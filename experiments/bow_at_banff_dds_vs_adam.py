#!/usr/bin/env python3
"""
GMD experiment: gradient-based (Adam) vs derivative-free (DDS) calibration of
Muskingum-Cunge reach roughness for the Bow River at Banff (Alberta, Canada).

This is the headline experiment for the dRoute methods paper. It demonstrates the
central claim of the library: because dRoute returns *exact* parameter gradients,
gradient-based calibration of per-reach parameters converges in far fewer model
evaluations than derivative-free search, and that advantage widens with the number
of calibrated parameters.

Domain
------
Semi-distributed Bow at Banff: 29 river reaches / 29 HRUs, hourly SUMMA
``averageRoutedRunoff`` forcing, observed streamflow at the Banff gauge. Routing
parameter: per-reach Manning's n (29 parameters), calibrated against KGE/NSE.

What it produces
----------------
- ``results/bow_dds_vs_adam.csv``      : objective vs. #model-evaluations (both methods)
- ``results/bow_convergence.png``      : convergence curves (the GMD figure)
- ``results/bow_hydrographs.png``      : observed vs. default/DDS/Adam hydrographs
- ``results/bow_summary.json``         : final metrics, wall-clock, #evaluations

Usage
-----
    python experiments/bow_at_banff_dds_vs_adam.py \
        --domain-dir /path/to/domain_Bow_at_Banff_semi_distributed \
        --cal-start 2008-10-01 --cal-end 2010-09-30 \
        --adam-epochs 200 --dds-evals 2000

The default --domain-dir points at the local SYMFLUENCE setup; override it for a
portable/archived copy when freezing the experiment for publication.

NOTE: This is the reproducibility scaffold for the GMD paper, not a finished result.
The numerics bind to the real dRoute API and the real Bow-at-Banff data; tune the
calibration window, parameterization, and optimizer settings before reporting.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import droute
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "droute is required. Build with: pip install -e .  (from the repo root)"
    ) from exc

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xarray is required: pip install xarray netcdf4") from exc

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DEFAULT_DOMAIN = (
    "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/"
    "domain_Bow_at_Banff_semi_distributed"
)

# Per-reach Manning's n calibration bounds (dimensionless).
N_MIN, N_MAX = 0.01, 0.10
N_DEFAULT = 0.035


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((obs - sim) ** 2) / denom) if denom > 0 else -np.inf


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    if len(sim) < 2:
        return -np.inf
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std() if obs.std() > 0 else 0.0
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else 0.0
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ---------------------------------------------------------------------------
# Data loading (Bow at Banff semi-distributed)
# ---------------------------------------------------------------------------
def load_topology(
    topo_path: Path,
) -> Tuple["droute.Network", np.ndarray, int, np.ndarray]:
    """Build a dRoute Network from a mizuRoute topology.nc file.

    Returns (network, hru_to_seg_idx, outlet_idx, seg_areas_m2).
    Manning's n defaults to N_DEFAULT when the topology has no ``mann_n`` field.
    """
    ds = xr.open_dataset(topo_path)
    seg_ids = ds["segId"].values.astype(int)
    down_ids = ds["downSegId"].values.astype(int)
    slopes = ds["slope"].values
    lengths = ds["length"].values
    hru_ids = ds["hruId"].values.astype(int)
    hru_to_seg = ds["hruToSegId"].values.astype(int)
    areas = ds["area"].values
    mann = ds["mann_n"].values if "mann_n" in ds else np.full(len(seg_ids), N_DEFAULT)
    ds.close()

    n_seg = len(seg_ids)
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}

    upstream = {i: [] for i in range(n_seg)}
    for i, d in enumerate(down_ids):
        if int(d) in id_to_idx:
            upstream[id_to_idx[int(d)]].append(i)

    outlet_idx = next(
        (i for i, d in enumerate(down_ids) if int(d) not in id_to_idx), n_seg - 1
    )

    network = droute.Network()
    for i in range(n_seg):
        reach = droute.Reach()
        reach.id = i
        reach.length = float(lengths[i])
        reach.slope = max(float(slopes[i]), 1e-4)
        reach.manning_n = float(mann[i])
        reach.geometry.width_coef = 7.2
        reach.geometry.width_exp = 0.5
        reach.geometry.depth_coef = 0.27
        reach.geometry.depth_exp = 0.3
        reach.upstream_junction_id = i
        d = int(down_ids[i])
        reach.downstream_junction_id = id_to_idx[d] if d in id_to_idx else -1
        network.add_reach(reach)

    for i in range(n_seg):
        junc = droute.Junction()
        junc.id = i
        junc.upstream_reach_ids = upstream[i]
        junc.downstream_reach_ids = [i]
        network.add_junction(junc)

    network.build_topology()

    # HRU -> segment index, and per-segment contributing area (m^2)
    hru_to_seg_idx = np.full(len(hru_ids), -1, dtype=int)
    seg_areas = np.zeros(n_seg)
    for k, hid in enumerate(hru_ids):
        sid = int(hru_to_seg[k])
        if sid in id_to_idx:
            hru_to_seg_idx[k] = id_to_idx[sid]
            seg_areas[id_to_idx[sid]] += float(areas[k])

    print(f"  Topology: {n_seg} reaches, outlet index {outlet_idx}, "
          f"basin area {seg_areas.sum() / 1e6:.0f} km^2")
    return network, hru_to_seg_idx, outlet_idx, seg_areas, np.asarray(lengths, float)


def load_forcing_and_obs(
    domain_dir: Path,
    hru_to_seg_idx: np.ndarray,
    n_seg: int,
    cal_start: Optional[str],
    cal_end: Optional[str],
    runoff_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load SUMMA runoff -> per-reach lateral inflow (m^3/s) and aligned obs (m^3/s).

    If runoff_file is given, use it (e.g. the SYMFLUENCE-calibrated SUMMA output);
    otherwise fall back to the first uncalibrated run under simulations/.
    SUMMA fill values (|runoff| > 1e3 m/s, e.g. uninitialised GRUs) are zeroed.
    """
    domain_name = domain_dir.name.replace("domain_", "")

    if runoff_file:
        summa_path = Path(runoff_file)
    else:
        summa_files = sorted(domain_dir.glob("simulations/*/SUMMA/*_timestep.nc"))
        if not summa_files:
            raise FileNotFoundError(f"No SUMMA *_timestep.nc under {domain_dir}/simulations")
        summa_path = summa_files[0]
    print(f"  SUMMA forcing: {summa_path}")

    ds = xr.open_dataset(summa_path)
    runoff_ms = ds["averageRoutedRunoff"].values.astype(float)  # (time, hru) in m/s
    times = pd.DatetimeIndex(ds["time"].values)
    ds.close()

    # Zero out SUMMA fill values (uninitialised GRUs report ~-2e4 m/s)
    n_fill = int((np.abs(runoff_ms) > 1e3).sum())
    if n_fill:
        bad_hru = np.where((np.abs(runoff_ms) > 1e3).any(axis=0))[0]
        print(f"  Masked {n_fill} fill values in HRUs {bad_hru.tolist()} -> 0")
        runoff_ms[np.abs(runoff_ms) > 1e3] = 0.0

    return runoff_ms, times, summa_path, domain_name


def build_lateral_inflow(
    runoff_ms: np.ndarray,
    hru_to_seg_idx: np.ndarray,
    seg_areas: np.ndarray,
    hru_areas: np.ndarray,
    n_seg: int,
) -> np.ndarray:
    """Convert HRU runoff (m/s) to per-reach lateral inflow (m^3/s)."""
    n_time, n_hru = runoff_ms.shape
    lateral = np.zeros((n_time, n_seg))
    for hru_idx in range(n_hru):
        seg = hru_to_seg_idx[hru_idx]
        if seg >= 0:
            lateral[:, seg] += runoff_ms[:, hru_idx] * hru_areas[hru_idx]
    return lateral


# ---------------------------------------------------------------------------
# Forward routing (Muskingum-Cunge) and objective
# ---------------------------------------------------------------------------
def route_mc(
    network: "droute.Network",
    lateral: np.ndarray,
    outlet: int,
    dt: float = 3600.0,
    record: bool = False,
) -> Tuple[np.ndarray, "droute.MuskingumCungeRouter"]:
    """Run Muskingum-Cunge forward, returning outlet discharge timeseries."""
    n_time, n_reach = lateral.shape
    cfg = droute.RouterConfig()
    cfg.dt = dt
    cfg.enable_gradients = record
    router = droute.MuskingumCungeRouter(network, cfg)
    if record:
        # The CoDiPack tape is a global singleton; without resetting it each call
        # it accumulates across Adam epochs and OOM-kills the process (~140 epochs).
        router.reset_gradients()
        router.start_recording()
    sim = np.zeros(n_time)
    for t in range(n_time):
        for r in range(n_reach):
            router.set_lateral_inflow(r, float(lateral[t, r]))
        router.route_timestep()
        if record:
            router.record_output(outlet)
        sim[t] = router.get_discharge(outlet)
    if record:
        router.stop_recording()
    return sim, router


def set_manning(network: "droute.Network", n_values: np.ndarray) -> None:
    for i, n in enumerate(n_values):
        network.get_reach(i).manning_n = float(np.clip(n, N_MIN, N_MAX))


# ---------------------------------------------------------------------------
# Parameterization: map a low-dimensional parameter vector onto 29 reaches
# ---------------------------------------------------------------------------
def make_groups(mode: str, n_seg: int, lengths: np.ndarray, n_groups: int) -> np.ndarray:
    """Return an int array (len n_seg) assigning each reach to a parameter group.

    mode='lumped'   -> 1 group (all reaches share one n)
    mode='grouped'  -> n_groups groups by reach-length quantiles
    mode='perreach' -> n_seg groups (one parameter per reach)
    """
    if mode == "lumped":
        return np.zeros(n_seg, dtype=int)
    if mode == "perreach":
        return np.arange(n_seg, dtype=int)
    if mode == "grouped":
        # quantile bins of reach length -> reproducible, physically meaningful strata
        ranks = np.argsort(np.argsort(lengths))
        return (ranks * n_groups // n_seg).astype(int)
    raise ValueError(f"unknown mode {mode}")


def expand(theta_g: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Broadcast per-group parameters to per-reach values."""
    return theta_g[groups]


# ---------------------------------------------------------------------------
# DDS (Tolson & Shoemaker, 2007) -- derivative-free baseline
# ---------------------------------------------------------------------------
def dds(
    objective,
    n_params: int,
    max_evals: int,
    rng: np.random.Generator,
    r: float = 0.2,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Minimize ``objective`` (a function of a [0,1]^n vector) with DDS.

    Returns (best_x, best_f, history) where history[i] is the best-so-far objective
    after i+1 evaluations.
    """
    x_best = rng.uniform(size=n_params)
    f_best = objective(x_best)
    history = [f_best]
    for i in range(1, max_evals):
        p = 1.0 - np.log(i + 1) / np.log(max_evals)  # prob. of perturbing each dim
        mask = rng.uniform(size=n_params) < p
        if not mask.any():
            mask[rng.integers(n_params)] = True
        x_new = x_best.copy()
        x_new[mask] += r * rng.standard_normal(mask.sum())
        # reflect at [0,1] bounds
        x_new = np.clip(x_new, 0.0, 1.0)
        f_new = objective(x_new)
        if f_new < f_best:
            x_best, f_best = x_new, f_new
        history.append(f_best)
    return x_best, f_best, np.array(history)


# ---------------------------------------------------------------------------
# Adam (gradient-based) using dRoute timeseries AD
# ---------------------------------------------------------------------------
def calibrate_adam(
    network: "droute.Network",
    lateral: np.ndarray,
    obs: np.ndarray,
    outlet: int,
    n_epochs: int,
    lr: float,
    groups: np.ndarray,
    dt: float = 3600.0,
) -> Dict[str, np.ndarray]:
    """Calibrate per-group log(Manning's n) with Adam + CoDiPack timeseries gradients.

    Operates over ``G = groups.max()+1`` parameters; per-reach gradients from the AD
    tape are scatter-added back to their group. Adam minimizes MSE (it needs a smooth
    loss) but we also record the calibration KGE at every epoch and keep the parameter
    set with the best calibration KGE seen so far.

    One forward routing pass per epoch (plus one reverse pass for the gradient).

    Returns dict with:
        best_n          : per-reach parameters at the best calibration KGE
        mse_history     : MSE per epoch
        kge_history     : calibration KGE per epoch
        best_kge_history: best-so-far calibration KGE per epoch (monotone)
    """
    if not HAS_TORCH:
        raise SystemExit("PyTorch required for the Adam path: pip install torch")

    valid = ~np.isnan(obs)
    n_reach = lateral.shape[1]
    G = int(groups.max()) + 1
    log_n = torch.tensor(
        np.log(np.full(G, N_DEFAULT)), dtype=torch.float64, requires_grad=True
    )
    opt = torch.optim.Adam([log_n], lr=lr)

    mse_hist, kge_hist, best_kge_hist = [], [], []
    best_kge = -np.inf
    best_n = expand(torch.exp(log_n).detach().numpy(), groups).copy()

    for epoch in range(n_epochs):
        opt.zero_grad()
        n_g = torch.exp(log_n)
        n_reach_vals = expand(n_g.detach().numpy(), groups)
        set_manning(network, n_reach_vals)

        sim, router = route_mc(network, lateral, outlet, dt, record=True)
        # MSE only over steps with observations (matches the KGE evaluation mask)
        resid = np.where(valid, sim - obs, 0.0)
        mse = float(np.sum(resid ** 2) / max(valid.sum(), 1))
        k = kge(sim, obs)
        mse_hist.append(mse)
        kge_hist.append(k)
        if k > best_kge:
            best_kge = k
            best_n = n_reach_vals.copy()
        best_kge_hist.append(best_kge)

        # dMSE/dQ_t = 2*(sim_t - obs_t)/N over valid steps
        dL_dQ = (2.0 / max(valid.sum(), 1)) * resid
        router.compute_gradients_timeseries(outlet, dL_dQ.tolist())
        grads = router.get_gradients()
        grad_reach = np.array(
            [grads.get(f"reach_{i}_manning_n", 0.0) for i in range(n_reach)]
        )
        # scatter per-reach gradients onto their group
        grad_g = np.zeros(G)
        np.add.at(grad_g, groups, grad_reach)

        # chain rule for log transform: dL/d(log n_g) = dL/dn_g * n_g
        log_n.grad = torch.tensor(
            grad_g * n_g.detach().numpy(), dtype=torch.float64
        )
        opt.step()
        if epoch % 20 == 0:
            print(f"    Adam epoch {epoch:4d}: MSE={mse:.4g}  KGE={k:.3f}")

    return {
        "best_n": best_n,
        "mse_history": np.array(mse_hist),
        "kge_history": np.array(kge_hist),
        "best_kge_history": np.array(best_kge_hist),
    }


# ---------------------------------------------------------------------------
# Helpers for fair comparison
# ---------------------------------------------------------------------------
def evaluate_params(
    network: "droute.Network",
    n_values: np.ndarray,
    lateral: np.ndarray,
    obs: np.ndarray,
    outlet: int,
) -> Dict[str, float]:
    """KGE/NSE of a parameter set on a given (lateral, obs) window."""
    set_manning(network, n_values)
    sim, _ = route_mc(network, lateral, outlet)
    return {"kge": kge(sim, obs), "nse": nse(sim, obs), "sim": sim}


def run_dds_multiseed(
    network: "droute.Network",
    lateral: np.ndarray,
    obs: np.ndarray,
    outlet: int,
    max_evals: int,
    seeds: list,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run DDS for several seeds over ``G = groups.max()+1`` parameters.

    Returns (best_n_overall_perreach, best_kge_curves) where best_kge_curves has
    shape (n_seeds, max_evals): best-so-far calibration KGE after each evaluation.
    """
    n_params = int(groups.max()) + 1
    curves = []
    best_overall_kge = -np.inf
    best_overall_n = None
    for s in seeds:
        rng = np.random.default_rng(s)

        def objective(x01: np.ndarray) -> float:
            n_g = N_MIN + x01 * (N_MAX - N_MIN)
            set_manning(network, expand(n_g, groups))
            sim, _ = route_mc(network, lateral, outlet)
            return 1.0 - kge(sim, obs)

        x_best, f_best, hist = dds(objective, n_params, max_evals, rng)
        curves.append(1.0 - hist)  # best-so-far KGE
        final_kge = 1.0 - f_best
        print(f"    DDS seed {s}: final cal KGE={final_kge:.3f}")
        if final_kge > best_overall_kge:
            best_overall_kge = final_kge
            best_overall_n = expand(N_MIN + x_best * (N_MAX - N_MIN), groups)
    return best_overall_n, np.array(curves)


# Convergence tolerance: a calibration is "converged" once best-so-far KGE is within
# CONV_TOL of the best value the run ultimately achieves. Using a fixed fraction of the
# best (the old "95% of best") is misleading when the optimum sits just above a decent
# default, so we measure approach to the *achieved optimum* instead.
CONV_TOL = 0.005

# One Adam epoch costs a forward pass plus a reverse (adjoint) pass; one DDS iteration is
# a single forward pass. Express both in forward-pass-equivalents so the cost axis is fair.
ADAM_PASSES_PER_EPOCH = 2


def evals_to_within(best_kge_curve: np.ndarray, optimum: float,
                    tol: float = CONV_TOL) -> Optional[int]:
    """First step (1-based) at which best-so-far KGE comes within tol of `optimum`."""
    hit = np.where(best_kge_curve >= optimum - tol)[0]
    return int(hit[0] + 1) if len(hit) else None


def run_one_parameterization(
    network, lat_cal, obs_cal, lat_val, obs_val, have_val, outlet,
    groups, mode_label, args, seeds,
) -> Dict:
    """Calibrate one parameterization with both DDS and Adam; return all results."""
    G = int(groups.max()) + 1
    print(f"\n=== Parameterization: {mode_label} ({G} parameters) ===")

    print(f"  DDS ({len(seeds)} seeds x {args.dds_evals} evals) ...")
    t0 = time.time()
    n_dds, dds_curves = run_dds_multiseed(
        network, lat_cal, obs_cal, outlet, args.dds_evals, seeds, groups
    )
    dds_wall = time.time() - t0
    dds_cal = evaluate_params(network, n_dds, lat_cal, obs_cal, outlet)
    dds_val = evaluate_params(network, n_dds, lat_val, obs_val, outlet) if have_val else None

    print(f"  Adam ({args.adam_epochs} epochs) ...")
    t0 = time.time()
    adam = calibrate_adam(
        network, lat_cal, obs_cal, outlet, args.adam_epochs, args.adam_lr, groups
    )
    adam_wall = time.time() - t0
    adam_cal = evaluate_params(network, adam["best_n"], lat_cal, obs_cal, outlet)
    adam_val = (
        evaluate_params(network, adam["best_n"], lat_val, obs_val, outlet)
        if have_val else None
    )

    dds_median = np.median(dds_curves, axis=0)
    # Converge toward the best KGE either method achieved in this mode, within CONV_TOL.
    mode_best = max(dds_cal["kge"], adam_cal["kge"])
    dds_steps = evals_to_within(dds_median, mode_best)
    adam_steps = evals_to_within(adam["best_kge_history"], mode_best)
    # Cost in forward-pass-equivalents (Adam epoch = fwd + reverse).
    dds_cost = dds_steps
    adam_cost = adam_steps * ADAM_PASSES_PER_EPOCH if adam_steps else None
    print(f"  {mode_label}: DDS cal/val KGE="
          f"{dds_cal['kge']:.3f}/{dds_val['kge'] if have_val else float('nan'):.3f}  "
          f"Adam cal/val KGE="
          f"{adam_cal['kge']:.3f}/{adam_val['kge'] if have_val else float('nan'):.3f}  "
          f"| cost(fwd-eq) DDS={dds_cost} Adam={adam_cost}")

    return {
        "mode": mode_label, "n_params": G,
        "n_dds": n_dds, "dds_curves": dds_curves, "dds_median": dds_median,
        "adam": adam, "n_adam": adam["best_n"],
        "dds_cal": dds_cal, "dds_val": dds_val,
        "adam_cal": adam_cal, "adam_val": adam_val,
        "dds_wall": dds_wall, "adam_wall": adam_wall,
        "mode_best": mode_best,
        "dds_cost": dds_cost, "adam_cost": adam_cost,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain-dir", default=DEFAULT_DOMAIN)
    ap.add_argument("--cal-start", default=None, help="calibration start, e.g. 2005-01-01")
    ap.add_argument("--cal-end", default=None, help="calibration end, e.g. 2006-12-31")
    ap.add_argument("--val-start", default=None, help="validation start, e.g. 2007-01-01")
    ap.add_argument("--val-end", default=None, help="validation end, e.g. 2007-12-31")
    ap.add_argument("--adam-epochs", type=int, default=300)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--dds-evals", type=int, default=1000)
    ap.add_argument("--dds-seeds", type=int, default=10, help="number of DDS seeds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true",
                    help="run the parameter-count sweep (lumped, grouped, per-reach)")
    ap.add_argument("--groups", type=int, default=4,
                    help="number of groups for the 'grouped' parameterization")
    ap.add_argument("--runoff-file", default=None,
                    help="SUMMA *_timestep.nc to route (default: first uncalibrated run)")
    ap.add_argument("--out-dir", default="experiments/results")
    args = ap.parse_args()

    domain_dir = Path(args.domain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading topology ...")
    topo = domain_dir / "settings" / "mizuRoute" / "topology.nc"
    network, hru_to_seg_idx, outlet, seg_areas, lengths = load_topology(topo)
    n_seg = len(seg_areas)

    print("Loading SUMMA forcing ...")
    runoff_ms, times, summa_path, domain_name = load_forcing_and_obs(
        domain_dir, hru_to_seg_idx, n_seg, args.cal_start, args.cal_end, args.runoff_file
    )

    ds = xr.open_dataset(topo)
    hru_areas = ds["area"].values
    ds.close()
    lateral_full = build_lateral_inflow(
        runoff_ms, hru_to_seg_idx, seg_areas, hru_areas, n_seg
    )

    obs_csv = (
        domain_dir / "data" / "observations" / "streamflow" / "preprocessed"
        / f"{domain_name}_streamflow_processed.csv"
    )
    obs_df = pd.read_csv(obs_csv, index_col="datetime", parse_dates=True)
    obs_full = obs_df.iloc[:, 0].reindex(times).values

    def window(start, end):
        if start is None:
            return lateral_full, obs_full, times
        sel = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end or times[-1]))
        idx = np.where(sel)[0]
        return lateral_full[idx], obs_full[idx], times[idx]

    lat_cal, obs_cal, t_cal = window(args.cal_start, args.cal_end)
    lat_val, obs_val, t_val = window(args.val_start, args.val_end)
    have_val = args.val_start is not None
    print(f"  Calibration: {t_cal[0]} .. {t_cal[-1]} ({len(t_cal)} steps, "
          f"{np.sum(~np.isnan(obs_cal))} obs)")
    if have_val:
        print(f"  Validation:  {t_val[0]} .. {t_val[-1]} ({len(t_val)} steps, "
              f"{np.sum(~np.isnan(obs_val))} obs)")

    # --- default run ---
    n_def = np.full(n_seg, N_DEFAULT)
    d_cal = evaluate_params(network, n_def, lat_cal, obs_cal, outlet)
    d_val = evaluate_params(network, n_def, lat_val, obs_val, outlet) if have_val else None
    print(f"  Default n={N_DEFAULT}: cal KGE={d_cal['kge']:.3f} NSE={d_cal['nse']:.3f}")

    seeds = [args.seed + i for i in range(args.dds_seeds)]

    # --- parameterizations to run ---
    if args.sweep:
        modes = [
            ("lumped", make_groups("lumped", n_seg, lengths, args.groups)),
            (f"grouped-{args.groups}", make_groups("grouped", n_seg, lengths, args.groups)),
            ("per-reach", make_groups("perreach", n_seg, lengths, args.groups)),
        ]
    else:
        modes = [("per-reach", make_groups("perreach", n_seg, lengths, args.groups))]

    results = [
        run_one_parameterization(
            network, lat_cal, obs_cal, lat_val, obs_val, have_val, outlet,
            groups, label, args, seeds,
        )
        for label, groups in modes
    ]

    # --- persist histories per mode ---
    for r in results:
        slug = r["mode"].replace("-", "_")
        pd.DataFrame(
            {"eval": np.arange(1, r["dds_curves"].shape[1] + 1),
             "best_kge_median": r["dds_median"],
             "best_kge_q25": np.percentile(r["dds_curves"], 25, axis=0),
             "best_kge_q75": np.percentile(r["dds_curves"], 75, axis=0)}
        ).to_csv(out_dir / f"bow_dds_{slug}.csv", index=False)
        pd.DataFrame(
            {"epoch": np.arange(1, len(r["adam"]["best_kge_history"]) + 1),
             "kge": r["adam"]["kge_history"],
             "best_kge": r["adam"]["best_kge_history"],
             "mse": r["adam"]["mse_history"]}
        ).to_csv(out_dir / f"bow_adam_{slug}.csv", index=False)

    summary = {
        "domain": domain_name,
        "n_reaches": int(n_seg),
        "calibration_window": [str(t_cal[0]), str(t_cal[-1])],
        "validation_window": [str(t_val[0]), str(t_val[-1])] if have_val else None,
        "dds_seeds": args.dds_seeds, "dds_evals": args.dds_evals,
        "adam_epochs": args.adam_epochs, "adam_lr": args.adam_lr,
        "default": {"cal_kge": d_cal["kge"], "cal_nse": d_cal["nse"],
                    "val_kge": d_val["kge"] if have_val else None},
        "parameterizations": [
            {"mode": r["mode"], "n_params": r["n_params"],
             "mode_best_cal_kge": r["mode_best"],
             "dds": {"cal_kge": r["dds_cal"]["kge"],
                     "val_kge": r["dds_val"]["kge"] if have_val else None,
                     "wall_s": r["dds_wall"],
                     "fwdeq_to_converge": r["dds_cost"]},
             "adam": {"cal_kge": r["adam_cal"]["kge"],
                      "val_kge": r["adam_val"]["kge"] if have_val else None,
                      "wall_s": r["adam_wall"],
                      "fwdeq_to_converge": r["adam_cost"]}}
            for r in results
        ],
    }
    (out_dir / "bow_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {out_dir / 'bow_summary.json'}")

    if not HAS_MPL:
        return 0

    # --- Figure 1: convergence for the per-reach parameterization (headline) ---
    perreach = next(r for r in results if r["mode"] == "per-reach")
    fig, ax = plt.subplots(figsize=(8, 5))
    dds_x = np.arange(1, perreach["dds_curves"].shape[1] + 1)
    ax.fill_between(dds_x, np.percentile(perreach["dds_curves"], 25, axis=0),
                    np.percentile(perreach["dds_curves"], 75, axis=0),
                    color="C0", alpha=0.25, label="DDS IQR (seeds)")
    ax.plot(dds_x, perreach["dds_median"], "C0-", lw=1.5, label="DDS median")
    # Adam plotted in forward-pass-equivalents (each epoch = forward + reverse pass)
    adam_x = ADAM_PASSES_PER_EPOCH * np.arange(1, len(perreach["adam"]["best_kge_history"]) + 1)
    ax.plot(adam_x, perreach["adam"]["best_kge_history"], "C3-", lw=1.5,
            label="Adam (dRoute AD)")
    ax.set_xlabel("forward-pass-equivalent model evaluations "
                  "(Adam epoch = fwd + reverse)")
    ax.set_ylabel("best-so-far calibration KGE")
    ax.set_title(f"Bow at Banff: gradient vs. derivative-free "
                 f"({n_seg} per-reach parameters)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "fig1_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: scaling with parameter count (sweep only) ---
    if args.sweep and len(results) > 1:
        nparams = [r["n_params"] for r in results]
        dds_skill = [r[("dds_val" if have_val else "dds_cal")]["kge"] for r in results]
        adam_skill = [r[("adam_val" if have_val else "adam_cal")]["kge"] for r in results]
        default_skill = d_val["kge"] if have_val else d_cal["kge"]
        label = "validation" if have_val else "calibration"

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        # Left: skill vs #params on an HONEST axis anchored at the uncalibrated default,
        # so the (lack of) gain from more parameters is not exaggerated by auto-zoom.
        axL.axhline(default_skill, color="gray", ls=":", lw=1.2,
                    label=f"default n (KGE={default_skill:.2f})")
        axL.plot(nparams, dds_skill, "C0-o", label="DDS")
        axL.plot(nparams, adam_skill, "C3-s", label="Adam")
        axL.set_xlabel("number of calibrated parameters")
        axL.set_ylabel(f"{label} KGE (best params)")
        axL.set_title("Skill vs. parameter count")
        axL.set_xscale("log")
        lo = min(default_skill, min(dds_skill), min(adam_skill))
        hi = max(max(dds_skill), max(adam_skill))
        axL.set_ylim(lo - 0.05, hi + 0.03)
        axL.legend(loc="best"); axL.grid(alpha=0.3)

        # Right: cost to converge (within CONV_TOL of the achieved optimum), in
        # forward-pass-equivalents. None -> not reached within budget (plot at cap).
        dds_cap = args.dds_evals
        adam_cap = args.adam_epochs * ADAM_PASSES_PER_EPOCH
        dds_c = [r["dds_cost"] or dds_cap for r in results]
        adam_c = [r["adam_cost"] or adam_cap for r in results]
        axR.plot(nparams, dds_c, "C0-o", label="DDS")
        axR.plot(nparams, adam_c, "C3-s", label="Adam")
        axR.set_xlabel("number of calibrated parameters")
        axR.set_ylabel(f"fwd-pass-equiv. evals to within {CONV_TOL} KGE of optimum")
        axR.set_title("Calibration cost vs. parameter count")
        axR.set_xscale("log"); axR.set_yscale("log")
        axR.legend(loc="best"); axR.grid(alpha=0.3)
        fig.savefig(out_dir / "fig2_scaling.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- hydrographs on the reporting window for the per-reach fit ---
    rep_lat, rep_obs, rep_t = (lat_val, obs_val, t_val) if have_val else (
        lat_cal, obs_cal, t_cal)
    sim_def = evaluate_params(network, n_def, rep_lat, rep_obs, outlet)["sim"]
    sim_dds = evaluate_params(network, perreach["n_dds"], rep_lat, rep_obs, outlet)["sim"]
    sim_adam = evaluate_params(network, perreach["n_adam"], rep_lat, rep_obs, outlet)["sim"]
    tag = "validation" if have_val else "calibration"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rep_t, rep_obs, "k-", lw=1.5, label="Observed", alpha=0.8)
    ax.plot(rep_t, sim_def, "--", lw=1, label="Default", alpha=0.6)
    ax.plot(rep_t, sim_dds, lw=1, label=f"DDS (KGE={kge(sim_dds, rep_obs):.2f})")
    ax.plot(rep_t, sim_adam, lw=1, label=f"Adam (KGE={kge(sim_adam, rep_obs):.2f})")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"Bow at Banff: routed hydrographs ({tag}, per-reach)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out_dir / "fig3_hydrographs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
