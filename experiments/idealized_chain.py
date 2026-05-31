#!/usr/bin/env python3
"""
Idealized MULTI-GAUGE routing experiment for the dRoute methods paper.

A long linear river chain with multi-day channel travel time and *heterogeneous* true
per-reach Manning's n. Observations are taken at several points along the chain (not
just the outlet). With multiple gauges the per-reach roughness field becomes
identifiable, so:

  - per-reach calibration recovers skill a single lumped n cannot -> skill RISES with
    parameter count (the point a single-gauge experiment cannot show, because per-reach
    n is equifinal there), and
  - gradient-based Adam converges in a near-constant number of passes while DDS cost
    grows with parameter dimension.

This is the idealized companion to the real multi-gauge Iceland experiment. Synthetic
observations = route(true n) at each gauge + small multiplicative noise. A warmup
window (channel spin-up while the long mainstem fills) is excluded from the objective.

Usage:
    python experiments/idealized_chain.py --reaches 50 --n-gauges 6 \
        --dds-evals 800 --dds-seeds 10 --adam-epochs 300
"""

import argparse
import json
from pathlib import Path

import numpy as np

import droute
from bow_at_banff_dds_vs_adam import (
    kge, nse, set_manning, dds, make_groups, expand, evals_to_within,
    N_MIN, N_MAX, N_DEFAULT, CONV_TOL, ADAM_PASSES_PER_EPOCH,
)

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


# ---------------------------------------------------------------------------
# Synthetic catchment
# ---------------------------------------------------------------------------
def build_chain(n_reaches, reach_len_m, slope):
    net = droute.Network()
    for i in range(n_reaches):
        r = droute.Reach()
        r.id = i; r.length = reach_len_m; r.slope = slope; r.manning_n = N_DEFAULT
        r.geometry.width_coef = 7.2; r.geometry.width_exp = 0.5
        r.geometry.depth_coef = 0.27; r.geometry.depth_exp = 0.3
        r.upstream_junction_id = i
        r.downstream_junction_id = i + 1 if i < n_reaches - 1 else -1
        net.add_reach(r)
    for i in range(n_reaches):
        j = droute.Junction()
        j.id = i; j.upstream_reach_ids = [i - 1] if i > 0 else []
        j.downstream_reach_ids = [i]
        net.add_junction(j)
    net.build_topology()
    return net


def synthetic_runoff(n_time, n_reach, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    lateral = np.full((n_time, n_reach), 2.0)
    for _ in range(12):
        t0 = rng.integers(0, n_time); amp = rng.uniform(5, 25); tau = rng.uniform(24, 120)
        pulse = amp * np.exp(-(t - t0) / tau) * (t >= t0)
        cover = rng.uniform(0.3, 1.0, n_reach)
        lateral += pulse[:, None] * cover[None, :]
    return lateral


def true_manning(n_reach, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_reach)
    trend = 0.025 + 0.045 * x
    local = 0.010 * np.sin(6 * np.pi * x) + rng.uniform(-0.006, 0.006, n_reach)
    return np.clip(trend + local, N_MIN, N_MAX)


# ---------------------------------------------------------------------------
# Multi-gauge routing + metrics
# ---------------------------------------------------------------------------
def route_multi(net, lateral, gauges, dt=3600.0, record=False):
    """Route; return sim of shape (n_time, n_gauges)."""
    n_time, n_reach = lateral.shape
    cfg = droute.RouterConfig(); cfg.dt = dt; cfg.enable_gradients = record
    rt = droute.MuskingumCungeRouter(net, cfg)
    if record:
        rt.start_recording()
    sim = np.zeros((n_time, len(gauges)))
    for t in range(n_time):
        for k in range(n_reach):
            rt.set_lateral_inflow(k, float(lateral[t, k]))
        rt.route_timestep()
        if record:
            rt.record_outputs(gauges)
        for gi, g in enumerate(gauges):
            sim[t, gi] = rt.get_discharge(g)
    if record:
        rt.stop_recording()
    return sim, rt


def mean_kge(sim, obs, warmup):
    return float(np.mean([kge(sim[warmup:, gi], obs[warmup:, gi])
                          for gi in range(sim.shape[1])]))


def mean_nse(sim, obs, warmup):
    return float(np.mean([nse(sim[warmup:, gi], obs[warmup:, gi])
                          for gi in range(sim.shape[1])]))


def evaluate(net, n_vals, lateral, obs, gauges, warmup):
    set_manning(net, n_vals)
    sim, _ = route_multi(net, lateral, gauges)
    return {"kge": mean_kge(sim, obs, warmup), "nse": mean_nse(sim, obs, warmup), "sim": sim}


# ---------------------------------------------------------------------------
# Multi-gauge DDS and Adam
# ---------------------------------------------------------------------------
def run_dds_multi(net, lateral, obs, gauges, warmup, groups, max_evals, seeds):
    n_params = int(groups.max()) + 1
    curves, best_kge, best_n = [], -np.inf, None
    for s in seeds:
        rng = np.random.default_rng(s)

        def objective(x01):
            set_manning(net, expand(N_MIN + x01 * (N_MAX - N_MIN), groups))
            sim, _ = route_multi(net, lateral, gauges)
            return 1.0 - mean_kge(sim, obs, warmup)

        x_best, f_best, hist = dds(objective, n_params, max_evals, rng)
        curves.append(1.0 - hist)
        if (1.0 - f_best) > best_kge:
            best_kge = 1.0 - f_best
            best_n = expand(N_MIN + x_best * (N_MAX - N_MIN), groups)
        print(f"    DDS seed {s}: mean cal KGE={1.0 - f_best:.3f}")
    return best_n, np.array(curves)


def calibrate_adam_multi(net, lateral, obs, gauges, warmup, groups, epochs, lr, dt=3600.0):
    if not HAS_TORCH:
        raise SystemExit("PyTorch required")
    n_reach = lateral.shape[1]; n_g = len(gauges); G = int(groups.max()) + 1
    T = lateral.shape[0]
    ndata = (T - warmup) * n_g
    log_n = torch.tensor(np.log(np.full(G, N_DEFAULT)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_n], lr=lr)
    kge_hist, best_kge_hist = [], []
    best_kge, best_n = -np.inf, expand(np.exp(log_n.detach().numpy()), groups).copy()
    for ep in range(epochs):
        opt.zero_grad()
        ng = torch.exp(log_n)
        n_reach_vals = expand(ng.detach().numpy(), groups)
        set_manning(net, n_reach_vals)
        sim, rt = route_multi(net, lateral, gauges, dt, record=True)
        k = mean_kge(sim, obs, warmup)
        kge_hist.append(k)
        if k > best_kge:
            best_kge, best_n = k, n_reach_vals.copy()
        best_kge_hist.append(best_kge)
        # MSE over post-warmup steps, summed across gauges; seed each gauge's adjoint
        dL = []
        for gi in range(n_g):
            res = sim[:, gi] - obs[:, gi]
            res[:warmup] = 0.0
            dL.append((2.0 / ndata * res).tolist())
        rt.compute_gradients_timeseries(list(gauges), dL)
        grads = rt.get_gradients()
        grad_reach = np.array([grads.get(f"reach_{i}_manning_n", 0.0) for i in range(n_reach)])
        grad_g = np.zeros(G); np.add.at(grad_g, groups, grad_reach)
        log_n.grad = torch.tensor(grad_g * ng.detach().numpy(), dtype=torch.float64)
        opt.step()
        if ep % 25 == 0:
            print(f"    Adam epoch {ep:4d}: mean KGE={k:.3f}")
    return {"best_n": best_n, "kge_history": np.array(kge_hist),
            "best_kge_history": np.array(best_kge_hist)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reaches", type=int, default=50)
    ap.add_argument("--reach-len-km", type=float, default=20.0)
    ap.add_argument("--slope", type=float, default=0.0008)
    ap.add_argument("--n-time", type=int, default=2500)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--n-gauges", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--dds-evals", type=int, default=800)
    ap.add_argument("--dds-seeds", type=int, default=10)
    ap.add_argument("--adam-epochs", type=int, default=300)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/results_idealized")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = args.reaches; outlet = n - 1
    net = build_chain(n, args.reach_len_km * 1000.0, args.slope)
    lengths = np.full(n, args.reach_len_km * 1000.0)
    # gauges evenly spaced along the chain, including the outlet
    gauges = sorted(set(np.linspace(n // args.n_gauges, n - 1, args.n_gauges).astype(int).tolist()))
    print(f"Chain: {n} reaches x {args.reach_len_km} km = {n*args.reach_len_km:.0f} km; "
          f"gauges at reaches {gauges}; warmup {args.warmup} steps")

    lateral = synthetic_runoff(args.n_time, n, args.seed)
    n_true = true_manning(n, args.seed)
    set_manning(net, n_true)
    obs_clean, _ = route_multi(net, lateral, gauges)
    rng = np.random.default_rng(args.seed + 999)
    obs = obs_clean * (1.0 + args.noise * rng.standard_normal(obs_clean.shape))

    cut = int(0.6 * args.n_time)
    # warmup applies within each window; validation warmup re-fills from window start
    lat_cal, obs_cal = lateral[:cut], obs[:cut]
    lat_val, obs_val = lateral[cut:], obs[cut:]

    d_cal = evaluate(net, np.full(n, N_DEFAULT), lat_cal, obs_cal, gauges, args.warmup)
    d_val = evaluate(net, np.full(n, N_DEFAULT), lat_val, obs_val, gauges, args.warmup)
    tr_cal = evaluate(net, n_true, lat_cal, obs_cal, gauges, args.warmup)
    tr_val = evaluate(net, n_true, lat_val, obs_val, gauges, args.warmup)
    print(f"  default n: cal {d_cal['kge']:.3f} / val {d_val['kge']:.3f}")
    print(f"  TRUE n   : cal {tr_cal['kge']:.3f} / val {tr_val['kge']:.3f}  (skill ceiling)")

    seeds = [args.seed + i for i in range(args.dds_seeds)]
    modes = [("lumped", make_groups("lumped", n, lengths, args.groups)),
             (f"grouped-{args.groups}", make_groups("grouped", n, lengths, args.groups)),
             ("per-reach", make_groups("perreach", n, lengths, args.groups))]

    results = []
    for label, groups in modes:
        G = int(groups.max()) + 1
        print(f"\n=== {label} ({G} params) ===")
        n_dds, dds_curves = run_dds_multi(net, lat_cal, obs_cal, gauges, args.warmup,
                                          groups, args.dds_evals, seeds)
        dds_med = np.median(dds_curves, axis=0)
        dds_cal = evaluate(net, n_dds, lat_cal, obs_cal, gauges, args.warmup)
        dds_val = evaluate(net, n_dds, lat_val, obs_val, gauges, args.warmup)
        adam = calibrate_adam_multi(net, lat_cal, obs_cal, gauges, args.warmup, groups,
                                    args.adam_epochs, args.adam_lr)
        adam_cal = evaluate(net, adam["best_n"], lat_cal, obs_cal, gauges, args.warmup)
        adam_val = evaluate(net, adam["best_n"], lat_val, obs_val, gauges, args.warmup)
        # Cost to within CONV_TOL of this mode's achieved optimum (cal-KGE space).
        # DDS: per-seed evals-to-target, then median across seeds (robust — comparing the
        # median *curve* to the best-seed optimum spuriously caps when the median plateaus
        # just below the bar). Adam: epochs-to-target x (fwd+reverse).
        mode_best = max(dds_cal["kge"], adam_cal["kge"])
        per_seed = [evals_to_within(dds_curves[s], mode_best) or args.dds_evals
                    for s in range(dds_curves.shape[0])]
        dds_cost = int(np.median(per_seed))
        as_ = evals_to_within(adam["best_kge_history"], mode_best)
        adam_cost = (as_ * ADAM_PASSES_PER_EPOCH) if as_ \
            else args.adam_epochs * ADAM_PASSES_PER_EPOCH
        print(f"  {label}: DDS cal/val={dds_cal['kge']:.3f}/{dds_val['kge']:.3f} "
              f"Adam cal/val={adam_cal['kge']:.3f}/{adam_val['kge']:.3f} "
              f"| cost(fwd-eq) DDS={dds_cost} Adam={adam_cost}")
        # persist curves so figures can be regenerated without re-running
        slug = label.replace("-", "_")
        import pandas as pd
        pd.DataFrame({"eval": np.arange(1, dds_curves.shape[1] + 1),
                      "best_kge_median": dds_med,
                      "best_kge_q25": np.percentile(dds_curves, 25, axis=0),
                      "best_kge_q75": np.percentile(dds_curves, 75, axis=0)}
                     ).to_csv(out / f"idealized_dds_{slug}.csv", index=False)
        pd.DataFrame({"epoch": np.arange(1, len(adam["best_kge_history"]) + 1),
                      "best_kge": adam["best_kge_history"]}
                     ).to_csv(out / f"idealized_adam_{slug}.csv", index=False)
        results.append({"mode": label, "n_params": G,
                        "dds_cal": dds_cal["kge"], "dds_val": dds_val["kge"],
                        "adam_cal": adam_cal["kge"], "adam_val": adam_val["kge"],
                        "dds_cost": dds_cost, "adam_cost": adam_cost,
                        "dds_curves": dds_curves, "adam_best": adam["best_kge_history"]})

    summary = {"n_reaches": n, "mainstem_km": n * args.reach_len_km, "gauges": gauges,
               "default": {"cal_kge": d_cal["kge"], "val_kge": d_val["kge"]},
               "true_ceiling": {"cal_kge": tr_cal["kge"], "val_kge": tr_val["kge"]},
               "parameterizations": [{k: r[k] for k in
                   ("mode", "n_params", "dds_cal", "dds_val", "adam_cal", "adam_val",
                    "dds_cost", "adam_cost")} for r in results]}
    (out / "idealized_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'idealized_summary.json'}")

    if not HAS_MPL:
        return 0
    nparams = [r["n_params"] for r in results]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.axhline(tr_val["kge"], color="green", ls="--", lw=1.2, label=f"true n ceiling ({tr_val['kge']:.2f})")
    axL.axhline(d_val["kge"], color="gray", ls=":", lw=1.2, label=f"default n ({d_val['kge']:.2f})")
    axL.plot(nparams, [r["dds_val"] for r in results], "C0-o", label="DDS")
    axL.plot(nparams, [r["adam_val"] for r in results], "C3-s", label="Adam")
    axL.set_xlabel("number of calibrated parameters"); axL.set_xscale("log")
    axL.set_ylabel("multi-gauge validation KGE")
    axL.set_title("Idealized chain (multi-gauge): skill vs. parameter count")
    axL.legend(loc="best"); axL.grid(alpha=0.3)
    axR.plot(nparams, [r["dds_cost"] for r in results], "C0-o", label="DDS")
    axR.plot(nparams, [r["adam_cost"] for r in results], "C3-s", label="Adam")
    axR.set_xlabel("number of calibrated parameters")
    axR.set_ylabel(f"fwd-pass-equiv. evals to within {CONV_TOL} KGE of optimum")
    axR.set_title("Idealized chain (multi-gauge): cost vs. parameter count")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.legend(loc="best"); axR.grid(alpha=0.3)
    fig.savefig(out / "idealized_scaling.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    pr = results[-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    dx = np.arange(1, pr["dds_curves"].shape[1] + 1)
    ax.fill_between(dx, np.percentile(pr["dds_curves"], 25, axis=0),
                    np.percentile(pr["dds_curves"], 75, axis=0), color="C0", alpha=0.25, label="DDS IQR")
    ax.plot(dx, np.median(pr["dds_curves"], axis=0), "C0-", lw=1.5, label="DDS median")
    ax.plot(ADAM_PASSES_PER_EPOCH * np.arange(1, len(pr["adam_best"]) + 1),
            pr["adam_best"], "C3-", lw=1.5, label="Adam (dRoute AD)")
    ax.set_xlabel("forward-pass-equivalent model evaluations (Adam = fwd + reverse)")
    ax.set_ylabel("best-so-far mean-gauge calibration KGE")
    ax.set_title(f"Idealized chain (multi-gauge): convergence ({n} params)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(out / "idealized_convergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
