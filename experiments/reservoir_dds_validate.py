# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of the GLOBAL reservoir operating-rule DDS path.

Exercises the real production components that the DDS calibration uses:
  - droute.calibration.parameter_manager.DRouteParameterManager.RESERVOIR_BOUNDS
    (the registered global reservoir parameter set + bounds), and
  - droute.calibration.worker.DRouteWorker._route_cpp_with_lakes /
    _apply_reservoir_operating_rules (route through the C++ lake router with the
    global reservoir rules applied).

A small DDS loop samples the four global reservoir parameters within their
registered bounds, drives them through the worker's routing, and scores KGE at
the Calgary outlet -- confirming the params flow end-to-end and that the
derivative-free optimiser improves the objective. (Complements the per-reservoir
Adam path in reservoir_rules_adam.py.)
"""
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from droute.calibration.worker import DRouteWorker
from droute.calibration.parameter_manager import DRouteParameterManager

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
OBS = f"{D}/observations/streamflow"
CAL = ("2011-01-01", "2012-12-31")
BND = DRouteParameterManager.RESERVOIR_BOUNDS          # the registered global reservoir bounds


def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o); s, o = s[m], o[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def setup_worker():
    """Build a DRouteWorker wired to the Calgary network/runoff/obs/lakes (daily)."""
    rn = gpd.read_file(glob.glob(f"{D}/shapefiles/river_network/*.shp")[0])
    seg_ids = rn["LINKNO"].astype(int).values
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    downstream = [id_to_idx.get(int(d), -1) for d in rn["DSLINKNO"].astype(int).values]
    outlet = int(np.argmax(rn["DSContArea"].astype(float).values))

    ds = xr.open_dataset(f"{D}/simulations/bow_calgary_v1/SUMMA/bow_calgary_v1_timestep.nc")
    gru = ds["gruId"].values.astype(int)
    time = pd.to_datetime(ds["time"].values)
    attr = xr.open_dataset(f"{D}/settings/SUMMA/attributes.nc")
    area = {int(h): float(a) for h, a in zip(attr["hruId"].values.astype(int),
                                              attr["HRUarea"].values.astype(float))}
    runoff = np.clip(ds["averageRoutedRunoff"].values, 0, None) * np.array([area[int(g)] for g in gru])[None, :]
    # hourly -> daily (m3/s); worker sums runoff per segment, so pre-multiply by area above
    daily = pd.DataFrame(runoff, index=time).resample("D").mean().loc[CAL[0]:CAL[1]]
    hru_to_seg = [id_to_idx.get(int(g), -1) for g in gru]

    obs = pd.read_csv(f"{OBS}/wsc_05BH004_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
    obs = obs.reindex(daily.index).values

    w = DRouteWorker(config={})
    w._runoff_data = daily.values
    w._network_config = {
        "n_segments": len(seg_ids), "downstream_idx": downstream, "outlet_indices": [outlet],
        "routing_order": list(range(len(seg_ids))),
        "slopes": rn["Slope"].astype(float).values.tolist(),
        "lengths": rn["Length"].astype(float).values.tolist(),
        "widths": [None] * len(seg_ids), "hru_to_seg_idx": hru_to_seg,
        "segment_ids": seg_ids.tolist(),
    }
    from pathlib import Path
    w._lake_config_path = Path(f"{D}/settings/dRoute/droute_lakes.yaml")
    w.routing_dt = 86400
    return w, outlet, obs


def evaluate(w, outlet, obs, params):
    """Route with the given global reservoir params via the worker's C++ lake path -> KGE."""
    routed = w._route_cpp_with_lakes(params)
    if routed is None:
        return -9.0
    return kge(routed[:, outlet], obs)


def main(n_iter=60, seed=0):
    w, outlet, obs = setup_worker()
    names = list(BND.keys())
    rng = np.random.default_rng(seed)

    def to_unit(p, v):
        b = BND[p]
        if b.get("transform") == "log":
            return (np.log(v) - np.log(b["min"])) / (np.log(b["max"]) - np.log(b["min"]))
        return (v - b["min"]) / (b["max"] - b["min"])

    def from_unit(p, u):
        b = BND[p]; u = min(max(u, 0.0), 1.0)
        if b.get("transform") == "log":
            return float(np.exp(np.log(b["min"]) + u * (np.log(b["max"]) - np.log(b["min"]))))
        return float(b["min"] + u * (b["max"] - b["min"]))

    # default rules (q_ref_mult=1, exp=1.5, q_min_frac=0, spill=1) as the DDS start
    x = np.array([to_unit("reservoir_q_ref_mult", 1.0), to_unit("reservoir_exp", 1.5),
                  to_unit("reservoir_q_min_frac", 0.0), to_unit("reservoir_spill_coef", 1.0)])
    x = np.clip(x, 0, 1)

    def decode(xv):
        return {names[i]: from_unit(names[i], xv[i]) for i in range(len(names))}

    best_x = x.copy()
    best = evaluate(w, outlet, obs, decode(x))
    k0 = best
    r = 0.2
    for it in range(1, n_iter + 1):
        trial = x.copy()
        probs = rng.uniform(size=len(x)) < max(1.0 - np.log(it) / np.log(n_iter), 1.0 / len(x))
        if not probs.any():
            probs[rng.integers(len(x))] = True
        trial[probs] += r * rng.normal(size=probs.sum())
        trial = np.clip(trial, 0, 1)
        score = evaluate(w, outlet, obs, decode(trial))
        if score > best:
            best, best_x, x = score, trial.copy(), trial.copy()
        if it % 10 == 0 or it == 1:
            print(f"  iter {it:3d}: best KGE={best:.4f}")

    print(f"\nGlobal reservoir DDS: KGE {k0:.4f} (default rules) -> {best:.4f} (calibrated)")
    print("calibrated global reservoir rules:")
    for k, v in decode(best_x).items():
        print(f"  {k} = {v:.3f}")
    return k0, best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=60)
    main(n_iter=ap.parse_args().iters)
