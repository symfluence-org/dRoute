#!/usr/bin/env python3
"""
Real-basin MULTI-GAUGE routing experiment (the field counterpart to idealized_chain).

Extracts a real Icelandic sub-basin from the national mizuRoute network (default: the
254-reach basin draining to segId 1890, which contains 29 LamaH-Ice gauges), routes the
existing calibrated SUMMA runoff through it with dRoute, and calibrates per-reach
Manning's n against the multiple internal gauges with gradient-based Adam vs
derivative-free DDS at increasing parameter dimension (lumped / grouped / per-reach).

This is the real-world test of the scaling argument: with many gauges the per-reach
roughness field is identifiable, so skill should rise with parameter count, and Adam
should reach the optimum in far fewer model passes than DDS as dimension grows.

Usage:
    python experiments/iceland_multigauge.py --outlet-seg 1890 --groups 8 \
        --dds-evals 600 --dds-seeds 8 --adam-epochs 300
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import droute
from bow_at_banff_dds_vs_adam import (
    kge, nse, set_manning, make_groups, evals_to_within,
    N_MIN, N_MAX, N_DEFAULT, CONV_TOL, ADAM_PASSES_PER_EPOCH,
)
from idealized_chain import (
    route_multi, mean_kge, mean_nse, evaluate, run_dds_multi, calibrate_adam_multi,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DOMAIN = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Iceland_National"
OBS_DIR = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/D_gauges/2_timeseries/daily"
RUNOFF = (f"{DOMAIN}/simulations/iceland_summa_glacier_v1/SUMMA/"
          "iceland_summa_glacier_v1_timestep.nc")


def upstream_reaches(seg, down, outlet_seg):
    """Return the set of segIds upstream of (and including) outlet_seg."""
    idx = {int(s): i for i, s in enumerate(seg)}
    up = {}
    for i, d in enumerate(down):
        up.setdefault(int(d), []).append(int(seg[i]))
    keep, stack = set(), [int(outlet_seg)]
    while stack:
        s = stack.pop()
        if s in keep:
            continue
        keep.add(s)
        stack.extend(up.get(s, []))
    return keep


def build_subnetwork(topo_path, outlet_seg):
    """Build a dRoute Network for the sub-basin draining to outlet_seg.

    Returns (network, sub_segids, segid_to_local, hru_to_local, hru_areas_local,
             local_lengths, outlet_local).
    """
    ds = xr.open_dataset(topo_path)
    seg = ds["segId"].values.astype(int)
    down = ds["downSegId"].values.astype(int)
    length = ds["length"].values
    slope = ds["slope"].values
    hru = ds["hruId"].values.astype(int)
    hru2seg = ds["hruToSegId"].values.astype(int)
    area = ds["area"].values
    ds.close()

    keep = upstream_reaches(seg, down, outlet_seg)
    sub = [int(s) for s in seg if int(s) in keep]            # sub-basin segIds
    loc = {s: i for i, s in enumerate(sub)}                  # segId -> local index
    n = len(sub)
    seg_pos = {int(s): i for i, s in enumerate(seg)}         # segId -> global row

    # upstream adjacency within the sub-basin
    up_local = {i: [] for i in range(n)}
    for s in sub:
        d = int(down[seg_pos[s]])
        if d in loc:
            up_local[loc[d]].append(loc[s])
    outlet_local = loc[int(outlet_seg)]

    net = droute.Network()
    for s in sub:
        gi = seg_pos[s]; i = loc[s]
        r = droute.Reach()
        r.id = i
        r.length = float(length[gi])
        r.slope = max(float(slope[gi]), 1e-4)
        r.manning_n = N_DEFAULT
        r.geometry.width_coef = 7.2; r.geometry.width_exp = 0.5
        r.geometry.depth_coef = 0.27; r.geometry.depth_exp = 0.3
        r.upstream_junction_id = i
        d = int(down[gi])
        r.downstream_junction_id = loc[d] if d in loc else -1
        net.add_reach(r)
    for i in range(n):
        j = droute.Junction()
        j.id = i; j.upstream_reach_ids = up_local[i]; j.downstream_reach_ids = [i]
        net.add_junction(j)
    net.build_topology()

    # HRU -> local reach, and per-HRU area
    hru_to_local = np.full(len(hru), -1, dtype=int)
    for k, hid in enumerate(hru):
        s = int(hru2seg[k])
        if s in loc:
            hru_to_local[k] = loc[s]
    lengths_local = np.array([float(length[seg_pos[s]]) for s in sub])
    print(f"  Sub-basin (outlet seg {outlet_seg}): {n} reaches, "
          f"mainstem-ish max len {lengths_local.max()/1000:.0f} km")
    return net, sub, loc, hru, hru_to_local, area, lengths_local, outlet_local, n


def build_lateral_daily(hru_to_local, hru_area, n_sub):
    """SUMMA hourly runoff (m/s) -> per-reach daily lateral inflow (m^3/s)."""
    ds = xr.open_dataset(RUNOFF)
    ro = ds["averageRoutedRunoff"].values.astype(float)   # (time, hru) m/s
    ro[np.abs(ro) > 1e3] = 0.0                             # mask fill (water HRUs)
    t = pd.DatetimeIndex(ds["time"].values)
    ds.close()
    n_time, n_hru = ro.shape
    lateral = np.zeros((n_time, n_sub))
    for h in range(n_hru):
        loc = hru_to_local[h]
        if loc >= 0:
            lateral[:, loc] += ro[:, h] * hru_area[h]      # m^3/s
    # aggregate hourly -> daily mean
    df = pd.DataFrame(lateral, index=t).resample("D").mean()
    return df.values, pd.DatetimeIndex(df.index)


def load_gauges(loc, outlet_seg, times):
    """Return (gauge_local_reaches, obs[T, n_gauges], gauge_names) for gauges in the basin."""
    gm = pd.read_csv(f"{DOMAIN}/settings/mizuRoute/gauge_segment_mapping.csv")
    rows = gm[gm["nearest_segment"].isin(loc.keys())]
    gauges, names, obs_cols = [], [], []
    for _, r in rows.iterrows():
        gid = int(r["id"]); seg = int(r["nearest_segment"])
        f = Path(OBS_DIR) / f"ID_{gid}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep=";")
        d["date"] = pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD))
        s = d.set_index("date")["qobs"].replace(-999, np.nan)
        aligned = s.reindex(times).values
        if np.sum(~np.isnan(aligned)) < 50:                # require some overlap
            continue
        gauges.append(loc[seg]); names.append(r["name"]); obs_cols.append(aligned)
    obs = np.array(obs_cols).T if obs_cols else np.zeros((len(times), 0))
    return gauges, obs, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outlet-seg", type=int, default=1890)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--cal-start", default="2004-01-01")
    ap.add_argument("--cal-end", default="2008-12-31")
    ap.add_argument("--val-start", default="2009-01-01")
    ap.add_argument("--val-end", default="2010-12-31")
    ap.add_argument("--warmup", type=int, default=180)     # days
    ap.add_argument("--dds-evals", type=int, default=600)
    ap.add_argument("--dds-seeds", type=int, default=8)
    ap.add_argument("--adam-epochs", type=int, default=300)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/results_iceland")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    topo = f"{DOMAIN}/settings/mizuRoute/topology.nc"

    print("Building sub-network ...")
    net, sub, loc, hru, hru2loc, area, lengths, outlet, n = build_subnetwork(topo, args.outlet_seg)
    # warm-up route (first-router-on-fresh-network produces a broken initial transient)
    _wcfg = droute.RouterConfig(); _wcfg.dt = 86400.0
    _w = droute.MuskingumCungeRouter(net, _wcfg)
    for k in range(n):
        _w.set_lateral_inflow(k, 1.0)
    _w.route_timestep()

    print("Loading SUMMA runoff -> daily lateral inflow ...")
    lateral, times = build_lateral_daily(hru2loc, area, n)
    print(f"  {lateral.shape[0]} daily steps, {times[0].date()}..{times[-1].date()}")

    print("Loading gauges ...")
    gauges, obs, names = load_gauges(loc, args.outlet_seg, times)
    print(f"  {len(gauges)} usable gauges in basin (of mapping); "
          f"obs coverage {np.mean(~np.isnan(obs)):.0%}")

    def window(a, b):
        m = (times >= pd.Timestamp(a)) & (times <= pd.Timestamp(b))
        i = np.where(m)[0]
        return lateral[i], obs[i], times[i]
    lat_cal, obs_cal, t_cal = window(args.cal_start, args.cal_end)
    lat_val, obs_val, t_val = window(args.val_start, args.val_end)
    print(f"  cal {t_cal[0].date()}..{t_cal[-1].date()} ({len(t_cal)}d), "
          f"val {t_val[0].date()}..{t_val[-1].date()} ({len(t_val)}d)")

    dt = 86400.0
    # default baseline (route at daily dt)
    def ev(nv, lat, ob):
        set_manning(net, nv); sim, _ = route_multi(net, lat, gauges, dt)
        return mean_kge(sim, ob, args.warmup), mean_nse(sim, ob, args.warmup)
    dk, dn = ev(np.full(n, N_DEFAULT), lat_cal, obs_cal)
    print(f"  default n: cal meanKGE={dk:.3f} NSE={dn:.3f}")

    seeds = [args.seed + i for i in range(args.dds_seeds)]
    modes = [("lumped", make_groups("lumped", n, lengths, args.groups)),
             (f"grouped-{args.groups}", make_groups("grouped", n, lengths, args.groups)),
             ("per-reach", make_groups("perreach", n, lengths, args.groups))]

    results = []
    for label, groups in modes:
        G = int(groups.max()) + 1
        print(f"\n=== {label} ({G} params) ===")
        n_dds, dds_curves = run_dds_multi(net, lat_cal, obs_cal, gauges, args.warmup,
                                          groups, args.dds_evals, seeds, dt=dt)
        dds_med = np.median(dds_curves, axis=0)
        dc = evaluate(net, n_dds, lat_cal, obs_cal, gauges, args.warmup, dt)
        dv = evaluate(net, n_dds, lat_val, obs_val, gauges, args.warmup, dt)
        adam = calibrate_adam_multi(net, lat_cal, obs_cal, gauges, args.warmup, groups,
                                    args.adam_epochs, args.adam_lr, dt=dt)
        ac = evaluate(net, adam["best_n"], lat_cal, obs_cal, gauges, args.warmup, dt)
        av = evaluate(net, adam["best_n"], lat_val, obs_val, gauges, args.warmup, dt)
        mode_best = max(dc["kge"], ac["kge"])
        per_seed = [evals_to_within(dds_curves[s], mode_best) or args.dds_evals
                    for s in range(dds_curves.shape[0])]
        dds_cost = int(np.median(per_seed))
        as_ = evals_to_within(adam["best_kge_history"], mode_best)
        adam_cost = (as_ * ADAM_PASSES_PER_EPOCH) if as_ else args.adam_epochs * ADAM_PASSES_PER_EPOCH
        print(f"  {label}: DDS cal/val={dc['kge']:.3f}/{dv['kge']:.3f} "
              f"Adam cal/val={ac['kge']:.3f}/{av['kge']:.3f} | cost(fwd-eq) DDS={dds_cost} Adam={adam_cost}")
        slug = label.replace("-", "_")
        pd.DataFrame({"eval": np.arange(1, dds_curves.shape[1] + 1),
                      "best_kge_median": dds_med,
                      "best_kge_q25": np.percentile(dds_curves, 25, axis=0),
                      "best_kge_q75": np.percentile(dds_curves, 75, axis=0)}
                     ).to_csv(out / f"iceland_dds_{slug}.csv", index=False)
        pd.DataFrame({"epoch": np.arange(1, len(adam["best_kge_history"]) + 1),
                      "best_kge": adam["best_kge_history"]}).to_csv(out / f"iceland_adam_{slug}.csv", index=False)
        results.append({"mode": label, "n_params": G, "dds_cal": dc["kge"], "dds_val": dv["kge"],
                        "adam_cal": ac["kge"], "adam_val": av["kge"],
                        "dds_cost": dds_cost, "adam_cost": adam_cost})

    summary = {"outlet_seg": args.outlet_seg, "n_reaches": n, "n_gauges": len(gauges),
               "gauge_names": names, "default_cal_kge": dk,
               "cal_window": [str(t_cal[0].date()), str(t_cal[-1].date())],
               "val_window": [str(t_val[0].date()), str(t_val[-1].date())],
               "parameterizations": results}
    (out / "iceland_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out/'iceland_summary.json'}")

    if HAS_MPL and results:
        nparams = [r["n_params"] for r in results]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        axL.axhline(dk, color="gray", ls=":", lw=1.2, label=f"default n ({dk:.2f})")
        axL.plot(nparams, [r["dds_val"] for r in results], "C0-o", label="DDS")
        axL.plot(nparams, [r["adam_val"] for r in results], "C3-s", label="Adam")
        axL.set_xscale("log"); axL.set_xlabel("number of calibrated parameters")
        axL.set_ylabel("multi-gauge validation KGE")
        axL.set_title(f"Iceland basin (seg {args.outlet_seg}, {n} reaches, {len(gauges)} gauges):\nskill vs. parameter count")
        axL.legend(); axL.grid(alpha=0.3)
        axR.plot(nparams, [r["dds_cost"] for r in results], "C0-o", label="DDS")
        axR.plot(nparams, [r["adam_cost"] for r in results], "C3-s", label="Adam")
        axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel("number of calibrated parameters")
        axR.set_ylabel(f"fwd-pass-equiv. evals to within {CONV_TOL} KGE")
        axR.set_title("calibration cost vs. parameter count")
        axR.legend(); axR.grid(alpha=0.3)
        fig.savefig(out / "iceland_scaling.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Wrote figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
