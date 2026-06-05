# SPDX-License-Identifier: Apache-2.0
"""Joint multi-gauge routing calibration on Bow at Calgary via dRoute AD gradients.

Calibrates ALL routing degrees of freedom together against the multi-gauge objective,
to stop the lake routing from over-attenuating the (well-calibrated) upstream gauges:

  * per-reach Manning's n                         (107 params)
  * every inline lake/reservoir rating            (q_ref/exp/q_min/spill_coef)
  * every subgrid lake store rating               (subgrid_q_ref/subgrid_exp)

Loss = mean over the CALIBRATION gauges of (1 - KGE_gauge); the analytical dKGE/dQ at
each gauge seeds a single reverse pass via compute_gradients_timeseries(reach_ids, ...).
Adam optimises all parameters jointly from dRoute's exact gradients. Daily timestep
(light AD tape over the multi-year window). Runs on the SUMMA-calibrated runoff (async-DDS).

All WSC gauges that map into the network are EVALUATED and saved for the figures (10 on the
Bow-at-Calgary hydrofabric), but only the subset routing can actually fit is used as a
calibration TARGET: gauges whose default-lakes KGE is below --min-calib-kge are dropped from
the loss, because a severely negative KGE there is volume bias (KGE's beta = sim_mean/obs_mean)
-- a runoff problem -- and routing conserves volume, so Manning's n / lake ratings move timing
and attenuation, not the mean. Including them would waste gradient on the unfixable.

This is the high-dimensional gradient calibration the library exists to make cheap:
~200 parameters, one forward + one reverse pass per epoch.
"""
import argparse
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
import droute

from reservoir_rules_adam import (
    D, OBS, RES, DT_DAY, CAL, build_network, load_inputs, kge, dloss_dsim, mask_spinup,
    BOUNDS as LAKE_BOUNDS,
)

# All WSC gauges on the Bow-at-Calgary hydrofabric (label, station, seg). The first five are the
# nested mainstem gauges (upstream -> downstream); the rest are tributary gauges.
ALL_GAUGES = [("Lake Louise", "05BA001", 291), ("Banff", "05BB001", 270),
              ("Seebe", "05BE004", 199), ("Cochrane", "05BH005", 390),
              ("Calgary (outlet)", "05BH004", 402),
              ("Pipestone R.", "05BA002", 314), ("Goat Ck", "05BC008", 261),
              ("Kananaskis R.", "05BF003", 179), ("Waiparous Ck", "05BG006", 245),
              ("Jumpingpound Ck", "05BH015", 139)]

MANNING_BND = (0.02, 0.12, "log")
SUBGRID_BND = {"sq_ref": (0.05, 1000.0, "log"), "sexp": (1.0, 3.0, "lin")}


def _unit(lo, hi, tr, v):
    if tr == "log":
        return (np.log(max(v, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (v - lo) / (hi - lo)


def _phys(lo, hi, tr, u):
    u = min(max(u, 0.0), 1.0)
    if tr == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def _dphys_du(lo, hi, tr, u):
    if tr == "log":
        return _phys(lo, hi, tr, u) * (np.log(hi) - np.log(lo))
    return hi - lo


def run(runoff_path, epochs=80, lr=0.02, min_calib_kge=-2.0):
    from droute.lake_preprocessor import apply_lake_config_to_network
    inp = load_inputs(runoff_path)
    seg_ids = inp["seg_ids"]; id_to_idx = inp["id_to_idx"]
    runoff = inp["daily_runoff"]; dates = inp["dates"]; n_seg = len(seg_ids)

    with open(f"{D}/settings/dRoute/droute_lakes.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cls = {"inline": raw.get("inline_lakes", {}) or {}, "subgrid": raw.get("subgrid_lakes", {}) or {}}
    inline_segs = [int(s) for s in cls["inline"] if int(s) in id_to_idx]
    subgrid_segs = [int(s) for s in cls["subgrid"] if int(s) in id_to_idx]

    gauges = [(lab, sid, seg, id_to_idx[seg]) for lab, sid, seg in ALL_GAUGES if seg in id_to_idx]
    gauge_idx = [g[3] for g in gauges]   # ALL gauges -- evaluated + saved for the figures
    obs = {}
    for lab, sid, seg, gi in gauges:
        o = pd.read_csv(f"{OBS}/wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
        obs[gi] = mask_spinup(o.reindex(dates).values, dates)   # exclude spin-up year
    calib_idx = list(gauge_idx)   # calibration-target subset; narrowed after default KGE is known

    # ---- build parameter vector (unit space) ----
    # each entry: (kind, key, name, lo, hi, tr); value lives in U[]
    specs = []
    for i in range(n_seg):
        specs.append(("manning", i, "manning", *MANNING_BND))
    for seg in inline_segs:
        for nm in ("q_ref", "exp", "q_min", "spill"):
            specs.append(("lake", seg, nm, *LAKE_BOUNDS[nm]))
    for seg in subgrid_segs:
        specs.append(("subgrid", seg, "sq_ref", *SUBGRID_BND["sq_ref"]))
        specs.append(("subgrid", seg, "sexp", *SUBGRID_BND["sexp"]))
    P = len(specs)
    print(f"network {n_seg} reaches | {len(inline_segs)} inline lakes | {len(subgrid_segs)} subgrid | "
          f"{P} parameters | {len(gauges)} gauges")

    def build_apply(U):
        net = build_network(seg_ids, inp["downstream_idx"], inp["lengths"], inp["slopes"])
        apply_lake_config_to_network(net, cls, id_to_idx)
        for k, (kind, key, nm, lo, hi, tr) in enumerate(specs):
            v = _phys(lo, hi, tr, U[k])
            if kind == "manning":
                net.get_reach(key).manning_n = v
            elif kind == "lake":
                r = net.get_reach(id_to_idx[key])
                setattr(r, {"q_ref": "lake_q_ref", "exp": "lake_exp",
                            "q_min": "lake_q_min", "spill": "lake_spill_coef"}[nm], v)
            else:
                r = net.get_reach(id_to_idx[key])
                setattr(r, "subgrid_q_ref" if nm == "sq_ref" else "subgrid_exp", v)
        return net

    def route(net, record):
        c = droute.RouterConfig(); c.dt = DT_DAY; c.enable_gradients = record
        rt = droute.MuskingumCungeRouter(net, c)
        order = np.asarray(net.topological_order(), dtype=int)
        if record:
            rt.reset_gradients(); rt.start_recording()
        Q = np.zeros((len(runoff), n_seg))
        for t in range(len(runoff)):
            for idx in order:
                rt.set_lateral_inflow(int(idx), float(runoff[t, idx]))
            rt.route_timestep()
            if record:
                rt.record_outputs(calib_idx)   # record only the calibration-target gauges
            Q[t, order] = rt.get_all_discharges()
        if record:
            rt.stop_recording()
        return Q, rt

    def mean_kge(Q, idxs=None):
        return float(np.nanmean([kge(Q[:, gi], obs[gi]) for gi in (idxs or gauge_idx)]))

    grad_attr = {"q_ref": "grad_lake_q_ref", "exp": "grad_lake_exp",
                 "q_min": "grad_lake_q_min", "spill": "grad_lake_spill_coef",
                 "sq_ref": "grad_subgrid_q_ref", "sexp": "grad_subgrid_exp"}

    # ---- init U from HydroLAKES defaults / manning 0.035 ----
    net0 = build_network(seg_ids, inp["downstream_idx"], inp["lengths"], inp["slopes"])
    apply_lake_config_to_network(net0, cls, id_to_idx)
    U = np.zeros(P)
    for k, (kind, key, nm, lo, hi, tr) in enumerate(specs):
        if kind == "manning":
            v = float(net0.get_reach(key).manning_n) or 0.035
        elif kind == "lake":
            r = net0.get_reach(id_to_idx[key])
            v = float(getattr(r, {"q_ref": "lake_q_ref", "exp": "lake_exp",
                                   "q_min": "lake_q_min", "spill": "lake_spill_coef"}[nm]))
            if nm == "exp" and v <= 0: v = 1.5
            if nm == "spill" and v <= 0: v = 1.0
        else:
            r = net0.get_reach(id_to_idx[key])
            v = float(getattr(r, "subgrid_q_ref" if nm == "sq_ref" else "subgrid_exp"))
            if nm == "sq_ref" and v <= 0: v = 1.0
            if nm == "sexp" and v <= 0: v = 1.0
        U[k] = min(max(_unit(lo, hi, tr, v), 0.0), 1.0)

    U_init = U.copy()   # default-lakes / HydroLAKES starting rules
    # baseline (no lakes) and default-lakes KGE for reference (all gauges)
    Qb = route(build_network(seg_ids, inp["downstream_idx"], inp["lengths"], inp["slopes"]), False)[0]
    Qd = route(build_apply(U_init), False)[0]
    kge_base = mean_kge(Qb); kge0 = mean_kge(Qd)
    # Narrow the calibration target to gauges routing can fit: drop default-lakes KGE < threshold
    # (severe volume bias = runoff problem, unfixable by routing). All gauges are still evaluated.
    kdef = {gi: kge(Qd[:, gi], obs[gi]) for gi in gauge_idx}
    calib_idx[:] = [gi for gi in gauge_idx if np.isfinite(kdef[gi]) and kdef[gi] >= min_calib_kge]
    dropped = [(lab, sid) for lab, sid, seg, gi in gauges if gi not in calib_idx]
    print(f"baseline (no lakes) mean KGE (all {len(gauge_idx)}) = {kge_base:.4f}")
    print(f"epoch  0: mean KGE (all) = {kge0:.4f}  (lakes, default rules)")
    if dropped:
        print(f"  dropped from calibration (default KGE < {min_calib_kge}, volume bias): "
              + ", ".join(f"{lab}({sid})" for lab, sid in dropped))
    print(f"  calibrating on {len(calib_idx)} gauges; default mean KGE (calib) = "
          f"{mean_kge(Qd, calib_idx):.4f}")

    m = np.zeros(P); v = np.zeros(P); b1, b2, eps = 0.9, 0.999, 1e-8
    best = {"kge": mean_kge(Qd, calib_idx), "U": U.copy(), "epoch": 0}
    G = len(calib_idx)
    for ep in range(1, epochs + 1):
        net = build_apply(U)
        Q, rt = route(net, record=True)
        dL = [(dloss_dsim(Q[:, gi], obs[gi]) / G).tolist() for gi in calib_idx]
        rt.compute_gradients_timeseries(calib_idx, dL)
        # read grads, Adam step in unit space
        g = np.zeros(P)
        for k, (kind, key, nm, lo, hi, tr) in enumerate(specs):
            r = net.get_reach(key if kind == "manning" else id_to_idx[key])
            gp = float(getattr(r, "grad_manning_n" if kind == "manning" else grad_attr[nm]))
            g[k] = gp * _dphys_du(lo, hi, tr, U[k])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** ep); vh = v / (1 - b2 ** ep)
        U = np.clip(U - lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        ke = mean_kge(route(build_apply(U), False)[0], calib_idx)   # optimization metric (calib gauges)
        if ke > best["kge"]:
            best = {"kge": ke, "U": U.copy(), "epoch": ep}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:2d}: calib mean KGE = {ke:.4f}  best = {best['kge']:.4f} (ep {best['epoch']})")

    # ---- final per-gauge report (baseline vs default-lakes vs joint-calibrated), ALL gauges ----
    Qc = route(build_apply(best["U"]), False)[0]
    is_calib = np.array([gi in calib_idx for _, _, _, gi in gauges])
    # save routed series at ALL gauges for figure generation (no recalibration needed)
    import os
    os.makedirs(RES, exist_ok=True)
    np.savez(f"{RES}/joint_calib.npz",
             dates=np.asarray(dates, dtype="datetime64[ns]"),
             gauge_labels=np.array([g[0] for g in gauges]),
             gauge_stations=np.array([g[1] for g in gauges]),
             gauge_segs=np.array([g[2] for g in gauges]),
             gauge_is_calib=is_calib,                       # True = calibration target, False = eval-only
             q_base=np.column_stack([Qb[:, gi] for _, _, _, gi in gauges]),
             q_lake_def=np.column_stack([Qd[:, gi] for _, _, _, gi in gauges]),
             q_lake_cal=np.column_stack([Qc[:, gi] for _, _, _, gi in gauges]),
             obs=np.column_stack([obs[gi] for _, _, _, gi in gauges]),
             kge_base=mean_kge(Qb, calib_idx), kge_def=mean_kge(Qd, calib_idx), kge_cal=best["kge"])
    print(f"\ncalib mean KGE -- baseline(no lakes) {mean_kge(Qb, calib_idx):.3f} | "
          f"lakes(default) {mean_kge(Qd, calib_idx):.3f} | "
          f"lakes(joint-calibrated) {best['kge']:.3f} (ep {best['epoch']})")
    print(f"\n{'gauge':18} {'station':9} {'cal?':>4} {'baseline':>9} {'lakes_def':>9} {'lakes_cal':>9}")
    for lab, sid, seg, gi in gauges:
        flag = "Y" if gi in calib_idx else "-"
        print(f"{lab:18} {sid:9} {flag:>4} {kge(Qb[:, gi], obs[gi]):>9.3f} "
              f"{kge(Qd[:, gi], obs[gi]):>9.3f} {kge(Qc[:, gi], obs[gi]):>9.3f}")
    return kge_base, kge0, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runoff", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--min-calib-kge", type=float, default=-2.0,
                    help="drop gauges with default-lakes KGE below this from the calibration target "
                         "(volume bias unfixable by routing); all gauges are still evaluated")
    a = ap.parse_args()
    run(a.runoff, epochs=a.epochs, lr=a.lr, min_calib_kge=a.min_calib_kge)
