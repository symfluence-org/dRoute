# SPDX-License-Identifier: Apache-2.0
"""Bow-at-Calgary (multigauge domain) — calibrate dRoute reservoir operating rules (AD).

Reuses the validated reservoir-rule Adam machinery (reservoir_rules_adam) but points it at the
NEW 116-reach multigauge domain + the async-DDS-calibrated SUMMA runoff. Learns per-reservoir
operating rules (q_ref, exp, q_min, spill) by gradient descent on outlet KGE, then routes the
calibrated-reservoir variant and reports KGE + monthly climatology at the nested gauges — to
quantify how much of the regulation gap calibrated reservoir routing closes.
"""
import sys, glob
import numpy as np
import pandas as pd

EXP = "/Users/darri.eythorsson/compHydro/code/dRoute/experiments"
sys.path.insert(0, EXP)
import reservoir_rules_adam as rr   # noqa: E402

# --- point the machinery at the new domain ---
rr.D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary_multigauge"
rr.OBS = f"{rr.D}/observations/streamflow"
rr.RES = f"{EXP}/results_calgary_multigauge"
RUNOFF = f"{rr.D}/optimization/SUMMA/async-dds_bow_calgary_v1/final_evaluation/bow_calgary_v1_timestep.nc"
import os; os.makedirs(rr.RES, exist_ok=True)

NESTED = [("Lake Louise", "05BA001", 315), ("Banff", "05BB001", 282),
          ("Seebe", "05BE004", 211), ("Cochrane", "05BH005", 438),
          ("Calgary (outlet)", "05BH004", 450)]


def route_all(net, daily):
    import droute
    c = droute.RouterConfig(); c.dt = rr.DT_DAY; c.enable_gradients = False
    rt = droute.MuskingumCungeRouter(net, c)
    order = np.asarray(net.topological_order(), dtype=int)
    nt, ns = daily.shape; out = np.zeros((nt, ns))
    for t in range(nt):
        for idx in order:
            rt.set_lateral_inflow(int(idx), float(daily[t, idx]))
        rt.route_timestep(); out[t, order] = rt.get_all_discharges()
    return out


def main():
    print("=== calibrating reservoir operating rules on the multigauge domain ===")
    kge0, best, res_idx, inp = rr.main(epochs=80, lr=0.015, runoff_path=RUNOFF)
    print(f"\n=== outlet reservoir-rule Adam: KGE {kge0:.3f} -> {best['kge']:.3f} ===")

    # route the three variants over ALL segments for nested-gauge + monthly eval
    sids = inp["id_to_idx"]; runoff = inp["daily_runoff"]; dates = inp["dates"]
    net0 = rr.build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    q_base = route_all(net0, runoff)
    netL = rr.build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    rr.apply_lakes_and_get_reservoirs(netL, sids); q_lake = route_all(netL, runoff)
    netC = rr.build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    ri = rr.apply_lakes_and_get_reservoirs(netC, sids); rr.set_reservoir_params(netC, ri, best["U"])
    q_cal = route_all(netC, runoff)

    cal = (dates >= pd.Timestamp(rr.CAL[0])) & (dates <= pd.Timestamp(rr.CAL[1]))
    print(f"\n=== KGE at nested gauges (calib {rr.CAL[0]}..{rr.CAL[1]}) ===")
    print(f"  {'gauge':22s} {'baseline':>9s} {'lakes':>9s} {'lakes+calib':>12s}")
    for label, sid, link in NESTED:
        idx = sids[link]
        o = pd.read_csv(f"{rr.OBS}/wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"].reindex(dates).values
        print(f"  {label:22s} {rr.kge(q_base[cal,idx],o[cal]):>+9.3f} {rr.kge(q_lake[cal,idx],o[cal]):>+9.3f} {rr.kge(q_cal[cal,idx],o[cal]):>+12.3f}")

    oidx = sids[450]
    obs_o = pd.read_csv(f"{rr.OBS}/wsc_05BH004_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"].reindex(dates)
    dfm = pd.DataFrame({"obs": obs_o.values, "base": q_base[:,oidx], "lake": q_lake[:,oidx], "calib": q_cal[:,oidx]}, index=dates)[cal]
    mc = dfm.groupby(dfm.index.month).mean()
    print("\n=== monthly mean at Calgary outlet (cms), calib period ===")
    print("  month: " + " ".join(f"{m:>5d}" for m in mc.index))
    for k in ("obs", "base", "lake", "calib"):
        print(f"  {k:6s}:" + " ".join(f"{v:5.0f}" for v in mc[k]))


if __name__ == "__main__":
    main()
