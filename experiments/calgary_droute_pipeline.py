# SPDX-License-Identifier: Apache-2.0
"""Bow-at-Calgary dRoute finishing pipeline (runoff-parameterized).

One entry point that takes a SUMMA runoff file and produces the full dRoute story:

  1. Route the runoff through the C++ network three ways, daily timestep:
       (a) BASELINE   - plain channels, no lakes
       (b) LAKES      - inline lakes/reservoirs with HydroLAKES-default operating rules
       (c) CALIBRATED - lakes with per-reservoir operating rules learned via AD gradients
  2. Evaluate each at the 5 nested Bow mainstem WSC gauges (KGE / NSE).
  3. Emit a metrics CSV, a per-gauge KGE bar chart, and nested hydrographs.

Run on the current (uncalibrated) runoff now, and re-run with --runoff pointing at the
SUMMA-calibrated runoff once the ASYNC-DDS calibration completes:

    python experiments/calgary_droute_pipeline.py --label baseline
    python experiments/calgary_droute_pipeline.py --runoff <calibrated.nc> --label calibrated
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reuse the validated daily routing + lake + reservoir-Adam machinery
from reservoir_rules_adam import (
    D, OBS, RES, DT_DAY, CAL, build_network, route_daily, apply_lakes_and_get_reservoirs,
    load_inputs, set_reservoir_params, from_unit, kge, main as calibrate_reservoirs,
)

NESTED = [("Lake Louise", "05BA001", 291), ("Banff", "05BB001", 270),
          ("Seebe", "05BE004", 199), ("Cochrane", "05BH005", 390),
          ("Calgary (outlet)", "05BH004", 402)]


def route_all(net, daily_runoff):
    """Daily route, return discharge for ALL segments (n_time, n_seg), reach-indexed."""
    import droute
    c = droute.RouterConfig(); c.dt = DT_DAY; c.enable_gradients = False
    rt = droute.MuskingumCungeRouter(net, c)
    order = np.asarray(net.topological_order(), dtype=int)
    nt, ns = daily_runoff.shape
    out = np.zeros((nt, ns))
    for t in range(nt):
        for idx in order:
            rt.set_lateral_inflow(int(idx), float(daily_runoff[t, idx]))
        rt.route_timestep()
        out[t, order] = rt.get_all_discharges()   # topo-ordered -> scatter to reach index
    return out


def nse(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    s, o = s[m], o[m]
    if len(s) < 10:
        return np.nan
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runoff", default=None, help="SUMMA runoff netCDF (default: current uncalibrated)")
    ap.add_argument("--label", default="baseline", help="tag for output files")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.015)
    a = ap.parse_args()

    import os
    os.makedirs(RES, exist_ok=True)

    # --- per-reservoir reservoir-rule calibration (also loads inputs/network) ---
    print(f"=== reservoir-rule calibration ({a.label}) ===")
    kge0, best, res_idx, inp = calibrate_reservoirs(epochs=a.epochs, lr=a.lr, runoff_path=a.runoff)

    dates = inp["dates"]
    seg_to_idx = {int(s): i for i, s in enumerate(inp["seg_ids"])}

    # --- three routed variants (daily) ---
    print("=== routing variants: baseline / lakes / calibrated ===")
    net0 = build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    q_base = route_all(net0, inp["daily_runoff"])

    net1 = build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    apply_lakes_and_get_reservoirs(net1, inp["id_to_idx"])
    q_lake = route_all(net1, inp["daily_runoff"])

    net2 = build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    ri2 = apply_lakes_and_get_reservoirs(net2, inp["id_to_idx"])
    set_reservoir_params(net2, ri2, best["U"])
    q_cal = route_all(net2, inp["daily_runoff"])

    variants = {"baseline (no lakes)": q_base, "lakes (default)": q_lake, "lakes (calibrated)": q_cal}

    # --- evaluate at the 5 nested gauges ---
    rows = []
    fig, axes = plt.subplots(len(NESTED), 1, figsize=(13, 2.3 * len(NESTED)), sharex=True)
    colors = {"baseline (no lakes)": "#d9772b", "lakes (default)": "#1f78b4", "lakes (calibrated)": "#2ca02c"}
    for ax, (label, sid, seg) in zip(axes, NESTED):
        idx = seg_to_idx[seg]
        obs = pd.read_csv(f"{OBS}/wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
        obs = obs.reindex(dates).values
        ax.plot(dates, obs, "k", lw=1.1, label="WSC obs", zorder=5)
        for vname, q in variants.items():
            rows.append((a.label, label, sid, seg, vname,
                         round(kge(q[:, idx], obs), 3), round(nse(q[:, idx], obs), 3)))
            ax.plot(dates, q[:, idx], color=colors[vname], lw=0.8, alpha=0.85, label=vname)
        ax.set_title(f"{label} [{sid}, seg {seg}]  obs mean {np.nanmean(obs):.0f} m³/s", fontsize=9)
        ax.set_ylabel("Q (m³/s)", fontsize=8); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Bow at Calgary — dRoute routing ({a.label}): baseline vs lakes vs calibrated reservoirs",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{RES}/pipeline_{a.label}_hydrographs.png", dpi=150, bbox_inches="tight")

    tbl = pd.DataFrame(rows, columns=["run", "gauge", "station", "seg", "variant", "KGE", "NSE"])
    tbl.to_csv(f"{RES}/pipeline_{a.label}_metrics.csv", index=False)

    # --- summary: mean KGE across gauges per variant ---
    print(f"\n=== {a.label}: reservoir-rule Adam KGE {kge0:.3f} -> {best['kge']:.3f} (outlet) ===")
    print(tbl.pivot_table(index="variant", values=["KGE", "NSE"], aggfunc="mean").round(3).to_string())
    print("\nper-gauge KGE:")
    print(tbl.pivot_table(index="gauge", columns="variant", values="KGE").round(3).to_string())
    print(f"\nsaved -> {RES}/pipeline_{a.label}_metrics.csv")
    print(f"saved -> {RES}/pipeline_{a.label}_hydrographs.png")


if __name__ == "__main__":
    main()
