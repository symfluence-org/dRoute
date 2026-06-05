# SPDX-License-Identifier: Apache-2.0
"""Evaluate Bow-at-Calgary routed flow (baseline vs lake-aware) against WSC gauges.

Loads the routed hydrographs from bow_at_calgary_lakes.py (calgary_routed.npz),
aggregates hourly -> daily, aligns with WSC daily-mean discharge at each nested
mainstem gauge, and reports KGE/NSE plus a multi-panel nested hydrograph figure.

Hydrology note: the SUMMA runoff has been calibrated (async-DDS), so the routed volumes
are unbiased and absolute KGE is meaningful at the mainstem gauges; the comparison is
baseline vs lakes (peak attenuation / reservoir regulation) over the nested multi-gauge
structure.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
OBS = f"{D}/observations/streamflow"
RES = "/Users/darri.eythorsson/compHydro/code/dRoute/experiments/results_calgary"

# nested Bow mainstem gauges, upstream -> downstream (label, station, seg)
NESTED = [
    ("Lake Louise", "05BA001", 291),
    ("Banff", "05BB001", 270),
    ("Seebe", "05BE004", 199),
    ("Cochrane", "05BH005", 390),
    ("Calgary (outlet)", "05BH004", 402),
]


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 10:
        return np.nan
    s, o = sim[m], obs[m]
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta = s.mean() / o.mean() if o.mean() > 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 10:
        return np.nan
    s, o = sim[m], obs[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def main():
    z = np.load(f"{RES}/calgary_routed.npz", allow_pickle=True)
    time = pd.to_datetime(z["time"])
    seg_ids = z["seg_ids"]
    seg_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    # hourly -> daily mean per series
    def daily(q_col):
        return pd.Series(q_col, index=time).resample("D").mean()

    rows = []
    fig, axes = plt.subplots(len(NESTED), 1, figsize=(13, 2.3 * len(NESTED)), sharex=True)
    for ax, (label, sid, seg) in zip(axes, NESTED):
        idx = seg_to_idx[seg]
        db = daily(z["q_base"][:, idx]); dl = daily(z["q_lake"][:, idx])
        obs = pd.read_csv(f"{OBS}/wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
        df = pd.DataFrame({"base": db, "lake": dl, "obs": obs}).dropna(subset=["obs"])
        df = df.loc["2011-01-01":"2014-12-31"]
        for tag in ("base", "lake"):
            rows.append((label, sid, seg, tag,
                         round(kge(df[tag].values, df["obs"].values), 3),
                         round(nse(df[tag].values, df["obs"].values), 3),
                         round(df[tag].mean(), 1), round(df["obs"].mean(), 1)))
        ax.plot(df.index, df["obs"], "k", lw=1.1, label="WSC obs", zorder=3)
        ax.plot(df.index, df["base"], color="#d9772b", lw=0.8, alpha=0.85, label="dRoute (no lakes)")
        ax.plot(df.index, df["lake"], color="#1f78b4", lw=0.8, alpha=0.85, label="dRoute (lakes)")
        ax.set_title(f"{label}  [{sid}, seg {seg}]  obs mean {df['obs'].mean():.0f} m³/s", fontsize=9)
        ax.set_ylabel("Q (m³/s)", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Bow at Calgary — nested multi-gauge routing (SUMMA → dRoute), baseline vs lake-aware",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f"{RES}/calgary_nested_hydrographs.png", dpi=150, bbox_inches="tight")

    tbl = pd.DataFrame(rows, columns=["gauge", "station", "seg", "run", "KGE", "NSE", "sim_mean", "obs_mean"])
    tbl.to_csv(f"{RES}/calgary_gauge_metrics.csv", index=False)
    print(tbl.to_string(index=False))
    print(f"\nsaved -> {RES}/calgary_nested_hydrographs.png")
    print(f"saved -> {RES}/calgary_gauge_metrics.csv")


if __name__ == "__main__":
    main()
