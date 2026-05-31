#!/usr/bin/env python3
"""Regenerate fig1/fig2 from cached sweep CSVs with the corrected cost metric.

Avoids re-running the full sweep: reads experiments/results/{bow_dds_*,bow_adam_*}.csv
and bow_summary.json, recomputes convergence cost as forward-pass-equivalent evals to
reach within CONV_TOL of each mode's achieved optimum, and redraws the figures.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bow_at_banff_dds_vs_adam import evals_to_within, CONV_TOL, ADAM_PASSES_PER_EPOCH

R = Path("experiments/results")
summary = json.loads((R / "bow_summary.json").read_text())
have_val = summary.get("validation_window") is not None
modes = [(p["mode"], p["mode"].replace("-", "_"), p) for p in summary["parameterizations"]]

dds = {slug: pd.read_csv(R / f"bow_dds_{slug}.csv") for _, slug, _ in modes}
adam = {slug: pd.read_csv(R / f"bow_adam_{slug}.csv") for _, slug, _ in modes}

# --- Figure 1: per-reach convergence ---
slug_pr = "per_reach"
d = dds[slug_pr]; a = adam[slug_pr]
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(d["eval"], d["best_kge_q25"], d["best_kge_q75"],
                color="C0", alpha=0.25, label="DDS IQR (seeds)")
ax.plot(d["eval"], d["best_kge_median"], "C0-", lw=1.5, label="DDS median")
ax.plot(ADAM_PASSES_PER_EPOCH * a["epoch"], a["best_kge"], "C3-", lw=1.5,
        label="Adam (dRoute AD)")
ax.set_xlabel("forward-pass-equivalent model evaluations (Adam epoch = fwd + reverse)")
ax.set_ylabel("best-so-far calibration KGE")
ax.set_title(f"Bow at Banff: gradient vs. derivative-free "
             f"({summary['n_reaches']} per-reach parameters)")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.savefig(R / "fig1_convergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# --- Figure 2: scaling ---
nparams = [p["n_params"] for _, _, p in modes]
dds_skill = [p["dds"]["val_kge" if have_val else "cal_kge"] for _, _, p in modes]
adam_skill = [p["adam"]["val_kge" if have_val else "cal_kge"] for _, _, p in modes]
default_skill = summary["default"]["val_kge" if have_val else "cal_kge"]
lab = "validation" if have_val else "calibration"

dds_cost, adam_cost = [], []
for mode, slug, p in modes:
    best = p.get("mode_best_cal_kge", max(p["dds"]["cal_kge"], p["adam"]["cal_kge"]))
    ds = evals_to_within(dds[slug]["best_kge_median"].values, best)
    as_ = evals_to_within(adam[slug]["best_kge"].values, best)
    dds_cost.append(ds if ds else dds[slug].shape[0])
    adam_cost.append(as_ * ADAM_PASSES_PER_EPOCH if as_ else adam[slug].shape[0] * ADAM_PASSES_PER_EPOCH)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
axL.axhline(default_skill, color="gray", ls=":", lw=1.2,
            label=f"default n (KGE={default_skill:.2f})")
axL.plot(nparams, dds_skill, "C0-o", label="DDS")
axL.plot(nparams, adam_skill, "C3-s", label="Adam")
axL.set_xlabel("number of calibrated parameters")
axL.set_ylabel(f"{lab} KGE (best params)")
axL.set_title("Skill vs. parameter count"); axL.set_xscale("log")
lo = min(default_skill, *dds_skill, *adam_skill); hi = max(*dds_skill, *adam_skill)
axL.set_ylim(lo - 0.05, hi + 0.03)
axL.legend(loc="best"); axL.grid(alpha=0.3)

axR.plot(nparams, dds_cost, "C0-o", label="DDS")
axR.plot(nparams, adam_cost, "C3-s", label="Adam")
axR.set_xlabel("number of calibrated parameters")
axR.set_ylabel(f"fwd-pass-equiv. evals to within {CONV_TOL} KGE of optimum")
axR.set_title("Calibration cost vs. parameter count")
axR.set_xscale("log"); axR.set_yscale("log")
axR.legend(loc="best"); axR.grid(alpha=0.3)
fig.savefig(R / "fig2_scaling.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("regenerated fig1_convergence.png, fig2_scaling.png")
print("nparams:", nparams)
print("DDS cost (fwd-eq):", dds_cost)
print("Adam cost (fwd-eq):", adam_cost)
print(f"skill {lab}: default={default_skill:.3f} dds={[round(x,3) for x in dds_skill]} adam={[round(x,3) for x in adam_skill]}")
