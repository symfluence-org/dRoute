---
title: "Joint SUMMA + dRoute calibration at Bow-at-Calgary"
date: "2026-06-08"
---

# Joint SUMMA + dRoute calibration at Bow-at-Calgary

**Question.** Does *joint* (co-optimized) SUMMA + dRoute calibration beat the *sequential* baseline
(SUMMA-then-reservoir-rules) at the regulated Bow-at-Calgary outlet (WSC 05BH004)?

**Answer.** Yes — outlet **KGE 0.868 vs the 0.83 sequential baseline**, measured identically (full
2011–2012 window, same KGE formula), with a robustness caveat noted below.

![Bow River at Calgary domain: WSC gauges (mainstem + tributary), TransAlta reservoirs and lakes, river network.](results_calgary/domain_map_gauges.png){width=4.3in}

## Setup

- **Domain.** `domain_Bow_at_Calgary_multigauge` — 116 subbasins, 9 interior WSC gauges + the Calgary
  outlet, correct drainage areas (Banff 2067, Calgary 7840 km²).
- **Chain.** SUMMA (land; runoff only — *land-only mode*, mizuRoute skipped) → dRoute Muskingum–Cunge
  routing + inline lakes + global reservoir operating rules. MC, not Saint-Venant (SV is
  ill-conditioned on this 116-reach domain).
- **Optimizer.** SYMFLUENCE COUPLED AsyncDDS, **22 joint parameters** (17 SUMMA + 5 dRoute: Manning *n*
  + 4 reservoir rules). 24 batches × pool 12 = **288 evaluations**, ~17 h, completed 2026-06-08 00:10.
- **Objective.** *Outlet-weighted* mean per-gauge KGE: the outlet (05BH004) carries weight 10
  (≈56 % of the score); the other gauges weight 1, acting as soft constraints. Flat-mean runs left the
  outlet near 0.73 — the weighting is what pushed it past the baseline.

## Result

Best weighted objective **0.7004** (iteration 23). Outlet hydrograph:

![Bow at Calgary outlet — joint-calibrated discharge vs observed, 2011–2012.](results_calgary_joint/fig1_outlet_hydrograph.png){width=6.2in}

Per-gauge KGE over the full 2011–2012 window:

| gauge   | KGE    | note                  |
|---------|--------|-----------------------|
| 05BH005 | 0.892  | Bow at Banff          |
| **05BH004** | **0.868** | **outlet — beats 0.83** |
| 05BE004 | 0.790  | Bow near Seebe        |
| 05BA001 | 0.567  | Bow at Lake Louise    |
| 05BB001 | 0.563  |                       |
| 05BG006 | 0.512  | Elbow at Bragg Creek  |
| 05BA002 | 0.497  |                       |
| 05BH015 | 0.062  | small tributary       |
| 05BF003 | 0.047  | small tributary       |
| 05BC008 | −5.5   | dropped (< KGE floor) |

![Nested multi-gauge hydrographs — joint SUMMA+dRoute (calibrated) vs observed.](results_calgary_joint/fig2_multigauge_hydrographs.png){width=6.4in}

![Per-gauge KGE; outlet in green, sequential baseline (0.83) dashed.](results_calgary_joint/fig3_kge_bars.png){width=5.6in}

- **Flat (unweighted) multi-gauge mean = 0.533** — higher than the earlier flat-mean run's 0.481, so
  outlet weighting *improved* the overall fit rather than trading it away.
- **Best parameters.** Manning *n* 0.047; reservoir q_ref_mult 0.83, exp 1.23, q_min_frac 0.32,
  spill_coef 0.15; + 17 SUMMA parameters (`bow_calgary_v1_asyncdds_best_params.json`).

## Robustness caveat

The 0.868 is an **annual-scale** number and is robust year-to-year (2011: 0.861, 2012: 0.872). The
remaining error is **seasonal volume timing**: within-year half-periods score lower —
Jan–Jun KGE ≈ 0.69 (spring freshet under-predicted, β ≈ 0.70) and Jul–Dec KGE ≈ 0.71 (autumn
over-predicted, β ≈ 1.22). These offsetting biases partly cancel at the annual scale that both the
objective and the 0.83 baseline use, so the comparison is fair and the joint approach wins — but the
absolute sub-seasonal skill is softer than 0.87 implies. The residual is the known headwater
**freshet / snowmelt-timing** gap (SUMMA runoff, no glaciers) — a land-model problem, not routing.

## Bottom line

- **Defensible.** Joint SUMMA+dRoute calibration with an outlet-weighted objective beats the sequential
  SUMMA-then-reservoir baseline at Bow-at-Calgary, **0.87 vs 0.83 KGE** (identical window and metric).
- **State carefully.** Sub-seasonal skill is ~0.69–0.71; the annual headline benefits from seasonal
  bias cancellation (true of both methods).

## Provenance

- Output: `domain_Bow_at_Calgary_multigauge/optimization/COUPLED/async-dds_bow_calgary_v1/`
  (`bow_calgary_v1_asyncdds_best_params.json`, `..._final_evaluation.json`,
  `final_evaluation/DROUTE/droute_streamflow.npz`).
- Code: dRoute PR **#11** (engine-agnostic MC|SV path, outlet-weighted objective, obs-dir resolution);
  SYMFLUENCE PR **#182** (pool-entry + settings, SUMMA land-only mode).
