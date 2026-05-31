# dRoute experiments (GMD paper)

Reproducibility scaffold for the dRoute methods paper. Each script is self-contained
and binds to the real `droute` API.

## Bow at Banff — DDS vs. Adam

`bow_at_banff_dds_vs_adam.py` calibrates per-reach Manning's n for the semi-distributed
Bow-at-Banff network (29 reaches) with Muskingum–Cunge routing, comparing:

- **DDS** (Tolson & Shoemaker, 2007) — derivative-free, the conventional hydrology baseline.
- **Adam** (Kingma & Ba, 2014) — gradient-based, using dRoute's CoDiPack timeseries AD.

### Run

```bash
# build droute first (from repo root): pip install -e .  with -DDMC_BUILD_PYTHON=ON
python experiments/bow_at_banff_dds_vs_adam.py \
    --domain-dir /path/to/domain_Bow_at_Banff_semi_distributed \
    --cal-start 2008-10-01 --cal-end 2010-09-30 \
    --adam-epochs 200 --dds-evals 2000
```

Outputs land in `experiments/results/` (history CSVs, `bow_summary.json`, convergence
and hydrograph PNGs).

### The headline GMD claim

Gradient-based calibration should reach a comparable or better objective in
**far fewer model evaluations** than DDS, and the gap should **widen as parameter
dimension grows** (per-reach n = 29 params). The paper figure is *best-so-far
objective vs. number of model evaluations* for both methods, plus a *parameter-count
sweep* (e.g. lumped n → grouped n → per-reach n) showing DDS evaluations exploding
while Adam stays roughly flat.

### Results (calibrated SUMMA runoff, cal 2006 / val 2007)

Run on SYMFLUENCE-calibrated Bow-at-Banff runoff (`--runoff-file ...best_calibrated/output/bow_calibrated_timestep.nc`):

- **Efficiency (fig1):** Adam reaches the per-reach KGE optimum (0.729 cal / 0.827 val)
  in ~50 forward-pass-equivalents; DDS median needs ~250 — about 5× more.
- **Cost scaling (fig2 right):** fwd-pass-equiv. evals to converge (within 0.005 KGE of
  the achieved optimum): DDS **8 → 34 → 118** for 1/4/29 params; Adam **flat at 44**.
  Crossover ≈ 6 params; gradient cost is dimension-independent.
- **Skill (fig2 left):** essentially flat (0.824 → 0.827 val) across 1/4/29 params —
  Bow-at-Banff is a *low routing-leverage* case (short reaches; SUMMA gamma routing
  already sets within-basin timing). Calibration's real lift is default 0.66 → 0.82.

So the efficiency/scaling claim holds; the *skill* gain from more routing parameters
does not — that needs a higher-leverage domain (longer river / larger network /
sub-daily metric).

### Done

1. ~~Fair objective axis~~ — both methods on best-so-far KGE vs. evaluations.
2. ~~Fair eval counting~~ — Adam counted in forward-pass-equivalents (fwd+reverse, 2×
   epochs); cost metric is "evals to within 0.005 KGE of the achieved optimum" (the old
   "95% of best" was trivially hit just above the default and made DDS look cheaper).
3. ~~Calibration/validation split~~ — cal 2006 / val 2007.
4. ~~Multiple DDS seeds~~ — 10 seeds, median + IQR band.

`regenerate_figs.py` rebuilds fig1/fig2 from cached CSVs without re-running the sweep.

### Remaining TODOs

5. **Gradient-correctness panel** — finite-difference vs. AD (reuse
   `tests/test_gradient_verification.cpp`) as a supporting figure.
6. **Higher-leverage domain** — to make the parameter-scaling *skill* argument land.
7. **Freeze the data** — archive calibrated SUMMA runoff, topology, observations; cite
   the Zenodo DOI in code/data-availability.
