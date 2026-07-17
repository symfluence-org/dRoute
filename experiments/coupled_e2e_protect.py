# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled SUMMA + dRoute e2e calibration with a SMALL-GAUGE-PROTECTING objective.

Same joint 26-param coupled chain as coupled_e2e_dds_calib.py (re-run SUMMA -> route -> multi-gauge
KGE), but the objective is worst-protecting instead of a plain mean, so the optimizer can't sacrifice
small/hard gauges (05BH015, 05BF003) to boost the mainstem aggregate:

    objective = (1 - W_CVAR) * mean(KGE)  +  W_CVAR * mean(worst-CVAR_K KGE)

over the fixed gauge subset, each KGE clipped at CLIP to bound catastrophic values. The worst-K
(CVaR / expected-shortfall) term puts half the weight on the laggards. Warm-started from the
mean-objective run's best parameters so the budget goes toward lifting the weak gauges.
"""

import json
import logging
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import numpy as np

import experiments._coupled_eval as E

logging.basicConfig(level=logging.INFO)

N_CORES = 10
KGE_FLOOR = -2.0          # gauge enters the objective subset if its warm-start KGE >= this
CLIP = -1.0               # clip each gauge KGE at this floor inside the objective
W_CVAR = 0.5              # weight on the worst-K (CVaR) term vs the mean
CVAR_K = 3               # number of worst gauges averaged for the CVaR term
SEED = 20260605
# COLD start (warm-starting from the mean-optimum traps DDS where the small gauges are already
# sacrificed). Starting from SUMMA defaults keeps the laggards healthy (05BH015 ~+0.22) and the
# CVaR objective is free to find a balanced optimum. Set WARM_START to a best.json path to warm-start.
WARM_START = None
RESULTS = Path(__file__).parent / "results_calgary" / "coupled_e2e_protect"
ASYNC_CFG = {
    'NUMBER_OF_ITERATIONS': 49, 'ASYNC_DDS_POOL_SIZE': 12, 'ASYNC_DDS_BATCH_SIZE': N_CORES,
    'DDS_R': 0.25, 'MAX_STAGNATION_BATCHES': 10_000, 'OPTIMIZATION_METRIC': 'KGE',
}

_W = {}
NAMES = None
LO = None
HI = None
N_SUMMA = 0
SUBSET = []


def _param_space(logger):
    from droute.calibration.parameter_manager import DRouteParameterManager
    from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
    spm = SUMMAParameterManager(E.CFG, logger, E.SUMMA_SETTINGS)
    dpm = DRouteParameterManager(E._droute_cfg(""), logger, E.DROUTE_SETTINGS)
    s_names, d_names = list(spm.all_param_names), list(dpm.all_param_names)
    sb, db = spm.param_bounds, dpm.param_bounds
    names = s_names + d_names
    lo = np.array([sb[n]['min'] for n in s_names] + [db[n]['min'] for n in d_names], float)
    hi = np.array([sb[n]['max'] for n in s_names] + [db[n]['max'] for n in d_names], float)
    s_init, d_init = spm.get_initial_parameters(), dpm.get_initial_parameters()
    x0 = np.array([float(np.mean(s_init[n])) for n in s_names] +
                  [float(d_init[n]) for n in d_names], float)
    return names, lo, hi, np.clip(x0, lo, hi), len(s_names)


def _init(names, lo, hi, n_summa, subset):
    global NAMES, LO, HI, N_SUMMA, SUBSET
    NAMES, LO, HI, N_SUMMA, SUBSET = names, np.asarray(lo, float), np.asarray(hi, float), n_summa, list(subset)
    run = Path(tempfile.mkdtemp(prefix="cplp_"))
    E.make_run_dir(run)
    _W['run'] = run
    _W['log'] = logging.getLogger('cw')


def _pergauge(x_norm):
    x = LO + np.asarray(x_norm) * (HI - LO)
    summa = {NAMES[i]: float(x[i]) for i in range(N_SUMMA)}
    droute = {NAMES[i]: float(x[i]) for i in range(N_SUMMA, len(NAMES))}
    run, log = _W['run'], _W['log']
    if not E.apply_summa(run, summa, log):
        return {}
    ts = E.run_summa(run, log)
    if ts is None:
        return {}
    return E.route_pergauge(ts, droute, log)


def _aggregate(perg):
    """Worst-protecting objective: blend of mean and CVaR (mean of worst-K), KGE clipped at CLIP."""
    vals = []
    for st in SUBSET:
        k = perg.get(st, np.nan)
        vals.append(CLIP if not np.isfinite(k) else max(float(k), CLIP))
    a = np.sort(np.asarray(vals, float))          # ascending -> worst first
    cvar = float(a[:min(CVAR_K, len(a))].mean())
    return (1.0 - W_CVAR) * float(a.mean()) + W_CVAR * cvar


def _plain_mean(perg):
    vals = [perg[st] for st in SUBSET if st in perg and np.isfinite(perg[st])]
    return float(np.mean(vals)) if vals else float('nan')


def evaluate(x_norm):
    try:
        perg = _pergauge(x_norm)
        if not perg:
            return -1e9
        return _aggregate(perg)
    except (RuntimeError, ValueError, OSError, KeyError, FloatingPointError):
        return -1e9


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    log = logging.getLogger('cplp')
    from symfluence.optimization.optimizers.algorithms.async_dds import AsyncDDSAlgorithm

    names, lo, hi, x0_default, n_summa = _param_space(log)
    if WARM_START is not None:
        warm = json.loads(Path(WARM_START).read_text())['best_params']
        x0 = np.clip(np.array([float(warm.get(n, x0_default[i])) for i, n in enumerate(names)], float), lo, hi)
        start = f"warm-started from {Path(WARM_START).name}"
    else:
        x0 = x0_default                              # cold start from SUMMA/dRoute defaults
        start = "cold start from defaults"
    log.info(f"joint param space: {len(names)} params ({n_summa} SUMMA + {len(names)-n_summa} dRoute), {start}")
    denorm = lambda xn: lo + np.asarray(xn) * (hi - lo)  # noqa: E731
    x0_norm = (x0 - lo) / (hi - lo)

    _init(names, lo, hi, n_summa, [])
    t0 = time.time()
    base_perg = _pergauge(x0_norm)
    subset = sorted([st for st, k in base_perg.items() if np.isfinite(k) and k >= KGE_FLOOR])
    global SUBSET
    SUBSET = subset
    base_obj, base_mean = _aggregate(base_perg), _plain_mean(base_perg)
    log.info(f"[baseline] {time.time()-t0:.0f}s | objective={base_obj:.4f} mean={base_mean:.4f} "
             f"over {len(subset)} gauges | per-gauge={ {k: round(v,3) for k,v in base_perg.items()} }")

    trace = []
    best = {'obj': base_obj, 'mean': base_mean, 'params': dict(zip(names, x0)), 'it': 0, 'perg': base_perg}

    def evaluate_population(pop_norm, iteration):
        return [s if s > -1e8 else PENALTY for s in pool.map(evaluate, [r for r in np.asarray(pop_norm)])]

    def record_iteration(it, score, params):
        trace.append({'batch': int(it), 'best_obj': round(float(score), 5),
                      'elapsed_min': round((time.time() - t0) / 60, 1)})
        (RESULTS / "trace.json").write_text(json.dumps(trace))

    def update_best(score, params, it):
        if score > best['obj']:
            best.update(obj=float(score), params=dict(params), it=int(it))
            _save(RESULTS, best, base_obj, base_mean, base_perg, subset)

    def log_progress(name, it, score, improvements, batch):
        log.info(f"[{name} batch {it}] best_obj={score:.4f} (+{improvements}/{batch}) "
                 f"[{(time.time()-t0)/60:.0f} min, base_obj {base_obj:.4f}]")

    algo = AsyncDDSAlgorithm(ASYNC_CFG, log)
    global PENALTY
    PENALTY = algo.penalty_score

    with mp.Pool(N_CORES, initializer=_init, initargs=(names, lo, hi, n_summa, subset)) as pool:
        result = algo.optimize(
            n_params=len(names), evaluate_solution=lambda x, it: evaluate(x),
            evaluate_population=evaluate_population, denormalize_params=lambda xn: dict(zip(names, denorm(xn))),
            record_iteration=record_iteration, update_best=update_best, log_progress=log_progress,
            num_processes=N_CORES, initial_guess=x0_norm,
        )
        best_x = np.array([best['params'][n] for n in names], float)
        best_perg = pool.apply(_pergauge, ((best_x - lo) / (hi - lo),))

    best['mean'] = _plain_mean(best_perg)
    _save(RESULTS, best, base_obj, base_mean, base_perg, subset, best_perg)
    log.info(f"\n[done] {(time.time()-t0)/60:.0f} min | base obj={base_obj:.4f} mean={base_mean:.4f} -> "
             f"best obj={best['obj']:.4f} mean={best['mean']:.4f} (batch {best['it']}) "
             f"| algo best={result['best_score']:.4f}")
    log.info(f"[best per-gauge] { {k: round(v,3) for k,v in best_perg.items() if np.isfinite(v)} }")


def _save(results, best, base_obj, base_mean, base_perg, subset, best_perg=None):
    out = dict(
        objective="0.5*mean + 0.5*mean(worst-%d), KGE clipped at %.1f" % (CVAR_K, CLIP),
        best_objective=best['obj'], best_mean_kge=best.get('mean'), baseline_objective=base_obj,
        baseline_mean_kge=base_mean, best_batch=best['it'], calib_gauges=list(subset),
        best_params={n: float(v) for n, v in best['params'].items()},
        baseline_per_gauge={k: (round(v, 4) if np.isfinite(v) else None) for k, v in base_perg.items()},
    )
    if best_perg is not None:
        out['best_per_gauge'] = {k: (round(v, 4) if np.isfinite(v) else None) for k, v in best_perg.items()}
    (results / "best.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
