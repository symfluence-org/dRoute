# SPDX-License-Identifier: GPL-3.0-or-later
"""True coupled SUMMA + dRoute e2e calibration on Bow-at-Calgary via Async-DDS.

Jointly calibrates the 17 SUMMA hydrology parameters (13 local + 4 basin, from the project config's
PARAMS_TO_CALIBRATE/BASIN_PARAMS_TO_CALIBRATE) AND the 9 dRoute routing transfer-function
coefficients. Each evaluation re-runs SUMMA over the spin-up+calibration window (2010-2012) in an
isolated per-worker run directory, then routes the fresh runoff through dRoute (SV + lakes) and
scores the multi-gauge KGE -- i.e. the full coupled chain, not a fixed-forcing routing calibration.

Optimizer: SYMFLUENCE's AsyncDDSAlgorithm (pool + tournament selection + DDS perturbation), with
each batch of ASYNC_DDS_BATCH_SIZE candidates evaluated concurrently in a multiprocessing Pool of
N_CORES workers (each worker owns an isolated SUMMA run dir). Total evals = baseline + pool +
NUMBER_OF_ITERATIONS * batch (~500). At ~14 min/eval this is a ~12 h run on 10 cores.
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
KGE_FLOOR = -2.0
SEED = 20260604
RESULTS = Path(__file__).parent / "results_calgary" / "coupled_e2e"
ASYNC_CFG = {  # 10 (pool) + 49*10 (batches) = 500 calibration evaluations (+1 baseline)
    'NUMBER_OF_ITERATIONS': 49, 'ASYNC_DDS_POOL_SIZE': 10, 'ASYNC_DDS_BATCH_SIZE': N_CORES,
    'DDS_R': 0.2, 'MAX_STAGNATION_BATCHES': 10_000, 'OPTIMIZATION_METRIC': 'KGE',
}

# module-level state inherited by spawn workers (set via the initializer args)
_W = {}
NAMES = None
LO = None
HI = None
N_SUMMA = 0
SUBSET = []


def _param_space(logger):
    """Joint (names, lo, hi, x0) over SUMMA (17) + dRoute (9), with a sane default initial vector."""
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
    """Worker initializer: set the shared param space + create an isolated SUMMA run dir."""
    global NAMES, LO, HI, N_SUMMA, SUBSET
    NAMES, LO, HI, N_SUMMA, SUBSET = names, np.asarray(lo, float), np.asarray(hi, float), n_summa, list(subset)
    run = Path(tempfile.mkdtemp(prefix="cpl_"))
    E.make_run_dir(run)
    _W['run'] = run
    _W['log'] = logging.getLogger('cw')


def _pergauge(x_norm):
    """Apply params -> run SUMMA -> route; return {station: KGE} (empty dict on failure)."""
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


def evaluate(x_norm):
    """Coupled objective: mean KGE over the fixed calibration gauge subset (-1e9 on failure)."""
    try:
        perg = _pergauge(x_norm)
        vals = [perg[st] for st in SUBSET if st in perg and np.isfinite(perg[st])]
        return float(np.mean(vals)) if vals else -1e9
    except (RuntimeError, ValueError, OSError, KeyError, FloatingPointError):
        return -1e9


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    log = logging.getLogger('cpl')
    from symfluence.optimization.optimizers.algorithms.async_dds import AsyncDDSAlgorithm

    names, lo, hi, x0, n_summa = _param_space(log)
    log.info(f"joint param space: {len(names)} params ({n_summa} SUMMA + {len(names)-n_summa} dRoute)")
    denorm = lambda xn: lo + np.asarray(xn) * (hi - lo)  # noqa: E731
    x0_norm = (x0 - lo) / (hi - lo)

    # baseline coupled eval (serial) -> fixed calibration gauge subset + starting score
    _init(names, lo, hi, n_summa, [])           # main-process worker dir + space (subset filled below)
    t0 = time.time()
    base_perg = _pergauge(x0_norm)
    subset = sorted([st for st, k in base_perg.items() if np.isfinite(k) and k >= KGE_FLOOR])
    global SUBSET
    SUBSET = subset
    base_mean = float(np.mean([base_perg[st] for st in subset])) if subset else -1e9
    log.info(f"[baseline] {time.time()-t0:.0f}s | mean KGE={base_mean:.4f} over {len(subset)}/"
             f"{len(base_perg)} gauges | per-gauge={ {k: round(v,3) for k,v in base_perg.items()} }")

    trace, best = [], {'score': base_mean, 'params': dict(zip(names, x0)), 'it': 0}

    def evaluate_population(pop_norm, iteration):
        return [s if s > -1e8 else PENALTY
                for s in pool.map(evaluate, [row for row in np.asarray(pop_norm)])]

    def record_iteration(it, score, params):
        trace.append({'batch': int(it), 'best_kge': round(float(score), 5),
                      'elapsed_min': round((time.time() - t0) / 60, 1)})
        (RESULTS / "trace.json").write_text(json.dumps(trace))

    def update_best(score, params, it):
        if score > best['score']:
            best.update(score=float(score), params=dict(params), it=int(it))
            _save(RESULTS, names, best, base_mean, base_perg, subset)

    def log_progress(name, it, score, improvements, batch):
        log.info(f"[{name} batch {it}] best={score:.4f} (+{improvements}/{batch}) "
                 f"[{(time.time()-t0)/60:.0f} min, baseline {base_mean:.4f}]")

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
        best_x = np.array([(best['params'][n]) for n in names], float)
        best_perg = pool.apply(_pergauge, ((best_x - lo) / (hi - lo),))

    _save(RESULTS, names, best, base_mean, base_perg, subset, best_perg)
    log.info(f"\n[done] {(time.time()-t0)/60:.0f} min | baseline={base_mean:.4f} -> "
             f"best={best['score']:.4f} (batch {best['it']}) | algo best={result['best_score']:.4f}")
    log.info(f"[best per-gauge] { {k: round(v,3) for k,v in best_perg.items() if np.isfinite(v)} }")


def _save(results, names, best, base_mean, base_perg, subset, best_perg=None):
    out = dict(
        best_mean_kge=best['score'], baseline_mean_kge=base_mean, best_batch=best['it'],
        calib_gauges=list(subset),
        best_params={n: float(v) for n, v in best['params'].items()},
        baseline_per_gauge={k: (round(v, 4) if np.isfinite(v) else None) for k, v in base_perg.items()},
    )
    if best_perg is not None:
        out['best_per_gauge'] = {k: (round(v, 4) if np.isfinite(v) else None) for k, v in best_perg.items()}
    (results / "best.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
