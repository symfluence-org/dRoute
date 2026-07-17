# SPDX-License-Identifier: GPL-3.0-or-later
"""Async-DDS calibration of the dRoute (SV + lakes) routing parameters on Bow-at-Calgary.

Calibrates the 9 transfer-function routing coefficients exposed by DRouteParameterManager
(per-reach Manning's n + inline/subgrid lake ratings) against the FIXED SUMMA runoff forcing,
maximizing the multi-gauge KGE used by the coupled objective. SUMMA output is held fixed (the
coupled e2e routes the cached *_timestep.nc runoff), so this calibrates the routing subspace.

Optimizer: SYMFLUENCE's own AsyncDDSAlgorithm (asynchronous parallel DDS -- maintains a pool of
best solutions, generates each batch by tournament selection + DDS perturbation in normalized
space). Parallelism is within each batch: evaluate_population routes ASYNC_DDS_BATCH_SIZE candidate
solutions concurrently in a multiprocessing Pool (N_CORES workers). Total model evaluations =
ASYNC_DDS_POOL_SIZE + NUMBER_OF_ITERATIONS * ASYNC_DDS_BATCH_SIZE (=> ~500 here). Each evaluation
routes the full 2010-2012 series at dt=12h with a 1-yr spin-up (same recipe as the coupled run).
"""

import contextlib
import io
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)

# ---- configuration ---------------------------------------------------------------------------
D = Path("/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary")
SETTINGS = D / "settings" / "dRoute"
CFG = {
    'RIVER_NETWORK_SHAPEFILE': str(D / "shapefiles" / "river_network" /
                                   "Bow_at_Calgary_riverNetwork_semidistributed.shp"),
    'SETTINGS_DROUTE_PATH': str(SETTINGS), 'DROUTE_REGIONALIZATION': 'transfer_function',
    'DROUTE_RUNOFF_FILE': str(D / "simulations" / "bow_calgary_v1" / "SUMMA" /
                              "bow_calgary_v1_timestep.nc"),
    'MULTI_GAUGE_OBS_DIR': str(D / "observations" / "streamflow"),
    'DROUTE_ROUTE_START': '2010-01-01', 'CALIBRATION_PERIOD_START': '2011-01-01',
    'CALIBRATION_PERIOD_END': '2012-12-31', 'DROUTE_ROUTING_DT_HOURS': 12.0,
}
DT_H = 12.0
N_CORES = 10
KGE_FLOOR = -2.0      # gauges with baseline KGE below this are excluded from the objective
SEED = 20260604
RESULTS = Path(__file__).parent / "results_calgary" / "coupled_dds"

# AsyncDDS config: 10 (pool) + 49*10 (batches) = 500 evaluations
ASYNC_CFG = {
    'NUMBER_OF_ITERATIONS': 49, 'ASYNC_DDS_POOL_SIZE': 10, 'ASYNC_DDS_BATCH_SIZE': N_CORES,
    'DDS_R': 0.2, 'MAX_STAGNATION_BATCHES': 10_000, 'OPTIMIZATION_METRIC': 'KGE',
}

_G = {}  # per-process state (filled by _init)


def _init(cfg, settings, subset_ridx):
    """Worker/process initializer: load fixed inputs + the regionalization once per process."""
    import droute  # noqa: F401  (ensure the extension loads in this process)
    from droute.calibration.parameter_manager import DRouteParameterManager
    from droute.calibration.worker import DRouteWorker
    log = logging.getLogger('w')
    w = DRouteWorker(cfg, log)
    inp = w._load_inputs(Path(settings))
    pm = DRouteParameterManager(cfg, log, Path(settings))
    pm._build_regionalization()
    _G.update(
        inp=inp, region=pm._region, names=list(pm.all_param_names),
        runoff=inp['daily'], ndays=inp['daily'].shape[0], i0=int(inp['i_eval0']),
        gauges=inp['gauges'], subset=set(subset_ridx),
        sub=int(round(86400.0 / (DT_H * 3600.0))),
    )


def _kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 +
                             (s.mean() / o.mean() - 1) ** 2))


def _route_pergauge(coeff_vec):
    """Route with the given TF coefficients; return {ridx: KGE} over all gauges (or {} on failure)."""
    import droute
    from droute.calibration.worker import build_droute_network
    g = _G
    coeffs = {n: float(v) for n, v in zip(g['names'], coeff_vec)}
    try:
        arr, pnames = g['region'].to_distributed(coeffs)
        per_reach = {p: arr[:, j].tolist() for j, p in enumerate(pnames)}
        net = build_droute_network(g['inp']['seg_ids'], g['inp']['downstream_idx'],
                                   g['inp']['lengths'], g['inp']['slopes'], g['inp']['lakes'],
                                   g['inp']['id_to_idx'], per_reach)
        c = droute.SaintVenantEnzymeConfig()
        c.dt = DT_H * 3600.0; c.n_nodes = 4
        c.enable_adjoint = False; c.use_enzyme_adjoint = False
        rt = droute.SaintVenantEnzyme(net, c)
        order = np.asarray(net.topological_order(), dtype=int)
        runoff, ndays, i0, sub = g['runoff'], g['ndays'], g['i0'], g['sub']
        Qd = {gg['ridx']: [] for gg in g['gauges']}
        with contextlib.redirect_stderr(io.StringIO()):
            for d in range(ndays):
                for _ in range(sub):
                    for idx in order:
                        rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
                    rt.route_timestep()
                if d >= i0:
                    for gg in g['gauges']:
                        Qd[gg['ridx']].append(rt.get_discharge(gg['ridx']))
        return {gg['ridx']: _kge(np.asarray(Qd[gg['ridx']], float), np.asarray(gg['obs'], float))
                for gg in g['gauges']}
    except (RuntimeError, ValueError, FloatingPointError, KeyError):
        return {}


def evaluate(coeff_vec):
    """Objective: mean KGE over the fixed calibration gauge subset (-1e9 on failure)."""
    perg = _route_pergauge(coeff_vec)
    vals = [perg[r] for r in _G['subset'] if r in perg and np.isfinite(perg[r])]
    return float(np.mean(vals)) if vals else -1e9


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    log = logging.getLogger('asyncdds')

    from droute.calibration.parameter_manager import DRouteParameterManager
    from symfluence.optimization.optimizers.algorithms.async_dds import AsyncDDSAlgorithm
    pm = DRouteParameterManager(CFG, log, SETTINGS)
    names = list(pm.all_param_names)
    lo = np.array([pm.param_bounds[n]['min'] for n in names], float)
    hi = np.array([pm.param_bounds[n]['max'] for n in names], float)
    x0 = np.array([pm.get_initial_parameters()[n] for n in names], float)
    denorm = lambda xn: lo + np.asarray(xn) * (hi - lo)  # noqa: E731

    # baseline (initial) per-gauge KGE -> fixed calibration gauge subset
    _init(CFG, SETTINGS, set())
    t0 = time.time()
    base_perg = _route_pergauge(x0)
    stations = {gg['ridx']: gg['station'] for gg in _G['gauges']}
    subset = sorted([r for r, k in base_perg.items() if np.isfinite(k) and k >= KGE_FLOOR])
    _G['subset'] = set(subset)
    base_mean = float(np.mean([base_perg[r] for r in subset]))
    log.info(f"[baseline] {time.time()-t0:.0f}s | mean KGE={base_mean:.4f} over {len(subset)}/"
             f"{len(base_perg)} gauges | per-gauge="
             f"{ {stations[r]: round(base_perg[r],3) for r in base_perg} }")

    trace, best = [], {'score': base_mean, 'params': dict(zip(names, x0)), 'it': 0}

    def evaluate_population(pop_norm, iteration):
        coeffs = [denorm(row) for row in np.asarray(pop_norm)]
        scores = pool.map(evaluate, coeffs)
        return [s if s > -1e8 else PENALTY for s in scores]

    def record_iteration(it, score, params):
        trace.append({'batch': int(it), 'best_kge': round(float(score), 5)})
        (RESULTS / "trace.json").write_text(json.dumps(trace))

    def update_best(score, params, it):
        if score > best['score']:
            best.update(score=float(score), params=dict(params), it=int(it))
            _save_best(RESULTS, names, best, base_mean, base_perg, stations, subset)

    def log_progress(name, it, score, improvements, batch):
        log.info(f"[{name} batch {it}] best={score:.4f} (+{improvements}/{batch}) "
                 f"[{time.time()-t0:.0f}s, baseline {base_mean:.4f}]")

    algo = AsyncDDSAlgorithm(ASYNC_CFG, log)
    global PENALTY
    PENALTY = algo.penalty_score

    with mp.Pool(N_CORES, initializer=_init, initargs=(CFG, SETTINGS, subset)) as pool:
        result = algo.optimize(
            n_params=len(names), evaluate_solution=lambda x, it: evaluate(denorm(x)),
            evaluate_population=evaluate_population, denormalize_params=lambda xn: dict(zip(names, denorm(xn))),
            record_iteration=record_iteration, update_best=update_best, log_progress=log_progress,
            num_processes=N_CORES, initial_guess=(x0 - lo) / (hi - lo),
        )
        # final per-gauge KGE of the best solution
        best_x = np.array([best['params'][n] for n in names], float)
        best_perg = pool.apply(_route_pergauge, (best_x,))

    _save_best(RESULTS, names, best, base_mean, base_perg, stations, subset, best_perg)
    log.info(f"\n[done] {time.time()-t0:.0f}s | baseline={base_mean:.4f} -> "
             f"best={best['score']:.4f} (batch {best['it']}) | algo best={result['best_score']:.4f}")
    log.info(f"[best per-gauge] { {stations[r]: round(best_perg[r],3) for r in best_perg if np.isfinite(best_perg.get(r, np.nan))} }")


def _save_best(results, names, best, base_mean, base_perg, stations, subset, best_perg=None):
    out = dict(
        best_mean_kge=best['score'], baseline_mean_kge=base_mean, best_batch=best['it'],
        calib_gauges=[stations[r] for r in subset],
        best_coeffs={n: float(v) for n, v in best['params'].items()},
        baseline_per_gauge={stations[r]: (round(base_perg[r], 4) if np.isfinite(base_perg[r])
                                          else None) for r in base_perg},
    )
    if best_perg is not None:
        out['best_per_gauge'] = {stations[r]: (round(best_perg[r], 4) if np.isfinite(best_perg.get(r, np.nan))
                                               else None) for r in base_perg}
    (results / "best.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
