# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared coupled SUMMA+dRoute evaluation for the e2e calibration.

One evaluation = apply the joint parameter vector -> run SUMMA over the spin-up+calibration window
in an isolated run directory -> route the fresh SUMMA runoff through dRoute (SV + lakes) with the
routing transfer-function coefficients -> multi-gauge KGE. Used by both the single-eval validation
and the parallel Async-DDS driver.
"""

import glob
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# Domain is parameterizable via env vars (default = the original Bow-at-Calgary domain) so the same
# driver can target the multigauge domain without disturbing the original / a running calibration.
D = Path(os.environ.get("COUPLED_DOMAIN_DIR",
                        "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"))
SUMMA_SETTINGS = D / "settings" / "SUMMA"
DROUTE_SETTINGS = D / "settings" / "dRoute"
SUMMA_EXE = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/installs/summa/bin/summa_sundials.exe"
RIVER_NET = Path(glob.glob(str(D / "shapefiles" / "river_network" / "*.shp"))[0])
OBS_DIR = D / "observations" / "streamflow"
PREFIX = "bow_calgary_v1"
SUMMA_SIM_END = "2012-12-31 23:00"   # spin-up 2010 + calib 2011-2012 (matches CFG CALIBRATION_PERIOD)
DT_H = 12.0
KGE_FLOOR = -2.0

import yaml  # noqa: E402

CFG = yaml.safe_load(open(os.environ.get("COUPLED_CONFIG", str(D.parent / "config_Bow_at_Calgary.yaml"))))


def _droute_cfg(runoff_file):
    return {
        'RIVER_NETWORK_SHAPEFILE': str(RIVER_NET), 'SETTINGS_DROUTE_PATH': str(DROUTE_SETTINGS),
        'DROUTE_REGIONALIZATION': 'transfer_function', 'DROUTE_RUNOFF_FILE': str(runoff_file),
        'MULTI_GAUGE_OBS_DIR': str(OBS_DIR), 'DROUTE_ROUTE_START': '2010-01-01',
        'CALIBRATION_PERIOD_START': '2011-01-01', 'CALIBRATION_PERIOD_END': '2012-12-31',
        'DROUTE_ROUTING_DT_HOURS': DT_H,
    }


def make_run_dir(root: Path) -> Path:
    """Create an isolated SUMMA run dir (settings copy + output dir + a private fileManager)."""
    root = Path(root)
    sdir = root / "SUMMA"
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not sdir.exists():
        shutil.copytree(SUMMA_SETTINGS, sdir)
    # private fileManager: point settings/output at this run dir, cap the sim window
    fm = (SUMMA_SETTINGS / "fileManager.txt").read_text().splitlines()
    new = []
    for ln in fm:
        s = ln.strip()
        if s.startswith("settingsPath"):
            new.append(f"settingsPath         '{sdir}/'")
        elif s.startswith("outputPath"):
            new.append(f"outputPath           '{out}/'")
        elif s.startswith("simEndTime"):
            new.append(f"simEndTime           '{SUMMA_SIM_END}'")
        else:
            new.append(ln)
    (sdir / "fileManager_run.txt").write_text("\n".join(new) + "\n")
    return root


def apply_summa(run_dir: Path, summa_params: dict, logger):
    """Write trialParams.nc (+ basin params) into the run dir's SUMMA settings."""
    from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
    sdir = Path(run_dir) / "SUMMA"
    pm = SUMMAParameterManager(CFG, logger, sdir)
    n_hru = len(pm.get_initial_parameters()[pm.all_param_names[0]])
    # broadcast each scalar trial value to the per-HRU/GRU array the manager expects
    arrs = {k: (np.full(n_hru, float(v)) if not hasattr(v, '__len__') else np.asarray(v, float))
            for k, v in summa_params.items()}
    return pm.update_model_files(arrs)


def run_summa(run_dir: Path, logger) -> Path:
    """Run SUMMA in the isolated run dir; return the path to the *_timestep.nc output."""
    sdir = Path(run_dir) / "SUMMA"
    out = Path(run_dir) / "out"
    fm = sdir / "fileManager_run.txt"
    log = out / "summa.log"
    with open(log, "w") as fh:
        rc = subprocess.run([SUMMA_EXE, "-m", str(fm)], stdout=fh, stderr=subprocess.STDOUT)
    ts = out / f"{PREFIX}_timestep.nc"
    if rc.returncode != 0 or not ts.exists():
        logger.error(f"SUMMA failed (rc={rc.returncode}); see {log}")
        return None
    return ts


def route_pergauge(runoff_file: Path, droute_coeffs: dict, logger) -> dict:
    """Route the fresh SUMMA runoff with the given dRoute TF coefficients; return {station: KGE}."""
    import contextlib
    import io

    import droute
    from droute.calibration.parameter_manager import DRouteParameterManager
    from droute.calibration.worker import DRouteWorker, build_droute_network
    cfg = _droute_cfg(runoff_file)
    w = DRouteWorker(cfg, logger)
    inp = w._load_inputs(DROUTE_SETTINGS)
    pm = DRouteParameterManager(cfg, logger, DROUTE_SETTINGS); pm._build_regionalization()
    arr, pnames = pm._region.to_distributed({k: float(v) for k, v in droute_coeffs.items()})
    per_reach = {p: arr[:, j].tolist() for j, p in enumerate(pnames)}
    net = build_droute_network(inp['seg_ids'], inp['downstream_idx'], inp['lengths'], inp['slopes'],
                               inp['lakes'], inp['id_to_idx'], per_reach)
    c = droute.SaintVenantEnzymeConfig()
    c.dt = DT_H * 3600.0; c.n_nodes = 4; c.enable_adjoint = False; c.use_enzyme_adjoint = False
    rt = droute.SaintVenantEnzyme(net, c)
    order = np.asarray(net.topological_order(), dtype=int)
    runoff = inp['daily']; i0 = int(inp['i_eval0']); ndays = runoff.shape[0]
    sub = int(round(86400.0 / (DT_H * 3600.0)))
    Qd = {g['ridx']: [] for g in inp['gauges']}
    with contextlib.redirect_stderr(io.StringIO()):
        for d in range(ndays):
            for _ in range(sub):
                for idx in order:
                    rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
                rt.route_timestep()
            if d >= i0:
                for g in inp['gauges']:
                    Qd[g['ridx']].append(rt.get_discharge(g['ridx']))
    out = {}
    for g in inp['gauges']:
        out[g['station']] = _kge(np.asarray(Qd[g['ridx']], float), np.asarray(g['obs'], float))
    return out


def _kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 +
                             (s.mean() / o.mean() - 1) ** 2))


if __name__ == "__main__":
    # single-eval validation with the INITIAL (default) parameters
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('val')
    from droute.calibration.parameter_manager import DRouteParameterManager
    from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
    spm = SUMMAParameterManager(CFG, log, SUMMA_SETTINGS)
    dpm = DRouteParameterManager(_droute_cfg(""), log, DROUTE_SETTINGS)
    summa0 = spm.get_initial_parameters()
    droute0 = dpm.get_initial_parameters()
    run = Path("/tmp/coupled_val")
    if run.exists():
        shutil.rmtree(run)
    make_run_dir(run)
    t = time.time()
    ok = apply_summa(run, summa0, log); print(f"[apply_summa] ok={ok}", flush=True)
    ts = run_summa(run, log)
    t_summa = time.time() - t
    print(f"[run_summa] {t_summa:.0f}s -> {ts}", flush=True)
    if ts is None:
        raise SystemExit("SUMMA run failed")
    t = time.time()
    perg = route_pergauge(ts, droute0, log)
    t_route = time.time() - t
    kept = [v for v in perg.values() if np.isfinite(v) and v >= KGE_FLOOR]
    print(f"[route] {t_route:.0f}s | mean KGE={np.mean(kept):.4f} over {len(kept)}/{len(perg)} gauges", flush=True)
    print("[per-gauge]", {k: round(v, 3) for k, v in perg.items()}, flush=True)
    print(f"[TOTAL coupled eval] {t_summa+t_route:.0f}s", flush=True)
