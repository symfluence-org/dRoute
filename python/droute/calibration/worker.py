# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SYMFLUENCE Team <dev@symfluence.org>

"""
dRoute Calibration Worker.

Worker implementation for dRoute routing parameter optimization with support for
both evolutionary and gradient-based calibration via automatic differentiation.
"""

import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.constants import ModelDefaults
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.evaluation.metrics import kge, nse
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask

# Try to import dRoute
try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False
    droute = None

# Seconds per day -- used to derive the Saint-Venant sub-step count from the daily routing window.
DT_DAY = 86400.0


def _kge_multigauge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Per-gauge KGE used by the Saint-Venant multi-gauge objective.

    Matches the validated standalone routing experiment: drops non-finite pairs and requires at
    least 10 overlapping points. Kept separate from :func:`symfluence.evaluation.metrics.kge`
    (used by the single-gauge Muskingum-Cunge path) so the SV multi-gauge results stay bit-identical
    to the standalone driver.
    """
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))


def _nse_multigauge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Per-gauge NSE companion to :func:`_kge_multigauge` (same finite-mask + min-overlap contract),
    so the multi-gauge objective can honour ``OPTIMIZATION_METRIC: NSE`` as well as KGE."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10:
        return np.nan
    return float(nse(o, s, transfo=1))


def build_droute_network(seg_ids, downstream_idx, lengths, slopes, lakes, id_to_idx,
                         per_reach: Optional[Dict[str, List[float]]] = None):
    """Build a dRoute Network with Manning's n + inline/subgrid lake config, optionally overriding
    per-reach routing params from a parameter-manager expansion (``per_reach[param][reach_idx]``).

    Used by the Saint-Venant routing path (transfer-function regionalized parameters expand to one
    value per reach); the Muskingum-Cunge path builds its network inline in
    :meth:`DRouteWorker._route_cpp_with_lakes`.
    """
    from droute.lake_preprocessor import apply_lake_config_to_network
    n = len(seg_ids)
    outlet_junc = n
    junc_up: Dict[int, List[int]] = {i: [] for i in range(n + 1)}
    for i in range(n):
        d = downstream_idx[i]
        junc_up[d if d >= 0 else outlet_junc].append(i)
    net = droute.Network()
    for jid in range(n + 1):
        j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]; net.add_junction(j)
    for i in range(n):
        r = droute.Reach(); r.id = i; r.length = float(lengths[i]); r.slope = max(float(slopes[i]), 0.001)
        r.manning_n = 0.035
        r.upstream_junction_id = i
        d = downstream_idx[i]; r.downstream_junction_id = d if d >= 0 else outlet_junc
        net.add_reach(r)
    net.build_topology()
    apply_lake_config_to_network(net, lakes, id_to_idx)
    # per-reach parameter overrides from the regionalized calibration
    if per_reach:
        setter = {'manning_n': 'manning_n', 'lake_q_ref': 'lake_q_ref', 'lake_exp': 'lake_exp',
                  'lake_q_min': 'lake_q_min', 'lake_spill_coef': 'lake_spill_coef',
                  'subgrid_q_ref': 'subgrid_q_ref', 'subgrid_exp': 'subgrid_exp'}
        for pname, vals in per_reach.items():
            attr = setter.get(pname)
            if attr is None:
                continue
            for i in range(n):
                setattr(net.get_reach(i), attr, float(vals[i]))
    return net


class DRouteWorker(BaseWorker):
    """
    Worker for dRoute routing parameter calibration.

    Two routing paths, selected by ``DROUTE_ROUTING_METHOD`` (default
    ``'muskingum_cunge'``):

    Muskingum-Cunge (``'muskingum_cunge'``):
    - Standard evolutionary optimization (evaluate -> route -> metrics)
    - Gradient-based optimization with AD when dRoute is compiled with CoDiPack/Enzyme
    - Efficient in-memory routing (no file I/O during calibration)
    - Optional inline reservoir operating rules (lakes config)
    - Single-gauge (outlet) KGE/NSE objective
    - Lumped calibration parameters: velocity, diffusivity, muskingum_k,
      muskingum_x, manning_n

    Saint-Venant + lakes (``'saint_venant'``):
    - Differentiable Saint-Venant solver with inline + subgrid lake routing
    - Transfer-function-regionalized per-reach parameters (Manning's n + lake
      rating coefficients), expanded from a low-dimensional coefficient set by
      :class:`DRouteParameterManager`
    - Multi-gauge KGE objective with a per-gauge floor
    - Coupling-aware: routes the upstream model's fresh runoff each evaluation
      (``coupling_source_dir``) and supports the dCoupler graph handoff via
      :meth:`materialize_metric_inputs`
    """

    # Default parameter bounds for dRoute calibration
    PARAM_BOUNDS = {
        'velocity': (0.1, 5.0),         # m/s
        'diffusivity': (100.0, 5000.0), # m^2/s
        'muskingum_k': (0.1, 24.0),     # hours
        'muskingum_x': (0.0, 0.5),      # dimensionless
        'manning_n': (0.01, 0.1),       # dimensionless
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dRoute worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Lazy-loaded components (Muskingum-Cunge path)
        self._network_config = None
        self._runoff_data = None
        self._observations = None
        self._time_index = None

        # Lazy-loaded components (Saint-Venant + lakes path)
        self._inp: Optional[Dict[str, Any]] = None  # cached SV inputs (runoff, network arrays, gauges)

        # Routing configuration
        self.routing_method = 'muskingum_cunge'
        self.routing_dt = 3600  # seconds

        if config:
            self.routing_method = config.get('DROUTE_ROUTING_METHOD', 'muskingum_cunge')
            self.routing_dt = config.get('DROUTE_ROUTING_DT', 3600)

    def _is_sv(self) -> bool:
        """True when configured for the Saint-Venant + lakes routing path (multi-gauge,
        transfer-function regionalized params, coupling-aware). Default is Muskingum-Cunge."""
        return str(self.routing_method).lower().replace('-', '_') in (
            'saint_venant', 'saintvenant', 'sv'
        )

    def _gv(self, key, default=None):
        """Config accessor used by the coupling-aware path (dict-key lookup with default)."""
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def _use_multigauge_path(self) -> bool:
        """True when the coupling-aware, multi-gauge routing path should be used (Saint-Venant
        OR a configured multi-gauge objective). That path is engine-agnostic: it routes via SV
        when ``_is_sv()`` else Muskingum-Cunge (+ lakes/reservoir rules), so coupled SUMMA+dRoute
        calibration works with either engine -- MC being the stable choice on domains where the
        transient SV solver is ill-conditioned (e.g. Bow-at-Calgary)."""
        return self._is_sv() or bool(self._gv('MULTI_GAUGE_CALIBRATION', False))

    def supports_native_gradients(self) -> bool:
        """
        Check if native gradient computation is available.

        dRoute supports native gradients via CoDiPack/Enzyme AD when compiled
        with DROUTE_ENABLE_AD=ON.

        Returns:
            True if dRoute has AD support and is available.
        """
        if not HAS_DROUTE:
            return False

        # Check if dRoute was built with AD
        return hasattr(droute, 'gradient') or hasattr(droute, 'compute_gradients')

    def _load_data(self) -> bool:
        """
        Load runoff and observation data for calibration.

        Returns:
            True if data loaded successfully.
        """
        if self._runoff_data is not None:
            return True

        try:
            if self.config is None:
                self.logger.error("Config not set for dRoute worker")
                return False

            data_dir = Path(self._cfg('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = self._cfg('DOMAIN_NAME', 'unknown')
            experiment_id = self._cfg('EXPERIMENT_ID', 'default')
            project_dir = data_dir / f"domain_{domain_name}"

            # Determine source model
            hydro_model = self._cfg('HYDROLOGICAL_MODEL', 'SUMMA')
            if ',' in str(hydro_model):
                from_model = str(hydro_model).split(',')[0].strip().upper()
            else:
                from_model = str(hydro_model).strip().upper()

            # Load runoff from source model
            runoff_dir = project_dir / f"simulations/{experiment_id}" / from_model
            runoff_file = self._find_runoff_file(runoff_dir)

            if runoff_file is None:
                self.logger.error(f"No runoff file found in {runoff_dir}")
                return False

            import xarray as xr
            ds = xr.open_dataset(runoff_file)

            # Find runoff variable
            runoff_var = None
            for var in ['averageRoutedRunoff', 'scalarTotalRunoff', 'runoff']:
                if var in ds:
                    runoff_var = var
                    break

            if runoff_var is None:
                for var in ds.data_vars:
                    if 'runoff' in var.lower():
                        runoff_var = var
                        break

            if runoff_var is None:
                self.logger.error("No runoff variable found")
                ds.close()
                return False

            self._runoff_data = ds[runoff_var].values
            if self._runoff_data.ndim == 1:
                self._runoff_data = self._runoff_data.reshape(-1, 1)

            self._time_index = pd.DatetimeIndex(ds.time.values)
            ds.close()

            # Load network configuration
            settings_dir = project_dir / 'settings' / 'dRoute'
            network_config_path = settings_dir / 'droute_network.yaml'
            # lake/reservoir operating-rule config (optional)
            lake_cfg = settings_dir / 'droute_lakes.yaml'
            self._lake_config_path = lake_cfg if lake_cfg.exists() else None

            if network_config_path.exists():
                import yaml
                with open(network_config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self._network_config = {
                    'n_segments': config['network']['n_segments'],
                    'downstream_idx': config['network']['downstream_idx'],
                    'outlet_indices': config['network']['outlet_indices'],
                    'routing_order': config['network']['routing_order'],
                    'slopes': config['geometry']['slopes'],
                    'lengths': config['geometry']['lengths'],
                    'widths': config['geometry']['widths'],
                    'hru_to_seg_idx': config['hru_mapping']['hru_to_seg_idx'],
                    # segment ids (LINKNO) needed to map lake/reservoir config -> reach index
                    'segment_ids': config['network'].get('segment_ids'),
                }
            else:
                self.logger.warning("Network config not found, trying mizuRoute topology")
                # Try mizuRoute topology
                from droute.network_adapter import DRouteNetworkAdapter
                adapter = DRouteNetworkAdapter(self.logger)
                mizu_topo = project_dir / 'settings' / 'mizuRoute' / 'mizuRoute_topology.nc'
                if mizu_topo.exists():
                    topology = adapter.load_topology(mizu_topo, format='netcdf')
                    self._network_config = adapter.to_droute_format(
                        topology,
                        routing_method=self.routing_method,
                        routing_dt=self.routing_dt
                    )
                else:
                    self.logger.error("No network configuration found")
                    return False

            # Load observations
            obs_file = resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' / f"{domain_name}_streamflow_processed.csv"
            if obs_file.exists():
                obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
                obs_cms = obs_df.iloc[:, 0]

                # Resample to daily if needed
                obs_freq = pd.infer_freq(obs_cms.index[:100])
                if obs_freq and ('H' in str(obs_freq) or 'h' in str(obs_freq)):
                    obs_daily = obs_cms.resample('D').mean()
                else:
                    obs_daily = obs_cms

                # Align with runoff time
                assert self._time_index is not None
                forcing_dates = pd.to_datetime(self._time_index).normalize()
                self._observations = obs_daily.reindex(forcing_dates).values
            else:
                self.logger.warning("Observations not found")
                self._observations = None

            self.logger.info(f"Loaded dRoute data: {len(self._runoff_data)} timesteps")
            return True

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error loading dRoute data: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    def _find_runoff_file(self, runoff_dir: Path) -> Optional[Path]:
        """Find runoff file in directory."""
        if not runoff_dir.exists():
            return None

        patterns = ['*_timestep.nc', '*_output.nc', '*_runs_def.nc', '*.nc']
        for pattern in patterns:
            files = list(runoff_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply routing parameters.

        For dRoute, parameters are passed directly to routing function,
        no file writing needed.

        Args:
            params: Parameter values
            settings_dir: Settings directory (unused for dRoute)
            **kwargs: Additional arguments

        Returns:
            True (always succeeds for dRoute)
        """
        # Always retain the params in-memory (the MC + reservoir-rule routing reads them directly).
        self._current_params = params
        # Coupling-aware multi-gauge path: persist via the parameter manager, which writes the
        # per-reach JSON in regionalized mode (SV) or droute_config.yaml in lumped mode (MC).
        if self._use_multigauge_path():
            from .parameter_manager import DRouteParameterManager
            pm = DRouteParameterManager(self.config, self.logger, Path(settings_dir))
            return pm.update_model_files(params)
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run dRoute routing.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if routing succeeded.
        """
        if self._use_multigauge_path() or kwargs.get('coupling_source_dir') is not None:
            return self._run_model_multigauge(config, settings_dir, output_dir, **kwargs)

        try:
            if not self._load_data():
                return False

            params = getattr(self, '_current_params', {})

            # Run routing
            routed = self._route_with_params(params)
            if routed is None:
                return False

            self._last_routed = routed

            # Save output if requested
            if kwargs.get('save_output', False) and output_dir:
                self._save_output(routed, output_dir, config)

            return True

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error running dRoute: {e}")
            self.logger.debug(traceback.format_exc())
            return False

    # Global reservoir operating-rule parameters this worker understands.
    RESERVOIR_PARAM_KEYS = (
        'reservoir_q_ref_mult', 'reservoir_exp', 'reservoir_q_min_frac', 'reservoir_spill_coef'
    )

    def _apply_reservoir_operating_rules(self, net, segid_to_index, params, lake_cls) -> int:
        """Apply GLOBAL reservoir operating-rule params to every inline reservoir.

        Each reservoir's HydroLAKES-initialised q_ref is scaled by
        reservoir_q_ref_mult; exp/spill are set globally; q_min is set as a
        fraction of the (scaled) q_ref. Returns the number of reservoirs modified.
        """
        q_ref_mult = float(params.get('reservoir_q_ref_mult', 1.0))
        exp = params.get('reservoir_exp')
        q_min_frac = params.get('reservoir_q_min_frac')
        spill = params.get('reservoir_spill_coef')
        n = 0
        for seg, rec in (lake_cls.get('inline') or {}).items():
            if int(rec.get('lake_type', 0)) != 1:        # reservoirs only (type 1)
                continue
            i = segid_to_index.get(int(seg))
            if i is None:
                continue
            r = net.get_reach(i)
            r.lake_q_ref = float(r.lake_q_ref) * q_ref_mult
            if exp is not None:
                r.lake_exp = float(exp)
            if spill is not None:
                r.lake_spill_coef = float(spill)
            if q_min_frac is not None:
                r.lake_q_min = float(q_min_frac) * float(r.lake_q_ref)
            n += 1
        return n

    def _route_cpp_with_lakes(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """Route through the C++ Muskingum-Cunge router with lake/reservoir routing.

        Required for reservoir operating-rule params to affect the objective (the
        pure-Python fallback ignores lakes). Returns per-segment discharge
        (n_time, n_seg) indexed by reach/segment, or None if prerequisites missing.
        """
        if not HAS_DROUTE:
            return None
        nc = self._network_config
        seg_ids = nc.get('segment_ids')
        if seg_ids is None:
            self.logger.warning("Network config has no segment_ids; cannot apply lakes (C++ path skipped)")
            return None
        import yaml
        from droute.lake_preprocessor import apply_lake_config_to_network

        n_time, n_hru = self._runoff_data.shape
        n_seg = nc['n_segments']
        hru_to_seg = nc['hru_to_seg_idx']
        downstream = np.asarray(nc['downstream_idx'], dtype=int)
        lengths = np.asarray(nc['lengths'], dtype=float)
        slopes = np.asarray(nc['slopes'], dtype=float)
        id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
        manning = float(params.get('manning_n', 0.035))

        seg_runoff = np.zeros((n_time, n_seg))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                seg_runoff[:, seg_idx] += self._runoff_data[:, hru_idx]

        # build network
        outlet_junc = n_seg
        junc_up = {i: [] for i in range(n_seg + 1)}
        for i in range(n_seg):
            d = int(downstream[i]); junc_up[d if d >= 0 else outlet_junc].append(i)
        net = droute.Network()
        for jid in range(n_seg + 1):
            j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]; net.add_junction(j)
        for i in range(n_seg):
            r = droute.Reach(); r.id = i; r.length = float(lengths[i])
            r.slope = max(float(slopes[i]), 0.001); r.manning_n = manning
            r.upstream_junction_id = i
            d = int(downstream[i]); r.downstream_junction_id = d if d >= 0 else outlet_junc
            net.add_reach(r)
        net.build_topology()

        # apply lakes + global reservoir operating-rule params
        if self._lake_config_path is not None:
            with open(self._lake_config_path, encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {}
            lake_cls = {'inline': raw.get('inline_lakes', {}) or {},
                        'subgrid': raw.get('subgrid_lakes', {}) or {}}
            apply_lake_config_to_network(net, lake_cls, id_to_idx, logger=self.logger)
            if any(k in params for k in self.RESERVOIR_PARAM_KEYS):
                nres = self._apply_reservoir_operating_rules(net, id_to_idx, params, lake_cls)
                self.logger.debug(f"Applied reservoir operating rules to {nres} reservoirs")

        cfg = droute.RouterConfig(); cfg.dt = float(self.routing_dt); cfg.enable_gradients = False
        router = droute.MuskingumCungeRouter(net, cfg)
        order = np.asarray(net.topological_order(), dtype=int)
        routed = np.zeros((n_time, n_seg))
        for t in range(n_time):
            for idx in order:
                router.set_lateral_inflow(int(idx), float(seg_runoff[t, idx]))
            router.route_timestep()
            routed[t, order] = router.get_all_discharges()   # get_all_discharges() is topo-ordered
        return routed

    def _route_with_params(self, params: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Execute routing with given parameters.

        Args:
            params: Routing parameters

        Returns:
            Routed streamflow array, or None on error
        """
        if self._runoff_data is None or self._network_config is None:
            return None

        # Prefer the C++ router when a lake/reservoir config is present (required for
        # reservoir operating-rule params to take effect); fall back to pure-Python MC.
        if getattr(self, '_lake_config_path', None) is not None or \
                any(k in params for k in self.RESERVOIR_PARAM_KEYS):
            routed = self._route_cpp_with_lakes(params)
            if routed is not None:
                return routed

        n_time, n_hru = self._runoff_data.shape
        n_segments = self._network_config['n_segments']
        hru_to_seg = self._network_config['hru_to_seg_idx']
        downstream = self._network_config['downstream_idx']
        routing_order = self._network_config['routing_order']
        lengths = np.array(self._network_config['lengths'])
        slopes = np.array(self._network_config['slopes'])

        # Aggregate runoff to segments
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += self._runoff_data[:, hru_idx]

        # Get velocity parameter (or use slope-based estimate)
        velocity = params.get('velocity', 1.0)
        if isinstance(velocity, (int, float)):
            velocities = np.full(n_segments, velocity)
        else:
            velocities = velocity * np.sqrt(slopes)

        velocities = np.clip(velocities, 0.1, 5.0)

        # Muskingum parameters
        musk_x = params.get('muskingum_x', 0.2)

        # Travel time and K
        K = lengths / velocities / 3600.0  # hours

        # Route
        routed = np.zeros((n_time, n_segments))
        dt_hours = self.routing_dt / 3600.0

        for t in range(n_time):
            for seg_idx in routing_order:
                Q_local = segment_runoff[t, seg_idx]

                # Upstream inflow
                Q_upstream = 0.0
                for up_idx in range(n_segments):
                    if downstream[up_idx] == seg_idx:
                        Q_upstream += routed[t, up_idx] if t > 0 else 0.0

                Q_prev = routed[t-1, seg_idx] if t > 0 else 0.0
                I = Q_local + Q_upstream
                I_prev = segment_runoff[t-1, seg_idx] + (
                    sum(routed[t-1, up_idx] for up_idx in range(n_segments)
                        if downstream[up_idx] == seg_idx) if t > 0 else 0.0
                )

                k = max(K[seg_idx], 0.1)
                denom = 2*k*(1-musk_x) + dt_hours
                C1 = (dt_hours - 2*k*musk_x) / denom
                C2 = (dt_hours + 2*k*musk_x) / denom
                C3 = (2*k*(1-musk_x) - dt_hours) / denom

                routed[t, seg_idx] = max(0, C1*I + C2*I_prev + C3*Q_prev)

        return routed

    def _save_output(self, routed: np.ndarray, output_dir: Path, config: Dict[str, Any]):
        """Save routed output to file."""
        import xarray as xr

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert self._network_config is not None
        outlet_indices = self._network_config['outlet_indices']
        outlet_idx = outlet_indices[0] if outlet_indices else 0

        ds = xr.Dataset(
            {
                'routedRunoff': (['time', 'seg'], routed.astype(np.float32)),
                'outletStreamflow': (['time'], routed[:, outlet_idx].astype(np.float32)),
            },
            coords={
                'time': self._time_index,
                'seg': np.arange(routed.shape[1]),
            }
        )

        domain_name = config.get('DOMAIN_NAME', 'unknown')
        output_file = output_dir / f"{domain_name}_droute_calibration.nc"
        ds.to_netcdf(output_file)
        ds.close()

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from routing output.

        Args:
            output_dir: Output directory (unused, uses in-memory results)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values.
        """
        # A coupled run uses _run_model_multigauge whenever coupling_source_dir is set (not only when
        # _use_multigauge_path() is True), writing droute_streamflow.npz but never setting _last_routed.
        # Key the metric path on that artifact so the run and metric paths agree -- otherwise the
        # single-gauge branch below sees _last_routed is None and returns the penalty score.
        npz_path = Path(output_dir) / 'droute_streamflow.npz'
        if self._use_multigauge_path() or npz_path.exists():
            return self._metrics_multigauge(output_dir, config, **kwargs)

        try:
            routed = getattr(self, '_last_routed', None)
            if routed is None:
                return {'kge': self.penalty_score, 'error': 'No routing results'}

            if self._observations is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            # Get outlet streamflow
            outlet_indices = self._network_config['outlet_indices']
            outlet_idx = outlet_indices[0] if outlet_indices else 0
            sim = routed[:, outlet_idx]

            obs = self._observations

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Remove NaN
            valid_mask = ~(np.isnan(sim) | np.isnan(obs))
            sim = sim[valid_mask]
            obs = obs[valid_mask]

            if len(sim) < 10:
                return {'kge': self.penalty_score, 'error': 'Insufficient data'}

            kge_val = kge(obs, sim, transfo=1)
            nse_val = nse(obs, sim, transfo=1)

            if np.isnan(kge_val):
                kge_val = self.penalty_score
            if np.isnan(nse_val):
                nse_val = self.penalty_score

            return {
                'kge': float(kge_val),
                'nse': float(nse_val),
                'n_points': len(sim)
            }

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error calculating dRoute metrics: {e}")
            return {'kge': self.penalty_score, 'error': str(e)}

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss with respect to routing parameters.

        Uses dRoute's AD capabilities when available.

        Args:
            params: Current parameter values
            metric: Metric to compute gradient for ('kge' or 'nse')

        Returns:
            Dictionary of parameter gradients, or None if AD unavailable.
        """
        if not HAS_DROUTE:
            self.logger.warning("dRoute not available for gradient computation")
            return None

        if not self.supports_native_gradients():
            self.logger.warning("dRoute not compiled with AD support")
            return None

        if not self._load_data():
            return None

        try:
            # Use dRoute's gradient computation
            gradients = droute.compute_gradients(
                runoff=self._runoff_data,
                network=self._network_config,
                params=params,
                observations=self._observations,
                metric=metric
            )
            return gradients

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error computing dRoute gradient: {e}")
            return None

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> tuple:
        """
        Evaluate loss and compute gradient in single pass.

        Args:
            params: Parameter values
            metric: Metric to evaluate

        Returns:
            Tuple of (loss_value, gradient_dict)
        """
        if not self.supports_native_gradients():
            # Fallback to separate evaluation
            self._current_params = params
            if self.run_model(self.config or {}, Path('.'), Path('.'), save_output=False):
                metrics = self.calculate_metrics(Path('.'), self.config or {})
                loss = -metrics.get('kge', self.penalty_score)
                return loss, None
            return self.penalty_score, None

        # Use AD for efficient value+gradient
        if not self._load_data():
            return self.penalty_score, None

        try:
            loss, gradients = droute.value_and_gradient(
                runoff=self._runoff_data,
                network=self._network_config,
                params=params,
                observations=self._observations,
                metric=metric
            )
            return float(loss), gradients

        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error in evaluate_with_gradient: {e}")
            return self.penalty_score, None

    # ===== Saint-Venant + lakes routing path =================================================
    # Routes SUMMA runoff through the differentiable Saint-Venant solver (inline + subgrid lakes),
    # applying transfer-function-regionalized per-reach params, scored against a multi-gauge KGE
    # objective. Coupling-aware: in a coupled run the COUPLED worker passes this iteration's fresh
    # SUMMA output as ``coupling_source_dir`` so routing reads fresh runoff each evaluation.

    def _load_inputs(self, settings_dir: Path, runoff_file: Optional[str] = None,
                     force: bool = False) -> Dict[str, Any]:
        # ``runoff_file`` overrides runoff discovery (coupled runs pass this iteration's fresh SUMMA
        # output); ``force`` re-reads even when cached, since the runoff changes every coupled eval.
        if self._inp is not None and not force:
            return self._inp
        import glob

        import geopandas as gpd
        import pandas as pd
        import xarray as xr

        # Resolve the domain root from config (SYMFLUENCE_DATA_DIR/domain_<DOMAIN_NAME>) rather than
        # settings_dir.parent.parent: in coupled/parallel runs settings_dir is a per-process dir
        # (.../process_N/settings/dRoute) whose .parent.parent is the process dir, which carries no
        # shapefiles/attributes. Fall back to the legacy relative path only when config lacks them.
        data_dir, domain_name = self._gv('SYMFLUENCE_DATA_DIR'), self._gv('DOMAIN_NAME')
        domain = (Path(data_dir) / f"domain_{domain_name}"
                  if data_dir and domain_name else Path(settings_dir).parent.parent)
        rn_path = self._gv('RIVER_NETWORK_SHAPEFILE')
        if not rn_path:
            rn_cands = glob.glob(str(domain / 'shapefiles' / 'river_network' / '*.shp'))
            if not rn_cands:
                raise FileNotFoundError(
                    f"no river network shapefile under {domain / 'shapefiles' / 'river_network'}")
            rn_path = rn_cands[0]
        rn = gpd.read_file(rn_path)
        seg_ids = rn['LINKNO'].astype(int).values
        id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
        downstream_idx = np.array([id_to_idx.get(int(d), -1) for d in rn['DSLINKNO'].astype(int).values])
        lengths = rn['Length'].astype(float).values
        slopes = rn['Slope'].astype(float).values

        runoff_path = runoff_file or self._gv('DROUTE_RUNOFF_FILE')
        if not runoff_path:
            ro_cands = glob.glob(str(domain / 'simulations' / '*' / 'SUMMA' / '*_timestep.nc'))
            if not ro_cands:
                raise FileNotFoundError(
                    f"no SUMMA runoff (*_timestep.nc) under {domain / 'simulations'}")
            runoff_path = ro_cands[0]
        ds = xr.open_dataset(runoff_path)
        runoff = np.clip(ds['averageRoutedRunoff'].values, 0, None)
        gru = ds['gruId'].values.astype(int)
        time = pd.to_datetime(ds['time'].values)
        attr = xr.open_dataset(domain / 'settings' / 'SUMMA' / 'attributes.nc')
        area_by_id = {int(h): float(a) for h, a in zip(attr['hruId'].values.astype(int),
                                                       attr['HRUarea'].values.astype(float))}
        n_seg = len(seg_ids)
        seg_runoff = np.zeros((len(time), n_seg))
        for jx, gid in enumerate(gru):
            i = id_to_idx.get(int(gid))
            if i is not None:
                seg_runoff[:, i] = runoff[:, jx] * area_by_id.get(int(gid), 0.0)
        cap = float(self._gv('DROUTE_RUNOFF_CAP', 50.0))
        daily = pd.DataFrame(np.clip(seg_runoff, 0, cap), index=time).resample('D').mean()
        # Calibration window: honour explicit *_START/_END, else parse the combined CALIBRATION_PERIOD.
        # The normalized config delivers it as a LIST ['start', 'end'] (older flat configs as a
        # "start, end" string) -- handle both, or extended-period runs silently fall back to the
        # legacy 2011-2012 default and drop the early calibration years (e.g. the 2005 flood).
        cal = self._gv('CALIBRATION_PERIOD')
        if isinstance(cal, (list, tuple)):
            cal_s, cal_e = (str(cal[0]).strip(), str(cal[1]).strip()) if len(cal) >= 2 else (None, None)
        else:
            cal = str(cal or '')
            cal_s, cal_e = ([p.strip() for p in cal.split(',')[:2]] + [None, None])[:2] if ',' in cal else (None, None)
        eval_start = self._gv('CALIBRATION_PERIOD_START') or cal_s or '2011-01-01'
        eval_end = self._gv('CALIBRATION_PERIOD_END') or cal_e or '2012-12-31'
        # Route from the experiment/spinup start (date only) so routing warms up before eval_start;
        # the eval window is selected downstream via i_eval0, so earlier routing is harmless.
        _rs = self._gv('DROUTE_ROUTE_START') or self._gv('EXPERIMENT_TIME_START') or eval_start
        route_start = str(_rs).split(',')[0].strip().split()[0]
        # A SUMMA eval that produced no output yields an empty runoff series; fail it cleanly (the
        # caller scores the penalty) rather than crashing on argmax/min of an empty sequence.
        if daily.empty:
            raise ValueError("dRoute received empty SUMMA runoff (0 timesteps): land model produced no output")
        # Guard against a stale EXPERIMENT_TIME_START default (the config factory defaults it to
        # '2010-01-01'): never start later than eval_start or the available runoff, or the early
        # calibration years (e.g. the 2005 flood) get silently clipped out of the window.
        data_start = str(daily.index.min().date())
        route_start = min(route_start, str(eval_start).split()[0], data_start)
        daily = daily.loc[route_start:eval_end]

        import yaml
        # Prefer an absolute SETTINGS_DROUTE_PATH (set in coupled/parallel runs where the per-process
        # settings dir doesn't carry droute_lakes.yaml); fall back to the passed settings_dir.
        droute_settings = self._gv('SETTINGS_DROUTE_PATH')
        lf = Path(settings_dir) / 'droute_lakes.yaml'
        if droute_settings and (Path(droute_settings) / 'droute_lakes.yaml').exists():
            lf = Path(droute_settings) / 'droute_lakes.yaml'
        raw = yaml.safe_load(open(lf, encoding='utf-8')) if lf.exists() else {}
        lakes = {'inline': (raw or {}).get('inline_lakes', {}) or {},
                 'subgrid': (raw or {}).get('subgrid_lakes', {}) or {}}

        # gauges from the mapping CSV (station, seg) + WSC daily obs. dRoute's native multi-gauge
        # layout (gauge_seg_mapping.csv + wsc_<station>_daily.csv) lives under observations/streamflow.
        # MULTI_GAUGE_OBS_DIR may point at a different layout owned by the land model's metric path
        # (e.g. LaMAH ID_*.csv in observations/streamflow/lamah_format), so only honour it when it
        # actually holds dRoute's mapping file; otherwise fall back to observations/streamflow.
        default_obs = domain / 'observations' / 'streamflow'
        obs_candidates = [self._gv('DROUTE_OBS_DIR'), self._gv('MULTI_GAUGE_OBS_DIR'), str(default_obs)]
        obs_dir = next((Path(d) for d in obs_candidates
                        if d and (Path(d) / 'gauge_seg_mapping.csv').exists()), default_obs)
        gmap = pd.read_csv(Path(obs_dir) / 'gauge_seg_mapping.csv')
        i_eval0 = int(np.argmax(daily.index >= pd.Timestamp(eval_start)))
        rec_dates = daily.index[i_eval0:]
        gauges = []
        for _, row in gmap.iterrows():
            seg = int(row['seg']); ridx = id_to_idx.get(seg)
            if ridx is None:
                continue
            o = pd.read_csv(Path(obs_dir) / f"wsc_{row['station']}_daily.csv",
                            parse_dates=['date']).set_index('date')['q_cms'].reindex(rec_dates).values
            if np.isfinite(o).sum() >= 10:
                gauges.append({'station': str(row['station']), 'ridx': int(ridx), 'obs': o.astype(float)})

        self._inp = dict(seg_ids=seg_ids, id_to_idx=id_to_idx, downstream_idx=downstream_idx,
                         lengths=lengths, slopes=slopes, daily=daily.values, dates=daily.index,
                         i_eval0=i_eval0, lakes=lakes, gauges=gauges, rn_path=rn_path)
        return self._inp

    def _run_model_multigauge(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Coupling-aware, multi-gauge routing. Routes the (fresh) SUMMA runoff through the dRoute
        network and saves per-gauge discharge. Engine-agnostic: Saint-Venant when ``_is_sv()`` else
        Muskingum-Cunge (+ inline lakes / reservoir operating rules). MC is the stable choice on
        domains where the transient SV solver is ill-conditioned."""
        # In a coupled run the COUPLED worker passes the upstream (land) model's fresh output dir as
        # ``coupling_source_dir``; route THAT iteration's SUMMA runoff rather than a stale project file.
        runoff_file = None
        csd = kwargs.get('coupling_source_dir')
        if csd:
            import glob
            cands = (sorted(glob.glob(str(Path(csd) / '**' / '*_timestep.nc'), recursive=True)) or
                     sorted(glob.glob(str(Path(csd) / '*_timestep.nc'))))
            if cands:
                runoff_file = cands[0]
            else:
                self.logger.error(f"coupling_source_dir {csd} has no *_timestep.nc SUMMA output")
                return False
        inp = self._load_inputs(Path(settings_dir), runoff_file=runoff_file, force=bool(runoff_file))
        per_reach = None
        pf = Path(settings_dir) / 'droute_routing_params.json'
        if pf.exists():
            per_reach = json.load(open(pf))['params']
        net = build_droute_network(inp['seg_ids'], inp['downstream_idx'], inp['lengths'],
                                   inp['slopes'], inp['lakes'], inp['id_to_idx'], per_reach)
        gauges = inp['gauges']
        Qd = self._route_gauges_sv(net, inp) if self._is_sv() else self._route_gauges_mc(net, inp)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(output_dir) / 'droute_streamflow.npz',
                 ridx=np.array([g['ridx'] for g in gauges]),
                 stations=np.array([g['station'] for g in gauges]),
                 Q=np.array([Qd[g['ridx']] for g in gauges]),
                 obs=np.array([g['obs'] for g in gauges]))
        return True

    def _route_gauges_sv(self, net, inp) -> Dict[int, List[float]]:
        """Daily-windowed Saint-Venant routing; returns {reach_idx: [discharge per eval day]}."""
        import contextlib
        import io
        dt_h = float(self._gv('DROUTE_ROUTING_DT_HOURS', 12.0))
        sub = int(round(DT_DAY / (dt_h * 3600.0)))
        c = droute.SaintVenantEnzymeConfig()
        c.dt = dt_h * 3600.0; c.n_nodes = int(self._gv('DROUTE_SV_NODES', 4))
        c.enable_adjoint = False; c.use_enzyme_adjoint = False
        rt = droute.SaintVenantEnzyme(net, c)
        order = np.asarray(net.topological_order(), dtype=int)
        runoff = inp['daily']; i0 = inp['i_eval0']; ndays = runoff.shape[0]
        gauges = inp['gauges']
        Qd: Dict[int, List[float]] = {g['ridx']: [] for g in gauges}
        with contextlib.redirect_stderr(io.StringIO()):
            for d in range(ndays):
                for _ in range(sub):
                    for idx in order:
                        rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
                    rt.route_timestep()
                if d >= i0:
                    for g in gauges:
                        Qd[g['ridx']].append(rt.get_discharge(g['ridx']))
        return Qd

    def _route_gauges_mc(self, net, inp) -> Dict[int, List[float]]:
        """Daily Muskingum-Cunge routing with inline lakes + global reservoir operating rules
        (from the in-memory calibration params); returns {reach_idx: [discharge per eval day]}.
        The stable engine at Calgary, where the transient SV solver blows up."""
        params = getattr(self, '_current_params', {}) or {}
        if any(k in params for k in self.RESERVOIR_PARAM_KEYS):
            self._apply_reservoir_operating_rules(net, inp['id_to_idx'], params, inp['lakes'])
        cfg = droute.RouterConfig(); cfg.dt = DT_DAY; cfg.enable_gradients = False
        rt = droute.MuskingumCungeRouter(net, cfg)
        order = np.asarray(net.topological_order(), dtype=int)
        runoff = inp['daily']; i0 = inp['i_eval0']; ndays = runoff.shape[0]
        n_seg = len(inp['seg_ids']); gauges = inp['gauges']
        Qd: Dict[int, List[float]] = {g['ridx']: [] for g in gauges}
        for d in range(ndays):
            for idx in order:
                rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
            rt.route_timestep()
            if d >= i0:
                allq = np.zeros(n_seg)
                allq[order] = rt.get_all_discharges()   # get_all_discharges() is topo-ordered
                for g in gauges:
                    Qd[g['ridx']].append(float(allq[g['ridx']]))
        return Qd

    def materialize_metric_inputs(self, graph_outputs: Dict[str, Any], output_dir: Path,
                                  settings_dir: Path, config: Dict[str, Any]) -> bool:
        """Graph->metrics handoff: turn the dCoupler routing-component discharge tensor into the
        per-gauge ``droute_streamflow.npz`` that :meth:`calculate_metrics` reads.

        The dCoupler graph emits raw flux tensors (``{component: {flux: tensor}}``); the routing
        component's ``discharge`` is ``[n_days, n_reach]`` (m^3/s) over the full routed window. This
        slices the evaluation window + the gauge reaches (reusing the same gauge/obs metadata and
        spin-up offset as the standalone routing run) so the objective is computed identically
        whether routing ran via the in-memory graph or the sequential file path.
        """
        discharge = self._extract_discharge(graph_outputs)
        if discharge is None:
            self.logger.error("dCoupler graph outputs contained no 'discharge' flux; "
                              "cannot materialize dRoute metric inputs")
            return False
        inp = self._load_inputs(Path(settings_dir))
        i0 = int(inp['i_eval0'])
        gauges = inp['gauges']
        n_eval = discharge.shape[0] - i0
        if n_eval <= 0:
            self.logger.error(f"Routed discharge ({discharge.shape[0]} days) shorter than the "
                              f"spin-up offset ({i0}); cannot evaluate")
            return False
        Q = np.array([discharge[i0:i0 + n_eval, g['ridx']] for g in gauges])
        obs = np.array([g['obs'][:n_eval] for g in gauges])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(output_dir) / 'droute_streamflow.npz',
                 ridx=np.array([g['ridx'] for g in gauges]),
                 stations=np.array([g['station'] for g in gauges]),
                 Q=Q, obs=obs)
        return True

    @staticmethod
    def _extract_discharge(graph_outputs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Pull the [n_days, n_reach] discharge array out of the graph outputs (prefer 'routing')."""
        def _to_np(t):
            return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
        if not isinstance(graph_outputs, dict):
            return None
        for key in ('routing', 'droute', 'DROUTE'):
            comp = graph_outputs.get(key)
            if isinstance(comp, dict) and 'discharge' in comp:
                return _to_np(comp['discharge'])
        for comp in graph_outputs.values():
            if isinstance(comp, dict) and 'discharge' in comp:
                return _to_np(comp['discharge'])
        return None

    def _metrics_multigauge(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        z = np.load(Path(output_dir) / 'droute_streamflow.npz', allow_pickle=True)
        Q, obs, stations = z['Q'], z['obs'], z['stations']
        floor = float(self._gv('MULTI_GAUGE_KGE_FLOOR', -2.0))
        # Optional per-gauge weighting (station -> weight) for the objective. Lets the mean emphasize
        # a target gauge (e.g. the regulated outlet) while keeping the others as soft constraints.
        # Absent/empty config -> all weights 1.0, i.e. the original flat mean.
        weights_cfg = self._gv('MULTI_GAUGE_WEIGHTS', None) or {}
        per = {}
        kept = []        # (kge, weight) for gauges above the floor
        nse_kept = []    # (nse, weight) per gauge for the NSE objective
        for i, st in enumerate(stations):
            st_s = str(st)
            w = float(weights_cfg.get(st_s, 1.0))
            k = _kge_multigauge(Q[i], obs[i])
            n = _nse_multigauge(Q[i], obs[i])
            per[f'KGE_{st_s}'] = k
            per[f'NSE_{st_s}'] = n
            if np.isfinite(k) and k >= floor:   # drop volume-biased gauges routing can't fit
                kept.append((k, w))
            if np.isfinite(n):
                nse_kept.append((n, w))
        wsum = sum(w for _, w in kept)
        mean_kge = float(sum(k * w for k, w in kept) / wsum) if wsum > 0 else floor
        nse_wsum = sum(w for _, w in nse_kept)
        mean_nse = float(sum(n * w for n, w in nse_kept) / nse_wsum) if nse_wsum > 0 else self.penalty_score
        # Expose both metrics (upper + lowercase aliases) so the optimizer's OPTIMIZATION_METRIC lookup
        # resolves whether it asks for NSE or KGE; calib_score (the maximization objective) follows it.
        per['KGE'] = mean_kge
        per['NSE'] = mean_nse
        per['kge'] = mean_kge
        per['nse'] = mean_nse
        metric_name = str(self._gv('OPTIMIZATION_METRIC', 'KGE') or 'KGE').upper()
        per['calib_score'] = mean_nse if metric_name == 'NSE' else mean_kge
        return per

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_droute_parameters_worker(task_data)


def _evaluate_droute_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary containing params, config, etc.

    Returns:
        Result dictionary with score and metrics.
    """
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    # Small random delay
    time.sleep(random.uniform(0.05, 0.2))  # nosec B311

    try:
        worker = DRouteWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 -- calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'dRoute worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }


__all__ = ['DRouteWorker']
