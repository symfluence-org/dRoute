# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Darri Eythorsson

"""
dRoute Model Runner.

Manages the execution of the dRoute routing model.
Supports both Python API mode (preferred) and subprocess fallback.
"""

import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.models.base import BaseModelRunner
from .mixins import DRouteConfigMixin
from .network_adapter import DRouteNetworkAdapter

# Try to import dRoute Python bindings
try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False
    droute = None


class DRouteRunner(BaseModelRunner, DRouteConfigMixin):  # type: ignore[misc]
    """
    Runner for the dRoute routing model.

    Supports two execution modes:
    1. Python API (default): Direct calls to dRoute Python bindings
       - Faster execution (no subprocess overhead)
       - Enables native gradient computation via AD
       - Requires dRoute compiled with Python bindings

    2. Subprocess mode: Executes dRoute as external command
       - Fallback when Python bindings unavailable
       - Works with any dRoute installation
       - No gradient support

    The runner handles:
    - Loading runoff data from source hydrological models
    - Configuring routing parameters
    - Executing routing computation
    - Writing routed streamflow output

    Attributes:
        config: Configuration settings
        logger: Logger instance
        execution_mode: 'python' or 'subprocess'
    """

    MODEL_NAME = "dRoute"

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the dRoute runner.

        Args:
            config: Configuration dictionary or SymfluenceConfig
            logger: Logger instance
            reporting_manager: Optional reporting manager for progress tracking
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # Determine execution mode
        preferred_mode = self.droute_execution_mode
        if preferred_mode == 'python' and not HAS_DROUTE:
            self.logger.warning(
                "dRoute Python bindings not available, falling back to subprocess mode. "
                "Install dRoute with: pip install droute"
            )
            self.execution_mode = 'subprocess'
        else:
            self.execution_mode = preferred_mode

        self.logger.debug(f"dRoute execution mode: {self.execution_mode}")

        # Initialize network adapter
        self.network_adapter = DRouteNetworkAdapter(logger)

        # Cached network and routing function
        self._network_config = None
        self._routing_fn = None

    def _should_create_output_dir(self) -> bool:
        """dRoute creates directories on-demand."""
        return False

    def run_droute(self):
        """
        Run the dRoute routing model.

        Orchestrates the complete routing workflow:
        1. Load runoff from source hydrological model
        2. Load network topology
        3. Execute routing
        4. Write output

        Returns:
            Path to output directory
        """
        self.logger.info("Starting dRoute run")

        with symfluence_error_handler(
            "dRoute model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            if self.execution_mode == 'python':
                return self._run_python_mode()
            else:
                return self._run_subprocess_mode()

    def _run_python_mode(self):
        """
        Run dRoute using Python API.

        This is the preferred mode as it:
        - Avoids subprocess overhead
        - Enables native gradient computation
        - Provides better error handling
        """
        self.logger.info("Running dRoute in Python API mode")

        # Load network configuration
        network_config = self._load_network_config()
        if network_config is None:
            raise ModelExecutionError("Failed to load network configuration")

        # Load runoff input from source model
        runoff_data, time_index = self._load_runoff_input()
        if runoff_data is None:
            raise ModelExecutionError("Failed to load runoff data")

        self.logger.info(
            f"Loaded runoff data: {runoff_data.shape[0]} timesteps, "
            f"{runoff_data.shape[1]} HRUs"
        )

        # Get routing parameters
        routing_method = self.droute_routing_method
        routing_dt = self.droute_routing_dt

        # Execute routing (C++ bindings with numpy fallback)
        routed_flow = None
        if HAS_DROUTE:
            try:
                routed_flow = self._route_with_droute(
                    runoff_data, network_config, routing_method, routing_dt
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"dRoute C++ bindings failed ({e}), falling back to numpy routing."
                )

        if routed_flow is None:
            try:
                routed_flow = self._route_numpy_fallback(
                    runoff_data, network_config, routing_method, routing_dt
                )
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Routing computation failed: {e}")
                self.logger.debug(traceback.format_exc())
                raise ModelExecutionError(f"dRoute routing failed: {e}") from e

        # Save output
        output_dir = self._get_output_dir()
        self._save_output(routed_flow, time_index, network_config, output_dir)

        self.logger.info(f"dRoute completed. Output: {output_dir}")
        return output_dir

    def _run_subprocess_mode(self):
        """
        Run dRoute as external subprocess.

        Fallback mode when Python bindings are unavailable.
        """
        self.logger.info("Running dRoute in subprocess mode")

        # Get executable path
        droute_exe = self.get_model_executable(
            install_path_key='DROUTE_INSTALL_PATH',
            default_install_subpath='installs/droute/bin',
            exe_name_key='DROUTE_EXE',
            default_exe_name='droute',
            must_exist=True
        )

        # Get config file path
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        config_file = settings_path / self.droute_config_file

        if not config_file.exists():
            raise ModelExecutionError(f"dRoute config file not found: {config_file}")

        # Setup output directory
        output_dir = self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup log directory
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'droute_log.txt'

        # Build command
        command = [str(droute_exe), str(config_file)]

        self.logger.debug(f"Running dRoute command: {' '.join(command)}")

        # Execute
        self.execute_subprocess(
            command,
            log_file,
            success_message="dRoute run completed successfully"
        )

        return output_dir

    def _load_network_config(self) -> Optional[Dict[str, Any]]:
        """Load network configuration from preprocessor output."""
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        network_config_path = settings_path / 'droute_network.yaml'

        if network_config_path.exists():
            try:
                import yaml
                with open(network_config_path, encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # Flatten nested structure for routing
                network_config = {
                    'n_segments': config['network']['n_segments'],
                    'downstream_idx': config['network']['downstream_idx'],
                    'outlet_indices': config['network']['outlet_indices'],
                    'routing_order': config['network']['routing_order'],
                    'slopes': config['geometry']['slopes'],
                    'lengths': config['geometry']['lengths'],
                    'widths': config['geometry']['widths'],
                    'hru_to_seg_idx': config['hru_mapping']['hru_to_seg_idx'],
                    'hru_areas': config['hru_mapping'].get('hru_areas', []),
                    'hru_ids': config['hru_mapping'].get('hru_ids', []),
                    'segment_ids': config['network'].get('segment_ids', []),
                }
                return network_config

            except Exception as e:  # noqa: BLE001 -- model execution resilience
                self.logger.error(f"Error loading network config: {e}")
                return None

        # Try to load from topology file directly
        topology_path = self._find_topology_file()
        if topology_path:
            topology = self.network_adapter.load_topology(
                topology_path,
                format=self.droute_topology_format
            )
            return self.network_adapter.to_droute_format(
                topology,
                routing_method=self.droute_routing_method,
                routing_dt=self.droute_routing_dt
            )

        return None

    def _find_topology_file(self) -> Optional[Path]:
        """Find topology file in standard locations."""
        # Check dRoute settings
        settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
        droute_topo = settings_path / self.droute_topology_file
        if droute_topo.exists():
            return droute_topo

        # Check mizuRoute settings
        mizu_topo = self.project_dir / 'settings' / 'mizuRoute' / 'mizuRoute_topology.nc'
        if mizu_topo.exists():
            return mizu_topo

        return None

    def _load_runoff_input(self):
        """
        Load runoff data from source hydrological model.

        Uses MODEL_CONFIGS for model-specific variable detection and
        fix_time_precision for time format correction.

        Returns:
            Tuple of (runoff_array, time_index) where:
            - runoff_array: 2D array [time, hru] of runoff values
            - time_index: DatetimeIndex of timestamps
        """
        import pandas as pd
        import xarray as xr

        from symfluence.models.utilities.runoff_loader import (
            detect_runoff_variable,
            fix_time_precision,
            resolve_runoff_file,
        )

        source_model = self.droute_from_model.upper()
        if source_model == 'DEFAULT':
            hydro_model = self._get_config_value(lambda: self.config.model.hydrological_model, default='SUMMA')
            source_model = str(hydro_model).split(',')[0].strip().upper() if ',' in str(hydro_model) else str(hydro_model).strip().upper()

        # Use shared runoff file resolution
        runoff_file = resolve_runoff_file(
            source_model=source_model,
            project_dir=self.project_dir,
            experiment_id=self.experiment_id,
            domain_name=self.domain_name,
            config=self.config,
        )

        if runoff_file is None:
            # Fallback: search input dir directly
            input_dir = self._get_input_dir()
            self.logger.debug(f"Shared resolver failed, looking in: {input_dir}")
            runoff_file = self._find_runoff_file(input_dir)

        if runoff_file is None:
            self.logger.error("No runoff file found for dRoute")
            return None, None

        self.logger.info(f"Loading runoff from: {runoff_file}")

        # Fix time precision before loading
        fix_time_precision(runoff_file, rounding_freq='h')

        try:
            ds = xr.open_dataset(runoff_file)

            runoff_var = detect_runoff_variable(ds, source_model)
            if runoff_var is None:
                self.logger.error(f"No runoff variable found in {runoff_file}")
                ds.close()
                return None, None

            self.logger.debug(f"Using runoff variable: {runoff_var}")

            runoff = ds[runoff_var].values

            if runoff.ndim == 1:
                runoff = runoff.reshape(-1, 1)
            elif runoff.ndim == 3:
                runoff = runoff.sum(axis=-1)

            time_index = pd.DatetimeIndex(ds.time.values)

            ds.close()
            return runoff, time_index

        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.error(f"Error loading runoff: {e}")
            self.logger.debug(traceback.format_exc())
            return None, None

    def _find_runoff_file(self, input_dir: Path) -> Optional[Path]:
        """Find runoff NetCDF file in input directory."""
        if not input_dir.exists():
            return None

        # Common runoff file patterns
        patterns = [
            '*_timestep.nc',
            '*_output.nc',
            '*_runs_def.nc',
            '*_runoff.nc',
            '*.nc',
        ]

        for pattern in patterns:
            files = list(input_dir.glob(pattern))
            if files:
                return files[0]

        return None

    def _route_with_droute(
        self,
        runoff: np.ndarray,
        network: Dict[str, Any],
        method: str,
        dt: int
    ) -> np.ndarray:
        """
        Route runoff using dRoute C++ bindings (pybind11).

        Builds a Network from junctions + reaches, creates a
        MuskingumCungeRouter, and routes timestep-by-timestep.
        """
        if not HAS_DROUTE:
            raise ImportError("dRoute Python bindings not available")

        n_time, n_hru = runoff.shape
        n_segments = network['n_segments']
        hru_to_seg = network['hru_to_seg_idx']
        downstream_idx = network['downstream_idx']
        lengths = network['lengths']
        slopes = network['slopes']
        hru_areas = np.array(network.get('hru_areas', np.ones(n_hru)))

        # Aggregate runoff to segments (m/s depth rate → m³/s)
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += runoff[:, hru_idx] * hru_areas[hru_idx]

        # Build Network using the C++ API: junctions + reaches.
        # Each reach flows from upstream_junction to downstream_junction.
        # Junction.upstream_reach_ids must list reaches that flow INTO it.
        net = droute.Network()

        outlet_junc_id = n_segments
        widths = network.get('widths', [None] * n_segments)
        mannings_n = float(self._get_config_value(
            lambda: self.config.model.droute.mannings_n if self.config.model and hasattr(self.config.model, 'droute') else None,
            default=0.035,
        ))

        # Collect upstream_reach_ids per junction before creating objects
        junc_upstream: dict = {i: [] for i in range(n_segments + 1)}
        for i in range(n_segments):
            ds = downstream_idx[i]
            ds_junc = ds if ds >= 0 else outlet_junc_id
            junc_upstream[ds_junc].append(i)

        # Create and add junctions with wiring set upfront
        for jid in range(n_segments + 1):
            junc = droute.Junction()
            junc.id = jid
            junc.upstream_reach_ids = junc_upstream[jid]
            net.add_junction(junc)

        # Create and add reaches
        for i in range(n_segments):
            reach = droute.Reach()
            reach.id = i
            reach.length = float(lengths[i])
            reach.slope = max(float(slopes[i]), 0.001)
            reach.manning_n = mannings_n
            reach.upstream_junction_id = i
            ds = downstream_idx[i]
            reach.downstream_junction_id = ds if ds >= 0 else outlet_junc_id

            if widths[i] is not None:
                geom = reach.geometry
                geom.width_coef = float(widths[i])
                geom.width_exp = 0.0
                reach.geometry = geom

            net.add_reach(reach)

        net.build_topology()
        topo_order = net.topological_order()
        self.logger.info(f"dRoute C++ network built: {net.num_reaches()} reaches, {net.num_junctions()} junctions")

        # Apply lake/reservoir config (settings/dRoute/droute_lakes.yaml) onto the
        # built network before the router is constructed. No-op if absent.
        self._apply_lakes(net, network)

        # Configure router
        config = droute.RouterConfig()
        config.dt = float(dt)
        config.enable_gradients = False

        router = droute.MuskingumCungeRouter(net, config)

        # Route timestep-by-timestep.
        # NOTE: get_all_discharges() returns values ordered by the network's
        # topological_order(), NOT by reach/segment index. Scatter them back to
        # segment-index columns via topo_order so routed_flow[:, i] is segment i.
        topo_arr = np.asarray(topo_order, dtype=int)
        routed_flow = np.zeros((n_time, n_segments))
        for t in range(n_time):
            for idx in topo_order:
                router.set_lateral_inflow(idx, float(segment_runoff[t, idx]))
            router.route_timestep()
            routed_flow[t, topo_arr] = router.get_all_discharges()

        return routed_flow

    def _apply_lakes(self, net, network: Dict[str, Any]) -> int:
        """Apply settings/dRoute/droute_lakes.yaml onto the built C++ network.

        Maps river-network segId -> reach index (reach.id == index) and sets the
        inline-lake / subgrid-lake fields so route_timestep() routes lakes as
        storage-discharge reservoirs. Returns reaches modified (0 if no config).
        """
        try:
            settings_path = self.get_config_path('SETTINGS_DROUTE_PATH', 'settings/dRoute/')
            lake_yaml = settings_path / 'droute_lakes.yaml'
            if not lake_yaml.exists():
                return 0
            import yaml
            from droute.lake_preprocessor import apply_lake_config_to_network
            with open(lake_yaml, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {}
            classification = {'inline': raw.get('inline_lakes', {}) or {},
                              'subgrid': raw.get('subgrid_lakes', {}) or {}}
            seg_ids = network['segment_ids']
            segid_to_index = {int(s): i for i, s in enumerate(seg_ids)}
            n = apply_lake_config_to_network(net, classification, segid_to_index, logger=self.logger)
            self.logger.info(
                f"dRoute lakes applied: {n} reaches "
                f"({len(classification['inline'])} inline, {len(classification['subgrid'])} subgrid)")
            return n
        except Exception as e:  # noqa: BLE001 - lakes optional, never block routing
            self.logger.warning(f"Lake config not applied: {type(e).__name__}: {e}")
            return 0

    def _route_numpy_fallback(
        self,
        runoff: np.ndarray,
        network: Dict[str, Any],
        method: str,
        dt: int
    ) -> np.ndarray:
        """
        Simple numpy-based routing fallback.

        Implements basic Muskingum-Cunge routing when dRoute unavailable.
        """
        self.logger.warning("Using numpy fallback routing (dRoute bindings unavailable)")

        n_time, n_hru = runoff.shape
        n_segments = network['n_segments']
        hru_to_seg = network['hru_to_seg_idx']
        downstream = network['downstream_idx']
        routing_order = network['routing_order']
        lengths = np.array(network['lengths'])
        slopes = np.array(network['slopes'])

        # Aggregate runoff to segments (convert m/s depth rate to m³/s)
        hru_areas = np.array(network.get('hru_areas', np.ones(n_hru)))
        segment_runoff = np.zeros((n_time, n_segments))
        for hru_idx, seg_idx in enumerate(hru_to_seg):
            if seg_idx >= 0 and hru_idx < n_hru:
                segment_runoff[:, seg_idx] += runoff[:, hru_idx] * hru_areas[hru_idx]

        # Simple Muskingum routing
        # Q_out = C1*I + C2*I_prev + C3*Q_prev
        routed = np.zeros((n_time, n_segments))

        # Estimate velocity from slope (Manning-like)
        velocity = 1.0 * np.sqrt(slopes)  # m/s
        velocity = np.clip(velocity, 0.1, 5.0)

        # Travel time
        travel_time = lengths / velocity / 3600.0  # hours

        # Muskingum parameters
        K = travel_time  # hours
        x = 0.2  # weighting factor

        # Routing coefficients (simplified)
        for t in range(n_time):
            for seg_idx in routing_order:
                # Local runoff
                Q_local = segment_runoff[t, seg_idx]

                # Upstream inflow
                Q_upstream = 0.0
                for up_idx in range(n_segments):
                    if downstream[up_idx] == seg_idx:
                        Q_upstream += routed[t, up_idx] if t > 0 else 0.0

                # Previous timestep
                Q_prev = routed[t-1, seg_idx] if t > 0 else 0.0
                I_prev = segment_runoff[t-1, seg_idx] + (
                    sum(routed[t-1, up_idx] for up_idx in range(n_segments)
                        if downstream[up_idx] == seg_idx) if t > 0 else 0.0
                )

                # Total inflow
                I = Q_local + Q_upstream

                # Muskingum routing (explicit scheme)
                k = max(K[seg_idx], 0.1)
                denom = 2*k*(1-x) + dt/3600
                C1 = (dt/3600 - 2*k*x) / denom
                C2 = (dt/3600 + 2*k*x) / denom
                C3 = (2*k*(1-x) - dt/3600) / denom

                routed[t, seg_idx] = max(0, C1*I + C2*I_prev + C3*Q_prev)

        return routed

    def _get_input_dir(self) -> Path:
        """Get input directory for runoff data from source model."""
        from_model = self.droute_from_model.upper()
        if from_model == 'DEFAULT':
            hydro_model = self._get_config_value(lambda: self.config.model.hydrological_model, default='SUMMA')
            if ',' in str(hydro_model):
                from_model = str(hydro_model).split(',')[0].strip().upper()
            else:
                from_model = str(hydro_model).strip().upper()

        experiment_output = self._get_config_value(lambda: getattr(self.config.paths, f'experiment_output_{from_model.lower()}', None), default=None)
        if experiment_output and experiment_output != 'default':
            return Path(experiment_output)

        return self.project_dir / f"simulations/{self.experiment_id}" / from_model

    def _get_output_dir(self) -> Path:
        """Get output directory for routed streamflow."""
        droute_output = self.droute_experiment_output
        if droute_output and droute_output != 'default':
            output_dir = Path(droute_output)
        else:
            output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'dRoute'

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_output(
        self,
        routed_flow: np.ndarray,
        time_index,
        network: Dict[str, Any],
        output_dir: Path
    ):
        """Save routed streamflow to NetCDF file."""
        import xarray as xr

        n_time, n_segments = routed_flow.shape
        outlet_indices = network['outlet_indices']

        # Create dataset
        ds = xr.Dataset(
            {
                'routedRunoff': (['time', 'seg'], routed_flow.astype(np.float32)),
            },
            coords={
                'time': time_index,
                'seg': np.arange(n_segments),
            }
        )

        ds['routedRunoff'].attrs = {
            'units': 'm3/s',
            'long_name': 'Routed streamflow',
        }

        # Add outlet streamflow as separate variable
        if outlet_indices:
            outlet_idx = outlet_indices[0]
            ds['outletStreamflow'] = (['time'], routed_flow[:, outlet_idx])
            ds['outletStreamflow'].attrs = {
                'units': 'm3/s',
                'long_name': 'Streamflow at outlet',
            }

        # Add metadata
        ds.attrs['routing_method'] = self.droute_routing_method
        ds.attrs['routing_dt'] = self.droute_routing_dt
        ds.attrs['n_outlets'] = len(outlet_indices)

        # Save
        output_file = output_dir / f"{self.experiment_id}_droute_output.nc"
        ds.to_netcdf(output_file)
        ds.close()

        self.logger.info(f"Saved dRoute output to {output_file}")

    def compute_gradients(
        self,
        params: Dict[str, float],
        runoff: np.ndarray,
        network: Dict[str, Any],
        obs: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradients using dRoute's AD capabilities.

        Args:
            params: Routing parameters (e.g., velocity, diffusivity)
            runoff: Input runoff array
            network: Network configuration
            obs: Observed streamflow for loss calculation

        Returns:
            Dictionary of parameter gradients, or None if AD unavailable
        """
        if not HAS_DROUTE:
            self.logger.warning("dRoute bindings not available for gradient computation")
            return None

        if not self.droute_enable_gradients:
            self.logger.warning("Gradient computation not enabled in config")
            return None

        try:
            # Check if AD is enabled in dRoute build
            if not hasattr(droute, 'gradient'):
                self.logger.warning("dRoute not compiled with AD support")
                return None

            # Create AD-enabled router
            net = droute.Network(
                n_segments=network['n_segments'],
                downstream=network['downstream_idx'],
                lengths=network['lengths'],
                slopes=network['slopes'],
                widths=network['widths']
            )

            router = droute.Router(net, enable_ad=True)

            # Compute gradients
            gradients = router.compute_gradients(
                runoff=runoff,
                params=params,
                observations=obs,
                metric='kge'
            )

            return gradients

        except Exception as e:  # noqa: BLE001 -- model execution resilience
            self.logger.error(f"Error computing gradients: {e}")
            self.logger.debug(traceback.format_exc())
            return None


__all__ = ['DRouteRunner']
