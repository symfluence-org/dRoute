# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SYMFLUENCE Team <dev@symfluence.org>

"""dRoute Parameter Manager.

Two calibration modes, selected by ``DROUTE_REGIONALIZATION`` (whose default depends on
``DROUTE_ROUTING_METHOD``):

Lumped (Muskingum-Cunge default):
    Calibrates scalar routing parameters (velocity, diffusivity, muskingum_k, muskingum_x,
    manning_n) plus optional global inline-reservoir operating-rule parameters. Writes
    ``droute_config.yaml`` for record-keeping; the worker applies the params in-memory.

Regionalized (Saint-Venant default — ``transfer_function`` / ``zones`` / ``distributed``):
    Calibrates the regionalization COEFFICIENTS (e.g. ``manning_n_a`` / ``manning_n_b``) of the
    per-reach Saint-Venant routing roughness and inline/subgrid lake rating coefficients, so a
    full-watershed routing calibration stays low-dimensional. ``update_model_files`` expands the
    coefficients to per-reach physical values and writes ``droute_routing_params.json`` that the
    dRoute worker applies to the network.

The package owns all parameter bounds (see :mod:`.bounds`), matching the JAX-model plugin pattern.

Configuration keys:
    DROUTE_ROUTING_METHOD       : 'muskingum_cunge' (default) | 'saint_venant'
    DROUTE_REGIONALIZATION      : 'lumped' | 'transfer_function' | 'zones' | 'distributed'
    DROUTE_PARAMS_TO_CALIBRATE  : comma-separated params (defaults differ per mode)
    DROUTE_PARAM_BOUNDS         : optional min/max overrides (lumped mode)
    RIVER_NETWORK_SHAPEFILE     : river-network shapefile (regionalized mode)
    SETTINGS_DROUTE_PATH        : dRoute settings dir (holds droute_lakes.yaml)
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.exceptions import ConfigurationError
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

from .bounds import (
    DEFAULT_REGION_PARAMS,
    DROUTE_REGION_PARAM_BOUNDS,
    RESERVOIR_BOUNDS,
    get_droute_bounds,
)
from .droute_regionalization import create_droute_regionalization


class DRouteParameterManager(BaseParameterManager):
    """Parameter manager for dRoute routing calibration (lumped MC or regionalized SV)."""

    # Exposed on the class for callers/tests that introspect the reservoir bounds.
    RESERVOIR_BOUNDS = RESERVOIR_BOUNDS

    def __init__(self, config: Any, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        # Lumped-mode parameter list (used only when not regionalized)
        params_str = self._get_config_value(
            lambda: self.config.model.droute.params_to_calibrate,
            default='velocity,diffusivity', dict_key='DROUTE_PARAMS_TO_CALIBRATE')
        self.droute_params = [p.strip() for p in str(params_str).split(',') if p.strip()]

        # Regionalized-mode state (lazy)
        self._region = None                 # lazy ParameterRegionalization
        self._n_reaches: int = 0
        self._phys_params: List[str] = []   # physical routing params being regionalized

    # ---- mode selection ---------------------------------------------------------------------
    def _get(self, default, key):
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def _routing_method(self) -> str:
        return str(self._get('muskingum_cunge', 'DROUTE_ROUTING_METHOD')).lower().replace('-', '_')

    def _regionalization_method(self) -> str:
        # Saint-Venant calibrations default to transfer-function regionalization; Muskingum-Cunge
        # defaults to lumped scalar parameters. An explicit DROUTE_REGIONALIZATION overrides.
        default = 'transfer_function' if self._routing_method() in ('saint_venant', 'saintvenant', 'sv') else 'lumped'
        return str(self._get(default, 'DROUTE_REGIONALIZATION')).lower().replace('-', '_')

    def _regionalized(self) -> bool:
        return self._regionalization_method() not in ('lumped', 'none', '')

    # ---- regionalized-mode helpers ----------------------------------------------------------
    def _region_params_to_calibrate(self) -> List[str]:
        raw = self._get(None, 'DROUTE_PARAMS_TO_CALIBRATE')
        if not raw or str(raw).lower() in ('default', 'all'):
            return list(DEFAULT_REGION_PARAMS)
        return [p.strip() for p in str(raw).split(',') if p.strip()]

    def _river_network_path(self) -> Path:
        p = self._get(None, 'RIVER_NETWORK_SHAPEFILE')
        if p:
            return Path(p)
        import glob
        cand = glob.glob(str(self.settings_dir.parent.parent / 'shapefiles' / 'river_network' / '*.shp'))
        if not cand:
            raise ConfigurationError("RIVER_NETWORK_SHAPEFILE not set and none found under shapefiles/river_network")
        return Path(cand[0])

    def _load_lakes(self) -> Dict[str, Dict]:
        import yaml
        f = Path(self.settings_dir) / 'droute_lakes.yaml'
        if not f.exists():
            return {'inline': {}, 'subgrid': {}}
        with open(f, encoding='utf-8') as fh:
            raw = yaml.safe_load(fh) or {}
        return {'inline': raw.get('inline_lakes', {}) or {}, 'subgrid': raw.get('subgrid_lakes', {}) or {}}

    def _n_reaches_from_network(self) -> int:
        import geopandas as gpd
        return len(gpd.read_file(self._river_network_path()))

    def _build_regionalization(self):
        if self._region is not None:
            return
        self._phys_params = self._region_params_to_calibrate()
        # only keep params with known bounds
        self._phys_params = [p for p in self._phys_params if p in DROUTE_REGION_PARAM_BOUNDS]
        param_bounds = {p: (DROUTE_REGION_PARAM_BOUNDS[p]['min'], DROUTE_REGION_PARAM_BOUNDS[p]['max'])
                        for p in self._phys_params}
        self._n_reaches = self._n_reaches_from_network()
        lakes = self._load_lakes()
        self._region = create_droute_regionalization(
            method=self._regionalization_method(),
            param_bounds=param_bounds,
            n_reaches=self._n_reaches,
            river_network_path=self._river_network_path(),
            lakes=lakes,
            logger=self.logger,
        )
        self.logger.info(
            f"dRoute regionalization '{self._regionalization_method()}': {len(self._phys_params)} "
            f"physical params over {self._n_reaches} reaches -> "
            f"{len(self._region.get_calibration_parameters())} calibration coefficients")

    # ---- BaseParameterManager contract ------------------------------------------------------
    def _get_parameter_names(self) -> List[str]:
        if self._regionalized():
            self._build_regionalization()
            return list(self._region.get_calibration_parameters().keys())
        return self.droute_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        if self._regionalized():
            self._build_regionalization()
            return {name: {'min': lo, 'max': hi, 'transform': 'linear'}
                    for name, (lo, hi) in self._region.get_calibration_parameters().items()}

        # Lumped: package-owned bounds, with optional config override (preserving transform meta).
        registry_bounds = get_droute_bounds()
        bounds: Dict[str, Dict[str, Any]] = {}
        for param in self.droute_params:
            if param in registry_bounds:
                bounds[param] = registry_bounds[param]
            elif param in RESERVOIR_BOUNDS:
                bounds[param] = dict(RESERVOIR_BOUNDS[param])
            else:
                self.logger.warning(f"No bounds found for dRoute param '{param}', using [0, 1]")
                bounds[param] = {'min': 0.0, 'max': 1.0}

        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='DROUTE_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        bounds = self.param_bounds
        if self._regionalized():
            # midpoint of each coefficient range (spatially-uniform params at the bound midpoints)
            return {name: 0.5 * (b['min'] + b['max']) for name, b in bounds.items()}

        # Lumped: midpoint, or geometric mean for log-transformed parameters.
        initial: Dict[str, float] = {}
        for param in self.droute_params:
            if param in bounds:
                b = bounds[param]
                if b.get('transform') == 'log' and b['min'] > 0:
                    initial[param] = math.sqrt(b['min'] * b['max'])
                else:
                    initial[param] = (b['min'] + b['max']) / 2.0
        return initial if initial else None

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        if self._regionalized():
            return self._update_regionalized(params)
        return self._update_lumped(params)

    # ---- mode-specific persistence ----------------------------------------------------------
    def _update_lumped(self, params: Dict[str, float]) -> bool:
        """Write lumped routing parameters to droute_config.yaml (record-keeping; the worker
        applies them in-memory)."""
        try:
            config_path = self.settings_dir / 'droute_config.yaml'
            if not config_path.exists():
                self.logger.debug("No droute_config.yaml to update (in-memory mode)")
                return True

            import yaml
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            if 'routing' not in config:
                config['routing'] = {}
            for param_name, value in params.items():
                config['routing'][param_name] = float(value)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            return True
        except Exception as e:  # noqa: BLE001 -- calibration resilience
            self.logger.error(f"Error updating dRoute config: {e}")
            return False

    def _update_regionalized(self, params: Dict[str, Any]) -> bool:
        """Expand calibration coefficients to per-reach physical params and write them out."""
        self._build_regionalization()
        try:
            arr, pnames = self._region.to_distributed({k: float(v) for k, v in params.items()})
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self.logger.error(f"dRoute regionalization expansion failed: {e}")
            return False
        # clamp each physical parameter to its bounds
        for j, p in enumerate(pnames):
            b = DROUTE_REGION_PARAM_BOUNDS.get(p)
            if b:
                arr[:, j] = np.clip(arr[:, j], b['min'], b['max'])
        per_reach = {p: arr[:, j].tolist() for j, p in enumerate(pnames)}
        out = Path(self.settings_dir) / 'droute_routing_params.json'
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'n_reaches': int(self._n_reaches), 'params': per_reach}, fh)
        self.logger.debug(f"Wrote per-reach routing params -> {out}")
        return True
