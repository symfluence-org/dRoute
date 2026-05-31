# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
dRoute Parameter Manager.

Handles parameter bounds, normalization, and configuration file updates
for dRoute routing parameter calibration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_droute_bounds


class DRouteParameterManager(BaseParameterManager):
    """
    Parameter manager for dRoute routing calibration.

    Manages core routing parameters:
    - velocity: Base flow velocity (m/s)
    - diffusivity: Diffusion coefficient (m2/s)
    - muskingum_k: Muskingum storage constant (hours)
    - muskingum_x: Muskingum weighting factor (dimensionless)
    - manning_n: Manning's roughness coefficient (dimensionless)

    Reservoir operating-rule parameters (global, applied to every inline
    reservoir; the worker scales each reservoir's HydroLAKES-initialised values):
    - reservoir_q_ref_mult: multiplier on each reservoir's reference release q_ref
    - reservoir_exp:        storage-discharge rating exponent
    - reservoir_q_min_frac: minimum regulated release as a fraction of q_ref
    - reservoir_spill_coef: above-full spill coefficient
    """

    # Default bounds for the global reservoir operating-rule parameters. These are
    # not in the shared SYMFLUENCE droute-bounds registry, so provide them here
    # (still overridable via the DROUTE_PARAM_BOUNDS config key).
    RESERVOIR_BOUNDS = {
        'reservoir_q_ref_mult': {'min': 0.2, 'max': 5.0, 'transform': 'log'},
        'reservoir_exp': {'min': 1.0, 'max': 3.0, 'transform': 'none'},
        'reservoir_q_min_frac': {'min': 0.0, 'max': 0.5, 'transform': 'none'},
        'reservoir_spill_coef': {'min': 0.1, 'max': 3.0, 'transform': 'none'},
    }

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.project_dir = (
            Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
            / f"domain_{self.domain_name}"
        )

        # Parse parameters to calibrate from config
        params_str = self._get_config_value(
            lambda: self.config.model.droute.params_to_calibrate, default='velocity,diffusivity', dict_key='DROUTE_PARAMS_TO_CALIBRATE'
        )
        self.droute_params = [
            p.strip() for p in params_str.split(',') if p.strip()
        ]

    def _get_parameter_names(self) -> List[str]:
        """Return list of dRoute parameters to calibrate."""
        return self.droute_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Load parameter bounds from registry defaults, with config overrides.

        Config key DROUTE_PARAM_BOUNDS overrides min/max while preserving
        transform metadata from the registry.
        """
        registry_bounds = get_droute_bounds()

        bounds = {}
        for param in self.droute_params:
            if param in registry_bounds:
                bounds[param] = registry_bounds[param]
            elif param in self.RESERVOIR_BOUNDS:
                bounds[param] = dict(self.RESERVOIR_BOUNDS[param])
            else:
                self.logger.warning(
                    f"No bounds found for dRoute param '{param}', using [0, 1]"
                )
                bounds[param] = {'min': 0.0, 'max': 1.0}

        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='DROUTE_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        Update dRoute configuration with calibrated parameters.

        For dRoute, parameters are primarily passed in-memory to the worker.
        This method writes to the YAML config file for record-keeping.

        Args:
            params: Dictionary of parameter name -> value

        Returns:
            True if update succeeded.
        """
        try:
            config_path = self.settings_dir / 'droute_config.yaml'
            if not config_path.exists():
                self.logger.debug("No droute_config.yaml to update (in-memory mode)")
                return True

            import yaml
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # Update routing parameters section
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

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """
        Get initial parameter values (midpoint of bounds).

        Uses geometric mean for log-transformed parameters.

        Returns:
            Dictionary of parameter name -> initial value, or None.
        """
        import math

        bounds = self.param_bounds
        initial = {}
        for param in self.droute_params:
            if param in bounds:
                b = bounds[param]
                if b.get('transform') == 'log' and b['min'] > 0:
                    initial[param] = math.sqrt(b['min'] * b['max'])
                else:
                    initial[param] = (b['min'] + b['max']) / 2.0
        return initial if initial else None
