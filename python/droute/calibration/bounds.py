# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SYMFLUENCE Team <dev@symfluence.org>

"""Calibration parameter bounds owned by the dRoute package.

Following the JAX-model plugin pattern, the model package owns its own parameter
bounds rather than relying on the SYMFLUENCE shared bounds registry. Three sets:

- :data:`DROUTE_LUMPED_BOUNDS` -- the lumped Muskingum-Cunge routing parameters
  (migrated verbatim from ``symfluence ... parameter_bounds_registry.DROUTE_PARAMS``).
- :data:`RESERVOIR_BOUNDS` -- the global inline-reservoir operating-rule parameters
  (used by the Muskingum-Cunge + lakes path).
- :data:`DROUTE_REGION_PARAM_BOUNDS` -- the per-reach Saint-Venant routing + lake
  rating parameters that the transfer-function regionalization calibrates.

Each entry is ``{'min': float, 'max': float, 'transform': 'linear'|'log'}``.
"""

from typing import Any, Dict, List

# --- Lumped Muskingum-Cunge routing parameters (was SYMFLUENCE DROUTE_PARAMS) -------------------
DROUTE_LUMPED_BOUNDS: Dict[str, Dict[str, Any]] = {
    'velocity':    {'min': 0.1,   'max': 5.0,    'transform': 'linear'},  # m/s
    'diffusivity': {'min': 100.0, 'max': 5000.0, 'transform': 'linear'},  # m^2/s
    'muskingum_k': {'min': 0.1,   'max': 24.0,   'transform': 'linear'},  # hours
    'muskingum_x': {'min': 0.0,   'max': 0.5,    'transform': 'linear'},  # dimensionless
    'manning_n':   {'min': 0.01,  'max': 0.1,    'transform': 'linear'},  # dimensionless
}

# --- Global inline-reservoir operating-rule parameters (Muskingum-Cunge + lakes path) ----------
RESERVOIR_BOUNDS: Dict[str, Dict[str, Any]] = {
    'reservoir_q_ref_mult': {'min': 0.2, 'max': 5.0, 'transform': 'log'},
    'reservoir_exp':        {'min': 1.0, 'max': 3.0, 'transform': 'linear'},
    'reservoir_q_min_frac': {'min': 0.0, 'max': 0.5, 'transform': 'linear'},
    'reservoir_spill_coef': {'min': 0.1, 'max': 3.0, 'transform': 'linear'},
}

# --- Per-reach Saint-Venant routing + lake rating parameters (regionalized) ---------------------
# Manning's n and the lake rating params mirror the standalone dRoute SV calibration experiments.
DROUTE_REGION_PARAM_BOUNDS: Dict[str, Dict[str, Any]] = {
    'manning_n':       {'min': 0.015, 'max': 0.12,   'transform': 'linear'},
    'lake_q_ref':      {'min': 0.05,  'max': 500.0,  'transform': 'log'},
    'lake_exp':        {'min': 1.0,   'max': 3.0,    'transform': 'linear'},
    'lake_q_min':      {'min': 0.0,   'max': 50.0,   'transform': 'linear'},
    'lake_spill_coef': {'min': 0.1,   'max': 3.0,    'transform': 'linear'},
    'subgrid_q_ref':   {'min': 0.05,  'max': 1000.0, 'transform': 'log'},
    'subgrid_exp':     {'min': 1.0,   'max': 3.0,    'transform': 'linear'},
}
DEFAULT_REGION_PARAMS: List[str] = ['manning_n', 'lake_q_ref', 'lake_q_min', 'lake_exp',
                                    'subgrid_q_ref', 'subgrid_exp']


def get_droute_bounds() -> Dict[str, Dict[str, Any]]:
    """Return the lumped Muskingum-Cunge routing parameter bounds (package-owned).

    Drop-in replacement for the former
    ``symfluence.optimization.core.parameter_bounds_registry.get_droute_bounds``.
    """
    return {k: dict(v) for k, v in DROUTE_LUMPED_BOUNDS.items()}
