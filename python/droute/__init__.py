# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Darri Eythorsson

"""
droute - Differentiable River Routing Library

This package provides:
1. C++ routing engine with Python bindings (via _droute_core extension)
2. SYMFLUENCE plugin integration for workflow orchestration

The C++ extension provides the core routing algorithms (Muskingum-Cunge, IRF,
Lag, Diffusive Wave, KWT) with optional automatic differentiation via
CoDiPack/Enzyme.

The SYMFLUENCE integration (activated via the register() entry point) provides:
- DRoutePreProcessor: Network topology setup
- DRouteRunner: Model execution (Python API or subprocess)
- DRoutePostProcessor: Result extraction and processing
- DRouteWorker: Calibration with gradient support
"""

from importlib import import_module
from ._version import __version__

__author__ = "Darri Eythorsson"

try:
    _module = import_module("_droute_core")
except ImportError as exc:
    raise ImportError(
        "droute requires the compiled extension module '_droute_core'. "
        "Please ensure the package is properly installed.\n"
        "Install with: pip install droute\n"
        "Or for development: pip install -e ."
    ) from exc

# Merge C++ extension exports, but preserve Python package identity
_preserve = {k: v for k, v in globals().items() if k in ('__file__', '__package__', '__path__', '__spec__', '__name__', '__loader__')}
globals().update(_module.__dict__)
globals().update(_preserve)

__all__ = getattr(
    _module,
    "__all__",
    [name for name in _module.__dict__ if not name.startswith("_")],
)


def register() -> None:
    """Register dRoute components with symfluence plugin registry."""
    from symfluence.core.registry import model_manifest
    from .config import DRouteConfig, DRouteConfigAdapter
    from .runner import DRouteRunner
    from .preprocessor import DRoutePreProcessor
    from .postprocessor import DRoutePostProcessor
    from .extractor import DRouteResultExtractor
    from .calibration.optimizer import DRouteModelOptimizer
    from .calibration.worker import DRouteWorker
    from .calibration.parameter_manager import DRouteParameterManager

    model_manifest(
        "DROUTE",
        config_adapter=DRouteConfigAdapter,
        config_schema=DRouteConfig,
        preprocessor=DRoutePreProcessor,
        runner=DRouteRunner,
        runner_method='run_droute',
        postprocessor=DRoutePostProcessor,
        result_extractor=DRouteResultExtractor,
        optimizer=DRouteModelOptimizer,
        worker=DRouteWorker,
        parameter_manager=DRouteParameterManager,
        build_instructions_module="droute.build_instructions",
    )
