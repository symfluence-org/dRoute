# dRoute: Differentiable River Routing Library

[![Build Status](https://github.com/DarriEy/dRoute/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/DarriEy/dRoute/actions/workflows/build-wheels.yml)
[![PyPI version](https://badge.fury.io/py/droute.svg)](https://badge.fury.io/py/droute)
[![Python Versions](https://img.shields.io/pypi/pyversions/droute.svg)](https://pypi.org/project/droute/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A differentiable river routing library for hydrological modeling. dRoute implements multiple routing methods with automatic differentiation support, enabling gradient-based parameter optimization and integration with machine learning workflows.

***Note dRoute is in active development - expect experimental code***

## Features

- **6 Routing Methods**: From simple lag routing to Saint-Venant equations
- **Dual AD Backends**: CoDiPack (tape-based) and Enzyme (source-to-source) 
- **Network Topology**: Full support for river networks with tributaries and confluences
- **PyTorch Integration**: Use dRoute gradients in ML training loops

## Routing Methods

| Method | Class | Physics | Speed | Use Case |
|--------|-------|---------|-------|----------|
| **Muskingum-Cunge** | `MuskingumCungeRouter` | Kinematic + diffusion approx | ~4,500/s | Production routing |
| **Lag** | `LagRouter` | Time delay buffer | ~20,000/s | Baseline comparison |
| **IRF** | `IRFRouter` | Gamma unit hydrograph | ~1,000/s | Fast calibration |
| **KWT-Soft** | `SoftGatedKWT` | Kinematic wave tracking | ~4,000/s | Differentiable Lagrangian |
| **Diffusive Wave** | `DiffusiveWaveIFT` | Diffusion wave PDE | ~3,000/s | Flood wave attenuation |
| **Saint-Venant** | `SaintVenantRouter` | Full shallow water eqs | ~100/s | High-fidelity benchmark |

## Quick Start

**Build requirements:** CMake 3.15+, a C++17 compiler, and Python development headers. Optional features (NetCDF, Enzyme, SUNDIALS) require those libraries installed.

## Build Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `DMC_BUILD_PYTHON` | OFF | Build Python bindings |
| `DMC_USE_CODIPACK` | ON | Enable CoDiPack AD |
| `DMC_ENABLE_ENZYME` | OFF | Enable Enzyme AD backend |
| `DMC_ENABLE_SUNDIALS` | OFF | Enable SUNDIALS for SVE solver |
| `DMC_ENABLE_NETCDF` | OFF | Enable NetCDF topology I/O |
| `DMC_ENABLE_OPENMP` | OFF | Enable OpenMP parallelization |
| `DMC_BUILD_TESTS` | ON | Build C++ test suite |

### Build with All Features

```bash
cmake -S . -B build \
    -DDMC_BUILD_PYTHON=ON \
    -DDMC_ENABLE_ENZYME=ON \
    -DDMC_ENABLE_SUNDIALS=ON \
    -DSUNDIALS_ROOT=/path/to/sundials/install \
    -DDMC_ENABLE_NETCDF=ON
    -DMC_ENABLE_OPENMP=ON

cmake --build build -j4
```

### macOS (Apple Silicon)

```bash
brew install cmake netcdf sundials

cmake -S . -B build -DDMC_BUILD_PYTHON=ON -DDMC_ENABLE_SUNDIALS=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```
### Python installation

```bash
git clone https://github.com/DarriEy/dRoute.git
cd dRoute
pip install -e .

# Or use PyPi

pip install droute
```

## Data 

### Download Sample Data

The example dataset is hosted as a GitHub release asset (v0.5.0).

```bash
python scripts/download_data.py
```

### Run with Sample Data

```bash
# Forward pass comparison (all methods)
python python/test_routing_with_data.py --data-dir data

# Fast optimization with Enzyme kernels (30 epochs in ~30s)
python python/test_routing_with_data.py --data-dir data --optimize --fast

# Include Saint-Venant high-fidelity benchmark
python python/test_routing_with_data.py --data-dir data --sve
```

## Architecture

```
dRoute/
├── include/dmc/
│   ├── router.hpp              # MuskingumCungeRouter with CoDiPack AD
│   ├── advanced_routing.hpp    # IRF, KWT, Diffusive routers  
│   ├── kernels_enzyme.hpp      # Enzyme-compatible kernels (all 5 methods)
│   ├── unified_router.hpp      # EnzymeRouter wrapper
│   ├── saint_venant_router.hpp # Full SVE with SUNDIALS CVODES
│   ├── network.hpp             # Network topology
│   └── types.hpp               # AD type definitions
├── python/
│   ├── bindings.cpp            # pybind11 bindings
│   └── test_routing_with_data.py
├── tests/                      # C++ test suite
└── CMakeLists.txt
```

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- pybind11 (auto-downloaded if not found)
- CoDiPack (auto-downloaded)
- Optional: Enzyme, SUNDIALS, NetCDF-C, OpenMP


## Citation

```bibtex
@software{dRoute2025,
  title={dRoute: Differentiable River Routing Library},
  author={Eythorsson, Darri},
  year={2024},
  url={https://github.com/DarriEy/dRoute}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [CoDiPack](https://github.com/SciCompKL/CoDiPack) - Tape-based automatic differentiation
- [Enzyme](https://enzyme.mit.edu/) - Source-to-source AD compiler plugin
- [SUNDIALS](https://computing.llnl.gov/projects/sundials) - Implicit ODE solvers
- [SUMMA](https://github.com/NCAR/summa) & [mizuRoute](https://github.com/NCAR/mizuRoute) - Hydrological modeling inspiration
