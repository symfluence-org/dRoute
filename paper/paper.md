---
title: 'dRoute: A differentiable river routing library for hydrological modeling'
tags:
  - hydrology
  - river routing
  - automatic differentiation
  - differentiable modeling
  - machine learning
  - Python
  - C++
authors:
  - name: Darri Eythorsson
    orcid: 0000-0000-0000-0000  # TODO: add ORCID
    affiliation: 1
    corresponding: true
affiliations:
  - name: Department of Civil Engineering, University of Calgary, Calgary, Alberta, Canada
    index: 1
date: 30 May 2026
bibliography: paper.bib
---

# Summary

`dRoute` is an open-source library for routing water through river networks that is
*differentiable end to end*: in addition to simulating discharge, it returns exact
derivatives of the simulated streamflow with respect to its physical parameters. It
implements six routing schemes that span the usual fidelity/cost trade-off in
hydrology — a simple lag, an impulse-response (gamma unit hydrograph) method,
Muskingum–Cunge, a soft-gated kinematic-wave tracking scheme, a diffusive-wave
solver, and a full Saint-Venant (dynamic-wave) benchmark — and routes them over
arbitrary river-network topologies with tributaries and confluences. The numerical
core is written in C++17 and exposed to Python through `pybind11`, so the same code
runs as a standalone routing engine, as a callable inside a Python workflow, or as a
gradient-producing layer inside a `PyTorch` [@paszke2019] training loop.

Derivatives are obtained by automatic differentiation (AD) rather than by hand-coded
adjoints or finite differences, through two interchangeable backends: `CoDiPack`
[@sagebaum2019], an operator-overloading (tape-based) approach, and `Enzyme`
[@moses2020], a compiler-level source-to-source approach. The two backends compute
the same gradients by different mechanisms, which both gives users a performance
choice and provides a built-in cross-check on gradient correctness. `dRoute`
interoperates with the SUMMA land model [@clark2015] and is compatible with
`mizuRoute` network-topology files [@mizukami2016], so it can ingest runoff from
existing modeling chains, and it exposes a Basic Model Interface (BMI) for coupling.

# Statement of need

River routing — translating distributed runoff into streamflow along a channel
network — is a standard component of large-scale hydrological and Earth-system
models. Its parameters (channel roughness, wave celerity, diffusivity, Muskingum
storage coefficients) are uncertain and are normally estimated by calibration.
Established routing tools are not differentiable, so calibration relies on
derivative-free search (e.g. DDS [@tolson2007], SCE-UA, particle swarm). These
methods are robust but scale poorly with the number of parameters, because the
required number of model evaluations grows sharply as parameter dimension increases —
exactly the regime entered when roughness or celerity is allowed to vary per reach in
a large network.

Differentiable modeling addresses this by making the simulator return gradients, so
that gradient-based optimizers and machine-learning components can be used directly
[@shen2023; @feng2022; @hoge2022]. This has produced large gains in process-based
hydrology, but the routing step has lagged: there is no widely available,
physically based routing library that emits exact parameter gradients across a
range of routing physics and over realistic network topologies. `dRoute` fills that
gap. Because it provides gradients, parameters can be learned with gradient descent
(e.g. Adam [@kingma2014]) or with quasi-Newton methods at a cost that is largely
independent of the number of parameters, and the routing operator can be embedded as
a layer inside a neural or hybrid model and trained jointly with it.

`dRoute` is aimed at hydrologists and Earth-system modelers who calibrate routing
parameters, and at researchers building differentiable or hybrid (physics + ML)
hydrological models. By offering six routing schemes behind a common interface, two
independent AD backends, network-topology support, and direct compatibility with
SUMMA/`mizuRoute` inputs, it lets users compare routing physics, verify gradients,
and adopt gradient-based calibration without re-implementing the underlying numerics.

# Functionality

- **Six routing methods** behind a common API: lag, impulse-response function (IRF),
  Muskingum–Cunge, soft-gated kinematic-wave tracking, diffusive wave, and full
  Saint-Venant (the last via SUNDIALS [@hindmarsh2005] implicit ODE solvers as a
  high-fidelity benchmark).
- **Exact parameter gradients** through any differentiable scheme, with timeseries
  gradient accumulation so that a single reverse pass differentiates the full
  simulation.
- **Two AD backends** — `CoDiPack` (tape-based) and `Enzyme` (source-to-source) — that
  cross-validate each other and let users trade compile-time for run-time cost.
- **River-network topology** with tributaries and confluences, loadable from
  `mizuRoute` topology files or from GeoJSON/CSV.
- **PyTorch integration**, so routing can be used as a differentiable layer in ML
  training, and **a Basic Model Interface (BMI)** for model coupling.
- **Python packaging** (`pip install droute`) with a compiled C++ core, plus a C++ and
  a Python test suite and continuous-integration wheel builds.

# Example

```python
import droute

# Build (or load) a river network, then route SUMMA runoff with Muskingum–Cunge
config = droute.RouterConfig()
config.dt = 3600.0            # hourly
config.enable_gradients = True
router = droute.MuskingumCungeRouter(network, config)

router.start_recording()
for t in range(n_timesteps):
    for r in range(n_reaches):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    router.record_output(outlet)      # store discharge on the AD tape
router.stop_recording()

# One reverse pass yields dLoss/dparam for every reach
router.compute_gradients_timeseries(outlet, dLoss_dQ)
grads = router.get_gradients()        # e.g. {"reach_0_manning_n": ...}
```

A worked example calibrating reach roughness for the Bow River at Banff (Alberta,
Canada) — comparing derivative-free DDS against gradient-based Adam using `dRoute`
gradients — is provided with the repository.

# Acknowledgements

`dRoute` builds on `CoDiPack` and `Enzyme` for automatic differentiation, `SUNDIALS`
for implicit ODE integration, and `pybind11` for Python bindings, and was developed
alongside the SUMMA and `mizuRoute` hydrological modeling ecosystem.

# References
