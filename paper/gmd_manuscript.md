# dRoute: a differentiable river routing library with six routing schemes and dual automatic-differentiation backends

**Target journal:** Geoscientific Model Development (GMD) — *Model description and evaluation paper*

**Authors:** Darri Eythorsson (University of Calgary), *et al.* (TODO)

---

> **Status: manuscript skeleton.** The model-description section (governing equations
> for all six schemes) is drafted; the introduction, evaluation, and discussion are
> outlined with the experiments to be produced (see `experiments/`). Equations are
> written to match the dRoute implementation in `include/dmc/kernels_enzyme.hpp`.

## Abstract (≈250 words — TODO finalize)

River routing is a standard component of distributed hydrological and Earth-system
models, yet operational routing tools are not differentiable, forcing parameter
calibration onto derivative-free search that scales poorly with parameter dimension.
We present **dRoute**, an open-source C++/Python library that solves river routing
with six schemes spanning the fidelity–cost spectrum — lag, impulse-response (gamma
unit hydrograph), Muskingum–Cunge, soft-gated kinematic-wave tracking, diffusive
wave, and full Saint-Venant — and returns **exact gradients** of simulated discharge
with respect to physical parameters via automatic differentiation (AD). Two
independent AD backends (operator-overloading CoDiPack and compiler-level Enzyme)
cross-validate each other. We verify gradient correctness against finite differences,
evaluate forward accuracy against the Saint-Venant reference and analytical flood-wave
solutions, and demonstrate gradient-based calibration of per-reach roughness for the
Bow River at Banff, showing that gradient descent (Adam) attains comparable or better
performance than Dynamically Dimensioned Search (DDS) in far fewer model evaluations,
with the advantage growing as parameter dimension increases. *(Quantitative results
TODO.)*

---

## 1 Introduction (outline)

- River routing in distributed/continental hydrology; role of mizuRoute, RAPID, SUMMA chains.
- Routing parameters (roughness, celerity, diffusivity, Muskingum K/X) are uncertain and calibrated.
- Conventional calibration = derivative-free (DDS, SCE-UA, PSO); robust but evaluation cost grows steeply with parameter count → forces lumped/regionalized parameters.
- Differentiable modeling in geosciences [Shen et al. 2023; Feng et al. 2022; Höge et al. 2022]: simulators that emit gradients enable gradient-based learning and hybrid physics+ML; routing has lagged.
- **Contribution:** (i) six routing schemes behind a common differentiable API; (ii) two independent AD backends with cross-validation; (iii) network-topology support and SUMMA/mizuRoute compatibility; (iv) demonstration that exact gradients change the calibration cost–dimension scaling.
- Paper roadmap.

## 2 Model description

### 2.1 Notation and common framework

Each reach $i$ has length $L_i$, bed slope $S_{0,i}$, Manning roughness $n_i$, and a
power-law hydraulic geometry relating top width $B$ and depth $h$ to discharge $Q$:

$$ B(Q) = a_w\,Q^{\,b_w}, \qquad h(Q) = a_d\,Q^{\,b_d}. $$

Reaches are connected through junctions forming a directed network; the topological
ordering $\pi$ guarantees that every upstream reach is routed before its downstream
neighbour within a timestep. The inflow to reach $i$ is the sum of upstream reach
outflows plus local lateral inflow $q_i$ (runoff aggregated from contributing HRUs):

$$ I_i(t) = \sum_{j \in \mathcal{U}(i)} Q_j(t) + q_i(t). $$

A bulk flow velocity $V$ is obtained from Manning's equation, and the kinematic-wave
celerity for a wide channel is

$$ c = \tfrac{5}{3}\,V. $$

All schemes share the same `RouterConfig` (timestep $\Delta t$, gradient flag,
substepping, smoothing parameters) and the same network/geometry inputs, so they are
directly interchangeable and comparable.

### 2.2 Lag routing

Pure advective translation by the reach travel time $\tau_i = L_i / c_i$:

$$ Q_i(t) = I_i\!\left(t - \tau_i\right). $$

Implemented as a fractional-delay buffer (linear interpolation between the two
bracketing timesteps), which keeps the operator differentiable in $\tau_i$ (hence in
$V_i$). No attenuation; serves as the cheapest baseline.

### 2.3 Impulse-response function (IRF)

Outflow is the convolution of inflow with a gamma unit hydrograph:

$$ Q_i(t) = \int_0^{\infty} I_i(t-s)\,h_i(s)\,\mathrm{d}s, \qquad
   h_i(s) = \frac{s^{\,k-1} e^{-s/\theta_i}}{\theta_i^{\,k}\,\Gamma(k)}, $$

with shape parameter $k$ (default $2.5$) and scale $\theta_i = \tau_i / k$ so that the
kernel mean equals the travel time $\tau_i$. The kernel is truncated at the time
covering 99 % of its mass and, in the differentiable variant, multiplied by a smooth
sigmoid cutoff $\sigma\!\big((T_\text{cut}-s)\,\beta/\theta_i\big)$ so that the
truncation point does not introduce a discontinuity in the gradient. The discrete
kernel is normalized to conserve mass.

### 2.4 Muskingum–Cunge

Storage routing with coefficients derived from the diffusive-wave analogy (Cunge,
1969). With $K_i = L_i / c_i$ and

$$ X_i = \frac{1}{2}\left(1 - \frac{Q_\text{ref}}{c_i\,B_i\,S_{0,i}\,L_i}\right), $$

the outflow update is

$$ Q_i^{\,t} = C_0\,I_i^{\,t} + C_1\,I_i^{\,t-1} + C_2\,Q_i^{\,t-1} + C_3\,q_i^{\,t}, $$

with, writing $D = 2K_i(1-X_i) + \Delta t$,

$$ C_0 = \frac{\Delta t - 2K_iX_i}{D},\quad
   C_1 = \frac{\Delta t + 2K_iX_i}{D},\quad
   C_2 = \frac{2K_i(1-X_i) - \Delta t}{D},\quad
   C_3 = \frac{2\,\Delta t}{D}. $$

$K_i$ is bounded below ($\geq 0.1\,\Delta t$) and $X_i$ is smooth-clamped to
$[x_\text{lo}, x_\text{hi}]$ to keep the scheme stable and the coefficients
differentiable; optional sub-stepping reduces numerical diffusion. This is the
recommended production scheme and the one used in the calibration experiment.

### 2.5 Soft-gated kinematic-wave tracking (KWT)

A Lagrangian discretization of the kinematic-wave equation

$$ \frac{\partial Q}{\partial t} + c\,\frac{\partial Q}{\partial x} = 0, \qquad
   c = \frac{\mathrm{d}Q}{\mathrm{d}A} = \tfrac{5}{3}V, $$

in which discrete flow parcels are advected downstream a distance $c\,\Delta t$ per
step. Parcel arrival at the reach outlet is normally a non-differentiable event; KWT
replaces the hard arrival test with a **soft sigmoid gate** of tunable steepness, so
that mass leaves a parcel smoothly as it crosses the outlet. An annealing schedule on
the gate steepness recovers the sharp kinematic limit while preserving usable
gradients during optimization. Provides advection-dominated routing with minimal
numerical diffusion in a fully differentiable Lagrangian form.

### 2.6 Diffusive wave

The diffusive-wave (zero-inertia) approximation adds physical attenuation:

$$ \frac{\partial Q}{\partial t} + c\,\frac{\partial Q}{\partial x}
   = D_h\,\frac{\partial^2 Q}{\partial x^2}, \qquad
   D_h = \frac{Q}{2\,B\,S_0}, $$

discretized on $N$ nodes per reach with upwind advection and a diffusion term and
advanced **implicitly** (the diffusion number $\mathrm{Df} = D_h\,\Delta t/\Delta x^2$
appears on the diagonal), giving a tridiagonal system solved by the Thomas algorithm.
Implicit time-stepping makes the scheme unconditionally stable but means gradients
cannot be obtained by naive reverse-mode through the solver; dRoute instead
differentiates the linear solve analytically via the **implicit-function theorem
(IFT)**, back-substituting the adjoint through the same tridiagonal factorization.
This captures flood-wave attenuation at moderate cost.

### 2.7 Saint-Venant (full dynamic wave)

The complete one-dimensional shallow-water equations,

$$ \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_\ell, \qquad
   \frac{\partial Q}{\partial t}
   + \frac{\partial}{\partial x}\!\left(\frac{Q^2}{A}\right)
   + gA\,\frac{\partial h}{\partial x}
   = gA\,(S_0 - S_f), $$

with friction slope from Manning, $S_f = n^2 Q|Q| / (A^2 R^{4/3})$ and hydraulic
radius $R$. The system is integrated with the SUNDIALS CVODES implicit BDF solver as
a high-fidelity reference. It is the most expensive scheme and is used in this paper
chiefly as the benchmark against which the cheaper schemes' forward accuracy is
assessed.

### 2.8 Network assembly

Within each timestep, reaches are evaluated in topological order; each junction sums
the outflows of its upstream reaches to form the inflow of the downstream reach. The
whole-network operator is therefore a composition of per-reach kernels, and because
every kernel is differentiable, the network map $\;\boldsymbol{\theta} \mapsto
\mathbf{Q}_\text{outlet}(t)\;$ is differentiable end to end.

## 3 Automatic differentiation

### 3.1 Two backends

- **CoDiPack** (operator overloading / tape): records the forward evaluation on a tape
  and replays it in reverse. Per-timestep outputs are stored with `record_output`, and
  `compute_gradients_timeseries` performs a single reverse sweep over the whole
  simulation, accumulating $\partial \mathcal{L}/\partial\theta$ for every parameter.
- **Enzyme** (compiler / source-to-source): differentiates the flat-array kernels at
  the LLVM-IR level, producing gradient code with no AD-specific types.

### 3.2 Gradient of the objective

For a loss $\mathcal{L}(\mathbf{Q}_\text{sim}, \mathbf{Q}_\text{obs})$ (e.g. MSE), the
seed $\partial\mathcal{L}/\partial Q^t$ is supplied per timestep and the reverse pass
returns $\partial\mathcal{L}/\partial\theta$ in one sweep, at cost independent of the
number of parameters — the property that underlies the calibration experiment.

### 3.3 Cross-validation of the two backends

The two backends compute identical gradients by different mechanisms; agreement
between them (and with finite differences, §4.1) is the correctness guarantee.

## 4 Evaluation (experiments to produce)

### 4.1 Gradient verification
- AD (both backends) vs. centered finite differences across schemes and parameters; report relative error. *(Basis: `tests/test_gradient_verification.cpp`, `tests/test_ad_backend_comparison.cpp`.)*

### 4.2 Forward accuracy
- Each cheaper scheme vs. Saint-Venant reference and vs. analytical flood-wave attenuation; quantify peak/timing/mass error.

### 4.3 Gradient-based vs. derivative-free calibration — **headline result**
- Domain: semi-distributed Bow River at Banff (29 reaches), hourly SUMMA `averageRoutedRunoff`, gauged discharge.
- Calibrate per-reach Manning's $n$; Adam (dRoute AD) vs. DDS (≥10 seeds).
- Figure 1: best-so-far KGE vs. number of forward model evaluations (Adam line, DDS median + IQR band) on a held-out validation period.
- Figure 2: parameter-count sweep (lumped → grouped → per-reach $n$) — DDS evaluations-to-target rising steeply while Adam stays ~flat.
- *(Script: `experiments/bow_at_banff_dds_vs_adam.py`.)*

### 4.4 Cost and scaling
- Wall-clock and forward-pass counts; AD reverse-pass overhead; CoDiPack vs. Enzyme runtime.

## 5 Discussion (outline)
- When differentiable routing helps (high-dimensional / hybrid ML) vs. when DDS suffices.
- Smoothing/soft-gating trade-offs (bias vs. gradient usability); annealing.
- Limitations: implicit-solver adjoints, non-smooth physics, memory of tape-based AD.

## 6 Conclusions (outline)

## Code and data availability
- Code: dRoute on GitHub + frozen Zenodo DOI (same release as the JOSS submission).
- Data: archive Bow-at-Banff SUMMA runoff, mizuRoute topology, and observed streamflow used in §4.3 (Zenodo).

## References
See `paper.bib` (shared with the JOSS submission); add RAPID, SCE-UA, and any
analytical flood-wave reference for §4.2.
