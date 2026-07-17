# Choosing an automatic-differentiation architecture for differentiable geoscientific models: a cross-paradigm evaluation on river routing

**Target journal:** Geoscientific Model Development (GMD) — *Model evaluation / methods for model assessment*

**Authors:** Darri Eythorsson (University of Calgary); co-authors TBD

---

> **Status: draft with results.** The routing physics (§2) is implemented identically in
> two independent codebases — **dRoute** (C++/CoDiPack/Enzyme, `include/dmc/`) and **jroute**
> (JAX/diffrax, `src/jroute/`) — which serve as a controlled cross-check throughout the
> evaluation (§5). Cross-paradigm gradient agreement, all-scheme gradient verification, the
> Saint-Venant adjoint-robustness case study, performance/scaling, and engineering-cost
> results (§5.1–5.5) are complete. The calibration-efficiency and identifiability results
> (§5.6) are from dRoute. One real multi-gauge case (Bow at Calgary, §5.6) is in preparation.

## Abstract

Differentiable modeling is reshaping geoscientific calibration and hybrid physics–machine-learning
(ML) modeling, and components such as river routing are increasingly expected to emit gradients of
their outputs with respect to physical parameters. Yet a prior question — *which automatic-
differentiation (AD) architecture* a differentiable geoscientific model should be built on — has not
been evaluated systematically. We use river routing as a representative operator and implement six
routing schemes (lag, impulse-response/gamma unit hydrograph, Muskingum–Cunge, soft-gated
kinematic-wave tracking, diffusive wave, and full Saint-Venant) twice, under three AD paradigms:
operator-overloading **tape** (CoDiPack), source-to-source **LLVM** (Enzyme), and trace-and-compile
**XLA** (JAX/diffrax). The two independent implementations — **dRoute** (C++) and **jroute** (JAX) —
act as a controlled cross-check. We find: (i) the tape and trace paradigms produce **identical**
Muskingum–Cunge gradients to $2\times10^{-14}$ despite sharing no source code, and all six schemes'
gradients match centered finite differences (to $\sim\!10^{-9}$ for the smooth schemes); (ii)
**adjoint robustness diverges sharply for stiff PDEs** — the trace paradigm's implicit-solver adjoint
returns finite, recovery-validated Saint-Venant gradients on a 116-reach regulated network with
reservoirs, where the source-to-source paradigm's hand-rolled CVODES adjoint diverges on transient
forcing; (iii) the paradigms handle method-specific AD hazards very differently (implicit tridiagonal
solves, data-dependent control flow, Lagrangian parcels), with the trace paradigm differentiating
straight through linear solves and using fixed-shape rewrites where the tape paradigm must fall back
to hand-derived or approximate gradients; and (iv) performance and engineering cost trade off cleanly
— the tape is fastest on small CPU problems and header-only to build, while the trace paradigm
compiles the solve ($\approx\!14\times$ speed-up), admits a matrix-free $O(N)$ linear solver, batches
over parameter ensembles, and runs on GPU, at the cost of static-shape rewrites and a heavier
compiler toolchain. We distill the results into a decision framework for choosing an AD architecture
for differentiable routing, and argue the lessons transfer to other geoscientific operators.

---

## 1 Introduction

River routing is a standard component of distributed hydrological and Earth-system models
(mizuRoute [Mizukami et al., 2016], RAPID [David et al., 2011], and the SUMMA [Clark et al., 2015] /
mizuRoute chain), and its parameters — roughness, celerity, diffusivity, Muskingum $K/X$ — are
uncertain and routinely calibrated. Conventional calibration is derivative-free (DDS
[Tolson & Shoemaker, 2007], SCE-UA, PSO): robust, but its evaluation cost grows steeply with parameter
count, forcing lumped or regionalized parameterizations. Differentiable modeling in the geosciences
[Shen et al., 2023; Feng et al., 2022; Höge et al., 2022] addresses this by making the simulator
return gradients, enabling gradient-based learning and hybrid physics–ML; routing has begun to follow.

The community discussion so far has centered on *whether* a model is differentiable. We argue that a
prior, under-examined question is *how* — on which AD architecture the differentiable model is built —
and that this choice has first-order consequences for correctness, robustness, performance, and
maintainability that a model builder must weigh before committing. AD architectures differ along at
least three axes: the **paradigm** (operator-overloading tape, source-to-source compilation, or
trace-and-compile), the **runtime** (native CPU vs vectorized XLA/GPU), and how each copes with the
**operator's structure** (implicit solvers, adaptive/data-dependent control flow, stiffness). These
interact: a paradigm that is trivial for an explicit update can be fragile for a stiff implicit PDE.

To evaluate the choice concretely rather than in the abstract, we implement the *same* six routing
schemes twice — once in C++ with the CoDiPack tape and the Enzyme source-to-source backends
(**dRoute**), once in JAX with XLA tracing and the diffrax implicit-adjoint machinery (**jroute**).
Because the two share no code but target the same physics, agreement between them is simultaneously a
correctness guarantee and the comparison itself, and *disagreement* localizes an architectural
strength or weakness.

**Contributions.** (i) The first cross-paradigm, cross-implementation evaluation of AD architectures
for a geoscientific operator, spanning six routing schemes and three AD paradigms. (ii) A machine-
precision mutual validation between two independent differentiable implementations. (iii) A stiff-PDE
adjoint-robustness result that separates the paradigms decisively. (iv) A performance/scaling and
engineering-cost characterization (tape memory, compilation, matrix-free scaling, batching, GPU
readiness, build friction). (v) A decision framework for choosing an AD architecture for
differentiable routing, with lessons that transfer to other differentiable geoscientific models.

## 2 Routing schemes (shared physics)

Both implementations solve the identical physics; we summarize it here and note per-scheme AD hazards
that §3 exploits.

### 2.1 Notation and common framework

Each reach $i$ has length $L_i$, bed slope $S_{0,i}$, Manning roughness $n_i$, and a power-law
hydraulic geometry relating top width $B$ and depth $h$ to discharge $Q$:

$$ B(Q) = a_w\,Q^{\,b_w}, \qquad h(Q) = a_d\,Q^{\,b_d}. $$

Reaches connect through junctions forming a directed network; a topological ordering $\pi$ guarantees
every upstream reach is routed before its downstream neighbour within a timestep. Inflow to reach $i$
is the sum of upstream outflows plus local lateral inflow $q_i$ (runoff aggregated from HRUs):

$$ I_i(t) = \sum_{j \in \mathcal{U}(i)} Q_j(t) + q_i(t). $$

A bulk velocity $V$ follows from Manning's equation and the wide-channel kinematic celerity is
$c = \tfrac{5}{3}V$. All schemes share the same timestep $\Delta t$, geometry, and network inputs and
are directly interchangeable. Lake/reservoir reaches route via a differentiable storage–discharge
rating $Q(S)$ with mass balance $\mathrm{d}S/\mathrm{d}t = I - Q(S)$ instead of the channel kernel;
the rating parameters (reference release, exponent, minimum release, spill) are learnable.

### 2.2 Lag routing

Pure advective translation by travel time $\tau_i = L_i/c_i$: $Q_i(t) = I_i(t-\tau_i)$, implemented as
a fractional-delay buffer (linear interpolation between bracketing timesteps). Cheapest baseline; no
attenuation. *AD hazard:* an integer time-lag makes the map piecewise-constant in $\tau_i$, so the
parameter gradient is quantized (§3).

### 2.3 Impulse-response function (IRF)

Outflow is inflow convolved with a gamma unit hydrograph
$h_i(s) = s^{\,k-1}e^{-s/\theta_i}/(\theta_i^{\,k}\Gamma(k))$, shape $k$ (default 2.5), scale
$\theta_i=\tau_i/k$ (kernel mean equal to $\tau_i$). The kernel is normalized (mass conservation).
*AD hazard:* the kernel length depends on the (parameter-dependent) travel time — a data-dependent
loop bound (§3).

### 2.4 Muskingum–Cunge

Storage routing with diffusive-analogy coefficients (Cunge, 1969). With $K_i=L_i/c_i$ and
$X_i = \tfrac12\!\left(1 - Q_\text{ref}/(c_iB_iS_{0,i}L_i)\right)$, and $D = 2K_i(1-X_i)+\Delta t$,

$$ Q_i^{\,t} = C_0 I_i^{\,t} + C_1 I_i^{\,t-1} + C_2 Q_i^{\,t-1} + C_3 q_i^{\,t}, $$

with $C_0=(\Delta t-2K_iX_i)/D$, $C_1=(\Delta t+2K_iX_i)/D$, $C_2=(2K_i(1-X_i)-\Delta t)/D$,
$C_3=2\Delta t/D$. $X_i$ is clamped to $[x_\text{lo},x_\text{hi}]$ and fixed sub-stepping stabilizes
short reaches. The recommended production scheme. *AD hazard:* the clamp is the load-bearing non-
smoothness; the two implementations make subtly different smoothing choices (§3, §5.1).

### 2.5 Soft-gated kinematic-wave tracking (KWT)

A Lagrangian discretization of $\partial_t Q + c\,\partial_x Q = 0$ in which flow parcels advect
$c\,\Delta t$ per step; the non-differentiable outlet-arrival event is replaced by a soft sigmoid gate
of tunable steepness. *AD hazard:* parcels are created and destroyed at runtime — a variable-length
data structure (§3). We also note (§5.3) that the celerity here depends on channel geometry, not on
Manning's $n$.

### 2.6 Diffusive wave

The zero-inertia approximation $\partial_t Q + c\,\partial_x Q = D_h\,\partial_{xx} Q$,
$D_h = Q/(2BS_0)$, discretized on $N$ nodes with an implicit (tridiagonal) update solved by the Thomas
algorithm. *AD hazard:* differentiating an implicit linear solve — a case where the paradigms diverge
most (hand-rolled implicit-function-theorem adjoint vs autodiff straight through the solve; §3, §5.3).

### 2.7 Saint-Venant (full dynamic wave)

The 1-D shallow-water equations,

$$ \partial_t A + \partial_x Q = q_\ell, \qquad
   \partial_t Q + \partial_x(Q^2/A) + gA\,\partial_x h = gA(S_0 - S_f), $$

with Manning friction slope $S_f = n^2 Q|Q|/(A^2R^{4/3})$, discretized method-of-lines (finite volume,
Rusanov flux) into a stiff ODE system. *AD hazard:* a stiff implicit integration whose adjoint is the
single hardest case in this study, and the one that separates the paradigms (§5.2).

### 2.8 Network assembly

Within a timestep, reaches are evaluated in topological order and junctions sum upstream outflows. The
whole-network operator is a composition of per-reach kernels; because every kernel is differentiable,
the map $\boldsymbol\theta \mapsto \mathbf{Q}_\text{outlet}(t)$ is differentiable end to end. In the
vectorized (jroute) implementation the topological order is realized as *level* generations so that a
whole level routes as one batched kernel while preserving the current-timestep upstream coupling.

## 3 Automatic-differentiation architectures

### 3.1 The three paradigms

- **Operator-overloading tape (CoDiPack [Sagebaum et al., 2019], in dRoute).** A custom `Real` type records every arithmetic
  operation on a tape during the forward pass; a single reverse sweep replays the tape to accumulate
  $\partial\mathcal{L}/\partial\theta$ at cost independent of the number of parameters. Header-only;
  differentiates arbitrary C++ control flow. Memory grows with the length of the recorded computation.
- **Source-to-source compilation (Enzyme [Moses & Churavy, 2020], in dRoute).** An LLVM plugin differentiates the compiled IR
  of flat-array kernels, emitting derivative code with no AD-specific types and, in principle, low
  overhead. Requires a matching Clang/LLVM + Enzyme toolchain; the AD transform can fail to compile or
  mis-handle constructs the analysis cannot type (§5.5).
- **Trace-and-compile (JAX/XLA [Bradbury et al., 2018] + diffrax [Kidger, 2021], in jroute).** The forward function is traced to an XLA graph
  and JIT-compiled; reverse-mode differentiation and implicit-solver adjoints are provided by the
  framework. Requires static shapes and functional (side-effect-free) code, but yields compilation,
  vectorization (`vmap`), device portability (CPU/GPU/TPU), and framework-level adjoints for implicit
  solvers.

For a loss $\mathcal{L}(\mathbf{Q}_\text{sim},\mathbf{Q}_\text{obs})$, all three return
$\partial\mathcal{L}/\partial\theta$ in one reverse pass at cost independent of parameter dimension —
the property that underlies the calibration efficiency of §5.6. The differences are in *which
operators each can differentiate, how robustly, how fast, and at what engineering cost*.

### 3.2 Method-specific AD hazards and how each paradigm meets them

| scheme | AD hazard | tape (CoDiPack) | source-to-source (Enzyme) | trace (JAX) |
|---|---|---|---|---|
| Lag | integer time-lag (quantized) | analytic approximation | — | fractional-lag buffer; gradient still quantized |
| IRF | data-dependent kernel length | soft-masked fixed max length | — | fixed max length + smooth mask (full AD) |
| MC | clamp non-smoothness | smooth clamp on tape | flat-array kernel | smooth clamp (full AD) |
| KWT | variable-length parcels | analytic approximation + attenuation heuristic | — | fixed-capacity parcel buffer (full AD) |
| DW | implicit tridiagonal solve | hand-rolled IFT adjoint + attenuation heuristic | — | autodiff straight through the linear solve |
| SV | stiff implicit integration | (not used) | CVODES continuous adjoint (fragile) | diffrax implicit adjoint (robust) |

Two hazards are worth highlighting because they most cleanly separate the paradigms. **Data-dependent
control flow** (IRF kernel length, KWT parcel count, adaptive sub-stepping) breaks a tape or naive
reverse-mode, which is why the tape implementation falls back to hand-derived or approximate gradients
for KWT and the diffusive wave; the trace implementation instead uses *static-shape* rewrites (a fixed
maximum kernel length with a smooth mask; a fixed-capacity parcel buffer with soft gates) that recover
full AD. **Implicit solves** (DW, SV) cannot be differentiated by replaying the solver; the tape
implementation supplies a hand-rolled implicit-function-theorem adjoint for the diffusive wave, while
the trace implementation differentiates straight through the framework's linear solve (an exact
vector–Jacobian product) and, for Saint-Venant, uses the framework's implicit-ODE adjoint (§5.2).

## 4 Two implementations as a controlled experiment

**dRoute** (C++/pybind11) implements the six schemes as flat-array and object kernels differentiated by
CoDiPack, with an Enzyme source-to-source path for Muskingum–Cunge and a CVODES-based Saint-Venant.
**jroute** (JAX) implements the same schemes as pure functions over pytrees: the network is compiled
to topological levels so a whole level routes as one `vmap`-ed kernel; the Saint-Venant scheme is a
method-of-lines system integrated by a stiff diffrax solver with a framework adjoint. The two are
driven by the *same* mizuRoute topology, SUMMA runoff forcing, and gauge observations, so every
comparison in §5 is apples-to-apples. Neither wraps the other; jroute's core runs with no C++
compiler present, and dRoute runs with no Python framework, which is what makes their agreement a
non-trivial check.

## 5 Evaluation

### 5.1 Cross-paradigm gradient agreement and verification

**Two independent paradigms, identical gradients.** For Muskingum–Cunge on the Bow-at-Banff network
(29 reaches), the tape (CoDiPack) and trace (JAX) implementations — sharing no code and using entirely
different AD mechanisms — produce the gradient of a mean-squared-error calibration loss with respect
to per-reach Manning's $n$ that agrees to a maximum relative difference of $2\times10^{-14}$, i.e.
machine precision. This is a strong mutual-validation result: two independent differentiable models of
the same physics converge on the same derivative.

**Every scheme against finite differences.** Each scheme's trace-mode gradient is verified against a
centered finite difference on a transient-forced chain (Table 1). The smooth schemes match to
$\sim\!10^{-9}$; the soft-gated KWT (with respect to a geometry parameter, §5.3) to $4\times10^{-8}$;
Saint-Venant to $7\times10^{-3}$, which is finite-difference-limited — its adaptive solver makes the
forward map mildly non-smooth, so FD is a poor oracle for it and its rigorous validation is parameter
recovery (§5.2), not FD. Verification of this kind is not ceremonial: it surfaced and fixed real
defects in both implementations (an unbounded AD-tape accumulation and Saint-Venant mass-balance and
gradient-sign errors in dRoute; a diffusive-wave outlet boundary-condition mass leak in jroute; and a
latent Enzyme miscompilation in dRoute, §5.5).

**Table 1.** Trace-mode (jroute) gradient vs centered finite difference, per scheme (4-reach chain,
transient forcing). Lag is omitted (quantized gradient).

| scheme | differentiated parameter | max relative error (AD vs FD) |
|---|---|---|
| Muskingum–Cunge | Manning $n$ | $1.0\times10^{-9}$ |
| IRF (gamma UH) | Manning $n$ | $7.9\times10^{-10}$ |
| Diffusive wave | Manning $n$ | $6.8\times10^{-10}$ |
| KWT (soft-gated) | width coefficient | $4.5\times10^{-8}$ |
| Saint-Venant | Manning $n$ | $7.3\times10^{-3}$ (FD-limited) |

**Reference validation against the operational tool.** Internal gradient checks establish that the
adjoint matches the forward each implementation defines; a complementary check is whether that
*forward* reproduces an established operational implementation. mizuRoute [Mizukami et al., 2016] —
the standard community routing tool — has native IRF (impulse-response unit hydrograph) and KWT
(kinematic-wave tracking) methods that correspond directly to two of the schemes here, so it is a
natural reference. We run mizuRoute in both modes (`route_opt` 1 and 2) on the semi-distributed
Bow-at-Banff network and, for each, feed jroute's matching scheme the identical basin-routed input
that mizuRoute network-routes (mizuRoute's `dlayRunoff`), comparing outlet hydrographs over a full
year (Table 2a). jroute reproduces the operational mizuRoute almost exactly: for **IRF**, Kling–Gupta
efficiency 0.999, Nash–Sutcliffe 1.000, correlation 1.000, volume and peak ratios 1.00, zero timing
lag; for **KWT**, KGE 0.972, NSE 0.999, correlation 1.000 (identical timing and shape), with a $\sim2\%$
lower volume and peak. That small KWT volume deficit is not an implementation error but the expected
signature of the soft-gate smoothing that makes the otherwise non-differentiable outlet-arrival event
differentiable (§2.5, §6) — the price of differentiability, quantified against the operational
reference. The near-exact agreement confirms that making these schemes differentiable did not alter
their physics, and — because dRoute and jroute share the same forward physics (§4) — validates the
shared implementation.

**Table 2a.** Reference validation: jroute vs the operational mizuRoute, network-routed at the
Bow-at-Banff outlet (2005; each scheme and mizuRoute fed the identical basin-routed `dlayRunoff`).

| scheme | KGE | NSE | corr | volume ratio | peak ratio | timing lag |
|---|---|---|---|---|---|---|
| IRF (vs mizuRoute IRF-UH) | 0.999 | 1.000 | 1.000 | 1.000 | 1.002 | 0 h |
| KWT (vs mizuRoute kinematic-wave tracking) | 0.972 | 0.999 | 1.000 | 0.981 | 0.981 | 0 h |

**Where the paradigms legitimately differ.** The Muskingum–Cunge gradient from the Enzyme
source-to-source path is itself *exact* — it matches a finite difference of its own forward to
$7\times10^{-7}$, including the gauge reach — but differs from the tape/trace gradient by $\approx7\%$.
The cause is not an AD error but a modeling choice: the Enzyme flat-array kernel is fully smoothed
(smooth maxima on the reference-discharge, celerity, and outflow floors; $\text{min\_flow}=10^{-10}$),
whereas the tape kernel uses hard floors and $\text{min\_flow}=10^{-6}$. Each paradigm's adjoint is
correct for the (slightly different) forward it differentiates — a reminder that AD faithfully returns
the gradient of *whatever* forward model it is given, so cross-paradigm agreement also depends on
matching the forward smoothing, not only the AD.

### 5.2 Adjoint robustness for stiff PDEs — the Saint-Venant case (headline)

The Saint-Venant scheme is the sternest test because its stiff implicit integration makes the adjoint
hard. Here the paradigms separate decisively.

**Trace paradigm (jroute).** The method-of-lines system is integrated by a stiff, L-stable diffrax
solver whose implicit-function-theorem adjoint is provided by the framework; the reverse pass is
therefore exact by construction (up to solver tolerance). It returns finite, correct gradients: on a
multi-reach chain the gradient matches finite difference to $5\times10^{-6}$ (smooth forcing); with
lake/reservoir reaches added, the lake-rating parameter gradient matches FD to $3\times10^{-5}$ and is
correctly localized to the lake reach; and on the full **116-reach Bow-at-Calgary network** — a
regulated, multi-gauge domain with reservoirs — the coupled solve runs to completion and yields a
finite gradient (39 of 116 reaches with nonzero sensitivity, exactly those draining to the outlet).
Because the forward map is mildly non-smooth (adaptive steps, area/flux clamps), FD is not a valid
oracle at scale; we validate instead by parameter recovery — gradient descent driven by the adjoint
recovers a known Manning field — the appropriate test for a non-smooth solver.

**Source-to-source paradigm (dRoute).** The Saint-Venant adjoint is a hand-rolled CVODES adjoint-
sensitivity (ASA) backward integration combined with forward-mode Enzyme Jacobians. Its forward solve
works, but on the *same* transient regulated network the backward (adjoint) integration diverges — the
CVODES corrector fails at $t=0$ and the backward solution overflows to non-finite values — across a
sweep of tolerances, initial conditions, and interpolation/Jacobian options. (In earlier work the
same adjoint was made to pass narrow, constant-forcing recovery tests; the divergence here on the
transient 116-reach case is the honest robustness statement.)

The architectural lesson is specific: for a stiff PDE, a **framework-provided implicit-solver adjoint**
(here, trace + diffrax) is markedly more robust than a **hand-rolled continuous adjoint over a
black-box stiff integrator** (source-to-source + CVODES-ASA), because the former differentiates the
same discrete steps the forward solver actually took, while the latter must reconstruct a consistent
backward trajectory — the step at which the hand-rolled adjoint fails. This is the clearest single
result separating the paradigms, and it is exactly the regime (high-fidelity dynamic-wave routing on
regulated networks) where differentiable routing is most wanted.

### 5.3 Handling method-specific hazards

**Implicit diffusive wave.** Both implementations are mass-conserving and give correct gradients, but
by different routes. dRoute differentiates the tridiagonal solve with a *hand-rolled* implicit-function
-theorem adjoint (back-substituting through the same factorization) plus a $0.85$ upstream-attenuation
heuristic for the cross-reach term, and it drops the data-dependent sub-step loop off the tape. jroute
instead differentiates straight through the batched linear solve — the framework supplies the exact
vector–Jacobian product of the solve — with fixed sub-steps, so no manual adjoint or attenuation
heuristic is needed and the data-dependent-loop problem disappears. jroute's diffusive-wave gradient
matches FD to $7\times10^{-10}$ (Table 1). The trace paradigm turns a bespoke, heuristic hand-derivation
into an automatic exact gradient.

**Variable-length parcels (KWT).** dRoute's KWT uses a runtime-variable parcel list and an *approximate*
gradient (a constant-celerity chain rule plus the $0.85$ attenuation), because a variable-length data
structure cannot be taped exactly. jroute uses a fixed-capacity circular parcel buffer with soft gates,
recovering full AD; its geometry-parameter gradient matches FD to $4\times10^{-8}$. In the course of
this comparison we also found that dRoute's KWT celerity reads Manning's $n$ but never uses it (celerity
is set by channel geometry), so its analytic *Manning* gradient is a fiction; jroute correctly returns
zero sensitivity to Manning for KWT and an exact gradient for the geometry parameters that actually
drive the celerity. Cross-implementation comparison thus caught a latent correctness issue that neither
implementation's internal checks had flagged.

**Lakes/reservoirs.** Both implementations route lake reaches by a differentiable storage–discharge
rating with mass balance; in jroute this is a shared component usable by all six schemes (exact
topological coupling for the level-scan schemes, a lagged coupling for the Lagrangian/lag schemes),
and the lake-rating parameters carry correct gradients (§5.2).

### 5.4 Performance and scaling

**Small CPU problems: the tape wins.** On the 29-reach Bow-at-Banff Muskingum–Cunge problem, forward
wall-clock is comparable across paradigms, but forward-plus-gradient is $\approx\!8\times$ faster for
the CoDiPack tape than for the JAX reverse pass — reverse-mode through a `lax.scan` over a small
network on CPU carries overhead the tape does not. On problems of this size and shape the tape is the
efficient choice.

**Compilation, scaling, batching, GPU: the trace paradigm wins.** The trace paradigm's advantages
appear at scale and off-CPU. (i) *Compilation*: JIT-compiling the Saint-Venant solve gives a
$\approx\!14\times$ speed-up over the eager solve (and is numerically identical to $10^{-13}$). (ii)
*Solver scaling*: swapping the dense $O(N^3)$ LU Newton solve for a matrix-free Krylov (GMRES) solver
that scales $\approx\!O(N)$ is $\approx\!4\times$ faster at 480 state dimensions and widening; on the
116-reach Bow-at-Calgary network the forward drops from $\approx\!3.2$ to $\approx\!0.28$ s/step (with
gradients matching the dense solver to $2\times10^{-6}$). (iii) *Batching*: a single `vmap` routes an
entire parameter ensemble in one compiled kernel — free Monte-Carlo / ensemble-Kalman forward — which
the sequential C++ router cannot express. (iv) *GPU readiness*: the compiled solve targets a CUDA
device automatically; GPU benchmarking is in preparation (the results here are CPU).

**Memory.** The tape's memory grows with the length of the recorded computation: the Bow calibration
must reset the CoDiPack tape each epoch and still OOM-terminates around 140 epochs without care. The
trace paradigm's reverse pass uses checkpointed adjoints with flat, tunable memory, and recomputes a
fresh graph each step, so no accumulation occurs — a robustness advantage for long simulations and
long training runs.

### 5.5 Engineering cost / build friction

The paradigms differ sharply in engineering cost, which is a real selection criterion for a model that
others must build and maintain.

- **Tape (CoDiPack):** header-only; compiles with any C++17 compiler; no external AD toolchain. Lowest
  friction.
- **Trace (JAX):** `pip install`; no compiler required for the pure-JAX core; runs on CPU/GPU with the
  same code. Low friction, at the cost of framework dependency and static-shape discipline.
- **Source-to-source (Enzyme):** highest friction. It requires a version-matched Clang/LLVM plus the
  Enzyme plugin (here `ClangEnzyme-21` against LLVM 21), and on macOS a code-signing step to survive
  library-validation; the build silently falls back to non-Enzyme if the plugin is not found. In this
  study the installed extension had in fact been built *without* Enzyme, so the source-to-source
  gradients were unavailable until a from-scratch Enzyme rebuild (which additionally needed a CMake
  policy override for a bundled dependency). During that rebuild we found that the multi-timestep
  Muskingum–Cunge Enzyme gradient had **never compiled**: the differentiated simulation used a
  runtime-sized `memcpy` whose element type Enzyme could not deduce. Replacing it with a typed copy
  fixed the compilation; the resulting adjoint is exact (§5.1). This fix, and the exposure of the
  previously-unbound multi-timestep MC adjoint, have been contributed upstream.

The pattern is that source-to-source offers the lowest theoretical overhead but the highest and most
brittle toolchain cost, while the tape and trace paradigms are far cheaper to build; the trace paradigm
uniquely decouples the differentiable science from any compiler toolchain.

### 5.6 Why it matters — calibration efficiency and identifiability (supporting)

The architecture question matters because differentiable routing pays off in calibration and hybrid
modeling. Using dRoute we confirm the practical value and its limits. On the semi-distributed
Bow-at-Banff network (29 reaches) driven by pre-calibrated SUMMA runoff, gradient-based calibration
(Adam) reaches the optimum (KGE 0.73 cal / 0.82 val) in $\approx\!50$ forward-pass-equivalent
evaluations, versus $\approx\!250$ for DDS (median of 10 seeds) — a $\sim\!5\times$ efficiency gain.
On an idealized multi-gauge chain with an identifiable heterogeneous roughness field, calibration
skill rises monotonically with parameter dimension toward the true-parameter ceiling (multi-gauge
validation KGE $0.957\to0.996$ from 1 to 50 parameters). But a single outlet gauge leaves per-reach
roughness equifinal (Bow-at-Banff validation KGE flat, $0.824\to0.827$ from 1 to 29 parameters):
differentiable routing's high-dimensional advantage is realized only when observations — multiple
gauges, remote sensing, or ML coupling — constrain the parameter field. Forward-accuracy checks
against a full Saint-Venant reference recover the expected fidelity ordering (diffusive $\approx$
Muskingum–Cunge $\gg$ KWT $>$ lag/IRF; Table 2). A purpose-built real multi-gauge case (Bow at
Calgary, 116 reaches, nested WSC gauges, reservoirs) is in preparation and is the natural setting for
the trace paradigm's Saint-Venant + reservoir capability (§5.2).

**Table 2.** Forward accuracy of the cheaper schemes against the full Saint-Venant reference (40-reach,
800-km chain).

| scheme | KGE vs SV | peak ratio | timing lag | volume ratio |
|---|---|---|---|---|
| Diffusive wave | 0.99 | 0.99 | 7 h | 1.01 |
| Muskingum–Cunge | 0.94 | 0.94 | 31 h | 1.03 |
| KWT (soft-gated) | 0.64 | 1.12 | −93 h | 0.88 |
| IRF (gamma UH) | 0.39 | 0.66 | 196 h | 1.00 |
| Lag | 0.37 | 0.71 | 205 h | 1.00 |

## 6 Discussion — a decision framework

The evaluation supports concrete guidance for choosing an AD architecture for a differentiable
geoscientific model, generalizing beyond routing.

- **Explicit updates on small/medium CPU problems, minimal build:** the **operator-overloading tape**
  is the pragmatic default — header-only, differentiates arbitrary control flow, and is fastest here.
  Watch tape memory on long simulations.
- **Stiff implicit PDEs, high fidelity, regulated systems, or where GPU/batching matters:** the
  **trace-and-compile** paradigm is preferable — framework implicit-solver adjoints are far more robust
  for stiff systems, static-shape rewrites recover full AD through data-dependent structures, and
  compilation, matrix-free scaling, `vmap` batching, and GPU come essentially for free. The cost is a
  functional/static-shape rewrite and a framework dependency.
- **Source-to-source** offers the lowest theoretical overhead but the highest toolchain fragility; it
  is attractive when an existing performant C/C++ kernel must be differentiated in place and the build
  environment is controlled, but its adjoints for stiff implicit solvers are hard to get robust, and
  the toolchain is the dominant practical risk.
- **Cross-implementation checking is worth its cost.** Two independent differentiable implementations
  agreeing to machine precision is a strong correctness guarantee, and — as the KWT Manning-gradient
  and the un-compiled Enzyme adjoint show — *disagreement* is a sensitive detector of latent defects
  that single-implementation verification misses. AD faithfully differentiates whatever forward it is
  given, so matching *forward* smoothing choices is as important as matching the AD when comparing.

**Smoothing trade-offs.** Differentiability for the otherwise non-smooth schemes is bought with smooth
surrogates (soft gates, smooth clamps); these keep gradients usable but introduce small biases (e.g.
KWT's volume leak, Table 2), and the two implementations' differing smoothing choices explain their
$\sim\!7\%$ Muskingum–Cunge gradient difference (§5.1). Annealing the gate steepness recovers the sharp
limit.

## 7 Conclusions

We evaluated automatic-differentiation architectures for differentiable river routing by implementing
six routing schemes twice — under operator-overloading tape, source-to-source, and trace-and-compile
paradigms — and comparing them head to head. The two independent implementations agree to machine
precision where they should ($2\times10^{-14}$ for Muskingum–Cunge), which validates both; where they
differ, the difference is diagnostic. The decisive architectural result is adjoint robustness for stiff
PDEs: the trace paradigm's framework implicit-solver adjoint yields finite, recovery-validated
Saint-Venant gradients on a 116-reach regulated network with reservoirs, where the source-to-source
paradigm's hand-rolled CVODES adjoint diverges. The tape paradigm is fastest and cheapest to build on
small CPU problems; the trace paradigm compiles, scales with a matrix-free solver, batches over
ensembles, and runs on GPU. We distilled these into a decision framework, and expect the lessons —
framework adjoints for stiff implicit solvers, static-shape rewrites for data-dependent structures,
cross-implementation checking, and matching forward smoothing when comparing AD paths — to transfer to
other differentiable geoscientific operators.

## Code and data availability

Two implementations: **dRoute** (C++/Python) at `github.com/symfluence-org/dRoute` and **jroute** (JAX)
at `github.com/DarriEy/jroute`, each at a frozen release with a Zenodo DOI. The comparison harness,
gradient-verification suite, and figure-generation scripts are in jroute `src/jroute/compare/`; dRoute
experiment scripts are in `experiments/`. Idealized-network experiments are fully synthetic and
reproducible from the scripts; the Bow SUMMA runoff, mizuRoute topology, and WSC observed streamflow
will be archived on Zenodo. Saint-Venant in dRoute requires a SUNDIALS-enabled build and, for the
source-to-source path, an Enzyme-enabled build (see repo notes); the jroute Saint-Venant requires only
`pip install`.

## References

All citations are in `paper.bib` (shared with the JOSS submission): differentiable modeling
[shen2023, feng2022, hoge2022]; routing/land models [mizukami2016, david2011, clark2015]; AD backends
[sagebaum2019 (CoDiPack), moses2020 (Enzyme), bradbury2018 (JAX), kidger2021 (diffrax)]; solvers and
optimizers [hindmarsh2005 (SUNDIALS/CVODES), tolson2007 (DDS), kingma2014 (Adam)]; routing theory and
metrics [cunge1969, gupta2009 (KGE), nash1970 (NSE)]. (Add SCE-UA/PSO references if those baselines are
retained in the final version.)

## Figure/script manifest (reproducibility)

| Figure (in text) | Script | Output |
|---|---|---|
| F1 gradient verification (all schemes, AD vs FD) | `jroute: python -m jroute.compare.report` | `paper/figures/f1_gradcheck.png` |
| F2 Muskingum–Cunge backend wall-clock (tape/source-to-source/trace) | `jroute: python -m jroute.compare.report` | `paper/figures/f2_mc_backend_timing.png` |
| F3 Saint-Venant adjoint robustness (trace works vs source-to-source diverges) | `jroute: python -m jroute.compare.report` | `paper/figures/f3_sv_adjoint_robustness.png` |
| Table 2a mizuRoute reference validation (IRF + KWT) | `jroute: experiments/mizuroute_validation.py --method {irf,kwt}` (mizuRoute `route_opt` 1/2) | metrics (stdout) |
| catalogue (6-scheme forward accuracy) | `dRoute: experiments/method_catalogue.py` | `results_catalogue/catalogue_hydrographs.png` |
| Bow-at-Banff convergence + scaling | `dRoute: experiments/bow_at_banff_dds_vs_adam.py` | `results/fig1_convergence.png`, `results/fig2_scaling.png` |
