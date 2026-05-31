# Adding lakes/reservoirs to dRoute — design investigation

**Question:** what would it take to add (differentiable) lake/reservoir routing to dRoute,
using HydroLAKES data obtained via SYMFLUENCE?

**Short answer:** moderate, and it's a *natural* fit — the storage-discharge update of a
lake is a per-node recurrence just like the Muskingum–Cunge outflow update, so the
existing CoDiPack timeseries-AD machinery differentiates it with no new AD
infrastructure. The novel/valuable part is exactly that: **differentiable** lake and
reservoir routing (mizuRoute/RAPID lakes are not differentiable), enabling gradient-based
learning of outflow parameters and reservoir operating rules, and HydroLAKES gives global
coverage.

## 1. Current state

- `include/dmc/network.hpp`: `Reach` (length, slope, `manning_n`, geometry, in/out-flow
  state, gradient storage) and `Junction` (`upstream_reach_ids`); `Network` builds a
  topological order. **No lake/reservoir entity exists.**
- Routing kernels (`include/dmc/kernels_enzyme.hpp`, `router.hpp`) iterate reaches in
  topological order; each reach's outflow is a recurrence in its state + upstream inflow.
- AD: per-timestep outputs are taped (`record_output`) and a single reverse pass
  (`compute_gradients_timeseries`) returns ∂L/∂θ — the same mechanism a lake would use.

## 2. Lake routing physics (storage-discharge), made differentiable

A lake/reservoir is a storage node with mass balance

  dS/dt = I(t) − Q_out(S, t),  discretized (e.g. implicit):
  Sᵗ⁺¹ = Sᵗ + Δt·(Iᵗ⁺¹ − Q_out(Sᵗ⁺¹)),  with Q_out a function of storage.

Methods to offer (mirroring mizuRoute, each differentiable):

1. **Linear reservoir (Döll D03-style)** — `Q_out = S/τ` (τ = residence time from
   HydroLAKES `Res_time`). Trivially smooth; learnable τ.
2. **Nonlinear storage–discharge (power law)** — `Q_out = Q_ref·(S/S_max)^b` (or
   `k·Sᵐ`). Smooth; learnable `b`/`k`, `m`, `Q_ref` (init from `Dis_avg`).
3. **Level-pool / weir** — head `h = S/A` (A from `Lake_area`), `Q_out = C·(h−h_spill)₊^{3/2}`.
   The spill threshold is non-smooth → reuse dRoute's existing **smooth-gate/soft-clamp**
   trick (as in the KWT soft gate and the Muskingum X clamp) for a differentiable surrogate.
4. **Regulated reservoir (Hanasaki H06)** — operating rules: target release from current
   storage, long-term mean inflow, and (optionally) demand, with min/max release clamps.
   Clamps → smooth-clamp for AD; learnable rule parameters. **Phase 2** (more complex).

`Lake_type` from HydroLAKES selects natural lake (1 → methods 1–3) vs reservoir
(2, 3 → method 4 or a parameterized power law).

## 3. Differentiability — why this is cheap

The storage recurrence `Sᵗ⁺¹ = f(Sᵗ, params, Iᵗ⁺¹)` is structurally identical to the
Muskingum–Cunge outflow recurrence already taped by CoDiPack. So:

- Lake outflow parameters (`τ`, `b`, `k`, `C`, `h_spill`, H06 rule coeffs) become `Real`
  (active AD type) fields, registered as inputs in `start_recording()`.
- The reverse pass already accumulates ∂L/∂(those params) over the whole timeseries.
- The only care needed is **smoothness** at thresholds (spillway, min/max release) — use
  the soft-clamp/sigmoid surrogates dRoute already has (smooth_clamp, smooth_max).

→ **No new AD machinery.** Same `record_output` / `compute_gradients_timeseries` path; the
gradient-verification harness (`experiments/gradient_verification.py`) extends directly to
lake parameters (AD vs finite difference).

## 4. Data model + topology changes

- **`Lake` state on the network.** Either (a) a `Reach` with `is_lake=true` + lake fields,
  or (b) a dedicated `Lake` struct keyed by the outlet reach. Fields: storage `S`
  (state, `Real`), surface area `A`, max storage `S_max` (= `Vol_total` or
  `Depth_avg`·`A`), outflow params, `lake_type`. Add gradient-storage fields mirroring
  `Reach`.
- **Routing loop:** in topological order, if a node is a lake, call the lake kernel
  (inflow = Σ upstream outflows + lateral; update `S`; emit `Q_out`) instead of the
  channel kernel. Order is unchanged.
- **Mass conservation:** lakes conserve mass by construction (dS/dt = I − Q_out); verify
  with the same steady-state test used for the Saint-Venant fix (outlet = inflow at
  steady state, storage stationary).
- **Topology I/O:** mizuRoute topology already carries `isLakeInlet`/lake flags; extend
  `topology_nc.hpp`/`network_io.hpp` to read an `isLake` flag + lake parameters, plus a
  lake→segment mapping (analogous to the gauge→segment mapping).

## 5. HydroLAKES data via SYMFLUENCE

- `symfluence data download hydrolakes --config <cfg>` →
  `attributes/lakes/domain_<name>_hydrolakes.gpkg` with `Hylak_id, Lake_type, Lake_area,
  Vol_total, Depth_avg, Dis_avg, Res_time, Elevation` (handler:
  `data/acquisition/handlers/hydrolakes.py`; cloud-direct).
- **Mapping:** spatial-join each lake polygon to the network reach whose outlet lies in /
  nearest the lake (same pattern as `gauge_segment_mapping.csv`), producing a
  lake→segment table.
- **Parameter init from attributes:** `S_init`, `S_max` ← `Vol_total` (mcm→m³);
  `A` ← `Lake_area` (km²→m²); `τ` ← `Res_time` (days→s); `Q_ref` ← `Dis_avg`;
  `h_max` ← `Depth_avg`. These give physically sensible priors; calibration then
  refines the outflow coefficients by gradient descent.

## 6. Implementation steps (and rough effort)

1. `Lake` fields/struct + state in `network.hpp`; gradient-storage fields. *(small)*
2. New `lake_kernels.hpp` (linear, nonlinear, level-pool; CoDiPack + Enzyme variants,
   smooth thresholds). *(moderate — ≈ one new routing scheme)*
3. Wire lakes into the network routing loop + mass-balance update. *(small–moderate)*
4. Topology/IO: read `isLake` + lake params + lake→segment mapping. *(moderate)*
5. pybind11 bindings: `Lake`, set/get storage & params, gradients. *(small)*
6. SYMFLUENCE glue: HydroLAKES gpkg → lake→segment mapping → dRoute lake config. *(moderate)*
7. Tests + gradient verification (FD vs AD for lake params) + a steady-state mass check.
   *(small, reuses existing harness)*
8. *(Phase 2)* Hanasaki H06 regulated-reservoir rules. *(larger)*

Core differentiable level-pool + linear/nonlinear reservoir ≈ the effort of adding one
routing scheme. H06 is a separate, larger increment.

## 7. Why it's worth it (paper angle)

- **Novel:** differentiable lake & reservoir routing — operational lake routers
  (mizuRoute, RAPID) are not differentiable, so outflow/operating-rule parameters are
  calibrated (if at all) by derivative-free search. dRoute could *learn* them by gradient
  descent, and embed regulated reservoirs in hybrid physics–ML models.
- **Global applicability** via HydroLAKES (~1.4M lakes).
- **Synergy with the multi-gauge identifiability result:** lake outflow parameters are
  often *more* identifiable than per-reach roughness (lakes leave a strong, localized
  signature on the hydrograph), so gradient-based calibration should pay off cleanly.

## 8. Open questions / risks

- Implicit vs explicit storage update: implicit (solve for `Sᵗ⁺¹`) is stable for fast
  lakes (small τ) but needs a differentiable solve (Newton with IFT adjoint, as already
  done for the diffusive-wave tridiagonal); explicit is simpler but may need sub-stepping
  for small-residence-time lakes.
- Lakes spanning multiple reaches / lake chains; bifurcations at lake outlets.
- Reservoir operating rules need demand/target data beyond HydroLAKES for H06 (could start
  with `Dis_avg`-based release and a learnable correction).
