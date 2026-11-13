# Transfers, bifurcations and lake/reservoir nodes

dRoute routes a river tree (each reach drains to one downstream reach). A national
hydrofabric such as Landslíkan also has two layers that one `downSegId` cannot express:

| Layer | File | What dRoute does |
|---|---|---|
| Transfers | `transfers.csv` | Directed extra edges that move water from one reach's outflow (or lake storage) to another reach's inflow |
| Storage nodes | `lakes.csv` | Level-pool lake/reservoir node at each lake outlet reach |

Supported routers: `MuskingumCungeRouter` (production) and `IRFRouter`. If a network
has transfers or level-pool lakes, every other router raises
`RuntimeError: <Router> is not supported with transfers or level-pool lake nodes`.
This includes the Enzyme routers.

## Quick start

```python
import droute
from droute.network_adapter import DRouteNetworkAdapter

adapter = DRouteNetworkAdapter()
# reads topology.nc, transfers.csv and lakes.csv from one directory
net, cfg, summary = adapter.load_hydrofabric(
    "data/build/v0.3_50m/droute",
    sim_start_year=1995.0,          # switches edges on/off by commissioned/decommissioned
    enable_groundwater=False,       # groundwater edges are hypotheses: off by default
)

router = droute.MuskingumCungeRouter(net, droute.RouterConfig())
for t in range(n_steps):
    for rid in net.topological_order():
        router.set_lateral_inflow(rid, runoff_m3s[t, rid])   # reach id == segment index
    router.route_timestep()

print(router.mass_balance().as_dict())
```

`adapter.apply_hydrofabric(net, cfg, transfers_path=..., lakes_path=...)` does the same
for a network you have already built.

## Processing order

`Network::build_topology()` sorts reaches topologically on the **combined graph**:
the tree edges plus one edge `from_reach → to_reach` per transfer. Groundwater edges are
included as well, even though they are lagged. If the combined graph has a cycle,
`build_topology()` (and so every router constructor) throws a `RuntimeError` that
names the reaches and the transfers on the cycle. A transfer that moves water
upstream (`topology_effect = reverse`) creates such a cycle when its target is
upstream of its source in the tree, so it is rejected.

Each timestep:

1. Clear the per-step transfer fluxes and release the groundwater stores.
2. For each reach, in order:
   - inflow = Σ upstream `(outflow − transfer_out)` + `transfer_in`;
   - outflow = channel routing, or the level-pool node for lake outlets;
   - apply the reach's outgoing transfers, which sets `transfer_out` on this reach
     and adds to `transfer_in` on the target.

`outflow` / `get_discharge()` is the gross outflow of a reach, which is what a gauge
upstream of the offtake measures. `net_outflow` / `get_net_discharge()` is what
continues downstream.

## Edge rules

Each source reach applies its edges in file order. Every edge takes
`min(demand, what is left of the outflow)`.

| `type` | demand | parameters |
|---|---|---|
| `headrace`, `diversion` | `min(fraction·Q, capacity, rule(t))` | `fraction`, `capacity` |
| `natural_bifurcation` | `fraction·Q` (constant share) | `fraction` |
| `groundwater` | `fraction·Q` goes into a linear store `S`, which releases `S·(1−e^{−dt/k})` per step to `to_unit`, so the leaked water arrives one step later at the earliest | `fraction`, `gw_k` |

* **Rule hook** for operated edges: `net.get_transfer("KAR-1").rule = f`, where
  `f(year, time_s, q_available)` returns the maximum withdrawal in m³/s, or `nan` for
  no constraint. The rule acts as a passive cap: it is not differentiated, but
  `fraction` and `capacity` still are. The `rule` column in transfers.csv is not
  interpreted yet; if it is filled, the loader logs a warning.
* **Time window:** an edge is active from `commissioned` (inclusive) to `decommissioned`
  (exclusive). Both are fractional calendar years, and the simulation year is
  `transfer_options.sim_start_year + time/365.2425 d`. If `sim_start_year` is NaN
  (the default), every edge is active.
* **Declared losses:** a `to_unit` of `0` (→ `to_reach = -1`) sends the water out of the
  system. The mass balance reports it as `declared_losses`.
* **Reservoir withdrawals:** an operated edge whose id is listed in the source lake's
  `operated_withdrawals` gets `from_storage = True` and draws from the lake's
  **storage** rather than from its spill. Its demand is
  `min(fraction·I_lake, capacity, rule)`, limited by the available water, and it can
  lower the lake to `lake_min_storage` below the crest.

### Defaults for empty fields (calibration parameters)

| Field | Default |
|---|---|
| headrace/diversion `fraction` | 1.0 |
| headrace/diversion `capacity_m3s` | `capacity_factor` (1.0) × prior mean flow at the source. The prior mean flow is the effective drainage area on the combined graph × 50 L/s/km² (`specific_runoff_m3s_km2`) |
| natural_bifurcation `fraction` | 0.25 |
| groundwater `fraction`, `gw_k` | 0.05, 30 days |

Override them with `transfer_defaults={...}` (see `network_adapter.TRANSFER_DEFAULTS`).

## Level-pool lake node

Storage `S` is measured from the outlet crest, so the stage is `h = max(S,0)/A` and

```
dS/dt = I − spill − Σ W_i,     spill = lake_weir_coef · h^lake_weir_exp
```

`lake_weir_coef = C_w · B`, with `C_w = 1.7` and `B = max(width prior, 5 m)`.
`lake_weir_exp` defaults to 1.5. The spill update is linearised-implicit, so it is
stable for any `dt`, and the storage update is exact in volume. `lake_min_storage` is:

* reservoirs: `−A · min(10 m, max_depth_m_wfd)`;
* natural lakes with operated withdrawals: `−A · 1 m`;
* other natural lakes: 0, so they cannot be drawn below the crest.

Lakes that share an outlet unit are merged into one node: their areas are summed and
their withdrawals combined. `storage_curve` is not used yet. Set `lake_model = 0` to
keep the older HydroLAKES rating-curve lake (`droute_lakes.yaml`).

## Differentiable parameters

The new parameters sit on the CoDiPack tape next to the channel parameters
(`Network.get_parameter_names()`). `get_gradients()` returns them under these names:

* `transfer_<id>_fraction`, `transfer_<id>_capacity` (only when a capacity is set),
  `transfer_<id>_gw_k` (groundwater edges);
* `reach_<i>_lake_weir_coef`, `reach_<i>_lake_weir_exp` (level-pool lakes).

All min/max operations are piecewise (hard), as in the existing lake routine. The
gradient comes from whichever branch is active, so a capacity that does not bind
gets a zero gradient. Gradients are checked against finite differences in
`tests/python/test_transfers_lakes.py` and `tests/test_transfers.cpp`.

## Mass balance

`router.mass_balance()` returns cumulative volumes in m³ since the router was created
or since `reset_mass_balance()`:

```
input = terminal_outflow + declared_losses + lake_storage_change
      + groundwater_storage_change + channel_storage_change + residual
```

`terminal_outflow` is the net outflow of the reaches that drain into no other reach
(sea or sink). For the IRF router the channel storage (water still in the kernel) is
exact, and the residual is at round-off level (~1e-14). For Muskingum-Cunge, the
channel storage is the Muskingum storage `K(XI+(1−X)O)`. That is exact within a step,
but K and X change between steps, and lakes release a step-mean flow that the next
channel integrates with the trapezoidal rule. A small residual therefore builds up
during transients and stops growing at steady state.

## SYMFLUENCE config keys

| Key | Default | Meaning |
|---|---|---|
| `DROUTE_TRANSFERS_FILE` | `default` | path to transfers.csv. `default` looks next to the topology file; `none` turns transfers off |
| `DROUTE_LAKES_FILE` | `default` | path to lakes.csv (level-pool nodes). Same `default` and `none` values |
| `DROUTE_ENABLE_GROUNDWATER_TRANSFERS` | `False` | apply `groundwater` edges |
| `DROUTE_TRANSFER_CAPACITY_FACTOR` | `1.0` | default capacity = factor × prior mean flow |
| `DROUTE_BIFURCATION_FRACTION` | `0.25` | default `natural_bifurcation` fraction |
| `DROUTE_LAKE_WEIR_C` | `1.7` | default weir coefficient `C_w` |

The runner sets `sim_start_year` from the first runoff timestamp.

## Input columns

`transfers.csv`: `transfer_id, type, operated, from_unit, to_unit, capacity_m3s,
fraction, rule, commissioned` are required. `decommissioned`, `status`,
`topology_effect` and `confidence` are optional. Rows whose `status` is not `resolved`
are skipped. Units are matched to `segId` in topology.nc.

`lakes.csv`: `unit_id` (the outlet segment) and `area_km2` are required. `reservoir`,
`operated_withdrawals` (comma-separated transfer ids), `max_depth_m_wfd` and
`storage_curve` are optional.
