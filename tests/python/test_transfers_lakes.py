"""Transfers (headrace, diversion, natural bifurcation, groundwater), level-pool
lake/reservoir nodes, cycle rejection, mass conservation and AD gradients on
small synthetic networks, plus an integration test on the Landslikan v0.3
hydrofabric (skipped when its files are absent)."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

droute = pytest.importorskip("droute")
from droute.network_adapter import DRouteNetworkAdapter  # noqa: E402

DT = 3600.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_net(downstream, length=2000.0, slope=0.002, width=10.0):
    """Network with reach id = index; downstream[i] = -1 for a terminal reach."""
    n = len(downstream)
    cfg = {
        'n_segments': n,
        'segment_ids': list(range(1, n + 1)),
        'downstream_idx': list(downstream),
        'lengths': [length] * n,
        'slopes': [slope] * n,
        'widths': [width] * n,
        'hru_to_seg_idx': list(range(n)),
        'hru_areas': [1e7] * n,
        'routing_order': DRouteNetworkAdapter()._compute_routing_order(np.array(downstream)),
    }
    return DRouteNetworkAdapter().build_network(cfg, mannings_n=0.035), cfg


def transfer(tid, kind, src, dst, fraction=1.0, capacity=None, operated=None, **kw):
    t = droute.Transfer()
    t.id = tid
    t.type = droute.parse_transfer_type(kind)
    t.operated = (kind in ('headrace', 'diversion')) if operated is None else operated
    t.from_reach = src
    t.to_reach = dst
    t.fraction = fraction
    t.capacity = capacity
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def make_lake(net, rid, area=1e6, weir_coef=20.0, reservoir=False, min_storage=0.0):
    r = net.get_reach(rid)
    r.is_lake = True
    r.lake_model = droute.LAKE_LEVEL_POOL
    r.lake_type = 1 if reservoir else 0
    r.lake_area = area
    r.lake_weir_coef = weir_coef
    r.lake_weir_exp = 1.5
    r.lake_min_storage = min_storage
    r.storage = 0.0


def run(net, lateral, steps, router_cls=None, config=None, every=None):
    cfg = config or droute.RouterConfig()
    cfg.dt = DT
    if config is None:
        cfg.enable_gradients = False
    router = (router_cls or droute.MuskingumCungeRouter)(net, cfg)
    for rid, q in lateral.items():
        router.set_lateral_inflow(rid, q)
    for k in range(steps):
        router.route_timestep()
        if every:
            every(router, k)
    return router


# ---------------------------------------------------------------------------
# edge types
# ---------------------------------------------------------------------------

# Two parallel branches: 0 -> 1 -> out,  2 -> 3 -> out
TWO_BRANCHES = [1, -1, 3, -1]


class TestOperatedEdges:
    def test_headrace_capacity_limits_withdrawal(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('H1', 'headrace', 0, 2, fraction=1.0, capacity=3.0))
        r = run(net, {0: 10.0, 1: 1.0, 2: 2.0, 3: 0.5}, 240)
        t = net.get_transfer('H1')
        assert t.last_flow == pytest.approx(3.0, rel=1e-6)
        assert r.get_discharge(0) == pytest.approx(10.0, rel=1e-4)        # gross (gauge above offtake)
        assert r.get_net_discharge(0) == pytest.approx(7.0, rel=1e-4)
        assert r.get_discharge(1) == pytest.approx(8.0, rel=1e-4)
        assert r.get_discharge(3) == pytest.approx(5.5, rel=1e-4)

    def test_diversion_fraction_without_capacity(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('D1', 'diversion', 0, 3, fraction=0.4))
        r = run(net, {0: 10.0}, 240)
        assert net.get_transfer('D1').last_flow == pytest.approx(4.0, rel=1e-4)
        assert r.get_discharge(1) == pytest.approx(6.0, rel=1e-4)
        assert r.get_discharge(3) == pytest.approx(4.0, rel=1e-4)

    def test_rule_hook_caps_operated_edge(self):
        net, _ = make_net(TWO_BRANCHES)
        t = transfer('H1', 'headrace', 0, 2, fraction=1.0, capacity=8.0)
        calls = []

        def rule(year, time_s, q_available):
            calls.append((year, time_s, q_available))
            return 1.5 if time_s >= 100 * DT else float('nan')

        t.rule = rule
        net.add_transfer(t)
        flows = []
        run(net, {0: 10.0}, 240, every=lambda r, k: flows.append(net.get_transfer('H1').last_flow))
        assert calls and len(calls) == 240
        assert flows[90] == pytest.approx(8.0, rel=1e-4)    # capacity only
        assert flows[-1] == pytest.approx(1.5, rel=1e-6)    # rule cap

    def test_commissioning_window(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('H1', 'headrace', 0, 2, capacity=5.0, commissioned=2000.0,
                                  decommissioned=2000.02))
        net.transfer_options.sim_start_year = 1999.99   # ~3.65 days before 2000.0
        flows = []
        run(net, {0: 10.0}, 24 * 15, every=lambda r, k: flows.append(net.get_transfer('H1').last_flow))
        year = lambda k: 1999.99 + k * DT / (365.2425 * 86400)  # noqa: E731
        before = [f for k, f in enumerate(flows) if year(k) < 2000.0]
        during = [f for k, f in enumerate(flows) if 2000.0 + 0.001 < year(k) < 2000.02]
        after = [f for k, f in enumerate(flows) if year(k) >= 2000.02]
        assert before and during and after
        assert max(before) == 0.0 and max(after) == 0.0
        assert min(during) > 4.9

    def test_no_year_means_always_active(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('H1', 'headrace', 0, 2, capacity=5.0, commissioned=2100.0))
        run(net, {0: 10.0}, 48)
        assert net.get_transfer('H1').last_flow > 4.9


class TestNaturalEdges:
    def test_natural_bifurcation_constant_fraction(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('B1', 'natural_bifurcation', 0, 2, fraction=0.3, operated=False))
        r = run(net, {0: 20.0}, 240)
        assert net.get_transfer('B1').last_flow == pytest.approx(6.0, rel=1e-4)
        assert r.get_discharge(1) == pytest.approx(14.0, rel=1e-4)
        assert r.get_discharge(3) == pytest.approx(6.0, rel=1e-4)

    def test_groundwater_off_by_default(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('G1', 'groundwater', 0, 2, fraction=0.2, operated=False))
        assert net.transfer_options.enable_groundwater is False
        r = run(net, {0: 10.0}, 48)
        assert net.get_transfer('G1').last_flow == 0.0
        assert r.get_discharge(3) == pytest.approx(0.0, abs=1e-3)

    def test_groundwater_is_lagged_linear_reservoir(self):
        net, _ = make_net(TWO_BRANCHES)
        k_s = 2 * 86400.0
        net.add_transfer(transfer('G1', 'groundwater', 0, 2, fraction=0.2, operated=False, gw_k=k_s))
        net.transfer_options.enable_groundwater = True
        recv = []
        run(net, {0: 10.0}, 24 * 30, every=lambda r, k: recv.append(net.get_reach(2).transfer_in))
        assert recv[0] == 0.0                 # nothing arrives in the first step
        assert 0.0 < recv[24] < 1.0           # still filling after a day
        assert recv[-1] == pytest.approx(2.0, rel=1e-3)   # steady state: 20% of Q
        # store ~ q * k at steady state
        assert net.get_transfer('G1').gw_store == pytest.approx(2.0 * k_s, rel=0.05)


# ---------------------------------------------------------------------------
# lakes and reservoirs
# ---------------------------------------------------------------------------

class TestLevelPoolLakes:
    def test_natural_lake_weir_equilibrium(self):
        net, _ = make_net([1, 2, -1])
        make_lake(net, 1, area=1e6, weir_coef=20.0)
        r = run(net, {0: 10.0}, 24 * 10)
        lake = net.get_reach(1)
        h = (10.0 / 20.0) ** (1 / 1.5)
        assert r.get_discharge(1) == pytest.approx(10.0, rel=1e-4)
        assert lake.storage == pytest.approx(1e6 * h, rel=1e-3)

    def test_lake_attenuates_and_conserves(self):
        net, _ = make_net([1, 2, -1])
        make_lake(net, 1, area=5e6, weir_coef=20.0)
        out = []
        r = run(net, {0: 10.0}, 24 * 3, every=lambda r, k: out.append(r.get_discharge(1)))
        assert out[5] < 10.0 * 0.5             # filling: outflow well below inflow
        mb = r.mass_balance()
        assert mb.lake_storage_change > 0.0

    def test_reservoir_outflow_is_spill_plus_withdrawals(self):
        net, _ = make_net(TWO_BRANCHES)
        make_lake(net, 0, area=2e6, weir_coef=15.0, reservoir=True, min_storage=-2e6 * 5)
        net.add_transfer(transfer('H1', 'headrace', 0, 2, fraction=1.0, capacity=4.0, from_storage=True))
        r = run(net, {0: 10.0}, 24 * 15)
        lake = net.get_reach(0)
        assert lake.transfer_out == pytest.approx(4.0, rel=1e-6)
        assert r.get_net_discharge(0) == pytest.approx(6.0, rel=1e-3)     # spill
        assert r.get_discharge(0) == pytest.approx(10.0, rel=1e-3)        # spill + withdrawal
        assert r.get_discharge(3) == pytest.approx(4.0, rel=1e-3)

    def test_reservoir_withdrawal_can_draw_down_below_crest(self):
        net, _ = make_net(TWO_BRANCHES)
        A = 1e6
        make_lake(net, 0, area=A, weir_coef=15.0, reservoir=True, min_storage=-A * 2.0)
        t = transfer('H1', 'headrace', 0, 2, fraction=1.0, capacity=5.0, from_storage=True)
        net.add_transfer(t)
        # demand = min(fraction*capacity, rule) = 5 m3/s > inflow of 1 m3/s: storage is drawn down
        run(net, {0: 1.0}, 24 * 20)
        lake = net.get_reach(0)
        assert lake.storage == pytest.approx(-A * 2.0, rel=1e-6)   # hits the drawdown floor
        assert lake.transfer_out == pytest.approx(1.0, rel=1e-6)   # then only inflow is available
        assert lake.net_outflow == pytest.approx(0.0, abs=1e-9)    # no spill below the crest


# ---------------------------------------------------------------------------
# ordering and cycles
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_order_respects_transfer_edges(self):
        # 0 -> 1 -> out ; 2 -> out ; transfer 1 -> 2 forces 2 after 1
        net, _ = make_net([1, -1, -1])
        net.add_transfer(transfer('D', 'diversion', 1, 2, fraction=0.5))
        net.build_topology()
        order = list(net.topological_order())
        assert order.index(1) < order.index(2)
        assert order.index(0) < order.index(1)

    def test_cycle_is_rejected(self):
        # 0 -> 1 -> 2 -> out ; transfer 2 -> 0 closes a loop
        net, _ = make_net([1, 2, -1])
        net.add_transfer(transfer('BACK', 'diversion', 2, 0, fraction=0.5))
        with pytest.raises(RuntimeError, match='cycle') as exc:
            net.build_topology()
        assert 'BACK' in str(exc.value)
        with pytest.raises(RuntimeError, match='cycle'):
            droute.MuskingumCungeRouter(net, droute.RouterConfig())

    def test_fractions_validated(self):
        net, _ = make_net([1, 2, -1, -1, -1])
        net.add_transfer(transfer('SKA-1', 'natural_bifurcation', 0, 3, fraction=0.6, operated=False))
        net.add_transfer(transfer('SKA-2', 'natural_bifurcation', 0, 4, fraction=0.5, operated=False))
        with pytest.raises(ValueError, match='SKA-1 \\+ SKA-2'):
            net.build_topology()
        net.get_transfer('SKA-2').fraction = 0.4
        net.build_topology()
        net.get_transfer('SKA-2').fraction = 1.5
        with pytest.raises(ValueError, match='outside'):
            net.build_topology()

    def test_invalid_endpoints(self):
        net, _ = make_net([1, -1])
        with pytest.raises(ValueError):
            net.add_transfer(transfer('X', 'diversion', 0, 99))
        with pytest.raises(ValueError):
            net.add_transfer(transfer('X', 'diversion', 0, 0))

    def test_unsupported_routers_raise(self):
        net, _ = make_net(TWO_BRANCHES)
        net.add_transfer(transfer('D', 'diversion', 0, 2, fraction=0.5))
        for cls in (droute.LagRouter, droute.DiffusiveWaveRouter, droute.KWTRouter):
            with pytest.raises(RuntimeError, match='not supported with transfers'):
                cls(net, droute.RouterConfig())

    def test_terminal_reaches(self):
        net, _ = make_net([1, -1, 3, -1])
        assert sorted(net.terminal_reaches()) == [1, 3]


# ---------------------------------------------------------------------------
# mass conservation
# ---------------------------------------------------------------------------

def full_network():
    """Every feature at once.

        0 -> 1 -> 2 (reservoir) -> 3 -> out      4 -> 5 -> 6 -> out
        2 =headrace (from storage)=> 5
        1 -natural bifurcation-> 6
        3 ~groundwater (lagged)~> 6
        0 ~groundwater~> leaves the system (declared loss)
    """
    net, _ = make_net([1, 2, 3, -1, 5, 6, -1])
    make_lake(net, 2, area=3e6, weir_coef=25.0, reservoir=True, min_storage=-3e6 * 3)
    net.add_transfer(transfer('HR', 'headrace', 2, 5, fraction=1.0, capacity=3.0, from_storage=True))
    net.add_transfer(transfer('BF', 'natural_bifurcation', 1, 6, fraction=0.2, operated=False))
    net.add_transfer(transfer('GW', 'groundwater', 3, 6, fraction=0.1, operated=False, gw_k=86400.0))
    net.add_transfer(transfer('LOSS', 'groundwater', 0, -1, fraction=0.05, operated=False, gw_k=86400.0))
    net.transfer_options.enable_groundwater = True
    return net


LATERAL = {0: 5.0, 1: 2.0, 2: 1.0, 3: 1.5, 4: 3.0, 5: 0.5, 6: 0.25}


def _varying(router, k):
    # time-varying runoff to exercise transients
    f = 1.0 + 0.8 * math.sin(2 * math.pi * k / 48.0)
    for rid, q in LATERAL.items():
        router.set_lateral_inflow(rid, q * f)


class TestMassConservation:
    def test_irf_conserves_exactly(self):
        net = full_network()
        cfg = droute.RouterConfig()
        cfg.enable_gradients = False
        cfg.irf_max_kernel_size = 200
        r = run(net, LATERAL, 24 * 10, router_cls=droute.IRFRouter, config=cfg, every=_varying)
        mb = r.mass_balance()
        assert mb.input > 0
        assert mb.declared_losses > 0
        assert mb.groundwater_storage_change > 0
        assert mb.lake_storage_change != 0
        assert abs(mb.relative_residual) < 1e-10, mb.as_dict()

    def test_mc_conserves(self):
        net = full_network()
        r = run(net, LATERAL, 24 * 10, every=_varying)
        mb = r.mass_balance()
        # Muskingum storage K(XI+(1-X)O) is exact within a step but K, X change
        # between steps, so a small transient residual remains.
        assert abs(mb.relative_residual) < 2e-3, mb.as_dict()

    def test_mc_steady_state_exact(self):
        net = full_network()
        r = run(net, LATERAL, 24 * 20)
        r.reset_mass_balance()
        for _ in range(24 * 5):
            r.route_timestep()
        mb = r.mass_balance()
        assert abs(mb.relative_residual) < 1e-6, mb.as_dict()
        # flux check: input = terminal outflow + losses
        q_in = sum(LATERAL.values())
        q_out = sum(r.get_net_discharge(t) for t in net.terminal_reaches())
        q_loss = net.get_transfer('LOSS').last_flow   # steady: leak in = release out
        assert q_out + q_loss == pytest.approx(q_in, rel=1e-4)


# ---------------------------------------------------------------------------
# differentiability
# ---------------------------------------------------------------------------

def _outlet_sum(net_factory, set_param, value, steps=48, outlet=3):
    net = net_factory()
    set_param(net, value)
    cfg = droute.RouterConfig()
    cfg.dt = DT
    cfg.enable_gradients = False
    r = droute.MuskingumCungeRouter(net, cfg)
    total = 0.0
    for k in range(steps):
        _varying(r, k)
        r.route_timestep()
        total += r.get_discharge(outlet)
    return total


def _ad_grads(net_factory, steps=48, outlet=3):
    net = net_factory()
    cfg = droute.RouterConfig()
    cfg.dt = DT
    cfg.enable_gradients = True
    r = droute.MuskingumCungeRouter(net, cfg)
    r.start_recording()
    for k in range(steps):
        _varying(r, k)
        r.route_timestep()
        r.record_output(outlet)
    r.stop_recording()
    r.compute_gradients_timeseries(outlet, [1.0] * steps)
    return r.get_gradients(), net


def _grad_net():
    net, _ = make_net([1, 2, 3, -1, 5, 6, -1])
    make_lake(net, 2, area=5e5, weir_coef=25.0)
    net.get_reach(2).storage = 2e5
    net.add_transfer(transfer('DIV', 'diversion', 1, 5, fraction=0.3, capacity=50.0))
    net.add_transfer(transfer('CAP', 'headrace', 0, 4, fraction=0.9, capacity=2.0))
    net.add_transfer(transfer('BF', 'natural_bifurcation', 3, 6, fraction=0.2, operated=False))
    return net


@pytest.mark.parametrize('name,setter,value', [
    ('transfer_DIV_fraction', lambda n, v: setattr(n.get_transfer('DIV'), 'fraction', v), 0.3),
    ('transfer_CAP_capacity', lambda n, v: setattr(n.get_transfer('CAP'), 'capacity', v), 2.0),
    ('reach_2_lake_weir_coef', lambda n, v: setattr(n.get_reach(2), 'lake_weir_coef', v), 25.0),
    ('reach_2_lake_weir_exp', lambda n, v: setattr(n.get_reach(2), 'lake_weir_exp', v), 1.5),
])
def test_mc_gradients_match_finite_differences(name, setter, value):
    grads, net = _ad_grads(_grad_net)
    if name not in grads:
        pytest.skip('dRoute built without AD')
    assert name in net.get_parameter_names()
    h = 1e-4 * max(abs(value), 1.0)
    fd = (_outlet_sum(_grad_net, setter, value + h) - _outlet_sum(_grad_net, setter, value - h)) / (2 * h)
    ad = grads[name]
    assert abs(fd) > 1e-6, f"{name}: parameter has no effect (fd={fd})"
    assert ad == pytest.approx(fd, rel=2e-3, abs=1e-6), f"{name}: AD={ad} FD={fd}"


def _grad_net_storage():
    # reservoir at 2 with a storage withdrawal 2 => 5, groundwater 1 ~> 6
    net, _ = make_net([1, 2, 3, -1, 5, 6, -1])
    make_lake(net, 2, area=5e5, weir_coef=25.0, reservoir=True, min_storage=-5e5 * 2)
    net.get_reach(2).storage = 2e5
    net.add_transfer(transfer('RES', 'headrace', 2, 5, fraction=0.8, capacity=4.0, from_storage=True))
    net.add_transfer(transfer('GW', 'groundwater', 1, 6, fraction=0.1, operated=False, gw_k=2 * 86400.0))
    net.transfer_options.enable_groundwater = True
    return net


@pytest.mark.parametrize('name,setter,value,outlet', [
    ('transfer_RES_fraction', lambda n, v: setattr(n.get_transfer('RES'), 'fraction', v), 0.8, 6),
    ('transfer_RES_capacity', lambda n, v: setattr(n.get_transfer('RES'), 'capacity', v), 4.0, 6),
    ('transfer_GW_fraction', lambda n, v: setattr(n.get_transfer('GW'), 'fraction', v), 0.1, 6),
    ('transfer_GW_gw_k', lambda n, v: setattr(n.get_transfer('GW'), 'gw_k', v), 2 * 86400.0, 6),
])
def test_mc_storage_and_groundwater_gradients(name, setter, value, outlet):
    grads, net = _ad_grads(_grad_net_storage, outlet=outlet)
    if name not in grads:
        pytest.skip('dRoute built without AD')
    h = 1e-4 * max(abs(value), 1.0)
    fd = (_outlet_sum(_grad_net_storage, setter, value + h, outlet=outlet)
          - _outlet_sum(_grad_net_storage, setter, value - h, outlet=outlet)) / (2 * h)
    ad = grads[name]
    assert abs(fd) > 1e-9, f"{name}: parameter has no effect (fd={fd})"
    assert ad == pytest.approx(fd, rel=2e-3, abs=1e-9), f"{name}: AD={ad} FD={fd}"


def test_bifurcation_fraction_gradient():
    # BF moves water from reach 3 (gauge above the split, gross flow unaffected)
    # to reach 6: the gradient is zero at 3 and positive at 6.
    grads3, _ = _ad_grads(_grad_net, outlet=3)
    if 'transfer_BF_fraction' not in grads3:
        pytest.skip('dRoute built without AD')
    assert grads3['transfer_BF_fraction'] == 0.0
    grads6, _ = _ad_grads(_grad_net, outlet=6)
    setter = lambda n, v: setattr(n.get_transfer('BF'), 'fraction', v)  # noqa: E731
    h = 1e-4
    fd = (_outlet_sum(_grad_net, setter, 0.2 + h, outlet=6)
          - _outlet_sum(_grad_net, setter, 0.2 - h, outlet=6)) / (2 * h)
    assert fd > 0
    assert grads6['transfer_BF_fraction'] == pytest.approx(fd, rel=2e-3)


def test_irf_exposes_transfer_and_lake_gradients():
    net = _grad_net()
    cfg = droute.RouterConfig()
    cfg.dt = DT
    cfg.enable_gradients = True
    cfg.irf_max_kernel_size = 100
    r = droute.IRFRouter(net, cfg)
    r.start_recording()
    for k in range(24):
        _varying(r, k)
        r.route_timestep()
    r.stop_recording()
    r.compute_gradients([3], [1.0])
    g = r.get_gradients()
    for key in ('transfer_DIV_fraction', 'transfer_CAP_capacity', 'reach_2_lake_weir_coef'):
        assert key in g


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

TRANSFERS_CSV = """transfer_id,type,operated,from_unit,to_unit,capacity_m3s,fraction,rule,commissioned,decommissioned,topology_effect,status
R-1,headrace,True,3,6,,,,1990.0,,cross_branch,resolved
B-1,natural_bifurcation,False,2,5,,0.4,,,,cross_branch,resolved
G-1,groundwater,False,4,7,,,,,,cross_branch,resolved
U-1,diversion,True,1,99,,,,,,bypass,unresolved
"""

LAKES_CSV = """lake_id,area_km2,reservoir,unit_id,name,operated_withdrawals,storage_curve,surface_elev_m,max_depth_m_wfd
1,2.0,True,3,Res,R-1,,500.0,4.0
2,0.5,False,3,Res-b,,,,
3,1.0,False,6,Lake,,,,
"""


def test_csv_loaders(tmp_path):
    # segIds 1..7: 1->2->3->4->out ; 5->6->7->out
    (tmp_path / 'transfers.csv').write_text(TRANSFERS_CSV, encoding='utf-8')
    (tmp_path / 'lakes.csv').write_text(LAKES_CSV, encoding='utf-8')
    net, cfg = make_net([1, 2, 3, -1, 5, 6, -1])
    summary = DRouteNetworkAdapter().apply_hydrofabric(
        net, cfg, transfers_path=tmp_path / 'transfers.csv', lakes_path=tmp_path / 'lakes.csv',
        sim_start_year=2000.0)
    assert sorted(summary['transfers']) == ['B-1', 'G-1', 'R-1']   # unresolved row skipped
    assert summary['lakes'] == 2                                   # two lakes share unit 3
    res = net.get_reach(2)                                         # segId 3 -> index 2
    assert res.is_lake and res.lake_model == droute.LAKE_LEVEL_POOL and res.lake_type == 1
    assert res.lake_area == pytest.approx(2.5e6)
    assert res.lake_min_storage == pytest.approx(-2.5e6 * 4.0)     # drawdown capped by max depth
    assert res.lake_weir_coef == pytest.approx(1.7 * 10.0)
    r1 = net.get_transfer('R-1')
    assert r1.from_reach == 2 and r1.to_reach == 5 and r1.from_storage
    # capacity prior on the combined graph: units 1..3 (30 km2) minus the 40% of
    # unit 1-2 flow (8 km2) that B-1 moves to the other branch -> 22 km2 * 50 L/s/km2
    assert r1.fraction == 1.0 and r1.capacity == pytest.approx(22 * 0.05)
    assert r1.commissioned == 1990.0
    b1 = net.get_transfer('B-1')
    assert b1.fraction == pytest.approx(0.4) and b1.capacity is None and not b1.from_storage
    assert net.get_transfer('G-1').type == droute.TransferType.groundwater
    assert net.transfer_options.enable_groundwater is False
    assert net.transfer_options.sim_start_year == 2000.0


# ---------------------------------------------------------------------------
# integration: Landslikan v0.3 national hydrofabric
# ---------------------------------------------------------------------------

V03 = Path(os.environ.get('LANDSLIKAN_DROUTE_DIR',
                          '/Users/darri/repos/landslikan/data/build/v0.3_50m/droute'))


@pytest.mark.slow
@pytest.mark.requires_data
@pytest.mark.skipif(not (V03 / 'topology.nc').exists() or not (V03 / 'transfers.csv').exists()
                    or not (V03 / 'lakes.csv').exists(), reason='Landslikan v0.3 files not found')
@pytest.mark.parametrize('router_name', ['mc', 'irf'])
def test_landslikan_v03_conservation(router_name):
    pytest.importorskip('xarray')
    adapter = DRouteNetworkAdapter()
    net, cfg, summary = adapter.load_hydrofabric(V03, sim_start_year=2020.0)
    assert len(summary['transfers']) == 19
    assert summary['lakes'] > 400
    rc = droute.RouterConfig()
    rc.dt = DT
    rc.enable_gradients = False
    if router_name == 'mc':
        router, spinup, window = droute.MuskingumCungeRouter(net, rc), 24 * 15, 24 * 5
    else:
        rc.irf_max_kernel_size = 48
        router, spinup, window = droute.IRFRouter(net, rc), 24 * 2, 24
    # constant 50 L/s/km2 runoff on every unit
    area = np.zeros(cfg['n_segments'])
    for h, s in enumerate(cfg['hru_to_seg_idx']):
        area[s] += cfg['hru_areas'][h]
    for rid in net.topological_order():
        router.set_lateral_inflow(rid, float(area[rid] * 0.05 / 1e6))
    for _ in range(spinup):
        router.route_timestep()
    full = router.mass_balance()
    router.reset_mass_balance()
    for _ in range(window):
        router.route_timestep()
    win = router.mass_balance()
    print(f"\n[{router_name}] full spin-up: {full.as_dict()}\n[{router_name}] window: {win.as_dict()}")
    assert full.input > 0 and win.input > 0
    if router_name == 'irf':
        assert abs(full.relative_residual) < 1e-9
        assert abs(win.relative_residual) < 1e-9
    else:
        assert abs(full.relative_residual) < 1e-2
        assert abs(win.relative_residual) < 1e-3
    # water moved through transfers during the window
    assert sum(net.get_transfer(t).last_flow for t in net.transfer_ids()) > 0


# ---------------------------------------------------------------------------
# SYMFLUENCE runner: output columns follow segment index with transfers
# ---------------------------------------------------------------------------

def test_runner_output_columns_with_transfers(tmp_path):
    """A transfer changes the topological order; routed columns must still be
    indexed by segment (regression: the runner scattered with the tree order)."""
    import logging
    import types
    runner_mod = pytest.importorskip('droute.runner')

    # segIds 1..6. Transfer 6 -> 1 (from the last segment to the first headwater)
    # forces segment 1 behind segment 6 in the combined order.
    downstream = [1, 2, -1, 4, 5, -1]
    n = len(downstream)
    network = {
        'n_segments': n, 'segment_ids': list(range(1, n + 1)), 'downstream_idx': downstream,
        'lengths': [2000.0] * n, 'slopes': [0.002] * n, 'widths': [10.0] * n,
        'hru_to_seg_idx': list(range(n)), 'hru_areas': [1.0] * n,
    }
    (tmp_path / 'transfers.csv').write_text(
        'transfer_id,type,operated,from_unit,to_unit,capacity_m3s,fraction,rule,commissioned\n'
        'X,diversion,True,6,1,,0.5,,\n', encoding='utf-8')
    stub = types.SimpleNamespace(
        logger=logging.getLogger('test'), config=None,
        network_adapter=DRouteNetworkAdapter(),
        droute_transfers_file=str(tmp_path / 'transfers.csv'), droute_lakes_file='none',
        droute_enable_groundwater_transfers=False, droute_transfer_capacity_factor=0.0,
        droute_bifurcation_fraction=0.25, droute_lake_weir_c=1.7,
        _get_config_value=lambda f, default=None: default,
        _apply_lakes=lambda net, network: 0,
    )
    R = runner_mod.DRouteRunner
    stub._resolve_layer_file = types.MethodType(R._resolve_layer_file, stub)
    stub._apply_hydrofabric = types.MethodType(R._apply_hydrofabric, stub)
    runoff = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (200, 1))   # m3/s per HRU (area 1)
    q = R._route_with_droute(stub, runoff, network, 'muskingum_cunge', 3600)
    # steady state: segment 6 (index 5) = 4+5+6 = 15 gross, half goes to segment 1
    assert q[-1, 5] == pytest.approx(15.0, rel=1e-3)
    assert q[-1, 0] == pytest.approx(1.0 + 7.5, rel=1e-3)
    assert q[-1, 2] == pytest.approx(1.0 + 7.5 + 2.0 + 3.0, rel=1e-3)
