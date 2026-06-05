# SPDX-License-Identifier: Apache-2.0
"""Per-reservoir operating-rule calibration on Bow at Calgary via dRoute AD gradients.

Each inline reservoir (lake_type==1 in droute_lakes.yaml) has a learnable storage-discharge
operating rule Q(S) = q_min + (q_ref - q_min)*frac^exp + spill_coef*overflow. This calibrates
those four parameters PER RESERVOIR (5 reservoirs -> 20 parameters) with Adam, using dRoute's
exact reverse-mode (CoDiPack) gradients -- the capability the library exists to demonstrate.

Routing is done at a DAILY timestep (hourly SUMMA runoff aggregated to daily) to keep the AD
tape light over the multi-year calibration window. The loss is (1 - KGE) at the Calgary outlet
vs WSC observations, with the analytical dKGE/dQ gradient (verified against finite differences,
cos = 1.0) seeding dRoute's reverse pass -- i.e. Adam optimises the actual calibration objective,
not an MSE surrogate. The best calibration KGE is kept.

Run AFTER the SUMMA hydrology calibration so the routed volumes are unbiased; works on the
current runoff too (machinery test).
"""
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import yaml
import droute

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
OBS = f"{D}/observations/streamflow"
RES = "/Users/darri.eythorsson/compHydro/code/dRoute/experiments/results_calgary"
DT_DAY = 86400.0
ROUTE_START = "2010-01-01"          # 1-yr routing spin-up so lake/subgrid stores equilibrate
CAL = ("2011-01-01", "2012-12-31")  # evaluation window (spin-up excluded from loss/metrics)

# per-reservoir parameter bounds (operating rule)
BOUNDS = {
    "q_ref": (0.05, 500.0, "log"),   # reference release (m^3/s)
    "exp":   (1.0, 3.0, "lin"),      # rating exponent
    "q_min": (0.0, 50.0, "lin"),     # minimum regulated release (m^3/s)
    "spill": (0.1, 3.0, "lin"),      # above-full spill coefficient
}


def build_network(seg_ids, downstream_idx, lengths, slopes, mannings_n=0.035):
    n = len(seg_ids); outlet_junc = n
    junc_up = {i: [] for i in range(n + 1)}
    for i in range(n):
        d = downstream_idx[i]; junc_up[d if d >= 0 else outlet_junc].append(i)
    net = droute.Network()
    for jid in range(n + 1):
        j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]; net.add_junction(j)
    for i in range(n):
        r = droute.Reach(); r.id = i; r.length = float(lengths[i]); r.slope = max(float(slopes[i]), 0.001)
        r.manning_n = mannings_n; r.upstream_junction_id = i
        d = downstream_idx[i]; r.downstream_junction_id = d if d >= 0 else outlet_junc
        net.add_reach(r)
    net.build_topology()
    return net


def load_inputs(runoff_path=None):
    runoff_path = runoff_path or f"{D}/simulations/bow_calgary_v1/SUMMA/bow_calgary_v1_timestep.nc"
    rn = gpd.read_file(glob.glob(f"{D}/shapefiles/river_network/*.shp")[0])
    seg_ids = rn["LINKNO"].astype(int).values
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    downstream_idx = np.array([id_to_idx.get(int(d), -1) for d in rn["DSLINKNO"].astype(int).values])
    lengths = rn["Length"].astype(float).values
    slopes = rn["Slope"].astype(float).values
    outlet_idx = int(np.argmax(rn["DSContArea"].astype(float).values))

    ds = xr.open_dataset(runoff_path)
    runoff = ds["averageRoutedRunoff"].values
    gru = ds["gruId"].values.astype(int)
    time = pd.to_datetime(ds["time"].values)
    attr = xr.open_dataset(f"{D}/settings/SUMMA/attributes.nc")
    area_by_id = {int(h): float(a) for h, a in zip(attr["hruId"].values.astype(int),
                                                    attr["HRUarea"].values.astype(float))}
    n_seg = len(seg_ids)
    seg_runoff = np.zeros((len(time), n_seg))
    for j, g in enumerate(gru):
        i = id_to_idx.get(int(g))
        if i is not None:
            seg_runoff[:, i] = np.clip(runoff[:, j], 0, None) * area_by_id.get(int(g), 0.0)

    # hourly -> daily mean lateral inflow (m^3/s); route from ROUTE_START (incl. spin-up year)
    daily = pd.DataFrame(seg_runoff, index=time).resample("D").mean()
    daily = daily.loc[ROUTE_START:CAL[1]]
    return dict(seg_ids=seg_ids, id_to_idx=id_to_idx, downstream_idx=downstream_idx,
                lengths=lengths, slopes=slopes, outlet_idx=outlet_idx,
                daily_runoff=daily.values, dates=daily.index)


def mask_spinup(obs, dates):
    """Set obs to NaN before the evaluation window so the routing spin-up year is
    excluded from the loss/metric automatically (kge/dloss_dsim skip NaN obs)."""
    obs = np.asarray(obs, dtype=float).copy()
    obs[pd.to_datetime(dates) < pd.Timestamp(CAL[0])] = np.nan
    return obs


def apply_lakes_and_get_reservoirs(net, id_to_idx):
    from droute.lake_preprocessor import apply_lake_config_to_network
    with open(f"{D}/settings/dRoute/droute_lakes.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cls = {"inline": raw.get("inline_lakes", {}), "subgrid": raw.get("subgrid_lakes", {})}
    apply_lake_config_to_network(net, cls, id_to_idx)
    # reservoirs = inline lakes with lake_type == 1
    res_idx = [id_to_idx[int(s)] for s, r in cls["inline"].items()
               if int(r.get("lake_type", 0)) == 1 and int(s) in id_to_idx]
    return sorted(res_idx)


def to_unit(name, val):
    lo, hi, tr = BOUNDS[name]
    if tr == "log":
        return (np.log(max(val, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (val - lo) / (hi - lo)


def from_unit(name, u):
    lo, hi, tr = BOUNDS[name]
    u = min(max(u, 0.0), 1.0)
    if tr == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def dval_du(name, u):
    """d(param)/d(unit) for chain rule from param-gradient to unit-gradient."""
    lo, hi, tr = BOUNDS[name]
    if tr == "log":
        return from_unit(name, u) * (np.log(hi) - np.log(lo))
    return hi - lo


def set_reservoir_params(net, res_idx, U):
    """U: dict {reach_idx: {name: unit_value}} -> set physical params on reaches."""
    for ri in res_idx:
        r = net.get_reach(ri)
        r.lake_q_ref = from_unit("q_ref", U[ri]["q_ref"])
        r.lake_exp = from_unit("exp", U[ri]["exp"])
        r.lake_q_min = from_unit("q_min", U[ri]["q_min"])
        r.lake_spill_coef = from_unit("spill", U[ri]["spill"])


def route_daily(net, daily_runoff, outlet_idx, record=False):
    n_t, n_seg = daily_runoff.shape
    c = droute.RouterConfig(); c.dt = DT_DAY; c.enable_gradients = record
    rt = droute.MuskingumCungeRouter(net, c)
    order = np.asarray(net.topological_order(), dtype=int)
    if record:
        rt.reset_gradients(); rt.start_recording()
    Q = np.zeros(n_t)
    for t in range(n_t):
        for idx in order:
            rt.set_lateral_inflow(int(idx), float(daily_runoff[t, idx]))
        rt.route_timestep()
        if record:
            rt.record_output(outlet_idx)
        Q[t] = rt.get_discharge(outlet_idx)
    if record:
        rt.stop_recording()
    return Q, rt


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def dloss_dsim(sim, obs):
    """Analytical gradient of the loss (1 - KGE) w.r.t. each daily routed discharge.

    Returns a full-length vector (0 where obs is missing) to seed dRoute's reverse
    pass directly on the actual calibration objective (KGE), not an MSE surrogate.
    Verified against central finite differences (cos = 1.0). Population statistics.
    """
    g = np.zeros_like(sim, dtype=float)
    m = np.isfinite(sim) & np.isfinite(obs)
    s = sim[m].astype(float); o = obs[m].astype(float); N = len(s)
    if N < 10 or s.std() == 0 or o.std() == 0:
        return g
    sb, ob, sig_s, sig_o = s.mean(), o.mean(), s.std(), o.std()
    Ss = np.sum((s - sb) ** 2); So = np.sum((o - ob) ** 2)
    r = np.sum((s - sb) * (o - ob)) / np.sqrt(Ss * So)
    alpha, beta = sig_s / sig_o, sb / ob
    ED = (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    sED = np.sqrt(ED) if ED > 0 else 1e-12
    dr = (o - ob) / np.sqrt(Ss * So) - r * (s - sb) / Ss
    dalpha = (s - sb) / (N * sig_s * sig_o)
    dbeta = (1.0 / N) / ob
    dED = 2 * (r - 1) * dr + 2 * (alpha - 1) * dalpha + 2 * (beta - 1) * dbeta
    g[m] = dED / (2 * sED)            # d(1 - KGE)/d sim_t
    return g


def main(epochs=60, lr=0.015, seed=0, runoff_path=None):
    inp = load_inputs(runoff_path)
    net = build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
    res_idx = apply_lakes_and_get_reservoirs(net, inp["id_to_idx"])
    print(f"{len(res_idx)} reservoirs at reaches {res_idx} -> {4*len(res_idx)} operating-rule params")

    obs = pd.read_csv(f"{OBS}/wsc_05BH004_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
    obs = mask_spinup(obs.reindex(inp["dates"]).values, inp["dates"])
    runoff = inp["daily_runoff"]; outlet = inp["outlet_idx"]

    # init units from current (HydroLAKES-derived) physical values
    U = {}
    for ri in res_idx:
        r = net.get_reach(ri)
        U[ri] = {"q_ref": to_unit("q_ref", max(float(r.lake_q_ref), 0.1)),
                 "exp": to_unit("exp", float(r.lake_exp) if r.lake_exp > 0 else 1.5),
                 "q_min": to_unit("q_min", float(r.lake_q_min)),
                 "spill": to_unit("spill", float(r.lake_spill_coef) if r.lake_spill_coef > 0 else 1.0)}

    # Adam state
    names = ["q_ref", "exp", "q_min", "spill"]
    m = {ri: {k: 0.0 for k in names} for ri in res_idx}
    v = {ri: {k: 0.0 for k in names} for ri in res_idx}
    b1, b2, eps = 0.9, 0.999, 1e-8

    set_reservoir_params(net, res_idx, U)
    Q0, _ = route_daily(net, runoff, outlet, record=False)
    kge0 = kge(Q0, obs)
    best = {"kge": kge0, "U": {ri: dict(U[ri]) for ri in res_idx}, "epoch": 0}
    print(f"epoch  0: KGE={kge0:.4f}  (initial HydroLAKES rules)")

    grad_attr = {"q_ref": "grad_lake_q_ref", "exp": "grad_lake_exp",
                 "q_min": "grad_lake_q_min", "spill": "grad_lake_spill_coef"}
    N = np.isfinite(obs).sum()
    for ep in range(1, epochs + 1):
        set_reservoir_params(net, res_idx, U)
        Q, rt = route_daily(net, runoff, outlet, record=True)
        dL_dQ = dloss_dsim(Q, obs)                     # d(1 - KGE) / d Q_t at outlet
        rt.compute_gradients_timeseries(outlet, dL_dQ.tolist())
        # Adam step in unit space (chain rule: dL/du = dL/dparam * dparam/du)
        for ri in res_idx:
            r = net.get_reach(ri)
            for k in names:
                g_param = float(getattr(r, grad_attr[k]))
                g = g_param * dval_du(k, U[ri][k])
                m[ri][k] = b1 * m[ri][k] + (1 - b1) * g
                v[ri][k] = b2 * v[ri][k] + (1 - b2) * g * g
                mh = m[ri][k] / (1 - b1 ** ep); vh = v[ri][k] / (1 - b2 ** ep)
                U[ri][k] = min(max(U[ri][k] - lr * mh / (np.sqrt(vh) + eps), 0.0), 1.0)
        set_reservoir_params(net, res_idx, U)
        Qe, _ = route_daily(net, runoff, outlet, record=False)
        ke = kge(Qe, obs)
        if ke > best["kge"]:
            best = {"kge": ke, "U": {ri: dict(U[ri]) for ri in res_idx}, "epoch": ep}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:2d}: KGE={ke:.4f}  best={best['kge']:.4f} (ep {best['epoch']})")

    print(f"\nInitial KGE {kge0:.4f} -> best KGE {best['kge']:.4f} at epoch {best['epoch']}")
    # report best per-reservoir rules
    print("Calibrated operating rules (best):")
    for ri in res_idx:
        seg = int(inp["seg_ids"][ri])
        print(f"  seg {seg}: q_ref={from_unit('q_ref',best['U'][ri]['q_ref']):.2f} "
              f"exp={from_unit('exp',best['U'][ri]['exp']):.2f} "
              f"q_min={from_unit('q_min',best['U'][ri]['q_min']):.2f} "
              f"spill={from_unit('spill',best['U'][ri]['spill']):.2f}")
    return kge0, best, res_idx, inp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--runoff", default=None, help="SUMMA runoff netCDF (default: current uncalibrated)")
    a = ap.parse_args()
    main(epochs=a.epochs, lr=a.lr, runoff_path=a.runoff)
