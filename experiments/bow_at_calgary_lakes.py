# SPDX-License-Identifier: Apache-2.0
"""Bow-at-Calgary lake-aware routing: SUMMA runoff -> dRoute, with vs without lakes.

Builds the C++ Muskingum-Cunge network from the semi-distributed river network
(LINKNO/DSLINKNO/Length/Slope), aggregates SUMMA averageRoutedRunoff to reaches
(LINKNO == GRU_ID == gruId), then routes twice: a plain-channel baseline and a
lake-aware run that applies settings/dRoute/droute_lakes.yaml (5 reservoirs +
6 natural lakes inline, 34 subgrid catchments). Saves routed flows + a hydrograph.
"""
import glob
import numpy as np
import geopandas as gpd
import xarray as xr
import yaml
import droute
from droute.lake_preprocessor import apply_lake_config_to_network

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
OUTDIR = "/Users/darri.eythorsson/compHydro/code/dRoute/experiments/results_calgary"
DT = 3600.0  # SUMMA output is hourly

import os
os.makedirs(OUTDIR, exist_ok=True)


def build_network(seg_ids, downstream_idx, lengths, slopes, widths=None, mannings_n=0.035):
    """Build a C++ droute.Network (junctions + reaches). reach.id == index."""
    n = len(seg_ids)
    outlet_junc = n
    junc_up = {i: [] for i in range(n + 1)}
    for i in range(n):
        ds = downstream_idx[i]
        junc_up[ds if ds >= 0 else outlet_junc].append(i)
    net = droute.Network()
    for jid in range(n + 1):
        j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]
        net.add_junction(j)
    for i in range(n):
        r = droute.Reach()
        r.id = i
        r.length = float(lengths[i])
        r.slope = max(float(slopes[i]), 0.001)
        r.manning_n = mannings_n
        r.upstream_junction_id = i
        ds = downstream_idx[i]
        r.downstream_junction_id = ds if ds >= 0 else outlet_junc
        if widths is not None and widths[i] is not None:
            g = r.geometry; g.width_coef = float(widths[i]); g.width_exp = 0.0; r.geometry = g
        net.add_reach(r)
    net.build_topology()
    return net


def route(net, seg_runoff, dt=DT):
    """Route per-reach lateral inflow (n_time, n_seg) -> discharge (n_time, n_seg)."""
    cfg = droute.RouterConfig(); cfg.dt = float(dt); cfg.enable_gradients = False
    router = droute.MuskingumCungeRouter(net, cfg)
    order = np.asarray(net.topological_order(), dtype=int)
    nt, ns = seg_runoff.shape
    out = np.zeros((nt, ns))
    for t in range(nt):
        for idx in order:
            router.set_lateral_inflow(int(idx), float(seg_runoff[t, idx]))
        router.route_timestep()
        # get_all_discharges() is in topological order -> scatter back to reach index
        out[t, order] = router.get_all_discharges()
    return out


def main():
    # --- river network -> topology ---
    rn = gpd.read_file(glob.glob(f"{D}/shapefiles/river_network/*.shp")[0])
    seg_ids = rn["LINKNO"].astype(int).values
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    ds_link = rn["DSLINKNO"].astype(int).values
    downstream_idx = np.array([id_to_idx.get(int(d), -1) for d in ds_link])
    lengths = rn["Length"].astype(float).values
    slopes = rn["Slope"].astype(float).values
    n_seg = len(seg_ids)
    print(f"network: {n_seg} reaches, {(downstream_idx < 0).sum()} outlet(s)")

    # --- SUMMA runoff -> per-reach lateral inflow (m3/s) ---
    ds = xr.open_dataset(f"{D}/simulations/bow_calgary_v1/SUMMA/bow_calgary_v1_timestep.nc")
    runoff = ds["averageRoutedRunoff"].values            # (time, gru) in m/s
    gru = ds["gruId"].values.astype(int)
    attr = xr.open_dataset(f"{D}/settings/SUMMA/attributes.nc")
    ahru = attr["hruId"].values.astype(int)
    area = attr["HRUarea"].values.astype(float)
    area_by_id = {int(h): float(a) for h, a in zip(ahru, area)}
    nt = runoff.shape[0]
    seg_runoff = np.zeros((nt, n_seg))
    for j, g in enumerate(gru):
        idx = id_to_idx.get(int(g))
        if idx is None:
            continue
        seg_runoff[:, idx] = np.clip(runoff[:, j], 0, None) * area_by_id.get(int(g), 0.0)
    print(f"runoff: {nt} steps, total mean inflow {seg_runoff.sum(axis=1).mean():.1f} m3/s")

    # --- baseline (no lakes) ---
    net0 = build_network(seg_ids, downstream_idx, lengths, slopes)
    q_base = route(net0, seg_runoff)

    # --- lake-aware ---
    net1 = build_network(seg_ids, downstream_idx, lengths, slopes)
    with open(f"{D}/settings/dRoute/droute_lakes.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cls = {"inline": raw.get("inline_lakes", {}), "subgrid": raw.get("subgrid_lakes", {})}
    nmod = apply_lake_config_to_network(net1, cls, id_to_idx)
    print(f"applied lakes to {nmod} reaches")
    q_lake = route(net1, seg_runoff)

    # --- outlet (max drainage area) ---
    outlet_idx = int(np.argmax(rn["DSContArea"].astype(float).values))
    outlet_seg = int(seg_ids[outlet_idx])
    time = ds["time"].values
    np.savez(f"{OUTDIR}/calgary_routed.npz",
             time=time, seg_ids=seg_ids, q_base=q_base, q_lake=q_lake,
             outlet_idx=outlet_idx, outlet_seg=outlet_seg)
    print(f"outlet seg {outlet_seg} (idx {outlet_idx}): "
          f"baseline mean={q_base[:, outlet_idx].mean():.1f} peak={q_base[:, outlet_idx].max():.1f} | "
          f"lakes mean={q_lake[:, outlet_idx].mean():.1f} peak={q_lake[:, outlet_idx].max():.1f} m3/s")
    print(f"saved -> {OUTDIR}/calgary_routed.npz")


if __name__ == "__main__":
    main()
