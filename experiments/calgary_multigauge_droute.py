# SPDX-License-Identifier: Apache-2.0
"""Bow-at-Calgary (multigauge domain) — route calibrated SUMMA runoff through dRoute.

Routes the async-DDS-calibrated SUMMA runoff through the C++ Muskingum-Cunge network on the
NEW 116-reach multigauge delineation, two ways:
  (a) BASELINE  - plain channels, no lakes
  (b) LAKES     - inline lakes/reservoirs with HydroLAKES-default storage-discharge rating

Regenerates the lake classification for the new topology (classify_lakes), evaluates at the
nested Bow mainstem WSC gauges (KGE), and prints monthly climatology at the outlet to show how
much of the regulation-driven seasonal gap the lake routing closes.
"""
import glob
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import droute
from pathlib import Path
from droute.lake_preprocessor import classify_lakes, apply_lake_config_to_network

D = Path("/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary_multigauge")
RUNOFF = D / "optimization/SUMMA/async-dds_bow_calgary_v1/final_evaluation/bow_calgary_v1_timestep.nc"
OBS = D / "observations/streamflow"
CAL = ("2011-01-01", "2012-12-31")
DT_DAY = 86400.0
# nested Bow mainstem gauges -> new-topology LINKNO (from rebuilt gauge_seg_mapping)
NESTED = [("Lake Louise", "05BA001", 315), ("Banff", "05BB001", 282),
          ("Seebe", "05BE004", 211), ("Cochrane", "05BH005", 438),
          ("Calgary (outlet)", "05BH004", 450)]


def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o); s, o = s[m], o[m]
    if len(s) < 10 or o.std() == 0: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)


def build_net(seg_ids, downstream_idx, lengths, slopes):
    n = len(seg_ids); outlet = n
    up = {i: [] for i in range(n+1)}
    for i in range(n):
        d = downstream_idx[i]; up[d if d >= 0 else outlet].append(i)
    net = droute.Network()
    for jid in range(n+1):
        j = droute.Junction(); j.id = jid; j.upstream_reach_ids = up[jid]; net.add_junction(j)
    for i in range(n):
        r = droute.Reach(); r.id = i; r.length = float(lengths[i]); r.slope = max(float(slopes[i]), 0.001)
        r.manning_n = 0.035; r.upstream_junction_id = i
        d = downstream_idx[i]; r.downstream_junction_id = d if d >= 0 else outlet
        net.add_reach(r)
    net.build_topology()
    return net


def route_daily(net, daily):
    c = droute.RouterConfig(); c.dt = DT_DAY; c.enable_gradients = False
    rt = droute.MuskingumCungeRouter(net, c)
    order = np.asarray(net.topological_order(), dtype=int)
    nt, ns = daily.shape; out = np.zeros((nt, ns))
    for t in range(nt):
        for idx in order:
            rt.set_lateral_inflow(int(idx), float(daily[t, idx]))
        rt.route_timestep()
        out[t, order] = rt.get_all_discharges()
    return out


def main():
    rn = __import__("geopandas").read_file(glob.glob(str(D/"shapefiles/river_network/*.shp"))[0])
    seg_ids = rn["LINKNO"].astype(int).values
    id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
    downstream_idx = np.array([id_to_idx.get(int(d), -1) for d in rn["DSLINKNO"].astype(int).values])
    lengths = rn["Length"].astype(float).values; slopes = rn["Slope"].astype(float).values
    n_seg = len(seg_ids)

    # --- regenerate lake config for the NEW topology ---
    print("=== classifying HydroLAKES for the 116-reach topology ===")
    lakes = classify_lakes(
        lakes_path=glob.glob(str(D/"data/attributes/lakes/*.gpkg"))[0],
        river_network_path=glob.glob(str(D/"shapefiles/river_network/*.shp"))[0],
        river_basins_path=glob.glob(str(D/"shapefiles/river_basins/*.shp"))[0],
    )
    (D/"settings/dRoute").mkdir(parents=True, exist_ok=True)
    with open(D/"settings/dRoute/droute_lakes.yaml", "w") as f:
        yaml.safe_dump({"inline_lakes": lakes["inline"], "subgrid_lakes": lakes["subgrid"]}, f)
    n_res = sum(1 for v in lakes["inline"].values() if int(v.get("lake_type", 0)) == 1)
    print(f"  inline={len(lakes['inline'])} ({n_res} reservoirs), subgrid catchments={len(lakes['subgrid'])}")

    # --- runoff -> per-reach lateral inflow (m3/s), daily ---
    ds = xr.open_dataset(RUNOFF)
    runoff = np.clip(ds["averageRoutedRunoff"].values, 0, None)
    gru = ds["gruId"].values.astype(int); time = pd.to_datetime(ds["time"].values)
    attr = xr.open_dataset(D/"settings/SUMMA/attributes.nc")
    area = {int(h): float(a) for h, a in zip(attr["hruId"].values.astype(int), attr["HRUarea"].values.astype(float))}
    seg_runoff = np.zeros((len(time), n_seg))
    for jx, gid in enumerate(gru):
        i = id_to_idx.get(int(gid))
        if i is not None:
            seg_runoff[:, i] = runoff[:, jx] * area.get(int(gid), 0.0)
    daily = pd.DataFrame(seg_runoff, index=time).resample("D").mean()
    dates = daily.index; arr = daily.values

    # --- route baseline vs lakes ---
    print("=== routing: baseline (no lakes) vs lakes (default rules) ===")
    q_base = route_daily(build_net(seg_ids, downstream_idx, lengths, slopes), arr)
    net_l = build_net(seg_ids, downstream_idx, lengths, slopes)
    apply_lake_config_to_network(net_l, lakes, id_to_idx)
    q_lake = route_daily(net_l, arr)

    # --- evaluate at nested gauges ---
    cal = (dates >= pd.Timestamp(CAL[0])) & (dates <= pd.Timestamp(CAL[1]))
    print(f"\n=== KGE at nested gauges (calibration period {CAL[0]}..{CAL[1]}) ===")
    print(f"  {'gauge':22s} {'baseline':>9s} {'lakes':>9s}")
    rows = []
    for label, sid, link in NESTED:
        idx = id_to_idx[link]
        o = pd.read_csv(OBS/f"wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"].reindex(dates).values
        kb, kl = kge(q_base[cal, idx], o[cal]), kge(q_lake[cal, idx], o[cal])
        rows.append((label, sid, kb, kl))
        print(f"  {label:22s} {kb:>+9.3f} {kl:>+9.3f}")

    # --- monthly climatology at the outlet (regulation gap) ---
    oidx = id_to_idx[450]
    obs_o = pd.read_csv(OBS/"wsc_05BH004_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"].reindex(dates)
    dfm = pd.DataFrame({"obs": obs_o.values, "base": q_base[:, oidx], "lake": q_lake[:, oidx]}, index=dates)
    dfm = dfm[cal]; mc = dfm.groupby(dfm.index.month).mean()
    print("\n=== monthly mean at Calgary outlet (cms), calibration period ===")
    print("  month: " + " ".join(f"{m:>5d}" for m in mc.index))
    for k in ("obs", "base", "lake"):
        print(f"  {k:5s}: " + " ".join(f"{v:5.0f}" for v in mc[k]))
    return rows


if __name__ == "__main__":
    main()
