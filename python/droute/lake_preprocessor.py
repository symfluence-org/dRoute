# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SYMFLUENCE Team <dev@symfluence.org>

"""
dRoute lake pre-processing: HydroLAKES -> spatial classification -> dRoute lake config.

Given a HydroLAKES layer (acquired via ``symfluence data download hydrolakes``), the
river network, and the catchment (river-basin) polygons, this classifies each lake as

  * INLINE  -- the lake intersects the river network; the mainstem flows through it, so
               the containing reach is routed as a lake/reservoir (storage-discharge).
  * SUBGRID -- the lake lies within a catchment but off the river network; it attenuates
               that catchment's local runoff (a fraction of the reach's lateral inflow
               is passed through an aggregate subgrid store).

and writes a ``droute_lakes.yaml`` that the network adapter applies to the dRoute
``Reach`` fields (``is_lake``/``lake_type``/``storage_max``/... and
``has_subgrid_lake``/``subgrid_lake_frac``/...). HydroLAKES attributes initialise the
(then learnable) rating-curve parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

MCM_TO_M3 = 1.0e6      # 10^6 m^3 per million-cubic-metre
KM2_TO_M2 = 1.0e6


def classify_lakes(
    lakes_path: Path,
    river_network_path: Path,
    river_basins_path: Optional[Path] = None,
    segid_field: str = "LINKNO",
    min_inline_frac: float = 0.30,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Classify HydroLAKES into inline (on-network) and subgrid (off-network) lakes.

    Returns {"inline": {segId: {...}}, "subgrid": {segId: {...}}} keyed by river-network
    segment id. Inline params init from Lake_area/Vol_total/Dis_avg/Lake_type; subgrid
    aggregates per catchment (lake-area fraction + summed volume/discharge).
    """
    import geopandas as gpd

    log = logger or logging.getLogger(__name__)
    lakes = gpd.read_file(lakes_path)
    rivers = gpd.read_file(river_network_path)
    # work in an equal-area-ish projected CRS for areas/intersections
    crs = rivers.estimate_utm_crs() if hasattr(rivers, "estimate_utm_crs") else 3857
    lakes = lakes.to_crs(crs)
    rivers = rivers.to_crs(crs)

    def attr(row, name, default=0.0):
        v = row.get(name, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    inline: Dict[int, Dict[str, Any]] = {}
    subgrid_lakes = []  # lakes not genuinely on the network (assigned to catchments below)

    # A lake is INLINE only if a reach genuinely flows THROUGH it: the river line must
    # be substantially contained in the lake polygon. We assign each lake to the reach
    # with the greatest reach-length-inside-lake (NOT max drainage area, which would grab
    # a mainstem reach the lake merely clips near a confluence), and require that the
    # overlap covers at least ``min_inline_frac`` of the reach. Lakes that only marginally
    # touch the network fall through to the subgrid (off-network) store.
    rivers_idx = rivers.reset_index(drop=True)
    sidx = rivers_idx.sindex
    for idx, lk in lakes.iterrows():
        geom = lk.geometry
        area_m2 = attr(lk, "Lake_area") * KM2_TO_M2
        vol_m3 = attr(lk, "Vol_total") * MCM_TO_M3
        depth = attr(lk, "Depth_avg")
        if vol_m3 <= 0 and area_m2 > 0 and depth > 0:
            vol_m3 = area_m2 * depth
        dis = attr(lk, "Dis_avg")
        ltype = int(attr(lk, "Lake_type", 1))         # 1=lake, 2=reservoir, 3=lake control
        droute_type = 0 if ltype == 1 else 1            # 0=natural, 1=reservoir/regulated

        # candidate reaches whose bbox intersects the lake
        cand = list(sidx.query(geom, predicate="intersects")) if geom is not None else []
        best_seg, best_overlap, best_frac = None, 0.0, 0.0
        for ci in cand:
            reach = rivers_idx.geometry.iloc[ci]
            if reach is None or reach.length <= 0:
                continue
            ov = reach.intersection(geom).length
            if ov > best_overlap:
                best_overlap = ov
                best_frac = ov / reach.length
                best_seg = int(rivers_idx.iloc[ci][segid_field])

        if best_seg is not None and best_frac >= min_inline_frac:
            rec = {"lake_type": droute_type, "lake_area": area_m2, "storage_max": vol_m3,
                   "lake_q_ref": dis if dis > 0 else None, "hylak_id": int(attr(lk, "Hylak_id")),
                   "overlap_frac": round(best_frac, 3)}
            cur = inline.get(best_seg)
            # if several lakes flow through one reach, keep the largest
            if cur is None or vol_m3 > cur.get("storage_max", 0):
                inline[best_seg] = rec
        else:
            subgrid_lakes.append({"geometry": geom, "area_m2": area_m2,
                                  "vol_m3": vol_m3, "dis": dis})

    subgrid: Dict[int, Dict[str, Any]] = {}
    if river_basins_path and subgrid_lakes:
        basins = gpd.read_file(river_basins_path).to_crs(crs)
        # basin id field: try common names, else index
        bid = next((f for f in (segid_field, "GRU_ID", "gru_id", "COMID", "hru_id")
                    if f in basins.columns), None)
        sg = gpd.GeoDataFrame(subgrid_lakes, geometry="geometry", crs=crs)
        sj = gpd.sjoin(sg, basins[[bid, "geometry"]] if bid else basins[["geometry"]],
                       predicate="within", how="left")
        for _, b in basins.iterrows():
            seg = int(b[bid]) if bid else int(_)
            inb = sj[sj["index_right"] == _] if "index_right" in sj else sj.iloc[0:0]
            if len(inb) == 0:
                continue
            lake_area = float(inb["area_m2"].sum())
            basin_area = float(b.geometry.area)
            frac = min(lake_area / basin_area, 0.95) if basin_area > 0 else 0.0
            if frac <= 1e-3:
                continue
            subgrid[seg] = {"subgrid_lake_frac": frac,
                            "subgrid_storage_max": float(inb["vol_m3"].sum()),
                            "subgrid_q_ref": float(inb["dis"].sum()) or None,
                            "n_lakes": int(len(inb))}

    log.info(f"Lake classification: {len(inline)} inline (on-network), "
             f"{len(subgrid)} catchments with subgrid lakes "
             f"({len(subgrid_lakes)} off-network lakes)")
    return {"inline": inline, "subgrid": subgrid}


def write_lake_config(classification: Dict[str, Any], out_yaml: Path,
                      logger: Optional[logging.Logger] = None) -> Path:
    """Write the classification to a droute_lakes.yaml the network adapter can apply."""
    import yaml
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({"inline_lakes": classification["inline"],
                        "subgrid_lakes": classification["subgrid"]}, f, default_flow_style=False)
    (logger or logging.getLogger(__name__)).info(f"Wrote dRoute lake config -> {out_yaml}")
    return out_yaml


def apply_lake_config_to_network(net, classification: Dict[str, Any], segid_to_index: Dict[int, int],
                                 logger: Optional[logging.Logger] = None) -> int:
    """Apply a lake classification onto a built dRoute Network (in-memory).

    segid_to_index maps river-network segId -> dRoute reach index. Sets inline reaches as
    lakes (is_lake + rating params init) and channel reaches with subgrid lakes
    (has_subgrid_lake + subgrid params). Returns the number of reaches modified.
    """
    log = logger or logging.getLogger(__name__)
    n = 0
    for seg, rec in classification.get("inline", {}).items():
        i = segid_to_index.get(int(seg))
        if i is None:
            continue
        r = net.get_reach(i)
        r.is_lake = True
        r.lake_type = int(rec.get("lake_type", 0))
        if rec.get("lake_area"):
            r.lake_area = float(rec["lake_area"])
        if rec.get("storage_max"):
            r.storage_max = float(rec["storage_max"])
            r.storage = 0.5 * float(rec["storage_max"])      # start half-full
        if rec.get("lake_q_ref"):
            r.lake_q_ref = float(rec["lake_q_ref"])
        if rec.get("lake_type", 0) == 1:                     # reservoir: small min release
            r.lake_q_min = 0.1 * float(rec.get("lake_q_ref") or 0.0)
        n += 1
    for seg, rec in classification.get("subgrid", {}).items():
        i = segid_to_index.get(int(seg))
        if i is None:
            continue
        r = net.get_reach(i)
        if r.is_lake:
            continue                                          # inline takes precedence
        r.has_subgrid_lake = True
        r.subgrid_lake_frac = float(rec.get("subgrid_lake_frac", 0.0))
        if rec.get("subgrid_storage_max"):
            r.subgrid_storage_max = float(rec["subgrid_storage_max"])
            r.subgrid_storage = 0.5 * float(rec["subgrid_storage_max"])
        if rec.get("subgrid_q_ref"):
            r.subgrid_q_ref = float(rec["subgrid_q_ref"])
        n += 1
    log.info(f"Applied lake config to {n} reaches")
    return n


# ---------------------------------------------------------------------------
# Level-pool lake nodes from a hydrofabric lakes table (e.g. Landslikan
# droute/lakes.csv: lake_id, unit_id, area_km2, reservoir, surface_elev_m,
# max_depth_m_wfd, operated_withdrawals, storage_curve, ...)
# ---------------------------------------------------------------------------

#: Defaults for level-pool lake nodes. All weir parameters are calibration
#: parameters (registered on the AD tape); these are only starting values.
LEVEL_POOL_DEFAULTS: Dict[str, float] = {
    "weir_c": 1.7,                 # broad-crested weir coefficient C_w [m^0.5/s]
    "min_crest_width_m": 5.0,      # crest width B = max(channel width prior, this)
    "weir_exp": 1.5,               # Q = C_w * B * h^weir_exp
    "reservoir_drawdown_m": 10.0,  # max drawdown below crest for reservoirs
    "regulated_drawdown_m": 1.0,   # natural lakes with operated withdrawals
    "initial_stage_m": 0.0,        # initial stage above the crest
}


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "t", "y")


def _float_or_none(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f   # NaN -> None


def load_lakes_table(lakes_csv: Path, logger: Optional[logging.Logger] = None) -> Dict[int, Dict[str, Any]]:
    """Read a lakes table and return one record per outlet unit.

    Several lakes can share an outlet unit (e.g. split polygons of one
    reservoir); they are merged into one storage node: areas are summed,
    ``reservoir`` is OR-ed and operated withdrawals are united.

    Returns ``{unit_id: {"lake_ids", "name", "area_m2", "reservoir",
    "surface_elev_m", "max_depth_m", "operated_withdrawals", "storage_curve"}}``.
    """
    import csv

    log = logger or logging.getLogger(__name__)
    lakes: Dict[int, Dict[str, Any]] = {}
    with open(lakes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            unit = _float_or_none(row.get("unit_id"))
            if unit is None:
                log.warning(f"lake {row.get('lake_id')}: no outlet unit_id, skipped")
                continue
            unit = int(unit)
            area = (_float_or_none(row.get("area_km2")) or 0.0) * KM2_TO_M2
            withdrawals = [w.strip() for w in str(row.get("operated_withdrawals") or "").split(",") if w.strip()]
            rec = lakes.get(unit)
            if rec is None:
                lakes[unit] = {
                    "lake_ids": [row.get("lake_id")],
                    "name": row.get("name") or "",
                    "area_m2": area,
                    "reservoir": _truthy(row.get("reservoir")),
                    "surface_elev_m": _float_or_none(row.get("surface_elev_m")),
                    "max_depth_m": _float_or_none(row.get("max_depth_m_wfd")),
                    "operated_withdrawals": withdrawals,
                    "storage_curve": (row.get("storage_curve") or "").strip(),
                }
            else:
                log.warning(f"lakes {rec['lake_ids']} and {row.get('lake_id')} share outlet unit {unit}; "
                            "merged into one storage node")
                rec["lake_ids"].append(row.get("lake_id"))
                rec["area_m2"] += area
                rec["reservoir"] = rec["reservoir"] or _truthy(row.get("reservoir"))
                depth = _float_or_none(row.get("max_depth_m_wfd"))
                if depth is not None:
                    rec["max_depth_m"] = max(depth, rec["max_depth_m"] or 0.0)
                rec["operated_withdrawals"] += [w for w in withdrawals if w not in rec["operated_withdrawals"]]
    return lakes


def apply_level_pool_lakes(net, lakes: Dict[int, Dict[str, Any]], segid_to_index: Dict[int, int],
                           widths: Optional[Any] = None, params: Optional[Dict[str, float]] = None,
                           logger: Optional[logging.Logger] = None) -> Dict[str, int]:
    """Turn each lake outlet reach of a built dRoute Network into a level-pool node.

    Storage is measured relative to the outlet crest (S = A * stage) and the
    outflow is the weir relation ``Q = lake_weir_coef * (S/A)^lake_weir_exp``
    with ``lake_weir_coef = C_w * max(width_prior, min_crest_width)``.
    Reservoirs (and lakes with operated withdrawals) may be drawn down below
    the crest by their storage withdrawals. ``storage_curve`` is not used yet.

    Returns ``{transfer_id: reach_index}`` for the operated withdrawals listed
    per lake; the caller marks those transfers ``from_storage``.
    """
    import droute

    log = logger or logging.getLogger(__name__)
    p = dict(LEVEL_POOL_DEFAULTS)
    p.update(params or {})
    withdrawal_lakes: Dict[str, int] = {}
    n = 0
    for unit, rec in lakes.items():
        i = segid_to_index.get(int(unit))
        if i is None:
            log.warning(f"lake {rec['lake_ids']} outlet unit {unit} not in topology; skipped")
            continue
        r = net.get_reach(i)
        area = max(float(rec["area_m2"]), 1.0)
        width = float(widths[i]) if widths is not None else 0.0
        crest = max(width, p["min_crest_width_m"])
        r.is_lake = True
        r.lake_model = droute.LAKE_LEVEL_POOL
        r.lake_type = 1 if rec["reservoir"] else 0
        r.lake_area = area
        r.lake_weir_coef = p["weir_c"] * crest
        r.lake_weir_exp = p["weir_exp"]
        r.storage = area * p["initial_stage_m"]
        if rec["reservoir"]:
            drawdown = p["reservoir_drawdown_m"]
            if rec.get("max_depth_m"):
                drawdown = min(drawdown, float(rec["max_depth_m"]))
        elif rec["operated_withdrawals"]:
            drawdown = p["regulated_drawdown_m"]
        else:
            drawdown = 0.0
        r.lake_min_storage = -area * drawdown
        if rec["storage_curve"]:
            log.info(f"lake {rec['lake_ids']}: storage_curve given but not used yet (level-pool A*h)")
        for tid in rec["operated_withdrawals"]:
            withdrawal_lakes[tid] = i
        n += 1
    log.info(f"Level-pool lake nodes: {n} ({sum(1 for r in lakes.values() if r['reservoir'])} reservoirs)")
    return withdrawal_lakes
