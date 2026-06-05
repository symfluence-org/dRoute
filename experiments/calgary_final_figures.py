# SPDX-License-Identifier: Apache-2.0
"""Final Bow-at-Calgary figures: domain map (with gauges) + nested hydrographs.

(1) domain_map_gauges.png  -- elevation + river network (width ~ stream order) +
    lakes/reservoirs + sub-basins + the 5 nested WSC gauge locations, labelled.
(2) final_hydrographs.png  -- observed vs dRoute baseline (no lakes) vs joint
    gradient-calibrated (lakes + subgrid + Manning) at each nested gauge, with KGE.

Reads the routed series saved by routing_multigauge_adam.py (joint_calib.npz).
"""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, ListedColormap
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

from droute.lake_preprocessor import classify_lakes

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
RES = "/Users/darri.eythorsson/compHydro/code/dRoute/experiments/results_calgary"

# WSC gauge coordinates (from GeoMet hydrometric-stations)
GAUGES = {
    "05BA001": ("Lake Louise", 51.429, -116.189),
    "05BB001": ("Banff", 51.172, -115.572),
    "05BE004": ("Seebe", 51.119, -115.033),
    "05BH005": ("Cochrane", 51.174, -114.466),
    "05BH004": ("Calgary", 51.050, -114.051),
}


def clip_raster(path, basins):
    with rasterio.open(path) as src:
        geoms = [g.__geo_interface__ for g in basins.to_crs(src.crs).geometry]
        nod = src.nodata if src.nodata is not None else -9999.0
        arr, tr = rio_mask(src, geoms, crop=True, nodata=nod, filled=True)
        arr = arr[0].astype(float); arr[arr == nod] = np.nan
        return arr, (tr.c, tr.c + tr.a * arr.shape[1], tr.f + tr.e * arr.shape[0], tr.f)


def make_map():
    rn = gpd.read_file(glob.glob(f"{D}/shapefiles/river_network/*.shp")[0]).to_crs(4326)
    basins = gpd.read_file(glob.glob(f"{D}/shapefiles/river_basins/*.shp")[0]).to_crs(4326)
    lakes = gpd.read_file(f"{D}/data/attributes/lakes/domain_Bow_at_Calgary_hydrolakes.gpkg").to_crs(4326)
    rn_path = glob.glob(f"{D}/shapefiles/river_network/*.shp")[0]
    rb_path = glob.glob(f"{D}/shapefiles/river_basins/*.shp")[0]
    lk_path = f"{D}/data/attributes/lakes/domain_Bow_at_Calgary_hydrolakes.gpkg"
    cls = classify_lakes(lk_path, rn_path, rb_path, segid_field="LINKNO", min_inline_frac=0.30)
    inline_ids = {r["hylak_id"] for r in cls["inline"].values()}
    res_ids = {r["hylak_id"] for r in cls["inline"].values() if r["lake_type"] == 1}
    lakes["inline"] = lakes["Hylak_id"].isin(inline_ids)
    lakes["is_res"] = lakes["Hylak_id"].isin(res_ids)

    so = rn["strmOrder"].astype(float) if "strmOrder" in rn else np.ones(len(rn))
    lw = 0.4 + 1.1 * (so - so.min()) / max(so.max() - so.min(), 1)
    bnds = basins.total_bounds; pad = 0.03
    wshd = gpd.GeoDataFrame(geometry=[basins.union_all()], crs=4326)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("white")
    dem, ext = clip_raster(f"{D}/data/attributes/elevation/dem/domain_Bow_at_Calgary_elv.tif", basins)
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), vert_exag=0.0008, dx=90, dy=90)
    shaded[~np.isfinite(dem)] = np.nan
    # truncate `terrain` to skip its bottom ~25% (the blue "below-sea-level" band) so the
    # low-elevation dry lowlands render green/tan, not as water. Hypsometric land tint.
    land = ListedColormap(plt.cm.terrain(np.linspace(0.25, 1.0, 256)))
    land.set_bad(alpha=0); gray = plt.cm.gray.copy(); gray.set_bad(alpha=0)
    ax.imshow(dem, extent=ext, cmap=land, origin="upper",
              vmin=np.nanpercentile(dem, 2), vmax=np.nanpercentile(dem, 98), zorder=1)
    ax.imshow(shaded, extent=ext, cmap=gray, alpha=0.35, origin="upper", zorder=2)
    wshd.boundary.plot(ax=ax, color="0.1", lw=1.4, zorder=8)
    basins.boundary.plot(ax=ax, color="0.3", lw=0.3, alpha=0.6, zorder=3)
    rn.plot(ax=ax, color="#1f4e8c", linewidth=lw, zorder=4)
    lakes[~lakes["inline"]].plot(ax=ax, facecolor="#7ab8e8", edgecolor="none", alpha=0.5, zorder=5)
    lakes[lakes["inline"] & ~lakes["is_res"]].plot(ax=ax, facecolor="#2e86d6", edgecolor="white", lw=0.6, zorder=6)
    lakes[lakes["inline"] & lakes["is_res"]].plot(ax=ax, facecolor="#d63a2e", edgecolor="white", lw=0.7, zorder=7)

    east_edge = bnds[2]
    for sid, (name, lat, lon) in GAUGES.items():
        ax.plot(lon, lat, marker="*", ms=18, mfc="#ffd400", mec="black", mew=1.0, zorder=10)
        # label to the left for gauges near the east edge (avoid clipping), else to the right
        left = lon > east_edge - 0.25
        ax.annotate(f"{name}\n{sid}", (lon, lat),
                    xytext=(-6 if left else 6, 6), textcoords="offset points",
                    ha="right" if left else "left", fontsize=8, weight="bold", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))
    ax.set_xlim(bnds[0] - pad, bnds[2] + pad); ax.set_ylim(bnds[1] - pad, bnds[3] + pad)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    sm = plt.cm.ScalarMappable(cmap=land,
            norm=plt.Normalize(np.nanpercentile(dem, 2), np.nanpercentile(dem, 98)))
    fig.colorbar(sm, ax=ax, shrink=0.55, label="Elevation (m)")
    handles = [
        plt.Line2D([0], [0], marker="*", ls="", mfc="#ffd400", mec="black", ms=13, label="WSC gauge"),
        mpatches.Patch(fc="#d63a2e", ec="white", label="Reservoir (inline)"),
        mpatches.Patch(fc="#2e86d6", ec="white", label="Natural lake (inline)"),
        mpatches.Patch(fc="#7ab8e8", label="Subgrid lake"),
        plt.Line2D([0], [0], color="#1f4e8c", lw=2, label="River network"),
        plt.Line2D([0], [0], color="0.1", lw=1.4, label="Watershed"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.set_title("Bow River at Calgary (WSC 05BH004) — domain, regulation & nested gauges",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{RES}/domain_map_gauges.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {RES}/domain_map_gauges.png")


def make_hydrographs():
    z = np.load(f"{RES}/joint_calib.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"]); labels = z["gauge_labels"]; stations = z["gauge_stations"]
    qb, qc, obs = z["q_base"], z["q_lake_cal"], z["obs"]
    # plot only the evaluation window (exclude the routing spin-up year)
    ev = dates >= pd.Timestamp("2011-01-01")
    dates, qb, qc, obs = dates[ev], qb[ev], qc[ev], obs[ev]

    def kge(s, o):
        m = np.isfinite(s) & np.isfinite(o); s, o = s[m], o[m]
        if len(s) < 10 or o.std() == 0: return np.nan
        r = np.corrcoef(s, o)[0, 1]
        return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.2 * n), sharex=True)
    for i, ax in enumerate(axes):
        kb, kc = kge(qb[:, i], obs[:, i]), kge(qc[:, i], obs[:, i])
        ax.plot(dates, obs[:, i], "k", lw=1.1, label="WSC obs", zorder=4)
        ax.plot(dates, qb[:, i], color="#d9772b", lw=0.8, alpha=0.85, label=f"baseline, no lakes (KGE {kb:.2f})")
        ax.plot(dates, qc[:, i], color="#2ca02c", lw=0.9, alpha=0.9, label=f"lakes, joint-calibrated (KGE {kc:.2f})")
        ax.set_title(f"{labels[i]}  [{stations[i]}, seg {int(z['gauge_segs'][i])}]  obs mean {np.nanmean(obs[:,i]):.0f} m³/s",
                     fontsize=9)
        ax.set_ylabel("Q (m³/s)", fontsize=8); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Bow at Calgary — nested-gauge hydrographs: dRoute baseline vs joint gradient-calibrated routing\n"
                 f"(SUMMA-calibrated runoff; mean KGE {float(z['kge_base']):.3f} → {float(z['kge_cal']):.3f})",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{RES}/final_hydrographs.png", dpi=160, bbox_inches="tight")
    print(f"saved -> {RES}/final_hydrographs.png")


if __name__ == "__main__":
    make_map()
    make_hydrographs()
