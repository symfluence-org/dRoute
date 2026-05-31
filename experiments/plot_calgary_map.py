# SPDX-License-Identifier: Apache-2.0
"""Bow-at-Calgary domain map: river network + lakes/reservoirs over elevation & landcover.

Two panels share basins (thin outlines) and the river network (line width ~ stream order).
Lakes are colored natural (blue) vs reservoir (red); inline (on-network) lakes get a bold
edge and the largest are labelled. Subgrid (off-network) lakes are drawn faint.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.plot import plotting_extent

from droute.lake_preprocessor import classify_lakes

D = "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Bow_at_Calgary"
OUT = f"{D}/plots/bow_at_calgary_map.png"

rn_path = glob.glob(f"{D}/shapefiles/river_network/*.shp")[0]
rb_path = glob.glob(f"{D}/shapefiles/river_basins/*.shp")[0]
lk_path = f"{D}/data/attributes/lakes/domain_Bow_at_Calgary_hydrolakes.gpkg"
dem_path = f"{D}/data/attributes/elevation/dem/domain_Bow_at_Calgary_elv.tif"
lc_path = f"{D}/data/attributes/landclass/domain_Bow_at_Calgary_land_classes.tif"

# --- vectors (plot in lat/lon) ---
PLOT_CRS = "EPSG:4326"
rivers = gpd.read_file(rn_path).to_crs(PLOT_CRS)
basins = gpd.read_file(rb_path).to_crs(PLOT_CRS)
lakes = gpd.read_file(lk_path).to_crs(PLOT_CRS)

# classify which lakes are inline (on-network) vs subgrid
cls = classify_lakes(lk_path, rn_path, rb_path, segid_field="LINKNO")
inline_ids = {r["hylak_id"] for r in cls["inline"].values()}
lakes["inline"] = lakes["Hylak_id"].isin(inline_ids)
lakes["is_res"] = lakes["Lake_type"].astype(float) >= 2  # 2=reservoir, 3=control

so = rivers["strmOrder"].astype(float) if "strmOrder" in rivers else np.ones(len(rivers))
lw = 0.4 + 0.9 * (so - so.min()) / max(so.max() - so.min(), 1)

bnds = basins.total_bounds  # minx,miny,maxx,maxy
pad = 0.02
ext = (bnds[0] - pad, bnds[2] + pad, bnds[1] - pad, bnds[3] + pad)

# basin union -> mask rasters to the watershed for a clean cartographic shape
wshd = basins.dissolve().geometry.values
wshd_gdf = gpd.GeoDataFrame(geometry=[basins.union_all()], crs=PLOT_CRS)


def clip_raster(path):
    with rasterio.open(path) as src:
        geoms = [g.__geo_interface__ for g in basins.to_crs(src.crs).geometry]
        nod = src.nodata if src.nodata is not None else -9999.0
        arr, transform = rio_mask(src, geoms, crop=True, nodata=nod, filled=True)
        arr = arr[0].astype(float)
        arr[arr == nod] = np.nan
        left = transform.c
        top = transform.f
        right = left + transform.a * arr.shape[1]
        bottom = top + transform.e * arr.shape[0]
        return arr, (left, right, bottom, top)


def draw_vectors(ax, label_lakes=False):
    wshd_gdf.boundary.plot(ax=ax, color="0.1", linewidth=1.4, zorder=8)
    basins.boundary.plot(ax=ax, color="0.3", linewidth=0.4, alpha=0.7)
    rivers.plot(ax=ax, color="#1f4e8c", linewidth=lw, zorder=4)
    sg = lakes[~lakes["inline"]]
    sg.plot(ax=ax, facecolor="#7ab8e8", edgecolor="none", alpha=0.5, zorder=5)
    nat = lakes[lakes["inline"] & ~lakes["is_res"]]
    res = lakes[lakes["inline"] & lakes["is_res"]]
    nat.plot(ax=ax, facecolor="#2e86d6", edgecolor="white", linewidth=0.6, zorder=6)
    res.plot(ax=ax, facecolor="#d63a2e", edgecolor="white", linewidth=0.7, zorder=7)
    if label_lakes:
        big = lakes[lakes["inline"]].sort_values("Lake_area", ascending=False).head(5)
        for _, r in big.iterrows():
            name = (r.get("Lake_name") or "").strip() or f"Hylak {int(r['Hylak_id'])}"
            c = r.geometry.centroid
            ax.annotate(name, (c.x, c.y), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="0.1", weight="bold",
                        path_effects=[], zorder=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

# --- panel 1: elevation hillshade (clipped to watershed) ---
dem, dem_ext = clip_raster(dem_path)
ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), vert_exag=0.0008,
                      dx=90, dy=90)
shaded[~np.isfinite(dem)] = np.nan          # don't paint hillshade outside watershed
terr = plt.cm.terrain.copy(); terr.set_bad(alpha=0.0)
gray = plt.cm.gray.copy(); gray.set_bad(alpha=0.0)
for a in (ax1, ax2):
    a.set_facecolor("white")
ax1.imshow(dem, extent=dem_ext, cmap=terr, origin="upper",
           vmin=np.nanpercentile(dem, 2), vmax=np.nanpercentile(dem, 98), zorder=1)
ax1.imshow(shaded, extent=dem_ext, cmap=gray, alpha=0.35, origin="upper", zorder=2)
draw_vectors(ax1, label_lakes=True)
ax1.set_title("Bow at Calgary — elevation (CopDEM90)", fontsize=12)
sm = plt.cm.ScalarMappable(cmap="terrain",
                           norm=plt.Normalize(np.nanpercentile(dem, 2), np.nanpercentile(dem, 98)))
fig.colorbar(sm, ax=ax1, shrink=0.6, label="Elevation (m)")

# --- panel 2: landcover (clipped to watershed) ---
lc, lc_ext = clip_raster(lc_path)
classes = np.unique(lc[np.isfinite(lc)]).astype(int)
cmap_lc = plt.cm.tab20(np.linspace(0, 1, max(len(classes), 1)))
lc_cmap = ListedColormap(cmap_lc)
bounds = np.append(classes - 0.5, classes[-1] + 0.5) if len(classes) else [0, 1]
norm = BoundaryNorm(bounds, lc_cmap.N) if len(classes) else None
ax2.imshow(lc, extent=lc_ext, cmap=lc_cmap, norm=norm, origin="upper", zorder=1, alpha=0.85)
draw_vectors(ax2, label_lakes=False)
ax2.set_title("Bow at Calgary — land cover (MODIS IGBP)", fontsize=12)

# shared legend for vectors
handles = [
    mpatches.Patch(facecolor="#d63a2e", edgecolor="white", label="Reservoir (inline)"),
    mpatches.Patch(facecolor="#2e86d6", edgecolor="white", label="Natural lake (inline)"),
    mpatches.Patch(facecolor="#7ab8e8", label="Subgrid lake (off-network)"),
    plt.Line2D([0], [0], color="#1f4e8c", lw=2, label="River network"),
    plt.Line2D([0], [0], color="0.25", lw=0.8, label="Sub-basins"),
]
ax1.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)

fig.suptitle(
    f"Bow River at Calgary (WSC 05BH004) — {len(basins)} sub-basins, {len(rivers)} reaches, "
    f"{int(lakes['inline'].sum())} inline lakes/reservoirs, {(~lakes['inline']).sum()} subgrid lakes",
    fontsize=13, weight="bold")

import os
os.makedirs(f"{D}/plots", exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("Wrote", OUT)
