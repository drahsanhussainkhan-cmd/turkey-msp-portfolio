"""
Integrated Marine Spatial Plan - Turkish Waters 2026
======================================================
Final synthesis map combining all findings from Projects 1-18
into a proposed spatial allocation framework for Turkish waters.
"""

import subprocess, sys, io, os

for pkg_name, import_name in [
    ("geopandas", "geopandas"), ("matplotlib", "matplotlib"),
    ("rasterio", "rasterio"), ("shapely", "shapely"),
    ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
]:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "-q"])

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping, LineString, Point
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp")

GEBCO_PATH = (BASE_DIR / "project 2 msp" /
              "GEBCO_19_Feb_2026_a2f2970736de bathymetry" /
              "gebco_2025_n43.0_s35.0_w25.0_e42.0.tif")
EEZ_PATH   = (BASE_DIR / "downloads" / "World_EEZ_v12_20231025" /
              "World_EEZ_v12_20231025" / "eez_v12.shp")
LAND_PATH  = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"
MPA_BASE   = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 19 msp scenario"
OUTPUT_PNG = OUTPUT_DIR / "turkey_integrated_msp.png"

# Zone codes (priority order: higher = overwrites lower)
Z_NODATA      = 0
Z_OPEN        = 1   # Open / general use
Z_MULTI_USE   = 2   # Multi-use zones
Z_FISHERY     = 3   # Sustainable fisheries
Z_BLUE_CARBON = 4   # Blue carbon conservation
Z_WIND        = 5   # Offshore wind development
Z_SHIPPING    = 6   # Shipping corridors
Z_EXCLUSION   = 7   # High-conflict exclusion
Z_MPA_PROP    = 8   # Proposed MPA expansion
Z_MPA_EXIST   = 9   # Existing MPAs (highest priority)

ZONE_META = {
    Z_MPA_EXIST:   {"name": "Existing MPAs",          "color": "#c0392b", "short": "MPA (existing)"},
    Z_MPA_PROP:    {"name": "Proposed MPA Expansion",  "color": "#e74c3c", "short": "MPA (proposed)"},
    Z_EXCLUSION:   {"name": "Exclusion Zones",         "color": "#2c3e50", "short": "Exclusion"},
    Z_SHIPPING:    {"name": "Shipping Corridors",      "color": "#8e44ad", "short": "Shipping"},
    Z_WIND:        {"name": "Offshore Wind Zones",     "color": "#f39c12", "short": "Wind Energy"},
    Z_BLUE_CARBON: {"name": "Blue Carbon Conservation","color": "#27ae60", "short": "Blue Carbon"},
    Z_FISHERY:     {"name": "Fisheries Management",    "color": "#3498db", "short": "Fisheries"},
    Z_MULTI_USE:   {"name": "Multi-Use Zones",         "color": "#1abc9c", "short": "Multi-Use"},
    Z_OPEN:        {"name": "General Use / Open",      "color": "#d5dbdb", "short": "Open"},
}

# Shipping routes (Project 10)
SHIPPING_ROUTES = [
    ("Bosphorus Strait",       [(28.98, 41.02), (29.05, 41.10), (29.12, 41.18)]),
    ("Black Sea Main",         [(29.0, 41.5), (32.0, 42.0), (36.0, 41.8), (41.0, 41.5)]),
    ("Aegean Main",            [(26.0, 38.5), (27.0, 39.5), (28.5, 40.5), (29.0, 41.0)]),
    ("Mediterranean Main",     [(26.0, 36.0), (30.0, 35.8), (33.0, 36.0),
                                (36.0, 36.2), (40.0, 36.5)]),
    ("Istanbul-Izmir Coastal", [(29.0, 41.0), (28.0, 40.2), (27.5, 39.5), (27.0, 38.5)]),
]
SHIP_BUFFER_KM = 8

# Offshore wind zones (moderate scenario, Project 16)
WIND_ZONES = [
    (26.0, 27.2, 38.0, 39.2),
    (26.5, 27.5, 39.5, 40.2),
    (27.8, 29.0, 40.4, 40.8),
]

# Blue carbon centres (Project 17 - seagrass + saltmarsh)
SEAGRASS_CENTRES = [
    (26.3, 39.0, 20), (26.8, 38.4, 25), (27.0, 37.6, 15),
    (26.5, 40.0, 14), (27.4, 37.0, 18), (27.1, 38.5, 10),
    (26.9, 37.2, 12), (30.3, 36.5, 12), (28.3, 36.6, 10),
]
SALTMARSH_DELTAS = [
    (36.0, 41.72, 10), (36.5, 41.42, 8), (26.9, 38.52, 7),
    (27.2, 37.50, 6), (33.9, 36.30, 5), (35.5, 36.82, 8),
    (28.6, 36.83, 4),
]

# Proposed MPA expansion areas (to reach ~10% coverage)
# Selected based on gap analysis (Project 9) + habitat suitability
PROPOSED_MPA_AREAS = [
    # Aegean - high biodiversity, seagrass
    (26.0, 27.0, 38.5, 39.5, "N. Aegean MPA"),
    (26.5, 27.5, 37.0, 37.8, "Datca-Bozburun MPA"),
    # Black Sea shelf - anchovy habitat
    (31.0, 34.0, 41.5, 42.2, "Central Black Sea MPA"),
    (36.0, 38.0, 41.3, 41.9, "Samsun Shelf MPA"),
    # Mediterranean
    (29.5, 31.0, 36.0, 36.7, "Antalya Bay MPA"),
    (33.5, 35.0, 36.0, 36.5, "Goksu-Mersin MPA"),
    # Deep sea representative
    (28.0, 30.0, 38.0, 39.5, "E. Aegean Deep MPA"),
    (34.0, 37.0, 37.0, 39.0, "Central Med. Deep MPA"),
]

# Fishing hotspots (Project 5)
FISHING_HOTSPOTS = [
    (36.5, 41.6, 0.8, 0.3), (32.0, 41.8, 0.5, 0.2),
    (28.5, 41.5, 0.6, 0.3), (40.5, 41.0, 0.4, 0.2),
    (26.5, 39.5, 0.5, 0.4), (26.0, 38.0, 0.6, 0.5),
    (27.5, 37.0, 0.4, 0.3), (30.5, 36.2, 0.8, 0.3),
    (34.0, 36.0, 0.5, 0.2), (35.5, 36.2, 0.3, 0.2),
    (28.8, 40.7, 0.3, 0.15),
]

# Cities
CITIES = [
    (29.0, 41.0, "Istanbul"),
    (27.1, 38.4, "Izmir"),
    (30.7, 36.9, "Antalya"),
    (34.6, 36.8, "Mersin"),
    (39.7, 41.0, "Trabzon"),
    (36.3, 41.3, "Samsun"),
]

AVG_LAT = 39.0
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(AVG_LAT))
KM_PER_DEG_LAT = 110.57

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("INTEGRATED MARINE SPATIAL PLAN - Turkish Waters 2026")
print("=" * 70)

print("\n[1/8] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_area_km2 = turkey_eez["AREA_KM2"].sum()
print(f"  EEZ loaded ({eez_area_km2:,.0f} km2)")

print("\n[2/8] Loading MPA polygons...")
mpa_frames = []
for i in range(3):
    shp = (MPA_BASE / f"WDPA_WDOECM_Feb2026_Public_TUR_shp_{i}" /
           "WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons.shp")
    if shp.exists():
        mpa_frames.append(gpd.read_file(shp))
        print(f"  Loaded split {i}: {len(mpa_frames[-1])} features")
mpas = pd.concat(mpa_frames, ignore_index=True)
mpas = gpd.GeoDataFrame(mpas, geometry="geometry", crs="EPSG:4326")
dedup_col = "WDPAID" if "WDPAID" in mpas.columns else "SITE_ID"
mpas = mpas.drop_duplicates(subset=dedup_col)
mpas_in_eez = gpd.clip(mpas, turkey_eez_dissolved)
print(f"  {len(mpas_in_eez)} unique MPAs in EEZ")

print("\n[3/8] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox_geom = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox_geom], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

print("\n[4/8] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    transform = src.transform
    ny, nx = bathy.shape
    pixel_res = (abs(transform[4]), abs(transform[0]))

dx_km = pixel_res[1] * KM_PER_DEG_LON
dy_km = pixel_res[0] * KM_PER_DEG_LAT
pixel_area_km2 = dx_km * dy_km
print(f"  GEBCO: {nx}x{ny}, pixel ~{dx_km:.3f} x {dy_km:.3f} km")

lon_min_r = transform[2]
lat_max_r = transform[5]
lons = np.arange(nx) * pixel_res[1] + lon_min_r + pixel_res[1] / 2
lats = lat_max_r - np.arange(ny) * pixel_res[0] - pixel_res[0] / 2
lon_grid, lat_grid = np.meshgrid(lons, lats)
depth = -bathy

# Masks
eez_mask = rasterize(
    [(mapping(eez_geom), 1)], out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8).astype(bool)
land_shapes = [(mapping(g), 1) for g in land_clip.geometry if g is not None]
land_mask = rasterize(
    land_shapes, out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8).astype(bool)
sea_mask = eez_mask & ~land_mask

shore_dist_px = distance_transform_edt(~land_mask)
shore_dist_km = shore_dist_px * dx_km

# ============================================================================
# 2. BUILD ZONE LAYERS
# ============================================================================
print("\n[5/8] Building zone layers...")

zones = np.zeros((ny, nx), dtype=np.uint8)

# --- Step 1: General use everywhere in EEZ ---
zones[sea_mask] = Z_OPEN

# --- Step 2: Fisheries management (shelf areas with fishing pressure) ---
print("  Building fisheries zones...")
fishing_pressure = np.zeros((ny, nx), dtype=np.float32)
for lon_c, lat_c, std_lon, std_lat in FISHING_HOTSPOTS:
    dist2 = ((lon_grid - lon_c) / std_lon)**2 + ((lat_grid - lat_c) / std_lat)**2
    fishing_pressure += np.exp(-dist2 / 2)
fishing_pressure += np.exp(-shore_dist_km / 20) * 0.3
fishing_pressure *= sea_mask
fp_max = fishing_pressure[sea_mask].max()
if fp_max > 0:
    fishing_pressure /= fp_max

fishery_mask = sea_mask & (depth > 0) & (depth <= 500) & (fishing_pressure >= 0.20)
zones[fishery_mask] = Z_FISHERY

# --- Step 3: Multi-use zones (moderate depth, moderate fishing, >50km from shore) ---
print("  Building multi-use zones...")
multi_mask = (
    sea_mask &
    (depth > 50) & (depth <= 1000) &
    (shore_dist_km > 30) &
    (fishing_pressure >= 0.10) & (fishing_pressure < 0.35)
)
zones[multi_mask] = Z_MULTI_USE

# --- Step 4: Blue carbon conservation ---
print("  Building blue carbon zones...")
blue_carbon = np.zeros((ny, nx), dtype=np.float32)
for lon_c, lat_c, rad in SEAGRASS_CENTRES:
    dist_km = np.sqrt(((lon_grid - lon_c) * KM_PER_DEG_LON)**2 +
                      ((lat_grid - lat_c) * KM_PER_DEG_LAT)**2)
    blue_carbon += np.exp(-0.5 * (dist_km / (rad * 0.5))**2)
for lon_c, lat_c, rad in SALTMARSH_DELTAS:
    dist_km = np.sqrt(((lon_grid - lon_c) * KM_PER_DEG_LON)**2 +
                      ((lat_grid - lat_c) * KM_PER_DEG_LAT)**2)
    blue_carbon += 0.8 * np.exp(-0.5 * (dist_km / (rad * 0.4))**2)

bc_mask = (
    sea_mask &
    (depth >= 0) & (depth <= 40) &
    (shore_dist_km <= 15) &
    (blue_carbon >= 0.15)
)
zones[bc_mask] = Z_BLUE_CARBON

# --- Step 5: Offshore wind (moderate scenario) ---
print("  Building wind energy zones...")
wind_mask = np.zeros((ny, nx), dtype=bool)
for lon1, lon2, lat1, lat2 in WIND_ZONES:
    wind_mask |= ((lon_grid >= lon1) & (lon_grid <= lon2) &
                  (lat_grid >= lat1) & (lat_grid <= lat2))
wind_suitable = wind_mask & (bathy >= -50) & (bathy < 0) & (shore_dist_km >= 5) & sea_mask
zones[wind_suitable] = Z_WIND

# --- Step 6: Shipping corridors ---
print("  Building shipping corridors...")
ship_lines = [LineString(coords) for _, coords in SHIPPING_ROUTES]
ship_union = unary_union(ship_lines)
ship_buffer_deg = SHIP_BUFFER_KM / KM_PER_DEG_LON
ship_buffered = ship_union.buffer(ship_buffer_deg)
ship_raster = rasterize(
    [(mapping(ship_buffered), 1)], out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8).astype(bool) & sea_mask
zones[ship_raster] = Z_SHIPPING

# --- Step 7: Exclusion zones (high conflict: shipping + wind or shipping + MPA) ---
print("  Building exclusion zones...")
excl_mask = ship_raster & (wind_suitable | bc_mask)
zones[excl_mask] = Z_EXCLUSION

# --- Step 8: Proposed MPA expansion ---
print("  Building proposed MPA expansion...")
for lon1, lon2, lat1, lat2, name in PROPOSED_MPA_AREAS:
    prop_mask = ((lon_grid >= lon1) & (lon_grid <= lon2) &
                 (lat_grid >= lat1) & (lat_grid <= lat2) & sea_mask)
    # Don't overwrite shipping or wind
    prop_mask = prop_mask & (zones != Z_SHIPPING) & (zones != Z_WIND) & (zones != Z_EXCLUSION)
    zones[prop_mask] = Z_MPA_PROP

# --- Step 9: Existing MPAs (highest priority) ---
print("  Adding existing MPAs...")
mpa_shapes = []
for _, row in mpas_in_eez.iterrows():
    g = row.geometry
    if g is not None and not g.is_empty:
        mpa_shapes.append((mapping(g), 1))
if mpa_shapes:
    mpa_raster = rasterize(
        mpa_shapes, out_shape=(ny, nx), transform=transform,
        fill=0, dtype=np.uint8).astype(bool) & sea_mask
    # Buffer existing MPAs slightly
    mpa_buf = binary_dilation(mpa_raster, structure=np.ones((7, 7))) & sea_mask
    zones[mpa_buf] = Z_MPA_EXIST

print("  Zone classification complete.")

# ============================================================================
# 3. COMPUTE STATISTICS
# ============================================================================
print("\n[6/8] Computing zone statistics...")

total_sea_px = sea_mask.sum()
zone_stats = {}

for zcode in sorted(ZONE_META.keys(), reverse=True):
    z_mask = zones == zcode
    n_px = int(z_mask.sum())
    area_km2 = float(n_px * pixel_area_km2)
    pct = float(n_px / total_sea_px * 100) if total_sea_px > 0 else 0
    mean_depth = float(depth[z_mask].mean()) if n_px > 0 else 0

    zone_stats[zcode] = {
        "name": ZONE_META[zcode]["name"],
        "short": ZONE_META[zcode]["short"],
        "color": ZONE_META[zcode]["color"],
        "area_km2": area_km2,
        "pct_eez": pct,
        "mean_depth": mean_depth,
        "n_pixels": n_px,
    }
    print(f"  {ZONE_META[zcode]['name']:30s}  {area_km2:>10,.0f} km2  ({pct:5.1f}%)")

total_classified = sum(s["area_km2"] for s in zone_stats.values())
mpa_total_pct = zone_stats[Z_MPA_EXIST]["pct_eez"] + zone_stats[Z_MPA_PROP]["pct_eez"]

# ============================================================================
# 4. CREATE FIGURE
# ============================================================================
print("\n[7/8] Creating figure...")

# A2-equivalent: ~594 x 420 mm -> large figsize
fig = plt.figure(figsize=(24, 16), dpi=300, facecolor="white")

# Main map takes most space, conflict matrix + stats in right column
gs_main = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[3, 1],
                           hspace=0.15, wspace=0.12,
                           left=0.03, right=0.97, top=0.90, bottom=0.04)

ax_map = fig.add_subplot(gs_main[0, 0])   # Main map (big)
ax_matrix = fig.add_subplot(gs_main[0, 1]) # Conflict matrix
ax_stats = fig.add_subplot(gs_main[1, 0])  # Stats table
ax_legend = fig.add_subplot(gs_main[1, 1]) # Legend / notes

# ---------- MAIN MAP ----------
print("  Drawing main map...")

# Land
land_clip.plot(ax=ax_map, color="#e8e8e8", edgecolor="#aab7b8", linewidth=0.4)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax_map, color="#00b4d8", linewidth=2.0,
                                   linestyle="-", alpha=0.8)

# Subtle bathymetry contours
bathy_display = np.ma.masked_where(~sea_mask, bathy)
ax_map.contour(lons, lats, bathy_display,
               levels=[-3000, -2000, -1000, -500, -200, -50],
               colors="#bdc3c7", linewidths=0.25, alpha=0.35)

# Zone raster
zone_display = np.ma.masked_where(zones == Z_NODATA, zones)
zone_codes_sorted = [Z_OPEN, Z_MULTI_USE, Z_FISHERY, Z_BLUE_CARBON,
                     Z_WIND, Z_SHIPPING, Z_EXCLUSION, Z_MPA_PROP, Z_MPA_EXIST]
zone_colors_list = [ZONE_META[z]["color"] for z in zone_codes_sorted]
cmap_z = ListedColormap(zone_colors_list)
bounds_z = [z - 0.5 for z in zone_codes_sorted] + [zone_codes_sorted[-1] + 0.5]
norm_z = BoundaryNorm(bounds_z, cmap_z.N)

ax_map.pcolormesh(lons, lats, zone_display, cmap=cmap_z, norm=norm_z,
                  alpha=0.7, shading="auto", rasterized=True)

# Shipping route lines on top
for route_name, coords in SHIPPING_ROUTES:
    rlons = [c[0] for c in coords]
    rlats = [c[1] for c in coords]
    ax_map.plot(rlons, rlats, color="#6c3483", linewidth=1.5, linestyle="--",
                alpha=0.7, zorder=5)

# MPA boundaries
for _, row in mpas_in_eez.iterrows():
    g = row.geometry
    if g is None or g.is_empty:
        continue
    try:
        gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").boundary.plot(
            ax=ax_map, color="#922b21", linewidth=1.2, alpha=0.9, zorder=6)
    except Exception:
        pass

# Cities
for cx, cy, cname in CITIES:
    ax_map.plot(cx, cy, marker="o", markersize=6, color="#2c3e50",
                markeredgecolor="white", markeredgewidth=1.0, zorder=8)
    ax_map.annotate(cname, xy=(cx, cy), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, fontweight="bold",
                    color="#2c3e50", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#bdc3c7", alpha=0.85, linewidth=0.5))

# Basin labels
basin_labels = [
    (34.5, 42.5, "B L A C K   S E A", 11),
    (26.0, 37.8, "A E G E A N\n    S E A", 10),
    (33.0, 35.2, "M E D I T E R R A N E A N   S E A", 10),
    (28.5, 40.55, "MARMARA", 8),
]
for bx, by, btxt, bfs in basin_labels:
    ax_map.text(bx, by, btxt, fontsize=bfs, fontstyle="italic",
                color="#566573", ha="center", va="center", alpha=0.5,
                fontweight="bold", zorder=3)

# Proposed MPA labels
for lon1, lon2, lat1, lat2, name in PROPOSED_MPA_AREAS:
    cx = (lon1 + lon2) / 2
    cy = (lat1 + lat2) / 2
    ax_map.text(cx, cy, name.replace(" MPA", ""), fontsize=5.5,
                color="#922b21", ha="center", va="center", alpha=0.7,
                fontweight="bold", fontstyle="italic", zorder=7)

# Wind zone labels
wind_labels = ["Aegean S.", "Aegean N.", "Marmara"]
for wl, (lon1, lon2, lat1, lat2) in zip(wind_labels, WIND_ZONES):
    ax_map.text((lon1+lon2)/2, (lat1+lat2)/2, f"Wind\n{wl}",
                fontsize=6, color="#7d6608", ha="center", va="center",
                fontweight="bold", alpha=0.8, zorder=7)

# North arrow
arr_x, arr_y = 41.5, 42.5
ax_map.annotate("N", xy=(arr_x, arr_y), fontsize=12, fontweight="bold",
                ha="center", va="bottom", color="#2c3e50")
ax_map.annotate("", xy=(arr_x, arr_y), xytext=(arr_x, arr_y - 0.8),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))

# Scale bar (approximate at 39N)
sb_x, sb_y = 25.5, 35.0
sb_len_deg = 200 / KM_PER_DEG_LON  # 200 km
ax_map.plot([sb_x, sb_x + sb_len_deg], [sb_y, sb_y], color="#2c3e50",
            linewidth=2.5, zorder=9)
ax_map.plot([sb_x, sb_x], [sb_y - 0.1, sb_y + 0.1], color="#2c3e50",
            linewidth=2, zorder=9)
ax_map.plot([sb_x + sb_len_deg, sb_x + sb_len_deg],
            [sb_y - 0.1, sb_y + 0.1], color="#2c3e50", linewidth=2, zorder=9)
ax_map.text(sb_x + sb_len_deg / 2, sb_y - 0.25, "200 km",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#2c3e50", zorder=9)

# Map frame
ax_map.set_xlim(24.5, 42.5)
ax_map.set_ylim(34.0, 43.5)
ax_map.set_xlabel("Longitude (E)", fontsize=10, fontweight="bold")
ax_map.set_ylabel("Latitude (N)", fontsize=10, fontweight="bold")
ax_map.grid(True, alpha=0.15, linestyle="--")
ax_map.tick_params(labelsize=9)
ax_map.set_aspect("equal")

# --- Inset: Regional context ---
ax_inset = inset_axes(ax_map, width="22%", height="28%", loc="upper left",
                      borderpad=1)
# Wider land for context
land_wide = gpd.clip(land, gpd.GeoDataFrame(
    geometry=[box(15, 30, 50, 48)], crs="EPSG:4326"))
land_wide.plot(ax=ax_inset, color="#d5d8dc", edgecolor="#95a5a6", linewidth=0.2)
# Turkey EEZ highlight
turkey_eez_dissolved.plot(ax=ax_inset, color="#85c1e9", alpha=0.4,
                          edgecolor="#00b4d8", linewidth=1)
# Study area box
sa_rect = mpatches.Rectangle((24.5, 34.0), 18, 9.5, linewidth=1.5,
                              edgecolor="#e74c3c", facecolor="none",
                              linestyle="-", zorder=5)
ax_inset.add_patch(sa_rect)
ax_inset.set_xlim(15, 50)
ax_inset.set_ylim(30, 48)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_title("Regional Context", fontsize=7, fontweight="bold", pad=2)
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#bdc3c7")

# ---------- CONFLICT MATRIX ----------
print("  Drawing conflict matrix...")

ax_matrix.set_title("Use Compatibility Matrix", fontsize=13,
                    fontweight="bold", pad=10)

matrix_zones = [Z_MPA_EXIST, Z_MPA_PROP, Z_SHIPPING, Z_WIND,
                Z_FISHERY, Z_BLUE_CARBON, Z_MULTI_USE]
matrix_labels = [ZONE_META[z]["short"] for z in matrix_zones]
n_mz = len(matrix_zones)

# Compatibility: 1=compatible, 0=conditional, -1=incompatible
compat = np.array([
    # MPA_E  MPA_P  SHIP  WIND  FISH  BC    MULTI
    [  1,     1,    -1,   -1,   -1,    1,    0  ],  # MPA existing
    [  1,     1,    -1,   -1,    0,    1,    0  ],  # MPA proposed
    [ -1,    -1,     1,   -1,    0,   -1,    0  ],  # Shipping
    [ -1,    -1,    -1,    1,   -1,   -1,    0  ],  # Wind
    [ -1,     0,     0,   -1,    1,    0,    1  ],  # Fisheries
    [  1,     1,    -1,   -1,    0,    1,    0  ],  # Blue carbon
    [  0,     0,     0,    0,    1,    0,    1  ],  # Multi-use
])

compat_colors = {-1: "#e74c3c", 0: "#f39c12", 1: "#27ae60"}
compat_labels_text = {-1: "X", 0: "?", 1: "OK"}

ax_matrix.set_xlim(-0.5, n_mz - 0.5)
ax_matrix.set_ylim(-0.5, n_mz - 0.5)
ax_matrix.set_aspect("equal")
ax_matrix.invert_yaxis()

for i in range(n_mz):
    for j in range(n_mz):
        val = compat[i, j]
        color = compat_colors[val]
        rect = mpatches.FancyBboxPatch(
            (j - 0.45, i - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.8,
        )
        ax_matrix.add_patch(rect)
        ax_matrix.text(j, i, compat_labels_text[val],
                      ha="center", va="center", fontsize=9,
                      fontweight="bold", color="white")

ax_matrix.set_xticks(range(n_mz))
ax_matrix.set_xticklabels(matrix_labels, fontsize=7.5, fontweight="bold",
                          rotation=45, ha="right")
ax_matrix.set_yticks(range(n_mz))
ax_matrix.set_yticklabels(matrix_labels, fontsize=7.5, fontweight="bold")
ax_matrix.tick_params(length=0)

# Matrix legend
ml_patches = [
    mpatches.Patch(facecolor="#27ae60", label="Compatible (OK)", alpha=0.8),
    mpatches.Patch(facecolor="#f39c12", label="Conditional (?)", alpha=0.8),
    mpatches.Patch(facecolor="#e74c3c", label="Incompatible (X)", alpha=0.8),
]
ax_matrix.legend(handles=ml_patches, loc="lower right",
                 bbox_to_anchor=(1.0, -0.12), fontsize=8,
                 framealpha=0.9, edgecolor="#bdc3c7", ncol=3)

# ---------- STATS TABLE ----------
print("  Drawing statistics table...")
ax_stats.axis("off")

col_labels = ["Zone", "Area (km2)", "% of EEZ", "Mean Depth (m)", "Status"]
row_data = []
display_order = [Z_MPA_EXIST, Z_MPA_PROP, Z_SHIPPING, Z_WIND, Z_BLUE_CARBON,
                 Z_FISHERY, Z_MULTI_USE, Z_EXCLUSION, Z_OPEN]
statuses = {
    Z_MPA_EXIST:   "Active",
    Z_MPA_PROP:    "Proposed",
    Z_SHIPPING:    "Active",
    Z_WIND:        "Proposed",
    Z_BLUE_CARBON: "Proposed",
    Z_FISHERY:     "Active/Proposed",
    Z_MULTI_USE:   "Proposed",
    Z_EXCLUSION:   "Proposed",
    Z_OPEN:        "Default",
}

for zc in display_order:
    s = zone_stats[zc]
    row_data.append([
        s["name"],
        f'{s["area_km2"]:,.0f}',
        f'{s["pct_eez"]:.1f}%',
        f'{s["mean_depth"]:,.0f}',
        statuses.get(zc, ""),
    ])

row_data.append(["TOTAL", f'{total_classified:,.0f}', "100%", "--", ""])

table = ax_stats.table(
    cellText=row_data, colLabels=col_labels,
    loc="center", cellLoc="center",
    colWidths=[0.28, 0.15, 0.12, 0.15, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#0d1b2a")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    cell.set_edgecolor("#34495e")

for i in range(1, len(row_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_edgecolor("#dfe6e9")
        if i <= len(display_order):
            zc = display_order[i - 1]
            if j == 0:
                cell.set_facecolor(ZONE_META[zc]["color"])
                dark_text = zc in [Z_WIND, Z_MULTI_USE, Z_OPEN, Z_BLUE_CARBON]
                cell.set_text_props(fontweight="bold",
                                    color="#2c3e50" if dark_text else "white",
                                    fontsize=8.5)
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
        else:
            cell.set_facecolor("#d5f5e3")
            cell.set_text_props(fontweight="bold")

ax_stats.set_title("Zone Allocation Summary", fontsize=13,
                   fontweight="bold", pad=8)

# ---------- LEGEND / NOTES ----------
print("  Drawing legend panel...")
ax_legend.axis("off")

# Build legend
y_pos = 0.95
ax_legend.text(0.05, y_pos, "MAP LEGEND", fontsize=12, fontweight="bold",
               color="#0d1b2a", transform=ax_legend.transAxes, va="top")
y_pos -= 0.07

for zc in display_order:
    m = ZONE_META[zc]
    rect = mpatches.FancyBboxPatch(
        (0.05, y_pos - 0.025), 0.08, 0.04,
        boxstyle="round,pad=0.005",
        facecolor=m["color"], edgecolor="#bdc3c7",
        linewidth=0.5, alpha=0.8,
        transform=ax_legend.transAxes, clip_on=False,
    )
    ax_legend.add_patch(rect)
    s = zone_stats[zc]
    ax_legend.text(0.16, y_pos, f'{m["name"]} ({s["pct_eez"]:.1f}%)',
                   fontsize=8.5, color="#2c3e50", transform=ax_legend.transAxes,
                   va="center")
    y_pos -= 0.065

y_pos -= 0.03
ax_legend.text(0.05, y_pos, "OTHER FEATURES", fontsize=10, fontweight="bold",
               color="#0d1b2a", transform=ax_legend.transAxes, va="top")
y_pos -= 0.06

other_items = [
    ("#00b4d8", "-", "EEZ Boundary"),
    ("#6c3483", "--", "Shipping Routes"),
    ("#922b21", "-", "MPA Boundaries"),
    ("#2c3e50", "o", "Major Cities"),
]
for color, style, label in other_items:
    ax_legend.text(0.05, y_pos, "--" if style != "o" else "o",
                   fontsize=10, color=color, fontweight="bold",
                   transform=ax_legend.transAxes, va="center",
                   fontfamily="monospace")
    ax_legend.text(0.12, y_pos, label, fontsize=8.5, color="#2c3e50",
                   transform=ax_legend.transAxes, va="center")
    y_pos -= 0.055

# Key metrics
y_pos -= 0.04
ax_legend.text(0.05, y_pos, "KEY METRICS", fontsize=10, fontweight="bold",
               color="#0d1b2a", transform=ax_legend.transAxes, va="top")
y_pos -= 0.06

metrics = [
    f"Total EEZ: {eez_area_km2:,.0f} km2",
    f"MPA Coverage (existing): {zone_stats[Z_MPA_EXIST]['pct_eez']:.1f}%",
    f"MPA Coverage (with proposed): {mpa_total_pct:.1f}%",
    f"Wind Energy Zone: {zone_stats[Z_WIND]['area_km2']:,.0f} km2",
    f"Protected + Managed: {100 - zone_stats[Z_OPEN]['pct_eez']:.1f}%",
]
for mtxt in metrics:
    ax_legend.text(0.07, y_pos, mtxt, fontsize=8, color="#2c3e50",
                   transform=ax_legend.transAxes, va="center")
    y_pos -= 0.05

# ---------- TITLES ----------
fig.suptitle("Integrated Marine Spatial Plan -- Turkish Waters 2026",
             fontsize=22, fontweight="bold", color="#0d1b2a", y=0.96)
fig.text(0.5, 0.935, "Proposed Spatial Allocation Framework",
         ha="center", fontsize=14, fontstyle="italic", color="#566573")

# Footer
fig.text(0.5, 0.01,
         "Sources: GEBCO 2025, WDPA Feb 2026, VLIZ EEZ v12, Natural Earth | "
         "Synthetic layers based on Projects 2-18 | CRS: EPSG:4326 (WGS84)",
         ha="center", fontsize=8, color="#95a5a6")

# ---------- Save ----------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUTPUT_PNG), dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)

file_size = OUTPUT_PNG.stat().st_size / (1024 * 1024)
print(f"\n  Figure saved: {OUTPUT_PNG}")
print(f"  File size: {file_size:.1f} MB")

# ============================================================================
# 5. PRINT RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("INTEGRATED MSP ZONE ALLOCATION SUMMARY")
print("=" * 70)

for zc in display_order:
    s = zone_stats[zc]
    print(f"  {s['name']:30s}  {s['area_km2']:>10,.0f} km2  ({s['pct_eez']:5.1f}%)"
          f"  depth: {s['mean_depth']:>6,.0f}m")

print(f"""
  {'='*60}
  FRAMEWORK OVERVIEW
  {'='*60}
  Total classified:            {total_classified:,.0f} km2
  MPA (existing + proposed):   {zone_stats[Z_MPA_EXIST]['area_km2'] + zone_stats[Z_MPA_PROP]['area_km2']:,.0f} km2 ({mpa_total_pct:.1f}%)
  Renewable energy zones:      {zone_stats[Z_WIND]['area_km2']:,.0f} km2
  Shipping corridors:          {zone_stats[Z_SHIPPING]['area_km2']:,.0f} km2
  Blue carbon conservation:    {zone_stats[Z_BLUE_CARBON]['area_km2']:,.0f} km2
  Fisheries management:        {zone_stats[Z_FISHERY]['area_km2']:,.0f} km2
  Multi-use:                   {zone_stats[Z_MULTI_USE]['area_km2']:,.0f} km2
  Conflict exclusion:          {zone_stats[Z_EXCLUSION]['area_km2']:,.0f} km2
  Open / general use:          {zone_stats[Z_OPEN]['area_km2']:,.0f} km2 ({zone_stats[Z_OPEN]['pct_eez']:.1f}%)

  SPATIAL PLAN HIGHLIGHTS:
  - Proposed MPA network reaches {mpa_total_pct:.1f}% of EEZ
  - 8 new MPA designations proposed across all basins
  - Wind energy moderate scenario (5 GW) allocated in Aegean/Marmara
  - Blue carbon zones protect {zone_stats[Z_BLUE_CARBON]['area_km2']:,.0f} km2 of seagrass/saltmarsh
  - {zone_stats[Z_EXCLUSION]['area_km2']:,.0f} km2 identified as high-conflict exclusion zones
  - {100 - zone_stats[Z_OPEN]['pct_eez']:.1f}% of EEZ under active spatial management
""")

print("=" * 70)
print("DONE - Integrated Marine Spatial Plan complete!")
print("=" * 70)
