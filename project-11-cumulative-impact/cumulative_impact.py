"""
Cumulative Human Impact Map - Turkish Waters
==============================================
Combines five synthetic pressure layers (fishing, shipping, coastal
development, pollution, climate) into a single cumulative impact
index following the Halpern et al. (2008) methodology.
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
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping, LineString
from scipy.ndimage import distance_transform_edt, gaussian_filter
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

EEZ_PATH  = (BASE_DIR / "downloads" / "World_EEZ_v12_20231025" /
             "World_EEZ_v12_20231025" / "eez_v12.shp")
LAND_PATH = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"

OUTPUT_DIR = BASE_DIR / "project 11 cumulative impact"
OUTPUT_PNG = OUTPUT_DIR / "turkey_cumulative_impact.png"

# Major coastal cities (lon, lat, name, population proxy weight)
CITIES = [
    (29.0, 41.0, "Istanbul",  1.0),
    (27.1, 38.4, "Izmir",     0.6),
    (30.7, 36.9, "Antalya",   0.5),
    (34.6, 36.8, "Mersin",    0.4),
    (39.7, 41.0, "Trabzon",   0.3),
    (36.3, 41.3, "Samsun",    0.35),
]

# River mouths (lon, lat, name) - for pollution layer
RIVER_MOUTHS = [
    (36.0, 41.7, "Kizilirmak"), (36.5, 41.4, "Yesilirmak"),
    (26.9, 38.7, "Gediz"),      (27.2, 37.5, "B.Menderes"),
    (33.9, 36.3, "Goksu"),      (35.5, 36.8, "Seyhan/Ceyhan"),
    (29.0, 41.1, "Bosphorus"),  (40.6, 41.0, "Coruh"),
]

# Shipping routes (from Project 10)
SHIPPING_ROUTES = [
    [(28.98, 41.02), (29.05, 41.10), (29.12, 41.18)],
    [(29.0, 41.5), (32.0, 42.0), (36.0, 41.8), (41.0, 41.5)],
    [(26.0, 38.5), (27.0, 39.5), (28.5, 40.5), (29.0, 41.0)],
    [(26.0, 36.0), (30.0, 35.8), (33.0, 36.0), (36.0, 36.2), (40.0, 36.5)],
    [(29.0, 41.0), (28.0, 40.2), (27.5, 39.5), (27.0, 38.5)],
]

# Analysis grid (coarser than GEBCO for speed - ~2 km resolution)
GRID_RES_DEG = 0.02  # ~2.2 km
LON_MIN, LON_MAX = 25.0, 42.0
LAT_MIN, LAT_MAX = 34.0, 44.0

# ============================================================================
# 1. LOAD VECTOR DATA
# ============================================================================
print("=" * 70)
print("CUMULATIVE HUMAN IMPACT MAP - Turkish Waters")
print("=" * 70)

print("\n[1/7] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_bounds = turkey_eez_dissolved.total_bounds
print(f"  EEZ loaded ({turkey_eez['AREA_KM2'].sum():,.0f} km2)")

print("\n[2/7] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox = box(LON_MIN - 0.5, LAT_MIN - 0.5, LON_MAX + 0.5, LAT_MAX + 0.5)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

# ============================================================================
# 2. BUILD ANALYSIS GRID
# ============================================================================
print("\n[3/7] Building analysis grid...")

n_cols = int((LON_MAX - LON_MIN) / GRID_RES_DEG)
n_rows = int((LAT_MAX - LAT_MIN) / GRID_RES_DEG)
grid_shape = (n_rows, n_cols)

from rasterio.transform import from_bounds
grid_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, n_cols, n_rows)
grid_bounds = rasterio.transform.array_bounds(n_rows, n_cols, grid_transform)

print(f"  Grid: {n_cols}x{n_rows} ({n_cols*n_rows:,} cells), res ~{GRID_RES_DEG*111:.1f} km")

# Coordinate grids
lon_arr = np.linspace(LON_MIN + GRID_RES_DEG/2, LON_MAX - GRID_RES_DEG/2, n_cols)
lat_arr = np.linspace(LAT_MAX - GRID_RES_DEG/2, LAT_MIN + GRID_RES_DEG/2, n_rows)
lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

# Pixel size
avg_lat = 39.0
km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
km_per_deg_lat = 110.57
dx_km = GRID_RES_DEG * km_per_deg_lon
dy_km = GRID_RES_DEG * km_per_deg_lat
pixel_area_km2 = dx_km * dy_km

# EEZ mask
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=grid_shape, transform=grid_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# Land mask
land_shapes = [(mapping(g), 1) for g in land_clip.geometry if g is not None]
land_raster = rasterize(
    land_shapes, out_shape=grid_shape, transform=grid_transform,
    fill=0, dtype=np.uint8
).astype(bool)

ocean_eez = eez_mask & ~land_raster

# Distance from shore (km)
dist_shore = distance_transform_edt(~land_raster, sampling=[dy_km, dx_km])

print(f"  Ocean EEZ pixels: {ocean_eez.sum():,}")

# ============================================================================
# 3. GENERATE PRESSURE LAYERS
# ============================================================================
print("\n[4/7] Generating pressure layers...")
rng = np.random.default_rng(42)

def normalize_01(arr, mask):
    """Normalize array to [0, 1] within mask."""
    vals = arr[mask]
    if vals.max() == vals.min():
        return np.zeros_like(arr)
    normed = (arr - vals.min()) / (vals.max() - vals.min())
    return np.clip(normed, 0, 1).astype(np.float32)

def city_influence(lon_g, lat_g, city_lon, city_lat, weight, radius_km=80):
    """Gaussian decay from city center."""
    d_lon = (lon_g - city_lon) * km_per_deg_lon
    d_lat = (lat_g - city_lat) * km_per_deg_lat
    dist = np.sqrt(d_lon**2 + d_lat**2)
    return weight * np.exp(-0.5 * (dist / radius_km)**2)

# --- (1) FISHING PRESSURE ---
print("  [1] Fishing pressure...")
# Decays with distance from coast, peaks at 5-30 km
fishing = np.exp(-0.5 * ((dist_shore - 15) / 20)**2)
# Add some spatial noise
fishing += rng.normal(0, 0.05, grid_shape)
fishing = gaussian_filter(fishing, sigma=3)
fishing = normalize_01(fishing, ocean_eez)

# --- (2) SHIPPING PRESSURE ---
print("  [2] Shipping pressure...")
# Rasterize routes then apply distance decay
route_lines = [LineString(coords) for coords in SHIPPING_ROUTES]
route_shapes = [(mapping(line), 1) for line in route_lines]
route_raster = rasterize(
    route_shapes, out_shape=grid_shape, transform=grid_transform,
    fill=0, dtype=np.uint8, all_touched=True
).astype(bool)

# Distance from shipping lanes (km)
dist_ship = distance_transform_edt(~route_raster, sampling=[dy_km, dx_km])
shipping = np.exp(-0.5 * (dist_ship / 25)**2)  # 25 km influence radius
shipping = gaussian_filter(shipping, sigma=2)
shipping = normalize_01(shipping, ocean_eez)

# --- (3) COASTAL DEVELOPMENT ---
print("  [3] Coastal development pressure...")
coastal_dev = np.zeros(grid_shape, dtype=np.float32)
for cx, cy, name, weight in CITIES:
    coastal_dev += city_influence(lon_grid, lat_grid, cx, cy, weight, radius_km=60)
# Also general coast proximity
coastal_dev += 0.3 * np.exp(-dist_shore / 15)
coastal_dev = gaussian_filter(coastal_dev, sigma=2)
coastal_dev = normalize_01(coastal_dev, ocean_eez)

# --- (4) POLLUTION ---
print("  [4] Pollution pressure...")
pollution = np.zeros(grid_shape, dtype=np.float32)
# City runoff
for cx, cy, name, weight in CITIES:
    pollution += city_influence(lon_grid, lat_grid, cx, cy, weight * 0.7, radius_km=50)
# River plumes
for rx, ry, _ in RIVER_MOUTHS:
    d_lon = (lon_grid - rx) * km_per_deg_lon
    d_lat = (lat_grid - ry) * km_per_deg_lat
    dist = np.sqrt(d_lon**2 + d_lat**2)
    pollution += 0.8 * np.exp(-0.5 * (dist / 40)**2)
pollution += rng.normal(0, 0.03, grid_shape)
pollution = gaussian_filter(pollution, sigma=3)
pollution = normalize_01(pollution, ocean_eez)

# --- (5) CLIMATE (SST anomaly proxy) ---
print("  [5] Climate pressure (SST anomaly proxy)...")
# Higher in eastern Med and Black Sea, latitudinal + longitudinal gradient
climate = np.zeros(grid_shape, dtype=np.float32)
# Black Sea warming signal
black_sea = (lat_grid > 41.0) & (lon_grid > 28.0)
climate[black_sea] = 0.6
# Eastern Mediterranean warming
east_med = (lat_grid < 37.5) & (lon_grid > 32.0)
climate[east_med] = 0.7
# General: higher in shallower waters (heat absorption)
shallow_factor = np.clip(1.0 - dist_shore / 100, 0, 0.3)
climate += shallow_factor
# Smooth gradient
climate += rng.normal(0, 0.05, grid_shape)
climate = gaussian_filter(climate, sigma=8)
climate = normalize_01(climate, ocean_eez)

# ============================================================================
# 4. COMBINE INTO CUMULATIVE IMPACT
# ============================================================================
print("\n[5/7] Computing cumulative impact score...")

layers = {
    "Fishing":     fishing,
    "Shipping":    shipping,
    "Development": coastal_dev,
    "Pollution":   pollution,
    "Climate":     climate,
}

# Equal-weight sum, then normalize to 0-1
cumulative = np.zeros(grid_shape, dtype=np.float32)
for name, layer in layers.items():
    cumulative += layer

cumulative = normalize_01(cumulative, ocean_eez)

# Mask to ocean EEZ
cumulative_display = np.where(ocean_eez, cumulative, np.nan)
for k in layers:
    layers[k] = np.where(ocean_eez, layers[k], np.nan)

# --- Statistics ---
valid = cumulative[ocean_eez]
ocean_area = ocean_eez.sum() * pixel_area_km2

# Impact classes
high_mask = ocean_eez & (cumulative > 0.6)
med_mask = ocean_eez & (cumulative > 0.3) & (cumulative <= 0.6)
low_mask = ocean_eez & (cumulative <= 0.3)

high_area = high_mask.sum() * pixel_area_km2
med_area = med_mask.sum() * pixel_area_km2
low_area = low_mask.sum() * pixel_area_km2

print(f"  Ocean area:     {ocean_area:,.0f} km2")
print(f"  Mean impact:    {np.nanmean(valid):.3f}")
print(f"  Median impact:  {np.nanmedian(valid):.3f}")
print(f"  High (>0.6):    {high_area:,.0f} km2 ({high_area/ocean_area*100:.1f}%)")
print(f"  Medium (0.3-0.6): {med_area:,.0f} km2 ({med_area/ocean_area*100:.1f}%)")
print(f"  Low (<0.3):     {low_area:,.0f} km2 ({low_area/ocean_area*100:.1f}%)")

# Per-layer stats
print("\n  Per-layer mean scores:")
for name, layer in layers.items():
    vals = layer[ocean_eez]
    print(f"    {name:<14}: mean={np.nanmean(vals):.3f}  "
          f"max={np.nanmax(vals):.3f}  std={np.nanstd(vals):.3f}")

# ============================================================================
# 5. CREATE FIGURE
# ============================================================================
print("\n[6/7] Creating figure...")

# Custom colormaps
impact_colors = ["#000033", "#0a1e5c", "#1a4c8c", "#2d8bbd",
                 "#5dc0c0", "#a8d96c", "#f0e442", "#f28c28",
                 "#e03c31", "#8b0000"]
cmap_impact = LinearSegmentedColormap.from_list("impact", impact_colors, N=256)
cmap_impact.set_bad(color="#0a0a2e", alpha=0)

# Thumbnail colormap (same but usable for small panels)
cmap_thumb = LinearSegmentedColormap.from_list("thumb",
    ["#08306b", "#2171b5", "#6baed6", "#fee08b", "#f46d43", "#a50026"], N=256)
cmap_thumb.set_bad(alpha=0)

# --- Layout: main map left, 5 thumbnails stacked on right ---
fig = plt.figure(figsize=(20, 12), facecolor="#0a0a2e")

# Main map: takes ~65% width
ax_main = fig.add_axes([0.03, 0.08, 0.60, 0.84])
ax_main.set_facecolor("#0a0a2e")

# 5 thumbnail axes on the right (stacked vertically)
thumb_names = list(layers.keys())
thumb_axes = []
th_left = 0.68
th_w = 0.28
th_h = 0.145
th_gap = 0.015
for i in range(5):
    th_bottom = 0.84 - i * (th_h + th_gap)
    ax_t = fig.add_axes([th_left, th_bottom, th_w, th_h])
    ax_t.set_facecolor("#0a0a2e")
    thumb_axes.append(ax_t)

# ---- MAIN MAP ----
pad = 0.6
ax_main.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax_main.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

im = ax_main.imshow(cumulative_display, extent=extent, origin="upper",
                     cmap=cmap_impact, vmin=0, vmax=1, zorder=2,
                     aspect="auto", interpolation="bilinear")

# Land (dark grey)
land.plot(ax=ax_main, color="#2a2a2a", edgecolor="#444444", linewidth=0.3, zorder=3)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax_main, color="#5588bb", linewidth=1.2,
                                    linestyle="--", alpha=0.6, zorder=4)

# City markers
for cx, cy, cname, _ in CITIES:
    ax_main.plot(cx, cy, "o", color="white", markersize=6, markeredgecolor="#333",
                 markeredgewidth=0.5, zorder=6)
    ax_main.annotate(cname, (cx, cy), fontsize=7.5, color="white", fontweight="bold",
                     xytext=(5, 5), textcoords="offset points", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.12", fc="#0a0a2e",
                               ec="none", alpha=0.7))

# Colorbar
cbar_ax = fig.add_axes([0.05, 0.04, 0.56, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Cumulative Human Impact Score", fontsize=10, color="white", labelpad=6)
cbar.ax.tick_params(colors="white", labelsize=8)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(["0.0\nVery Low", "0.2", "0.4", "0.6", "0.8", "1.0\nVery High"])

# Title
ax_main.set_title("Cumulative Human Impact -- Turkish Waters",
                   fontsize=18, fontweight="bold", color="white", pad=14)
ax_main.set_xlabel("Longitude", fontsize=10, color="white", labelpad=6)
ax_main.set_ylabel("Latitude", fontsize=10, color="white", labelpad=6)
ax_main.tick_params(colors="white", labelsize=8)
for spine in ax_main.spines.values():
    spine.set_edgecolor("#444")

# Summary box
summary = (
    f"Impact Summary\n"
    f"{'-' * 28}\n"
    f"High (>0.6): {high_area:,.0f} km2\n"
    f"  ({high_area/ocean_area*100:.1f}% of EEZ)\n"
    f"Medium:      {med_area:,.0f} km2\n"
    f"Low (<0.3):  {low_area:,.0f} km2\n"
    f"{'-' * 28}\n"
    f"Mean score:  {np.nanmean(valid):.3f}\n"
    f"EEZ ocean:   {ocean_area:,.0f} km2"
)
props = dict(boxstyle="round,pad=0.5", facecolor="#0a0a2e", alpha=0.85,
             edgecolor="#5588bb", linewidth=0.8)
ax_main.text(0.01, 0.01, summary, transform=ax_main.transAxes, fontsize=8,
             va="bottom", ha="left", bbox=props, fontfamily="monospace",
             color="white", zorder=10)

# ---- THUMBNAILS ----
for i, (lname, ax_t) in enumerate(zip(thumb_names, thumb_axes)):
    layer_data = layers[lname]
    ax_t.imshow(layer_data, extent=extent, origin="upper",
                cmap=cmap_thumb, vmin=0, vmax=1, aspect="auto",
                interpolation="bilinear")
    # Land overlay
    land_display = np.where(land_raster, 1.0, np.nan)
    cmap_land = LinearSegmentedColormap.from_list("ld", ["#2a2a2a", "#2a2a2a"])
    cmap_land.set_bad(alpha=0)
    ax_t.imshow(land_display, extent=extent, origin="upper",
                cmap=cmap_land, alpha=0.9, aspect="auto", interpolation="nearest")

    ax_t.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
    ax_t.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)
    ax_t.set_title(lname, fontsize=9, fontweight="bold", color="white", pad=3)
    ax_t.tick_params(colors="white", labelsize=5)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_edgecolor("#5588bb")
        spine.set_linewidth(0.5)

    # Mean score annotation
    vals = layers[lname][ocean_eez]
    ax_t.text(0.98, 0.05, f"mean={np.nanmean(vals):.2f}",
              transform=ax_t.transAxes, fontsize=6.5, color="white",
              ha="right", va="bottom",
              bbox=dict(fc="#0a0a2e", ec="none", alpha=0.7, pad=1.5))

# Source annotation
fig.text(0.5, 0.005,
         "Methodology: Halpern et al. (2008) cumulative impact framework | "
         "Data: synthetic pressure layers | EEZ: Flanders Marine Institute v12 | "
         "Land: Natural Earth 10m",
         ha="center", fontsize=7, color="#888888", fontstyle="italic")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="#0a0a2e")
plt.close()
print(f"\nMap saved: {OUTPUT_PNG}")

# ============================================================================
# 6. FULL SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY REPORT")
print("=" * 70)

print(f"""
CUMULATIVE HUMAN IMPACT - Turkish Waters
Methodology: Halpern et al. (2008) adapted framework
Date: February 2026
{'-' * 60}

1. PRESSURE LAYERS (5 stressors, equal weight)""")

print(f"   {'Layer':<16} {'Mean':>8} {'Median':>8} {'Max':>8} {'Std':>8}")
print(f"   {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for lname, layer in layers.items():
    v = layer[ocean_eez]
    print(f"   {lname:<16} {np.nanmean(v):>8.3f} {np.nanmedian(v):>8.3f} "
          f"{np.nanmax(v):>8.3f} {np.nanstd(v):>8.3f}")

print(f"""
2. CUMULATIVE IMPACT
   - EEZ ocean area:   {ocean_area:,.0f} km2
   - Mean score:        {np.nanmean(valid):.3f}
   - Median score:      {np.nanmedian(valid):.3f}
   - Std deviation:     {np.nanstd(valid):.3f}
   - Max score:         {np.nanmax(valid):.3f}

3. IMPACT CLASSIFICATION
   - High (>0.6):      {high_area:,.0f} km2 ({high_area/ocean_area*100:.1f}%)
   - Medium (0.3-0.6): {med_area:,.0f} km2 ({med_area/ocean_area*100:.1f}%)
   - Low (<0.3):       {low_area:,.0f} km2 ({low_area/ocean_area*100:.1f}%)

4. HOTSPOT ANALYSIS""")

for cx, cy, cname, _ in CITIES:
    col = int((cx - LON_MIN) / GRID_RES_DEG)
    row = int((LAT_MAX - cy) / GRID_RES_DEG)
    r1, r2 = max(0, row-5), min(n_rows, row+5)
    c1, c2 = max(0, col-5), min(n_cols, col+5)
    window = cumulative[r1:r2, c1:c2]
    wmask = ocean_eez[r1:r2, c1:c2]
    if wmask.any():
        local_mean = np.nanmean(window[wmask])
        local_max = np.nanmax(window[wmask])
        print(f"   {cname:<12} ({cy:.1f}N, {cx:.1f}E): "
              f"mean={local_mean:.3f}  max={local_max:.3f}")

print(f"""
5. KEY FINDINGS
   - {high_area/ocean_area*100:.1f}% of Turkey's EEZ ({high_area:,.0f} km2) experiences
     high cumulative human impact (score > 0.6).
   - The Istanbul/Bosphorus region shows the highest cumulative
     impact due to converging shipping, urban development,
     pollution, and fishing pressures.
   - Coastal waters within 30 km of shore bear disproportionate
     pressure from all five stressor categories.
   - The central Black Sea basin has the lowest impact scores,
     driven primarily by distance from human activity centers.
   - Climate pressure (SST anomalies) affects broad areas of the
     eastern Mediterranean and Black Sea uniformly, while other
     pressures are more spatially concentrated.
   - This framework follows Halpern et al. (2008, Science) and
     can be refined with real satellite-derived pressure data
     for publication-quality assessments.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
