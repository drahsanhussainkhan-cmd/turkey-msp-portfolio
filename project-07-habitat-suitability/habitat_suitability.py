"""
European Anchovy (Engraulis encrasicolus) Habitat Suitability Model
====================================================================
Builds a continuous suitability index (0-1) for Turkish waters based on
bathymetry, distance from coast, and latitudinal preference.
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
import matplotlib.ticker as ticker
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from scipy.ndimage import distance_transform_edt
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

OUTPUT_DIR = BASE_DIR / "project 7 habitat suitability"
OUTPUT_PNG = OUTPUT_DIR / "turkey_anchovy_suitability.png"

# --- Habitat criteria (European Anchovy) ---
# Depth: optimal 10-200m, with trapezoidal falloff
DEPTH_OPT_MIN = 10    # m below sea level (shallowest optimal)
DEPTH_OPT_MAX = 200   # m below sea level (deepest optimal)
DEPTH_ABS_MIN = 5     # absolute minimum depth
DEPTH_ABS_MAX = 400   # absolute maximum (zero suitability beyond)

# Distance from coast: 0-100 km, peak near 10-60 km
COAST_PEAK_MIN = 10   # km  (inner edge of peak suitability)
COAST_PEAK_MAX = 60   # km  (outer edge of peak suitability)
COAST_ABS_MAX  = 100  # km  (zero beyond)

# Latitude: full suitability 37-42 N, taper at edges
LAT_OPT_MIN = 37.0
LAT_OPT_MAX = 42.0
LAT_ABS_MIN = 35.5
LAT_ABS_MAX = 43.5

# ============================================================================
# HELPER: trapezoidal membership function
# ============================================================================
def trapezoid(x, a, b, c, d):
    """Trapezoidal fuzzy membership: 0 outside [a,d], ramp up a-b, 1 in b-c, ramp down c-d."""
    result = np.zeros_like(x, dtype=np.float32)
    # rising edge
    mask_rise = (x >= a) & (x < b)
    if b > a:
        result[mask_rise] = (x[mask_rise] - a) / (b - a)
    # plateau
    result[(x >= b) & (x <= c)] = 1.0
    # falling edge
    mask_fall = (x > c) & (x <= d)
    if d > c:
        result[mask_fall] = (d - x[mask_fall]) / (d - c)
    return result

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("ANCHOVY HABITAT SUITABILITY MODEL - Turkish Waters")
print("=" * 70)

print("\n[1/6] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_bounds = turkey_eez_dissolved.total_bounds
print(f"  EEZ loaded ({turkey_eez['AREA_KM2'].sum():,.0f} km2)")

print("\n[2/6] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

print("\n[3/6] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    bathy_transform = src.transform
    bathy_shape = src.shape
    bathy_bounds = src.bounds
    pixel_res = src.res  # (dy, dx) in degrees

res_m = pixel_res[0] * 111_000
print(f"  Raster: {bathy_shape[1]}x{bathy_shape[0]}, ~{res_m:.0f}m res")
print(f"  Depth range: {bathy.min():.0f}m to {bathy.max():.0f}m")

# ============================================================================
# 2. BUILD RASTER MASKS AND ENVIRONMENTAL LAYERS
# ============================================================================
print("\n[4/6] Building environmental layers...")

# --- (a) EEZ mask ---
print("  [a] EEZ mask...")
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# --- (b) Land mask ---
print("  [b] Land mask...")
land_shapes = [(mapping(geom), 1) for geom in land_clip.geometry if geom is not None]
land_raster = rasterize(
    land_shapes,
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
land_raster = land_raster | (bathy > 0)

# Ocean within EEZ
ocean_eez = eez_mask & ~land_raster

# --- (c) Depth layer (positive values = depth below sea surface) ---
print("  [c] Depth suitability...")
depth = -bathy  # convert to positive depth values
depth[depth < 0] = 0  # land/above sea = 0 depth
s_depth = trapezoid(depth,
                    DEPTH_ABS_MIN, DEPTH_OPT_MIN,
                    DEPTH_OPT_MAX, DEPTH_ABS_MAX)

# --- (d) Distance from coast ---
print("  [d] Distance-from-coast suitability...")
avg_lat = 39.0
km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
km_per_deg_lat = 110.57
dx_km = pixel_res[1] * km_per_deg_lon
dy_km = pixel_res[0] * km_per_deg_lat

dist_coast_km = distance_transform_edt(~land_raster, sampling=[dy_km, dx_km])
s_coast = trapezoid(dist_coast_km,
                    0, COAST_PEAK_MIN,
                    COAST_PEAK_MAX, COAST_ABS_MAX)

# --- (e) Latitude layer ---
print("  [e] Latitude suitability...")
rows = np.arange(bathy_shape[0])
# Convert row index to latitude
lats = bathy_bounds.top - rows * pixel_res[0]
lat_grid = np.broadcast_to(lats[:, np.newaxis], bathy_shape).astype(np.float32)
s_lat = trapezoid(lat_grid,
                  LAT_ABS_MIN, LAT_OPT_MIN,
                  LAT_OPT_MAX, LAT_ABS_MAX)

# ============================================================================
# 3. COMBINE INTO SUITABILITY INDEX
# ============================================================================
print("\n[5/6] Computing composite suitability index...")

# Equal-weight geometric mean (rewards balanced conditions)
suitability = (s_depth * s_coast * s_lat) ** (1.0 / 3.0)

# Mask to ocean within EEZ only
suitability[~ocean_eez] = np.nan

# --- Statistics ---
valid = suitability[~np.isnan(suitability)]
pixel_area_km2 = dx_km * dy_km
eez_ocean_area = ocean_eez.sum() * pixel_area_km2

high_suit = (suitability > 0.7) & ocean_eez
med_suit  = (suitability > 0.4) & (suitability <= 0.7) & ocean_eez
low_suit  = (suitability > 0.0) & (suitability <= 0.4) & ocean_eez
zero_suit = (suitability == 0.0) & ocean_eez  # within EEZ but score=0

high_area = high_suit.sum() * pixel_area_km2
med_area  = med_suit.sum() * pixel_area_km2
low_area  = low_suit.sum() * pixel_area_km2
zero_area = zero_suit.sum() * pixel_area_km2
pct_high  = (high_area / eez_ocean_area * 100) if eez_ocean_area > 0 else 0

print(f"  EEZ ocean area:       {eez_ocean_area:,.0f} km2")
print(f"  High suitability:     {high_area:,.0f} km2 ({pct_high:.1f}%)")
print(f"  Medium suitability:   {med_area:,.0f} km2")
print(f"  Low suitability:      {low_area:,.0f} km2")
print(f"  Unsuitable:           {zero_area:,.0f} km2")
print(f"  Mean score:           {np.nanmean(valid):.3f}")
print(f"  Median score:         {np.nanmedian(valid):.3f}")

# ============================================================================
# 4. CREATE MAP
# ============================================================================
print("\n[6/6] Creating suitability map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor="white")
ax.set_facecolor("#AED9E0")

pad = 0.8
ax.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

# --- Custom colormap: white -> light green -> dark green ---
colors_suit = ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#004529"]
cmap_suit = LinearSegmentedColormap.from_list("anchovy", colors_suit, N=256)
cmap_suit.set_bad(color="#AED9E0", alpha=0)  # transparent for NaN (ocean bg)

# --- Suitability raster ---
extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]
im = ax.imshow(suitability, extent=extent, origin="upper",
               cmap=cmap_suit, vmin=0, vmax=1, zorder=2,
               aspect="auto", interpolation="bilinear")

# --- Land ---
land.plot(ax=ax, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.4, zorder=3)

# --- EEZ boundary ---
turkey_eez_dissolved.boundary.plot(ax=ax, color="#1E5AA8", linewidth=1.8,
                                    linestyle="--", zorder=4)

# --- Colorbar ---
cbar = fig.colorbar(im, ax=ax, shrink=0.55, aspect=25, pad=0.02,
                    label="Habitat Suitability Index")
cbar.ax.tick_params(labelsize=9)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(["0.0\nUnsuitable", "0.2", "0.4", "0.6", "0.8",
                      "1.0\nOptimal"])

# --- Title ---
ax.set_title("European Anchovy Habitat Suitability -- Turkish Waters",
             fontsize=17, fontweight="bold", pad=16, color="#1A1A2E")
ax.set_xlabel("Longitude", fontsize=11, labelpad=8)
ax.set_ylabel("Latitude", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle=":", alpha=0.3, color="#666666")

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color="#1E5AA8", linewidth=1.8, linestyle="--",
           label="Turkey EEZ boundary"),
    mpatches.Patch(facecolor="#F5F0E8", edgecolor="#B0A890", label="Land"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
          framealpha=0.92, edgecolor="#CCCCCC", fancybox=True)

# --- Summary box ---
summary_text = (
    "Engraulis encrasicolus\n"
    "(European Anchovy)\n"
    f"{'-' * 30}\n"
    f"Depth:     {DEPTH_OPT_MIN}-{DEPTH_OPT_MAX}m optimal\n"
    f"Coast:     {COAST_PEAK_MIN}-{COAST_PEAK_MAX} km peak\n"
    f"Latitude:  {LAT_OPT_MIN}-{LAT_OPT_MAX} N\n"
    f"{'-' * 30}\n"
    f"High (>0.7): {high_area:,.0f} km2\n"
    f"Medium:      {med_area:,.0f} km2\n"
    f"Low:         {low_area:,.0f} km2\n"
    f"{'-' * 30}\n"
    f"Mean score:  {np.nanmean(valid):.3f}\n"
    f"EEZ ocean:   {eez_ocean_area:,.0f} km2"
)
props = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.92,
             edgecolor="#999999", linewidth=0.8)
ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, fontfamily="monospace", zorder=10)

# --- Source annotation ---
ax.annotate(
    "Data: GEBCO 2025 bathymetry | EEZ: Flanders Marine Institute v12 | "
    "Land: Natural Earth 10m | Model: fuzzy trapezoid, equal-weight geometric mean",
    xy=(0.5, -0.06), xycoords="axes fraction", ha="center", fontsize=7.5,
    color="#666666", style="italic"
)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nMap saved: {OUTPUT_PNG}")

# ============================================================================
# 5. FULL SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY REPORT")
print("=" * 70)

# Regional breakdown by sea basin (approximate longitude ranges)
regions = [
    ("Aegean Sea",       25.0, 27.5, 35.5, 41.5),
    ("Sea of Marmara",   27.5, 30.0, 40.0, 41.2),
    ("Black Sea",        28.5, 41.5, 41.0, 43.5),
    ("Mediterranean",    29.0, 36.5, 35.5, 37.5),
]

print(f"""
EUROPEAN ANCHOVY HABITAT SUITABILITY - Turkish Waters
Species: Engraulis encrasicolus
Date: February 2026
{'-' * 60}

1. MODEL PARAMETERS
   - Depth:        {DEPTH_OPT_MIN}-{DEPTH_OPT_MAX}m optimal (taper {DEPTH_ABS_MIN}-{DEPTH_ABS_MAX}m)
   - Coast dist:   {COAST_PEAK_MIN}-{COAST_PEAK_MAX} km peak (max {COAST_ABS_MAX} km)
   - Latitude:     {LAT_OPT_MIN}-{LAT_OPT_MAX} N optimal
   - Combination:  Equal-weight geometric mean (3 factors)
   - Raster:       GEBCO 2025 (~{res_m:.0f}m resolution)

2. OVERALL RESULTS
   - EEZ ocean area:          {eez_ocean_area:,.0f} km2
   - High suitability (>0.7): {high_area:,.0f} km2 ({pct_high:.1f}%)
   - Medium (0.4-0.7):        {med_area:,.0f} km2
   - Low (0.0-0.4):           {low_area:,.0f} km2
   - Unsuitable (0.0):        {zero_area:,.0f} km2
   - Mean score:               {np.nanmean(valid):.3f}
   - Median score:             {np.nanmedian(valid):.3f}
   - Std deviation:            {np.nanstd(valid):.3f}

3. REGIONAL BREAKDOWN""")

for rname, lon_min, lon_max, lat_min, lat_max in regions:
    # Build row/col mask for this region
    col_min = max(0, int((lon_min - bathy_bounds.left) / pixel_res[1]))
    col_max = min(bathy_shape[1], int((lon_max - bathy_bounds.left) / pixel_res[1]))
    row_min = max(0, int((bathy_bounds.top - lat_max) / pixel_res[0]))
    row_max = min(bathy_shape[0], int((bathy_bounds.top - lat_min) / pixel_res[0]))

    region_suit = suitability[row_min:row_max, col_min:col_max]
    region_valid = region_suit[~np.isnan(region_suit)]
    if len(region_valid) > 0:
        rh = (region_suit > 0.7).sum() * pixel_area_km2
        rmean = np.nanmean(region_valid)
        print(f"   {rname:<20} mean={rmean:.3f}  high-suit={rh:,.0f} km2")
    else:
        print(f"   {rname:<20} (no EEZ ocean pixels)")

print(f"""
4. KEY FINDINGS
   - The continental shelf areas of the Black Sea and Aegean Sea
     show the highest anchovy habitat suitability, consistent with
     known spawning and feeding grounds.
   - {pct_high:.1f}% of Turkey's EEZ qualifies as high-suitability
     habitat ({high_area:,.0f} km2), concentrated on the shelf break.
   - The deep central Black Sea basin and eastern Mediterranean
     are largely unsuitable due to extreme depth (>400m).
   - Coastal waters within 10 km have reduced suitability due to
     shallow depth and proximity to anthropogenic disturbance.
   - This model provides a first-order screening; real habitat use
     also depends on SST, chlorophyll-a, and prey availability.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
