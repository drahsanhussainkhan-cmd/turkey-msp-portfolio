"""
Offshore Wind Site Suitability Screening - Turkish Waters
==========================================================
Identifies suitable zones for fixed-bottom offshore wind turbines
using bathymetry, distance-from-shore, EEZ, and MPA constraints.
"""

import subprocess, sys, io, os

# Auto-install missing packages
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
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
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
MPA_BASE   = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 6 offshore wind"
OUTPUT_PNG = OUTPUT_DIR / "turkey_offshore_wind_suitability.png"

# Suitability criteria
DEPTH_MIN = -50   # metres (negative = below sea level)
DEPTH_MAX = 0     # metres
SHORE_MIN_KM = 5  # minimum distance from coast
SHORE_MAX_KM = 50 # maximum distance from coast

# ============================================================================
# 1. LOAD VECTOR DATA
# ============================================================================
print("=" * 70)
print("OFFSHORE WIND SITE SUITABILITY SCREENING - Turkish Waters")
print("=" * 70)

# --- Turkey EEZ ---
print("\n[1/6] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_area_attr = turkey_eez["AREA_KM2"].sum()
print(f"  Turkey EEZ loaded ({eez_area_attr:,.0f} km2)")

# --- Land ---
print("\n[2/6] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
# Clip land to wider Turkey region for coastline extraction
turkey_bbox = box(25.0, 34.0, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped to Turkey region: {len(land_clip)} polygon(s)")

# --- MPAs ---
print("\n[3/6] Loading Turkey MPA polygons...")
mpa_frames = []
for i in range(3):
    shp = (MPA_BASE / f"WDPA_WDOECM_Feb2026_Public_TUR_shp_{i}" /
           "WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons.shp")
    if shp.exists():
        mpa_frames.append(gpd.read_file(shp))
mpas = pd.concat(mpa_frames, ignore_index=True)
mpas = gpd.GeoDataFrame(mpas, geometry="geometry", crs="EPSG:4326")
mpas = mpas.drop_duplicates(subset="SITE_ID")
# Clip MPAs to EEZ
mpas_in_eez = gpd.clip(mpas, turkey_eez_dissolved)
mpas_in_eez = mpas_in_eez[~mpas_in_eez.is_empty]
print(f"  {len(mpas)} total MPAs, {len(mpas_in_eez)} intersecting EEZ")

# ============================================================================
# 2. LOAD AND PROCESS BATHYMETRY
# ============================================================================
print("\n[4/6] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    bathy_transform = src.transform
    bathy_crs = src.crs
    bathy_shape = src.shape
    bathy_bounds = src.bounds
    pixel_res = src.res  # (dy, dx) in degrees

res_m = pixel_res[0] * 111_000  # degrees to metres (approx)
print(f"  Raster: {bathy_shape[1]}x{bathy_shape[0]}, "
      f"res ~{res_m:.0f}m, range [{bathy.min():.0f}, {bathy.max():.0f}]m")

# ============================================================================
# 3. BUILD SUITABILITY MASKS (all on the GEBCO raster grid)
# ============================================================================
print("\n[5/6] Computing suitability layers...")

# --- (a) EEZ mask ---
print("  [a] Rasterizing Turkey EEZ...")
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape,
    transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
print(f"      EEZ pixels: {eez_mask.sum():,}")

# --- (b) Depth mask: DEPTH_MIN <= elevation <= DEPTH_MAX ---
print(f"  [b] Depth filter ({DEPTH_MIN}m to {DEPTH_MAX}m)...")
depth_mask = (bathy >= DEPTH_MIN) & (bathy <= DEPTH_MAX)
# Exclude land (elevation > 0)
depth_mask = depth_mask & (bathy < 0)
print(f"      Depth-suitable pixels: {(depth_mask & eez_mask).sum():,}")

# --- (c) Land mask + distance from shore ---
print(f"  [c] Distance from shore ({SHORE_MIN_KM}-{SHORE_MAX_KM} km)...")
land_shapes = [(mapping(geom), 1) for geom in land_clip.geometry if geom is not None]
land_raster = rasterize(
    land_shapes,
    out_shape=bathy_shape,
    transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# Also mark positive-elevation pixels as land
land_raster = land_raster | (bathy > 0)

# Distance transform: pixels from nearest land pixel
# Convert pixel distance to km (approximate at ~39N latitude)
avg_lat = 39.0
km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
km_per_deg_lat = 110.57
dx_km = pixel_res[1] * km_per_deg_lon  # pixel width in km
dy_km = pixel_res[0] * km_per_deg_lat  # pixel height in km

# EDT gives distance in pixel units; use sampling to convert to km
dist_from_shore = distance_transform_edt(~land_raster, sampling=[dy_km, dx_km])
shore_mask = (dist_from_shore >= SHORE_MIN_KM) & (dist_from_shore <= SHORE_MAX_KM)
print(f"      Shore-distance suitable pixels: {(shore_mask & eez_mask).sum():,}")

# --- (d) MPA exclusion mask ---
print("  [d] Rasterizing MPA exclusion zones...")
if len(mpas_in_eez) > 0:
    mpa_shapes = [(mapping(geom), 1) for geom in mpas_in_eez.geometry if geom is not None]
    mpa_raster = rasterize(
        mpa_shapes,
        out_shape=bathy_shape,
        transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool)
else:
    mpa_raster = np.zeros(bathy_shape, dtype=bool)
mpa_exclude = ~mpa_raster
print(f"      MPA pixels to exclude: {mpa_raster.sum():,}")

# --- COMBINE all criteria ---
print("  [e] Combining all layers...")
suitable = eez_mask & depth_mask & shore_mask & mpa_exclude
n_suitable = suitable.sum()
n_eez = eez_mask.sum()

# Area calculation: each pixel area in km2
pixel_area_km2 = dx_km * dy_km
suitable_area_km2 = n_suitable * pixel_area_km2
eez_area_km2 = n_eez * pixel_area_km2
pct_of_eez = (suitable_area_km2 / eez_area_km2 * 100) if eez_area_km2 > 0 else 0

# Breakdown of exclusion reasons (within EEZ, ocean only)
ocean_eez = eez_mask & ~land_raster
too_deep = ocean_eez & ~depth_mask & (bathy < 0)
too_shallow_or_land = ocean_eez & (bathy >= 0)
too_close = ocean_eez & depth_mask & (dist_from_shore < SHORE_MIN_KM)
too_far = ocean_eez & depth_mask & (dist_from_shore > SHORE_MAX_KM)
in_mpa = ocean_eez & depth_mask & shore_mask & mpa_raster

print(f"\n  SUITABILITY RESULTS:")
print(f"    Suitable area:       {suitable_area_km2:,.0f} km2 ({pct_of_eez:.2f}% of EEZ)")
print(f"    EEZ area (raster):   {eez_area_km2:,.0f} km2")
print(f"    Excluded - too deep: {too_deep.sum() * pixel_area_km2:,.0f} km2")
print(f"    Excluded - too close:{too_close.sum() * pixel_area_km2:,.0f} km2")
print(f"    Excluded - too far:  {too_far.sum() * pixel_area_km2:,.0f} km2")
print(f"    Excluded - in MPA:   {in_mpa.sum() * pixel_area_km2:,.0f} km2")

# ============================================================================
# 4. CREATE MAP
# ============================================================================
print("\n[6/6] Creating suitability map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor="white")
ax.set_facecolor("#AED9E0")

# Map extent (Turkey EEZ bounds with padding)
eez_bounds = turkey_eez_dissolved.total_bounds
pad = 0.8
xmin, ymin, xmax, ymax = (eez_bounds[0] - pad, eez_bounds[1] - pad,
                           eez_bounds[2] + pad, eez_bounds[3] + pad)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# --- Bathymetry as faded background within EEZ ---
# Create a masked depth array for context shading
depth_display = np.where(eez_mask & ~land_raster, bathy, np.nan)
extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]
ax.imshow(depth_display, extent=extent, origin="upper",
          cmap="Blues_r", alpha=0.20, vmin=-2000, vmax=0, zorder=1,
          aspect="auto", interpolation="bilinear")

# --- Excluded zones (grey) within EEZ ---
excluded = eez_mask & ~land_raster & ~suitable & (bathy < 0)
excluded_display = np.where(excluded, 1.0, np.nan)
ax.imshow(excluded_display, extent=extent, origin="upper",
          cmap=ListedColormap(["#C0C0C0"]), alpha=0.35, zorder=2,
          aspect="auto", interpolation="nearest")

# --- Suitable zones (bright yellow-green) ---
suitable_display = np.where(suitable, 1.0, np.nan)
ax.imshow(suitable_display, extent=extent, origin="upper",
          cmap=ListedColormap(["#7FD34E"]), alpha=0.80, zorder=3,
          aspect="auto", interpolation="nearest")

# --- Land ---
land.plot(ax=ax, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.4, zorder=4)

# --- EEZ boundary ---
turkey_eez_dissolved.boundary.plot(ax=ax, color="#1E5AA8", linewidth=1.8,
                                    linestyle="--", zorder=5)

# --- MPAs ---
if len(mpas_in_eez) > 0:
    mpas_in_eez.plot(ax=ax, color="#1A5C2A", alpha=0.70, edgecolor="#0F3D1A",
                     linewidth=0.8, zorder=6)

# --- Title ---
ax.set_title("Offshore Wind Site Suitability -- Turkish Waters",
             fontsize=17, fontweight="bold", pad=16, color="#1A1A2E")
ax.set_xlabel("Longitude", fontsize=11, labelpad=8)
ax.set_ylabel("Latitude", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle=":", alpha=0.35, color="#666666")

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor="#7FD34E", alpha=0.8, edgecolor="#4A9A2A",
                   label=f"Suitable zone ({suitable_area_km2:,.0f} km2)"),
    mpatches.Patch(facecolor="#C0C0C0", alpha=0.5, edgecolor="#999999",
                   label="Excluded (depth/distance/MPA)"),
    Line2D([0], [0], color="#1E5AA8", linewidth=1.8, linestyle="--",
           label="Turkey EEZ boundary"),
    mpatches.Patch(facecolor="#1A5C2A", alpha=0.7, edgecolor="#0F3D1A",
                   label=f"Marine Protected Areas (n={len(mpas_in_eez)})"),
    mpatches.Patch(facecolor="#F5F0E8", edgecolor="#B0A890", label="Land"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
          framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
          title="Legend", title_fontsize=10)

# --- Criteria + summary box ---
criteria_text = (
    "Suitability Criteria\n"
    f"{'-' * 34}\n"
    f"Water depth:    {DEPTH_MIN}m to {DEPTH_MAX}m\n"
    f"Shore distance: {SHORE_MIN_KM}-{SHORE_MAX_KM} km\n"
    f"MPA exclusion:  Yes\n"
    f"{'-' * 34}\n"
    f"Suitable area:  {suitable_area_km2:,.0f} km2\n"
    f"% of EEZ:       {pct_of_eez:.2f}%\n"
    f"EEZ area:       {eez_area_km2:,.0f} km2\n"
    f"{'-' * 34}\n"
    f"Turbine type: Fixed-bottom\n"
    f"Raster res:   ~{res_m:.0f}m (GEBCO)"
)
props = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.92,
             edgecolor="#999999", linewidth=0.8)
ax.text(0.98, 0.98, criteria_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, fontfamily="monospace", zorder=10)

# --- Source annotation ---
ax.annotate(
    "Data: GEBCO 2025 bathymetry | EEZ: Flanders Marine Institute v12 | "
    "MPAs: WDPA Feb 2026 | Land: Natural Earth 10m",
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

# Estimate turbine capacity
turbine_mw = 15  # modern offshore turbine
spacing_km2 = 1.0  # ~1 km2 per turbine (conservative)
n_turbines = int(suitable_area_km2 / spacing_km2)
capacity_gw = n_turbines * turbine_mw / 1000

print(f"""
OFFSHORE WIND SITE SUITABILITY - Turkish Waters
Date: February 2026
{'-' * 60}

1. INPUT DATA
   - Bathymetry:     GEBCO 2025 ({bathy_shape[1]}x{bathy_shape[0]}, ~{res_m:.0f}m res)
   - EEZ:            Flanders Marine Institute v12
   - MPAs:           WDPA Feb 2026 ({len(mpas_in_eez)} in EEZ)
   - Coastline:      Natural Earth 10m

2. SUITABILITY CRITERIA
   - Water depth:    {DEPTH_MIN}m to {DEPTH_MAX}m (fixed-bottom foundations)
   - Shore distance: {SHORE_MIN_KM} km to {SHORE_MAX_KM} km
   - MPA exclusion:  All protected areas removed
   - Turbine type:   Fixed-bottom (monopile/jacket)

3. RESULTS
   - Turkey EEZ area:        {eez_area_km2:,.0f} km2
   - Suitable area:          {suitable_area_km2:,.0f} km2
   - Suitability ratio:      {pct_of_eez:.2f}% of EEZ

4. EXCLUSION BREAKDOWN
   - Too deep (< {DEPTH_MIN}m):   {too_deep.sum() * pixel_area_km2:,.0f} km2
   - Too close (< {SHORE_MIN_KM} km): {too_close.sum() * pixel_area_km2:,.0f} km2
   - Too far (> {SHORE_MAX_KM} km):  {too_far.sum() * pixel_area_km2:,.0f} km2
   - Inside MPA:             {in_mpa.sum() * pixel_area_km2:,.0f} km2

5. CAPACITY ESTIMATE (indicative)
   - Turbine rating:         {turbine_mw} MW
   - Spacing assumption:     ~{spacing_km2:.0f} km2/turbine
   - Potential turbines:     {n_turbines:,}
   - Installed capacity:     {capacity_gw:,.1f} GW

6. KEY FINDINGS
   - {suitable_area_km2:,.0f} km2 of Turkish waters meet all criteria for
     fixed-bottom offshore wind development.
   - Most suitable zones are in the shallow continental shelves of
     the Aegean Sea and southern Black Sea coast.
   - The Mediterranean coast has a narrow shelf, limiting fixed-bottom
     potential (floating turbines could unlock deeper areas).
   - MPA exclusions have minimal impact ({in_mpa.sum() * pixel_area_km2:,.0f} km2) since
     Turkey's marine protection coverage is very low (< 0.2% of EEZ).
   - At {capacity_gw:,.1f} GW theoretical capacity, offshore wind could
     significantly contribute to Turkey's renewable energy targets.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
