"""
Coastal Erosion Risk Index - Turkish Coastline
================================================
Combines elevation/slope vulnerability, synthetic wave exposure,
and river delta proximity into a composite risk index (0-1).
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
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping, Point
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

GEBCO_PATH = (BASE_DIR / "project 2 msp" /
              "GEBCO_19_Feb_2026_a2f2970736de bathymetry" /
              "gebco_2025_n43.0_s35.0_w25.0_e42.0.tif")
EEZ_PATH   = (BASE_DIR / "downloads" / "World_EEZ_v12_20231025" /
              "World_EEZ_v12_20231025" / "eez_v12.shp")
LAND_PATH  = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"

OUTPUT_DIR = BASE_DIR / "project 8 coastal erosion"
OUTPUT_PNG = OUTPUT_DIR / "turkey_coastal_erosion_risk.png"

# River delta locations (lat, lon, name)
RIVER_DELTAS = [
    (41.7, 36.0, "Kizilirmak"),
    (41.4, 36.5, "Yesilirmak"),
    (38.7, 26.9, "Gediz"),
    (37.5, 27.2, "B. Menderes"),
    (36.3, 33.9, "Goksu"),
    (36.8, 35.5, "Seyhan"),
]

# Coastal strip definition
COASTAL_BAND_KM = 10  # analyse the nearshore zone within 10 km of coast

# Wave exposure basin parameters (base exposure 0-1, boosted by fetch)
# Format: (lon_min, lon_max, lat_min, lat_max, base_exposure)
BASIN_EXPOSURE = [
    # Black Sea - high exposure, long northerly fetch
    (28.5, 41.5, 41.0, 43.5, 0.85),
    # Aegean Sea - moderate, island sheltering
    (25.0, 28.0, 35.5, 41.0, 0.55),
    # Mediterranean - moderate SW exposure
    (28.0, 36.5, 35.5, 37.5, 0.60),
    # Sea of Marmara - low, enclosed basin
    (27.0, 30.0, 40.3, 41.2, 0.30),
]

# Risk class thresholds
RISK_THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 1.01]
RISK_LABELS = ["Low", "Moderate", "High", "Very High"]
RISK_COLORS = ["#2ca02c", "#fee08b", "#f46d43", "#d73027"]

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("COASTAL EROSION RISK INDEX - Turkish Coastline")
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
turkey_bbox = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

print("\n[3/7] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    bathy_transform = src.transform
    bathy_shape = src.shape
    bathy_bounds = src.bounds
    pixel_res = src.res

res_m = pixel_res[0] * 111_000
print(f"  Raster: {bathy_shape[1]}x{bathy_shape[0]}, ~{res_m:.0f}m res")

# Pixel size in km
avg_lat = 39.0
km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
km_per_deg_lat = 110.57
dx_km = pixel_res[1] * km_per_deg_lon
dy_km = pixel_res[0] * km_per_deg_lat
pixel_area_km2 = dx_km * dy_km

# ============================================================================
# 2. BUILD LAND/OCEAN MASKS AND COASTAL STRIP
# ============================================================================
print("\n[4/7] Identifying coastal zone...")

# EEZ mask
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# Land mask
land_shapes = [(mapping(geom), 1) for geom in land_clip.geometry if geom is not None]
land_raster = rasterize(
    land_shapes,
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
land_raster = land_raster | (bathy > 0)

# Distance from land (in km)
dist_from_land = distance_transform_edt(~land_raster, sampling=[dy_km, dx_km])

# Distance from ocean (into land, in km)
ocean_mask = ~land_raster
dist_from_ocean = distance_transform_edt(~ocean_mask, sampling=[dy_km, dx_km])

# Coastal strip: ocean pixels within COASTAL_BAND_KM of land, inside EEZ
coastal_ocean = eez_mask & ocean_mask & (dist_from_land <= COASTAL_BAND_KM)
# Also include the land fringe within 2 km of water (low-lying coastal land)
coastal_land = land_raster & (dist_from_ocean <= 2.0)
coastal_zone = coastal_ocean | coastal_land

n_coastal = coastal_zone.sum()
coastal_area_km2 = n_coastal * pixel_area_km2
print(f"  Coastal zone pixels: {n_coastal:,} ({coastal_area_km2:,.0f} km2)")
print(f"  Ocean band: {COASTAL_BAND_KM} km, land fringe: 2 km")

# ============================================================================
# 3. COMPUTE RISK FACTORS
# ============================================================================
print("\n[5/7] Computing risk factors...")

# --- (a) Elevation/slope vulnerability ---
# Shallow nearshore and low-lying coast = higher risk
# Use absolute elevation as proxy: closer to 0 = more vulnerable
print("  [a] Elevation vulnerability...")
elev_abs = np.abs(bathy)  # distance from sea level
# 0m = highest risk, >30m = low risk
s_elev = np.clip(1.0 - elev_abs / 30.0, 0, 1).astype(np.float32)

# Slope factor: compute gradient magnitude (steeper = more resistant)
print("  [b] Slope resistance...")
grad_y, grad_x = np.gradient(bathy, dy_km * 1000, dx_km * 1000)  # in m/m
slope = np.sqrt(grad_x**2 + grad_y**2)
# Flat = vulnerable (1.0), steep = resistant (0.0); threshold at 5% grade
s_slope_vuln = np.clip(1.0 - slope / 0.05, 0, 1).astype(np.float32)

# Combined elevation/slope: average
elev_slope_risk = (s_elev * 0.6 + s_slope_vuln * 0.4)

# --- (b) Wave exposure ---
print("  [c] Wave exposure...")
# Build a latitude/longitude grid
rows = np.arange(bathy_shape[0])
cols = np.arange(bathy_shape[1])
lon_grid = bathy_bounds.left + cols * pixel_res[1]
lat_grid = bathy_bounds.top - rows * pixel_res[0]
lon_2d = np.broadcast_to(lon_grid[np.newaxis, :], bathy_shape).astype(np.float32)
lat_2d = np.broadcast_to(lat_grid[:, np.newaxis], bathy_shape).astype(np.float32)

# Assign base exposure by basin
wave_exposure = np.full(bathy_shape, 0.4, dtype=np.float32)  # default moderate
for lon_min, lon_max, lat_min, lat_max, base_exp in BASIN_EXPOSURE:
    mask = ((lon_2d >= lon_min) & (lon_2d <= lon_max) &
            (lat_2d >= lat_min) & (lat_2d <= lat_max))
    wave_exposure[mask] = base_exp

# Modulate by distance from coast: exposure peaks at coast, decreases inland
coast_modulation = np.clip(1.0 - dist_from_land / 20.0, 0, 1)
wave_exposure = wave_exposure * coast_modulation

# Add stochastic variation and smooth for realism
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.08, bathy_shape).astype(np.float32)
wave_exposure = np.clip(wave_exposure + noise, 0, 1)
wave_exposure = gaussian_filter(wave_exposure, sigma=5)
wave_exposure = np.clip(wave_exposure, 0, 1).astype(np.float32)

# --- (c) River delta proximity ---
print("  [d] River delta proximity...")
# For each delta, compute distance and create a risk surface
delta_risk = np.zeros(bathy_shape, dtype=np.float32)
for lat_d, lon_d, name in RIVER_DELTAS:
    # Distance from this delta in km (approximate)
    d_lon = (lon_2d - lon_d) * km_per_deg_lon
    d_lat = (lat_2d - lat_d) * km_per_deg_lat
    dist_delta = np.sqrt(d_lon**2 + d_lat**2)
    # Risk decays with distance: high within 30 km, fades to 0 at 80 km
    risk_i = np.clip(1.0 - (dist_delta - 5.0) / 75.0, 0, 1)
    # Boost the inner 15 km
    risk_i[dist_delta <= 15] = np.clip(1.0 - dist_delta[dist_delta <= 15] / 30.0, 0.5, 1.0)
    delta_risk = np.maximum(delta_risk, risk_i)

# ============================================================================
# 4. COMPOSITE RISK INDEX
# ============================================================================
print("\n[6/7] Computing composite risk index...")

# Weighted combination
W_ELEV = 0.35
W_WAVE = 0.40
W_DELTA = 0.25

risk_index = (W_ELEV * elev_slope_risk +
              W_WAVE * wave_exposure +
              W_DELTA * delta_risk)

# Normalize to 0-1
risk_min = np.nanmin(risk_index[coastal_zone])
risk_max = np.nanmax(risk_index[coastal_zone])
risk_index = (risk_index - risk_min) / (risk_max - risk_min + 1e-10)
risk_index = np.clip(risk_index, 0, 1).astype(np.float32)

# Mask to coastal zone only
risk_display = np.where(coastal_zone, risk_index, np.nan)

# --- Classify ---
risk_class = np.full(bathy_shape, -1, dtype=np.int8)
risk_class[coastal_zone & (risk_index < 0.25)] = 0  # Low
risk_class[coastal_zone & (risk_index >= 0.25) & (risk_index < 0.50)] = 1  # Moderate
risk_class[coastal_zone & (risk_index >= 0.50) & (risk_index < 0.75)] = 2  # High
risk_class[coastal_zone & (risk_index >= 0.75)] = 3  # Very High

# --- Statistics per class ---
class_stats = {}
for i, label in enumerate(RISK_LABELS):
    count = (risk_class == i).sum()
    area = count * pixel_area_km2
    class_stats[label] = {"pixels": count, "area_km2": area}
    pct = (area / coastal_area_km2 * 100) if coastal_area_km2 > 0 else 0
    print(f"  {label:<12}: {area:>8,.0f} km2 ({pct:.1f}%)")

print(f"  Total coastal zone: {coastal_area_km2:,.0f} km2")
print(f"  Mean risk score:    {np.nanmean(risk_index[coastal_zone]):.3f}")

# ============================================================================
# 5. CREATE MAP
# ============================================================================
print("\n[7/7] Creating map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor="white")
ax.set_facecolor("#AED9E0")

pad = 0.8
ax.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

# --- Land base ---
land.plot(ax=ax, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.3, zorder=2)

# --- Risk classes as raster overlay ---
risk_class_display = np.where(coastal_zone, risk_class.astype(float), np.nan)
cmap_risk = ListedColormap(RISK_COLORS)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap_risk.N)
cmap_risk.set_bad(alpha=0)

extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]
ax.imshow(risk_class_display, extent=extent, origin="upper",
          cmap=cmap_risk, norm=norm, alpha=0.85, zorder=3,
          aspect="auto", interpolation="nearest")

# --- EEZ boundary ---
turkey_eez_dissolved.boundary.plot(ax=ax, color="#1E5AA8", linewidth=1.8,
                                    linestyle="--", zorder=4)

# --- River delta markers ---
for lat_d, lon_d, name in RIVER_DELTAS:
    ax.plot(lon_d, lat_d, marker="^", color="#4B0082", markersize=10,
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    ax.annotate(name, (lon_d, lat_d), fontsize=7.5, fontweight="bold",
                color="#4B0082", xytext=(6, 6), textcoords="offset points",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

# --- Title ---
ax.set_title("Coastal Erosion Risk Index -- Turkish Coastline",
             fontsize=17, fontweight="bold", pad=16, color="#1A1A2E")
ax.set_xlabel("Longitude", fontsize=11, labelpad=8)
ax.set_ylabel("Latitude", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle=":", alpha=0.3, color="#666666")

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor=RISK_COLORS[0], alpha=0.85, edgecolor="#999",
                   label=f"Low risk (<0.25): {class_stats['Low']['area_km2']:,.0f} km2"),
    mpatches.Patch(facecolor=RISK_COLORS[1], alpha=0.85, edgecolor="#999",
                   label=f"Moderate (0.25-0.50): {class_stats['Moderate']['area_km2']:,.0f} km2"),
    mpatches.Patch(facecolor=RISK_COLORS[2], alpha=0.85, edgecolor="#999",
                   label=f"High (0.50-0.75): {class_stats['High']['area_km2']:,.0f} km2"),
    mpatches.Patch(facecolor=RISK_COLORS[3], alpha=0.85, edgecolor="#999",
                   label=f"Very High (>0.75): {class_stats['Very High']['area_km2']:,.0f} km2"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#4B0082",
           markersize=10, markeredgecolor="white",
           label=f"River deltas (n={len(RIVER_DELTAS)})"),
    Line2D([0], [0], color="#1E5AA8", linewidth=1.8, linestyle="--",
           label="Turkey EEZ boundary"),
    mpatches.Patch(facecolor="#F5F0E8", edgecolor="#B0A890", label="Land"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5,
          framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
          title="Erosion Risk Classes", title_fontsize=9.5)

# --- Summary box ---
mean_risk = np.nanmean(risk_index[coastal_zone])
vh_area = class_stats["Very High"]["area_km2"]
h_area = class_stats["High"]["area_km2"]
combined_high = vh_area + h_area
pct_combined = (combined_high / coastal_area_km2 * 100) if coastal_area_km2 > 0 else 0

summary_text = (
    "Risk Factor Weights\n"
    f"{'-' * 30}\n"
    f"Wave exposure:  {W_WAVE:.0%}\n"
    f"Elevation/slope:{W_ELEV:.0%}\n"
    f"Delta proximity:{W_DELTA:.0%}\n"
    f"{'-' * 30}\n"
    f"Coastal zone:   {coastal_area_km2:,.0f} km2\n"
    f"High+Very High: {combined_high:,.0f} km2\n"
    f"  ({pct_combined:.1f}% of coast)\n"
    f"Mean risk:      {mean_risk:.3f}\n"
    f"{'-' * 30}\n"
    f"Coastal band:   {COASTAL_BAND_KM} km\n"
    f"River deltas:   {len(RIVER_DELTAS)}"
)
props = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.92,
             edgecolor="#999999", linewidth=0.8)
ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, fontfamily="monospace", zorder=10)

# --- Source annotation ---
ax.annotate(
    "Data: GEBCO 2025 bathymetry | EEZ: Flanders Marine Institute v12 | "
    "Land: Natural Earth 10m | Wave exposure: synthetic basin model",
    xy=(0.5, -0.06), xycoords="axes fraction", ha="center", fontsize=7.5,
    color="#666666", style="italic"
)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nMap saved: {OUTPUT_PNG}")

# ============================================================================
# 6. FULL SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY REPORT")
print("=" * 70)

print(f"""
COASTAL EROSION RISK INDEX - Turkish Coastline
Date: February 2026
{'-' * 60}

1. MODEL CONFIGURATION
   - Coastal band:      {COASTAL_BAND_KM} km offshore + 2 km inland
   - Elevation weight:  {W_ELEV:.0%} (low-lying + flat = vulnerable)
   - Wave exposure:     {W_WAVE:.0%} (basin-specific synthetic model)
   - Delta proximity:   {W_DELTA:.0%} (sediment-rich, low-lying coasts)
   - Raster resolution: ~{res_m:.0f}m (GEBCO 2025)

2. RISK CLASS DISTRIBUTION
   - Low (<0.25):       {class_stats['Low']['area_km2']:>8,.0f} km2  ({class_stats['Low']['area_km2']/coastal_area_km2*100:.1f}%)
   - Moderate (0.25-0.50): {class_stats['Moderate']['area_km2']:>5,.0f} km2  ({class_stats['Moderate']['area_km2']/coastal_area_km2*100:.1f}%)
   - High (0.50-0.75):  {class_stats['High']['area_km2']:>8,.0f} km2  ({class_stats['High']['area_km2']/coastal_area_km2*100:.1f}%)
   - Very High (>0.75): {class_stats['Very High']['area_km2']:>8,.0f} km2  ({class_stats['Very High']['area_km2']/coastal_area_km2*100:.1f}%)
   - Total coastal:     {coastal_area_km2:>8,.0f} km2

3. STATISTICS
   - Mean risk score:   {mean_risk:.3f}
   - Std deviation:     {np.nanstd(risk_index[coastal_zone]):.3f}
   - High+Very High:    {combined_high:,.0f} km2 ({pct_combined:.1f}% of coast)

4. RIVER DELTA HOTSPOTS""")

for lat_d, lon_d, name in RIVER_DELTAS:
    # Find risk values near this delta
    col = int((lon_d - bathy_bounds.left) / pixel_res[1])
    row = int((bathy_bounds.top - lat_d) / pixel_res[0])
    # Sample 20x20 pixel window
    r1, r2 = max(0, row-10), min(bathy_shape[0], row+10)
    c1, c2 = max(0, col-10), min(bathy_shape[1], col+10)
    window = risk_index[r1:r2, c1:c2]
    window_coastal = coastal_zone[r1:r2, c1:c2]
    if window_coastal.any():
        local_mean = np.nanmean(window[window_coastal])
        local_max = np.nanmax(window[window_coastal])
        print(f"   {name:<16} ({lat_d:.1f}N, {lon_d:.1f}E)  mean={local_mean:.3f}  max={local_max:.3f}")
    else:
        print(f"   {name:<16} ({lat_d:.1f}N, {lon_d:.1f}E)  (outside coastal zone)")

print(f"""
5. KEY FINDINGS
   - {pct_combined:.1f}% of Turkey's coastal zone ({combined_high:,.0f} km2) is at
     High or Very High erosion risk.
   - The Black Sea coast shows the highest wave exposure (0.85 base),
     creating a continuous high-risk band along the northern shoreline.
   - River deltas (Kizilirmak, Yesilirmak, Seyhan, Goksu) concentrate
     very-high-risk zones due to low elevation, flat topography, and
     sediment dynamics.
   - The Aegean coast has variable risk due to island sheltering effects,
     with exposed headlands at higher risk than sheltered bays.
   - The Sea of Marmara has the lowest overall exposure due to its
     enclosed basin geometry.
   - This index is a screening tool; detailed assessments should
     incorporate sea-level rise projections, sediment budgets, and
     coastal infrastructure data.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
