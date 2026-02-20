"""
Offshore Wind Cumulative Impact Assessment - Turkish Waters
=============================================================
Models three development scenarios (Conservative, Moderate, Ambitious)
and quantifies cumulative impacts on fishing, shipping, habitat,
visual amenity, and construction disturbance.
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
import matplotlib.ticker as mticker
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
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

OUTPUT_DIR = BASE_DIR / "project 16 wind impact"
OUTPUT_PNG = OUTPUT_DIR / "turkey_wind_cumulative_impact.png"

# Wind farm parameters
TURBINE_MW       = 15       # MW per turbine (next-gen)
TURBINE_SPACING  = 1.0      # km2 per turbine (typical for 15 MW class)
CAPACITY_FACTOR  = 0.40     # offshore Turkey ~35-45%

# Scenario definitions
# Each scenario: (name, capacity_mw, zones)
# Zones: list of (lon_min, lon_max, lat_min, lat_max) bounding boxes
SCENARIOS = {
    "Conservative": {
        "capacity_mw": 500,
        "color": "#27ae60",
        "color_light": "#a9dfbf",
        "zones": [
            # Small Aegean cluster near Izmir
            (26.3, 27.0, 38.2, 38.8),
        ],
        "label": "500 MW",
    },
    "Moderate": {
        "capacity_mw": 5_000,
        "color": "#e67e22",
        "color_light": "#f5cba7",
        "zones": [
            # Multiple Aegean sites
            (26.0, 27.2, 38.0, 39.2),
            (26.5, 27.5, 39.5, 40.2),
            # Marmara shelf
            (27.8, 29.0, 40.4, 40.8),
        ],
        "label": "5 GW",
    },
    "Ambitious": {
        "capacity_mw": 41_700,
        "color": "#c0392b",
        "color_light": "#f1948a",
        "zones": [
            # Full suitable area from Project 6
            (26.0, 27.5, 37.5, 40.5),  # Aegean shelf
            (27.5, 29.5, 40.3, 41.0),  # Marmara shelf
            (28.5, 32.0, 41.3, 42.0),  # W Black Sea
            (29.5, 32.0, 36.0, 36.8),  # Antalya shelf
            (33.0, 36.0, 36.0, 36.8),  # Mersin shelf
        ],
        "label": "41.7 GW",
    },
}

# Shipping routes (from Project 10)
SHIPPING_ROUTES = [
    ("Bosphorus Strait",       [(28.98, 41.02), (29.05, 41.10), (29.12, 41.18)]),
    ("Black Sea Main",         [(29.0, 41.5), (32.0, 42.0), (36.0, 41.8), (41.0, 41.5)]),
    ("Aegean Main",            [(26.0, 38.5), (27.0, 39.5), (28.5, 40.5), (29.0, 41.0)]),
    ("Mediterranean Main",     [(26.0, 36.0), (30.0, 35.8), (33.0, 36.0),
                                (36.0, 36.2), (40.0, 36.5)]),
    ("Istanbul-Izmir Coastal", [(29.0, 41.0), (28.0, 40.2), (27.5, 39.5), (27.0, 38.5)]),
]
SHIP_BUFFER_KM = 5

# Fishing hotspots (from Project 5)
FISHING_HOTSPOTS = [
    (36.5, 41.6, 0.8, 0.3),
    (32.0, 41.8, 0.5, 0.2),
    (28.5, 41.5, 0.6, 0.3),
    (40.5, 41.0, 0.4, 0.2),
    (26.5, 39.5, 0.5, 0.4),
    (26.0, 38.0, 0.6, 0.5),
    (27.5, 37.0, 0.4, 0.3),
    (30.5, 36.2, 0.8, 0.3),
    (34.0, 36.0, 0.5, 0.2),
    (35.5, 36.2, 0.3, 0.2),
    (28.8, 40.7, 0.3, 0.15),
]

# Visual impact buffer from shore
VISUAL_BUFFER_KM = 15

# Noise / construction disturbance buffer around each wind zone
NOISE_BUFFER_KM = 10

# Grid parameters (match GEBCO extent)
AVG_LAT = 39.0
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(AVG_LAT))
KM_PER_DEG_LAT = 110.57

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("OFFSHORE WIND CUMULATIVE IMPACT ASSESSMENT - Turkish Waters")
print("=" * 70)

print("\n[1/7] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_bounds = turkey_eez_dissolved.total_bounds
eez_area_km2 = turkey_eez["AREA_KM2"].sum()
print(f"  EEZ loaded ({eez_area_km2:,.0f} km2)")

print("\n[2/7] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox_geom = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox_geom], crs="EPSG:4326"))
land_union = unary_union(land_clip.geometry)
print(f"  Land clipped: {len(land_clip)} polygon(s)")

print("\n[3/7] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    transform = src.transform
    bathy_crs = src.crs
    ny, nx = bathy.shape
    pixel_res = (abs(transform[4]), abs(transform[0]))  # lat_res, lon_res

dx_km = pixel_res[1] * KM_PER_DEG_LON
dy_km = pixel_res[0] * KM_PER_DEG_LAT
pixel_area_km2 = dx_km * dy_km
print(f"  GEBCO: {nx}x{ny}, pixel ~{dx_km:.3f} x {dy_km:.3f} km ({pixel_area_km2:.4f} km2/px)")

# Build raster coordinate arrays
lon_min_r = transform[2]
lat_max_r = transform[5]
lons = np.arange(nx) * pixel_res[1] + lon_min_r + pixel_res[1] / 2
lats = lat_max_r - np.arange(ny) * pixel_res[0] - pixel_res[0] / 2
lon_grid, lat_grid = np.meshgrid(lons, lats)

# EEZ mask
print("  Building EEZ raster mask...")
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool)

# Land mask
land_shapes = [(mapping(g), 1) for g in land_clip.geometry if g is not None]
land_mask = rasterize(
    land_shapes, out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool)

sea_mask = eez_mask & ~land_mask

# Shore distance (km)
print("  Computing shore distance...")
shore_dist_px = distance_transform_edt(~land_mask)
shore_dist_km = shore_dist_px * dx_km

# ============================================================================
# 2. BUILD IMPACT LAYERS
# ============================================================================
print("\n[4/7] Building spatial layers...")

# --- Fishing intensity raster ---
print("  Building fishing intensity field...")
np.random.seed(42)
fishing_intensity = np.zeros((ny, nx), dtype=np.float32)
for lon_c, lat_c, std_lon, std_lat in FISHING_HOTSPOTS:
    dist2 = ((lon_grid - lon_c) / std_lon) ** 2 + ((lat_grid - lat_c) / std_lat) ** 2
    fishing_intensity += np.exp(-dist2 / 2)
fishing_intensity /= fishing_intensity.max()
fishing_intensity *= sea_mask

# --- Shipping corridor raster (5 km buffer) ---
print("  Building shipping corridor mask...")
from shapely.geometry import LineString as LS
ship_lines = []
for _, coords in SHIPPING_ROUTES:
    ship_lines.append(LS(coords))
ship_union = unary_union(ship_lines)
# Buffer in degrees (~5 km)
ship_buffer_deg = SHIP_BUFFER_KM / KM_PER_DEG_LON
ship_buffered = ship_union.buffer(ship_buffer_deg)
ship_mask = rasterize(
    [(mapping(ship_buffered), 1)],
    out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool) & sea_mask

# --- Anchovy habitat suitability (from Project 7 - simplified) ---
print("  Building habitat suitability layer...")
# Depth suitability: 10-200m optimal
depth_vals = -bathy  # positive depths
depth_suit = np.zeros_like(depth_vals)
mask_d = (depth_vals >= 10) & (depth_vals <= 200)
depth_suit[mask_d] = np.where(
    depth_vals[mask_d] <= 100,
    depth_vals[mask_d] / 100,
    1.0 - (depth_vals[mask_d] - 100) / 200,
)
depth_suit = np.clip(depth_suit, 0, 1)

# Latitude suitability: 37-42 optimal
lat_suit = np.zeros_like(lat_grid)
lat_ok = (lat_grid >= 36) & (lat_grid <= 43)
lat_suit[lat_ok] = np.where(
    lat_grid[lat_ok] <= 39.5,
    (lat_grid[lat_ok] - 36) / 3.5,
    1.0 - (lat_grid[lat_ok] - 39.5) / 3.5,
)
lat_suit = np.clip(lat_suit, 0, 1)

habitat_suit = np.sqrt(depth_suit * lat_suit) * sea_mask

# --- Seagrass proxy: shallow coastal (<30m, <10km from shore) ---
seagrass = ((depth_vals > 0) & (depth_vals < 30) &
            (shore_dist_km < 10) & sea_mask).astype(np.float32)

# ============================================================================
# 3. SCENARIO IMPACT CALCULATIONS
# ============================================================================
print("\n[5/7] Computing scenario impacts...")

results = {}

for scenario_name, scenario in SCENARIOS.items():
    print(f"\n  --- {scenario_name} ({scenario['label']}) ---")

    # Build wind farm mask from scenario zones
    wind_mask = np.zeros((ny, nx), dtype=bool)
    for lon_min_z, lon_max_z, lat_min_z, lat_max_z in scenario["zones"]:
        zone_mask = ((lon_grid >= lon_min_z) & (lon_grid <= lon_max_z) &
                     (lat_grid >= lat_min_z) & (lat_grid <= lat_max_z))
        wind_mask |= zone_mask

    # Apply suitability constraints (same as Project 6)
    # Depth: 0-50m
    depth_ok = (bathy >= -50) & (bathy < 0)
    # Shore distance: 5-50 km
    shore_ok = (shore_dist_km >= 5) & (shore_dist_km <= 50)

    wind_suitable = wind_mask & depth_ok & shore_ok & sea_mask
    wind_area_km2 = float(wind_suitable.sum() * pixel_area_km2)

    # Scale to target capacity
    target_km2 = scenario["capacity_mw"] / TURBINE_MW * TURBINE_SPACING
    if wind_area_km2 > 0:
        area_scale = min(1.0, target_km2 / wind_area_km2)
    else:
        area_scale = 0
    effective_area_km2 = wind_area_km2 * area_scale

    n_turbines = int(scenario["capacity_mw"] / TURBINE_MW)
    annual_gwh = scenario["capacity_mw"] * CAPACITY_FACTOR * 8760 / 1000

    print(f"    Suitable area: {wind_area_km2:,.0f} km2")
    print(f"    Effective footprint: {effective_area_km2:,.0f} km2")
    print(f"    Turbines ({TURBINE_MW} MW): {n_turbines:,}")

    # --- Impact 1: Fishing displacement ---
    fishing_in_wind = fishing_intensity[wind_suitable].sum()
    total_fishing = fishing_intensity[sea_mask].sum()
    fishing_displaced_pct = (fishing_in_wind / total_fishing * 100) * area_scale if total_fishing > 0 else 0
    fishing_displaced_km2 = effective_area_km2 * (fishing_in_wind / wind_suitable.sum()) if wind_suitable.sum() > 0 else 0
    # Scale fishing displacement by intensity
    fishing_displaced_km2 = min(fishing_displaced_km2, effective_area_km2)
    print(f"    Fishing displaced: {fishing_displaced_km2:,.0f} km2 ({fishing_displaced_pct:.1f}% of total)")

    # --- Impact 2: Shipping conflict ---
    ship_overlap = (wind_suitable & ship_mask).sum() * pixel_area_km2 * area_scale
    print(f"    Shipping conflict: {ship_overlap:,.0f} km2")

    # --- Impact 3: Habitat affected ---
    habitat_affected = habitat_suit[wind_suitable].sum() * pixel_area_km2 * area_scale
    habitat_affected = min(habitat_affected, effective_area_km2 * 0.8)
    seagrass_affected = seagrass[wind_suitable].sum() * pixel_area_km2 * area_scale
    print(f"    Habitat affected: {habitat_affected:,.0f} km2")
    print(f"    Seagrass affected: {seagrass_affected:,.0f} km2")

    # --- Impact 4: Visual impact (within 15 km of shore) ---
    visual_zone = wind_suitable & (shore_dist_km <= VISUAL_BUFFER_KM)
    visual_km2 = visual_zone.sum() * pixel_area_km2 * area_scale
    print(f"    Visual impact zone: {visual_km2:,.0f} km2")

    # --- Impact 5: Noise/construction disturbance ---
    # Buffer the wind zone by NOISE_BUFFER_KM
    noise_buffer_px = NOISE_BUFFER_KM / dx_km
    from scipy.ndimage import binary_dilation
    struct_size = max(3, int(noise_buffer_px * 2 + 1))
    struct = np.ones((struct_size, struct_size), dtype=bool)
    wind_dilated = binary_dilation(wind_suitable, structure=struct)
    noise_zone = wind_dilated & sea_mask & ~wind_suitable
    noise_km2 = noise_zone.sum() * pixel_area_km2 * area_scale
    print(f"    Noise disturbance zone: {noise_km2:,.0f} km2")

    # Total cumulative impact footprint
    total_impact_km2 = effective_area_km2 + noise_km2
    total_impact_pct = total_impact_km2 / eez_area_km2 * 100

    results[scenario_name] = {
        "capacity_mw": scenario["capacity_mw"],
        "label": scenario["label"],
        "n_turbines": n_turbines,
        "annual_gwh": annual_gwh,
        "footprint_km2": effective_area_km2,
        "fishing_displaced_km2": fishing_displaced_km2,
        "fishing_displaced_pct": fishing_displaced_pct,
        "shipping_conflict_km2": ship_overlap,
        "habitat_affected_km2": habitat_affected,
        "seagrass_affected_km2": seagrass_affected,
        "visual_impact_km2": visual_km2,
        "noise_disturbance_km2": noise_km2,
        "total_impact_km2": total_impact_km2,
        "total_impact_pct": total_impact_pct,
        "wind_mask": wind_suitable,
        "area_scale": area_scale,
        "color": scenario["color"],
        "color_light": scenario["color_light"],
    }

# ============================================================================
# 4. CREATE FIGURE
# ============================================================================
print("\n\n[6/7] Creating figure...")

fig = plt.figure(figsize=(20, 12), dpi=300, facecolor="white")

# Layout: 2x2 grid
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                      left=0.05, right=0.97, top=0.90, bottom=0.06)

ax_map   = fig.add_subplot(gs[0, 0])  # Scenario map
ax_bar   = fig.add_subplot(gs[0, 1])  # Stacked bar chart
ax_radar = fig.add_subplot(gs[1, 0], polar=True)  # Radar chart
ax_table = fig.add_subplot(gs[1, 1])  # Summary table

# ---------- Panel A: Scenario Footprint Map ----------
print("  Drawing scenario map...")

# Land and EEZ
land_clip.plot(ax=ax_map, color="#d5d8dc", edgecolor="#95a5a6", linewidth=0.3)
turkey_eez_dissolved.boundary.plot(ax=ax_map, color="#00b4d8", linewidth=1.0,
                                   linestyle="--", alpha=0.5)

# Bathymetry contours (subtle)
bathy_display = np.ma.masked_where(~sea_mask, bathy)
contour_levels = [-2000, -1000, -500, -200, -50]
ax_map.contour(lons, lats, bathy_display, levels=contour_levels,
               colors="#bdc3c7", linewidths=0.3, alpha=0.4)

# Shipping routes
for route_name, coords in SHIPPING_ROUTES:
    rlons = [c[0] for c in coords]
    rlats = [c[1] for c in coords]
    ax_map.plot(rlons, rlats, color="#95a5a6", linewidth=1, linestyle=":",
                alpha=0.5)

# Plot each scenario's footprint
for scenario_name in ["Ambitious", "Moderate", "Conservative"]:
    r = results[scenario_name]
    mask = r["wind_mask"]
    masked_data = np.ma.masked_where(~mask, np.ones_like(mask, dtype=float))
    ax_map.pcolormesh(lons, lats, masked_data,
                      cmap=LinearSegmentedColormap.from_list(
                          "sc", [r["color_light"], r["color"]]),
                      alpha=0.6, shading="auto")

# Legend patches
legend_patches = []
for sn in ["Conservative", "Moderate", "Ambitious"]:
    r = results[sn]
    legend_patches.append(mpatches.Patch(
        facecolor=r["color"], alpha=0.6,
        label=f'{sn} ({r["label"]}, {r["footprint_km2"]:,.0f} km2)',
    ))
legend_patches.append(Line2D([0], [0], color="#00b4d8", linewidth=1,
                             linestyle="--", label="Turkey EEZ"))
legend_patches.append(Line2D([0], [0], color="#95a5a6", linewidth=1,
                             linestyle=":", label="Shipping Routes"))

ax_map.legend(handles=legend_patches, loc="lower left", fontsize=7,
              framealpha=0.9, edgecolor="#bdc3c7")

ax_map.set_xlim(24.5, 42.5)
ax_map.set_ylim(34.5, 43.0)
ax_map.set_xlabel("Longitude", fontsize=9)
ax_map.set_ylabel("Latitude", fontsize=9)
ax_map.set_title("(a) Scenario Footprints", fontsize=13, fontweight="bold", pad=10)
ax_map.grid(True, alpha=0.2, linestyle="--")
ax_map.tick_params(labelsize=8)
ax_map.set_aspect("equal")

# ---------- Panel B: Stacked Bar Chart ----------
print("  Drawing stacked bar chart...")

categories = ["Fishing\nDisplaced", "Shipping\nConflict", "Habitat\nAffected",
              "Visual\nImpact", "Noise\nDisturbance"]

x_pos = np.arange(len(SCENARIOS))
bar_width = 0.55

scenario_names = list(SCENARIOS.keys())

for cat_idx, (cat_label, key) in enumerate(zip(
    categories,
    ["fishing_displaced_km2", "shipping_conflict_km2", "habitat_affected_km2",
     "visual_impact_km2", "noise_disturbance_km2"]
)):
    pass  # We'll use grouped bars instead

# Grouped bar chart (clearer than stacked for comparisons)
n_cats = len(categories)
n_scenarios = len(scenario_names)
group_width = 0.7
bar_w = group_width / n_scenarios
x_cats = np.arange(n_cats)

impact_keys = ["fishing_displaced_km2", "shipping_conflict_km2",
               "habitat_affected_km2", "visual_impact_km2",
               "noise_disturbance_km2"]

for s_idx, sn in enumerate(scenario_names):
    r = results[sn]
    values = [r[k] for k in impact_keys]
    offset = (s_idx - n_scenarios / 2 + 0.5) * bar_w
    bars = ax_bar.bar(x_cats + offset, values, width=bar_w,
                      color=r["color"], alpha=0.8, edgecolor="white",
                      linewidth=0.5, label=f'{sn} ({r["label"]})')
    # Value labels on bars (only if > 10)
    for b, v in zip(bars, values):
        if v > 10:
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                        f'{v:,.0f}', ha="center", va="bottom", fontsize=6,
                        fontweight="bold", color=r["color"], rotation=0)

ax_bar.set_xticks(x_cats)
ax_bar.set_xticklabels(categories, fontsize=9, fontweight="bold")
ax_bar.set_ylabel("Impact Area (km2)", fontsize=10, fontweight="bold")
ax_bar.set_title("(b) Impact Comparison by Category", fontsize=13,
                 fontweight="bold", pad=10)
ax_bar.legend(fontsize=8, framealpha=0.9, edgecolor="#bdc3c7")
ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--")
ax_bar.tick_params(labelsize=8)

# ---------- Panel C: Radar Chart ----------
print("  Drawing radar chart...")

radar_categories = ["Fishing\nDisplacement", "Shipping\nConflict",
                    "Habitat\nImpact", "Visual\nImpact",
                    "Noise\nDisturbance", "EEZ %\nUsed"]
n_vars = len(radar_categories)

# Normalise each dimension to 0-1 (relative to ambitious scenario)
amb = results["Ambitious"]
radar_keys = ["fishing_displaced_km2", "shipping_conflict_km2",
              "habitat_affected_km2", "visual_impact_km2",
              "noise_disturbance_km2", "total_impact_pct"]

max_vals = [max(results[sn][k] for sn in scenario_names) for k in radar_keys]
# Avoid division by zero
max_vals = [m if m > 0 else 1 for m in max_vals]

angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)

for sn in scenario_names:
    r = results[sn]
    values = [r[k] / max_vals[i] for i, k in enumerate(radar_keys)]
    values += values[:1]
    ax_radar.plot(angles, values, 'o-', linewidth=2, color=r["color"],
                  label=f'{sn} ({r["label"]})', markersize=4)
    ax_radar.fill(angles, values, alpha=0.1, color=r["color"])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_categories, fontsize=8, fontweight="bold")
ax_radar.set_ylim(0, 1.15)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7,
                          color="#7f8c8d")
ax_radar.set_title("(c) Impact Profile (normalised)", fontsize=13,
                   fontweight="bold", pad=20)
ax_radar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15),
                fontsize=8, framealpha=0.9, edgecolor="#bdc3c7")
ax_radar.grid(True, alpha=0.3)

# ---------- Panel D: Summary Table ----------
print("  Drawing summary table...")

ax_table.axis("off")

# Table data
col_labels = ["Metric", "Conservative\n(500 MW)", "Moderate\n(5 GW)",
              "Ambitious\n(41.7 GW)"]

row_data = [
    ["Capacity (MW)",
     f'{results["Conservative"]["capacity_mw"]:,}',
     f'{results["Moderate"]["capacity_mw"]:,}',
     f'{results["Ambitious"]["capacity_mw"]:,}'],
    ["Turbines (15 MW)",
     f'{results["Conservative"]["n_turbines"]:,}',
     f'{results["Moderate"]["n_turbines"]:,}',
     f'{results["Ambitious"]["n_turbines"]:,}'],
    ["Annual Energy (GWh)",
     f'{results["Conservative"]["annual_gwh"]:,.0f}',
     f'{results["Moderate"]["annual_gwh"]:,.0f}',
     f'{results["Ambitious"]["annual_gwh"]:,.0f}'],
    ["Footprint (km2)",
     f'{results["Conservative"]["footprint_km2"]:,.0f}',
     f'{results["Moderate"]["footprint_km2"]:,.0f}',
     f'{results["Ambitious"]["footprint_km2"]:,.0f}'],
    ["Fishing Displaced (km2)",
     f'{results["Conservative"]["fishing_displaced_km2"]:,.0f}',
     f'{results["Moderate"]["fishing_displaced_km2"]:,.0f}',
     f'{results["Ambitious"]["fishing_displaced_km2"]:,.0f}'],
    ["Shipping Conflict (km2)",
     f'{results["Conservative"]["shipping_conflict_km2"]:,.0f}',
     f'{results["Moderate"]["shipping_conflict_km2"]:,.0f}',
     f'{results["Ambitious"]["shipping_conflict_km2"]:,.0f}'],
    ["Habitat Affected (km2)",
     f'{results["Conservative"]["habitat_affected_km2"]:,.0f}',
     f'{results["Moderate"]["habitat_affected_km2"]:,.0f}',
     f'{results["Ambitious"]["habitat_affected_km2"]:,.0f}'],
    ["Seagrass Affected (km2)",
     f'{results["Conservative"]["seagrass_affected_km2"]:,.0f}',
     f'{results["Moderate"]["seagrass_affected_km2"]:,.0f}',
     f'{results["Ambitious"]["seagrass_affected_km2"]:,.0f}'],
    ["Visual Impact (km2)",
     f'{results["Conservative"]["visual_impact_km2"]:,.0f}',
     f'{results["Moderate"]["visual_impact_km2"]:,.0f}',
     f'{results["Ambitious"]["visual_impact_km2"]:,.0f}'],
    ["Noise Disturbance (km2)",
     f'{results["Conservative"]["noise_disturbance_km2"]:,.0f}',
     f'{results["Moderate"]["noise_disturbance_km2"]:,.0f}',
     f'{results["Ambitious"]["noise_disturbance_km2"]:,.0f}'],
    ["Total Impact (km2)",
     f'{results["Conservative"]["total_impact_km2"]:,.0f}',
     f'{results["Moderate"]["total_impact_km2"]:,.0f}',
     f'{results["Ambitious"]["total_impact_km2"]:,.0f}'],
    ["EEZ Impact (%)",
     f'{results["Conservative"]["total_impact_pct"]:.2f}%',
     f'{results["Moderate"]["total_impact_pct"]:.2f}%',
     f'{results["Ambitious"]["total_impact_pct"]:.2f}%'],
]

table = ax_table.table(
    cellText=row_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.45)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#0d1b2a")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_edgecolor("#34495e")

# Style scenario columns
scenario_colors_table = ["white", "#d5f5e3", "#fdebd0", "#fadbd8"]
for i in range(1, len(row_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if j == 0:
            cell.set_text_props(fontweight="bold", ha="left")
            cell.set_facecolor("#f8f9fa")
        else:
            cell.set_facecolor(scenario_colors_table[j] if i % 2 == 0 else "white")
        cell.set_edgecolor("#dfe6e9")

ax_table.set_title("(d) Impact Summary Table", fontsize=13,
                   fontweight="bold", pad=10)

# ---------- Super title ----------
fig.suptitle("Offshore Wind Cumulative Impact Assessment -- Turkish Waters",
             fontsize=18, fontweight="bold", color="#1a2634", y=0.97)

# Source annotation
fig.text(0.99, 0.005,
         "Sources: GEBCO 2025, VLIZ EEZ v12, Natural Earth | Synthetic AIS & habitat data",
         ha="right", va="bottom", fontsize=7, color="#bdc3c7")

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
print("CUMULATIVE IMPACT ASSESSMENT RESULTS")
print("=" * 70)

for sn in scenario_names:
    r = results[sn]
    print(f"""
  {sn.upper()} SCENARIO ({r['label']})
  {'-'*60}
  Installed Capacity:       {r['capacity_mw']:,} MW
  Number of Turbines:       {r['n_turbines']:,} x {TURBINE_MW} MW
  Annual Energy:            {r['annual_gwh']:,.0f} GWh
  Capacity Factor:          {CAPACITY_FACTOR*100:.0f}%
  Development Footprint:    {r['footprint_km2']:,.0f} km2

  IMPACT BREAKDOWN:
    Fishing Displaced:      {r['fishing_displaced_km2']:,.0f} km2 ({r['fishing_displaced_pct']:.1f}% of total)
    Shipping Conflict:      {r['shipping_conflict_km2']:,.0f} km2
    Habitat Affected:       {r['habitat_affected_km2']:,.0f} km2
    Seagrass Affected:      {r['seagrass_affected_km2']:,.0f} km2
    Visual Impact Zone:     {r['visual_impact_km2']:,.0f} km2
    Noise Disturbance:      {r['noise_disturbance_km2']:,.0f} km2

  CUMULATIVE:
    Total Impact Area:      {r['total_impact_km2']:,.0f} km2
    % of EEZ:               {r['total_impact_pct']:.2f}%""")

# Scaling analysis
print(f"""
  SCALING ANALYSIS
  {'-'*60}
  Impact per GW installed:
    Conservative: {results['Conservative']['total_impact_km2']/0.5:,.0f} km2/GW
    Moderate:     {results['Moderate']['total_impact_km2']/5:,.0f} km2/GW
    Ambitious:    {results['Ambitious']['total_impact_km2']/41.7:,.0f} km2/GW

  Energy output per km2 of impact:
    Conservative: {results['Conservative']['annual_gwh']/max(results['Conservative']['total_impact_km2'],1):,.1f} GWh/km2
    Moderate:     {results['Moderate']['annual_gwh']/max(results['Moderate']['total_impact_km2'],1):,.1f} GWh/km2
    Ambitious:    {results['Ambitious']['annual_gwh']/max(results['Ambitious']['total_impact_km2'],1):,.1f} GWh/km2
""")

print("=" * 70)
print("DONE - Wind cumulative impact assessment complete!")
print("=" * 70)
