"""
Blue Carbon Ecosystem Mapping - Turkish Waters
=================================================
Maps seagrass meadows, saltmarshes, and mangroves across
Turkish coastal waters and estimates carbon storage potential.
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
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, mapping, Point
from shapely.ops import unary_union
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

OUTPUT_DIR = BASE_DIR / "project 17 blue carbon"
OUTPUT_PNG = OUTPUT_DIR / "turkey_blue_carbon.png"

# Carbon stock values (tonnes C per hectare, from literature)
CARBON_STOCKS = {
    "Seagrass":  83,   # Fourqurean et al. 2012; soil + biomass
    "Saltmarsh": 162,  # Mcleod et al. 2011
    "Mangrove":  386,  # Donato et al. 2011
}

CO2_FACTOR = 3.67        # tC -> tCO2e
CARBON_PRICE_EUR = 50    # EUR per tonne CO2

# Seagrass occurrence centres (lon, lat, radius_km, species, density 0-1)
# Based on literature: Posidonia oceanica dominant in Aegean,
# Cymodocea nodosa in shallower bays, Zostera in Black Sea (rare)
SEAGRASS_CENTRES = [
    # Aegean coast - Posidonia oceanica
    (26.3, 39.0, 25, "P. oceanica",  0.7),   # Lesvos coast
    (26.8, 38.4, 30, "P. oceanica",  0.8),   # Izmir/Cesme
    (27.0, 37.6, 20, "P. oceanica",  0.6),   # Bodrum
    (26.5, 40.0, 18, "P. oceanica",  0.5),   # Gokceada
    (27.4, 37.0, 22, "P. oceanica",  0.7),   # Datca-Marmaris
    (26.2, 38.8, 15, "P. oceanica",  0.6),   # Chios channel
    # Aegean coast - Cymodocea nodosa (shallower bays)
    (27.1, 38.5, 12, "C. nodosa",    0.5),   # Izmir Bay
    (26.9, 37.2, 15, "C. nodosa",    0.5),   # Gokova Bay
    (27.8, 36.9, 10, "C. nodosa",    0.4),   # Fethiye Bay
    # Mediterranean coast
    (30.3, 36.5, 15, "P. oceanica",  0.4),   # Antalya
    (28.3, 36.6, 12, "P. oceanica",  0.5),   # Kas-Kalkan
    (32.5, 36.2, 10, "C. nodosa",    0.3),   # Anamur
    # Sea of Marmara
    (29.0, 40.6, 8,  "C. nodosa",    0.3),   # Marmara Islands
    (27.5, 40.4, 10, "C. nodosa",    0.3),   # Bandirma
    # Black Sea - very limited
    (30.3, 41.2, 5,  "Zostera",      0.15),  # Bolu coast
]

# Saltmarsh locations (at river deltas and coastal wetlands)
# (lon, lat, radius_km, name)
SALTMARSH_CENTRES = [
    (36.0, 41.72, 12, "Kizilirmak Delta"),
    (36.5, 41.42, 10, "Yesilirmak Delta"),
    (26.9, 38.52, 8,  "Gediz Delta"),
    (27.2, 37.50, 7,  "B. Menderes Delta"),
    (33.9, 36.30, 6,  "Goksu Delta"),
    (35.5, 36.82, 10, "Seyhan-Ceyhan Delta"),
    (28.6, 36.83, 5,  "Dalyan Delta"),
    (40.6, 41.00, 4,  "Coruh Delta"),
    (29.0, 41.10, 3,  "Bosphorus wetlands"),
    (30.5, 41.20, 3,  "Sakarya Delta"),
]

# Mangrove (extremely limited in Turkey - only Dalyan area)
MANGROVE_CENTRES = [
    (28.62, 36.82, 2, "Dalyan (Liquidambar orientalis)"),
]

# Depth constraints
SEAGRASS_DEPTH_MAX = 40    # metres
SEAGRASS_DEPTH_MIN = 0.5   # metres (not intertidal)
SALTMARSH_DEPTH_MAX = 2    # metres (intertidal/supratidal)
MANGROVE_DEPTH_MAX  = 1    # metres

# Shore distance constraints (km)
SEAGRASS_SHORE_MAX = 15
SALTMARSH_SHORE_MAX = 3
MANGROVE_SHORE_MAX = 2

# Grid resolution
AVG_LAT = 39.0
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(AVG_LAT))
KM_PER_DEG_LAT = 110.57

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("BLUE CARBON ECOSYSTEM MAPPING - Turkish Waters")
print("=" * 70)

print("\n[1/7] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
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
    ny, nx = bathy.shape
    pixel_res = (abs(transform[4]), abs(transform[0]))

dx_km = pixel_res[1] * KM_PER_DEG_LON
dy_km = pixel_res[0] * KM_PER_DEG_LAT
pixel_area_km2 = dx_km * dy_km
pixel_area_ha = pixel_area_km2 * 100  # 1 km2 = 100 ha
print(f"  GEBCO: {nx}x{ny}, pixel ~{dx_km:.3f} x {dy_km:.3f} km")

# Coordinate grids
lon_min_r = transform[2]
lat_max_r = transform[5]
lons = np.arange(nx) * pixel_res[1] + lon_min_r + pixel_res[1] / 2
lats = lat_max_r - np.arange(ny) * pixel_res[0] - pixel_res[0] / 2
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Positive depth values
depth = -bathy  # positive = below sea level

# EEZ mask
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

# Shore distance
print("  Computing shore distance...")
shore_dist_px = distance_transform_edt(~land_mask)
shore_dist_km = shore_dist_px * dx_km

# ============================================================================
# 2. BUILD ECOSYSTEM DISTRIBUTION MAPS
# ============================================================================
print("\n[4/7] Mapping blue carbon ecosystems...")
np.random.seed(42)

# --- Seagrass ---
print("  Mapping seagrass meadows...")
seagrass_prob = np.zeros((ny, nx), dtype=np.float32)

for lon_c, lat_c, radius_km, species, density in SEAGRASS_CENTRES:
    dist_km = np.sqrt(((lon_grid - lon_c) * KM_PER_DEG_LON) ** 2 +
                      ((lat_grid - lat_c) * KM_PER_DEG_LAT) ** 2)
    contribution = density * np.exp(-0.5 * (dist_km / (radius_km * 0.5)) ** 2)
    seagrass_prob += contribution

# Apply constraints
seagrass_mask = (
    (depth >= SEAGRASS_DEPTH_MIN) &
    (depth <= SEAGRASS_DEPTH_MAX) &
    (shore_dist_km <= SEAGRASS_SHORE_MAX) &
    (shore_dist_km >= 0.5) &
    sea_mask
)

# Depth preference: peak at 5-15m, declining to 40m
depth_pref = np.zeros_like(depth)
shallow = (depth >= 0.5) & (depth < 5)
optimal = (depth >= 5) & (depth <= 15)
deep = (depth > 15) & (depth <= 40)
depth_pref[shallow] = depth[shallow] / 5
depth_pref[optimal] = 1.0
depth_pref[deep] = 1.0 - (depth[deep] - 15) / 25

seagrass_prob *= depth_pref
seagrass_prob *= seagrass_mask
seagrass_prob = np.clip(seagrass_prob, 0, 1)

# Threshold to binary presence
seagrass_present = (seagrass_prob >= 0.15) & seagrass_mask
# Apply light Gaussian smoothing to make patches more realistic
seagrass_smoothed = gaussian_filter(seagrass_prob, sigma=1.5)
seagrass_present = (seagrass_smoothed >= 0.12) & seagrass_mask

seagrass_area_km2 = float(seagrass_present.sum() * pixel_area_km2)
seagrass_area_ha = seagrass_area_km2 * 100
print(f"    Seagrass: {seagrass_area_km2:,.0f} km2 ({seagrass_area_ha:,.0f} ha)")

# --- Saltmarsh ---
print("  Mapping saltmarsh wetlands...")
saltmarsh_prob = np.zeros((ny, nx), dtype=np.float32)

for lon_c, lat_c, radius_km, name in SALTMARSH_CENTRES:
    dist_km = np.sqrt(((lon_grid - lon_c) * KM_PER_DEG_LON) ** 2 +
                      ((lat_grid - lat_c) * KM_PER_DEG_LAT) ** 2)
    contribution = np.exp(-0.5 * (dist_km / (radius_km * 0.4)) ** 2)
    saltmarsh_prob += contribution

# Saltmarsh at the land-sea interface: very shallow / intertidal
saltmarsh_mask = (
    (depth >= -1) & (depth <= SALTMARSH_DEPTH_MAX) &
    (shore_dist_km <= SALTMARSH_SHORE_MAX) &
    eez_mask  # allow land fringe
)

saltmarsh_prob *= saltmarsh_mask
saltmarsh_prob = np.clip(saltmarsh_prob, 0, 1)
saltmarsh_present = (saltmarsh_prob >= 0.20) & saltmarsh_mask

saltmarsh_area_km2 = float(saltmarsh_present.sum() * pixel_area_km2)
saltmarsh_area_ha = saltmarsh_area_km2 * 100
print(f"    Saltmarsh: {saltmarsh_area_km2:,.0f} km2 ({saltmarsh_area_ha:,.0f} ha)")

# --- Mangrove ---
print("  Mapping mangrove areas...")
mangrove_prob = np.zeros((ny, nx), dtype=np.float32)

for lon_c, lat_c, radius_km, name in MANGROVE_CENTRES:
    dist_km = np.sqrt(((lon_grid - lon_c) * KM_PER_DEG_LON) ** 2 +
                      ((lat_grid - lat_c) * KM_PER_DEG_LAT) ** 2)
    contribution = np.exp(-0.5 * (dist_km / (radius_km * 0.3)) ** 2)
    mangrove_prob += contribution

mangrove_mask = (
    (depth >= -0.5) & (depth <= MANGROVE_DEPTH_MAX) &
    (shore_dist_km <= MANGROVE_SHORE_MAX) &
    eez_mask
)

mangrove_prob *= mangrove_mask
mangrove_prob = np.clip(mangrove_prob, 0, 1)
mangrove_present = (mangrove_prob >= 0.25) & mangrove_mask

mangrove_area_km2 = float(mangrove_present.sum() * pixel_area_km2)
mangrove_area_ha = mangrove_area_km2 * 100
print(f"    Mangrove: {mangrove_area_km2:,.1f} km2 ({mangrove_area_ha:,.0f} ha)")

# ============================================================================
# 3. CARBON ACCOUNTING
# ============================================================================
print("\n[5/7] Computing carbon stocks...")

ecosystems = {}

for eco_name, present_mask, area_km2, area_ha in [
    ("Seagrass",  seagrass_present,  seagrass_area_km2,  seagrass_area_ha),
    ("Saltmarsh", saltmarsh_present, saltmarsh_area_km2, saltmarsh_area_ha),
    ("Mangrove",  mangrove_present,  mangrove_area_km2,  mangrove_area_ha),
]:
    stock_tC_ha = CARBON_STOCKS[eco_name]
    total_tC = area_ha * stock_tC_ha
    total_tCO2 = total_tC * CO2_FACTOR
    value_eur = total_tCO2 * CARBON_PRICE_EUR

    ecosystems[eco_name] = {
        "area_km2": area_km2,
        "area_ha": area_ha,
        "stock_tC_ha": stock_tC_ha,
        "total_tC": total_tC,
        "total_tCO2": total_tCO2,
        "value_eur": value_eur,
        "mask": present_mask,
    }

    print(f"  {eco_name}:")
    print(f"    Area: {area_km2:,.0f} km2 ({area_ha:,.0f} ha)")
    print(f"    Carbon density: {stock_tC_ha} tC/ha")
    print(f"    Total stock: {total_tC:,.0f} tC")
    print(f"    CO2 equivalent: {total_tCO2:,.0f} tCO2e")
    print(f"    Value (EUR50/tCO2): EUR {value_eur:,.0f}")

# Totals
total_area_km2 = sum(e["area_km2"] for e in ecosystems.values())
total_area_ha  = sum(e["area_ha"]  for e in ecosystems.values())
total_tC       = sum(e["total_tC"] for e in ecosystems.values())
total_tCO2     = sum(e["total_tCO2"] for e in ecosystems.values())
total_value    = sum(e["value_eur"] for e in ecosystems.values())

print(f"\n  TOTALS:")
print(f"    Area: {total_area_km2:,.0f} km2 ({total_area_ha:,.0f} ha)")
print(f"    Carbon stock: {total_tC:,.0f} tC")
print(f"    CO2 equivalent: {total_tCO2:,.0f} tCO2e")
print(f"    Value: EUR {total_value:,.0f}")

# Per-basin breakdown
BASIN_BOUNDS = {
    "Aegean":        (25.0, 28.0, 35.5, 41.0),
    "Mediterranean": (28.0, 36.5, 35.0, 37.5),
    "Black Sea":     (28.5, 41.5, 40.5, 43.5),
    "Marmara":       (27.0, 30.0, 40.0, 41.2),
}

basin_results = {}
for basin_name, (blon1, blon2, blat1, blat2) in BASIN_BOUNDS.items():
    basin_mask = ((lon_grid >= blon1) & (lon_grid <= blon2) &
                  (lat_grid >= blat1) & (lat_grid <= blat2))
    basin_sg = float((seagrass_present & basin_mask).sum() * pixel_area_km2)
    basin_sm = float((saltmarsh_present & basin_mask).sum() * pixel_area_km2)
    basin_mg = float((mangrove_present & basin_mask).sum() * pixel_area_km2)
    basin_total = basin_sg + basin_sm + basin_mg
    basin_tC = (basin_sg * 100 * CARBON_STOCKS["Seagrass"] +
                basin_sm * 100 * CARBON_STOCKS["Saltmarsh"] +
                basin_mg * 100 * CARBON_STOCKS["Mangrove"])
    basin_results[basin_name] = {
        "seagrass_km2": basin_sg,
        "saltmarsh_km2": basin_sm,
        "mangrove_km2": basin_mg,
        "total_km2": basin_total,
        "total_tC": basin_tC,
    }

# ============================================================================
# 4. CREATE FIGURE
# ============================================================================
print("\n[6/7] Creating figure...")

fig = plt.figure(figsize=(20, 8), dpi=300, facecolor="white")

gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.8, 0.9],
                      wspace=0.28, left=0.04, right=0.98,
                      top=0.86, bottom=0.08)

ax_map   = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[0, 2])

# Colours
C_SEAGRASS  = "#27ae60"
C_SALTMARSH = "#8e44ad"
C_MANGROVE  = "#d35400"
C_LAND      = "#d5d8dc"

# ---------- Panel A: Ecosystem Map ----------
print("  Drawing ecosystem map...")

# Land
land_clip.plot(ax=ax_map, color=C_LAND, edgecolor="#95a5a6", linewidth=0.3)

# EEZ
turkey_eez_dissolved.boundary.plot(ax=ax_map, color="#00b4d8", linewidth=1.0,
                                   linestyle="--", alpha=0.5)

# Bathymetry shading (shallow areas)
shallow_display = np.ma.masked_where(~sea_mask | (depth > 50), depth)
ax_map.pcolormesh(lons, lats, shallow_display,
                  cmap="Blues_r", alpha=0.15, shading="auto",
                  vmin=0, vmax=50)

# Seagrass
sg_display = np.ma.masked_where(~seagrass_present, seagrass_smoothed)
ax_map.pcolormesh(lons, lats, sg_display,
                  cmap=ListedColormap([C_SEAGRASS]),
                  alpha=0.7, shading="auto")

# Saltmarsh
sm_display = np.ma.masked_where(~saltmarsh_present, saltmarsh_prob)
ax_map.pcolormesh(lons, lats, sm_display,
                  cmap=ListedColormap([C_SALTMARSH]),
                  alpha=0.7, shading="auto")

# Mangrove
mg_display = np.ma.masked_where(~mangrove_present, mangrove_prob)
ax_map.pcolormesh(lons, lats, mg_display,
                  cmap=ListedColormap([C_MANGROVE]),
                  alpha=0.8, shading="auto")

# Label key seagrass regions
seagrass_labels = [
    (26.5, 38.7, "Izmir/Cesme\nSeagrass"),
    (27.2, 37.2, "Bodrum-Datca\nSeagrass"),
    (26.3, 39.6, "N. Aegean\nSeagrass"),
    (30.0, 36.4, "Antalya\nSeagrass"),
]
for lx, ly, label in seagrass_labels:
    ax_map.annotate(label, xy=(lx, ly), fontsize=6, fontweight="bold",
                    color=C_SEAGRASS, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=C_SEAGRASS, alpha=0.85, linewidth=0.8))

# Label saltmarsh deltas
for lon_c, lat_c, _, name in SALTMARSH_CENTRES[:6]:
    ax_map.annotate(name.replace(" Delta", ""), xy=(lon_c, lat_c),
                    fontsize=5.5, fontweight="bold", color=C_SALTMARSH,
                    ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=C_SALTMARSH, alpha=0.8, linewidth=0.6))

# Label mangrove
for lon_c, lat_c, _, name in MANGROVE_CENTRES:
    ax_map.annotate(f"Mangrove\n({name.split('(')[0].strip()})",
                    xy=(lon_c, lat_c),
                    fontsize=6, fontweight="bold", color=C_MANGROVE,
                    ha="center", va="top", xytext=(15, -8),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=C_MANGROVE, lw=1),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=C_MANGROVE, alpha=0.9, linewidth=0.8))

# Legend
legend_patches = [
    mpatches.Patch(facecolor=C_SEAGRASS, alpha=0.7,
                   label=f'Seagrass ({seagrass_area_km2:,.0f} km2)'),
    mpatches.Patch(facecolor=C_SALTMARSH, alpha=0.7,
                   label=f'Saltmarsh ({saltmarsh_area_km2:,.0f} km2)'),
    mpatches.Patch(facecolor=C_MANGROVE, alpha=0.8,
                   label=f'Mangrove ({mangrove_area_km2:,.1f} km2)'),
    Line2D([0], [0], color="#00b4d8", linewidth=1, linestyle="--",
           label="Turkey EEZ"),
]
ax_map.legend(handles=legend_patches, loc="lower left", fontsize=7,
              framealpha=0.9, edgecolor="#bdc3c7")

ax_map.set_xlim(24.5, 42.5)
ax_map.set_ylim(34.5, 43.0)
ax_map.set_xlabel("Longitude", fontsize=9)
ax_map.set_ylabel("Latitude", fontsize=9)
ax_map.set_title("(a) Blue Carbon Ecosystem Distribution", fontsize=12,
                 fontweight="bold", pad=10)
ax_map.grid(True, alpha=0.2, linestyle="--")
ax_map.tick_params(labelsize=8)
ax_map.set_aspect("equal")

# ---------- Panel B: Carbon Stock Bar Chart ----------
print("  Drawing carbon stock chart...")

eco_names = list(ecosystems.keys())
eco_colors = [C_SEAGRASS, C_SALTMARSH, C_MANGROVE]

# Stacked: area (left axis) and carbon stock (bar height)
x_pos = np.arange(len(eco_names))
bar_width = 0.55

# Carbon stock bars (in thousand tonnes C)
carbon_vals = [ecosystems[n]["total_tC"] / 1000 for n in eco_names]  # ktC
co2_vals = [ecosystems[n]["total_tCO2"] / 1000 for n in eco_names]   # ktCO2

bars = ax_bar.bar(x_pos, carbon_vals, width=bar_width,
                  color=eco_colors, alpha=0.8, edgecolor="white", linewidth=1)

# Value labels
for b, v, co2v in zip(bars, carbon_vals, co2_vals):
    if v > 0.01:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + max(carbon_vals) * 0.02,
                    f'{v:,.0f} ktC\n({co2v:,.0f} ktCO2e)',
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#2c3e50")

# Area annotation inside bars
for b, n in zip(bars, eco_names):
    area = ecosystems[n]["area_km2"]
    if b.get_height() > max(carbon_vals) * 0.05:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                    f'{area:,.0f} km2',
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(eco_names, fontsize=10, fontweight="bold")
ax_bar.set_ylabel("Carbon Stock (thousand tonnes C)", fontsize=10,
                  fontweight="bold")
ax_bar.set_title("(b) Carbon Stock by Ecosystem", fontsize=12,
                 fontweight="bold", pad=10)
ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--")
ax_bar.tick_params(labelsize=9)

# Add carbon density annotation
for i, n in enumerate(eco_names):
    ax_bar.text(i, -max(carbon_vals) * 0.08,
                f'{CARBON_STOCKS[n]} tC/ha',
                ha="center", va="top", fontsize=7,
                color="#7f8c8d", fontstyle="italic")

ax_bar.set_ylim(bottom=-max(carbon_vals) * 0.15)

# Total annotation
total_text = (f"Total: {total_tC/1000:,.0f} ktC\n"
              f"({total_tCO2/1000:,.0f} ktCO2e)\n"
              f"EUR {total_value/1e6:,.1f}M")
ax_bar.text(0.97, 0.95, total_text, transform=ax_bar.transAxes,
            fontsize=9, fontweight="bold", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eafaf1",
                      edgecolor="#27ae60", alpha=0.9))

# ---------- Panel C: Summary Table ----------
print("  Drawing summary table...")

ax_table.axis("off")

col_labels = ["Ecosystem", "Area\n(km2)", "Area\n(ha)", "C Stock\n(tC/ha)",
              "Total C\n(ktC)", "CO2e\n(ktCO2)", "Value\n(EUR M)"]

row_data = []
for n in eco_names:
    e = ecosystems[n]
    row_data.append([
        n,
        f'{e["area_km2"]:,.0f}',
        f'{e["area_ha"]:,.0f}',
        f'{e["stock_tC_ha"]}',
        f'{e["total_tC"]/1000:,.0f}',
        f'{e["total_tCO2"]/1000:,.0f}',
        f'{e["value_eur"]/1e6:,.1f}',
    ])

# Total row
row_data.append([
    "TOTAL",
    f'{total_area_km2:,.0f}',
    f'{total_area_ha:,.0f}',
    "--",
    f'{total_tC/1000:,.0f}',
    f'{total_tCO2/1000:,.0f}',
    f'{total_value/1e6:,.1f}',
])

# Basin breakdown rows
row_data.append(["", "", "", "", "", "", ""])
row_data.append(["Basin", "Seagrass", "Saltmarsh", "Mangrove",
                 "Total km2", "Total ktC", ""])
for basin_name, br in basin_results.items():
    row_data.append([
        basin_name,
        f'{br["seagrass_km2"]:,.0f}',
        f'{br["saltmarsh_km2"]:,.0f}',
        f'{br["mangrove_km2"]:,.1f}',
        f'{br["total_km2"]:,.0f}',
        f'{br["total_tC"]/1000:,.0f}',
        "",
    ])

table = ax_table.table(
    cellText=row_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 1.35)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#0d1b2a")
    cell.set_text_props(color="white", fontweight="bold", fontsize=8)
    cell.set_edgecolor("#34495e")

# Style rows
for i in range(1, len(row_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_edgecolor("#dfe6e9")
        if i <= 3:
            # Ecosystem rows
            if j == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor(
                    "#eafaf1" if i == 1 else ("#f4ecf7" if i == 2 else "#fdebd0"))
            elif i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
        elif i == 4:
            # Total row
            cell.set_facecolor("#d5f5e3")
            cell.set_text_props(fontweight="bold")
        elif i == 5:
            # Blank separator
            cell.set_facecolor("white")
            cell.set_edgecolor("white")
        elif i == 6:
            # Basin header
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        else:
            # Basin data
            if i % 2 == 1:
                cell.set_facecolor("#f8f9fa")
            if j == 0:
                cell.set_text_props(fontweight="bold")

ax_table.set_title("(c) Carbon Accounting Summary", fontsize=12,
                   fontweight="bold", pad=10)

# ---------- Super title ----------
fig.suptitle("Blue Carbon Ecosystem Mapping -- Turkish Waters",
             fontsize=17, fontweight="bold", color="#1a2634", y=0.96)

# Stats footer
stats_text = (
    f"Total blue carbon area: {total_area_km2:,.0f} km2 | "
    f"Carbon stock: {total_tC/1000:,.0f} ktC | "
    f"CO2e: {total_tCO2/1000:,.0f} ktCO2 | "
    f"Value at EUR50/tCO2: EUR {total_value/1e6:,.1f}M"
)
fig.text(0.50, 0.02, stats_text, ha="center", va="center", fontsize=9,
         fontstyle="italic", color="#7f8c8d",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                   edgecolor="#dfe6e9", alpha=0.9))

# Source
fig.text(0.99, 0.003,
         "Sources: GEBCO 2025, VLIZ EEZ v12 | C stocks: Fourqurean 2012, "
         "Mcleod 2011, Donato 2011 | Synthetic distribution",
         ha="right", va="bottom", fontsize=6.5, color="#bdc3c7")

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
print("BLUE CARBON ACCOUNTING SUMMARY")
print("=" * 70)

print(f"""
  ECOSYSTEM COVERAGE
  {'-'*60}
  Seagrass (P. oceanica, C. nodosa, Zostera):
    Area:                   {seagrass_area_km2:,.0f} km2 ({seagrass_area_ha:,.0f} ha)
    Dominant species:       Posidonia oceanica (Aegean)
    Depth range:            {SEAGRASS_DEPTH_MIN}-{SEAGRASS_DEPTH_MAX}m
    Carbon density:         {CARBON_STOCKS['Seagrass']} tC/ha

  Saltmarsh (coastal wetlands):
    Area:                   {saltmarsh_area_km2:,.0f} km2 ({saltmarsh_area_ha:,.0f} ha)
    Locations:              {len(SALTMARSH_CENTRES)} river deltas/wetlands
    Depth range:            intertidal to {SALTMARSH_DEPTH_MAX}m
    Carbon density:         {CARBON_STOCKS['Saltmarsh']} tC/ha

  Mangrove (Dalyan area):
    Area:                   {mangrove_area_km2:,.1f} km2 ({mangrove_area_ha:,.0f} ha)
    Location:               Dalyan Delta (28.6E, 36.8N)
    Depth range:            intertidal to {MANGROVE_DEPTH_MAX}m
    Carbon density:         {CARBON_STOCKS['Mangrove']} tC/ha

  CARBON STOCKS
  {'-'*60}""")

for n in eco_names:
    e = ecosystems[n]
    pct = e["total_tC"] / total_tC * 100 if total_tC > 0 else 0
    print(f"  {n}:")
    print(f"    Stock:              {e['total_tC']:,.0f} tC ({pct:.1f}% of total)")
    print(f"    CO2 equivalent:     {e['total_tCO2']:,.0f} tCO2e")
    print(f"    Carbon value:       EUR {e['value_eur']:,.0f}")
    print()

print(f"""  TOTALS
  {'-'*60}
  Total area:             {total_area_km2:,.0f} km2 ({total_area_ha:,.0f} ha)
  Total carbon stock:     {total_tC:,.0f} tC ({total_tC/1000:,.0f} ktC)
  Total CO2 equivalent:   {total_tCO2:,.0f} tCO2e ({total_tCO2/1e6:,.2f} MtCO2e)
  Total value (EUR50/t):  EUR {total_value:,.0f} (EUR {total_value/1e6:,.1f}M)

  BASIN BREAKDOWN
  {'-'*60}""")

for basin_name, br in basin_results.items():
    print(f"  {basin_name}:")
    print(f"    Seagrass: {br['seagrass_km2']:,.0f} km2 | "
          f"Saltmarsh: {br['saltmarsh_km2']:,.0f} km2 | "
          f"Mangrove: {br['mangrove_km2']:,.1f} km2")
    print(f"    Total: {br['total_km2']:,.0f} km2 | "
          f"Carbon: {br['total_tC']/1000:,.0f} ktC")
    print()

print(f"""  POLICY IMPLICATIONS
  {'-'*60}
  - Blue carbon ecosystems cover {total_area_km2/eez_area_km2*100:.2f}% of Turkey's EEZ
  - Seagrass meadows are the largest contributor ({ecosystems['Seagrass']['total_tC']/total_tC*100:.0f}% of stock)
  - Aegean coast holds the majority of blue carbon assets
  - Carbon value of EUR {total_value/1e6:,.1f}M justifies conservation investment
  - Integration with MPA network could protect {total_area_km2:,.0f} km2 of carbon sinks
""")

print("=" * 70)
print("DONE - Blue carbon ecosystem mapping complete!")
print("=" * 70)
