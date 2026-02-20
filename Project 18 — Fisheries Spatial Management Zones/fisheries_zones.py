"""
Fisheries Spatial Management Zones - Turkish Waters
======================================================
Divides the Turkish EEZ into five functional management zones
based on habitat sensitivity, fishing pressure, MPA locations,
and depth, following an ecosystem-based fisheries management approach.
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
from shapely.geometry import box, mapping
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
MPA_BASE   = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 18 fisheries zones"
OUTPUT_PNG = OUTPUT_DIR / "turkey_fisheries_zones.png"

# Zone codes
Z_NODATA       = 0
Z_NO_TAKE      = 1
Z_SEASONAL     = 2
Z_EFFORT_RED   = 3
Z_SUSTAINABLE  = 4
Z_OPEN_ACCESS  = 5

ZONE_NAMES = {
    Z_NO_TAKE:     "No-Take Zone (NTZ)",
    Z_SEASONAL:    "Seasonal Closure",
    Z_EFFORT_RED:  "Effort Reduction",
    Z_SUSTAINABLE: "Sustainable Fishing",
    Z_OPEN_ACCESS: "Open Access",
}

ZONE_COLORS = {
    Z_NO_TAKE:     "#c0392b",
    Z_SEASONAL:    "#e67e22",
    Z_EFFORT_RED:  "#f1c40f",
    Z_SUSTAINABLE: "#82e0aa",
    Z_OPEN_ACCESS: "#85c1e9",
}

ZONE_RESTRICTIONS = {
    Z_NO_TAKE:     "All fishing prohibited year-round",
    Z_SEASONAL:    "Closed Apr-Sep (spawning season)",
    Z_EFFORT_RED:  "50% effort cap, gear restrictions",
    Z_SUSTAINABLE: "Licensed fishing, quota-managed",
    Z_OPEN_ACCESS: "Open fishing, monitoring required",
}

ZONE_RATIONALE = {
    Z_NO_TAKE:     "MPA + high habitat sensitivity",
    Z_SEASONAL:    "High anchovy suitability (>0.7)",
    Z_EFFORT_RED:  "Mod. sensitivity + high pressure",
    Z_SUSTAINABLE: "Low sensitivity + mod. pressure",
    Z_OPEN_ACCESS: "Deep water, low sensitivity",
}

# Fishing hotspot centres (from Project 5)
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

# Coordinate constants
AVG_LAT = 39.0
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(AVG_LAT))
KM_PER_DEG_LAT = 110.57

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("FISHERIES SPATIAL MANAGEMENT ZONES - Turkish Waters")
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

# Coordinate grids
lon_min_r = transform[2]
lat_max_r = transform[5]
lons = np.arange(nx) * pixel_res[1] + lon_min_r + pixel_res[1] / 2
lats = lat_max_r - np.arange(ny) * pixel_res[0] - pixel_res[0] / 2
lon_grid, lat_grid = np.meshgrid(lons, lats)
depth = -bathy  # positive = below sea level

# ============================================================================
# 2. BUILD INPUT LAYERS
# ============================================================================
print("\n[5/8] Building input layers...")

# --- EEZ mask ---
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool)

land_shapes = [(mapping(g), 1) for g in land_clip.geometry if g is not None]
land_mask = rasterize(
    land_shapes, out_shape=(ny, nx), transform=transform,
    fill=0, dtype=np.uint8,
).astype(bool)

sea_mask = eez_mask & ~land_mask

# --- MPA raster mask ---
print("  Rasterising MPA boundaries...")
mpa_shapes = []
for _, row in mpas_in_eez.iterrows():
    g = row.geometry
    if g is not None and not g.is_empty:
        mpa_shapes.append((mapping(g), 1))

if mpa_shapes:
    mpa_mask = rasterize(
        mpa_shapes, out_shape=(ny, nx), transform=transform,
        fill=0, dtype=np.uint8,
    ).astype(bool) & sea_mask
else:
    mpa_mask = np.zeros((ny, nx), dtype=bool)

# Expand MPA influence with a buffer (~5 km)
from scipy.ndimage import binary_dilation
mpa_buffer_px = max(3, int(5 / dx_km))
struct = np.ones((mpa_buffer_px * 2 + 1, mpa_buffer_px * 2 + 1), dtype=bool)
mpa_expanded = binary_dilation(mpa_mask, structure=struct) & sea_mask
print(f"  MPA mask: {mpa_mask.sum():,} px ({mpa_mask.sum() * pixel_area_km2:,.0f} km2)")
print(f"  MPA buffered: {mpa_expanded.sum():,} px ({mpa_expanded.sum() * pixel_area_km2:,.0f} km2)")

# --- Habitat suitability (anchovy, from Project 7 logic) ---
print("  Computing habitat suitability...")

# Depth suitability: 10-200m optimal
depth_suit = np.zeros_like(depth)
d_shallow = (depth >= 5) & (depth < 10)
d_opt     = (depth >= 10) & (depth <= 100)
d_deep    = (depth > 100) & (depth <= 200)
depth_suit[d_shallow] = (depth[d_shallow] - 5) / 5
depth_suit[d_opt]     = 1.0
depth_suit[d_deep]    = 1.0 - (depth[d_deep] - 100) / 100
depth_suit = np.clip(depth_suit, 0, 1)

# Shore distance suitability (10-60 km peak)
shore_dist_px = distance_transform_edt(~land_mask)
shore_dist_km = shore_dist_px * dx_km

coast_suit = np.zeros_like(shore_dist_km)
c_near = (shore_dist_km >= 2) & (shore_dist_km < 10)
c_opt  = (shore_dist_km >= 10) & (shore_dist_km <= 60)
c_far  = (shore_dist_km > 60) & (shore_dist_km <= 100)
coast_suit[c_near] = (shore_dist_km[c_near] - 2) / 8
coast_suit[c_opt]  = 1.0
coast_suit[c_far]  = 1.0 - (shore_dist_km[c_far] - 60) / 40
coast_suit = np.clip(coast_suit, 0, 1)

# Latitude suitability: 37-42 optimal
lat_suit = np.zeros_like(lat_grid)
l_south = (lat_grid >= 36) & (lat_grid < 37)
l_opt   = (lat_grid >= 37) & (lat_grid <= 42)
l_north = (lat_grid > 42) & (lat_grid <= 43)
lat_suit[l_south] = (lat_grid[l_south] - 36)
lat_suit[l_opt]   = 1.0
lat_suit[l_north] = 1.0 - (lat_grid[l_north] - 42)
lat_suit = np.clip(lat_suit, 0, 1)

# Combined suitability (geometric mean)
habitat_suit = np.cbrt(depth_suit * coast_suit * lat_suit)
habitat_suit *= sea_mask
habitat_suit = gaussian_filter(habitat_suit, sigma=2)
habitat_suit *= sea_mask
print(f"  Habitat suitability: mean={habitat_suit[sea_mask].mean():.3f}, "
      f"max={habitat_suit[sea_mask].max():.3f}")

# --- Fishing pressure ---
print("  Computing fishing pressure field...")
np.random.seed(42)
fishing_pressure = np.zeros((ny, nx), dtype=np.float32)
for lon_c, lat_c, std_lon, std_lat in FISHING_HOTSPOTS:
    dist2 = ((lon_grid - lon_c) / std_lon) ** 2 + ((lat_grid - lat_c) / std_lat) ** 2
    fishing_pressure += np.exp(-dist2 / 2)

# Add coastal fishing strip
coastal_boost = np.exp(-shore_dist_km / 20) * 0.3
fishing_pressure += coastal_boost

fishing_pressure *= sea_mask
fp_max = fishing_pressure[sea_mask].max()
if fp_max > 0:
    fishing_pressure /= fp_max
fishing_pressure = gaussian_filter(fishing_pressure, sigma=1.5)
fishing_pressure *= sea_mask
# Re-normalise
fp_max = fishing_pressure[sea_mask].max()
if fp_max > 0:
    fishing_pressure /= fp_max
print(f"  Fishing pressure: mean={fishing_pressure[sea_mask].mean():.3f}, "
      f"max={fishing_pressure[sea_mask].max():.3f}")

# ============================================================================
# 3. CLASSIFY MANAGEMENT ZONES
# ============================================================================
print("\n[6/8] Classifying management zones...")

zones = np.zeros((ny, nx), dtype=np.uint8)

# Priority order (higher priority overwrites lower):
# 5. Open Access    -> deep water, low sensitivity
# 4. Sustainable    -> low sensitivity, moderate pressure
# 3. Effort Reduction -> moderate sensitivity, high pressure
# 2. Seasonal Closure -> high anchovy habitat (>0.7)
# 1. No-Take Zone   -> MPA + high sensitivity

# Step 1: Start with Open Access everywhere in EEZ
zones[sea_mask] = Z_OPEN_ACCESS

# Step 2: Sustainable Fishing - depth < 500m and low-moderate habitat
sustainable = (
    sea_mask &
    (depth > 0) & (depth <= 500) &
    (habitat_suit <= 0.5) &
    (fishing_pressure >= 0.15) & (fishing_pressure < 0.50)
)
zones[sustainable] = Z_SUSTAINABLE

# Step 3: Effort Reduction - moderate sensitivity + high fishing pressure
effort_red = (
    sea_mask &
    (habitat_suit >= 0.3) & (habitat_suit < 0.7) &
    (fishing_pressure >= 0.45)
)
zones[effort_red] = Z_EFFORT_RED

# Step 4: Seasonal Closure - high anchovy habitat suitability
seasonal = (
    sea_mask &
    (habitat_suit >= 0.7)
)
zones[seasonal] = Z_SEASONAL

# Step 5: No-Take Zones - MPA expanded + high sensitivity
no_take = sea_mask & mpa_expanded
zones[no_take] = Z_NO_TAKE

# Also add high-sensitivity coastal areas as NTZ candidates
# (shallow reef-like areas with very high suitability near MPAs)
ntz_extra = (
    sea_mask &
    mpa_mask &
    (habitat_suit >= 0.5)
)
zones[ntz_extra] = Z_NO_TAKE

print("  Zone classification complete.")

# ============================================================================
# 4. COMPUTE STATISTICS
# ============================================================================
print("\n[7/8] Computing zone statistics...")

zone_stats = {}
total_sea_px = sea_mask.sum()

for zcode in [Z_NO_TAKE, Z_SEASONAL, Z_EFFORT_RED, Z_SUSTAINABLE, Z_OPEN_ACCESS]:
    z_mask = zones == zcode
    n_px = z_mask.sum()
    area_km2 = float(n_px * pixel_area_km2)
    pct = float(n_px / total_sea_px * 100) if total_sea_px > 0 else 0

    # Mean fishing pressure in this zone
    fp_mean = float(fishing_pressure[z_mask].mean()) if n_px > 0 else 0
    # Mean habitat suitability
    hs_mean = float(habitat_suit[z_mask].mean()) if n_px > 0 else 0
    # Mean depth
    d_mean = float(depth[z_mask].mean()) if n_px > 0 else 0

    zone_stats[zcode] = {
        "name": ZONE_NAMES[zcode],
        "area_km2": area_km2,
        "pct_eez": pct,
        "mean_fishing": fp_mean,
        "mean_habitat": hs_mean,
        "mean_depth_m": d_mean,
        "restriction": ZONE_RESTRICTIONS[zcode],
        "rationale": ZONE_RATIONALE[zcode],
    }

    print(f"  {ZONE_NAMES[zcode]:25s}  {area_km2:>10,.0f} km2  ({pct:5.1f}%)  "
          f"depth={d_mean:6.0f}m  fish={fp_mean:.2f}  hab={hs_mean:.2f}")

total_classified = sum(s["area_km2"] for s in zone_stats.values())

# ============================================================================
# 5. CREATE FIGURE
# ============================================================================
print("\n[8/8] Creating figure...")

fig = plt.figure(figsize=(20, 9), dpi=300, facecolor="white")

gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.6, 1.0],
                      wspace=0.25, left=0.04, right=0.98,
                      top=0.87, bottom=0.07)

ax_map   = fig.add_subplot(gs[0, 0])
ax_pie   = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[0, 2])

# ---------- Panel A: Zone Map ----------
print("  Drawing zone map...")

# Land base
land_clip.plot(ax=ax_map, color="#d5d8dc", edgecolor="#95a5a6", linewidth=0.3)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax_map, color="#00b4d8", linewidth=1.2,
                                   linestyle="--", alpha=0.6)

# Zone raster
zone_display = np.ma.masked_where(zones == Z_NODATA, zones)
cmap_zones = ListedColormap([
    ZONE_COLORS[Z_NO_TAKE],
    ZONE_COLORS[Z_SEASONAL],
    ZONE_COLORS[Z_EFFORT_RED],
    ZONE_COLORS[Z_SUSTAINABLE],
    ZONE_COLORS[Z_OPEN_ACCESS],
])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = BoundaryNorm(bounds, cmap_zones.N)

ax_map.pcolormesh(lons, lats, zone_display, cmap=cmap_zones, norm=norm,
                  alpha=0.75, shading="auto")

# MPA outlines
for _, row in mpas_in_eez.iterrows():
    g = row.geometry
    if g is None or g.is_empty:
        continue
    try:
        gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").boundary.plot(
            ax=ax_map, color="black", linewidth=1.0, alpha=0.8)
    except Exception:
        pass

# Depth contours
bathy_display = np.ma.masked_where(~sea_mask, bathy)
cs = ax_map.contour(lons, lats, bathy_display,
                    levels=[-2000, -1000, -500, -200, -50],
                    colors="#7f8c8d", linewidths=0.3, alpha=0.3)

# Basin labels
basin_labels = [
    (34.0, 42.2, "BLACK SEA"),
    (26.0, 38.0, "AEGEAN\nSEA"),
    (32.0, 35.5, "MEDITERRANEAN SEA"),
    (28.5, 40.6, "MARMARA"),
]
for bx, by, btxt in basin_labels:
    ax_map.text(bx, by, btxt, fontsize=7, fontstyle="italic",
                color="#566573", ha="center", va="center", alpha=0.6)

# Legend
legend_patches = []
for zcode in [Z_NO_TAKE, Z_SEASONAL, Z_EFFORT_RED, Z_SUSTAINABLE, Z_OPEN_ACCESS]:
    s = zone_stats[zcode]
    legend_patches.append(mpatches.Patch(
        facecolor=ZONE_COLORS[zcode], alpha=0.75, edgecolor="#bdc3c7", linewidth=0.5,
        label=f'{s["name"]} ({s["pct_eez"]:.1f}%)',
    ))
legend_patches.append(Line2D([0], [0], color="black", linewidth=1.0,
                             label="MPA boundaries"))
legend_patches.append(Line2D([0], [0], color="#00b4d8", linewidth=1.2,
                             linestyle="--", label="Turkey EEZ"))

ax_map.legend(handles=legend_patches, loc="lower left", fontsize=7,
              framealpha=0.9, edgecolor="#bdc3c7", ncol=1)

ax_map.set_xlim(24.5, 42.5)
ax_map.set_ylim(34.5, 43.0)
ax_map.set_xlabel("Longitude", fontsize=9)
ax_map.set_ylabel("Latitude", fontsize=9)
ax_map.set_title("(a) Fisheries Management Zones", fontsize=13,
                 fontweight="bold", pad=10)
ax_map.grid(True, alpha=0.15, linestyle="--")
ax_map.tick_params(labelsize=8)
ax_map.set_aspect("equal")

# ---------- Panel B: Pie Chart ----------
print("  Drawing pie chart...")

pie_labels = []
pie_sizes = []
pie_colors = []
pie_explode = []

for zcode in [Z_NO_TAKE, Z_SEASONAL, Z_EFFORT_RED, Z_SUSTAINABLE, Z_OPEN_ACCESS]:
    s = zone_stats[zcode]
    short = s["name"].split("(")[0].strip()
    if s["name"] == "No-Take Zone (NTZ)":
        short = "No-Take (NTZ)"
    pie_labels.append(short)
    pie_sizes.append(s["pct_eez"])
    pie_colors.append(ZONE_COLORS[zcode])
    pie_explode.append(0.03 if zcode == Z_NO_TAKE else 0)

wedges, texts, autotexts = ax_pie.pie(
    pie_sizes, labels=None, colors=pie_colors, explode=pie_explode,
    autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
    pctdistance=0.75, startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=1.5, alpha=0.85),
)

for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")

ax_pie.legend(wedges, pie_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), fontsize=7.5,
              framealpha=0.9, edgecolor="#bdc3c7", ncol=1)

ax_pie.set_title("(b) EEZ Allocation (%)", fontsize=13,
                 fontweight="bold", pad=10)

# Total area annotation
ax_pie.text(0, 0, f'{total_classified:,.0f}\nkm2',
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#2c3e50")

# ---------- Panel C: Summary Table ----------
print("  Drawing summary table...")

ax_table.axis("off")

col_labels = ["Zone", "Area (km2)", "% EEZ", "Depth (m)", "Restrictions"]

row_data = []
for zcode in [Z_NO_TAKE, Z_SEASONAL, Z_EFFORT_RED, Z_SUSTAINABLE, Z_OPEN_ACCESS]:
    s = zone_stats[zcode]
    short = s["name"]
    if short == "No-Take Zone (NTZ)":
        short = "No-Take (NTZ)"
    row_data.append([
        short,
        f'{s["area_km2"]:,.0f}',
        f'{s["pct_eez"]:.1f}%',
        f'{s["mean_depth_m"]:,.0f}',
        s["restriction"],
    ])

row_data.append([
    "TOTAL",
    f'{total_classified:,.0f}',
    "100%",
    "--",
    "",
])

table = ax_table.table(
    cellText=row_data,
    colLabels=col_labels,
    loc="upper center",
    cellLoc="center",
    colWidths=[0.18, 0.13, 0.08, 0.10, 0.51],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.6)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#0d1b2a")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_edgecolor("#34495e")

# Style zone rows
zone_codes_order = [Z_NO_TAKE, Z_SEASONAL, Z_EFFORT_RED, Z_SUSTAINABLE, Z_OPEN_ACCESS]
for i in range(1, len(row_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_edgecolor("#dfe6e9")
        if i <= 5:
            zc = zone_codes_order[i - 1]
            if j == 0:
                cell.set_facecolor(ZONE_COLORS[zc])
                cell.set_text_props(fontweight="bold", color="white" if zc == Z_NO_TAKE else "#2c3e50")
                cell.set_alpha(0.85)
            elif j == 4:
                cell.set_text_props(fontsize=7, ha="left")
            if i % 2 == 0 and j > 0:
                cell.set_facecolor("#f8f9fa")
        elif i == 6:
            cell.set_facecolor("#d5f5e3")
            cell.set_text_props(fontweight="bold")

# Rationale sub-table
rationale_data = [["Zone", "Classification Rationale"]]
for zcode in zone_codes_order:
    s = zone_stats[zcode]
    short = s["name"]
    if short == "No-Take Zone (NTZ)":
        short = "No-Take (NTZ)"
    rationale_data.append([short, s["rationale"]])

table2 = ax_table.table(
    cellText=rationale_data[1:],
    colLabels=rationale_data[0],
    loc="lower center",
    cellLoc="center",
    colWidths=[0.30, 0.70],
)
table2.auto_set_font_size(False)
table2.set_fontsize(7.5)
table2.scale(1.0, 1.45)

for j in range(2):
    cell = table2[0, j]
    cell.set_facecolor("#1a5276")
    cell.set_text_props(color="white", fontweight="bold", fontsize=8)
    cell.set_edgecolor("#34495e")

for i in range(1, 6):
    for j in range(2):
        cell = table2[i, j]
        cell.set_edgecolor("#dfe6e9")
        zc = zone_codes_order[i - 1]
        if j == 0:
            cell.set_facecolor(ZONE_COLORS[zc])
            cell.set_text_props(fontweight="bold",
                                color="white" if zc == Z_NO_TAKE else "#2c3e50",
                                fontsize=7)
            cell.set_alpha(0.85)
        elif i % 2 == 0:
            cell.set_facecolor("#f8f9fa")

ax_table.set_title("(c) Zone Summary & Rationale", fontsize=13,
                   fontweight="bold", pad=10)

# ---------- Super title ----------
fig.suptitle("Fisheries Spatial Management Zones -- Turkish Waters",
             fontsize=18, fontweight="bold", color="#1a2634", y=0.96)

# Stats footer
protected_pct = zone_stats[Z_NO_TAKE]["pct_eez"] + zone_stats[Z_SEASONAL]["pct_eez"]
managed_pct = protected_pct + zone_stats[Z_EFFORT_RED]["pct_eez"]
fig.text(0.50, 0.015,
         f'Protected (NTZ + Seasonal): {protected_pct:.1f}% | '
         f'Managed (+ Effort Red.): {managed_pct:.1f}% | '
         f'Total EEZ classified: {total_classified:,.0f} km2',
         ha="center", va="center", fontsize=9, fontstyle="italic",
         color="#7f8c8d",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                   edgecolor="#dfe6e9", alpha=0.9))

# Source
fig.text(0.99, 0.003,
         "Sources: GEBCO 2025, WDPA Feb 2026, VLIZ EEZ v12 | "
         "Synthetic habitat & fishing pressure layers",
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
# 6. PRINT RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FISHERIES MANAGEMENT ZONE SUMMARY")
print("=" * 70)

for zcode in zone_codes_order:
    s = zone_stats[zcode]
    print(f"""
  {s['name'].upper()}
  {'-'*60}
  Area:                 {s['area_km2']:>10,.0f} km2 ({s['pct_eez']:.1f}% of EEZ)
  Mean depth:           {s['mean_depth_m']:>10,.0f} m
  Mean fishing pressure:{s['mean_fishing']:>10.3f}
  Mean habitat suit.:   {s['mean_habitat']:>10.3f}
  Restrictions:         {s['restriction']}
  Rationale:            {s['rationale']}""")

print(f"""
  {'='*60}
  OVERALL FRAMEWORK
  {'='*60}
  Total EEZ classified:       {total_classified:,.0f} km2
  Fully protected (NTZ):      {zone_stats[Z_NO_TAKE]['area_km2']:,.0f} km2 ({zone_stats[Z_NO_TAKE]['pct_eez']:.1f}%)
  Seasonally closed:          {zone_stats[Z_SEASONAL]['area_km2']:,.0f} km2 ({zone_stats[Z_SEASONAL]['pct_eez']:.1f}%)
  Effort-managed:             {zone_stats[Z_EFFORT_RED]['area_km2']:,.0f} km2 ({zone_stats[Z_EFFORT_RED]['pct_eez']:.1f}%)
  Sustainable fishing:        {zone_stats[Z_SUSTAINABLE]['area_km2']:,.0f} km2 ({zone_stats[Z_SUSTAINABLE]['pct_eez']:.1f}%)
  Open access:                {zone_stats[Z_OPEN_ACCESS]['area_km2']:,.0f} km2 ({zone_stats[Z_OPEN_ACCESS]['pct_eez']:.1f}%)

  Protection summary:
    Strict protection:        {zone_stats[Z_NO_TAKE]['pct_eez']:.1f}%
    Protected + seasonal:     {protected_pct:.1f}%
    Total managed:            {managed_pct:.1f}%
""")

print("=" * 70)
print("DONE - Fisheries management zones complete!")
print("=" * 70)
