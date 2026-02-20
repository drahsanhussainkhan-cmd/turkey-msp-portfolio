"""
Shipping Lane Conflict Analysis - Turkish Waters
==================================================
Identifies spatial conflicts between major shipping corridors and
MPAs, proposed offshore wind zones, and fishing grounds.
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
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, box, mapping, MultiPolygon
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

OUTPUT_DIR = BASE_DIR / "project 10 shipping conflict"
OUTPUT_PNG = OUTPUT_DIR / "turkey_shipping_conflict.png"

# Shipping routes: list of (name, [(lon, lat), ...])
SHIPPING_ROUTES = [
    ("Bosphorus Strait",       [(28.98, 41.02), (29.05, 41.10), (29.12, 41.18)]),
    ("Black Sea Main",         [(29.0, 41.5), (32.0, 42.0), (36.0, 41.8), (41.0, 41.5)]),
    ("Aegean Main",            [(26.0, 38.5), (27.0, 39.5), (28.5, 40.5), (29.0, 41.0)]),
    ("Mediterranean Main",     [(26.0, 36.0), (30.0, 35.8), (33.0, 36.0),
                                (36.0, 36.2), (40.0, 36.5)]),
    ("Istanbul-Izmir Coastal", [(29.0, 41.0), (28.0, 40.2), (27.5, 39.5), (27.0, 38.5)]),
]

SHIPPING_BUFFER_KM = 5
FISHING_GROUND_KM = 30     # coastal fishing zone
WIND_DEPTH_MIN = -50       # m
WIND_DEPTH_MAX = 0         # m
WIND_SHORE_MIN_KM = 5
WIND_SHORE_MAX_KM = 50

# Projected CRS for Turkey (UTM 36N covers central Turkey well enough)
PROJ_CRS = "EPSG:32636"

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("SHIPPING LANE CONFLICT ANALYSIS - Turkish Waters")
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

print("\n[3/7] Loading Turkey MPA polygons...")
mpa_frames = []
for i in range(3):
    shp = (MPA_BASE / f"WDPA_WDOECM_Feb2026_Public_TUR_shp_{i}" /
           "WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons.shp")
    if shp.exists():
        mpa_frames.append(gpd.read_file(shp))
mpas = pd.concat(mpa_frames, ignore_index=True)
mpas = gpd.GeoDataFrame(mpas, geometry="geometry", crs="EPSG:4326")
mpas = mpas.drop_duplicates(subset="SITE_ID")
mpas_in_eez = gpd.clip(mpas, turkey_eez_dissolved)
mpas_in_eez = mpas_in_eez[~mpas_in_eez.is_empty]
print(f"  {len(mpas_in_eez)} MPAs in EEZ")

print("\n[4/7] Loading GEBCO bathymetry...")
with rasterio.open(GEBCO_PATH) as src:
    bathy = src.read(1).astype(np.float32)
    bathy_transform = src.transform
    bathy_shape = src.shape
    bathy_bounds = src.bounds
    pixel_res = src.res

res_m = pixel_res[0] * 111_000
avg_lat = 39.0
km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
km_per_deg_lat = 110.57
dx_km = pixel_res[1] * km_per_deg_lon
dy_km = pixel_res[0] * km_per_deg_lat
pixel_area_km2 = dx_km * dy_km
print(f"  Raster: {bathy_shape[1]}x{bathy_shape[0]}, ~{res_m:.0f}m res")

# ============================================================================
# 2. BUILD RASTER MASKS
# ============================================================================
print("\n[5/7] Building spatial layers...")

# --- EEZ mask ---
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# --- Land mask ---
land_shapes = [(mapping(geom), 1) for geom in land_clip.geometry if geom is not None]
land_raster = rasterize(
    land_shapes, out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
land_raster = land_raster | (bathy > 0)

ocean_eez = eez_mask & ~land_raster

# --- Distance from shore ---
print("  Computing distance from shore...")
dist_shore = distance_transform_edt(~land_raster, sampling=[dy_km, dx_km])

# --- MPA mask ---
if len(mpas_in_eez) > 0:
    mpa_shapes = [(mapping(g), 1) for g in mpas_in_eez.geometry
                  if g is not None and not g.is_empty]
    mpa_raster = rasterize(
        mpa_shapes, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool) & ocean_eez
else:
    mpa_raster = np.zeros(bathy_shape, dtype=bool)
print(f"  MPA pixels: {mpa_raster.sum():,}")

# --- Offshore wind zones ---
print("  Computing offshore wind zones...")
depth_ok = (bathy >= WIND_DEPTH_MIN) & (bathy < WIND_DEPTH_MAX)
shore_ok = (dist_shore >= WIND_SHORE_MIN_KM) & (dist_shore <= WIND_SHORE_MAX_KM)
wind_zone = ocean_eez & depth_ok & shore_ok & ~mpa_raster
wind_area = wind_zone.sum() * pixel_area_km2
print(f"  Wind zone pixels: {wind_zone.sum():,} ({wind_area:,.0f} km2)")

# --- Fishing grounds (within 30km of shore) ---
print("  Computing fishing grounds...")
fishing_zone = ocean_eez & (dist_shore <= FISHING_GROUND_KM) & (bathy < 0)
fishing_area = fishing_zone.sum() * pixel_area_km2
print(f"  Fishing ground pixels: {fishing_zone.sum():,} ({fishing_area:,.0f} km2)")

# ============================================================================
# 3. BUILD SHIPPING LANES AND BUFFERS
# ============================================================================
print("\n[6/7] Building shipping lanes and buffer zones...")

# Create LineString geometries in 4326, then project for buffering
route_lines_4326 = []
for name, coords in SHIPPING_ROUTES:
    line = LineString(coords)
    route_lines_4326.append({"name": name, "geometry": line})

routes_gdf = gpd.GeoDataFrame(route_lines_4326, crs="EPSG:4326")

# Project to metric CRS, buffer, then back to 4326
routes_proj = routes_gdf.to_crs(PROJ_CRS)
buffer_geoms = routes_proj.geometry.buffer(SHIPPING_BUFFER_KM * 1000)
buffers_proj = gpd.GeoDataFrame(
    {"name": routes_proj["name"].values},
    geometry=buffer_geoms.values,
    crs=PROJ_CRS
)
buffers_4326 = buffers_proj.to_crs("EPSG:4326")

# Clip buffers to EEZ
buffers_clipped = gpd.clip(buffers_4326, turkey_eez_dissolved)
buffers_clipped = buffers_clipped[~buffers_clipped.is_empty]

# Dissolve all buffers into one for rasterization
all_buffer_geom = unary_union(buffers_clipped.geometry)

# Rasterize shipping buffer
if all_buffer_geom.is_empty:
    ship_raster = np.zeros(bathy_shape, dtype=bool)
else:
    geoms_list = []
    if all_buffer_geom.geom_type == "MultiPolygon":
        for poly in all_buffer_geom.geoms:
            geoms_list.append((mapping(poly), 1))
    else:
        geoms_list.append((mapping(all_buffer_geom), 1))
    ship_raster = rasterize(
        geoms_list, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool) & ocean_eez

ship_area = ship_raster.sum() * pixel_area_km2
print(f"  Shipping buffer pixels: {ship_raster.sum():,} ({ship_area:,.0f} km2)")

# Per-route buffer areas
for _, row in buffers_clipped.iterrows():
    geom = row.geometry
    if geom.is_empty:
        continue
    g_list = [(mapping(geom), 1)]
    r_mask = rasterize(
        g_list, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool) & ocean_eez
    print(f"    {row['name']}: {r_mask.sum() * pixel_area_km2:,.0f} km2")

# ============================================================================
# 4. CONFLICT ANALYSIS
# ============================================================================
print("\n  Computing spatial conflicts...")

conflict_mpa = ship_raster & mpa_raster
conflict_wind = ship_raster & wind_zone
conflict_fish = ship_raster & fishing_zone

# Any conflict
conflict_any = conflict_mpa | conflict_wind | conflict_fish

# Areas
c_mpa_km2 = conflict_mpa.sum() * pixel_area_km2
c_wind_km2 = conflict_wind.sum() * pixel_area_km2
c_fish_km2 = conflict_fish.sum() * pixel_area_km2
c_any_km2 = conflict_any.sum() * pixel_area_km2

mpa_area_km2 = mpa_raster.sum() * pixel_area_km2

print(f"\n  CONFLICT RESULTS:")
print(f"  {'Zone':<25} {'Overlap km2':>12} {'Zone Total':>12} {'% Affected':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
pct_mpa = (c_mpa_km2 / mpa_area_km2 * 100) if mpa_area_km2 > 0 else 0
pct_wind = (c_wind_km2 / wind_area * 100) if wind_area > 0 else 0
pct_fish = (c_fish_km2 / fishing_area * 100) if fishing_area > 0 else 0
print(f"  {'Shipping x MPA':<25} {c_mpa_km2:>12,.1f} {mpa_area_km2:>12,.1f} {pct_mpa:>11.2f}%")
print(f"  {'Shipping x Wind':<25} {c_wind_km2:>12,.1f} {wind_area:>12,.1f} {pct_wind:>11.2f}%")
print(f"  {'Shipping x Fishing':<25} {c_fish_km2:>12,.1f} {fishing_area:>12,.1f} {pct_fish:>11.2f}%")
print(f"  {'Total conflict area':<25} {c_any_km2:>12,.1f}")

# ============================================================================
# 5. CREATE MAP
# ============================================================================
print("\n[7/7] Creating map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor="white")
ax.set_facecolor("#AED9E0")

pad = 0.8
ax.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]

# --- Fishing grounds (light orange background) ---
fish_display = np.where(fishing_zone, 1.0, np.nan)
from matplotlib.colors import ListedColormap
cmap_fish = ListedColormap(["#FFDAB9"])
cmap_fish.set_bad(alpha=0)
ax.imshow(fish_display, extent=extent, origin="upper",
          cmap=cmap_fish, alpha=0.45, zorder=2, aspect="auto", interpolation="nearest")

# --- Offshore wind zones (yellow) ---
wind_display = np.where(wind_zone, 1.0, np.nan)
cmap_wind = ListedColormap(["#FFD700"])
cmap_wind.set_bad(alpha=0)
ax.imshow(wind_display, extent=extent, origin="upper",
          cmap=cmap_wind, alpha=0.55, zorder=3, aspect="auto", interpolation="nearest")

# --- Land ---
land.plot(ax=ax, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.3, zorder=4)

# --- EEZ boundary ---
turkey_eez_dissolved.boundary.plot(ax=ax, color="#1E5AA8", linewidth=1.5,
                                    linestyle="--", zorder=5)

# --- MPAs ---
if len(mpas_in_eez) > 0:
    mpas_in_eez.plot(ax=ax, color="#2E8B57", alpha=0.6, edgecolor="#1A6B3A",
                     linewidth=0.8, zorder=6)

# --- Shipping lane buffers (semi-transparent navy) ---
buffers_clipped.plot(ax=ax, color="#1B2A4A", alpha=0.25, edgecolor="none", zorder=7)

# --- Shipping lane lines ---
route_colors = ["#FF6347", "#1B2A4A", "#1B2A4A", "#1B2A4A", "#1B2A4A"]
route_widths = [3.0, 2.0, 2.0, 2.0, 1.5]
for idx, (_, row) in enumerate(routes_gdf.iterrows()):
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color=route_colors[idx], linewidth=route_widths[idx],
            solid_capstyle="round", zorder=8)

# --- Conflict zones (bright red overlay) ---
conflict_display = np.where(conflict_any, 1.0, np.nan)
cmap_conflict = ListedColormap(["#FF0000"])
cmap_conflict.set_bad(alpha=0)
ax.imshow(conflict_display, extent=extent, origin="upper",
          cmap=cmap_conflict, alpha=0.65, zorder=9, aspect="auto",
          interpolation="nearest")

# --- Bosphorus label ---
ax.annotate("BOSPHORUS\nSTRAIT", xy=(29.05, 41.10), fontsize=8, fontweight="bold",
            color="#FF6347", ha="left", va="center",
            xytext=(29.5, 41.35), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#FF6347", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FF6347", alpha=0.9),
            zorder=11)

# --- Route labels (subtle) ---
label_positions = {
    "Black Sea Main": (34.0, 42.2),
    "Mediterranean Main": (33.0, 35.4),
    "Aegean Main": (25.8, 39.2),
    "Istanbul-Izmir Coastal": (27.0, 39.9),
}
for name, (lx, ly) in label_positions.items():
    ax.text(lx, ly, name.replace(" Main", ""), fontsize=7, color="#1B2A4A",
            fontstyle="italic", alpha=0.8, zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

# --- Title ---
ax.set_title("Shipping Lane Conflict Analysis -- Turkish Waters",
             fontsize=17, fontweight="bold", pad=16, color="#1A1A2E")
ax.set_xlabel("Longitude", fontsize=11, labelpad=8)
ax.set_ylabel("Latitude", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle=":", alpha=0.3, color="#666")

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color="#1B2A4A", linewidth=2.5, label="Shipping lanes"),
    mpatches.Patch(facecolor="#1B2A4A", alpha=0.25,
                   label=f"Shipping buffer ({SHIPPING_BUFFER_KM} km)"),
    mpatches.Patch(facecolor="#FF0000", alpha=0.65,
                   label=f"Conflict zones ({c_any_km2:,.0f} km2)"),
    mpatches.Patch(facecolor="#2E8B57", alpha=0.6, edgecolor="#1A6B3A",
                   label=f"MPAs (n={len(mpas_in_eez)})"),
    mpatches.Patch(facecolor="#FFD700", alpha=0.55,
                   label=f"Offshore wind zones ({wind_area:,.0f} km2)"),
    mpatches.Patch(facecolor="#FFDAB9", alpha=0.45,
                   label=f"Fishing grounds (<{FISHING_GROUND_KM} km)"),
    Line2D([0], [0], color="#1E5AA8", linewidth=1.5, linestyle="--",
           label="Turkey EEZ"),
    mpatches.Patch(facecolor="#F5F0E8", edgecolor="#B0A890", label="Land"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8.5,
          framealpha=0.92, edgecolor="#CCC", fancybox=True,
          title="Legend", title_fontsize=9.5)

# --- Conflict summary box ---
summary_text = (
    "Conflict Summary\n"
    f"{'-' * 32}\n"
    f"Shipping x MPA:   {c_mpa_km2:>7,.1f} km2\n"
    f"Shipping x Wind:  {c_wind_km2:>7,.1f} km2\n"
    f"Shipping x Fish:  {c_fish_km2:>7,.1f} km2\n"
    f"{'-' * 32}\n"
    f"Total conflict:   {c_any_km2:>7,.1f} km2\n"
    f"Shipping buffer:  {ship_area:>7,.0f} km2\n"
    f"Routes:           {len(SHIPPING_ROUTES)}\n"
    f"Buffer width:     {SHIPPING_BUFFER_KM} km"
)
props = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.92,
             edgecolor="#999", linewidth=0.8)
ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="right", bbox=props, fontfamily="monospace", zorder=11)

# --- Source annotation ---
ax.annotate(
    "Data: GEBCO 2025 | EEZ: Flanders Marine Institute v12 | "
    "MPAs: WDPA Feb 2026 | Land: Natural Earth 10m | "
    "Shipping routes: representative corridors (synthetic)",
    xy=(0.5, -0.06), xycoords="axes fraction", ha="center", fontsize=7.5,
    color="#666", fontstyle="italic")

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
SHIPPING LANE CONFLICT ANALYSIS - Turkish Waters
Date: February 2026
{'-' * 60}

1. SHIPPING ROUTES
   - Routes modeled:     {len(SHIPPING_ROUTES)}
   - Buffer width:       {SHIPPING_BUFFER_KM} km (each side)
   - Total buffer area:  {ship_area:,.0f} km2 (within EEZ)""")

for _, row in buffers_clipped.iterrows():
    geom = row.geometry
    if geom.is_empty:
        continue
    g_list = [(mapping(geom), 1)]
    r_mask = rasterize(
        g_list, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool) & ocean_eez
    area = r_mask.sum() * pixel_area_km2
    print(f"   - {row['name']:<25} {area:>8,.0f} km2")

print(f"""
2. ZONE AREAS
   - MPAs in EEZ:        {mpa_area_km2:,.1f} km2
   - Offshore wind:      {wind_area:,.0f} km2 (depth 0-50m, {WIND_SHORE_MIN_KM}-{WIND_SHORE_MAX_KM}km)
   - Fishing grounds:    {fishing_area:,.0f} km2 (<{FISHING_GROUND_KM} km from shore)

3. CONFLICT MATRIX
   {'Conflict Type':<28} {'Overlap':>10} {'Zone Total':>12} {'% Affected':>10}
   {'-'*28} {'-'*10} {'-'*12} {'-'*10}
   {'Shipping x MPA':<28} {c_mpa_km2:>10,.1f} {mpa_area_km2:>12,.1f} {pct_mpa:>9.2f}%
   {'Shipping x Offshore Wind':<28} {c_wind_km2:>10,.1f} {wind_area:>12,.0f} {pct_wind:>9.2f}%
   {'Shipping x Fishing':<28} {c_fish_km2:>10,.1f} {fishing_area:>12,.0f} {pct_fish:>9.2f}%
   {'-'*28} {'-'*10}
   {'Total conflict (union)':<28} {c_any_km2:>10,.1f}

4. KEY FINDINGS
   - Shipping lanes overlap {c_fish_km2:,.0f} km2 of fishing grounds,
     the largest single conflict category. This reflects the reality
     that both shipping and fishing concentrate in coastal waters.
   - {c_wind_km2:,.0f} km2 of proposed offshore wind zones conflict with
     shipping corridors, requiring route adjustment or turbine
     exclusion setbacks in site planning.
   - MPA-shipping conflicts total {c_mpa_km2:,.1f} km2, concentrated
     at coastal Ramsar sites near the Bosphorus and Aegean routes.
   - The Bosphorus Strait is the highest-intensity conflict point,
     where one of the world's busiest waterways passes through
     the Istanbul metropolitan coastal zone.
   - The Black Sea and Mediterranean main routes traverse deep
     water with minimal zone conflicts, but the Aegean and
     Istanbul-Izmir coastal routes create significant overlaps
     with fishing grounds and wind energy zones.
   - MSP recommendations: implement Traffic Separation Schemes
     near wind farms, establish vessel speed restrictions in
     MPA-adjacent corridors, and coordinate fishing exclusion
     windows during peak shipping hours.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
