"""
Tidal Energy Resource Assessment - Turkish Waters
===================================================
Generates synthetic tidal current velocity fields for Turkish strait
and coastal systems, computes power density, and identifies viable
tidal energy sites after applying depth, shipping, and MPA constraints.
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping, LineString
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

OUTPUT_DIR = BASE_DIR / "project 12 tidal energy"
OUTPUT_PNG = OUTPUT_DIR / "turkey_tidal_energy.png"

RHO = 1025.0           # seawater density kg/m3
DEPTH_LIMIT = 40        # max depth for tidal turbines (m)
SHIP_BUFFER_KM = 5      # exclusion around shipping corridors

# Strait / channel shipping routes to exclude
STRAIT_ROUTES = [
    # Bosphorus corridor
    [(28.98, 41.00), (29.00, 41.05), (29.03, 41.10), (29.06, 41.15), (29.10, 41.20)],
    # Dardanelles corridor
    [(26.18, 40.05), (26.30, 40.10), (26.45, 40.15), (26.60, 40.20), (26.70, 40.25)],
]

# Tidal current hotspots (lon, lat, peak velocity m/s, radius km)
# These define Gaussian-shaped current centres
TIDAL_HOTSPOTS = [
    # Bosphorus - world-class currents
    (29.04, 41.08, 3.2, 3),   # central Bosphorus narrows
    (29.02, 41.04, 2.8, 3),   # southern Bosphorus
    (29.06, 41.14, 2.5, 3),   # northern Bosphorus
    # Dardanelles
    (26.40, 40.12, 2.2, 5),   # central Dardanelles
    (26.25, 40.07, 1.8, 4),   # southern Dardanelles
    (26.60, 40.20, 1.6, 4),   # northern Dardanelles
    # Aegean island channels
    (26.15, 39.10, 1.2, 8),   # Chios Strait
    (27.00, 37.60, 0.9, 6),   # Kos Channel
    (26.50, 38.50, 0.8, 7),   # Lesvos channel
    (26.80, 40.40, 0.7, 5),   # near Marmara entrance
    # Black Sea coastal promontories
    (30.50, 41.20, 0.4, 10),  # Bolu coast
    (36.00, 41.70, 0.3, 12),  # Samsun cape
    (34.00, 42.00, 0.3, 10),  # Sinop peninsula
    # Mediterranean
    (30.00, 36.10, 0.2, 15),  # Antalya coast
    (35.00, 36.30, 0.15, 12), # Mersin coast
]

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("TIDAL ENERGY RESOURCE ASSESSMENT - Turkish Waters")
print("=" * 70)

print("\n[1/8] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_bounds = turkey_eez_dissolved.total_bounds
print(f"  EEZ loaded ({turkey_eez['AREA_KM2'].sum():,.0f} km2)")

print("\n[2/8] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

print("\n[3/8] Loading MPAs...")
mpa_frames = []
for i in range(3):
    shp = (MPA_BASE / f"WDPA_WDOECM_Feb2026_Public_TUR_shp_{i}" /
           "WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons.shp")
    if shp.exists():
        mpa_frames.append(gpd.read_file(shp))
mpas = pd.concat(mpa_frames, ignore_index=True)
mpas = gpd.GeoDataFrame(mpas, geometry="geometry", crs="EPSG:4326").drop_duplicates(subset="SITE_ID")
mpas_in_eez = gpd.clip(mpas, turkey_eez_dissolved)
mpas_in_eez = mpas_in_eez[~mpas_in_eez.is_empty]
print(f"  {len(mpas_in_eez)} MPAs in EEZ")

print("\n[4/8] Loading GEBCO bathymetry...")
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
# 2. BUILD MASKS
# ============================================================================
print("\n[5/8] Building masks...")

# Coordinate grids
cols = np.arange(bathy_shape[1])
rows = np.arange(bathy_shape[0])
lon_arr = bathy_bounds.left + cols * pixel_res[1]
lat_arr = bathy_bounds.top - rows * pixel_res[0]
lon_grid = np.broadcast_to(lon_arr[np.newaxis, :], bathy_shape).astype(np.float32)
lat_grid = np.broadcast_to(lat_arr[:, np.newaxis], bathy_shape).astype(np.float32)

# EEZ
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# Land
land_shapes = [(mapping(g), 1) for g in land_clip.geometry if g is not None]
land_raster = rasterize(
    land_shapes, out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
land_raster = land_raster | (bathy > 0)

ocean_eez = eez_mask & ~land_raster

# Depth constraint: only areas <= DEPTH_LIMIT m deep
depth = -bathy  # positive depth below sea level
depth_ok = (depth > 0) & (depth <= DEPTH_LIMIT)

# MPA exclusion
if len(mpas_in_eez) > 0:
    mpa_shapes = [(mapping(g), 1) for g in mpas_in_eez.geometry
                  if g is not None and not g.is_empty]
    mpa_raster = rasterize(
        mpa_shapes, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool)
else:
    mpa_raster = np.zeros(bathy_shape, dtype=bool)

# Shipping exclusion (buffer around strait routes)
print("  Building shipping exclusion zones...")
route_lines = [LineString(coords) for coords in STRAIT_ROUTES]
routes_gdf = gpd.GeoDataFrame(geometry=route_lines, crs="EPSG:4326")
routes_proj = routes_gdf.to_crs("EPSG:32636")
buffer_geoms = routes_proj.geometry.buffer(SHIP_BUFFER_KM * 1000)
buffers_proj = gpd.GeoDataFrame(geometry=buffer_geoms.values, crs="EPSG:32636")
buffers_4326 = buffers_proj.to_crs("EPSG:4326")
all_buf = unary_union(buffers_4326.geometry)

if not all_buf.is_empty:
    buf_shapes = []
    if all_buf.geom_type == "MultiPolygon":
        for p in all_buf.geoms:
            buf_shapes.append((mapping(p), 1))
    else:
        buf_shapes.append((mapping(all_buf), 1))
    ship_raster = rasterize(
        buf_shapes, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool)
else:
    ship_raster = np.zeros(bathy_shape, dtype=bool)

print(f"  Shipping exclusion pixels: {ship_raster.sum():,}")
print(f"  MPA exclusion pixels: {mpa_raster.sum():,}")

# ============================================================================
# 3. GENERATE TIDAL CURRENT VELOCITY FIELD
# ============================================================================
print("\n[6/8] Generating tidal current velocity field...")

velocity = np.full(bathy_shape, 0.05, dtype=np.float32)  # background 0.05 m/s

for hlon, hlat, v_peak, radius_km in TIDAL_HOTSPOTS:
    d_lon = (lon_grid - hlon) * km_per_deg_lon
    d_lat = (lat_grid - hlat) * km_per_deg_lat
    dist = np.sqrt(d_lon**2 + d_lat**2)
    contribution = v_peak * np.exp(-0.5 * (dist / radius_km)**2)
    velocity = np.maximum(velocity, contribution)

# Add slight noise for realism
rng = np.random.default_rng(42)
velocity += rng.normal(0, 0.02, bathy_shape).astype(np.float32)
velocity = np.clip(velocity, 0.01, 5.0)

# Light smoothing
velocity = gaussian_filter(velocity, sigma=1.5).astype(np.float32)
velocity = np.clip(velocity, 0.01, 5.0)

# Mask: only ocean in EEZ
velocity_display = np.where(ocean_eez, velocity, np.nan)

# Stats
v_ocean = velocity[ocean_eez]
print(f"  Velocity range: {np.nanmin(v_ocean):.3f} - {np.nanmax(v_ocean):.3f} m/s")
print(f"  Mean velocity:  {np.nanmean(v_ocean):.3f} m/s")

# ============================================================================
# 4. COMPUTE POWER DENSITY
# ============================================================================
print("\n[7/8] Computing tidal power density...")

# P = 0.5 * rho * v^3 (W/m2)
power_density = 0.5 * RHO * velocity**3

power_display = np.where(ocean_eez, power_density, np.nan)

# Suitable zones: ocean + depth + not shipping + not MPA + power > 100 W/m2
suitable = ocean_eez & depth_ok & ~ship_raster & ~mpa_raster & (power_density >= 100)
marginal = ocean_eez & depth_ok & ~ship_raster & ~mpa_raster & (power_density >= 10) & (power_density < 100)

suitable_area = suitable.sum() * pixel_area_km2
marginal_area = marginal.sum() * pixel_area_km2

# Technical power potential
# Assume 30% capture efficiency, turbine swept area factor
# Power per km2: average power density * 1e6 m2 * efficiency
eff = 0.30
suitable_mean_power = np.nanmean(power_density[suitable]) if suitable.any() else 0
# Total technical potential in MW
# For each suitable pixel: P_density * pixel_area_m2 * efficiency / 1e6
pixel_area_m2 = pixel_area_km2 * 1e6
if suitable.any():
    total_power_mw = np.sum(power_density[suitable]) * pixel_area_m2 * eff / 1e6
    total_power_gw = total_power_mw / 1000
else:
    total_power_mw = 0
    total_power_gw = 0

print(f"  Suitable area (>=100 W/m2):  {suitable_area:,.1f} km2")
print(f"  Marginal area (10-100 W/m2): {marginal_area:,.1f} km2")
print(f"  Mean power density (suitable): {suitable_mean_power:,.0f} W/m2")
print(f"  Technical potential: {total_power_mw:,.0f} MW ({total_power_gw:,.1f} GW)")

# Per-hotspot analysis
print("\n  Key site analysis:")
hotspot_results = []
key_sites = [
    ("Bosphorus", 29.04, 41.08, 8),
    ("Dardanelles", 26.40, 40.12, 10),
    ("Aegean Channels", 26.15, 39.10, 15),
]
for sname, slon, slat, srad in key_sites:
    d_lon = (lon_grid - slon) * km_per_deg_lon
    d_lat = (lat_grid - slat) * km_per_deg_lat
    dist = np.sqrt(d_lon**2 + d_lat**2)
    site_mask = (dist <= srad) & ocean_eez
    if site_mask.any():
        sv = velocity[site_mask]
        sp = power_density[site_mask]
        ss = suitable & site_mask
        sa = ss.sum() * pixel_area_km2
        smw = np.sum(power_density[ss]) * pixel_area_m2 * eff / 1e6 if ss.any() else 0
        print(f"    {sname:<18} v_max={np.max(sv):.2f} m/s  "
              f"P_max={np.max(sp):,.0f} W/m2  "
              f"suitable={sa:.1f} km2  potential={smw:,.0f} MW")
        hotspot_results.append({
            "name": sname, "v_max": np.max(sv), "p_max": np.max(sp),
            "area_km2": sa, "mw": smw
        })

# ============================================================================
# 5. CREATE MAP
# ============================================================================
print("\n[8/8] Creating map...")

fig = plt.figure(figsize=(18, 12), facecolor="#0d1b2a")

# Main map
ax = fig.add_axes([0.05, 0.08, 0.82, 0.84])
ax.set_facecolor("#0d1b2a")

pad = 0.6
ax.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

# Custom colormap: dark blue -> cyan -> white
power_colors = ["#0d1b2a", "#0a3055", "#0f4c81", "#1a7cb8",
                "#35a8d5", "#5ccde6", "#8eeef7", "#c0fcff", "#ffffff"]
cmap_power = LinearSegmentedColormap.from_list("tidal", power_colors, N=256)
cmap_power.set_bad(color="#0d1b2a", alpha=0)

extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]

# Power density (log scale for dynamic range)
# Clip to min 0.01 for log
power_log_display = np.where(ocean_eez, np.clip(power_density, 0.01, None), np.nan)
im = ax.imshow(power_log_display, extent=extent, origin="upper",
               cmap=cmap_power, norm=LogNorm(vmin=0.01, vmax=20000),
               zorder=2, aspect="auto", interpolation="bilinear")

# Land
land.plot(ax=ax, color="#2a2a2a", edgecolor="#444", linewidth=0.3, zorder=3)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax, color="#3388aa", linewidth=1.0,
                                    linestyle="--", alpha=0.5, zorder=4)

# Suitable zones outline
if suitable.any():
    from matplotlib.colors import ListedColormap
    suit_display = np.where(suitable, 1.0, np.nan)
    cmap_suit = ListedColormap(["#00ffff"])
    cmap_suit.set_bad(alpha=0)
    ax.imshow(suit_display, extent=extent, origin="upper",
              cmap=cmap_suit, alpha=0.4, zorder=5, aspect="auto",
              interpolation="nearest")

# Shipping exclusion zones (faint red)
ship_display = np.where(ship_raster & ocean_eez, 1.0, np.nan)
cmap_ship = ListedColormap(["#ff4444"])
cmap_ship.set_bad(alpha=0)
ax.imshow(ship_display, extent=extent, origin="upper",
          cmap=cmap_ship, alpha=0.25, zorder=5, aspect="auto",
          interpolation="nearest")

# --- Labels ---
# Bosphorus
ax.annotate("BOSPHORUS\nSTRAIT", xy=(29.04, 41.08), fontsize=10, fontweight="bold",
            color="#00ffff", ha="left", va="center",
            xytext=(29.8, 41.5), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#00ffff", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d1b2a", ec="#00ffff", alpha=0.9),
            zorder=8)

# Dardanelles
ax.annotate("DARDANELLES\nSTRAIT", xy=(26.40, 40.12), fontsize=9, fontweight="bold",
            color="#00dddd", ha="right", va="center",
            xytext=(25.4, 40.6), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#00dddd", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d1b2a", ec="#00dddd", alpha=0.9),
            zorder=8)

# Colorbar
cbar_ax = fig.add_axes([0.05, 0.04, 0.78, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Tidal Power Density (W/m2)", fontsize=10, color="white", labelpad=6)
cbar.ax.tick_params(colors="white", labelsize=8)

# Title
ax.set_title("Tidal Energy Resource Assessment -- Turkish Waters",
             fontsize=18, fontweight="bold", color="white", pad=14)
ax.set_xlabel("Longitude", fontsize=10, color="white", labelpad=6)
ax.set_ylabel("Latitude", fontsize=10, color="white", labelpad=6)
ax.tick_params(colors="white", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor="#00ffff", alpha=0.4,
                   label=f"Suitable sites (>100 W/m2, {suitable_area:,.1f} km2)"),
    mpatches.Patch(facecolor="#ff4444", alpha=0.25,
                   label="Shipping exclusion zone"),
    Line2D([0], [0], color="#3388aa", linewidth=1, linestyle="--", alpha=0.5,
           label="Turkey EEZ"),
    mpatches.Patch(facecolor="#2a2a2a", edgecolor="#444", label="Land"),
]
leg = ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
                framealpha=0.85, edgecolor="#3388aa", fancybox=True,
                labelcolor="white", facecolor="#0d1b2a")
leg.get_title()

# --- Summary box ---
summary = (
    "Tidal Energy Potential\n"
    f"{'-' * 30}\n"
    f"Suitable area:  {suitable_area:,.1f} km2\n"
    f"Marginal area:  {marginal_area:,.1f} km2\n"
    f"Mean P (suit):  {suitable_mean_power:,.0f} W/m2\n"
    f"{'-' * 30}\n"
    f"Technical pot.:  {total_power_mw:,.0f} MW\n"
    f"                ({total_power_gw:,.1f} GW)\n"
    f"Efficiency:      {eff:.0%}\n"
    f"Depth limit:     {DEPTH_LIMIT}m\n"
    f"{'-' * 30}\n"
    f"P = 0.5*rho*v^3\n"
    f"rho = {RHO:.0f} kg/m3"
)
props = dict(boxstyle="round,pad=0.5", facecolor="#0d1b2a", alpha=0.9,
             edgecolor="#00bbcc", linewidth=0.8)
ax.text(0.98, 0.98, summary, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right", bbox=props, fontfamily="monospace",
        color="#ccffff", zorder=10)

# --- INSET: Bosphorus zoom ---
ax_inset = fig.add_axes([0.60, 0.52, 0.28, 0.38])
ax_inset.set_facecolor("#0d1b2a")

# Bosphorus extent
bos_lon_min, bos_lon_max = 28.85, 29.25
bos_lat_min, bos_lat_max = 40.95, 41.25
ax_inset.set_xlim(bos_lon_min, bos_lon_max)
ax_inset.set_ylim(bos_lat_min, bos_lat_max)

ax_inset.imshow(power_log_display, extent=extent, origin="upper",
                cmap=cmap_power, norm=LogNorm(vmin=0.01, vmax=20000),
                zorder=2, aspect="auto", interpolation="bilinear")

land.plot(ax=ax_inset, color="#2a2a2a", edgecolor="#555", linewidth=0.5, zorder=3)

# Suitable zones in inset
if suitable.any():
    ax_inset.imshow(suit_display, extent=extent, origin="upper",
                    cmap=cmap_suit, alpha=0.35, zorder=5, aspect="auto",
                    interpolation="nearest")

# Shipping exclusion in inset
ax_inset.imshow(ship_display, extent=extent, origin="upper",
                cmap=cmap_ship, alpha=0.3, zorder=5, aspect="auto",
                interpolation="nearest")

ax_inset.set_title("Bosphorus Strait (Detail)", fontsize=9,
                    fontweight="bold", color="#00ffff", pad=5)
ax_inset.tick_params(colors="white", labelsize=6)
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#00bbcc")
    spine.set_linewidth(1.5)

# Velocity annotation in inset
ax_inset.text(0.5, 0.05, "Peak: 3.2 m/s | ~16,800 W/m2",
              transform=ax_inset.transAxes, fontsize=7.5, color="#00ffff",
              ha="center", va="bottom",
              bbox=dict(fc="#0d1b2a", ec="#00bbcc", alpha=0.85, pad=2))

# Mark inset area on main map
rect = mpatches.Rectangle((bos_lon_min, bos_lat_min),
                            bos_lon_max - bos_lon_min,
                            bos_lat_max - bos_lat_min,
                            linewidth=1.5, edgecolor="#00ffff",
                            facecolor="none", linestyle="-", zorder=9)
ax.add_patch(rect)

# Source annotation
fig.text(0.5, 0.005,
         "Methodology: P = 0.5 * rho * v^3 | Currents: synthetic basin model | "
         "Bathymetry: GEBCO 2025 | EEZ: Flanders Marine Institute v12",
         ha="center", fontsize=7, color="#667788", fontstyle="italic")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="#0d1b2a")
plt.close()
print(f"\nMap saved: {OUTPUT_PNG}")

# ============================================================================
# 6. FULL SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY REPORT")
print("=" * 70)

print(f"""
TIDAL ENERGY RESOURCE ASSESSMENT - Turkish Waters
Date: February 2026
{'-' * 60}

1. METHODOLOGY
   - Power density:   P = 0.5 * rho * v^3
   - Seawater rho:    {RHO:.0f} kg/m3
   - Capture eff.:    {eff:.0%}
   - Depth limit:     {DEPTH_LIMIT}m (tidal turbine constraint)
   - Shipping excl.:  {SHIP_BUFFER_KM} km buffer (Bosphorus & Dardanelles)
   - MPA exclusion:   {len(mpas_in_eez)} sites removed

2. VELOCITY FIELD
   - EEZ mean:        {np.nanmean(v_ocean):.3f} m/s
   - EEZ max:         {np.nanmax(v_ocean):.2f} m/s
   - Bosphorus peak:  3.2 m/s
   - Dardanelles peak:2.2 m/s

3. RESOURCE SUMMARY
   - Suitable area (>=100 W/m2):  {suitable_area:,.1f} km2
   - Marginal area (10-100 W/m2): {marginal_area:,.1f} km2
   - Mean power (suitable):       {suitable_mean_power:,.0f} W/m2
   - Technical potential:          {total_power_mw:,.0f} MW ({total_power_gw:,.1f} GW)

4. KEY SITES""")

for hr in hotspot_results:
    print(f"   {hr['name']:<18} v_max={hr['v_max']:.2f} m/s  "
          f"P_max={hr['p_max']:>8,.0f} W/m2  "
          f"area={hr['area_km2']:.1f} km2  pot={hr['mw']:,.0f} MW")

print(f"""
5. KEY FINDINGS
   - The Bosphorus Strait is Turkey's premier tidal energy site,
     with peak currents of 3.2 m/s yielding ~16,800 W/m2 power
     density -- among the highest in the world.
   - The Dardanelles Strait offers a secondary resource with
     sustained currents of 1.5-2.2 m/s along its 60 km length.
   - Shipping lane exclusions significantly reduce the exploitable
     area in both straits, as they are critical maritime corridors
     handling 40,000+ vessels/year (Bosphorus).
   - Aegean island channels provide modest tidal resources (0.5-1.2
     m/s) suitable for small community-scale turbines.
   - Black Sea and Mediterranean coasts have negligible tidal
     resources (<0.5 m/s) unsuitable for energy extraction.
   - Technical potential of {total_power_mw:,.0f} MW assumes {eff:.0%} capture
     efficiency across all suitable pixels -- realistic deployments
     would target the highest-power 10-20% of sites.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
