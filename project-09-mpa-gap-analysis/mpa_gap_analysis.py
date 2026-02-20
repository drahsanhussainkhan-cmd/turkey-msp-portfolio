"""
Marine Protected Area Gap Analysis - Turkish Waters
=====================================================
Identifies which depth-based habitat zones are underrepresented
inside Turkey's MPA network relative to the CBD 30x30 target.
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

OUTPUT_DIR = BASE_DIR / "project 9 mpa gap analysis"
OUTPUT_PNG = OUTPUT_DIR / "turkey_mpa_gap_analysis.png"

CBD_TARGET = 30.0  # CBD 30x30 target percent

# Habitat zones: (label, depth_min, depth_max, color)
# depth values are NEGATIVE (below sea level)
HABITAT_ZONES = [
    ("Intertidal",    0,   -10,   "#a6dba0"),
    ("Shallow Shelf", -10,  -50,  "#5aae61"),
    ("Mid Shelf",     -50,  -200, "#1b7837"),
    ("Deep Shelf",    -200, -500, "#762a83"),
    ("Deep Sea",      -500, -6000, "#40004b"),
]

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("MPA GAP ANALYSIS - Turkish Waters")
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
mpas_in_eez = gpd.clip(mpas, turkey_eez_dissolved)
mpas_in_eez = mpas_in_eez[~mpas_in_eez.is_empty]
print(f"  {len(mpas)} total MPAs, {len(mpas_in_eez)} intersecting EEZ")

# List MPA names
for _, row in mpas_in_eez.iterrows():
    name = row.get("NAME_ENG") or row.get("NAME") or "Unknown"
    print(f"    - {name}")

print("\n[4/6] Loading GEBCO bathymetry...")
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
print("\n[5/6] Building analysis layers...")

# EEZ mask
eez_mask = rasterize(
    [(mapping(eez_geom), 1)],
    out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)

# Land mask
land_shapes = [(mapping(geom), 1) for geom in land_clip.geometry if geom is not None]
land_raster = rasterize(
    land_shapes, out_shape=bathy_shape, transform=bathy_transform,
    fill=0, dtype=np.uint8
).astype(bool)
land_raster = land_raster | (bathy > 0)

# Ocean in EEZ
ocean_eez = eez_mask & ~land_raster

# MPA mask
if len(mpas_in_eez) > 0:
    mpa_shapes = [(mapping(geom), 1) for geom in mpas_in_eez.geometry
                  if geom is not None and not geom.is_empty]
    mpa_raster = rasterize(
        mpa_shapes, out_shape=bathy_shape, transform=bathy_transform,
        fill=0, dtype=np.uint8
    ).astype(bool)
else:
    mpa_raster = np.zeros(bathy_shape, dtype=bool)

mpa_ocean = mpa_raster & ocean_eez
print(f"  EEZ ocean pixels: {ocean_eez.sum():,}")
print(f"  MPA ocean pixels: {mpa_ocean.sum():,}")

# ============================================================================
# 3. HABITAT ZONE CLASSIFICATION AND GAP ANALYSIS
# ============================================================================
print("\n[6/6] Computing gap analysis per habitat zone...")

# Classify each ocean pixel into a habitat zone
habitat_grid = np.full(bathy_shape, -1, dtype=np.int8)
for i, (label, d_upper, d_lower, _) in enumerate(HABITAT_ZONES):
    # d_upper is the shallower boundary (closer to 0), d_lower is deeper (more negative)
    mask = ocean_eez & (bathy <= d_upper) & (bathy > d_lower)
    habitat_grid[mask] = i

# Compute stats per zone
gap_results = []
for i, (label, d_upper, d_lower, color) in enumerate(HABITAT_ZONES):
    zone_mask = habitat_grid == i
    total_px = zone_mask.sum()
    mpa_px = (zone_mask & mpa_ocean).sum()
    total_area = total_px * pixel_area_km2
    mpa_area = mpa_px * pixel_area_km2
    pct_protected = (mpa_area / total_area * 100) if total_area > 0 else 0
    gap_pct = max(0, CBD_TARGET - pct_protected)
    gap_area = (gap_pct / 100.0) * total_area

    gap_results.append({
        "zone": label,
        "depth_range": f"{d_upper}m to {d_lower}m",
        "total_km2": total_area,
        "mpa_km2": mpa_area,
        "pct_protected": pct_protected,
        "gap_pct": gap_pct,
        "gap_area_km2": gap_area,
        "color": color,
    })

gap_df = pd.DataFrame(gap_results)

# Print table
print(f"\n  {'Zone':<16} {'Depth':<16} {'Total km2':>12} {'MPA km2':>10} "
      f"{'Protected':>10} {'Gap to 30%':>10} {'Gap km2':>10}")
print(f"  {'-'*16} {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for _, r in gap_df.iterrows():
    print(f"  {r['zone']:<16} {r['depth_range']:<16} {r['total_km2']:>12,.0f} "
          f"{r['mpa_km2']:>10,.1f} {r['pct_protected']:>9.2f}% "
          f"{r['gap_pct']:>9.2f}% {r['gap_area_km2']:>10,.0f}")

total_ocean = gap_df["total_km2"].sum()
total_mpa = gap_df["mpa_km2"].sum()
overall_pct = (total_mpa / total_ocean * 100) if total_ocean > 0 else 0
overall_gap_area = max(0, (CBD_TARGET / 100.0) * total_ocean - total_mpa)

print(f"\n  Overall: {total_ocean:,.0f} km2 ocean, {total_mpa:,.1f} km2 protected "
      f"({overall_pct:.2f}%), gap = {overall_gap_area:,.0f} km2 needed for 30%")

# ============================================================================
# 4. CREATE FIGURE (two panels)
# ============================================================================
print("\nCreating figure...")

fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(20, 10), facecolor="white",
                                      gridspec_kw={"width_ratios": [1.3, 1]})

# ---- PANEL A: HABITAT ZONE MAP ----
ax_map.set_facecolor("#AED9E0")
pad = 0.8
ax_map.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax_map.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

# Habitat zones raster
zone_colors = [hz[3] for hz in HABITAT_ZONES]
cmap_zones = ListedColormap(zone_colors)
bounds_z = [-0.5 + i for i in range(len(HABITAT_ZONES) + 1)]
norm_z = BoundaryNorm(bounds_z, cmap_zones.N)
cmap_zones.set_bad(alpha=0)

habitat_display = np.where(ocean_eez, habitat_grid.astype(float), np.nan)
habitat_display[habitat_grid == -1] = np.nan

extent = [bathy_bounds.left, bathy_bounds.right, bathy_bounds.bottom, bathy_bounds.top]
ax_map.imshow(habitat_display, extent=extent, origin="upper",
              cmap=cmap_zones, norm=norm_z, alpha=0.75, zorder=2,
              aspect="auto", interpolation="nearest")

# Land
land.plot(ax=ax_map, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.3, zorder=3)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax_map, color="#1E5AA8", linewidth=1.5,
                                    linestyle="--", zorder=4)

# MPAs with hatching for visibility
if len(mpas_in_eez) > 0:
    mpas_in_eez.plot(ax=ax_map, facecolor="none", edgecolor="#FF4500",
                     linewidth=1.5, hatch="///", zorder=5)

# Map legend
map_legend = []
for label, _, _, color in HABITAT_ZONES:
    map_legend.append(mpatches.Patch(facecolor=color, alpha=0.75,
                                     edgecolor="#666", label=label))
map_legend.append(mpatches.Patch(facecolor="none", edgecolor="#FF4500",
                                 hatch="///", label=f"MPAs (n={len(mpas_in_eez)})"))
map_legend.append(Line2D([0], [0], color="#1E5AA8", linewidth=1.5,
                          linestyle="--", label="Turkey EEZ"))
ax_map.legend(handles=map_legend, loc="lower left", fontsize=8,
              framealpha=0.92, edgecolor="#CCC", fancybox=True,
              title="Habitat Zones", title_fontsize=9)

ax_map.set_title("A. Habitat Zones with MPA Overlay",
                 fontsize=13, fontweight="bold", pad=10)
ax_map.set_xlabel("Longitude", fontsize=10)
ax_map.set_ylabel("Latitude", fontsize=10)
ax_map.tick_params(labelsize=8)
ax_map.grid(True, linestyle=":", alpha=0.3, color="#666")

# Summary box on map
map_summary = (
    f"Overall MPA coverage\n"
    f"{'-' * 26}\n"
    f"EEZ ocean:  {total_ocean:,.0f} km2\n"
    f"Protected:  {total_mpa:,.1f} km2\n"
    f"Coverage:   {overall_pct:.2f}%\n"
    f"CBD target: {CBD_TARGET:.0f}%\n"
    f"Gap:        {overall_gap_area:,.0f} km2"
)
props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92,
             edgecolor="#999", linewidth=0.8)
ax_map.text(0.98, 0.98, map_summary, transform=ax_map.transAxes, fontsize=8.5,
            va="top", ha="right", bbox=props, fontfamily="monospace", zorder=10)

# ---- PANEL B: BAR CHART ----
ax_bar.set_facecolor("white")

zones = gap_df["zone"].tolist()
pct_vals = gap_df["pct_protected"].tolist()
colors = gap_df["color"].tolist()
y_pos = np.arange(len(zones))

bars = ax_bar.barh(y_pos, pct_vals, height=0.6, color=colors, alpha=0.85,
                    edgecolor="#555", linewidth=0.6)

# 30% target line
ax_bar.axvline(x=CBD_TARGET, color="#D32F2F", linewidth=2.2, linestyle="--",
               label=f"CBD 30x30 Target ({CBD_TARGET:.0f}%)", zorder=5)

# Shade the gap region for each bar
for i, (pct, total) in enumerate(zip(pct_vals, gap_df["total_km2"])):
    if pct < CBD_TARGET:
        gap_needed = (CBD_TARGET - pct) / 100.0 * total
        ax_bar.barh(i, CBD_TARGET - pct, left=pct, height=0.6,
                    color="#FFCDD2", alpha=0.6, edgecolor="#E57373", linewidth=0.5)

# Value labels on bars
for i, (pct, mpa_a, total_a) in enumerate(zip(pct_vals,
                                                gap_df["mpa_km2"],
                                                gap_df["total_km2"])):
    if pct > 1:
        ax_bar.text(pct - 0.3, i, f"{pct:.1f}%", va="center", ha="right",
                    fontsize=9, fontweight="bold", color="white")
    else:
        ax_bar.text(pct + 0.5, i, f"{pct:.2f}%", va="center", ha="left",
                    fontsize=9, fontweight="bold", color="#333")

    # Gap annotation on right
    gap_km = max(0, (CBD_TARGET / 100.0) * total_a - mpa_a)
    if gap_km > 0:
        ax_bar.text(CBD_TARGET + 1, i, f"+{gap_km:,.0f} km2 needed",
                    va="center", ha="left", fontsize=7.5, color="#B71C1C",
                    fontstyle="italic")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(zones, fontsize=10)
ax_bar.set_xlabel("% Protected", fontsize=11)
ax_bar.set_xlim(0, max(CBD_TARGET * 2, max(pct_vals) * 1.5, 65))
ax_bar.invert_yaxis()
ax_bar.set_title("B. Protection Gap per Habitat Zone",
                 fontsize=13, fontweight="bold", pad=10)
ax_bar.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax_bar.grid(axis="x", linestyle=":", alpha=0.4)
ax_bar.tick_params(labelsize=9)

# Pink = gap shading note
ax_bar.annotate("Pink shading = additional area needed to reach 30% target",
                xy=(0.5, -0.08), xycoords="axes fraction", ha="center",
                fontsize=8, color="#B71C1C", fontstyle="italic")

# ---- SUPER TITLE ----
fig.suptitle("Marine Protected Area Gap Analysis -- Turkish Waters",
             fontsize=18, fontweight="bold", color="#1A1A2E", y=1.01)

# ---- Source annotation ----
fig.text(0.5, -0.03,
         "Data: GEBCO 2025 bathymetry | EEZ: Flanders Marine Institute v12 | "
         "MPAs: WDPA Feb 2026 | Land: Natural Earth 10m | "
         "Target: CBD Kunming-Montreal 30x30 Framework",
         ha="center", fontsize=7.5, color="#666", fontstyle="italic")

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

print(f"""
MPA GAP ANALYSIS - Turkish Waters
Date: February 2026
Reference target: CBD Kunming-Montreal Global Biodiversity Framework (30x30)
{'-' * 60}

1. DATA SOURCES
   - Bathymetry:  GEBCO 2025 (~{res_m:.0f}m resolution)
   - EEZ:         Flanders Marine Institute v12
   - MPAs:        WDPA Feb 2026 ({len(mpas_in_eez)} sites in EEZ)
   - Habitat:     Depth-based classification (5 zones)

2. GAP ANALYSIS TABLE""")

print(f"   {'Zone':<16} {'Depth':<16} {'Total':>10} {'In MPA':>10} "
      f"{'Prot.':>8} {'Gap':>8} {'Needed':>10}")
print(f"   {'':.<16} {'':.<16} {'(km2)':>10} {'(km2)':>10} "
      f"{'(%)':>8} {'(%)':>8} {'(km2)':>10}")
print(f"   {'-'*90}")
for _, r in gap_df.iterrows():
    print(f"   {r['zone']:<16} {r['depth_range']:<16} {r['total_km2']:>10,.0f} "
          f"{r['mpa_km2']:>10,.1f} {r['pct_protected']:>7.2f}% "
          f"{r['gap_pct']:>7.2f}% {r['gap_area_km2']:>10,.0f}")
print(f"   {'-'*90}")
print(f"   {'TOTAL':<16} {'':16} {total_ocean:>10,.0f} "
      f"{total_mpa:>10,.1f} {overall_pct:>7.2f}% "
      f"{max(0,CBD_TARGET-overall_pct):>7.2f}% {overall_gap_area:>10,.0f}")

print(f"""
3. MPA SITES IN EEZ""")
for _, row in mpas_in_eez.iterrows():
    name = row.get("NAME_ENG") or row.get("NAME") or "Unknown"
    desig = row.get("DESIG_ENG") or row.get("DESIG") or ""
    realm = row.get("REALM", "")
    print(f"   - {name} ({desig}) [{realm}]")

# Identify which zone has the largest gap
worst = gap_df.loc[gap_df["gap_area_km2"].idxmax()]
best_prot = gap_df.loc[gap_df["pct_protected"].idxmax()]

print(f"""
4. KEY FINDINGS
   - Turkey's overall marine protection stands at {overall_pct:.2f}% of its EEZ,
     far below the CBD 30x30 target of {CBD_TARGET:.0f}%.
   - An additional {overall_gap_area:,.0f} km2 would need to be designated
     as protected to reach the 30% target.
   - The largest absolute gap is in the {worst['zone']} zone
     ({worst['gap_area_km2']:,.0f} km2 of additional protection needed).
   - All five habitat zones are critically underprotected, with no
     zone exceeding {best_prot['pct_protected']:.1f}% coverage.
   - Current MPAs are predominantly coastal Ramsar wetland sites that
     provide minimal coverage of shelf and deep-sea habitats.
   - Priority expansion areas should include:
     * Continental shelf habitats (shallow + mid shelf) for fisheries
       and biodiversity
     * Deep-sea zones in the Black Sea and eastern Mediterranean
       for unique chemosynthetic and cold-water coral communities
   - Turkey lacks any large offshore marine reserves, unlike
     Mediterranean neighbors (e.g., Greece, Italy).

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
