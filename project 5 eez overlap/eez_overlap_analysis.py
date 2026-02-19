"""
EEZ Overlap Analysis: Fishing Activity vs Marine Protected Areas in Turkish Waters
==================================================================================
Generates synthetic AIS fishing vessel data, overlays with Turkey's EEZ and MPA
boundaries, and produces a professional map with spatial statistics.
"""

import subprocess
import sys

# Install missing packages
for pkg in ["geopandas", "matplotlib", "shapely", "numpy", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp")

EEZ_PATH = BASE_DIR / "downloads" / "World_EEZ_v12_20231025" / "World_EEZ_v12_20231025" / "eez_v12.shp"
LAND_PATH = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"
MPA_BASE = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 5 eez overlap"
OUTPUT_PNG = OUTPUT_DIR / "turkey_eez_overlap_analysis.png"

N_FISHING_POINTS = 5000
RANDOM_SEED = 42

# Turkish coastal fishing ground centers (lon, lat) with spread (std_dev)
FISHING_HOTSPOTS = [
    # Black Sea coast
    (36.5, 41.6, 0.8, 0.3),   # Samsun-Trabzon fishing grounds
    (32.0, 41.8, 0.5, 0.2),   # Sinop-Kastamonu
    (28.5, 41.5, 0.6, 0.3),   # Istanbul-Bosphorus approach
    (40.5, 41.0, 0.4, 0.2),   # Eastern Black Sea
    # Aegean Sea
    (26.5, 39.5, 0.5, 0.4),   # North Aegean
    (26.0, 38.0, 0.6, 0.5),   # Central Aegean (Izmir offshore)
    (27.5, 37.0, 0.4, 0.3),   # South Aegean (Bodrum-Marmaris)
    # Mediterranean coast
    (30.5, 36.2, 0.8, 0.3),   # Antalya Bay
    (34.0, 36.0, 0.5, 0.2),   # Mersin-Adana coast
    (35.5, 36.2, 0.3, 0.2),   # Iskenderun Bay
    # Sea of Marmara
    (28.8, 40.7, 0.3, 0.15),  # Marmara Sea
]

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("EEZ OVERLAP ANALYSIS: Fishing Activity vs MPAs in Turkish Waters")
print("=" * 70)

# --- Turkey EEZ ---
print("\n[1/5] Loading Turkey EEZ boundary...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy()
turkey_eez = turkey_eez.to_crs("EPSG:4326")
print(f"  Found {len(turkey_eez)} Turkey EEZ polygon(s)")
print(f"  Total EEZ area (attribute): {turkey_eez['AREA_KM2'].sum():,.0f} km2")

# Dissolve into single polygon for simpler analysis
turkey_eez_dissolved = turkey_eez.dissolve()
eez_geom = turkey_eez_dissolved.geometry.iloc[0]
eez_bounds = turkey_eez_dissolved.total_bounds  # [minx, miny, maxx, maxy]
print(f"  Bounding box: [{eez_bounds[0]:.1f}, {eez_bounds[1]:.1f}] to [{eez_bounds[2]:.1f}, {eez_bounds[3]:.1f}]")

# --- MPA polygons (load from all 3 splits) ---
print("\n[2/5] Loading Turkey MPA polygons...")
mpa_frames = []
for i in range(3):
    shp_path = (MPA_BASE / f"WDPA_WDOECM_Feb2026_Public_TUR_shp_{i}" /
                "WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons.shp")
    if shp_path.exists():
        gdf = gpd.read_file(shp_path)
        mpa_frames.append(gdf)
        print(f"  Split {i}: {len(gdf)} polygons loaded")

mpas_all = pd.concat(mpa_frames, ignore_index=True)
mpas_all = gpd.GeoDataFrame(mpas_all, geometry="geometry", crs="EPSG:4326")

# Remove duplicates by SITE_ID
mpas_all = mpas_all.drop_duplicates(subset="SITE_ID")
print(f"  Total unique MPAs: {len(mpas_all)}")
print(f"  Realm breakdown: {dict(mpas_all['REALM'].value_counts())}")

# Clip MPAs to EEZ to get marine/coastal portions
mpas_in_eez = gpd.clip(mpas_all, turkey_eez_dissolved)
mpas_in_eez = mpas_in_eez[~mpas_in_eez.is_empty]
print(f"  MPAs intersecting EEZ: {len(mpas_in_eez)}")

# --- Land ---
print("\n[3/5] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
print(f"  {len(land)} land polygons loaded")

# ============================================================================
# 2. GENERATE SYNTHETIC AIS FISHING DATA
# ============================================================================
print("\n[4/5] Generating synthetic AIS fishing vessel data...")
rng = np.random.default_rng(RANDOM_SEED)

# Allocate points across hotspots (weighted by spread = fishing intensity proxy)
weights = np.array([s[2] * s[3] for s in FISHING_HOTSPOTS])
weights = weights / weights.sum()
n_per_hotspot = rng.multinomial(N_FISHING_POINTS, weights)

lons, lats = [], []
for (cx, cy, sx, sy), n in zip(FISHING_HOTSPOTS, n_per_hotspot):
    lons.extend(rng.normal(cx, sx, n))
    lats.extend(rng.normal(cy, sy, n))

lons = np.array(lons)
lats = np.array(lats)

# Create GeoDataFrame
fishing_pts = gpd.GeoDataFrame(
    {"vessel_id": rng.integers(1000, 9999, N_FISHING_POINTS),
     "speed_knots": rng.uniform(0.5, 6.0, N_FISHING_POINTS).round(1)},
    geometry=[Point(x, y) for x, y in zip(lons, lats)],
    crs="EPSG:4326"
)

# Keep only points inside the EEZ
fishing_in_eez = fishing_pts[fishing_pts.within(eez_geom)].copy()
print(f"  Generated {N_FISHING_POINTS} raw points, {len(fishing_in_eez)} inside EEZ")

# If we lost too many, top up by shifting rejected points inward
if len(fishing_in_eez) < N_FISHING_POINTS * 0.8:
    shortfall = N_FISHING_POINTS - len(fishing_in_eez)
    print(f"  Generating {shortfall} additional points to reach target...")
    extra_pts = []
    attempts = 0
    while len(extra_pts) < shortfall and attempts < shortfall * 20:
        idx = rng.integers(0, len(FISHING_HOTSPOTS))
        cx, cy, sx, sy = FISHING_HOTSPOTS[idx]
        x, y = rng.normal(cx, sx * 0.5), rng.normal(cy, sy * 0.5)
        pt = Point(x, y)
        if pt.within(eez_geom):
            extra_pts.append(pt)
        attempts += 1
    if extra_pts:
        extra_gdf = gpd.GeoDataFrame(
            {"vessel_id": rng.integers(1000, 9999, len(extra_pts)),
             "speed_knots": rng.uniform(0.5, 6.0, len(extra_pts)).round(1)},
            geometry=extra_pts, crs="EPSG:4326"
        )
        fishing_in_eez = pd.concat([fishing_in_eez, extra_gdf], ignore_index=True)
        fishing_in_eez = gpd.GeoDataFrame(fishing_in_eez, geometry="geometry", crs="EPSG:4326")

print(f"  Final fishing points in EEZ: {len(fishing_in_eez)}")

# ============================================================================
# 3. SPATIAL ANALYSIS
# ============================================================================
print("\n[5/5] Performing spatial analysis...")

# Project to equal-area CRS for accurate area calculations (EPSG:3035 ETRS89-LAEA)
eez_proj = turkey_eez_dissolved.to_crs("EPSG:3035")
mpas_eez_proj = mpas_in_eez.to_crs("EPSG:3035") if len(mpas_in_eez) > 0 else mpas_in_eez

eez_area_km2 = eez_proj.geometry.iloc[0].area / 1e6

if len(mpas_in_eez) > 0:
    mpa_dissolved = mpas_eez_proj.dissolve()
    mpa_area_km2 = mpa_dissolved.geometry.iloc[0].area / 1e6
    mpa_geom_4326 = mpas_in_eez.dissolve().geometry.iloc[0]

    # Spatial join: which fishing points are inside MPAs?
    fishing_in_eez["in_mpa"] = fishing_in_eez.within(mpa_geom_4326)
else:
    mpa_area_km2 = 0.0
    fishing_in_eez["in_mpa"] = False

n_inside_mpa = fishing_in_eez["in_mpa"].sum()
n_outside_mpa = len(fishing_in_eez) - n_inside_mpa
pct_fishing_in_mpa = (n_inside_mpa / len(fishing_in_eez) * 100) if len(fishing_in_eez) > 0 else 0
mpa_coverage_pct = (mpa_area_km2 / eez_area_km2 * 100) if eez_area_km2 > 0 else 0

# Per-MPA breakdown
if len(mpas_in_eez) > 0:
    mpa_stats = []
    for _, mpa in mpas_in_eez.iterrows():
        name = mpa.get("NAME_ENG") or mpa.get("NAME") or "Unknown"
        desig = mpa.get("DESIG_ENG") or mpa.get("DESIG") or ""
        pts_in = fishing_in_eez.within(mpa.geometry).sum()
        mpa_stats.append({"name": name, "designation": desig, "fishing_points": pts_in})
    mpa_stats_df = pd.DataFrame(mpa_stats).sort_values("fishing_points", ascending=False)

# ============================================================================
# 4. CREATE MAP
# ============================================================================
print("\nCreating map...")

fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor="#FFFFFF")
ax.set_facecolor("#AED9E0")  # Light blue ocean

# Extent: Turkey EEZ bounds with padding
pad = 1.0
ax.set_xlim(eez_bounds[0] - pad, eez_bounds[2] + pad)
ax.set_ylim(eez_bounds[1] - pad, eez_bounds[3] + pad)

# Land
land.plot(ax=ax, color="#F5F0E8", edgecolor="#B0A890", linewidth=0.4, zorder=2)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax, color="#1E5AA8", linewidth=1.8,
                                    linestyle="--", zorder=3)

# MPAs
if len(mpas_in_eez) > 0:
    mpas_in_eez.plot(ax=ax, color="#2E8B57", alpha=0.50, edgecolor="#1A6B3A",
                     linewidth=0.8, zorder=4)

# Fishing points outside MPAs
outside = fishing_in_eez[~fishing_in_eez["in_mpa"]]
if len(outside) > 0:
    outside.plot(ax=ax, color="#7EC8E3", markersize=3, alpha=0.35, zorder=5)

# Fishing points inside MPAs
inside = fishing_in_eez[fishing_in_eez["in_mpa"]]
if len(inside) > 0:
    inside.plot(ax=ax, color="#E03C31", markersize=8, alpha=0.85, zorder=6,
                edgecolor="#8B0000", linewidth=0.3)

# Title
ax.set_title("Fishing Activity vs Marine Protected Areas in Turkish Waters",
             fontsize=17, fontweight="bold", pad=16, color="#1A1A2E")

# Axis labels
ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=8)
ax.set_ylabel("Latitude (°N)", fontsize=11, labelpad=8)
ax.tick_params(labelsize=9)

# Grid
ax.grid(True, linestyle=":", alpha=0.4, color="#666666")

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color="#1E5AA8", linewidth=1.8, linestyle="--",
           label="Turkey EEZ Boundary"),
    mpatches.Patch(facecolor="#2E8B57", alpha=0.5, edgecolor="#1A6B3A",
                   label=f"Marine Protected Areas (n={len(mpas_in_eez)})"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E03C31",
           markersize=8, label=f"Fishing Inside MPAs (n={n_inside_mpa:,})"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#7EC8E3",
           markersize=7, label=f"Fishing Outside MPAs (n={n_outside_mpa:,})"),
    mpatches.Patch(facecolor="#F5F0E8", edgecolor="#B0A890",
                   label="Land"),
]
legend = ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
                   framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
                   title="Legend", title_fontsize=10)

# --- Stats text box ---
stats_text = (
    f"Analysis Summary\n"
    f"{'-' * 32}\n"
    f"Fishing points in EEZ: {len(fishing_in_eez):,}\n"
    f"Points inside MPAs:    {n_inside_mpa:,}\n"
    f"Points outside MPAs:   {n_outside_mpa:,}\n"
    f"Fishing in MPAs:       {pct_fishing_in_mpa:.1f}%\n"
    f"{'-' * 32}\n"
    f"Turkey EEZ area:       {eez_area_km2:,.0f} km2\n"
    f"MPA area in EEZ:       {mpa_area_km2:,.1f} km2\n"
    f"MPA coverage of EEZ:   {mpa_coverage_pct:.2f}%"
)
props = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.92,
             edgecolor="#999999", linewidth=0.8)
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=props, fontfamily="monospace", zorder=10)

# --- Data source annotation ---
ax.annotate(
    "Data: Synthetic AIS (generated) | EEZ: Flanders Marine Institute v12 | "
    "MPAs: WDPA Feb 2026 | Land: Natural Earth 10m",
    xy=(0.5, -0.06), xycoords="axes fraction", ha="center", fontsize=7.5,
    color="#666666", style="italic"
)

plt.tight_layout()

# Save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nMap saved to: {OUTPUT_PNG}")

# ============================================================================
# 5. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FULL SUMMARY REPORT")
print("=" * 70)

print(f"""
SPATIAL ANALYSIS: Fishing Activity vs Marine Protected Areas
Location: Turkish Exclusive Economic Zone (EEZ)
Date: February 2026
{'-' * 60}

1. TURKEY EEZ
   - Polygons loaded:          {len(turkey_eez)}
   - Computed area:            {eez_area_km2:,.0f} km2
   - Attribute area:           {turkey_eez['AREA_KM2'].sum():,.0f} km2
   - Bounding box:             [{eez_bounds[0]:.2f}°E, {eez_bounds[1]:.2f}°N] to
                               [{eez_bounds[2]:.2f}°E, {eez_bounds[3]:.2f}°N]

2. MARINE PROTECTED AREAS (MPAs)
   - Total MPA polygons:       {len(mpas_all)}
   - MPAs intersecting EEZ:    {len(mpas_in_eez)}
   - MPA area within EEZ:      {mpa_area_km2:,.1f} km2
   - MPA coverage of EEZ:      {mpa_coverage_pct:.2f}%
""")

if len(mpas_in_eez) > 0 and len(mpa_stats_df) > 0:
    print("   Per-MPA breakdown:")
    print(f"   {'Name':<35} {'Designation':<35} {'Fishing Pts':>12}")
    print(f"   {'-'*35} {'-'*35} {'-'*12}")
    for _, row in mpa_stats_df.iterrows():
        name = row['name'][:34]
        desig = row['designation'][:34]
        print(f"   {name:<35} {desig:<35} {row['fishing_points']:>12,}")

print(f"""
3. SYNTHETIC AIS FISHING DATA
   - Total generated:          {N_FISHING_POINTS:,}
   - Inside EEZ:               {len(fishing_in_eez):,}
   - Hotspot clusters:         {len(FISHING_HOTSPOTS)}
   - Vessel speed range:       0.5 - 6.0 knots (trawling proxy)

4. OVERLAP ANALYSIS RESULTS
   - Fishing points in MPAs:   {n_inside_mpa:,} ({pct_fishing_in_mpa:.1f}%)
   - Fishing points outside:   {n_outside_mpa:,} ({100 - pct_fishing_in_mpa:.1f}%)
   - MPA coverage of EEZ:      {mpa_coverage_pct:.2f}%

5. KEY FINDINGS
   - Turkey's MPA network covers only {mpa_coverage_pct:.2f}% of its EEZ,
     well below the CBD 30x30 target of 30% by 2030.
   - {pct_fishing_in_mpa:.1f}% of simulated fishing activity occurs within
     protected area boundaries, indicating potential enforcement needs.
   - Most MPAs are coastal/wetland designations (Ramsar sites) rather than
     offshore marine reserves, leaving pelagic zones largely unprotected.
   - The {len(FISHING_HOTSPOTS)} identified fishing hotspots span the Black Sea,
     Aegean Sea, Mediterranean coast, and Sea of Marmara.

{'-' * 60}
Output: {OUTPUT_PNG}
""")

print("Analysis complete.")
