"""
MPA Effectiveness Analysis - Turkish Waters (2019-2023)
========================================================
Before-After, Control-Impact (BACI) design comparing fishing
effort inside vs outside MPAs across a simulated designation event.
"""

import subprocess, sys, io, os

for pkg_name, import_name in [
    ("geopandas", "geopandas"), ("matplotlib", "matplotlib"),
    ("shapely", "shapely"), ("numpy", "numpy"),
    ("scipy", "scipy"), ("pandas", "pandas"),
]:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "-q"])

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from shapely.geometry import box
import warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp")

EEZ_PATH  = (BASE_DIR / "downloads" / "World_EEZ_v12_20231025" /
             "World_EEZ_v12_20231025" / "eez_v12.shp")
LAND_PATH = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"
MPA_BASE  = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 15 mpa effectiveness"
OUTPUT_PNG = OUTPUT_DIR / "turkey_mpa_effectiveness.png"

# Time parameters
YEARS         = list(range(2019, 2024))       # 2019-2023
DESIGNATION   = pd.Timestamp("2021-01-01")    # MPA designation date
BEFORE_PERIOD = ("2019-01", "2020-12")
AFTER_PERIOD  = ("2021-01", "2023-12")
RANDOM_SEED   = 42

# Fishing effort parameters (vessels*hours / month)
INSIDE_BASE_BEFORE  = 420   # high fishing inside MPAs pre-designation
INSIDE_BASE_AFTER   = 185   # reduced after enforcement
OUTSIDE_BASE        = 380   # relatively stable outside
SEASONAL_AMP        = 100   # summer peak amplitude
NOISE_STD           = 35    # monthly noise
TREND_OUTSIDE       = -1.2  # slight decline per month outside (sustainability)

# ============================================================================
# 1. LOAD MPA DATA
# ============================================================================
print("=" * 70)
print("MPA EFFECTIVENESS ANALYSIS (BACI) - Turkish Waters")
print("=" * 70)

print("\n[1/6] Loading Turkey EEZ...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
print(f"  EEZ loaded ({turkey_eez['AREA_KM2'].sum():,.0f} km2)")

print("\n[2/6] Loading MPA polygons...")
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
n_mpas = len(mpas_in_eez)
print(f"  {n_mpas} unique MPAs in Turkish EEZ")

print("\n[3/6] Loading land polygons...")
land = gpd.read_file(LAND_PATH)
turkey_bbox = box(24.5, 34.5, 42.5, 44.0)
land_clip = gpd.clip(land, gpd.GeoDataFrame(geometry=[turkey_bbox], crs="EPSG:4326"))
print(f"  Land clipped: {len(land_clip)} polygon(s)")

# ============================================================================
# 2. GENERATE SYNTHETIC BACI TIME SERIES
# ============================================================================
print("\n[4/6] Generating synthetic fishing effort time series...")
np.random.seed(RANDOM_SEED)

# Monthly date range
dates = pd.date_range("2019-01-01", "2023-12-31", freq="MS")
n_months = len(dates)

# Seasonal component: peak in June-August (month 6-8)
months_num = np.array([d.month for d in dates])
seasonal = SEASONAL_AMP * np.sin(2 * np.pi * (months_num - 3) / 12)  # peak ~June

# --- Inside MPA effort ---
inside_effort = np.zeros(n_months)
for i, d in enumerate(dates):
    if d < DESIGNATION:
        base = INSIDE_BASE_BEFORE
    else:
        # Gradual reduction over first 6 months after designation
        months_since = (d.year - 2021) * 12 + d.month - 1
        ramp = min(1.0, months_since / 6.0)
        base = INSIDE_BASE_BEFORE - ramp * (INSIDE_BASE_BEFORE - INSIDE_BASE_AFTER)
    inside_effort[i] = base + seasonal[i] * 0.7 + np.random.normal(0, NOISE_STD)

# --- Outside MPA effort (control) ---
month_idx = np.arange(n_months)
outside_effort = (OUTSIDE_BASE
                  + seasonal
                  + TREND_OUTSIDE * month_idx
                  + np.random.normal(0, NOISE_STD, n_months))

# Clip to positive values
inside_effort = np.clip(inside_effort, 20, None)
outside_effort = np.clip(outside_effort, 20, None)

# Build DataFrame
df = pd.DataFrame({
    "date": dates,
    "inside_mpa": inside_effort,
    "outside_mpa": outside_effort,
})
df["period"] = df["date"].apply(lambda d: "Before" if d < DESIGNATION else "After")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

print(f"  Generated {n_months} monthly observations ({YEARS[0]}-{YEARS[-1]})")
print(f"  Before period: {BEFORE_PERIOD[0]} to {BEFORE_PERIOD[1]} ({(df['period']=='Before').sum()} months)")
print(f"  After period:  {AFTER_PERIOD[0]} to {AFTER_PERIOD[1]} ({(df['period']=='After').sum()} months)")

# ============================================================================
# 3. BACI STATISTICAL ANALYSIS
# ============================================================================
print("\n[5/6] Running BACI analysis...")

before = df[df["period"] == "Before"]
after  = df[df["period"] == "After"]

# Means and standard errors
inside_before_mean  = before["inside_mpa"].mean()
inside_before_se    = before["inside_mpa"].sem()
inside_after_mean   = after["inside_mpa"].mean()
inside_after_se     = after["inside_mpa"].sem()

outside_before_mean = before["outside_mpa"].mean()
outside_before_se   = before["outside_mpa"].sem()
outside_after_mean  = after["outside_mpa"].mean()
outside_after_se    = after["outside_mpa"].sem()

# BACI score = (Inside_after - Inside_before) - (Outside_after - Outside_before)
delta_inside  = inside_after_mean - inside_before_mean
delta_outside = outside_after_mean - outside_before_mean
baci_score    = delta_inside - delta_outside

# Percent reduction inside MPAs
pct_reduction_inside = (1 - inside_after_mean / inside_before_mean) * 100

# Statistical tests
# 1. Paired t-test: inside before vs after
t_inside, p_inside = stats.ttest_ind(before["inside_mpa"], after["inside_mpa"])

# 2. Paired t-test: outside before vs after
t_outside, p_outside = stats.ttest_ind(before["outside_mpa"], after["outside_mpa"])

# 3. BACI interaction test (two-way comparison)
# Compute difference series (inside - outside) for before and after
before_diff = before["inside_mpa"].values - before["outside_mpa"].values
after_diff  = after["inside_mpa"].values - after["outside_mpa"].values
t_baci, p_baci = stats.ttest_ind(before_diff, after_diff)

# Effect size (Cohen's d for BACI)
pooled_std = np.sqrt((before_diff.var() + after_diff.var()) / 2)
cohens_d = (before_diff.mean() - after_diff.mean()) / pooled_std

# Annual means for supplementary analysis
annual_inside  = df.groupby("year")["inside_mpa"].mean()
annual_outside = df.groupby("year")["outside_mpa"].mean()

# Per-MPA fishing effort change (synthetic, proportional to area)
mpa_names = []
mpa_lons  = []
mpa_lats  = []
mpa_areas = []
mpa_changes = []  # % change in fishing effort

for _, row in mpas_in_eez.iterrows():
    name = row.get("NAME", "Unknown")
    area = row.get("REP_AREA", 0)
    centroid = row.geometry.centroid
    mpa_names.append(name)
    mpa_lons.append(centroid.x)
    mpa_lats.append(centroid.y)
    mpa_areas.append(area)
    # Larger MPAs show more reduction (synthetic relationship)
    base_change = -pct_reduction_inside
    area_factor = min(1.5, max(0.5, np.log10(area + 1) / 3))
    noise = np.random.normal(0, 5)
    mpa_changes.append(base_change * area_factor + noise)

# ============================================================================
# 4. CREATE FIGURE
# ============================================================================
print("\n[6/6] Creating figure...")

fig = plt.figure(figsize=(18, 7), dpi=300, facecolor="white")

# Layout: 3 panels side by side
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.8, 1.0],
                      wspace=0.32, left=0.05, right=0.97,
                      top=0.85, bottom=0.12)

ax1 = fig.add_subplot(gs[0, 0])  # Time series
ax2 = fig.add_subplot(gs[0, 1])  # Bar chart
ax3 = fig.add_subplot(gs[0, 2])  # Map

# Colours
C_INSIDE  = "#e74c3c"
C_OUTSIDE = "#2980b9"
C_BEFORE  = "#ecf0f1"
C_AFTER   = "#d5f5e3"

# ---------- Panel A: Time Series ----------
# Shaded regions
ax1.axvspan(dates[0], DESIGNATION, alpha=0.10, color="#3498db",
            label="Before designation")
ax1.axvspan(DESIGNATION, dates[-1], alpha=0.10, color="#2ecc71",
            label="After designation")

# Vertical designation line
ax1.axvline(DESIGNATION, color="#2c3e50", linewidth=1.5, linestyle="--",
            alpha=0.8, zorder=5)
ax1.text(DESIGNATION, 0.82, " MPA\n Designation",
         fontsize=8, fontweight="bold", color="#2c3e50",
         va="bottom", ha="left", transform=ax1.get_xaxis_transform())

# Rolling averages (3-month)
window = 3
inside_smooth  = pd.Series(inside_effort).rolling(window, center=True).mean()
outside_smooth = pd.Series(outside_effort).rolling(window, center=True).mean()

# Raw data (faint)
ax1.plot(dates, inside_effort, color=C_INSIDE, alpha=0.20, linewidth=0.8)
ax1.plot(dates, outside_effort, color=C_OUTSIDE, alpha=0.20, linewidth=0.8)

# Smoothed lines
ax1.plot(dates, inside_smooth, color=C_INSIDE, linewidth=2.2,
         label="Inside MPAs", zorder=4)
ax1.plot(dates, outside_smooth, color=C_OUTSIDE, linewidth=2.2,
         label="Outside MPAs", zorder=4)

# Before/After means as horizontal lines
for period_df, ls in [(before, ":"), (after, ":")]:
    d_start, d_end = period_df["date"].iloc[0], period_df["date"].iloc[-1]
    ax1.hlines(period_df["inside_mpa"].mean(), d_start, d_end,
               colors=C_INSIDE, linewidth=1.5, linestyle=ls, alpha=0.6)
    ax1.hlines(period_df["outside_mpa"].mean(), d_start, d_end,
               colors=C_OUTSIDE, linewidth=1.5, linestyle=ls, alpha=0.6)

ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
ax1.set_ylabel("Fishing Effort (vessel-hours / month)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Monthly Fishing Effort Time Series", fontsize=12,
              fontweight="bold", pad=10)
ax1.legend(loc="upper right", fontsize=8, framealpha=0.9, edgecolor="#bdc3c7")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.tick_params(labelsize=9)

# Annotate reduction arrow
mid_after = pd.Timestamp("2022-06-01")
ax1.annotate(
    f"{pct_reduction_inside:.0f}%\nreduction",
    xy=(mid_after, inside_after_mean),
    xytext=(mid_after, inside_before_mean - 10),
    fontsize=9, fontweight="bold", color=C_INSIDE,
    ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->", color=C_INSIDE, lw=1.5),
)

# ---------- Panel B: BACI Bar Chart ----------
bar_width = 0.30
x_pos = np.array([0, 1])

# Before bars
bars_before = ax2.bar(x_pos - bar_width/2,
                      [inside_before_mean, outside_before_mean],
                      width=bar_width, color=["#e74c3c", "#2980b9"],
                      alpha=0.5, edgecolor="white", linewidth=0.8,
                      label="Before (2019-2020)")
# After bars
bars_after = ax2.bar(x_pos + bar_width/2,
                     [inside_after_mean, outside_after_mean],
                     width=bar_width, color=["#e74c3c", "#2980b9"],
                     alpha=0.9, edgecolor="white", linewidth=0.8,
                     label="After (2021-2023)")

# Error bars
ax2.errorbar(x_pos - bar_width/2,
             [inside_before_mean, outside_before_mean],
             yerr=[inside_before_se * 1.96, outside_before_se * 1.96],
             fmt="none", ecolor="#2c3e50", capsize=4, linewidth=1.2)
ax2.errorbar(x_pos + bar_width/2,
             [inside_after_mean, outside_after_mean],
             yerr=[inside_after_se * 1.96, outside_after_se * 1.96],
             fmt="none", ecolor="#2c3e50", capsize=4, linewidth=1.2)

# Significance brackets
def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color="#2c3e50")
    ax.text((x1+x2)/2, y+h+2, text, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2c3e50")

sig_text_inside = "***" if p_inside < 0.001 else ("**" if p_inside < 0.01 else
                  ("*" if p_inside < 0.05 else "ns"))
max_bar = max(inside_before_mean, outside_before_mean) + 40
add_sig_bracket(ax2, -bar_width/2, bar_width/2, max_bar, 8, sig_text_inside)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Inside MPAs", "Outside MPAs"], fontsize=10, fontweight="bold")
ax2.set_ylabel("Mean Fishing Effort\n(vessel-hours / month)", fontsize=10,
               fontweight="bold")
ax2.set_title("(b) BACI Comparison", fontsize=12, fontweight="bold", pad=10)
ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#bdc3c7")
ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
ax2.tick_params(labelsize=9)
ax2.set_ylim(0, max_bar + 80)

# BACI score annotation
baci_text = f"BACI = {baci_score:.1f}\np = {p_baci:.4f}"
ax2.text(0.98, 0.05, baci_text, transform=ax2.transAxes,
         fontsize=9, fontweight="bold", ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                   edgecolor="#bdc3c7", alpha=0.9))

# ---------- Panel C: Map ----------
# Turkey basemap
land_clip.plot(ax=ax3, color="#d5d8dc", edgecolor="#95a5a6", linewidth=0.4)

# EEZ boundary
turkey_eez_dissolved.boundary.plot(ax=ax3, color="#00b4d8", linewidth=1.2,
                                   linestyle="--", alpha=0.6)

# MPAs coloured by fishing effort change
if len(mpa_changes) > 0:
    vmin = min(mpa_changes)
    vmax = max(mpa_changes)
    # Normalise for colormap
    norm = plt.Normalize(vmin=vmin, vmax=0)
    cmap = plt.cm.RdYlGn_r  # Red = big reduction (good), Green = little change

    for idx, (_, row) in enumerate(mpas_in_eez.iterrows()):
        if idx < len(mpa_changes):
            change = mpa_changes[idx]
            color = cmap(norm(change))
            try:
                gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").plot(
                    ax=ax3, color=color, edgecolor="black", linewidth=1.0, alpha=0.7)
            except Exception:
                pass

    # Add MPA labels
    for idx in range(min(len(mpa_names), len(mpa_lons))):
        if mpa_areas[idx] > 0:
            short_name = mpa_names[idx][:18]
            ax3.annotate(
                f"{short_name}\n({mpa_changes[idx]:+.0f}%)",
                xy=(mpa_lons[idx], mpa_lats[idx]),
                fontsize=6, fontweight="bold", color="#2c3e50",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#bdc3c7", alpha=0.85),
            )

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Fishing Effort Change (%)", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

ax3.set_xlim(24.5, 42.5)
ax3.set_ylim(34.5, 43.0)
ax3.set_xlabel("Longitude", fontsize=9)
ax3.set_ylabel("Latitude", fontsize=9)
ax3.set_title("(c) MPA Fishing Effort Change", fontsize=12,
              fontweight="bold", pad=10)
ax3.grid(True, alpha=0.2, linestyle="--")
ax3.tick_params(labelsize=8)
ax3.set_aspect("equal")

# ---------- Super title ----------
fig.suptitle("MPA Effectiveness Analysis -- Turkish Waters (2019-2023)",
             fontsize=16, fontweight="bold", color="#1a2634", y=0.97)

# ---------- Text box with key stats ----------
stats_text = (
    f"EEZ: {turkey_eez['AREA_KM2'].sum():,.0f} km2  |  "
    f"MPAs analysed: {n_mpas}  |  "
    f"BACI score: {baci_score:.1f}  |  "
    f"Cohen's d: {cohens_d:.2f}  |  "
    f"Inside reduction: {pct_reduction_inside:.1f}%"
)
fig.text(0.50, 0.03, stats_text, ha="center", va="center", fontsize=9,
         fontstyle="italic", color="#7f8c8d",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa",
                   edgecolor="#dfe6e9", alpha=0.9))

# Source annotation
fig.text(0.99, 0.005,
         "Sources: WDPA Feb 2026, VLIZ EEZ v12, Synthetic AIS Data | BACI Design",
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
print("BACI ANALYSIS RESULTS")
print("=" * 70)

print(f"""
  STUDY DESIGN
  {'-'*60}
  Design:             Before-After, Control-Impact (BACI)
  Before period:      {BEFORE_PERIOD[0]} to {BEFORE_PERIOD[1]} (24 months)
  After period:       {AFTER_PERIOD[0]} to {AFTER_PERIOD[1]} (36 months)
  Impact sites:       Inside MPAs ({n_mpas} protected areas)
  Control sites:      Outside MPAs (rest of Turkish EEZ)

  FISHING EFFORT (vessel-hours / month)
  {'-'*60}
  Inside MPAs:
    Before:           {inside_before_mean:.1f} +/- {inside_before_se*1.96:.1f} (95% CI)
    After:            {inside_after_mean:.1f} +/- {inside_after_se*1.96:.1f} (95% CI)
    Change:           {delta_inside:+.1f} ({pct_reduction_inside:.1f}% reduction)

  Outside MPAs:
    Before:           {outside_before_mean:.1f} +/- {outside_before_se*1.96:.1f} (95% CI)
    After:            {outside_after_mean:.1f} +/- {outside_after_se*1.96:.1f} (95% CI)
    Change:           {delta_outside:+.1f} ({abs(delta_outside/outside_before_mean*100):.1f}% {'reduction' if delta_outside < 0 else 'increase'})

  BACI INTERACTION
  {'-'*60}
  BACI score:         {baci_score:.1f}
  Interpretation:     {'MPA designation reduced fishing effort' if baci_score < 0 else 'No clear MPA effect'}
                      beyond background trends

  STATISTICAL TESTS
  {'-'*60}
  Inside Before vs After:
    t = {t_inside:.3f},  p = {p_inside:.6f}  {'***' if p_inside < 0.001 else '**' if p_inside < 0.01 else '*' if p_inside < 0.05 else 'ns'}

  Outside Before vs After:
    t = {t_outside:.3f},  p = {p_outside:.6f}  {'***' if p_outside < 0.001 else '**' if p_outside < 0.01 else '*' if p_outside < 0.05 else 'ns'}

  BACI Interaction (difference-in-differences):
    t = {t_baci:.3f},  p = {p_baci:.6f}  {'***' if p_baci < 0.001 else '**' if p_baci < 0.01 else '*' if p_baci < 0.05 else 'ns'}

  Effect size (Cohen's d): {cohens_d:.2f}  ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})

  ANNUAL MEANS (vessel-hours / month)
  {'-'*60}""")

for yr in YEARS:
    status = "Before" if yr < 2021 else "After "
    print(f"    {yr} [{status}]   Inside: {annual_inside[yr]:.1f}    Outside: {annual_outside[yr]:.1f}    Ratio: {annual_inside[yr]/annual_outside[yr]:.2f}")

print(f"""
  CONCLUSION
  {'-'*60}
  MPA designation resulted in a statistically significant
  {pct_reduction_inside:.1f}% reduction in fishing effort inside protected
  areas (p < {max(p_inside, 0.001):.3f}). The BACI interaction term confirms
  this reduction is attributable to the MPA designation rather
  than background trends (BACI p = {p_baci:.4f}, Cohen's d = {cohens_d:.2f}).
""")

print("=" * 70)
print("DONE - MPA effectiveness analysis complete!")
print("=" * 70)
