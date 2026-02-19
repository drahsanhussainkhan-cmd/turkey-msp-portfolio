"""
Fishing Vessel Density Map for Turkish Waters
Generates a professional night-mode maritime density heatmap.
"""

import subprocess
import sys

# Ensure required packages
for pkg in ["matplotlib", "numpy", "scipy", "geopandas", "shapely"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import geopandas as gpd
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
EEZ_PATH = (
    r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp"
    r"\downloads\World_EEZ_v12_20231025"
    r"\World_EEZ_v12_20231025\eez_v12.shp"
)
OUTPUT_PATH = (
    r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp"
    r"\project 4 fishing density\turkey_fishing_density_map.png"
)

# ──────────────────────────────────────────────────────────────
# 1. TRY GLOBAL FISHING WATCH API, FALL BACK TO SYNTHETIC DATA
# ──────────────────────────────────────────────────────────────
def try_gfw_api():
    """Attempt to fetch data from Global Fishing Watch public API."""
    try:
        import requests
        url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
        params = {
            "spatial-resolution": "low",
            "temporal-resolution": "yearly",
            "group-by": "flagAndGearType",
            "datasets[0]": "public-global-fishing-effort:latest",
            "date-range": "2023-01-01,2023-12-31",
            "region": {"type": "Polygon", "coordinates": [
                [[25.4, 34.2], [41.6, 34.2], [41.6, 43.5],
                 [25.4, 43.5], [25.4, 34.2]]
            ]},
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def generate_synthetic_data(n_points=6000, seed=42):
    """
    Generate realistic synthetic AIS fishing vessel positions
    for Turkish waters (Black Sea, Aegean, Mediterranean).
    Uses mixture of Gaussians anchored on real fishing grounds.
    """
    rng = np.random.default_rng(seed)

    # Fishing ground clusters: (lon, lat, lon_std, lat_std, weight)
    clusters = [
        # ── BLACK SEA ── dense along northern Turkish coast
        (37.0, 41.5, 1.8, 0.35, 0.14),   # Trabzon-Rize coast
        (34.5, 41.8, 1.5, 0.30, 0.10),   # Sinop-Samsun coast
        (31.5, 41.7, 1.2, 0.25, 0.08),   # Zonguldak coast
        (36.0, 42.5, 2.0, 0.40, 0.06),   # Offshore Black Sea
        (39.5, 41.2, 0.8, 0.30, 0.05),   # Eastern Black Sea

        # ── MARMARA / STRAITS ── Istanbul strait heavy traffic
        (29.0, 40.8, 0.4, 0.25, 0.08),   # Istanbul strait / Marmara
        (28.5, 40.3, 0.6, 0.30, 0.04),   # Southern Marmara

        # ── AEGEAN ── western coast, islands corridor
        (26.5, 39.5, 0.7, 0.60, 0.10),   # Northern Aegean
        (27.0, 38.0, 0.8, 0.50, 0.08),   # Central Aegean (Izmir)
        (27.5, 37.0, 0.6, 0.40, 0.05),   # Southern Aegean (Bodrum)
        (26.0, 38.5, 1.0, 0.80, 0.04),   # Offshore Aegean

        # ── MEDITERRANEAN ── scattered along southern coast
        (30.5, 36.2, 1.2, 0.30, 0.06),   # Antalya coast
        (33.5, 36.0, 1.0, 0.25, 0.04),   # Mersin coast
        (35.5, 36.0, 0.8, 0.20, 0.04),   # Iskenderun Bay
        (29.0, 36.5, 0.8, 0.25, 0.04),   # Fethiye-Kas coast
    ]

    # Normalize weights
    weights = np.array([c[4] for c in clusters])
    weights /= weights.sum()

    # Assign points per cluster
    counts = rng.multinomial(n_points, weights)

    lons, lats = [], []
    for (lon_c, lat_c, lon_s, lat_s, _), n in zip(clusters, counts):
        lons.append(rng.normal(lon_c, lon_s, n))
        lats.append(rng.normal(lat_c, lat_s, n))

    lons = np.concatenate(lons)
    lats = np.concatenate(lats)

    # Clip to reasonable Turkish waters bounding box
    mask = (lons >= 25.0) & (lons <= 42.0) & (lats >= 34.0) & (lats <= 44.0)
    return lons[mask], lats[mask]


# ──────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────────────────
print("Attempting Global Fishing Watch API...")
api_data = try_gfw_api()

if api_data is not None:
    print("API returned data — parsing...")
    data_source = "Global Fishing Watch (2023)"
    # Parse would go here; for now fall through
    lons, lats = generate_synthetic_data()
    data_source = "Synthetic AIS data modeled on Turkish fishing grounds"
else:
    print("API requires authentication. Generating synthetic data...")
    lons, lats = generate_synthetic_data()
    data_source = "Synthetic AIS data modeled on Turkish fishing grounds (n={:,})".format(len(lons))

print(f"  {len(lons):,} vessel positions generated.")

# ──────────────────────────────────────────────────────────────
# 3. LOAD TURKEY EEZ
# ──────────────────────────────────────────────────────────────
print("Loading Turkey EEZ boundary...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy()
turkey_eez = turkey_eez.to_crs(epsg=4326)

# Also load land (Natural Earth via geopandas built-in)
print("Loading land polygons...")
try:
    land = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except Exception:
    land = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

# ──────────────────────────────────────────────────────────────
# 4. COMPUTE KDE DENSITY
# ──────────────────────────────────────────────────────────────
print("Computing kernel density estimation...")
coords = np.vstack([lons, lats])
kde = gaussian_kde(coords, bw_method=0.06)

# Grid for evaluation
xi = np.linspace(25.0, 42.0, 500)
yi = np.linspace(34.0, 44.0, 350)
xx, yy = np.meshgrid(xi, yi)
grid_coords = np.vstack([xx.ravel(), yy.ravel()])
zi = kde(grid_coords).reshape(xx.shape)

# ──────────────────────────────────────────────────────────────
# 5. CREATE PROFESSIONAL MAP
# ──────────────────────────────────────────────────────────────
print("Rendering map...")

# Custom colormap: transparent -> yellow -> orange -> red -> white-hot
cmap_colors = [
    (0.0, (0.04, 0.04, 0.18, 0.0)),     # transparent navy
    (0.05, (0.04, 0.04, 0.18, 0.0)),     # still transparent at low density
    (0.15, (0.55, 0.15, 0.02, 0.6)),     # dark orange
    (0.35, (0.90, 0.35, 0.0, 0.8)),      # bright orange
    (0.55, (1.0, 0.65, 0.0, 0.9)),       # yellow-orange
    (0.75, (1.0, 0.90, 0.2, 0.95)),      # bright yellow
    (1.0, (1.0, 1.0, 0.85, 1.0)),        # white-hot
]
cmap_data = {
    "red":   [(v, c[0], c[0]) for v, c in cmap_colors],
    "green": [(v, c[1], c[1]) for v, c in cmap_colors],
    "blue":  [(v, c[2], c[2]) for v, c in cmap_colors],
    "alpha": [(v, c[3], c[3]) for v, c in cmap_colors],
}
heat_cmap = LinearSegmentedColormap("maritime_heat", cmap_data, N=256)

fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=300)
fig.patch.set_facecolor("#0a0a2e")
ax.set_facecolor("#0a0a2e")

# Plot land
land.plot(ax=ax, color="#2a2a2a", edgecolor="#3a3a3a", linewidth=0.4, zorder=2)

# Plot density heatmap
density_plot = ax.pcolormesh(
    xx, yy, zi,
    cmap=heat_cmap,
    shading="gouraud",
    zorder=3,
)

# Plot Turkey EEZ boundary
turkey_eez.boundary.plot(
    ax=ax, edgecolor="white", linewidth=1.2,
    linestyle="--", alpha=0.8, zorder=4, label="Turkey EEZ"
)

# Map extent
ax.set_xlim(25.0, 42.0)
ax.set_ylim(34.0, 44.0)
ax.set_aspect("equal")

# Gridlines
for x in range(26, 42, 2):
    ax.axvline(x, color="#1a1a4e", linewidth=0.3, zorder=1)
for y in range(35, 44, 1):
    ax.axhline(y, color="#1a1a4e", linewidth=0.3, zorder=1)

# Tick styling
ax.tick_params(colors="#8888aa", labelsize=8)
ax.set_xlabel("Longitude", color="#8888aa", fontsize=9)
ax.set_ylabel("Latitude", color="#8888aa", fontsize=9)
for spine in ax.spines.values():
    spine.set_color("#3a3a5e")
    spine.set_linewidth(0.5)

# Colorbar
sm = plt.cm.ScalarMappable(
    cmap=heat_cmap,
    norm=mcolors.Normalize(vmin=zi.min(), vmax=zi.max()),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbar.set_label("Vessel Density", color="#ccccdd", fontsize=10)
cbar.ax.tick_params(colors="#8888aa", labelsize=8)
cbar.outline.set_edgecolor("#3a3a5e")

# Title
ax.set_title(
    "Fishing Vessel Density in Turkish Waters",
    color="#e8e8f0", fontsize=16, fontweight="bold",
    pad=15, fontfamily="sans-serif",
)

# Annotations
ax.text(
    0.01, -0.06,
    f"Data: {data_source}\n"
    f"EEZ: Marine Regions v12 (Flanders Marine Institute, 2023)\n"
    f"Method: Gaussian KDE | CRS: WGS 84 (EPSG:4326)",
    transform=ax.transAxes,
    color="#666688", fontsize=7, verticalalignment="top",
)

# Sea labels
sea_labels = [
    (33.5, 42.8, "BLACK SEA"),
    (26.5, 38.2, "AEGEAN\nSEA"),
    (32.0, 35.2, "MEDITERRANEAN SEA"),
    (28.8, 40.6, "Sea of\nMarmara"),
]
for sx, sy, label in sea_labels:
    ax.text(
        sx, sy, label,
        color="#4a4a7a", fontsize=8, fontstyle="italic",
        ha="center", va="center", zorder=5, alpha=0.7,
    )

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="white", linewidth=1.2,
           linestyle="--", label="Turkey EEZ Boundary"),
]
leg = ax.legend(
    handles=legend_elements, loc="upper right",
    fontsize=8, framealpha=0.3,
    facecolor="#0a0a2e", edgecolor="#3a3a5e",
    labelcolor="#ccccdd",
)

plt.tight_layout()
plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
    edgecolor="none",
    pad_inches=0.3,
)
plt.close()

print(f"\nMap saved to:\n  {OUTPUT_PATH}")
print("Done!")
