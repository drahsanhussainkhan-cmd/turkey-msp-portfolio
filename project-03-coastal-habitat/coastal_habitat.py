import subprocess
import sys
import requests
import warnings
warnings.filterwarnings('ignore')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

for pkg in ["requests", "pandas", "geopandas", "matplotlib", "shapely"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, box

print("Fetching Posidonia oceanica data from OBIS...")

def get_obis_data(species_name):
    url = "https://api.obis.org/v3/occurrence"
    params = {
        "scientificname": species_name,
        "geometry": "POLYGON((25 35,42 35,42 43,25 43,25 35))",
        "size": 5000
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        records = data.get("results", [])
        print(f"Found {len(records)} records for {species_name}")
        return records
    except Exception as e:
        print(f"Error fetching {species_name}: {e}")
        return []

posidonia_records = get_obis_data("Posidonia oceanica")
cymodocea_records = get_obis_data("Cymodocea nodosa")

def records_to_gdf(records):
    points = []
    for r in records:
        lat = r.get("decimalLatitude")
        lon = r.get("decimalLongitude")
        if lat and lon:
            points.append(Point(lon, lat))
    if points:
        return gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

posidonia_gdf = records_to_gdf(posidonia_records)
cymodocea_gdf = records_to_gdf(cymodocea_records)

print(f"Posidonia points: {len(posidonia_gdf)}")
print(f"Cymodocea points: {len(cymodocea_gdf)}")

# Load EEZ
eez_path = r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp\downloads\World_EEZ_v12_20231025\World_EEZ_v12_20231025\eez_v12.shp"
eez = gpd.read_file(eez_path)
turkey_eez = eez[eez['SOVEREIGN1'] == 'Turkey']

# Load land
land_path = r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp\downloads\ne_10m_land\ne_10m_land.shp"
world = gpd.read_file(land_path)
turkey_box = box(24, 34, 45, 44)
land_clip = world.clip(turkey_box)

# Create map
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_facecolor('#AED9E0')
fig.patch.set_facecolor('#ffffff')

land_clip.plot(ax=ax, color='#F5F0E8', edgecolor='#AAAAAA', linewidth=0.5, zorder=2)
turkey_eez.plot(ax=ax, color='none', edgecolor='#2166AC', linewidth=1.5, linestyle='--', zorder=3)

if len(posidonia_gdf) > 0:
    posidonia_gdf.plot(ax=ax, color='#2d6a4f', markersize=20, marker='o', alpha=0.7, zorder=5)

if len(cymodocea_gdf) > 0:
    cymodocea_gdf.plot(ax=ax, color='#52b788', markersize=20, marker='^', alpha=0.7, zorder=5)

ax.set_xlim(24, 45)
ax.set_ylim(34, 44)
ax.set_title("Coastal Habitat Distribution in Turkish Waters", fontsize=18, fontweight='bold', pad=15, color='#1a1a2e')
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.grid(True, alpha=0.3, linestyle=':')

legend_elements = [
    mpatches.Patch(facecolor='#F5F0E8', edgecolor='#AAAAAA', label='Land'),
    mpatches.Patch(facecolor='#AED9E0', label='Ocean'),
    plt.Line2D([0], [0], color='#2166AC', linestyle='--', linewidth=1.5, label='Turkey EEZ'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2d6a4f', markersize=10, label=f'Posidonia oceanica (n={len(posidonia_gdf)})'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#52b788', markersize=10, label=f'Cymodocea nodosa (n={len(cymodocea_gdf)})'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
ax.annotate('Data: OBIS, Marine Regions EEZ v12, Natural Earth | Projection: WGS84',
            xy=(0.01, 0.02), xycoords='axes fraction', fontsize=7, color='#555555')

plt.tight_layout()
output_path = r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp\project 3 coastal habitat\turkey_coastal_habitat_map.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Map saved!")
plt.show()