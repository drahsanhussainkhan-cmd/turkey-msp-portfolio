"""
Interactive MSP Web Map - Turkish Waters
==========================================
Combines findings from all MSP portfolio projects (5-12) into a single
interactive Folium web map with toggleable layers, popups, and legends.
"""

import subprocess, sys, io, os

for pkg_name, import_name in [
    ("geopandas", "geopandas"), ("folium", "folium"),
    ("shapely", "shapely"), ("numpy", "numpy"),
    ("pandas", "pandas"), ("branca", "branca"),
]:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "-q"])

import geopandas as gpd
import folium
from folium import plugins
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import LineString, Point, mapping
import branca.colormap as cm
import json
import warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp")

EEZ_PATH = (BASE_DIR / "downloads" / "World_EEZ_v12_20231025" /
            "World_EEZ_v12_20231025" / "eez_v12.shp")
LAND_PATH = BASE_DIR / "downloads" / "ne_10m_land" / "ne_10m_land.shp"
MPA_BASE  = BASE_DIR / "downloads" / "WDPA_WDOECM_Feb2026_Public_TUR_shp"

OUTPUT_DIR = BASE_DIR / "project 13 interactive webmap"
OUTPUT_HTML = OUTPUT_DIR / "turkey_msp_interactive.html"

# --- Data from previous projects ---

# Project 5: Fishing hotspots (lon, lat, std_lon, std_lat)
FISHING_HOTSPOTS = [
    (36.5, 41.6, 0.8, 0.3, "Samsun-Trabzon"),
    (32.0, 41.8, 0.5, 0.2, "Sinop-Kastamonu"),
    (28.5, 41.5, 0.6, 0.3, "Istanbul-Bosphorus"),
    (40.5, 41.0, 0.4, 0.2, "Eastern Black Sea"),
    (26.5, 39.5, 0.5, 0.4, "North Aegean"),
    (26.0, 38.0, 0.6, 0.5, "Central Aegean (Izmir)"),
    (27.5, 37.0, 0.4, 0.3, "South Aegean (Bodrum)"),
    (30.5, 36.2, 0.8, 0.3, "Antalya Bay"),
    (34.0, 36.0, 0.5, 0.2, "Mersin-Adana"),
    (35.5, 36.2, 0.3, 0.2, "Iskenderun Bay"),
    (28.8, 40.7, 0.3, 0.15, "Marmara Sea"),
]

# Project 10: Shipping routes
SHIPPING_ROUTES = [
    ("Bosphorus Strait",       [(28.98, 41.02), (29.05, 41.10), (29.12, 41.18)]),
    ("Black Sea Main",         [(29.0, 41.5), (32.0, 42.0), (36.0, 41.8), (41.0, 41.5)]),
    ("Aegean Main",            [(26.0, 38.5), (27.0, 39.5), (28.5, 40.5), (29.0, 41.0)]),
    ("Mediterranean Main",     [(26.0, 36.0), (30.0, 35.8), (33.0, 36.0),
                                (36.0, 36.2), (40.0, 36.5)]),
    ("Istanbul-Izmir Coastal", [(29.0, 41.0), (28.0, 40.2), (27.5, 39.5), (27.0, 38.5)]),
]

# Project 8: River deltas (coastal erosion)
RIVER_DELTAS = [
    (41.7, 36.0, "Kizilirmak Delta",   "Very High", 0.913),
    (41.4, 36.5, "Yesilirmak Delta",    "Very High", 0.870),
    (38.7, 26.9, "Gediz Delta",         "High",      0.785),
    (37.5, 27.2, "B. Menderes Delta",   "High",      0.760),
    (36.3, 33.9, "Goksu Delta",         "High",      0.730),
    (36.8, 35.5, "Seyhan Delta",        "High",      0.750),
]

# Project 11: Coastal cities (cumulative impact)
CITIES = [
    (29.0, 41.0, "Istanbul",   1.0,  0.949),
    (27.1, 38.4, "Izmir",      0.6,  0.720),
    (30.7, 36.9, "Antalya",    0.5,  0.650),
    (34.6, 36.8, "Mersin",     0.4,  0.580),
    (39.7, 41.0, "Trabzon",    0.3,  0.510),
    (36.3, 41.3, "Samsun",     0.35, 0.540),
]

# Project 12: Tidal energy hotspots (lon, lat, velocity m/s, radius km, name)
TIDAL_HOTSPOTS = [
    (29.04, 41.08, 3.2, 3,  "Central Bosphorus"),
    (29.02, 41.04, 2.8, 3,  "Southern Bosphorus"),
    (29.06, 41.14, 2.5, 3,  "Northern Bosphorus"),
    (26.40, 40.12, 2.2, 5,  "Central Dardanelles"),
    (26.25, 40.07, 1.8, 4,  "Southern Dardanelles"),
    (26.60, 40.20, 1.6, 4,  "Northern Dardanelles"),
    (26.15, 39.10, 1.2, 8,  "Chios Strait"),
    (27.00, 37.60, 0.9, 6,  "Kos Channel"),
    (26.50, 38.50, 0.8, 7,  "Lesvos Channel"),
    (26.80, 40.40, 0.7, 5,  "Marmara Entrance"),
    (30.50, 41.20, 0.4, 10, "Bolu Coast"),
    (36.00, 41.70, 0.3, 12, "Samsun Cape"),
    (34.00, 42.00, 0.3, 10, "Sinop Peninsula"),
    (30.00, 36.10, 0.2, 15, "Antalya Coast"),
    (35.00, 36.30, 0.15, 12, "Mersin Coast"),
]

# Project 6: Offshore wind suitability zones (representative polygons)
# Approximated as rectangular zones where depth 0-50m and 5-50km offshore
WIND_ZONES = [
    ("Aegean Shelf",    [(26.0, 38.0), (27.5, 38.0), (27.5, 40.0), (26.0, 40.0)]),
    ("Marmara Shelf",   [(27.5, 40.3), (29.5, 40.3), (29.5, 41.0), (27.5, 41.0)]),
    ("W Black Sea",     [(28.5, 41.3), (32.0, 41.3), (32.0, 42.0), (28.5, 42.0)]),
    ("Antalya Shelf",   [(29.5, 36.0), (32.0, 36.0), (32.0, 36.8), (29.5, 36.8)]),
    ("Mersin Shelf",    [(33.0, 36.0), (36.0, 36.0), (36.0, 36.8), (33.0, 36.8)]),
]

# Project 9: MPA gap analysis - 5 habitat zones
HABITAT_ZONES = [
    ("Littoral (0-10m)",       0.12),
    ("Infralittoral (10-40m)", 0.08),
    ("Circalittoral (40-200m)", 0.10),
    ("Bathyal (200-2000m)",    0.05),
    ("Abyssal (>2000m)",       0.00),
]

# ============================================================================
# 1. LOAD VECTOR DATA
# ============================================================================
print("=" * 70)
print("INTERACTIVE MSP WEB MAP - Turkish Waters")
print("=" * 70)

print("\n[1/6] Loading Turkey EEZ boundary...")
eez_all = gpd.read_file(EEZ_PATH)
turkey_eez = eez_all[eez_all["SOVEREIGN1"] == "Turkey"].copy().to_crs("EPSG:4326")
turkey_eez_dissolved = turkey_eez.dissolve()
eez_area = turkey_eez["AREA_KM2"].sum()
print(f"  EEZ loaded ({eez_area:,.0f} km2)")

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
print(f"  {len(mpas_in_eez)} unique MPAs in Turkish EEZ")

# ============================================================================
# 2. CREATE BASE MAP
# ============================================================================
print("\n[3/6] Creating base map...")

# Center on Turkey
center_lat, center_lon = 39.0, 33.5
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles=None,
    control_scale=True,
    prefer_canvas=True,
)

# Add multiple basemap options
folium.TileLayer(
    "cartodbdark_matter",
    name="CartoDB Dark",
    control=True,
).add_to(m)

folium.TileLayer(
    "openstreetmap",
    name="OpenStreetMap",
    control=True,
).add_to(m)

folium.TileLayer(
    "cartodbpositron",
    name="CartoDB Light",
    control=True,
).add_to(m)

# ============================================================================
# 3. ADD LAYERS
# ============================================================================
print("\n[4/6] Adding data layers...")

# --- Layer 1: EEZ Boundary ---
print("  Adding EEZ boundary...")
eez_fg = folium.FeatureGroup(name="Turkey EEZ Boundary", show=True)
eez_geojson = turkey_eez_dissolved.to_json()
folium.GeoJson(
    eez_geojson,
    style_function=lambda x: {
        "fillColor": "transparent",
        "color": "#00d4ff",
        "weight": 2.5,
        "dashArray": "8 4",
        "fillOpacity": 0,
    },
    tooltip=f"Turkey EEZ ({eez_area:,.0f} km2)",
).add_to(eez_fg)
eez_fg.add_to(m)

# --- Layer 2: MPAs with popups ---
print("  Adding MPA polygons...")
mpa_fg = folium.FeatureGroup(name="Marine Protected Areas", show=True)

for _, row in mpas_in_eez.iterrows():
    name = row.get("NAME", "Unknown MPA")
    desig = row.get("DESIG_ENG", row.get("DESIG", "N/A"))
    status = row.get("STATUS", "N/A")
    area_km2 = row.get("REP_AREA", 0)
    iucn = row.get("IUCN_CAT", "N/A")

    popup_html = f"""
    <div style="font-family: Arial; min-width: 200px;">
        <h4 style="color: #2ecc71; margin-bottom: 5px;">{name}</h4>
        <table style="font-size: 12px;">
            <tr><td><b>Designation:</b></td><td>{desig}</td></tr>
            <tr><td><b>Status:</b></td><td>{status}</td></tr>
            <tr><td><b>IUCN Category:</b></td><td>{iucn}</td></tr>
            <tr><td><b>Area:</b></td><td>{area_km2:,.1f} km2</td></tr>
        </table>
    </div>
    """

    try:
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda x: {
                "fillColor": "#2ecc71",
                "color": "#27ae60",
                "weight": 2,
                "fillOpacity": 0.35,
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=name,
        ).add_to(mpa_fg)
    except Exception:
        pass

mpa_fg.add_to(m)

# --- Layer 3: Offshore Wind Suitability Zones ---
print("  Adding offshore wind zones...")
wind_fg = folium.FeatureGroup(name="Offshore Wind Zones (Proj 6)", show=True)

for zone_name, coords in WIND_ZONES:
    # coords are (lon, lat) -> folium wants (lat, lon)
    poly_coords = [(lat, lon) for lon, lat in coords]
    folium.Polygon(
        locations=poly_coords,
        color="#ffd700",
        weight=2,
        fill=True,
        fill_color="#ffd700",
        fill_opacity=0.2,
        popup=folium.Popup(
            f"<b>{zone_name}</b><br>Offshore Wind Screening Zone<br>"
            f"Criteria: Depth 0-50m, Shore 5-50km, MPA-free",
            max_width=250,
        ),
        tooltip=f"Wind Zone: {zone_name}",
    ).add_to(wind_fg)

wind_fg.add_to(m)

# --- Layer 4: Shipping Routes ---
print("  Adding shipping routes...")
ship_fg = folium.FeatureGroup(name="Shipping Routes (Proj 10)", show=True)

ship_colors = ["#ff4444", "#ff7744", "#ff9944", "#ffbb44", "#ffdd44"]
for idx, (route_name, coords) in enumerate(SHIPPING_ROUTES):
    # coords are (lon, lat) -> folium wants (lat, lon)
    route_latlon = [(lat, lon) for lon, lat in coords]
    folium.PolyLine(
        locations=route_latlon,
        color=ship_colors[idx % len(ship_colors)],
        weight=3,
        opacity=0.8,
        dash_array="10 5",
        popup=folium.Popup(
            f"<b>{route_name}</b><br>5 km buffer zone<br>"
            f"Source: AIS shipping data (synthetic)",
            max_width=250,
        ),
        tooltip=route_name,
    ).add_to(ship_fg)

ship_fg.add_to(m)

# --- Layer 5: Fishing Grounds (Heatmap) ---
print("  Adding fishing heatmap...")
np.random.seed(42)
heat_points = []
for lon, lat, std_lon, std_lat, name in FISHING_HOTSPOTS:
    n = 100
    pts_lon = np.random.normal(lon, std_lon * 0.3, n)
    pts_lat = np.random.normal(lat, std_lat * 0.3, n)
    for px, py in zip(pts_lat, pts_lon):
        heat_points.append([float(px), float(py), 1.0])

heat_fg = folium.FeatureGroup(name="Fishing Activity Heatmap (Proj 5)", show=True)
plugins.HeatMap(
    heat_points,
    min_opacity=0.3,
    max_val=3.0,
    radius=15,
    blur=20,
    gradient={0.2: "#0000ff", 0.4: "#00ffff", 0.6: "#00ff00",
              0.8: "#ffff00", 1.0: "#ff0000"},
).add_to(heat_fg)
heat_fg.add_to(m)

# --- Layer 6: Fishing Ground Markers ---
print("  Adding fishing ground markers...")
fish_marker_fg = folium.FeatureGroup(name="Fishing Ground Centers (Proj 5)", show=False)

for lon, lat, std_lon, std_lat, name in FISHING_HOTSPOTS:
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color="#3498db",
        fill=True,
        fill_color="#3498db",
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{name}</b><br>Fishing Ground Center<br>"
            f"Spread: {std_lon:.1f} x {std_lat:.1f} deg",
            max_width=200,
        ),
        tooltip=f"Fishing: {name}",
    ).add_to(fish_marker_fg)

fish_marker_fg.add_to(m)

# --- Layer 7: Coastal Erosion Hotspots ---
print("  Adding coastal erosion hotspots...")
erosion_fg = folium.FeatureGroup(name="Coastal Erosion Hotspots (Proj 8)", show=True)

erosion_colors = {"Very High": "#d73027", "High": "#f46d43"}
for lat, lon, name, risk_class, risk_score in RIVER_DELTAS:
    color = erosion_colors.get(risk_class, "#f46d43")
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(
            f"<div style='font-family:Arial; min-width:180px;'>"
            f"<h4 style='color:{color};'>{name}</h4>"
            f"<b>Risk Class:</b> {risk_class}<br>"
            f"<b>Risk Score:</b> {risk_score:.3f}<br>"
            f"<b>Factors:</b> Wave exposure (40%), "
            f"Elevation/Slope (35%), Delta proximity (25%)</div>",
            max_width=280,
        ),
        tooltip=f"Erosion: {name} ({risk_class})",
    ).add_to(erosion_fg)

    # Add a pulsing ring around Very High risk deltas
    if risk_class == "Very High":
        folium.Circle(
            location=[lat, lon],
            radius=15000,
            color=color,
            weight=1.5,
            fill=False,
            dash_array="5 5",
            opacity=0.5,
        ).add_to(erosion_fg)

erosion_fg.add_to(m)

# --- Layer 8: Cumulative Impact Cities ---
print("  Adding cumulative impact markers...")
city_fg = folium.FeatureGroup(name="Cumulative Impact Cities (Proj 11)", show=True)

for lon, lat, name, pop_weight, impact_score in CITIES:
    # Size proportional to impact
    radius = int(10 + impact_score * 15)

    # Color gradient: green (low) -> yellow -> red (high)
    if impact_score >= 0.8:
        color = "#e74c3c"
        impact_class = "Critical"
    elif impact_score >= 0.6:
        color = "#f39c12"
        impact_class = "High"
    else:
        color = "#e67e22"
        impact_class = "Moderate"

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=folium.Popup(
            f"<div style='font-family:Arial; min-width:200px;'>"
            f"<h4 style='color:{color};'>{name}</h4>"
            f"<b>Impact Score:</b> {impact_score:.3f}<br>"
            f"<b>Impact Class:</b> {impact_class}<br>"
            f"<b>Population Weight:</b> {pop_weight:.1f}<br>"
            f"<hr style='margin:5px 0;'>"
            f"<small>Halpern et al. framework: fishing, shipping,<br>"
            f"development, pollution, climate pressures</small></div>",
            max_width=300,
        ),
        tooltip=f"{name}: Impact {impact_score:.2f}",
    ).add_to(city_fg)

city_fg.add_to(m)

# --- Layer 9: Tidal Energy Sites ---
print("  Adding tidal energy sites...")
tidal_fg = folium.FeatureGroup(name="Tidal Energy Sites (Proj 12)", show=True)

for lon, lat, velocity, radius_km, name in TIDAL_HOTSPOTS:
    # Power density: P = 0.5 * rho * v^3
    power_density = 0.5 * 1025.0 * velocity ** 3
    viable = velocity >= 1.5

    if velocity >= 2.5:
        color = "#e74c3c"
        tier = "Excellent (>2.5 m/s)"
    elif velocity >= 1.5:
        color = "#f39c12"
        tier = "Good (1.5-2.5 m/s)"
    elif velocity >= 0.5:
        color = "#3498db"
        tier = "Marginal (0.5-1.5 m/s)"
    else:
        color = "#95a5a6"
        tier = "Low (<0.5 m/s)"

    folium.CircleMarker(
        location=[lat, lon],
        radius=max(4, int(velocity * 4)),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<div style='font-family:Arial; min-width:200px;'>"
            f"<h4 style='color:{color};'>{name}</h4>"
            f"<b>Peak Velocity:</b> {velocity:.1f} m/s<br>"
            f"<b>Power Density:</b> {power_density:,.0f} W/m2<br>"
            f"<b>Resource Tier:</b> {tier}<br>"
            f"<b>Influence Radius:</b> {radius_km} km<br>"
            f"<b>Viable for turbines:</b> {'Yes' if viable else 'No'}"
            f"</div>",
            max_width=280,
        ),
        tooltip=f"Tidal: {name} ({velocity:.1f} m/s)",
    ).add_to(tidal_fg)

    # Add influence area circle for viable sites
    if viable:
        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
        ).add_to(tidal_fg)

tidal_fg.add_to(m)

# --- Layer 10: MPA Gap Analysis Summary (markers at zone centroids) ---
print("  Adding MPA gap analysis markers...")
gap_fg = folium.FeatureGroup(name="MPA Gap Analysis (Proj 9)", show=False)

# Place markers along the EEZ for each habitat zone
gap_positions = [
    (40.5, 30.0),  # Littoral
    (39.5, 28.0),  # Infralittoral
    (38.0, 32.0),  # Circalittoral
    (37.0, 35.0),  # Bathyal
    (36.5, 38.0),  # Abyssal
]

gap_colors = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#d73027"]

for idx, ((zone_name, protection_pct), (glat, glon)) in enumerate(
    zip(HABITAT_ZONES, gap_positions)
):
    gap_needed = max(0, 30.0 - protection_pct * 100)
    folium.Marker(
        location=[glat, glon],
        icon=folium.DivIcon(
            html=f'<div style="background:{gap_colors[idx]};color:white;'
                 f'padding:3px 6px;border-radius:4px;font-size:11px;'
                 f'font-weight:bold;white-space:nowrap;text-align:center;'
                 f'border:1px solid white;">'
                 f'{zone_name}<br>{protection_pct*100:.1f}% protected</div>',
            icon_size=(140, 40),
            icon_anchor=(70, 20),
        ),
        popup=folium.Popup(
            f"<div style='font-family:Arial;'>"
            f"<h4>{zone_name}</h4>"
            f"<b>Current Protection:</b> {protection_pct*100:.2f}%<br>"
            f"<b>CBD 30x30 Target:</b> 30%<br>"
            f"<b>Gap:</b> {gap_needed:.1f}%<br>"
            f"<hr style='margin:5px 0;'>"
            f"<small>Overall: 0.10% protected<br>"
            f"Total gap: 72,757 km2 needed</small></div>",
            max_width=250,
        ),
    ).add_to(gap_fg)

gap_fg.add_to(m)

# ============================================================================
# 4. ADD MAP CONTROLS & OVERLAYS
# ============================================================================
print("\n[5/6] Adding controls and overlays...")

# Layer control
folium.LayerControl(collapsed=False, position="topright").add_to(m)

# Fullscreen button
plugins.Fullscreen(
    position="topleft",
    title="Fullscreen",
    title_cancel="Exit Fullscreen",
).add_to(m)

# Minimap
plugins.MiniMap(
    toggle_display=True,
    position="bottomright",
    tile_layer="cartodbdark_matter",
    zoom_level_offset=-5,
    width=150,
    height=120,
).add_to(m)

# Measure tool
plugins.MeasureControl(
    position="bottomleft",
    primary_length_unit="kilometers",
    secondary_length_unit="miles",
    primary_area_unit="sqkilometers",
).add_to(m)

# Mouse position
plugins.MousePosition(
    position="bottomleft",
    separator=" | ",
    prefix="Cursor:",
    num_digits=4,
).add_to(m)

# --- Title overlay ---
title_html = """
<div style="
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(13, 27, 42, 0.92);
    color: white;
    padding: 12px 28px;
    border-radius: 8px;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid rgba(0, 212, 255, 0.4);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    pointer-events: none;
">
    Marine Spatial Planning - Turkish Waters
    <span style="font-size: 12px; opacity: 0.7; display: block; text-align: center; margin-top: 2px;">
        Interactive MSP Portfolio | Projects 5-12
    </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# --- Legend overlay ---
legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 10px;
    z-index: 9998;
    background: rgba(13, 27, 42, 0.92);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    max-width: 220px;
    line-height: 1.6;
">
    <div style="font-weight:bold; font-size:13px; margin-bottom:8px;
                border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:5px;">
        Legend
    </div>
    <div><span style="color:#00d4ff;">- - -</span> EEZ Boundary</div>
    <div><span style="color:#2ecc71;">&#9632;</span> Marine Protected Areas</div>
    <div><span style="color:#ffd700;">&#9632;</span> Offshore Wind Zones</div>
    <div><span style="color:#ff4444;">---</span> Shipping Routes</div>
    <div><span style="color:#ff6600;">&#9679;</span> Fishing Heatmap</div>
    <div style="margin-top:4px; font-weight:bold; font-size:11px; opacity:0.7;">
        Coastal Erosion Risk
    </div>
    <div><span style="color:#d73027;">&#9679;</span> Very High Risk Delta</div>
    <div><span style="color:#f46d43;">&#9679;</span> High Risk Delta</div>
    <div style="margin-top:4px; font-weight:bold; font-size:11px; opacity:0.7;">
        Tidal Energy (m/s)
    </div>
    <div><span style="color:#e74c3c;">&#9679;</span> Excellent (&gt;2.5)</div>
    <div><span style="color:#f39c12;">&#9679;</span> Good (1.5-2.5)</div>
    <div><span style="color:#3498db;">&#9679;</span> Marginal (0.5-1.5)</div>
    <div style="margin-top:4px; font-weight:bold; font-size:11px; opacity:0.7;">
        Cumulative Impact
    </div>
    <div><span style="color:#e74c3c;">&#9679;</span> Critical (&gt;0.8)</div>
    <div><span style="color:#f39c12;">&#9679;</span> High (0.6-0.8)</div>
    <div><span style="color:#e67e22;">&#9679;</span> Moderate (&lt;0.6)</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# --- Summary statistics panel ---
stats_html = """
<div id="stats-panel" style="
    position: fixed;
    top: 80px;
    right: 10px;
    z-index: 9997;
    background: rgba(13, 27, 42, 0.90);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    max-width: 200px;
    line-height: 1.7;
    cursor: pointer;
" onclick="this.style.display='none'">
    <div style="font-weight:bold; font-size:12px; margin-bottom:6px;
                border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:4px;">
        Key Findings
    </div>
    <div>EEZ Area: ~243,000 km2</div>
    <div>MPA Coverage: 0.10%</div>
    <div>Wind Potential: 2,778 km2</div>
    <div>Wind Capacity: 41.7 GW</div>
    <div>Tidal Potential: 750 MW</div>
    <div>Erosion Risk (H+VH): 12.1%</div>
    <div>High Impact Area: 15.5%</div>
    <div>Habitat Suitable: 15.1%</div>
    <div style="margin-top:6px; font-size:10px; opacity:0.5; text-align:center;">
        Click to dismiss
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(stats_html))

# ============================================================================
# 5. SAVE MAP
# ============================================================================
print("\n[6/6] Saving interactive map...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
m.save(str(OUTPUT_HTML))
file_size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
print(f"  Saved: {OUTPUT_HTML}")
print(f"  File size: {file_size_mb:.1f} MB")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("INTERACTIVE WEB MAP - Summary")
print("=" * 70)

print(f"""
  Output:       {OUTPUT_HTML.name}
  File size:    {file_size_mb:.1f} MB
  Basemaps:     CartoDB Dark, OpenStreetMap, CartoDB Light
  Center:       {center_lat}N, {center_lon}E (zoom {6})

  LAYERS INCLUDED:
  ---------------------------------------------------------------
  1. Turkey EEZ Boundary         - Dashed cyan outline
  2. Marine Protected Areas      - Green polygons with popups
  3. Offshore Wind Zones         - Yellow screening areas (Proj 6)
  4. Shipping Routes             - Red dashed lines (Proj 10)
  5. Fishing Activity Heatmap    - Blue-to-red heat overlay (Proj 5)
  6. Fishing Ground Centers      - Blue markers (Proj 5, hidden)
  7. Coastal Erosion Hotspots    - Red/orange circles (Proj 8)
  8. Cumulative Impact Cities    - Sized circles (Proj 11)
  9. Tidal Energy Sites          - Colored by velocity (Proj 12)
  10. MPA Gap Analysis           - Zone labels (Proj 9, hidden)

  CONTROLS:
  ---------------------------------------------------------------
  - Layer toggle (top-right)     - Fullscreen button (top-left)
  - Minimap (bottom-right)       - Measure tool (bottom-left)
  - Mouse coordinates            - Scale bar
  - Legend (bottom-left)         - Key Findings panel (top-right)

  Open {OUTPUT_HTML.name} in any web browser to explore the map.
""")

print("=" * 70)
print("DONE - Interactive MSP web map generated successfully!")
print("=" * 70)
