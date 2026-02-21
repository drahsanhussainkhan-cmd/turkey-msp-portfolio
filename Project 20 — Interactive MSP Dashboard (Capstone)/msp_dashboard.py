"""
Marine Spatial Planning Dashboard - Turkish Waters
=====================================================
Interactive Streamlit dashboard summarising all findings
from the MSP portfolio (Projects 1-19).

Run with:  streamlit run msp_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Turkey MSP Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # project 1 msp/

MAP_IMAGES = {
    "integrated_msp":    BASE_DIR / "project 19 msp scenario"     / "turkey_integrated_msp.png",
    "cumulative_impact": BASE_DIR / "project 11 cumulative impact" / "turkey_cumulative_impact.png",
    "mpa_gap":           BASE_DIR / "project 9 mpa gap analysis"  / "turkey_mpa_gap_analysis.png",
    "offshore_wind":     BASE_DIR / "project 6 offshore wind"     / "turkey_offshore_wind_suitability.png",
    "shipping_conflict": BASE_DIR / "project 10 shipping conflict"/ "turkey_shipping_conflict.png",
    "coastal_erosion":   BASE_DIR / "project 8 coastal erosion"   / "turkey_coastal_erosion_risk.png",
    "tidal_energy":      BASE_DIR / "project 12 tidal energy"     / "turkey_tidal_energy.png",
    "blue_carbon":       BASE_DIR / "project 17 blue carbon"      / "turkey_blue_carbon.png",
    "fisheries_zones":   BASE_DIR / "project 18 fisheries zones"  / "turkey_fisheries_zones.png",
    "mpa_effectiveness": BASE_DIR / "project 15 mpa effectiveness"/ "turkey_mpa_effectiveness.png",
    "wind_impact":       BASE_DIR / "project 16 wind impact"      / "turkey_wind_cumulative_impact.png",
    "habitat":           BASE_DIR / "project 7 habitat suitability"/ "turkey_anchovy_suitability.png",
}

# ============================================================================
# HARDCODED METRICS FROM PROJECTS 1-19
# ============================================================================
METRICS = {
    "eez_area_km2": 243_783,
    "coastline_km": 8_333,
    "mpa_existing_km2": 237.5,
    "mpa_coverage_pct": 0.10,
    "mpa_count": 5,
    "cbd_target_pct": 30.0,
    "mpa_gap_km2": 72_757,
    "wind_suitable_km2": 2_778,
    "wind_capacity_gw": 41.7,
    "wind_suitable_pct": 1.13,
    "tidal_potential_mw": 750,
    "tidal_area_km2": 18.3,
    "tidal_peak_ms": 3.2,
    "cumulative_high_km2": 41_102,
    "cumulative_high_pct": 15.5,
    "cumulative_hotspot": "Istanbul",
    "cumulative_hotspot_score": 0.949,
    "shipping_conflict_km2": 5_829,
    "wind_conflict_km2": 138,
    "mpa_conflict_km2": 0.3,
    "erosion_high_km2": 7_659,
    "erosion_high_pct": 12.1,
    "erosion_top_delta": "Kizilirmak",
    "erosion_top_score": 0.913,
    "blue_carbon_area_km2": 550,
    "blue_carbon_tC": 4_878_000,
    "blue_carbon_value_eur": 895_000_000,
    "habitat_high_km2": 36_621,
    "habitat_high_pct": 15.1,
    "habitat_top_basin": "Aegean Sea",
    "mpa_effectiveness_reduction": 51.2,
    "baci_score": -175.2,
    "baci_cohens_d": 2.95,
    "msp_managed_pct": 38.8,
    "msp_mpa_proposed_pct": 11.7,
}

# Wind scenarios from Project 16
WIND_SCENARIOS = {
    "Conservative (500 MW)": {
        "capacity_mw": 500, "turbines": 33, "annual_gwh": 1_752,
        "footprint_km2": 8, "fishing_km2": 2, "shipping_km2": 0,
        "habitat_km2": 5, "visual_km2": 8, "noise_km2": 385,
        "total_impact_km2": 393, "eez_pct": 0.15,
    },
    "Moderate (5 GW)": {
        "capacity_mw": 5_000, "turbines": 333, "annual_gwh": 17_520,
        "footprint_km2": 323, "fishing_km2": 103, "shipping_km2": 94,
        "habitat_km2": 178, "visual_km2": 323, "noise_km2": 3_685,
        "total_impact_km2": 4_008, "eez_pct": 1.53,
    },
    "Ambitious (41.7 GW)": {
        "capacity_mw": 41_700, "turbines": 2_780, "annual_gwh": 146_117,
        "footprint_km2": 1_648, "fishing_km2": 363, "shipping_km2": 110,
        "habitat_km2": 602, "visual_km2": 1_646, "noise_km2": 16_422,
        "total_impact_km2": 18_070, "eez_pct": 6.89,
    },
}

# City cumulative impact scores
CITY_SCORES = {
    "Istanbul": 0.949, "Izmir": 0.720, "Antalya": 0.650,
    "Mersin": 0.580, "Samsun": 0.540, "Trabzon": 0.510,
}

# Erosion delta scores
DELTA_SCORES = {
    "Kizilirmak": 0.913, "Yesilirmak": 0.870, "Gediz": 0.785,
    "B. Menderes": 0.760, "Seyhan": 0.750, "Goksu": 0.730,
}

# Habitat zones for MPA gap
HABITAT_ZONES = {
    "Littoral (0-10m)": {"protection": 0.12, "area": 4_200},
    "Infralittoral (10-40m)": {"protection": 0.08, "area": 12_500},
    "Circalittoral (40-200m)": {"protection": 0.10, "area": 38_600},
    "Bathyal (200-2000m)": {"protection": 0.05, "area": 135_000},
    "Abyssal (>2000m)": {"protection": 0.00, "area": 71_933},
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Dark navy theme overrides */
    .stApp {
        background-color: #0d1b2a;
    }
    .main .block-container {
        padding-top: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1b2838 0%, #162032 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00b4d8;
        margin-bottom: 10px;
    }
    .metric-card.critical { border-left-color: #e74c3c; }
    .metric-card.warning  { border-left-color: #f39c12; }
    .metric-card.good     { border-left-color: #2ecc71; }
    .metric-card.info     { border-left-color: #00b4d8; }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ecf0f1;
        margin: 0;
    }
    .metric-label {
        font-size: 13px;
        color: #7f8c8d;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 12px;
        margin: 4px 0 0 0;
    }
    .metric-delta.bad  { color: #e74c3c; }
    .metric-delta.good { color: #2ecc71; }
    .metric-delta.neutral { color: #95a5a6; }

    /* Section headers */
    .section-header {
        color: #00b4d8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #1a3a5c;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a1628;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        color: #00b4d8;
    }

    /* Image containers */
    .map-container {
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def metric_card(label, value, delta=None, delta_good=True, status="info"):
    """Render a styled metric card."""
    delta_html = ""
    if delta:
        delta_cls = "good" if delta_good else "bad"
        delta_html = f'<p class="metric-delta {delta_cls}">{delta}</p>'

    return f"""
    <div class="metric-card {status}">
        <p class="metric-label">{label}</p>
        <p class="metric-value">{value}</p>
        {delta_html}
    </div>
    """


def show_map(key, caption=""):
    """Display a map image if it exists."""
    path = MAP_IMAGES.get(key)
    if path and path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Map image not found: {key}")


def plotly_dark_layout(fig, title=""):
    """Apply dark theme to a plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#ecf0f1")),
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#131f30",
        font=dict(color="#bdc3c7", size=12),
        legend=dict(bgcolor="rgba(13,27,42,0.8)", bordercolor="#1a3a5c"),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    fig.update_xaxes(gridcolor="#1a3a5c", zerolinecolor="#1a3a5c")
    fig.update_yaxes(gridcolor="#1a3a5c", zerolinecolor="#1a3a5c")
    return fig


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("# 🌊 MSP Dashboard")
    st.markdown("**Turkish Waters 2026**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Conservation", "Energy", "Conflicts",
         "Cumulative Impact", "Coastal Risk", "Scenario Planner"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### Portfolio Summary")
    st.markdown(f"- **19** projects completed")
    st.markdown(f"- **{METRICS['eez_area_km2']:,}** km2 EEZ analysed")
    st.markdown(f"- **4** marine basins covered")
    st.markdown(f"- **{METRICS['coastline_km']:,}** km coastline")

    st.markdown("---")
    st.markdown(
        "<small style='color:#566573;'>MSP Portfolio | Feb 2026<br>"
        "Data: GEBCO, WDPA, VLIZ EEZ</small>",
        unsafe_allow_html=True,
    )


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "Overview":
    st.markdown("# Integrated Marine Spatial Plan — Turkish Waters")
    st.markdown(
        '<p class="section-header">Key Performance Indicators</p>',
        unsafe_allow_html=True,
    )

    # Row 1: Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "EEZ Area", f"{METRICS['eez_area_km2']:,} km2",
            delta="4 marine basins", delta_good=True, status="info",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "MPA Coverage", f"{METRICS['mpa_coverage_pct']:.1f}%",
            delta=f"Target: 30% (gap: {METRICS['mpa_gap_km2']:,} km2)",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Wind Energy Potential", f"{METRICS['wind_capacity_gw']} GW",
            delta=f"{METRICS['wind_suitable_km2']:,} km2 suitable",
            delta_good=True, status="good",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Cumulative Impact", f"{METRICS['cumulative_high_pct']}%",
            delta=f"{METRICS['cumulative_high_km2']:,} km2 high impact",
            delta_good=False, status="warning",
        ), unsafe_allow_html=True)

    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(metric_card(
            "Blue Carbon Value", f"EUR {METRICS['blue_carbon_value_eur']/1e6:,.0f}M",
            delta=f"{METRICS['blue_carbon_area_km2']:,} km2 ecosystems",
            delta_good=True, status="good",
        ), unsafe_allow_html=True)
    with c6:
        st.markdown(metric_card(
            "Tidal Energy", f"{METRICS['tidal_potential_mw']} MW",
            delta=f"Peak: {METRICS['tidal_peak_ms']} m/s (Bosphorus)",
            delta_good=True, status="info",
        ), unsafe_allow_html=True)
    with c7:
        st.markdown(metric_card(
            "Spatial Conflicts", f"{METRICS['shipping_conflict_km2']:,} km2",
            delta="Shipping vs fishing (largest)",
            delta_good=False, status="warning",
        ), unsafe_allow_html=True)
    with c8:
        st.markdown(metric_card(
            "Erosion High Risk", f"{METRICS['erosion_high_pct']}%",
            delta=f"{METRICS['erosion_high_km2']:,} km2 coastline",
            delta_good=False, status="warning",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Integrated MSP map
    st.markdown(
        '<p class="section-header">Integrated Marine Spatial Plan</p>',
        unsafe_allow_html=True,
    )
    show_map("integrated_msp", "Proposed spatial allocation framework (Project 19)")

    # MSP zone allocation chart
    st.markdown(
        '<p class="section-header">Spatial Allocation Breakdown</p>',
        unsafe_allow_html=True,
    )

    zone_data = {
        "Existing MPAs": 571, "Proposed MPAs": 27_995,
        "Shipping Corridors": 31_147, "Wind Energy": 188,
        "Blue Carbon": 352, "Fisheries Mgmt": 32_580,
        "Multi-Use": 1_550, "Exclusion": 186, "Open / General": 149_215,
    }
    zone_colors = {
        "Existing MPAs": "#c0392b", "Proposed MPAs": "#e74c3c",
        "Shipping Corridors": "#8e44ad", "Wind Energy": "#f39c12",
        "Blue Carbon": "#27ae60", "Fisheries Mgmt": "#3498db",
        "Multi-Use": "#1abc9c", "Exclusion": "#2c3e50",
        "Open / General": "#d5dbdb",
    }

    fig_zone = go.Figure(data=[go.Pie(
        labels=list(zone_data.keys()),
        values=list(zone_data.values()),
        marker=dict(colors=[zone_colors[k] for k in zone_data]),
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}<br>%{value:,.0f} km2<br>%{percent}<extra></extra>",
    )])
    plotly_dark_layout(fig_zone, "EEZ Zone Allocation (km2)")
    fig_zone.update_layout(height=450)
    st.plotly_chart(fig_zone, use_container_width=True)


# ============================================================================
# PAGE: CONSERVATION
# ============================================================================

elif page == "Conservation":
    st.markdown("# Marine Conservation Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card(
            "Current MPA Coverage", f"{METRICS['mpa_coverage_pct']:.2f}%",
            delta=f"Only {METRICS['mpa_existing_km2']} km2 protected",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "CBD 30x30 Gap", f"{METRICS['mpa_gap_km2']:,} km2",
            delta="29.9% additional coverage needed",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "MPA Effectiveness", f"{METRICS['mpa_effectiveness_reduction']:.0f}% reduction",
            delta=f"BACI Cohen's d = {METRICS['baci_cohens_d']:.2f} (large)",
            delta_good=True, status="good",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # MPA gap by habitat zone
    st.markdown(
        '<p class="section-header">Protection Gap by Habitat Zone</p>',
        unsafe_allow_html=True,
    )

    zones = list(HABITAT_ZONES.keys())
    current_pct = [HABITAT_ZONES[z]["protection"] for z in zones]
    gap_pct = [30.0 - HABITAT_ZONES[z]["protection"] for z in zones]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=zones, y=current_pct, name="Current Protection (%)",
        marker_color="#27ae60", text=[f"{v:.2f}%" for v in current_pct],
        textposition="outside",
    ))
    fig_gap.add_trace(go.Bar(
        x=zones, y=gap_pct, name="Gap to 30% Target",
        marker_color="#e74c3c", text=[f"{v:.1f}%" for v in gap_pct],
        textposition="outside",
    ))
    plotly_dark_layout(fig_gap, "MPA Coverage vs CBD 30x30 Target")
    fig_gap.update_layout(barmode="stack", height=400)
    st.plotly_chart(fig_gap, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_map("mpa_gap", "MPA Gap Analysis (Project 9)")
    with col2:
        show_map("mpa_effectiveness", "MPA Effectiveness - BACI Design (Project 15)")


# ============================================================================
# PAGE: ENERGY
# ============================================================================

elif page == "Energy":
    st.markdown("# Renewable Energy Potential")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(
            "Wind Capacity", f"{METRICS['wind_capacity_gw']} GW",
            status="good",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Wind Suitable Area", f"{METRICS['wind_suitable_km2']:,} km2",
            delta=f"{METRICS['wind_suitable_pct']}% of EEZ", status="info",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Tidal Potential", f"{METRICS['tidal_potential_mw']} MW",
            status="good",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(
            "Peak Tidal Current", f"{METRICS['tidal_peak_ms']} m/s",
            delta="Bosphorus Strait", status="info",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Wind scenario comparison
    st.markdown(
        '<p class="section-header">Offshore Wind Scenario Comparison</p>',
        unsafe_allow_html=True,
    )

    selected_scenario = st.selectbox(
        "Select Wind Development Scenario",
        list(WIND_SCENARIOS.keys()),
        index=1,
    )

    sc = WIND_SCENARIOS[selected_scenario]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Capacity", f"{sc['capacity_mw']:,} MW")
    with mc2:
        st.metric("Turbines", f"{sc['turbines']:,}")
    with mc3:
        st.metric("Annual Energy", f"{sc['annual_gwh']:,} GWh")
    with mc4:
        st.metric("EEZ Impact", f"{sc['eez_pct']:.2f}%")

    # Impact breakdown chart
    impact_cats = ["Fishing", "Shipping", "Habitat", "Visual", "Noise"]
    impact_colors_map = ["#e74c3c", "#8e44ad", "#27ae60", "#f39c12", "#3498db"]

    fig_impact = go.Figure()
    for sn, color in zip(WIND_SCENARIOS.keys(),
                         ["#27ae60", "#f39c12", "#e74c3c"]):
        s = WIND_SCENARIOS[sn]
        vals = [s["fishing_km2"], s["shipping_km2"], s["habitat_km2"],
                s["visual_km2"], s["noise_km2"]]
        fig_impact.add_trace(go.Bar(
            x=impact_cats, y=vals, name=sn,
            marker_color=color, opacity=0.8,
        ))
    plotly_dark_layout(fig_impact, "Impact by Category (km2)")
    fig_impact.update_layout(barmode="group", height=400)
    st.plotly_chart(fig_impact, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_map("offshore_wind", "Offshore Wind Suitability (Project 6)")
    with col2:
        show_map("tidal_energy", "Tidal Energy Assessment (Project 12)")

    st.markdown("---")
    show_map("wind_impact", "Wind Cumulative Impact Assessment (Project 16)")


# ============================================================================
# PAGE: CONFLICTS
# ============================================================================

elif page == "Conflicts":
    st.markdown("# Spatial Conflict Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card(
            "Total Conflict Area", f"{METRICS['shipping_conflict_km2']:,} km2",
            delta="Shipping vs all uses", status="warning",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Wind Zone Conflicts", f"{METRICS['wind_conflict_km2']} km2",
            delta="Shipping vs wind zones", status="warning",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "MPA Conflicts", f"{METRICS['mpa_conflict_km2']} km2",
            delta="Shipping vs MPAs", status="info",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Conflict breakdown
    conflict_data = {
        "Fishing Grounds": METRICS["shipping_conflict_km2"],
        "Wind Energy Zones": METRICS["wind_conflict_km2"],
        "MPAs": METRICS["mpa_conflict_km2"],
    }

    fig_conf = go.Figure(data=[go.Bar(
        x=list(conflict_data.keys()),
        y=list(conflict_data.values()),
        marker_color=["#e74c3c", "#f39c12", "#27ae60"],
        text=[f"{v:,.0f} km2" for v in conflict_data.values()],
        textposition="outside",
    )])
    plotly_dark_layout(fig_conf, "Shipping Corridor Conflicts by Category")
    fig_conf.update_layout(height=350, yaxis_type="log",
                           yaxis_title="Conflict Area (km2, log scale)")
    st.plotly_chart(fig_conf, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_map("shipping_conflict", "Shipping Conflict Analysis (Project 10)")
    with col2:
        show_map("fisheries_zones", "Fisheries Management Zones (Project 18)")


# ============================================================================
# PAGE: CUMULATIVE IMPACT
# ============================================================================

elif page == "Cumulative Impact":
    st.markdown("# Cumulative Human Impact")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card(
            "High Impact Area", f"{METRICS['cumulative_high_pct']}%",
            delta=f"{METRICS['cumulative_high_km2']:,} km2",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Primary Hotspot", METRICS["cumulative_hotspot"],
            delta=f"Score: {METRICS['cumulative_hotspot_score']:.3f}",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Blue Carbon at Risk", f"EUR {METRICS['blue_carbon_value_eur']/1e6:,.0f}M",
            delta=f"{METRICS['blue_carbon_area_km2']} km2 ecosystems",
            delta_good=False, status="warning",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # City impact scores
    st.markdown(
        '<p class="section-header">Cumulative Impact by Coastal City</p>',
        unsafe_allow_html=True,
    )

    cities = list(CITY_SCORES.keys())
    scores = list(CITY_SCORES.values())
    colors = ["#e74c3c" if s >= 0.8 else "#f39c12" if s >= 0.6 else "#e67e22"
              for s in scores]

    fig_city = go.Figure(data=[go.Bar(
        x=cities, y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    )])
    plotly_dark_layout(fig_city, "Cumulative Impact Score by City (Halpern et al. Framework)")
    fig_city.update_layout(height=380, yaxis_range=[0, 1.1],
                           yaxis_title="Impact Score (0-1)")
    # Add threshold line
    fig_city.add_hline(y=0.8, line_dash="dash", line_color="#e74c3c",
                       annotation_text="Critical threshold",
                       annotation_font_color="#e74c3c")
    st.plotly_chart(fig_city, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_map("cumulative_impact", "Cumulative Impact Map (Project 11)")
    with col2:
        show_map("blue_carbon", "Blue Carbon Ecosystems (Project 17)")


# ============================================================================
# PAGE: COASTAL RISK
# ============================================================================

elif page == "Coastal Risk":
    st.markdown("# Coastal Erosion Risk")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card(
            "High + Very High Risk", f"{METRICS['erosion_high_pct']}%",
            delta=f"{METRICS['erosion_high_km2']:,} km2 of coastline",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "Highest Risk Delta", METRICS["erosion_top_delta"],
            delta=f"Score: {METRICS['erosion_top_score']:.3f}",
            delta_good=False, status="critical",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "Deltas Assessed", "6",
            delta="Black Sea, Aegean, Mediterranean",
            delta_good=True, status="info",
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Delta risk scores
    st.markdown(
        '<p class="section-header">Erosion Risk by River Delta</p>',
        unsafe_allow_html=True,
    )

    deltas = list(DELTA_SCORES.keys())
    d_scores = list(DELTA_SCORES.values())
    d_colors = ["#d73027" if s >= 0.85 else "#f46d43" for s in d_scores]
    basins = ["Black Sea", "Black Sea", "Aegean", "Aegean",
              "Mediterranean", "Mediterranean"]

    fig_delta = go.Figure(data=[go.Bar(
        x=deltas, y=d_scores,
        marker_color=d_colors,
        text=[f"{s:.3f}" for s in d_scores],
        textposition="outside",
        hovertemplate="%{x}<br>Score: %{y:.3f}<br>Basin: %{customdata}<extra></extra>",
        customdata=basins,
    )])
    plotly_dark_layout(fig_delta, "Coastal Erosion Risk Score by Delta")
    fig_delta.update_layout(height=380, yaxis_range=[0, 1.05],
                            yaxis_title="Risk Score (0-1)")
    fig_delta.add_hline(y=0.75, line_dash="dash", line_color="#f39c12",
                        annotation_text="High risk threshold",
                        annotation_font_color="#f39c12")
    st.plotly_chart(fig_delta, use_container_width=True)

    show_map("coastal_erosion", "Coastal Erosion Risk Index (Project 8)")

    st.markdown("---")
    show_map("habitat", "European Anchovy Habitat Suitability (Project 7)")


# ============================================================================
# PAGE: SCENARIO PLANNER
# ============================================================================

elif page == "Scenario Planner":
    st.markdown("# Interactive Scenario Planner")
    st.markdown("Adjust targets and explore trade-offs in real time.")

    st.markdown("---")

    # MPA Target Slider
    st.markdown(
        '<p class="section-header">MPA Coverage Target</p>',
        unsafe_allow_html=True,
    )

    mpa_target = st.slider(
        "Set MPA coverage target (%)",
        min_value=0.1, max_value=30.0, value=10.0, step=0.5,
        format="%.1f%%",
    )

    current_km2 = METRICS["mpa_existing_km2"]
    target_km2 = METRICS["eez_area_km2"] * mpa_target / 100
    gap_km2 = max(0, target_km2 - current_km2)
    gap_pct = max(0, mpa_target - METRICS["mpa_coverage_pct"])

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(metric_card(
            "Current Coverage", f"{METRICS['mpa_coverage_pct']:.2f}%",
            delta=f"{current_km2:,.1f} km2", status="critical",
        ), unsafe_allow_html=True)
    with mc2:
        color = "good" if mpa_target <= 5 else ("warning" if mpa_target <= 15 else "critical")
        st.markdown(metric_card(
            "Target Coverage", f"{mpa_target:.1f}%",
            delta=f"{target_km2:,.0f} km2 needed", status=color,
        ), unsafe_allow_html=True)
    with mc3:
        st.markdown(metric_card(
            "Additional Area Needed", f"{gap_km2:,.0f} km2",
            delta=f"+{gap_pct:.1f}% above current", delta_good=False,
            status="warning" if gap_km2 > 0 else "good",
        ), unsafe_allow_html=True)
    with mc4:
        # Estimate cost (rough: EUR 500/km2/year management)
        annual_cost = gap_km2 * 500
        st.markdown(metric_card(
            "Est. Annual Mgmt Cost", f"EUR {annual_cost/1e6:,.1f}M/yr",
            delta="At EUR 500/km2/year", status="info",
        ), unsafe_allow_html=True)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mpa_target,
        delta={"reference": METRICS["mpa_coverage_pct"],
               "increasing": {"color": "#27ae60"}},
        number={"suffix": "%", "font": {"size": 40, "color": "#ecf0f1"}},
        gauge={
            "axis": {"range": [0, 30], "tickcolor": "#bdc3c7",
                     "tickfont": {"color": "#bdc3c7"}},
            "bar": {"color": "#00b4d8"},
            "bgcolor": "#131f30",
            "bordercolor": "#1a3a5c",
            "steps": [
                {"range": [0, 0.1], "color": "#e74c3c"},
                {"range": [0.1, 10], "color": "#f39c12"},
                {"range": [10, 20], "color": "#f1c40f"},
                {"range": [20, 30], "color": "#27ae60"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 3},
                "thickness": 0.8, "value": 30,
            },
        },
    ))
    plotly_dark_layout(fig_gauge, "MPA Coverage Progress Toward CBD 30x30")
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # Wind scenario selector
    st.markdown(
        '<p class="section-header">Wind Energy Development Scenario</p>',
        unsafe_allow_html=True,
    )

    wind_choice = st.selectbox(
        "Select scenario", list(WIND_SCENARIOS.keys()), index=1,
        key="wind_planner",
    )
    wsc = WIND_SCENARIOS[wind_choice]

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown(metric_card(
            "Capacity", f"{wsc['capacity_mw']:,} MW",
            delta=f"{wsc['turbines']:,} turbines", status="good",
        ), unsafe_allow_html=True)
    with wc2:
        st.markdown(metric_card(
            "Annual Energy", f"{wsc['annual_gwh']:,} GWh",
            delta=f"40% capacity factor", status="good",
        ), unsafe_allow_html=True)
    with wc3:
        st.markdown(metric_card(
            "Total Impact", f"{wsc['total_impact_km2']:,} km2",
            delta=f"{wsc['eez_pct']:.2f}% of EEZ",
            delta_good=wsc['eez_pct'] < 2, status="warning",
        ), unsafe_allow_html=True)

    # Radar chart for selected scenario
    categories = ["Fishing", "Shipping", "Habitat", "Visual", "Noise"]
    amb = WIND_SCENARIOS["Ambitious (41.7 GW)"]
    max_vals = [amb["fishing_km2"], max(amb["shipping_km2"], 1),
                amb["habitat_km2"], amb["visual_km2"], amb["noise_km2"]]
    vals = [wsc["fishing_km2"] / max_vals[0],
            wsc["shipping_km2"] / max(max_vals[1], 1),
            wsc["habitat_km2"] / max_vals[2],
            wsc["visual_km2"] / max_vals[3],
            wsc["noise_km2"] / max_vals[4]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0,180,216,0.2)",
        line=dict(color="#00b4d8", width=2),
        name=wind_choice,
    ))
    plotly_dark_layout(fig_radar, f"Impact Profile: {wind_choice}")
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#131f30",
            radialaxis=dict(visible=True, range=[0, 1.1],
                           gridcolor="#1a3a5c", tickfont=dict(color="#566573")),
            angularaxis=dict(gridcolor="#1a3a5c"),
        ),
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Combined scenario summary
    st.markdown(
        '<p class="section-header">Combined Scenario Summary</p>',
        unsafe_allow_html=True,
    )

    total_protected = current_km2 + gap_km2
    total_protected_pct = total_protected / METRICS["eez_area_km2"] * 100
    wind_area = wsc["footprint_km2"]
    remaining = METRICS["eez_area_km2"] - total_protected - wind_area - wsc["total_impact_km2"]

    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | **MPA Coverage** | {total_protected_pct:.1f}% ({total_protected:,.0f} km2) |
    | **Wind Energy Footprint** | {wind_area:,.0f} km2 ({wsc['capacity_mw']:,} MW) |
    | **Wind Impact Zone** | {wsc['total_impact_km2']:,} km2 |
    | **Remaining EEZ** | {max(remaining,0):,.0f} km2 |
    | **Annual Clean Energy** | {wsc['annual_gwh']:,} GWh |
    | **Est. MPA Mgmt Cost** | EUR {gap_km2 * 500 / 1e6:,.1f}M/year |
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#566573; font-size:12px;'>"
    "Marine Spatial Planning Dashboard | Turkish Waters 2026 | "
    "Data: GEBCO, WDPA, VLIZ EEZ v12 | Projects 1-19<br>"
    "Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
