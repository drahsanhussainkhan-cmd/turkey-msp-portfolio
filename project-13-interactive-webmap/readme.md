# Project 13 — Interactive MSP Web Map

A fully interactive web map synthesizing all key findings from the Turkey 
Marine Spatial Planning portfolio into a single explorable interface.

## What This Project Demonstrates
- Interactive web mapping with Folium
- Multi-layer GIS data integration
- User interface design for non-technical stakeholders
- Synthesizing multiple analyses into one deliverable
- This is the kind of product NGOs and government clients actually use

## Tools Used
- Python (Folium, GeoPandas, Pandas)

## How to View
Open `turkey_msp_interactive.html` in any web browser. No installation needed.

## Map Layers
| # | Layer | Source | Default |
|---|-------|--------|---------|
| 1 | Turkey EEZ Boundary | Marine Regions v12 | Visible |
| 2 | Marine Protected Areas | WDPA Feb 2026 | Visible |
| 3 | Offshore Wind Zones | Project 6 | Visible |
| 4 | Shipping Routes | Project 10 | Visible |
| 5 | Fishing Activity Heatmap | Project 5 | Visible |
| 6 | Fishing Ground Centers | Project 5 | Hidden |
| 7 | Coastal Erosion Hotspots | Project 8 | Visible |
| 8 | Cumulative Impact Cities | Project 11 | Visible |
| 9 | Tidal Energy Sites | Project 12 | Visible |
| 10 | MPA Gap Analysis Zones | Project 9 | Hidden |

## Features
- 3 basemaps: CartoDB Dark (default), OpenStreetMap, CartoDB Light
- Layer toggle control panel
- Fullscreen mode
- Minimap overview
- Measure tool
- Mouse coordinates display
- Scale bar
- Legend panel
- Key findings summary overlay
- MPA popups with name, designation, IUCN category, and area

## Key Findings Summarized
- Turkey EEZ: 243,313 km²
- MPA coverage: 0.10% (300x below CBD 30x30 target)
- Offshore wind potential: 41.7 GW
- Tidal energy potential: 750 MW
- High cumulative impact: 15.5% of EEZ
- Shipping/fishing conflict: 5,829 km²
- Coastal erosion hotspot: Kizilirmak Delta (risk score 1.0)

## Output
Open [turkey_msp_interactive.html](turkey_msp_interactive.html) in your browser.
