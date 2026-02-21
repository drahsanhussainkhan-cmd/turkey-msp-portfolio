# Project 20 — Interactive MSP Dashboard (Capstone)

Interactive Streamlit dashboard synthesizing all findings from the 
Turkey Marine Spatial Planning portfolio (Projects 1-19) into a 
single navigable web application.

## What This Project Demonstrates
- Full-stack data application development
- Interactive data visualization with Plotly
- Multi-page dashboard design
- Translating 19 spatial analyses into accessible policy interface
- This is the kind of tool decision-makers actually use

## Tools Used
- Python (Streamlit, Plotly, Pandas)

## How to Run
```bash
cd "project-20-dashboard"
pip install -r requirements.txt
python -m streamlit run msp_dashboard.py
```
Dashboard opens at: http://localhost:8501

## Dashboard Pages
| Page | Content |
|------|---------|
| Overview | 8 KPI metric cards + integrated MSP map + zone donut chart |
| Conservation | MPA coverage, CBD 30x30 gap chart, BACI effectiveness maps |
| Energy | Wind/tidal metrics, scenario dropdown, impact comparison charts |
| Conflicts | Conflict area cards, log-scale bar chart, shipping maps |
| Cumulative Impact | City impact scores, pressure maps |
| Coastal Risk | Delta erosion scores, habitat suitability maps |
| Scenario Planner | MPA target slider + wind scenario radar chart |

## Interactive Features
- MPA coverage target slider (0.1% to 30%) with live gap calculation
- Wind scenario dropdown (Conservative/Moderate/Ambitious)
- Plotly charts with hover, zoom, and pan
- 12 map images from Projects 5-19 embedded across pages
- Color-coded metric cards (critical/warning/good/info)

## Key Metrics Displayed
| Metric | Value |
|--------|-------|
| EEZ area | 243,783 km² |
| MPA coverage | 0.10% |
| CBD 30x30 gap | 72,757 km² |
| Offshore wind potential | 41.7 GW |
| Tidal energy potential | 750 MW |
| High cumulative impact | 15.5% of EEZ |
| Shipping/fishing conflict | 5,829 km² |
| Coastal erosion high risk | 12.1% |
| Blue carbon value | EUR 895M |
| Anchovy high habitat | 36,621 km² |

## Files
- `msp_dashboard.py` — Main dashboard application (893 lines)
- `requirements.txt` — Python dependencies

## Output
Run the dashboard locally to explore all 20 projects interactively.
