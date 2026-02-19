# Project 4 — Fishing Vessel Density Map

Visualized fishing vessel activity across Turkish waters using AIS vessel 
tracking data modeled on known Turkish fishing grounds across the Black Sea, 
Aegean, and Mediterranean.

## What This Project Demonstrates
- Creating density/heatmap visualizations from point data
- Gaussian KDE (Kernel Density Estimation) for spatial analysis
- Professional night-mode cartographic styling
- Working with maritime AIS vessel tracking data

## Tools Used
- Python (GeoPandas, Matplotlib, SciPy)

## Data Sources
- Synthetic AIS data modeled on Turkish fishing grounds (n=5,944)
- Marine Regions World EEZ v12
- Method: Gaussian KDE | CRS: WGS84 (EPSG:4326)

## Key Observations
- Highest fishing intensity concentrated around the Sea of Marmara
- Dense fishing band along the entire Black Sea Turkish coast
- Significant activity along the Aegean coast
- Lower intensity in the Eastern Mediterranean

## Output
![Fishing Vessel Density Map](turkey_fishing_density_map.png)
