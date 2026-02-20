# Project 11 — Cumulative Human Impact Mapping

Cumulative human impact assessment for Turkish waters using the Halpern et al. 
(2008) framework, combining five marine pressure layers into a single 
composite impact score.

## What This Project Demonstrates
- Cumulative impact methodology (Halpern et al. 2008)
- Multi-layer pressure synthesis and normalization
- Complex multi-panel figure design
- City-level hotspot analysis
- Publication-quality scientific visualization

## Tools Used
- Python (NumPy, GeoPandas, Matplotlib, SciPy)

## Methodology
Based on Halpern et al. (2008) cumulative impact framework. Five pressure 
layers combined with equal weights and sum normalization to produce a 
composite impact score (0-1 scale).

## Pressure Layers
| Layer | Mean Score | Max | Std |
|-------|-----------|-----|-----|
| Climate (SST anomaly) | 0.658 | 1.000 | 0.257 |
| Fishing | 0.418 | 1.000 | 0.403 |
| Shipping | 0.353 | 1.000 | 0.365 |
| Development | 0.118 | 1.000 | 0.158 |
| Pollution | 0.105 | 1.000 | 0.179 |

## Impact Summary
| Class | Area | % of EEZ |
|-------|------|----------|
| High (>0.6) | 41,102 km² | 15.5% |
| Medium (0.3–0.6) | 113,092 km² | 42.7% |
| Low (<0.3) | 110,950 km² | 41.8% |
| Mean score | 0.365 | |
| EEZ ocean area | 265,144 km² | |

## City Hotspot Scores
| City | Mean Impact | Max Impact |
|------|-------------|------------|
| Istanbul | 0.949 | 0.992 |
| Izmir | 0.749 | 0.765 |
| Samsun | 0.739 | 0.762 |
| Mersin | 0.609 | 0.616 |
| Trabzon | 0.502 | 0.530 |
| Antalya | 0.436 | 0.439 |

## Key Findings
- Istanbul/Bosphorus is the maximum impact hotspot — all 5 pressures converge
- Climate stress is the most spatially widespread pressure (mean 0.658)
- Black Sea coastal band shows elevated impact from fishing + shipping + climate
- Central Black Sea deep basin has lowest impact — remote from all pressures
- 15.5% of Turkish EEZ classified as high cumulative impact
- Development and pollution tightly clustered around cities and river mouths

## Output
![Cumulative Impact Map](turkey_cumulative_impact.png)
