# GEM-2 Analysis Tool — documentation

Read the pages in order for a first pass through the app, or go straight to the topic you need.

| # | Page | Read it for |
|---|---|---|
| 1 | [Getting started](getting-started.md) | Installing, running the app locally or online, the sample survey, running the tests |
| 2 | [Input data](input-data.md) | GEM-2 exports (WinGEM, `.gbf`-converted), legacy multi-sheet files, Status flags, distance along line, CSV vs XLSX precision |
| 3 | [Corrections & filters](corrections.md) | Despiking, clipping, sensor height, drift, GPS lag, calibration, EC at 25 °C, PCA, smoothing, heading filter |
| 4 | [Profiles, scoring & interpolation](profiles-and-scoring.md) | Representative profiles, the signal-to-noise score, the seven interpolation methods, the graph editor |
| 5 | [Sensor physics & inversion](sensor-physics.md) | Forward model, EC / MS from I / Q, depth of investigation, multi-height calibration, layered-earth inversion, EMagPy export |
| 6 | [Interpretation aids](interpretation.md) | Frequency selection, magnetic viscosity, anomaly spectra, power-line noise |
| 7 | [2D mapping](mapping.md) | Pseudo-sections, area maps, Surfer gridding methods, colours, projections, map filters, combined surveys |
| 8 | [Soil tools](soil-tools.md) | Sampling design, soil-property calibration, management zones |
| 9 | [Outputs](outputs.md) | Every file the app can download |
| 10 | [Architecture](architecture.md) | Module layout, tests, design notes — for developers |
| 11 | [Troubleshooting](troubleshooting.md) | Common warnings and what to do |
| 12 | [References](references.md) | Full bibliography |

Design notes written while building the 2D contouring live in [`superpowers/`](superpowers/).
