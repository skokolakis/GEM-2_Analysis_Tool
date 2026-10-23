# Troubleshooting

[← Documentation index](README.md)

Each file's warnings (expander under the file name) list every step the app took and anything it skipped — start there.

### No usable frequencies or sheets found
- Check **Distance step** in the sidebar (default 0.5 m) — a step larger than the profile skips every channel or sheet. For GEM-2 files the app shows the reading spacing and offers a step that suits it (**Use … m**).
- Make sure the first column of a legacy sheet is numeric distance.
- Check that traces have ≥ 2 valid points after blanks are removed (more for some [interpolation methods](profiles-and-scoring.md#interpolation-methods)).

### "Reduced precision detected" (CSV only)
- Expected — GEM CSV exports round EC to integers and MS to one decimal.
- Use the XLSX export for full-precision analysis; scores may differ slightly between the two. See [Input data](input-data.md#csv-vs-xlsx-precision).

### Different interpolation methods give different scores
- Methods differ in how they smooth (or preserve) variability between data points: `linear` preserves inter-point noise; `cubic` / `pchip` reduce it; `polynomial` is a trend fit and inflates scores.
- Use **Generate all-methods export** (under **Export all files**) to compare all seven on your data.
- `pchip` is a good general-purpose default; `akima` is preferred when outliers are suspected.

### Akima needs ≥ 5 points
- Columns with fewer valid points are skipped with a warning. Switch to `pchip` (≥ 2 points) or `linear` for sparse data.

### Scores rose after turning on a filter
- Despiking, the running mean and PCA lower the noise σ. Compare scores only between runs with the same [corrections & filters](corrections.md).

### Lines of a `.gbf`-converted file are wrong
- Without a `Line` column, lines are inferred from constant X / Y runs, coordinate restarts, or time gaps — the warnings say which. See [Input data](input-data.md#tables-converted-from-gbf-files).

### No area map
- The area map needs several lines with coordinates (at least three distinct line positions). Repeat passes along one transect are treated as a transect — use the pseudo-section. Kriging needs roughly 30 or more block medians. See [2D mapping](mapping.md#area-map-plan-view).

### A tool needs columns the file lacks
The summary card under each file lists the tools the file supports and, for the others, what they need. A tool that cannot run says so where it would appear.
- **Coordinates (X/Y or Lat/Lon)** — area maps, combined surveys, the GPS-lag estimate and the soil tools. Export positions from WinGEM / EMExport, or merge GPS positions into the table. See [Input data](input-data.md#gem-2-exports-recommended).
- **In-phase and quadrature (`I_…Hz`, `Q_…Hz`)** — anomaly spectrum and multi-height calibration. Export the I / Q columns as well as EC / MS.
- **EC channels** — soil tools. With I / Q columns, **Recompute EC / MS from I / Q** under Sensor geometry (advanced settings) adds them.
- **A Time column (`Time[ms]` or `Time[hhmmss.sss]`)** — base-station drift and the GPS-lag estimate.
- **Two or more frequencies** — pseudo-sections.
