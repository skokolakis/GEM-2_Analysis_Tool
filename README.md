# Representative Incision Tool — GEM Multi-Frequency Analysis

## Overview

**RIs_v2.py** is a Streamlit web application for analyzing multi-frequency electromagnetic induction (EMI) survey data. It automatically identifies the **most representative frequencies** (EC and MS channels) for geophysical profiling by scoring them based on signal-to-noise ratio.
The tool can be also be used online via https://gemris.streamlit.app

### Key Features

- **Dual-format support**: Upload GEM instrument data as `.csv` or `.xlsx` (full precision), or legacy multi-sheet XLSX
- **Every GEM-2 channel**: apparent conductivity (EC), susceptibility (MS), raw in-phase (I) and quadrature (Q) in ppm, and the power-line noise, quadrature-sum and total-EC channels
- **Quality flags and distance options**: readings flagged in the `Status` column are dropped; distance along each line from the `Y` column, coordinates, reading number or event markers
- **Optional ranking** (sidebar toggle, off by default): frequencies ranked by signal-to-noise score (amplitude ÷ noise)
- **Interactive graph editor**: Customize axis limits, line styles, titles, and visibility
- **Multi-format downloads**: Export interpolated profiles (XLSX), scores (CSV), and plots (PNG)
- **Per-frequency detail plots**: Individual traces with mean ± 1σ envelope
- **Seven interpolation methods**: Linear, cubic, nearest, quadratic, PCHIP, Akima, and polynomial
- **Batch export**: Single XLSX packaging all files and modes; or run all 7 methods simultaneously for direct comparison
- **Data quality warnings**: Auto-detects and warns about precision loss in CSV exports
- **2D contouring** (optional): plan-view area maps (thin-plate spline, ordinary kriging or linear gridding) and distance × frequency pseudo-sections

---

## Installation

### Requirements
- Python 3.9+
- Dependencies: `streamlit` (≥ 1.26), `pandas`, `numpy`, `scipy` (≥ 1.10), `matplotlib` (≥ 3.5), `openpyxl`, `pykrige` (≥ 1.7), `pyproj` (≥ 3.3)
- Tests: `pip install -r requirements-dev.txt`, then `python -m pytest`

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd GEM_representative_insicision

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run RIs_v2.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Usage

### Uploading Data

**GEM instrument format** (recommended):
- Single file (CSV or XLSX) with all frequencies and lines in one table, as written by WinGEM / EMExport
- Required columns: `Line`, `Y`, and one or more channel columns matching the patterns:
  - EC: `EC{freq}Hz[mS/m]`
  - MS: `MSusc{freq}Hz[1/1000]`
  - In-phase / quadrature: `I_{freq}Hz`, `Q_{freq}Hz` (ppm)
- Optional columns used when present: `X`/`Y` or `Lat`/`Lon`, `Sample`, `Mark` (event markers), `Status` (quality flag), `PowerLn` (power-line noise, mG), `QSum`, `TotalEC[mS/m]`
- Each mode found in the file gets its own tab: **EC**, **MS**, **I**, **Q** and **AUX** (power-line noise, quadrature sum, total EC)

### GEM data options (sidebar)

- **Drop readings with a Status flag** (on by default) — the GEM-2 writes a non-zero `Status` when a reading has a problem such as ADC overload (GEM-2 Manual v3.8).
- **Distance along line**:

  | Option | Distance of each reading |
  |---|---|
  | Y column | The exported `Y` value (WinGEM grid surveys) |
  | Projection on the survey axis | Coordinates projected on the main axis of the whole survey — repeat passes walked in either direction share distances |
  | Path length along each line | Cumulative distance from each line's first reading |
  | Reading number × spacing | Reading order within the line × the spacing you enter |
  | Between event markers | Markers placed every *spacing* metres, readings spaced evenly between them (dead reckoning); readings outside the first and last marker are not used |

- **Event markers** are drawn as dotted vertical lines on every profile.

### Corrections & filters (sidebar)

Applied to the EC, MS, I and Q channels of GEM files, before profiles and maps, in this order. Every step that runs is listed with the file's warnings.

| Step | What it does | Needs |
|---|---|---|
| Despike | Blanks readings that differ from their running median by more than *threshold* × the robust noise σ (1.4826 · MAD of second differences / √6). Readings within half a window of a line end are not tested | — |
| Clip to percentiles | Blanks values outside the chosen percentile range of each channel | — |
| Sensor-height correction | Fits *a + b·exp(c·h)* of each channel against the height column (Vilhelmsen & Døssing, 2022), falling back to a straight line, and moves every reading to the reference height. Assumes geology is not correlated with height | Height column (e.g. drone altitude above ground) |
| Temperature drift | Linear coefficient per channel from the base-station occupations, subtracted relative to their mean temperature | Temperature column and ≥ 3 base-station lines |
| Base-station drift | Mean of each base-station occupation, trend in time (piecewise linear or straight line), subtracted relative to the first occupation; base-station lines are then removed (USGS GEM-2 practice) | Time column and ≥ 2 base-station lines |
| Background EC | Shifts each EC channel so its median equals a known background value (GEM-2 Manual) | — |
| Reference calibration | Gain and offset per channel from reference values (e.g. apparent conductivity predicted from ERT, Lavoué et al., 2010; Mester et al., 2011, or TDR, Dragonetti et al., 2018): least-squares line, or matched mean and standard deviation | CSV with the survey's coordinate columns and one column per channel |
| EC at 25 °C | EC × (0.4470 + 1.4034·e^(−T/26.815)) (Sheets & Hendrickx, 1995; Corwin & Lesch, 2005) | Soil temperature |
| PCA noise reduction | Keeps the first *k* principal components of the standardised channels of each mode (Minsley et al., 2010) | — |
| Running mean | Centred moving average along each line | — |
| GPS lag | Each reading takes the track position *lag* seconds earlier. **Estimate GPS lag** (under each file) picks the lag that minimises differences between neighbouring lines (González Jiménez et al., 2022) | Time column, coordinates |
| Heading filter | Keeps readings walked within ± tolerance of a heading — e.g. to check for heading error on zig-zag surveys | Coordinates |

> Despiking, the running mean and PCA lower the noise σ, so with scoring on the scores rise; the sidebar warns when they are combined.

---

## Sensor physics

The physics tools use a layered-earth forward model of the GEM-2 (`emphysics.py`): vertical magnetic dipoles (horizontal co-planar coils) over horizontal layers, quasi-static, with the bucking coil subtracted (Won et al., 1996; Ward & Hohmann, 1988). The Hankel integral is evaluated by Gauss–Legendre quadrature with its large-wavenumber limit integrated analytically; tests check it against a brute-force integral, McNeill's (1980) low-induction-number limit and the analytic static response of a susceptible half-space.

**Sensor geometry** (sidebar): Tx–Rx separation (1.66 m), Tx–bucking-coil distance (1.035 m) and sensor height. Check the coil distances against your sensor's `.gem` configuration.

| Tool | Where | What it does |
|---|---|---|
| Recompute EC / MS from I / Q | Sensor geometry → checkbox | For every frequency with `I_` and `Q_` columns, finds the homogeneous half-space (conductivity and susceptibility) that reproduces each reading (Huang & Won, 2000), for your geometry and height. Overwrites exported EC / MS; adds them to I/Q-only exports. Readings with quadrature ≤ 0 are left blank |
| Frequency information | EC tab | Skin depth, induction number and depth of investigation (depth above which 70 % of the quadrature response originates, from the forward model) for the median EC of each frequency, with cumulative-sensitivity curves |
| Forward model | Main page | Background and target layered models → in-phase, quadrature and apparent EC per frequency; a difference over 3 × the noise level is marked detectable. Works without an upload — use it to choose frequencies before a survey |
| Multi-height calibration | Under each GEM file | Readings over one spot at several heights (own line(s), `Height` column) → half-space plus one additive offset per frequency and component (after Minsley et al., 2014). Download the offsets and load them as **Calibration offsets** in the sidebar |
| Lines to leave out | GEM data | Removes calibration or test lines from profiles and maps |

> Above about 40 kHz the in-phase response can include a dielectric-permittivity contribution (Benech et al., 2016); the MS and I tabs say so when such frequencies are present.

### Layered-earth inversion and export

Under each GEM file, **Layered-earth inversion** inverts the quadrature of the mean profiles for a smooth layered conductivity model at stations along the line:

- Fixed layers growing geometrically from the first thickness to the depth of the half-space; parameters are log₁₀ conductivity.
- Gauss–Newton with vertical smoothing (Occam-style, Constable et al., 1987) and, between neighbouring stations, lateral smoothing (laterally constrained inversion, Auken & Christiansen, 2004). Lateral constraint 0 gives independent 1D inversions.
- The regularisation weight starts at the chosen value and is halved each iteration until χ² per datum reaches 1.
- Data errors: the larger of *relative error × |data| + floor* and, optionally, the **measured between-pass noise** of each frequency — the same σ the scoring uses (converted to ppm for EC data).
- EC-only files are converted to quadrature with the half-space model, which undoes a half-space conversion like the one in WinGEM.
- Downloads: the model as CSV (distance, depth from / to, EC) and an **EMagPy survey file** (McLachlan et al., 2021): coordinates in metres, one column per frequency named `HCP{separation}f{frequency}h{height}` in mS/m, with `_err` columns from the between-pass noise. EMagPy models a plain loop pair, so the GEM-2 bucking coil is not modelled there.

> Calibrate first (multi-height offsets, reference calibration): offsets in I/Q bias inverted models (Minsley et al., 2014).

**Legacy format** (multi-sheet XLSX):
- One frequency per Excel sheet
- Column 0 = distance (metres), Columns 1+ = one per survey line/trace
- Select measurement mode (EC or MS) from sidebar

### Understanding the Score

Scoring is optional: switch on **Frequency scoring** at the top of the sidebar. With it off (the default) the app shows each mode's channels, profiles, maps and exports without scores. AUX channels are never scored.

$$\text{Score} = \frac{A}{\sigma_{\text{noise}}}$$

- **A (Amplitude)** = max − min of the representative (mean) profile, taken where at least two passes overlap (single-pass files: the whole profile)
- **σ_noise** depends on the number of passes:
  - **Multi-trace (≥ 2 passes)** — sample standard deviation across passes (ddof = 1) at each distance step where ≥ 2 passes were measured, pooled as √(mean variance); captures instrument noise, positioning uncertainty, and short-term drift
  - **Single-trace fallback** — estimated from the *measured* samples (not the interpolated grid) by second differences, which cancel a locally linear trend: σ ≈ 1.4826 · MAD(Δ²y) / √6. It does not depend on the distance step or interpolation method
  - **Zero or undefined noise** — the score is left blank and ranked last, with a warning
- Each trace is only used inside its own measured distance range — nothing is extrapolated

The ranking table and exports show the **noise method** and number of **traces** behind each score. When some frequencies in a file fall back to the single-trace estimate, a warning says their scores are not strictly comparable with the others.

**Higher score = cleaner, larger-contrast signal** → better for detailed profiling.

> **Why sample std (ddof = 1)?** The passes are a finite sample of the measurement process whose noise is being estimated, so the sample formula applies. With two passes the population formula (ddof = 0) would understate σ by a factor of √½ ≈ 0.71.

### Customization

1. **Sidebar settings**:
   - Measurement mode (EC / MS) — applies to legacy files; GEM files show both automatically
   - Interpolation method (7 options — see table below)
   - Distance step (m) — controls interpolation grid density

2. **Graph editor** (per mode, per file):
   - Toggle visibility of individual frequencies
   - Set axis limits (auto or manual)
   - Adjust line width and line style
   - Edit axis labels and plot title
   - Show/hide individual traces and ±1σ envelope

3. **Downloads** (per file):
   - **Interpolated profiles** — XLSX with distance column + mean for each frequency
   - **Scores** — CSV with amplitude, noise, and score for each frequency
   - **Overview plot** — PNG showing all frequencies on one axes

4. **Batch export** (all files combined):
   - **Selected method** — single XLSX with all files, modes, and frequencies for the currently chosen interpolation method
   - **All methods** — runs all 7 interpolation methods across all uploaded files and exports every result side-by-side for direct comparison; includes a `Scores` summary sheet

---

## Data Quality Notes

### CSV vs XLSX Precision

The GEM instrument exports different precision levels:

| Format | EC precision | MS precision | Note |
|---|---|---|---|
| `.csv` | Integer (no decimals) | 1 d.p. | **Reduced precision** — scores may differ slightly |
| `.xlsx` | 3+ d.p. | 4+ d.p. | **Full instrument precision** — recommended for analysis |

The app **automatically detects and warns** when a CSV has reduced precision. For quantitative frequency comparison, always use the XLSX file.

### Single-Trace Files

Files with only one line (one trace per frequency) still produce valid scores:
- Between-trace std is undefined → fallback to the second-difference noise estimate on the measured samples
- Ranking still works correctly; high score = large amplitude with low point-to-point noise
- Single-trace scores are less reliable than multi-pass results — multiple passes are always preferable

---

## Interpolation Methods

All methods create a common distance grid using `np.linspace` and interpolate each trace onto it; grid points outside a trace's own measured range are left blank for that trace. At a repeated distance within a trace only the first reading is kept (with a warning) — averaging would lower that trace's noise alone.

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **nearest** | 1 | Assigns each grid point the value of the closest data point. Useful for step-like signals. |
| **quadratic** | 3 | Quadratic B-spline (`make_interp_spline`, k = 2). Smoother than linear with modest curvature. |
| **cubic** | 4 | Cubic spline with continuous second derivative (`CubicSpline`). Best for dense, smooth profiles; may overshoot at sharp boundaries. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial. Shape-preserving and monotone within each interval — avoids the overshoot of cubic splines. Good default for near-monotone geophysical profiles. |
| **akima** | 5 | Akima (1970) local spline. Derives slopes from neighbouring points only, making it robust to isolated outliers that would disturb a global cubic spline. |
| **polynomial** | 3 | **Trend fit, not an interpolant.** Global least-squares polynomial (degree = min(n − 1, 5)); exact only for ≤ 6 points. Smooths each trace, which lowers σ and inflates the score — do not compare its scores with the other methods. Shown as "polynomial (trend fit)" in the sidebar. |

The **Batch Export — all methods** option runs all seven methods in one step and writes a single XLSX whose `Scores` sheet lists every (file, mode, frequency, method) combination side-by-side for direct comparison. Its `Exact interpolant` column is `False` for `polynomial`, whose scores are not comparable with the others.

---

## 2D Contouring

Two optional views, switched on in the sidebar under **2D contouring**. Both are off by default.

### Pseudo-section (distance × frequency)

Stacks the line-averaged profiles of all frequencies (the mean across survey lines at each distance — meaningful for repeat passes of one transect) into one contour plot: distance on x, frequency on y. The y-axis is logarithmic when every label is a distinct positive frequency, otherwise categorical. Profiles are aligned on the distance range they all cover — nothing is extrapolated at this step. Labels in kHz are converted to Hz. White lines mark the measured frequencies; colours between them are interpolated.

> The frequency axis is **not a calibrated depth axis**. Under the low-induction-number (LIN) approximation the depth response of a loop–loop sensor is set by coil geometry, not frequency (McNeill, 1980; Callegary et al., 2007). Analyses of small broadband sensors nevertheless find that the practical depth of investigation grows roughly with the square root of the skin depth, so lower frequencies see somewhat deeper (Huang, 2005). Read the axis as a qualitative trend at most.

### Area map (plan view)

For GEM files covering an area (several lines with X/Y or Lat/Lon coordinates), one frequency at a time. Narrow corridor surveys are accepted when they have at least three distinct line positions; repeat passes along one transect (lines < 1 m apart) are treated as a transect — use the pseudo-section for those.

1. **Coordinates** — `Lat`/`Latitude` + `Lon`/`Long`/`Longitude` columns are used as degrees when they hold at least 10 GPS fixes (rows logged as 0, 0 are treated as "no fix" and dropped); otherwise `X`/`Y`. Degrees are projected to local metres (equirectangular projection of a spherical Earth about the survey centroid; distances good to a few tenths of a percent at site scale). The sidebar override (metres / degrees) applies to `X`/`Y` only; `Lat`/`Lon` columns are always read as degrees.
2. **Line levelling** (optional) — shifts each line to the survey median to remove line-to-line offsets (striping). It also removes any real gradient across lines, so compare with levelling off. For more advanced levelling see Mauring & Kihle (2006).
3. **Block-median reduction** — one median point per grid cell so densely sampled lines do not dominate. Default cell size = √(bounding-box area / number of readings), about one node per reading.
4. **Gridding**

   | Method | Notes |
   |---|---|
   | Thin-plate spline | Minimum-curvature (biharmonic) surface (Briggs, 1974; Sandwell, 1987). Smoothing 0 interpolates exactly. |
   | Ordinary kriging | Widely used for mapping apparent electrical conductivity (Corwin & Lesch, 2005); for regression / cokriging alternatives that use calibration samples see Lesch et al. (1995). The single omnidirectional variogram (spherical / exponential / gaussian) is fitted to log-spaced lag classes up to half the maximum distance with Cressie (1985) weights, which lets the nugget pick up measurement noise where the model shape allows (spherical and exponential models, being linear near the origin, can absorb noise into their slope; check the fitted nugget shown under the map) (Oliver & Webster, 2014); a small nugget floor keeps the gaussian model numerically stable. Kriging needs roughly 30 or more block medians to fit a variogram. Kriging uses the 64 nearest points. A second panel maps the kriging standard deviation. Uses at most 4,000 block medians: with the automatic cell size the cell grows until that holds; with a manual cell size, choose a larger cell if the limit is hit. |
   | Linear | Delaunay triangulation; blank outside the data hull. |

5. **Blanking** — grid nodes farther than the blanking distance from any block median are left blank. The default keeps the gaps between survey lines filled: the larger of 2 × median point spacing and 1.5 × the 90th-percentile distance from grid nodes inside the survey to the nearest data point.
6. **Cross-validation** — 5-fold RMSE / MAE on the block medians, to compare methods on your data (Li & Heap, 2011). Folds are random, so along densely sampled lines the errors are optimistic; use them to rank methods rather than as absolute accuracy.

Contour colours span the 2nd–98th percentile of the gridded values; values beyond that range are shown in the end colours.

**Projection.** `Lat`/`Lon` are projected to local metres about the survey centre (default) or to **UTM (WGS 84)**, zone of the survey centre (sidebar *Projection of Lat/Lon*). For files whose `X`/`Y` are already projected, enter their **EPSG code** (e.g. 32634 for UTM 34N) so the exports carry it.

**Map processing** (expander under each map), applied to the gridded values:

| Option | What it does |
|---|---|
| Despike | Replaces cells that differ from their window median by more than *threshold* × 1.4826 × the median absolute difference |
| Low-pass | Moving average over *window* × *window* cells (blank cells neither spread nor bias it) |
| High-pass | Removes the regional trend: the value minus its *window* × *window* moving average |
| Deconvolve the sensor footprint | Lateral Tikhonov deconvolution of the coils' low-induction-number footprint (receiver minus bucking coil, sensor height from *Sensor geometry*; the coil axis defaults to the survey-line direction). Sharpens apparent-conductivity maps. It does not resolve depth: the 3D multichannel deconvolution of Guillemoteau et al. (2017) needs several coil geometries, while a GEM-2 has one geometry at several frequencies, and at low induction number the frequencies share the same footprint |

The footprint kernel is the dot product of the transmitter's and receiver's quasi-static electric fields (Born approximation). Tests check that its depth profile matches McNeill's (1980) HCP depth response.

**Histogram & statistics** — count, mean, standard deviation, minimum, 2nd / 50th / 98th percentile and maximum of the gridded values, with a histogram.

**Downloads:** PNG; grid CSV (`x, y, value` [, `variance`] [, `lon, lat`]); ESRI ASCII grid (`.asc`); single-band float32 **GeoTIFF** (`.tif`, no-data −9999); and, when the CRS is known (UTM or a user EPSG code), an ESRI **`.prj`** file to sit next to the `.asc`. With local-metre projection the `.asc` and GeoTIFF are not georeferenced — choose UTM, or use the CSV `lon`/`lat` columns.

### Combined surveys

With the area map on and two or more GEM files uploaded, **Combined surveys** (below the files) grids a channel the files share:

- **Merged (edge-matched)** — the surveys become one; line labels are prefixed with the survey number. With *Edge-match levels* each survey after the first is shifted by the median difference to the surveys before it, over readings closer than the edge-matching distance (as in edge matching of adjacent grids in archaeological-prospection software). The offsets are listed.
- **Difference (B − A)** — both surveys are gridded on one grid in the projection of A, and A is subtracted from B (time-lapse). Nodes blank in either survey stay blank; colours are centred on zero.

---

## Output Files

### Per-file downloads

**Interpolated profiles (`.xlsx`)**
One sheet per frequency with columns:
- `Distance (m)` — common interpolation grid
- `{Frequency}_mean` — representative (mean across lines) profile

**Scores (`.csv`)** — only with scoring on
One row per frequency:
- `mean_std` — noise level (σ) in same units as measurement
- `amplitude` — dynamic range of the profile
- `score` — final ranking metric (amplitude ÷ noise); blank if σ is zero or undefined
- `noise_method` — `between-trace` or `intra-profile`
- `n_traces` — number of passes used

**Overview plot (`.png`)**
All frequencies on a single axes, legend showing score per frequency.

### Batch export (`.xlsx`)

A single workbook combining all uploaded files:

| Sheet | Contents |
|---|---|
| `Scores` | One row per (file, mode, frequency) with Score, Amplitude, Noise (σ), Noise method, Traces — only with scoring on |
| `{stem}_{mode}` | Distance column + one `{freq}_mean` column per frequency, joined on distance (a frequency is blank where it has no data) |

The **all-methods** batch export adds a `Method` column to the `Scores` sheet and creates separate data sheets per (file, mode, method) combination.

---

## Architecture

```
gem_io.py      — GEM-2 export channels, Status flags, event markers, distance along line
pipeline.py    — PrepSettings and prepare_gem_table(): the raw table before profiles and maps
corrections.py — despike, clip, height, drift, temperature, calibration, EC25, PCA, GPS lag, heading
emphysics.py   — layered-earth forward model, I/Q ↔ EC/MS conversion, skin depth, sensitivity, multi-height fit
inversion.py   — smooth 1D / laterally constrained inversion, stations, EMagPy export
gridtools.py   — grid filters, statistics, survey merging, footprint deconvolution, UTM, GeoTIFF, .prj
ui_tools.py    — Streamlit panels for the physics and inversion tools
contouring.py  — area maps, pseudo-sections, unit labels
RIs_v2.py
├── Configuration & constants
├── GEM format detection
│   ├── is_gem_format()
│   ├── pivot_gem_frequency()
│   └── parse_gem_dataframe()
├── Data processing (cached)
│   ├── process_sheet()          — core interpolation & scoring
│   ├── process_file()           — legacy multi-sheet XLSX
│   └── process_gem_file()       — GEM CSV / XLSX
├── Interpolation helpers
│   └── _interpolate_with_method()
├── Plotting
│   ├── make_overview_figure()
│   └── make_sheet_figure()
├── Export helpers
│   ├── build_excel_download()
│   ├── build_scores_csv()
│   ├── build_batch_xlsx()
│   ├── build_all_methods_batch_xlsx()
│   └── fig_to_png()
├── UI components
│   ├── render_graph_editor()
│   └── _render_mode_section()
├── Dispatch
│   ├── render_legacy_results()
│   └── render_gem_results()
└── Streamlit app — main()
```

**No external APIs or databases** — all processing is local and deterministic.

---

## Troubleshooting

### "No usable sheets found"
- Reduce `Distance step` in sidebar (default: 0.5 m)
- Ensure first column is numeric distance
- Check that traces have ≥ 2 valid points after NaN removal

### "Reduced precision detected" warning (CSV only)
- Expected — GEM CSV exports round EC values to integers
- Use the XLSX file for full-precision analysis
- Scores may differ slightly between CSV and XLSX due to rounding

### Different interpolation methods give different scores
- Methods differ in how they smooth (or preserve) high-frequency variability between data points
- `linear` preserves inter-point noise; `cubic`/`pchip` reduce it
- Use the **all-methods batch export** to compare scores across all seven methods for your dataset
- `pchip` is a good general-purpose default; `akima` is preferred when outliers are suspected

### Akima method requires ≥ 5 points
- If fewer than 5 valid points remain after NaN removal, the column is skipped with a warning
- Switch to `pchip` (≥ 2 points) or `linear` for sparse data

---

## References

1. Won, I.J., Keiswetter, D.A., Fields, G.R.A. & Sutton, L.C. (1996). GEM-2: A new multifrequency electromagnetic sensor. *Journal of Environmental and Engineering Geophysics*, **1**(2), 129–137. https://doi.org/10.4133/JEEG1.2.129

2. McNeill, J.D. (1980). *Electromagnetic terrain conductivity measurement at low induction numbers*. Technical Note TN-6, Geonics Limited, Mississauga, Canada.

3. Callegary, J.B., Ferré, T.P.A. & Groom, R.W. (2007). Vertical spatial sensitivity and exploration depth of low-induction-number electromagnetic induction instruments. *Vadose Zone Journal*, **6**(1), 158–167. https://doi.org/10.2136/vzj2006.0120

4. Delefortrie, S., Saey, T., Van De Vijver, E., De Smedt, P., Missiaen, T., Demerre, I. & Van Meirvenne, M. (2014). Frequency domain electromagnetic induction survey in the intertidal zone: data acquisition and correction procedures. *Journal of Applied Geophysics*, **100**, 119–130. https://doi.org/10.1016/j.jappgeo.2013.10.017

5. De Smedt, P., Van Meirvenne, M., Herremans, D., De Reu, J., Saey, T., Meerschman, E., Crombé, P. & De Clercq, W. (2013). The 3-D reconstruction of medieval wetland reclamation through electromagnetic induction survey. *Scientific Reports*, **3**, 1517. https://doi.org/10.1038/srep01517

6. Reynolds, J.M. (2011). *An Introduction to Applied and Environmental Geophysics* (2nd ed.). Wiley-Blackwell.

7. Sheriff, R.E. & Geldart, L.P. (1995). *Exploration Seismology* (2nd ed.). Cambridge University Press.

8. Bakulin, A., Silvestrov, I. & Protasov, M. (2022). Signal-to-noise ratio computation for challenging land data. *Geophysical Prospecting*, **70**, 629–638. https://doi.org/10.1111/1365-2478.13183

9. Akima, H. (1970). A new method of interpolation and smooth curve fitting based on local procedures. *Journal of the ACM*, **17**(4), 589–602. https://doi.org/10.1145/321607.321609

10. Fritsch, F.N. & Carlson, R.E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, **17**(2), 238–246. https://doi.org/10.1137/0717021

11. Briggs, I.C. (1974). Machine contouring using minimum curvature. *Geophysics*, **39**(1), 39–48. https://doi.org/10.1190/1.1440410

12. Sandwell, D.T. (1987). Biharmonic spline interpolation of GEOS-3 and SEASAT altimeter data. *Geophysical Research Letters*, **14**(2), 139–142. https://doi.org/10.1029/GL014i002p00139

13. Lesch, S.M., Strauss, D.J. & Rhoades, J.D. (1995). Spatial prediction of soil salinity using electromagnetic induction techniques: 1. Statistical prediction models: A comparison of multiple linear regression and cokriging. *Water Resources Research*, **31**(2), 373–386. https://doi.org/10.1029/94WR02179

14. Corwin, D.L. & Lesch, S.M. (2005). Apparent soil electrical conductivity measurements in agriculture. *Computers and Electronics in Agriculture*, **46**, 11–43. https://doi.org/10.1016/j.compag.2004.10.005

15. Oliver, M.A. & Webster, R. (2014). A tutorial guide to geostatistics: Computing and modelling variograms and kriging. *Catena*, **113**, 56–69. https://doi.org/10.1016/j.catena.2013.09.006

16. Li, J. & Heap, A.D. (2011). A review of comparative studies of spatial interpolation methods in environmental sciences: Performance and impact factors. *Ecological Informatics*, **6**, 228–241. https://doi.org/10.1016/j.ecoinf.2010.12.003

17. Mauring, E. & Kihle, O. (2006). Leveling aerogeophysical data using a moving differential median filter. *Geophysics*, **71**(1), L5–L11. https://doi.org/10.1190/1.2163912

18. Huang, H. (2005). Depth of investigation for small broadband electromagnetic sensors. *Geophysics*, **70**(6), G135–G142. https://doi.org/10.1190/1.2122412

19. Cressie, N. (1985). Fitting variogram models by weighted least squares. *Mathematical Geology*, **17**(5), 563–586. https://doi.org/10.1007/BF01032109

20. Geophex Ltd. (2004). *GEM-2 Manual*, version 3.8. Raleigh, NC.

21. Minsley, B.J., Smith, B.D., Hammack, R., Sams, J.I. & Veloski, G. (2010). Calibration and filtering strategies for frequency domain electromagnetic data. *SAGEEP 2010*. https://doi.org/10.4133/1.3445431

22. Vilhelmsen, T.B. & Døssing, A. (2022). Drone-towed controlled-source electromagnetic (CSEM) system for near-surface geophysical prospecting. *Geoscientific Instrumentation, Methods and Data Systems*, **11**, 435–450. https://doi.org/10.5194/gi-11-435-2022

23. González Jiménez, A. et al. (2022). Correcting on-the-go field measurement–coordinate mismatch by minimizing nearest neighbor difference. *Sensors*, **22**(4), 1496. https://doi.org/10.3390/s22041496

24. Lavoué, F., van der Kruk, J., Rings, J., André, F., Moghadas, D., Huisman, J.A., Lambot, S. & Weihermüller, L. (2010). Electromagnetic induction calibration using apparent electrical conductivity modelling based on electrical resistivity tomography. *Near Surface Geophysics*, **8**(6), 553–561. https://doi.org/10.3997/1873-0604.2010037

25. Mester, A., van der Kruk, J., Zimmermann, E. & Vereecken, H. (2011). Quantitative two-layer conductivity inversion of multi-configuration electromagnetic induction measurements. *Vadose Zone Journal*, **10**(4), 1319–1330. https://doi.org/10.2136/vzj2011.0035

26. Dragonetti, G., Comegna, A., Ajeel, A., Deidda, G.P., Lamaddalena, N., Rodriguez, G., Vignoli, G. & Coppola, A. (2018). Calibrating electromagnetic induction conductivities with time-domain reflectometry measurements. *Hydrology and Earth System Sciences*, **22**, 1509–1523. https://doi.org/10.5194/hess-22-1509-2018

27. Sheets, K.R. & Hendrickx, J.M.H. (1995). Noninvasive soil water content measurement using electromagnetic induction. *Water Resources Research*, **31**(10), 2401–2409. https://doi.org/10.1029/95WR01949

28. Huang, H. & Won, I.J. (2000). Conductivity and susceptibility mapping using broadband electromagnetic sensors. *Journal of Environmental and Engineering Geophysics*, **5**(4), 31–41. https://doi.org/10.4133/JEEG5.4.31

29. Ward, S.H. & Hohmann, G.W. (1988). Electromagnetic theory for geophysical applications. In M.N. Nabighian (ed.), *Electromagnetic Methods in Applied Geophysics*, Vol. 1, 131–311. Society of Exploration Geophysicists.

30. Minsley, B.J., Kass, M.A., Hodges, G. & Smith, B.D. (2014). Multielevation calibration of frequency-domain electromagnetic data. *Geophysics*, **79**(5), E201–E216. https://doi.org/10.1190/GEO2013-0320.1

31. Benech, C., Lombard, P., Rejiba, F. & Tabbagh, A. (2016). Demonstrating the contribution of dielectric permittivity to the in-phase EMI response of soils: example from an archaeological site in Bahrain. *Near Surface Geophysics*, **14**, 337–344. https://doi.org/10.3997/1873-0604.2016023

32. Constable, S.C., Parker, R.L. & Constable, C.G. (1987). Occam's inversion: a practical algorithm for generating smooth models from electromagnetic sounding data. *Geophysics*, **52**(3), 289–300. https://doi.org/10.1190/1.1442303

33. Auken, E. & Christiansen, A.V. (2004). Layered and laterally constrained 2D inversion of resistivity data. *Geophysics*, **69**(3), 752–761. https://doi.org/10.1190/1.1759461

34. McLachlan, P., Blanchy, G. & Binley, A. (2021). EMagPy: open-source standalone software for processing, forward modeling and inversion of electromagnetic induction data. *Computers & Geosciences*, **146**, 104561. https://doi.org/10.1016/j.cageo.2020.104561

35. Guillemoteau, J., Christensen, N.B., Jacobsen, B.H. & Tronicke, J. (2017). Fast 3D multichannel deconvolution of electromagnetic induction loop-loop apparent conductivity data sets acquired at low induction numbers. *Geophysics*, **82**(6), E357–E369. https://doi.org/10.1190/geo2016-0518.1

---

## License

See `LICENSE` file for terms.

---

## Citation

See `CITATION.cff` file.
