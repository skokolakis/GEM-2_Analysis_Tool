# Representative Incision Tool — GEM Multi-Frequency Analysis

## Overview

**RIs_v2.py** is a Streamlit web application for analyzing multi-frequency electromagnetic induction (EMI) survey data. It automatically identifies the **most representative frequencies** (EC and MS channels) for geophysical profiling by scoring them based on signal-to-noise ratio.
The tool can be also be used online via https://gemris.streamlit.app

### Key Features

- **Dual-format support**: Upload GEM instrument data as `.csv` or `.xlsx` (full precision), or legacy multi-sheet XLSX
- **Intelligent ranking**: Frequencies ranked by signal-to-noise score (amplitude ÷ noise)
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
- Dependencies: `streamlit` (≥ 1.26), `pandas`, `numpy`, `scipy` (≥ 1.10), `matplotlib` (≥ 3.5), `openpyxl`, `pykrige` (≥ 1.7)
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
- Single file (CSV or XLSX) with all frequencies and lines in one table
- Required columns: `Line`, `Y`, and one or more frequency columns matching the patterns:
  - EC: `EC{freq}Hz[mS/m]`
  - MS: `MSusc{freq}Hz[1/1000]`
- Automatically detects and shows both **EC** and **MS** results in separate tabs

**Legacy format** (multi-sheet XLSX):
- One frequency per Excel sheet
- Column 0 = distance (metres), Columns 1+ = one per survey line/trace
- Select measurement mode (EC or MS) from sidebar

### Understanding the Score

$$\text{Score} = \frac{A}{\sigma_{\text{noise}}}$$

- **A (Amplitude)** = max − min of the representative (mean) profile across all passes
- **σ_noise** depends on the number of passes:
  - **Multi-trace (≥ 2 passes)** — population standard deviation across traces at each distance step (ddof = 0); captures instrument noise, positioning uncertainty, and short-term drift
  - **Single-trace fallback** — residual std after subtracting a rolling-mean smoother (window = max(5, N/10)); decomposes the profile into a geological trend and high-frequency noise component
  - **Near-zero noise** — if σ < 10⁻⁸, score collapses to raw amplitude A to avoid numerical instability

**Higher score = cleaner, larger-contrast signal** → better for detailed profiling.

> **Why population std (ddof = 0)?** The survey passes represent the complete set of measurements — not a sample from a larger population — so the population formula is statistically appropriate.

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
- Between-trace std = NaN → fallback to intra-profile SNR via rolling-window residuals
- Ranking still works correctly; high score = large amplitude with low residual noise
- Single-trace scores are less reliable than multi-pass results — multiple passes are always preferable

---

## Interpolation Methods

All methods create a common distance grid using `np.linspace` and interpolate each trace onto it. Duplicate distance values are averaged before interpolation.

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **nearest** | 1 | Assigns each grid point the value of the closest data point. Useful for step-like signals. |
| **quadratic** | 3 | Quadratic B-spline (`make_interp_spline`, k = 2). Smoother than linear with modest curvature. |
| **cubic** | 4 | Cubic spline with continuous second derivative (`CubicSpline`). Best for dense, smooth profiles; may overshoot at sharp boundaries. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial. Shape-preserving and monotone within each interval — avoids the overshoot of cubic splines. Good default for near-monotone geophysical profiles. |
| **akima** | 5 | Akima (1970) local spline. Derives slopes from neighbouring points only, making it robust to isolated outliers that would disturb a global cubic spline. |
| **polynomial** | 3 | Global least-squares polynomial fit (degree = min(n − 1, 5)). Suitable for very smooth, low-point-count profiles; avoid for long profiles where Runge oscillations can appear. |

The **Batch Export — all methods** option runs all seven methods in one step and writes a single XLSX whose `Scores` sheet lists every (file, mode, frequency, method) combination side-by-side for direct comparison.

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
   | Ordinary kriging | Widely used for mapping apparent electrical conductivity (Corwin & Lesch, 2005); for regression / cokriging alternatives that use calibration samples see Lesch et al. (1995). The single omnidirectional variogram (spherical / exponential / gaussian) is fitted to log-spaced lag classes up to half the maximum distance with Cressie (1985) weights, so the nugget reflects measurement noise (Oliver & Webster, 2014); a small nugget floor keeps the gaussian model numerically stable. Kriging uses the 64 nearest points. A second panel maps the kriging standard deviation. Uses at most 4,000 block medians: with the automatic cell size the cell grows until that holds; with a manual cell size, choose a larger cell if the limit is hit. |
   | Linear | Delaunay triangulation; blank outside the data hull. |

5. **Blanking** — grid nodes farther than the blanking distance from any block median are left blank. The default keeps the gaps between survey lines filled: the larger of 2 × median point spacing and 1.5 × the 90th-percentile distance from grid nodes inside the survey to the nearest data point.
6. **Cross-validation** — 5-fold RMSE / MAE on the block medians, to compare methods on your data (Li & Heap, 2011). Folds are random, so along densely sampled lines the errors are optimistic; use them to rank methods rather than as absolute accuracy.

Contour colours span the 2nd–98th percentile of the gridded values; values beyond that range are shown in the end colours.

**Downloads:** PNG; grid CSV (`x, y, value` [, `variance`] [, `lon, lat`]); ESRI ASCII grid (`.asc`) for QGIS / ArcGIS. With degree input the `.asc` is in local metres and is not georeferenced — use the CSV `lon`/`lat` columns.

---

## Output Files

### Per-file downloads

**Interpolated profiles (`.xlsx`)**
One sheet per frequency with columns:
- `Distance (m)` — common interpolation grid
- `{Frequency}_mean` — representative (mean across lines) profile

**Scores (`.csv`)**
One row per frequency:
- `mean_std` — noise level (σ) in same units as measurement
- `amplitude` — dynamic range of the profile
- `score` — final ranking metric (amplitude ÷ noise)

**Overview plot (`.png`)**
All frequencies on a single axes, legend showing score per frequency.

### Batch export (`.xlsx`)

A single workbook combining all uploaded files:

| Sheet | Contents |
|---|---|
| `Scores` | One row per (file, mode, frequency) with Score, Amplitude, Noise (σ) |
| `{stem}_{mode}` | Distance column + one `{freq}_mean` column per frequency |

The **all-methods** batch export adds a `Method` column to the `Scores` sheet and creates separate data sheets per (file, mode, method) combination.

---

## Architecture

```
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

---

## License

See `LICENSE` file for terms.

---

## Citation

See `CITATION.cff` file.
