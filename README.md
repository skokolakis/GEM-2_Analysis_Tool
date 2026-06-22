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

---

## Installation

### Requirements
- Python 3.9+
- Dependencies: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`

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

---

## License

See `LICENSE` file for terms.

---

## Citation

See `CITATION.cff` file.
