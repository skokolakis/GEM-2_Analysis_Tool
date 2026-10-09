# Profiles, scoring & interpolation

[← Documentation index](README.md)

The core of the app: every survey line (pass) of a channel is resampled onto a common distance grid, the passes are averaged into a **representative profile**, and — optionally — each frequency is scored by how far its signal stands above the noise.

## Processing pipeline

| Step | What happens |
|---|---|
| **1. Ingest** | The file is read (CSV or XLSX) and its format detected ([Input data](input-data.md)). For GEM files, flagged readings are dropped, the enabled [corrections & filters](corrections.md) run, and the distance of each reading along its line is found. Each channel becomes a matrix with distance as rows and survey lines as columns. |
| **2. Interpolate** | All traces are resampled onto a common, evenly spaced distance grid (`np.linspace`, spacing = **Distance step**) with the chosen [interpolation method](#interpolation-methods). |
| **3. Score** (optional) | With **Frequency scoring** on, the mean profile and noise are computed and frequencies ranked by descending score. |
| **4. Visualise** | An overview plot shows all mean profiles; per-frequency plots show the individual traces, the mean profile and the ±1σ envelope. |
| **5. Export** | Per-file and batch downloads — see [Outputs](outputs.md). |

Each trace is used only inside its own measured distance range — grid points outside it are left blank for that trace, so nothing is extrapolated. At a repeated distance within a trace only the first reading is kept (with a warning): averaging would lower that trace's noise alone.

## Sidebar → Processing

| Setting | Effect |
|---|---|
| **Frequency scoring** | Off by default. With it off the app shows each mode's channels, profiles, maps and exports without scores. AUX channels are never scored. |
| **Measurement mode** | EC or MS — applies to legacy files only; GEM files show every mode. |
| **Distance step (m)** | Spacing of the common interpolation grid (default 0.5 m). |
| **Interpolation method** | One of the seven methods below. |

## Scoring

$$\text{Score} = \frac{A}{\sigma_{\text{noise}}}$$

- **A (amplitude)** = max − min of the representative (mean) profile, taken where at least two passes overlap (single-pass files: the whole profile). It is the total geophysical contrast resolved at that frequency.
- **σ_noise** depends on the number of passes:
  - **Multi-trace (≥ 2 passes)** — sample standard deviation across passes (ddof = 1) at each distance step where ≥ 2 passes were measured, pooled as √(mean variance). It captures instrument noise, positioning uncertainty and short-term drift.
  - **Single-trace fallback** — estimated from the *measured* samples (not the interpolated grid) by second differences, which cancel a locally linear trend: for white noise var(Δ²y) = 6σ², so σ ≈ 1.4826 · MAD(Δ²y) / √6. The median absolute deviation ignores the few large differences at sharp boundaries, and the estimate does not depend on the distance step or interpolation method.
  - **Zero or undefined noise** (σ < 10⁻⁸ or not estimable) — the score is left blank and ranked last, with a warning.

**Higher score = cleaner, larger-contrast signal** → better for detailed profiling. This is the signal-to-noise ratio, the standard geophysical criterion for data quality (Sheriff & Geldart, 1995; Bakulin et al., 2022).

The ranking table and exports show the **noise method** and number of **traces** behind each score. When some frequencies in a file fall back to the single-trace estimate, a warning says their scores are not strictly comparable with the others.

> **Why sample std (ddof = 1)?** The passes are a finite sample of the measurement process whose noise is being estimated, so the sample formula applies. With two passes the population formula (ddof = 0) would understate σ by a factor of √½ ≈ 0.71.

> **Filters change scores.** Despiking, the running mean, PCA and the polynomial trend fit all lower σ and so raise the score. Compare scores only between runs with the same settings.

To go from scores to a short list of non-redundant frequencies, see [Frequency selection](interpretation.md#frequency-selection).

### Survey design: the representative incision

The score is most meaningful on a **representative incision**: a survey transect, typically 50–200 m long, that crosses the main geophysical contrasts expected across the site — for example from a high-EC wetland fringe into a low-EC sandy terrace. Repeated passes (at least two) along the same line allow assessment of instrument drift, operator-induced variability and short-term environmental noise; this replicated design is what makes the between-pass standard deviation a meaningful noise estimate.

### Why frequency matters

The **skin depth** δ = √(2 / ωμσ) is the depth at which the primary field falls to 1/e of its surface value; it shrinks as frequency and conductivity rise. Under the **low-induction-number (LIN)** approximation (McNeill, 1980), valid when the induction number B = s/δ ≪ 1 (s = coil separation), quadrature maps to apparent conductivity and in-phase to apparent susceptibility, and the depth response is set by the coil geometry, not by frequency (McNeill, 1980; Callegary et al., 2007). As B grows — high frequencies, conductive ground — the response moves towards the surface and LIN breaks down (Callegary et al., 2007; Delefortrie et al., 2014); the scoring is most reliable when LIN holds at every frequency tested. For small broadband sensors the practical depth of investigation grows roughly with √δ, so lower frequencies see somewhat deeper (Huang, 2005) — a qualitative trend, not a depth scale. What differs most between frequencies at one site is the noise (Vilhelmsen & Døssing, 2022), which is what the score measures. The [Frequency information](sensor-physics.md#tools) panel shows skin depth, induction number and depth of investigation for your data.

## Interpolation methods

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **nearest** | 1 | Assigns each grid point the value of the closest data point. Useful for step-like signals. |
| **quadratic** | 3 | Quadratic B-spline (`make_interp_spline`, k = 2). Smoother than linear with modest curvature. |
| **cubic** | 4 | Cubic spline with continuous second derivative (`CubicSpline`). Best for dense, smooth profiles; may overshoot at sharp boundaries. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial (Fritsch & Carlson, 1980). Shape-preserving and monotone within each interval — avoids the overshoot of cubic splines. Good default for near-monotone geophysical profiles. |
| **akima** | 5 | Akima (1970) local spline. Derives slopes from neighbouring points only, so it is robust to isolated outliers that would disturb a global cubic spline. |
| **polynomial** | 3 | **Trend fit, not an interpolant.** Global least-squares polynomial (degree = min(n − 1, 5)); exact only for ≤ 6 points. Smooths each trace, which lowers σ and inflates the score — do not compare its scores with the other methods. Shown as "polynomial (trend fit)" in the sidebar. |

Columns with fewer than the minimum number of points are skipped with a warning. The **Batch export — all methods** option runs all seven methods in one go for direct comparison ([Outputs](outputs.md#batch-export)).

## Viewing the results

Each mode tab shows:

- **Channels** — the frequencies / channels found, or the **Frequency ranking** table when scoring is on.
- **Representative profiles** — every frequency's mean profile on one axes (scores in the legend when scoring is on), with event markers as dotted lines.
- **Graph editor** — choose the **X axis** (distance along line, X, Y, Lat/Lon, time, reading number, or a channel for cross-plots) and the **Y axis** (the representative profiles, or any channel of the file — readings are then plotted one colour per line); toggle frequencies; set axis limits, line width and style, labels and title; show or hide individual traces and the ±1σ envelope.
- **Per-frequency detail plots** — individual traces (thin, semi-transparent), the mean profile (bold) and the ±1σ envelope.
- **Downloads** — see [Outputs](outputs.md).

Full citations: [References](references.md).
