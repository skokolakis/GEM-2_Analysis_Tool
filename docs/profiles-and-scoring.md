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

## Sidebar → Profiles → Profile interpolation & scoring

| Setting | Effect |
|---|---|
| **Frequency scoring** | Off by default. With it off the app shows each mode's channels, profiles, maps and exports without scores. AUX channels are never scored. |
| **Distance step (m)** | Spacing of the common interpolation grid (default 0.5 m). |
| **Interpolation method** | Linear or PCHIP, below. |

## Scoring

$$\text{Score} = \frac{A}{\sigma_{\text{noise}}}$$

- **A (amplitude)** = max − min of the representative (mean) profile after a running median over 1 m (at least 3 grid points), taken where at least two passes overlap (single-pass files: the whole profile). It is the total geophysical contrast resolved at that frequency. The plain max − min is an extreme-value statistic — one spike sets it, and noise pushes it up most at the noisiest frequencies, partly cancelling the penalty in σ. Real anomalies are wider than the GEM-2's footprint (coils 1.66 m apart), so the 1 m median keeps them; it trims the very top of a peak by about the change over one grid step (≈ 10 % for an anomaly 3 m wide at a 0.5 m step), the same at every frequency, so the ranking is unchanged.
- **σ_noise** depends on the number of passes:
  - **Multi-trace (≥ 2 passes)** — sample standard deviation across passes (ddof = 1) at each distance step where ≥ 2 passes were measured, pooled as √(mean variance). It captures instrument noise, positioning uncertainty and short-term drift.
  - **Single-trace fallback** — estimated from the *measured* samples (not the interpolated grid) by second differences, which cancel a locally linear trend: for white noise var(Δ²y) = 6σ², so σ ≈ 1.4826 · MAD(Δ²y) / √6. The median absolute deviation ignores the few large differences at sharp boundaries, and the estimate does not depend on the distance step or interpolation method.
  - **Lines at different positions** — the passes are grouped into transects by their position across the line direction (lines less than 1 m apart are one transect). The between-pass σ is pooled only over transects walked at least twice, and A is the median range of their mean profiles. When every transect was walked once (an area survey), σ is the single-trace estimate pooled over the lines as √(mean σ²) and A is the median range of the lines. Lines over different ground differ by the geology, not by noise, so they are never compared with each other. Files without coordinates are taken as repeat passes of one transect.
  - **Zero or undefined noise** (σ < 10⁻⁸ or not estimable) — the score is left blank and ranked last, with a warning.

**Higher score = cleaner, larger-contrast signal** → better for detailed profiling. The score is a **contrast-to-noise ratio** (range of the signal over the noise σ), related to but not the same as the RMS signal-to-noise ratio used in seismic data quality (Sheriff & Geldart, 1995; Bakulin et al., 2022).

The ranking table and exports show the **noise method** and number of **traces** behind each score. When some frequencies in a file fall back to the single-trace estimate, a warning says their scores are not strictly comparable with the others.

> **Why sample std (ddof = 1)?** The passes are a finite sample of the measurement process whose noise is being estimated, so the sample formula applies. With two passes the population formula (ddof = 0) would understate σ by a factor of √½ ≈ 0.71.

> **Filters change scores.** Despiking lowers σ and so raises the score. Compare scores only between runs with the same settings.

To go from scores to a short list of non-redundant frequencies, see [Frequency selection](interpretation.md#frequency-selection).

### Survey design: the representative incision

The score is most meaningful on a **representative incision**: a survey transect, typically 50–200 m long, that crosses the main geophysical contrasts expected across the site — for example from a high-EC wetland fringe into a low-EC sandy terrace. Repeated passes (at least two) along the same line allow assessment of instrument drift, operator-induced variability and short-term environmental noise; this replicated design is what makes the between-pass standard deviation a meaningful noise estimate.

### Why frequency matters

The **skin depth** δ = √(2 / ωμσ) is the depth at which the primary field falls to 1/e of its surface value; it shrinks as frequency and conductivity rise. Under the **low-induction-number (LIN)** approximation (McNeill, 1980), valid when the induction number B = s/δ ≪ 1 (s = coil separation), quadrature maps to apparent conductivity and in-phase to apparent susceptibility, and the depth response is set by the coil geometry, not by frequency (McNeill, 1980; Callegary et al., 2007). As B grows — high frequencies, conductive ground — the response moves towards the surface and LIN breaks down (Callegary et al., 2007; Delefortrie et al., 2014); the scoring is most reliable when LIN holds at every frequency tested. For small broadband sensors the practical depth of investigation grows roughly with √δ, so lower frequencies see somewhat deeper (Huang, 2005) — a qualitative trend, not a depth scale. What differs most between frequencies at one site is the noise (Vilhelmsen & Døssing, 2022), which is what the score measures. The [Frequency information](sensor-physics.md#tools) panel shows skin depth, induction number and depth of investigation for your data.

## Interpolation methods

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial (Fritsch & Carlson, 1980). Smooth, shape-preserving and monotone within each interval, so it never overshoots the readings. |

Columns with fewer than two points are skipped with a warning. GEM-2 readings (every 0.1–0.2 m) are usually denser than the profile grid, so the two methods give nearly the same profiles and scores. Splines that overshoot (cubic, quadratic), the nearest-neighbour step and the polynomial trend fit, which smoothed each trace and inflated its score, were removed.

## Viewing the results

Each mode tab shows:

- **Channels** — the frequencies / channels found, or the **Frequency ranking** table when scoring is on.
- **Representative profiles** (collapsed; click to open) — every frequency's mean profile on one axes (scores in the legend when scoring is on), with event markers as dotted lines.
- **Graph editor** — choose the **X axis** (distance along line, X, Y, Lat/Lon, time, reading number, or a channel for cross-plots) and the **Y axis** (the representative profiles, or any channel of the file — readings are then plotted one colour per line); toggle frequencies; set axis limits, line width and style, labels and title; show or hide individual traces and the ±1σ envelope.
- **Per-frequency detail plots** (per-channel on the AUX tab) — individual traces (thin, semi-transparent), the mean profile (bold) and the ±1σ envelope.
- **Downloads** — see [Outputs](outputs.md).

Full citations: [References](references.md).
