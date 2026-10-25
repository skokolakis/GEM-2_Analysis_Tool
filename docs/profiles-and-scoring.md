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
| **Interpolation method** | Linear, PCHIP, nearest or kriging, below. With scoring on, kriging shows a warning that its scores are higher. |

## Scoring

Each frequency is scored by its **signal-to-noise ratio** — the ratio of signal power to noise power (Sheriff & Geldart, 1995), in decibels as is usual for data-quality assessment (Bakulin et al., 2022):

$$\mathrm{SNR_{dB}} = 10\,\log_{10}\frac{P_s}{P_n}, \qquad P_n = \sigma_{\text{noise}}^2, \qquad P_s = \operatorname{var}(\bar p) - \sigma_{\text{noise}}^2\left\langle \tfrac{1}{k} \right\rangle$$

- **p̄** is the representative (mean) profile, where at least two passes overlap (single-pass files: the measured readings), and **k** the number of passes averaged at each distance.
- **P_s (signal power)** is the variance of p̄ about its mean less the noise still in it. Averaging k passes leaves σ²/k of noise at each distance, so var(p̄) = P_s + σ²⟨1/k⟩ on average; subtracting that term makes P_s unbiased, so a noisier frequency does not gain signal from its own noise (a single pass: k = 1, P_s = var(y) − σ²). The signal is the variation of the profile about its mean — the contrasts the frequency resolves; a constant level carries no information about the ground.
- **P_n (noise power)** is σ²_noise, with σ estimated as follows:
  - **Multi-trace (≥ 2 passes)** — sample standard deviation across passes (ddof = 1) at each distance step where ≥ 2 passes were measured, pooled as √(mean variance). It captures instrument noise, positioning uncertainty and short-term drift.
  - **Single-trace fallback** — estimated from the *measured* samples (not the interpolated grid) by second differences, which cancel a locally linear trend: for white noise var(Δ²y) = 6σ², so σ ≈ 1.4826 · MAD(Δ²y) / √6. The median absolute deviation ignores the few large differences at sharp boundaries, and the estimate does not depend on the distance step or interpolation method.
  - **Lines at different positions** — the passes are grouped into transects by their position across the line direction (lines less than 1 m apart are one transect). The between-pass σ is pooled only over transects walked at least twice, and P_s over their mean profiles (weighted by length). When every transect was walked once (an area survey), σ is the single-trace estimate pooled over the lines as √(mean σ²) and P_s is pooled over the lines. Lines over different ground differ by the geology, not by noise, so they are never compared with each other. Files without coordinates are taken as repeat passes of one transect.
  - **Zero or undefined noise** (σ < 10⁻⁸ or not estimable) — the score is left blank and ranked last, with a warning.
- **No signal above the noise** (P_s ≤ 0) — the score is left blank and ranked last, with a warning.

**Higher SNR = more of the measured variation is ground, less is noise** → better for detailed profiling. 0 dB means equal signal and noise power; every 10 dB is a tenfold ratio. The ranking and exports show the SNR in dB with the signal σ (√P_s) and the noise σ in the data's units.

The ranking table and exports show the **noise method** and number of **traces** behind each score. When some frequencies in a file fall back to the single-trace estimate, a warning says their scores are not strictly comparable with the others.

> **Why sample std (ddof = 1)?** The passes are a finite sample of the measurement process whose noise is being estimated, so the sample formula applies. With two passes the population formula (ddof = 0) would understate σ by a factor of √½ ≈ 0.71.

> **Filters change scores.** Despiking and kriging lower σ and so raise the score. Compare scores only between runs with the same settings.

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
| **nearest** | 2 | Value of the closest reading (`scipy.interpolate.interp1d`, `kind="nearest"`). Nothing is averaged, so σ is that of the readings themselves — the most conservative score. The profile is stepped, and each value sits up to half the reading spacing from where it was measured. |
| **kriging** | 2 | 1D ordinary kriging along distance. Each grid point is estimated from the 16 readings around it (a moving neighbourhood, so long profiles stay fast), with a gaussian variogram fitted by the same weighted least squares as the area maps (Cressie, 1985). One variogram is fitted per channel, from pairs of readings within each pass (never across passes), so every pass is smoothed alike and a short pass borrows the structure of the others; the fit uses lags up to twice the neighbourhood width, the lags the estimates use. The gaussian model is parabolic near the origin, like an EMI signal averaged over the sensor footprint; the spherical and exponential models are linear there and, on steep profiles, took the noise for slope and fitted no nugget. The nugget is treated as measurement noise and filtered out (Cressie, 1993): the profile is continuous and the smoothest of the four, but it no longer passes through the readings. |

Passes with fewer than two readings are skipped with a warning; kriging skips a channel whose variogram cannot be fitted (it needs a few dozen readings over all its passes). GEM-2 readings (every 0.1–0.2 m) are usually denser than the profile grid, so linear, PCHIP and nearest give nearly the same profiles and scores. Kriging is different in kind: by filtering the nugget it removes much of the noise from each pass, so the between-pass σ falls and the score rises — on a synthetic profile with σ = 0.5 it measured σ ≈ 0.13, about 10 dB more. Its mean profile is closer to the true signal (there, a third of the linear profile's error), so it suits displaying and exporting smooth profiles; compare its scores only with other kriging runs. Splines that overshoot (cubic, quadratic) and the polynomial trend fit were removed.

## Viewing the results

Each mode tab shows:

- **Channels** — the frequencies / channels found, or the **Frequency ranking** table when scoring is on.
- **Representative profiles** (collapsed; click to open) — every frequency's mean profile on one axes (scores in the legend when scoring is on), with event markers as dotted lines.
- **Graph editor** — choose the **X axis** (distance along line, X, Y, Lat/Lon, time, reading number, or a channel for cross-plots) and the **Y axis** (the representative profiles, or any channel of the file — readings are then plotted one colour per line); toggle frequencies; set axis limits, line width and style, labels and title; show or hide individual traces and the ±1σ envelope.
- **Per-frequency detail plots** (per-channel on the AUX tab) — individual traces (thin, semi-transparent), the mean profile (bold) and the ±1σ envelope.
- **Downloads** — see [Outputs](outputs.md).

Full citations: [References](references.md).
