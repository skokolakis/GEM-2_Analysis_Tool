# Outputs

[← Documentation index](README.md)

## Per-file downloads

Under each mode tab, **Downloads**:

| File | Content |
|---|---|
| Interpolated profiles (`.xlsx`) | One sheet per frequency: `Distance (m)` (the common interpolation grid) and `{Frequency}_mean` (the representative profile, mean across lines) |
| Scores (`.csv`) — scoring on only | One row per frequency: `mean_std` (noise σ, in the measurement's units), `amplitude`, `score` (amplitude ÷ noise; blank if σ is zero or undefined), `noise_method` (`between-trace` or `intra-profile`), `n_traces` (passes used) |
| Overview plot (`.png`) | All frequencies on one axes; scores in the legend when scoring is on |

## Batch export

Shown once, above the per-file results, when at least one file was processed.

**Selected method** — one workbook for all uploaded files with the current interpolation method:

| Sheet | Contents |
|---|---|
| `Scores` | One row per (file, mode, frequency) with Score, Amplitude, Noise (σ), Noise method, Traces — scoring on only |
| `{stem}_{mode}` | Distance column + one `{freq}_mean` column per frequency, joined on distance (a frequency is blank where it has no data) |

**All methods** — runs all seven [interpolation methods](profiles-and-scoring.md#interpolation-methods) across all files. Adds a `Method` column to `Scores` and writes one data sheet per (file, mode, method). Its `Exact interpolant` column is `False` for `polynomial`, whose scores are not comparable with the others.

## Elsewhere in the app

| Where | Files | Details |
|---|---|---|
| Area maps | PNG, grid CSV, ESRI `.asc`, GeoTIFF, `.prj` | [2D mapping → Downloads](mapping.md#downloads) |
| Layered-earth inversion | Model CSV, EMagPy survey file | [Sensor physics → Inversion](sensor-physics.md#layered-earth-inversion) |
| Multi-height calibration | I / Q offsets CSV (reload as *Calibration offsets*) | [Sensor physics](sensor-physics.md#multi-height-calibration) |
| Soil tools | Sampling sites, predicted property, management zones (CSV) | [Soil tools](soil-tools.md) |
