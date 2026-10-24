# Outputs

[← Documentation index](README.md)

## Per-file downloads

Under each mode tab, **Downloads**:

| File | Content |
|---|---|
| Interpolated profiles (`.xlsx`) | One sheet per frequency: `Distance (m)` (the common interpolation grid) and `{Frequency}_mean` (the representative profile, mean across lines) |
| Scores (`.csv`) — scoring on only | One row per frequency: `mean_std` (noise σ, in the measurement's units), `signal_std` (signal σ = √P_s), `snr` (P_s / P_n), `score` (SNR in dB; blank if σ is zero or undefined or no signal rises above the noise), `noise_method` (`between-trace` or `intra-profile`), `n_traces` (passes used) |
| Overview plot (`.png`) | All frequencies on one axes; scores in the legend when scoring is on |

## Batch export

Under **Export all files**, near the end of the page, when at least one file was processed.

One workbook for all uploaded files with the current interpolation method:

| Sheet | Contents |
|---|---|
| `Scores` | One row per (file, mode, frequency) with SNR (dB), Signal (σ), Noise (σ), Noise method, Traces — scoring on only |
| `{stem}_{mode}` | Distance column + one `{freq}_mean` column per frequency, joined on distance (a frequency is blank where it has no data) |

## Elsewhere in the app

| Where | Files | Details |
|---|---|---|
| Area maps | PNG, grid CSV, ESRI `.asc`, GeoTIFF, `.prj` | [2D mapping → Downloads](mapping.md#downloads) |
| Layered-earth inversion | Model CSV, EMagPy survey file | [Sensor physics → Inversion](sensor-physics.md#layered-earth-inversion) |
| Multi-height calibration | I / Q offsets CSV (reload as *Calibration offsets*) | [Sensor physics](sensor-physics.md#multi-height-calibration) |
| Soil tools | Sampling sites, predicted property, management zones (CSV) | [Soil tools](soil-tools.md) |
