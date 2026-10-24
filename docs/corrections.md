# Corrections & filters

[← Documentation index](README.md)

Sidebar → **Corrections & filters**. The steps apply to the EC, MS, I and Q channels of GEM files, before profiles and maps are built, in the order below. Every step that runs is listed with the file's warnings. Implementation: [`corrections.py`](../corrections.py), orchestrated by [`pipeline.py`](../pipeline.py).

Despike, clipping, base-station lines and the GPS lag are always shown; the other steps, and the despike window and threshold, appear with **Show advanced settings** — see [Basic and advanced settings](getting-started.md#basic-and-advanced-settings).

| Step | What it does | Needs |
|---|---|---|
| Despike | Blanks readings that differ from their running median by more than *threshold* × the robust noise σ (1.4826 · MAD of second differences / √6). Readings within half a window of a line end are not tested | — |
| Clip to percentiles | Blanks values outside the chosen percentile range of each channel | — |
| Sensor-height correction | With **Recompute EC / MS from I / Q**, each reading is converted at its own height (in 2 cm steps), so the half-space model removes the height effect and no curve is fitted. Otherwise fits *a + b·exp(c·h)* of each channel against the height column (Vilhelmsen & Døssing, 2022), falling back to a straight line, and moves every reading to the reference height — this assumes the ground does not change with height | Height column (sensor height above ground, e.g. drone altitude) |
| Temperature drift | Linear coefficient per channel from the base-station occupations, subtracted relative to their mean temperature. Temperature and instrument drift both change through the day, so with a time column and ≥ 4 occupations the coefficient is fitted together with a time trend (the drift correction then removes the time part); with 3 it also takes up drift that follows the temperature, and a warning says so — as it does when temperature and time rise together (\|r\| > 0.9) and cannot be told apart | Temperature column and ≥ 3 base-station lines |
| Base-station drift | Mean of each base-station occupation, trend in time (piecewise linear or straight line), subtracted relative to the first occupation; base-station lines are then removed (USGS GEM-2 practice) | Time column and ≥ 2 base-station lines |
| Calibration offsets | Additive I / Q offsets per frequency, loaded from the [multi-height calibration](sensor-physics.md#multi-height-calibration) download | Offsets CSV |
| GPS lag | Each reading takes the track position *lag* seconds earlier. **Estimate GPS lag** (under each file, in Calibration) picks the lag that minimises differences between neighbouring lines (González Jiménez et al., 2022) | Time column, coordinates |
| Reference calibration | Gain and offset per channel from reference values (e.g. apparent conductivity predicted from ERT, Lavoué et al., 2010; Mester et al., 2011, or TDR, Dragonetti et al., 2018): least-squares line, or matched mean and standard deviation | CSV with the survey's coordinate columns and one column per channel |
| EC at 25 °C | EC × (0.4470 + 1.4034·e^(−T/26.815)) (Sheets & Hendrickx, 1995; Corwin & Lesch, 2005) | Soil temperature |

When **Recompute EC / MS from I / Q** is on ([Sensor physics](sensor-physics.md)), the steps up to and including the calibration offsets run on the raw readings first, EC / MS are then recomputed, and the remaining steps run on the result.

> **Effect on scores.** Despiking lowers the noise σ, so with scoring on the scores rise. The sidebar warns when they are combined — compare scores only between runs with the same filters.

The running mean, PCA noise reduction, the known-background EC shift and the heading filter were removed: the first two lowered σ and inflated the scores, a background shift duplicates the reference calibration, and leaving out lines does what the heading filter did. EC at 25 °C stays, since it is what makes surveys on different days comparable.

> **Calibrate before inverting.** Offsets in I / Q bias inverted models (Minsley et al., 2014). See [Sensor physics](sensor-physics.md).

Full citations: [References](references.md).
