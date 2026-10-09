# Input data

[← Documentation index](README.md)

The app accepts `.csv` and `.xlsx` files. A file is treated as a **GEM-2 export** when its columns match the GEM-2 layout; otherwise it is read as a **legacy multi-sheet** workbook.

## GEM-2 exports (recommended)

One table holding all frequencies and lines, as written by WinGEM / EMExport.

**Required columns:** `Line`, `Y`, and one or more channel columns:

| Channel | Column pattern | Unit |
|---|---|---|
| Apparent conductivity (EC) | `EC{freq}Hz[mS/m]` | mS/m |
| Apparent susceptibility (MS) | `MSusc{freq}Hz[1/1000]` | 10⁻³ SI |
| In-phase (I) | `I_{freq}Hz` | ppm |
| Quadrature (Q) | `Q_{freq}Hz` | ppm |

**Optional columns**, used when present: `X` / `Y` or `Lat` / `Lon`, `Sample`, `Mark` (event markers), `Status` (quality flag), `PowerLn` (power-line noise, mG), `QSum`, `TotalEC[mS/m]`, time, height and temperature columns.

Each mode found in the file gets its own tab: **EC**, **MS**, **I**, **Q** and **AUX** (power-line noise, quadrature sum, total EC, and the magnetic-viscosity channels when [computed](interpretation.md#magnetic-viscosity)).

### Tables converted from `.gbf` files

- In-phase / quadrature named `Ip_{freq}Hz` / `Qd_{freq}Hz` are renamed to `I_` / `Q_`.
- Without a `Line` column, lines are taken from runs of constant X (lines walked along Y) or constant Y. A new line also starts wherever the along-line coordinate restarts (e.g. Y runs 0 → 77 m, then back to 0), so a grid is split correctly even when X never changes. Failing that, lines are split at pauses of more than 3 s in the time column.
- The file's warnings say how the lines were found.

## GEM data options (sidebar)

- **Drop readings with a Status flag** (on by default) — the GEM-2 writes a non-zero `Status` when a reading has a problem such as ADC overload (GEM-2 Manual v3.8).
- **Lines to leave out** — comma-separated line labels, e.g. calibration or test lines recorded in the same file. They are removed from profiles and maps.
- **Distance along line**:

  | Option | Distance of each reading |
  |---|---|
  | Y column | The exported `Y` value (WinGEM grid surveys) |
  | Projection on the survey axis | Coordinates projected on the main axis of the whole survey — repeat passes walked in either direction share distances |
  | Path length along each line | Cumulative distance from each line's first reading |
  | Reading number × spacing | Reading order within the line × the spacing you enter |
  | Between event markers | Markers placed every *spacing* metres, readings spaced evenly between them (dead reckoning); readings outside the first and last marker are not used |

- **Event markers** are drawn as dotted vertical lines on every profile.
- **Sensor geometry** and **Corrections & filters** are described in [Sensor physics](sensor-physics.md) and [Corrections & filters](corrections.md).

## Legacy multi-sheet XLSX

| Element | Requirement |
|---|---|
| Sheets | One frequency per sheet (sheet name = frequency label) |
| Column 0 | Distance along the transect (m, numeric) |
| Columns 1+ | One column per survey line / pass (numeric) |
| Mode | Choose **Measurement mode** (EC or MS) in the sidebar — GEM files show every mode automatically |

## CSV vs XLSX precision

The GEM software exports different precision levels:

| Format | EC precision | MS precision | Note |
|---|---|---|---|
| `.csv` | Integer (no decimals) | 1 d.p. | **Reduced precision** — scores may differ slightly |
| `.xlsx` | 3+ d.p. | 4+ d.p. | **Full instrument precision** — recommended for analysis |

The app detects reduced-precision CSVs and warns. For quantitative frequency comparison use the XLSX file.

## Single-pass files

Files with only one line (one trace per frequency) still produce scores, using a second-difference noise estimate on the measured samples — see [Scoring](profiles-and-scoring.md#scoring). Single-pass scores are less reliable; repeat passes are always preferable.
