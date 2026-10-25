"""
GEM-2 Analysis Tool — Streamlit UI

Run with:
    streamlit run RIs_v2.py
"""
from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import PchipInterpolator, interp1d

import contouring as ctr
import corrections
import emphysics
import gem_io
import gridtools
import pipeline
import ui_tools

# ---------------------------------------------------------------------------
# Configuration (all in one place, easily overridden via Streamlit widgets)
# ---------------------------------------------------------------------------
DEFAULT_DISTANCE_STEP = 0.5
DEFAULT_INTERP_KIND = "linear"
SCORE_EPSILON = 1e-8

# Noise estimators recorded in each score_dict ("noise_method")
NOISE_BETWEEN = "between-trace"   # ≥ 2 overlapping passes
NOISE_INTRA = "intra-profile"     # single usable pass
EXCEL_SHEET_NAME_MAX = 31

MODES = {
    "EC": "Electrical conductivity",
    "MS": "Magnetic susceptibility",
}
# Modes that get 2D contouring. Raw in-phase and quadrature (ppm) grow with
# frequency, so their pseudo-sections show that trend rather than the ground;
# AUX channels are not frequencies but can be mapped (e.g. power-line noise).
PSEUDOSECTION_MODES = tuple(MODES)
AREA_MAP_MODES = (*MODES, "AUX")
PREP_APPLIES_TO = "Applied to every GEM file before its profiles, maps and further analysis."

# Readings are usually denser than the profile grid, so linear and PCHIP
# differ little; nearest keeps single readings, kriging filters the noise.
INTERP_METHODS = ["linear", "pchip", "nearest", "kriging"]
PROFILE_KRIGE_NEIGHBOURS = 16       # readings around each grid point (moving neighbourhood)
PROFILE_VARIOGRAM_SPANS = 2         # variogram fitted up to this many neighbourhood widths
PROFILE_KRIGE_CHUNK = 4096          # grid points per batched kriging solve

LINE_STYLES = {
    "Solid": "-",
    "Dashed": "--",
    "Dash-dot": "-.",
    "Dotted": ":",
}

# Publication-style figures for every plot in the app: boxed axes with inward
# ticks, a light grid, and the colour-blind-safe Okabe & Ito (2008) palette
# (yellow last: it is faint on white).
PLOT_STYLE = {
    "axes.prop_cycle": plt.cycler(color=[
        "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#F0E442",
    ]),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "0.15",
    "axes.titlepad": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": False,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "lines.linewidth": 1.4,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "0.8",
    "legend.framealpha": 0.9,
    "image.cmap": "viridis",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PLOT_STYLE)

# GEM instrument format detection (column patterns live in gem_io)
GEM_EC_PATTERN = gem_io.CHANNEL_PATTERNS["EC"]
GEM_MS_PATTERN = gem_io.CHANNEL_PATTERNS["MS"]
GEM_REQUIRED_COLS = {'Line', 'Y'}

SCORING_HELP = (
    "Rank frequencies by their signal-to-noise ratio (signal power / noise power, in dB). Off: "
    "profiles, maps and exports are produced without scores."
)
SMOOTHING_SCORE_WARNING = (
    "Despiking lowers the noise σ, so scores rise. Compare scores only between runs with "
    "the same filters."
)
KRIGING_SCORE_WARNING = (
    "Kriging filters the nugget (noise) out of each pass, so the noise σ falls and scores "
    "rise. Compare scores only between runs with the same interpolation method."
)
DISTANCE_HELP = (
    "How the distance of each reading along its line is found. "
    "Projection: coordinates projected on the main survey axis — use for "
    "repeat passes walked in either direction. Path length: restarts at the "
    "first reading of each line. Event markers: markers placed every "
    "'spacing' metres, readings spaced evenly between them."
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


SCIENCE_HELP = {                 # tooltips (the "?" icon) of the science choices
    "altitude": (
        "Column with the sensor height above ground of each reading. With 'Recompute EC / MS "
        "from I / Q', each reading is converted at its own height (the physics removes the "
        "height effect). Otherwise readings are corrected to the reference height with a "
        "fitted height response, which also removes any change of the ground that goes with "
        "height."
    ),
    "altitude_ref": (
        "Height every reading is corrected to; 0 uses the median height."
    ),
    "area_map": (
        "Grids the readings over the survey area from their coordinates and draws a contour "
        "map of one channel, on the EC, MS and AUX tabs."
    ),
    "bucking": (
        "Distance from the transmitter to the bucking coil, which cancels the primary field "
        "at the receiver (1.035 m on the GEM-2). 0 models a plain coil pair."
    ),
    "cell": (
        "Grid spacing of the map. Auto gives about one grid node per reading. Smaller cells "
        "draw a finer map but cannot add detail between lines that were not measured."
    ),
    "clip": (
        "Blanks the most extreme values of each channel. It always removes the strongest "
        "readings of the survey, which are often the targets themselves (metal, kilns, clay "
        "lenses, saline patches): use the despike filter for isolated spikes."
    ),
    "clip_range": (
        "Values below the lower or above the upper percentile of the whole survey are blanked."
    ),
    "colour_lines": (
        "Draws black contour lines at the colour boundaries on top of the map."
    ),
    "colour_map": (
        "Colours from low to high values. Viridis and cividis change evenly in brightness, so "
        "equal steps in value look equal; rainbow scales (Surfer's default rainbow, jet, "
        "turbo) show more detail at a glance but make some value steps look like sharp "
        "edges."
    ),
    "colour_range": (
        "Values the colours span. Percentiles (default 2–98 %) keep a few extreme readings "
        "from washing out the rest; the full range shows every value in colour, as Surfer "
        "does; fixed values give several maps the same scale. Values beyond the range take "
        "the end colours (arrows on the colour bar)."
    ),
    "colour_scale": (
        "How values map to colours. Linear: equal value steps get equal colour steps. "
        "Logarithmic: equal ratios do (positive values only), for skewed data such as EC. "
        "Histogram-equalised: each colour covers the same share of the map, which shows "
        "the most contrast in skewed data but stretches small differences."
    ),
    "combined": (
        "Merged: one map from several surveys, each levelled to the others where they "
        "overlap. Difference: the change between two surveys of the same area (B − A)."
    ),
    "despike": (
        "Blanks single readings that jump away from their neighbours (metal, knocks) by more "
        "than the threshold × the noise σ."
    ),
    "despike_threshold": (
        "How far from the running median a reading must be, in units of the robust noise σ, "
        "to count as a spike. Lower values remove more."
    ),
    "despike_window": (
        "Number of readings in the running median each reading is compared with."
    ),
    "distance_step": (
        "Spacing of the common distance grid every pass is resampled onto, and so of the "
        "representative profile."
    ),
    "drift_model": (
        "How instrument drift between base-station visits is followed: piecewise linear "
        "through every visit, or one straight line through all of them."
    ),
    "ec25": (
        "Converts EC to its value at 25 °C (EC rises about 2 % per °C), so surveys made at "
        "different temperatures can be compared."
    ),
    "edge_distance": (
        "Readings of two surveys closer than this are compared to find the level shift "
        "between them."
    ),
    "edge_match": (
        "Shifts each survey by the median difference to the earlier ones where they overlap, "
        "removing calibration offsets between days."
    ),
    "gridding": (
        "How values between readings are estimated (the methods of Surfer's Grid Data). "
        "Thin-plate spline: smooth surface through the data. Ordinary kriging: weights from "
        "the spatial correlation (variogram), with an error map. Linear: flat triangles "
        "between readings. Minimum curvature: the smoothest surface through the data, common "
        "for geophysical maps. Inverse distance: weighted mean, nearer readings count more; "
        "leaves bull's-eyes around readings. Natural neighbour: weights from the area each "
        "reading would give up; smooth and never beyond the data range."
    ),
    "height": (
        "Height of the coils above the ground. The response weakens with height and comes "
        "from deeper on average. Used by the physics tools, the EC / MS recompute and the "
        "inversion."
    ),
    "idw_power": (
        "How fast a reading's weight falls with distance. 2 is the usual choice; higher "
        "powers follow the nearest readings more closely, lower ones average more."
    ),
    "idw_smoothing": (
        "Adds this distance to every reading's distance, so no reading is matched exactly "
        "and bull's-eyes soften. 0 passes through the readings."
    ),
    "interpolation": (
        "How each pass is resampled onto the common grid. Linear joins the readings with "
        "straight lines; PCHIP is smooth and, like linear, never overshoots the readings. "
        "Nearest takes the closest reading: nothing is averaged, so the noise σ is that of "
        "the readings, but the profile is stepped and positions shift by up to half the "
        "reading spacing. Kriging weights the 16 readings around each grid point by a "
        "gaussian variogram fitted to the passes of the channel, and treats the nugget as "
        "noise: the smoothest profile, but the noise σ falls and scores rise, so compare "
        "scores only within one method. Kriging is the slowest."
    ),
    "lag_channel": (
        "Channel compared between neighbouring lines; one with clear anomalies works best."
    ),
    "levels": (
        "Number of colour bands between the lowest and highest mapped value."
    ),
    "map_despike": (
        "How far from the local median a cell must be, in robust σ units, to be replaced."
    ),
    "map_display": (
        "Filled contours group values into the colour bands. A continuous image colours "
        "every grid cell by its own value, without banding (Surfer's image map)."
    ),
    "map_filter": (
        "Despike replaces isolated outlier cells. Low-pass smooths. High-pass removes the "
        "regional trend to show local anomalies."
    ),
    "map_window": (
        "Size of the moving window, in cells per side."
    ),
    "min_curvature_tension": (
        "0 gives the smoothest surface (minimum curvature), which may overshoot between "
        "readings; values towards 1 pull the surface tight between them, like a stretched "
        "membrane (Smith & Wessel, 1990). 0.25–0.35 suits steep anomalies."
    ),
    "pseudosection": (
        "Frequency against distance for each line, on the EC and MS tabs (raw in-phase and "
        "quadrature grow with frequency, so they are left out). Lower frequencies see deeper "
        "on average, but the frequency axis is not a calibrated depth (McNeill, 1980)."
    ),
    "ref_method": (
        "Regression: least-squares line through the matched pairs. Moments: gain and offset "
        "that match the mean and spread of the reference values."
    ),
    "ref_radius": (
        "Survey readings within this distance of a reference point are matched to it (their "
        "median)."
    ),
    "separation": (
        "Distance from the transmitter to the receiver coil (1.66 m on the GEM-2). Wider "
        "separation sees deeper."
    ),
    "spacing": (
        "Reading number: distance walked between two readings. Markers: distance between two "
        "event markers; readings in between are spread evenly (dead reckoning)."
    ),
    "temperature": (
        "Column with the instrument or air temperature. Its effect is measured at the base "
        "station and removed from every reading."
    ),
    "variogram": (
        "Shape of the spatial correlation used by kriging. Spherical and exponential suit "
        "rough fields (they rise linearly at short distances); gaussian suits very smooth "
        "ones."
    ),
}


# ---------------------------------------------------------------------------
# Graph editor options
# ---------------------------------------------------------------------------

PROFILE_X = "Distance along line (m)"          # graph-editor choices that keep the profile view
PROFILE_Y = "Representative profiles"
TIME_X = "Time (s)"


@dataclass
class GraphOptions:
    """Visual settings collected from the graph editor panel."""
    selected_sheets: list[str] = field(default_factory=list)  # empty → show all
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    show_grid: bool = True
    line_width: float = 2.0
    line_style: str = "-"
    show_traces: bool = True
    show_envelope: bool = True
    plot_title: str = ""   # empty → use auto-generated default
    x_label: str = ""      # empty → "Distance (m)"
    y_label: str = ""      # empty → profile_label(mode)
    x_column: str = PROFILE_X   # what is plotted (plot_axis_options)
    y_column: str = PROFILE_Y

    @property
    def readings(self) -> bool:
        """True when a column is chosen for Y: plot the readings instead of the profiles."""
        return self.y_column != PROFILE_Y


@dataclass(frozen=True)
class ContourSettings:
    """2D contouring options collected from the sidebar."""
    area_map: bool = False
    pseudosection: bool = False
    coord_mode: str = "auto"              # "auto" | "metres" | "degrees"
    method: str = "spline"                # key of contouring.METHODS
    smoothing: float = 0.0                # thin-plate spline only
    variogram_model: str = "spherical"    # kriging only
    cell_size: float | None = None        # None → automatic
    blank_distance: float | None = None   # None → automatic
    level_lines: bool = False
    n_levels: int = 20
    projection: str = "local"             # key of contouring.PROJECTIONS (degrees input)
    xy_epsg: int | None = None            # CRS of metre X/Y when known
    method_options: tuple = ()            # (name, value) pairs, see contouring.METHOD_DEFAULTS
    colours: ctr.ColourStyle = ctr.ColourStyle()
    show_points: bool = True


PSEUDOSECTION_CAPTION = (
    "Frequency axis is not a calibrated depth axis: under LIN the depth response "
    "is set by coil geometry (McNeill, 1980); at most, lower frequencies see "
    "somewhat deeper (Huang, 2005). "
    "White lines mark measured frequencies; values between them are interpolated."
)
LEVELLING_HELP = (
    "Shifts each line so its median equals the survey median (removes "
    "line-to-line offsets / striping). Also removes any real gradient across "
    "lines — compare with levelling off."
)

APP_VERSION = "2.1"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
DOCS_URL = ui_tools.DOCS_URL

ABOUT_MD = f"""
**GEM-2 Analysis Tool** — v{APP_VERSION} · [documentation]({DOCS_URL}/README.md) ·
cite as [doi:10.5281/zenodo.18302817](https://doi.org/10.5281/zenodo.18302817)

Profiles, frequency ranking, corrections, sensor physics, inversion,
2D mapping and soil tools for multi-frequency GEM-2 surveys. Every
setting has a **?** tooltip with the science behind it.

#### How the GEM-2 measures

The GEM-2 (Won et al., 1996) is a broadband frequency-domain EMI
sensor. A transmitter coil drives a time-varying primary magnetic
field that induces eddy currents in the ground; their secondary field
is measured at the receiver as in-phase (I) and quadrature (Q) parts,
in ppm of the primary. A bucking coil cancels the primary at the
receiver.

Under the **low-induction-number (LIN)** approximation (McNeill, 1980)
the two parts decouple:

| Component | Physical quantity | Unit |
|---|---|---|
| **Quadrature** (out-of-phase) | Apparent electrical conductivity (EC) | mS/m |
| **In-phase** | Apparent magnetic susceptibility (MS) | 10⁻³ SI |

**EC** rises with moisture, clay content and mineralogy, salinity and
fine texture; low values indicate dry, sandy or gravelly ground
(Reynolds, 2011). **MS** is raised by ferrimagnetic minerals
(maghemite, magnetite), pedogenic enhancement of topsoil, burning and
anthropogenic enrichment (hearths, kilns, iron-rich fills) and mafic
parent material.

The decoupling is not perfect: a viscous (frequency-dependent)
susceptibility adds to the quadrature (Simon et al., 2015 — see
*Magnetic viscosity*), and above about 40 kHz the in-phase can include
a dielectric-permittivity contribution (Benech et al., 2016).

#### Frequency and depth

The **skin depth**

$$\\delta = \\sqrt{{\\frac{{2}}{{\\omega \\mu \\sigma}}}}$$

is the depth at which the primary field falls to 1/e ≈ 37 % of its
surface value; it shrinks as frequency and conductivity rise. LIN
holds when the induction number B = s/δ ≪ 1 (s = coil separation).
Then δ is much larger than the sensor, and the depth response is set
by the coil geometry, **not** by frequency (McNeill, 1980; Callegary
et al., 2007). As B grows — high frequencies, conductive ground — the
response moves towards the surface and LIN breaks down (Callegary et
al., 2007; Delefortrie et al., 2014). For small broadband sensors the
practical depth of investigation grows roughly with √δ, so lower
frequencies see *somewhat* deeper (Huang, 2005): a qualitative trend,
not a depth scale. Depth itself comes from varying the coil geometry
or height, or from inverting a layered model (e.g. De Smedt et al.,
2013; *Layered-earth inversion* below). The **Frequency information**
panel (EC tab) shows δ, B and the depth of investigation for your data.

What differs most between frequencies at one site is therefore the
**noise** — the GEM-2 shares its transmitter power among its
frequencies, and the noise level depends strongly on the frequency
choice (Vilhelmsen & Døssing, 2022) — and whether LIN still holds.
That is why the best frequency is site-specific, and what the score
measures.

---

#### Representative incision and scoring

A **representative incision** is a survey transect, typically
50–200 m long, that crosses the main geophysical contrasts expected
across the site — for example from a high-EC wetland fringe into a
low-EC sandy terrace — walked at least twice. The repeat passes reveal
instrument drift, operator-induced variability and short-term
environmental noise; this replicated design is what makes the
between-pass spread a meaningful noise estimate.

Each frequency receives a **signal-to-noise ratio** (SNR), the ratio
of signal power to noise power (Sheriff & Geldart, 1995), in decibels
as is usual for data quality (Bakulin et al., 2022):

$$\\mathrm{{SNR_{{dB}}}} = 10\\,\\log_{{10}}\\frac{{P_s}}{{P_n}}, \\qquad P_n = \\sigma_{{\\text{{noise}}}}^2, \\qquad P_s = \\operatorname{{var}}(\\bar p) - \\sigma_{{\\text{{noise}}}}^2 \\left\\langle \\tfrac{{1}}{{k}} \\right\\rangle$$

| Symbol | Meaning |
|---|---|
| **p̄** | Mean profile of the passes, where at least two overlap (single-pass files: the measured readings) |
| **P_s** | Signal power — the variance of p̄ about its mean, less the noise still in it: averaging k passes leaves σ²/k of noise at each distance (single pass: k = 1). So P_s does not grow with the noise, however many passes there are. |
| **P_n** | Noise power, σ², with σ estimated from the data as below. |

- **≥ 2 passes (recommended):** at each distance step where at least
  two passes were measured, the sample variance across passes
  (ddof = 1); σ is the square root of the mean of these variances.
  The passes are a finite sample of the measurement process, so the
  sample formula applies (with two passes the population formula would
  understate σ by √½ ≈ 0.71). Each pass is used only inside its own
  measured range — nothing is extrapolated.
- **Lines at different positions** (an area survey): lines are grouped
  into transects by their position across the line direction (less
  than 1 m apart = one transect). σ and A are taken only within each
  transect walked at least twice — or within each line when every
  transect was walked once — since lines over different ground differ
  by the geology, not by noise.
- **1 pass (fallback):** from the *measured* samples, not the
  interpolated grid. Second differences Δ²y cancel a locally linear
  trend; for white noise var(Δ²y) = 6σ², so
  σ ≈ 1.4826 · MAD(Δ²y) / √6. The median absolute deviation ignores
  the few large differences at sharp boundaries, and the estimate does
  not depend on the distance step or interpolation method.
  Single-pass scores are less reliable; the ranking warns when they
  are mixed with multi-pass scores.
- **σ < 10⁻⁸ or not estimable:** the score is left blank and ranked
  last, with a warning.

0 dB means as much signal as noise power; every 10 dB is ten times
more. When P_s ≤ 0 — no signal rises above the noise — the score is
left blank and ranked last.

> **Filters change scores.** Despiking and kriging lower σ and so
> raise the score. Compare scores only between runs with the same
> settings.

**Frequency selection** under the ranking keeps up to three
high-scoring frequencies whose mean profiles are not strongly
correlated, since correlated frequencies carry the same information
(Minsley et al., 2010; Vilhelmsen & Døssing, 2022).

---

#### Processing pipeline

| Step | What happens |
|---|---|
| **1. Ingest** | CSV or XLSX. A GEM-2 export is recognised by its columns (`Line`, `Y`, `EC…Hz[mS/m]`, `MSusc…Hz[1/1000]`, `I_…Hz`, `Q_…Hz`, or `Ip_` / `Qd_` from `.gbf` conversions). Readings with a non-zero `Status` are dropped (optional). |
| **2. Correct** | The enabled **Corrections & filters** run in a fixed order — despike, clip, sensor height, temperature and base-station drift, calibration offsets, GPS lag, reference calibration, EC at 25 °C. Optionally EC / MS are recomputed from I / Q after the calibration offsets. |
| **3. Locate** | The distance of each reading along its line is found (sidebar **Distance along line**); each channel becomes a distance × line matrix. |
| **4. Interpolate** | Every pass is resampled onto a common distance grid (**Distance step**) inside its own measured range. At a repeated distance within a pass only the first reading is kept. |
| **5. Score** (optional) | With **Frequency scoring** on, the mean profile, σ and score of each frequency; frequencies ranked by descending score. |
| **6. Visualise & export** | Overview and per-frequency plots (passes, mean profile, ±1σ envelope), graph editor, maps, and per-file and batch downloads. |

> **Precision note:** GEM CSV exports round EC to integers and MS to
> one decimal place. For quantitative frequency comparison use the
> XLSX export, which keeps the instrument's full precision.

---

#### Interpolation methods

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial (Fritsch & Carlson, 1980). Smooth, shape-preserving and monotone within each interval, so it never overshoots the readings. |
| **nearest** | 2 | Value of the closest reading. Nothing is averaged, so σ is that of the readings; the profile is stepped and positions shift by up to half the reading spacing. |
| **kriging** | 2 | 1D ordinary kriging from the 16 readings around each grid point, with a gaussian variogram fitted over short lags to the passes of each channel (Cressie, 1985). The nugget is treated as measurement noise and filtered out (Cressie, 1993): the smoothest profile, but σ falls and scores rise. |

GEM-2 readings (every 0.1–0.2 m) are usually denser than the profile
grid, so linear, PCHIP and nearest give nearly the same profiles and
scores; kriging scores are higher and comparable only with each other.

---

#### Tools

| Area | What it does | Key references |
|---|---|---|
| **Corrections & filters** | Despiking, percentile clipping, sensor-height correction, temperature and base-station drift, calibration offsets, GPS lag, reference calibration (ERT / TDR), EC at 25 °C | Vilhelmsen & Døssing (2022); González Jiménez et al. (2022); Lavoué et al. (2010); Mester et al. (2011); Dragonetti et al. (2018); Sheets & Hendrickx (1995); Corwin & Lesch (2005) |
| **Sensor physics** | Layered-earth forward model with the bucking coil; EC / MS recomputed from I / Q by a half-space fit; frequency information; survey-planning forward model; multi-height calibration | Won et al. (1996); Ward & Hohmann (1988); McNeill (1980); Huang & Won (2000); Minsley et al. (2014) |
| **Layered-earth inversion** | Smooth layered conductivity along the line from the quadrature, laterally constrained, with data errors from the measured noise (between passes, or within the line); EMagPy export | Constable et al. (1987); Auken & Christiansen (2004); McLachlan et al. (2021) |
| **Interpretation aids** | Redundancy-aware frequency selection, magnetic viscosity from two frequencies, anomaly spectrum (Argand diagram), power-line noise map | Minsley et al. (2010); Simon et al. (2015); Huang & Won (2003); Geophex (2004) |
| **2D mapping** | Pseudo-sections (the frequency axis is not a depth axis); area maps with line levelling, block-median reduction and six gridding methods — kriging with a fitted variogram, minimum curvature, thin-plate spline, natural neighbour, linear and inverse distance — cross-validation, map filters and combined surveys | Briggs (1974); Sandwell (1987); Smith & Wessel (1990); Park et al. (2006); Cressie (1985); Oliver & Webster (2014); Corwin & Lesch (2005); Lesch et al. (1995); Mauring & Kihle (2006); Li & Heap (2011) |
| **Soil tools** | Response-surface sampling design, soil-property calibration on ln(EC), fuzzy c-means management zones | Lesch (2005); Lesch et al. (1995); Bezdek (1981); Fridgen et al. (2004); Boydell & McBratney (2002) |

Methods in full: [documentation]({DOCS_URL}/README.md).

---

#### Supported file formats

| Format | Notes |
|---|---|
| **GEM-2 `.xlsx`** | Full instrument precision — recommended for scoring |
| **GEM-2 `.csv`** | EC rounded to integers; may slightly affect scores |
| **GEM-2 I / Q** | `I_…Hz` / `Q_…Hz` in ppm, plus `PowerLn`, `QSum`, `TotalEC[mS/m]` when present |
| **`.gbf`-converted tables** | `Ip_` / `Qd_` columns; lines found from X / Y or time when there is no `Line` column |

---

#### Glossary

| Term | Definition |
|---|---|
| **EMI** | Frequency-domain electromagnetic induction |
| **I / Q** | In-phase / quadrature part of the secondary field, in ppm of the primary |
| **EC** | Apparent electrical conductivity (mS/m) — from the quadrature |
| **MS** | Apparent magnetic susceptibility (10⁻³ SI) — from the in-phase |
| **LIN** | Low-induction-number approximation (B = s/δ ≪ 1), under which EC and MS decouple (McNeill, 1980) |
| **Skin depth (δ)** | Depth at which the primary field falls to 1/e; decreases with frequency and conductivity |
| **SNR** | Signal-to-noise ratio, 10 log₁₀(P_s / P_n) dB — the score used for frequency ranking |
| **P_s, P_n** | Signal power (variance of the mean profile less its remaining noise) and noise power (σ²_noise) |
| **σ_noise** | Pooled sample std across passes (multi-pass) or second-difference MAD estimate on the measured samples (single pass) |
| **Representative incision** | A transect designed to sample the full range of subsurface variability at a site |
| **PCHIP** | Shape-preserving piecewise cubic interpolation that avoids overshoot |
| **Nugget** | Variogram value at zero distance: measurement noise and variation shorter than the reading spacing |

---

#### References
"""


def about_references_markdown() -> str:
    """Bibliography of the About panel, read from docs/references.md so the
    app and the documentation always cite the same works."""
    try:
        text = (DOCS_DIR / "references.md").read_text(encoding="utf-8")
    except OSError:
        return f"See [References]({DOCS_URL}/references.md)."
    return "\n".join(line for line in text.splitlines() if line.startswith("- "))


# ---------------------------------------------------------------------------
# GEM format detection & pivoting
# ---------------------------------------------------------------------------

def is_gem_format(df: pd.DataFrame) -> bool:
    """Return True if *df* has the GEM instrument column structure (after gem_io.normalise_columns)."""
    df, _ = gem_io.normalise_columns(df)
    cols = set(str(c) for c in df.columns)
    if not GEM_REQUIRED_COLS.issubset(cols):
        return False
    channels = gem_io.find_channels(df.columns)
    return any(channels[m] for m in gem_io.FREQUENCY_MODES)


def pivot_gem_frequency(
    df: pd.DataFrame,
    value_col: str,
    distance_col: str = "Y",
    line_col: str = "Line",
) -> pd.DataFrame:
    """
    Pivot a GEM dataframe for one frequency column into process_sheet format.

    Returns a DataFrame with column 0 = distance (Y values) and
    columns 1+ = one column per unique Line value.

    Repeated readings at the same Y within a line keep only the first reading:
    averaging them would lower that trace's noise alone (biasing the
    between-trace σ) and silently merge a line walked out-and-back.
    """
    subset = df[[distance_col, line_col, value_col]].copy()
    subset[distance_col] = pd.to_numeric(subset[distance_col], errors="coerce")
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=[distance_col, value_col])
    subset = subset.drop_duplicates(subset=[line_col, distance_col], keep="first")

    pivoted = subset.pivot_table(
        index=distance_col,
        columns=line_col,
        values=value_col,
        aggfunc="first",
    )

    # Rename columns to "Line_0", "Line_1", "Line_L1", etc.
    pivoted.columns = [gem_io.trace_column(c) for c in pivoted.columns]

    # Reset index so column 0 = distance (what process_sheet expects)
    pivoted = pivoted.reset_index()
    return pivoted


def parse_gem_dataframe(
    df: pd.DataFrame,
    warnings: list[str] | None = None,
    distance_col: str = "Y",
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Parse a GEM-format DataFrame into pivoted DataFrames per mode and channel.

    Pivoting messages (e.g. dropped duplicate readings) are appended to *warnings*.

    Returns
    -------
    {mode: {label: pivoted_df}} for every mode in gem_io.GEM_MODES, e.g.
    {"EC": {"4525Hz": pivoted_df, ...}, "Q": {...}, "AUX": {"PowerLn": ...}}
    """
    result: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in gem_io.GEM_MODES}

    if warnings is not None:
        d = pd.to_numeric(df[distance_col], errors="coerce")
        keyed = pd.DataFrame({"Line": df["Line"], "d": d}).dropna(subset=["d"])
        dup = keyed.duplicated(keep="first")
        if dup.any():
            warnings.append(
                f"{ui_tools.plural(int(dup.sum()), 'repeated reading')} at the same distance within "
                f"{ui_tools.plural(keyed.loc[dup, 'Line'].nunique(), 'line')}: kept the first reading "
                "of each. If a line was walked out-and-back under one Line number, "
                "give each direction its own Line."
            )

    for mode, channels in gem_io.find_channels(df.columns).items():
        for label, col in channels.items():
            result[mode][label] = pivot_gem_frequency(df, value_col=col, distance_col=distance_col)

    return result


def _empty_modes() -> dict[str, dict]:
    return {m: {} for m in gem_io.GEM_MODES}


@st.cache_data(show_spinner=False, max_entries=64)
def process_gem_file(
    file_bytes: bytes,
    file_name: str,
    distance_step: float = DEFAULT_DISTANCE_STEP,
    interp_kind: str = DEFAULT_INTERP_KIND,
    prep: pipeline.PrepSettings | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, dict]], list[str]]:
    """
    Process a GEM-format file (CSV or XLSX).

    *prep* selects quality filtering and how distance along each line is
    found (default: drop flagged readings, distance = Y column).

    Returns
    -------
    output_data : {mode: {channel: interp_df, ...}} for every mode in gem_io.GEM_MODES
    scores      : {mode: {channel: score_dict, ...}}
    warnings    : list of warning strings
    """
    warnings: list[str] = []

    # Read raw data based on extension
    if file_name.lower().endswith(".csv"):
        try:
            raw_df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            warnings.append(f"Could not read CSV: {exc}")
            return _empty_modes(), _empty_modes(), warnings
    else:
        try:
            raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        except Exception as exc:
            warnings.append(f"Could not read Excel: {exc}")
            return _empty_modes(), _empty_modes(), warnings

    # ── Precision check ─────────────────────────────────────────────────────
    # GEM CSV exports often round EC values to integers and MS to 1 d.p.,
    # while the XLSX retains full instrument precision (3+ decimal places).
    # Detect this and warn the user so they know scores may differ.
    if file_name.lower().endswith(".csv"):
        low_prec_cols: list[str] = []
        for col in raw_df.columns:
            col_str = str(col)
            if GEM_EC_PATTERN.match(col_str) or GEM_MS_PATTERN.match(col_str):
                vals = pd.to_numeric(raw_df[col], errors="coerce").dropna()
                # Flag columns where every value is an integer (no fractional part)
                if len(vals) > 0 and (vals % 1 == 0).all():
                    low_prec_cols.append(col_str)
        if low_prec_cols:
            warnings.append(
                "Reduced precision detected: the following columns contain only "
                f"integer values — {', '.join(low_prec_cols)}. "
                "GEM CSV exports typically round measurements, which causes scores to "
                "differ slightly from the XLSX equivalent. "
                "Use the XLSX file for full instrument precision."
            )

    try:
        prepared, messages = prepared_table(file_bytes, file_name, prep)   # shared with the panels
    except ValueError as exc:
        warnings.append(f"Could not prepare the data: {exc}")
        return _empty_modes(), _empty_modes(), warnings
    warnings.extend(messages)

    gem_data = parse_gem_dataframe(prepared, warnings, distance_col=pipeline.DISTANCE_COL)
    tracks = trace_tracks(prepared)
    if tracks is not None:
        warnings.append(SEPARATE_TRACKS_WARNING.format(n=len(set(tracks.values()))))

    output_data: dict[str, dict[str, pd.DataFrame]] = _empty_modes()
    scores: dict[str, dict[str, dict]] = _empty_modes()

    for mode_key in gem_io.GEM_MODES:
        for freq_label, pivoted_df in gem_data[mode_key].items():
            interp_df, score_dict, error, col_warnings = process_sheet(
                pivoted_df, distance_step, interp_kind, tracks
            )

            for w in col_warnings:
                warnings.append(f"{mode_key} {freq_label}: {w}")

            if error:
                warnings.append(f"{mode_key} {freq_label}: skipped — {error}")
                continue

            output_data[mode_key][freq_label] = interp_df
            scores[mode_key][freq_label] = score_dict

        mixed = noise_method_warning(scores[mode_key])
        if mixed:
            warnings.append(f"{mode_key}: {mixed}")

    return output_data, scores, warnings


SEPARATE_TRACKS_WARNING = (
    "The lines lie at {n} different positions across the survey, so they are not all repeat "
    "passes of one transect. Noise is measured between repeat passes of the same transect "
    "(within each line where a transect was walked once) and the score is the median over the "
    "transects. The mean profiles and ±1σ envelopes still average lines over different ground: "
    "use the area map for the whole survey, and pick one transect for the pseudo-section and "
    "the inversion."
)


def trace_tracks(table: pd.DataFrame) -> dict[str, int] | None:
    """
    Transect of each trace column ({trace column: track}) when the lines of
    *table* lie at more than one position (gem_io.track_groups); None when
    they are all repeat passes of one transect or their positions are unknown.
    """
    try:
        groups = gem_io.track_groups(table)
    except (KeyError, ValueError):
        return None
    if not groups or len(set(groups.values())) < 2:
        return None
    return {gem_io.trace_column(line): track for line, track in groups.items()}


@st.cache_data(show_spinner=False, max_entries=64)    # up to 3 settings variants per file
def prepared_table(
    file_bytes: bytes, file_name: str, prep: pipeline.PrepSettings | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Raw GEM table after pipeline.prepare_gem_table (raises on unreadable input)."""
    return pipeline.prepare_gem_table(read_raw_table(file_bytes, file_name), prep)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

# Minimum point requirements per method
_METHOD_MIN_POINTS = {
    "linear": 2,
    "pchip": 2,
    "nearest": 2,
    "kriging": 2,       # the variogram is fitted to all passes of the channel
}


def _interpolate_with_method(
    xp: np.ndarray, yp: np.ndarray, target_x: np.ndarray, method: str,
    variogram: ctr.VariogramFit | None = None,
) -> np.ndarray:
    """Interpolate *yp* at *xp* onto *target_x* using the named method (*variogram*: kriging)."""
    if method == "linear":
        return np.interp(target_x, xp, yp)

    if method == "pchip":
        return PchipInterpolator(xp, yp)(target_x)

    if method == "nearest":
        return interp1d(xp, yp, kind="nearest", bounds_error=False, assume_sorted=True)(target_x)

    if method == "kriging":
        return _krige_profile(xp, yp, target_x, variogram)

    raise ValueError(f"Unknown interpolation method: {method!r}")


def _profile_variogram(passes: list[tuple[np.ndarray, np.ndarray]]) -> ctr.VariogramFit | None:
    """
    Gaussian variogram along the line, pooled over the (distance, value)
    *passes* of one channel by contouring.fit_variogram, up to
    PROFILE_VARIOGRAM_SPANS kriging neighbourhood widths: the lags the
    estimates use. Each pass sits on its own row, farther from the others
    than that, so only pairs within a pass count. One fit per channel keeps
    long surveys fast and smooths every pass alike.

    The gaussian model is parabolic near the origin, like an EMI signal
    averaged over the sensor footprint: spherical and exponential models,
    linear there, took the noise for slope on steep profiles (nugget 0).
    None when no pass varies. Raises ValueError when no fit is possible.
    """
    if all(np.ptp(y) == 0 for _, y in passes):
        return None
    spacing = float(np.median(np.concatenate([np.diff(x) for x, _ in passes])))
    max_lag = PROFILE_VARIOGRAM_SPANS * PROFILE_KRIGE_NEIGHBOURS * spacing
    rows = np.concatenate([np.full(len(x), 2.0 * max_lag * i) for i, (x, _) in enumerate(passes)])
    try:
        return ctr.fit_variogram(
            np.concatenate([x for x, _ in passes]), rows, np.concatenate([y for _, y in passes]),
            "gaussian", max_lag=max_lag,
        )
    except ctr.KrigingError as exc:
        raise ValueError("too few readings to fit a variogram, use linear or PCHIP") from exc


def _krige_profile(
    xp: np.ndarray, yp: np.ndarray, target_x: np.ndarray, variogram: ctr.VariogramFit | None = None
) -> np.ndarray:
    """
    1D ordinary kriging along distance (NaN outside the readings), with the
    channel's *variogram* (default: fitted to this pass, _profile_variogram).

    Each grid point is estimated from the PROFILE_KRIGE_NEIGHBOURS readings
    around it, so the cost grows linearly with the profile length. The
    nugget counts as measurement noise and is filtered out: gamma between a
    grid point and a reading at the same place is the nugget, not 0
    (Cressie, 1993), so the profile is continuous instead of jumping to
    each reading it happens to hit.
    """
    from pykrige.variogram_models import gaussian_variogram_model

    out = np.full(target_x.shape, np.nan)
    inside = np.flatnonzero((target_x >= xp[0]) & (target_x <= xp[-1]))
    vg = variogram or _profile_variogram([(xp, yp)])
    sill = vg.psill + vg.nugget if vg else 0.0
    if not sill > 0:                        # no variation the variogram could see
        out[inside] = np.interp(target_x[inside], xp, yp)
        return out

    def gamma(h):   # scaled to unit sill: the weights do not depend on the scale
        return gaussian_variogram_model([vg.psill / sill, vg.range, vg.nugget / sill], h)

    k = min(PROFILE_KRIGE_NEIGHBOURS, len(xp))
    for lo in range(0, inside.size, PROFILE_KRIGE_CHUNK):   # bounds the memory of fine grids
        idx = inside[lo:lo + PROFILE_KRIGE_CHUNK]
        t = target_x[idx]
        start = np.clip(np.searchsorted(xp, t) - k // 2, 0, len(xp) - k)
        nb = start[:, None] + np.arange(k)
        xn = xp[nb]
        a = np.ones((len(t), k + 1, k + 1))
        a[:, :k, :k] = gamma(np.abs(xn[:, :, None] - xn[:, None, :]))
        a[:, np.arange(k), np.arange(k)] = 0.0      # gamma(0) = 0 between a reading and itself
        a[:, k, k] = 0.0
        b = np.ones((len(t), k + 1))
        b[:, :k] = gamma(np.abs(t[:, None] - xn))   # the nugget stays: noise is filtered
        try:
            w = np.linalg.solve(a, b[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"kriging system is singular ({exc})") from exc
        out[idx] = np.einsum("mk,mk->m", w[:, :k], yp[nb])
    return out


# ---------------------------------------------------------------------------
# Core processing (pure functions — no side effects, no sys.exit)
# ---------------------------------------------------------------------------

def process_sheet(
    df: pd.DataFrame,
    distance_step: float = DEFAULT_DISTANCE_STEP,
    interp_kind: str = DEFAULT_INTERP_KIND,
    tracks: dict[str, int] | None = None,
) -> tuple[pd.DataFrame | None, dict | None, str | None, list[str]]:
    """
    Process a single sheet.

    *tracks* maps trace columns to transects (gem_io.track_groups): only
    traces on one transect are repeat passes, so the between-trace noise and
    the signal power are taken within each transect walked at least twice,
    or within each trace when every transect was walked once. None: all
    traces are repeat passes of one transect.

    The score is the signal-to-noise ratio in dB, 10 log10(P_s / P_n): the
    noise power P_n = sigma^2, and the signal power P_s is the variance of
    the mean profile about its mean minus the noise left in it, sigma^2 x
    mean(1 / k) with k passes at each distance (single pass: the variance of
    the measured readings minus sigma^2), so P_s is unbiased however many
    passes were averaged.

    Returns
    -------
    interpolated_df : DataFrame indexed by common distance, columns = original trace
                      columns; NaN outside each trace's own measured range
    score_dict      : {"mean_std": σ_noise, "signal_std": √P_s, "snr": P_s / P_n,
                       "score": SNR in dB, "noise_method": NOISE_BETWEEN | NOISE_INTRA,
                       "n_traces": int, "passes": passes per transect behind σ (median)}
                      score is NaN when the noise is zero or undefined, or
                      when no signal rises above the noise
                      score is NaN when the noise is zero or undefined
    error           : human-readable reason for failure, or None on success
    col_warnings    : list of per-column warning strings surfaced to the UI
    """
    col_warnings: list[str] = []

    if df.shape[1] < 2:
        return None, None, "fewer than 2 columns", col_warnings

    distance = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    line_data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    if np.all(np.isnan(distance)):
        return None, None, "distance column is all NaN", col_warnings

    min_dist = float(np.nanmin(distance))
    max_dist = float(np.nanmax(distance))

    if not (np.isfinite(min_dist) and np.isfinite(max_dist)):
        return None, None, "non-finite distance range", col_warnings

    span = max_dist - min_dist
    if span < distance_step:
        return None, None, f"distance span ({span:.2f} m) < step ({distance_step} m)", col_warnings

    # Use linspace to avoid float accumulation past max_dist
    n_points = int(round(span / distance_step)) + 1
    common_dist = np.linspace(min_dist, max_dist, n_points)

    readings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in line_data.columns:
        y = line_data[col].values
        mask = ~np.isnan(distance) & ~np.isnan(y)

        if mask.sum() < 2:
            log.debug("  Column %s: fewer than 2 valid points, skipped.", col)
            continue

        # Keep the first reading at a repeated distance rather than averaging:
        # averaging lowers this trace's noise alone and biases the between-trace σ.
        df_xy = pd.DataFrame({"d": distance[mask], "y": y[mask]}).sort_values(
            "d", kind="stable"
        )
        n_dup = int(df_xy["d"].duplicated().sum())
        if n_dup:
            col_warnings.append(
                f"column '{col}': {ui_tools.plural(n_dup, 'repeated distance value')}, kept the first reading"
            )
            df_xy = df_xy.drop_duplicates(subset="d", keep="first")

        if df_xy.shape[0] < 2:
            continue

        min_pts = _METHOD_MIN_POINTS.get(interp_kind, 2)
        if df_xy.shape[0] < min_pts:
            col_warnings.append(
                f"column '{col}': need ≥{min_pts} points for {interp_kind} interpolation, skipped"
            )
            continue
        readings[col] = (df_xy["d"].values, df_xy["y"].values)

    variogram = None
    if interp_kind == "kriging" and readings:
        try:
            variogram = _profile_variogram(list(readings.values()))
        except ValueError as exc:
            return None, None, f"kriging failed — {exc}", col_warnings

    interpolated_lines: dict[str, np.ndarray] = {}
    raw_samples: dict[str, np.ndarray] = {}
    for col, (xp, yp) in readings.items():
        try:
            values = _interpolate_with_method(xp, yp, common_dist, interp_kind, variogram)
        except Exception as exc:
            col_warnings.append(f"column '{col}': interpolation failed — {exc}")
            continue
        # Never extrapolate: a pass only contributes where it was measured
        inside = (common_dist >= xp[0]) & (common_dist <= xp[-1])
        interpolated_lines[col] = np.where(inside, values, np.nan)
        raw_samples[col] = yp

    if not interpolated_lines:
        return None, None, "no columns survived interpolation", col_warnings

    interpolated_df = pd.DataFrame(interpolated_lines, index=common_dist)
    interpolated_df.index.name = "Distance (m)"

    n_traces = interpolated_df.shape[1]
    groups: dict[int, list[str]] = {}
    for col in interpolated_df.columns:
        groups.setdefault(0 if tracks is None else tracks.get(col, -1 - len(groups)), []).append(col)
    repeats = [cols for cols in groups.values() if len(cols) >= 2]

    if repeats:
        # Between-trace noise, only where ≥ 2 passes of one transect were
        # measured. The passes are a finite sample of the measurement process,
        # hence ddof=1; the point-wise variances are pooled as sqrt(mean(var)).
        # Lines at different positions differ by the ground, not by noise.
        variances, profiles, passes = [], [], []
        for cols in repeats:
            sub = interpolated_df[cols]
            count = sub.notna().sum(axis=1)
            overlap = count >= 2
            if not overlap.any():
                continue
            variances.append(sub[overlap].var(axis=1, skipna=True, ddof=1).to_numpy())
            # mean profile and the number of passes averaged at each distance
            profiles.append((sub[overlap].mean(axis=1, skipna=True).to_numpy(), count[overlap].to_numpy()))
            passes.append(len(cols))
        if not variances:
            return None, None, "traces do not overlap in distance", col_warnings
        noise = float(np.sqrt(np.nanmean(np.concatenate(variances))))
        signal_power = _signal_power([p for p, _ in profiles], [1.0 / k for _, k in profiles], noise)
        noise_method = NOISE_BETWEEN
    else:
        # Every transect walked once: noise and signal within each trace, on
        # its measured readings (the noise estimate comes from them too).
        sigmas = np.array([_intra_profile_noise(raw_samples[c]) for c in interpolated_df.columns])
        noise = float(np.sqrt(np.nanmean(sigmas ** 2))) if np.isfinite(sigmas).any() else float("nan")
        traces = [raw_samples[c] for c in interpolated_df.columns]
        signal_power = _signal_power(traces, [np.ones(len(t)) for t in traces], noise)
        passes = [1]
        noise_method = NOISE_INTRA

    snr, score = np.nan, np.nan
    if not (np.isfinite(noise) and noise >= SCORE_EPSILON):
        # Zero or undefined noise: a ratio would be meaningless.
        col_warnings.append(f"{noise_method} noise is zero or undefined — score left blank")
    elif not signal_power > 0:
        snr = 0.0
        col_warnings.append("no signal rises above the noise (signal power ≤ 0) — score left blank")
    else:
        snr = signal_power / noise ** 2
        score = 10.0 * np.log10(snr)

    score_dict = {
        "mean_std": noise,
        "signal_std": float(np.sqrt(signal_power)) if signal_power > 0 else 0.0,
        "snr": snr,
        "score": score,
        "noise_method": noise_method,
        "n_traces": n_traces,
        "passes": int(np.median(passes)),
    }
    return interpolated_df, score_dict, None, col_warnings


def _signal_power(profiles: list[np.ndarray], noise_shares: list[np.ndarray], noise: float) -> float:
    """
    Signal power pooled over transects: for each profile its variance about
    its mean (ddof = 1) minus the noise left in it, noise^2 x mean(share),
    the share being 1 / (passes averaged) at each distance; weighted by the
    number of distances. NaN when no profile has two values.
    """
    total, weight = 0.0, 0
    for p, share in zip(profiles, noise_shares):
        p, share = np.asarray(p, dtype=float), np.asarray(share, dtype=float)
        ok = np.isfinite(p)
        if ok.sum() < 2:
            continue
        power = float(np.var(p[ok], ddof=1)) - (noise ** 2 * float(np.mean(share[ok])) if np.isfinite(noise) else 0.0)
        total += power * ok.sum()
        weight += int(ok.sum())
    return total / weight if weight else float("nan")


def _intra_profile_noise(y: np.ndarray) -> float:
    """
    Noise σ of a single pass, from its measured samples (not the interpolated
    grid), so it does not depend on the distance step or interpolation method.

    Second differences cancel a locally linear trend, and for white noise
    var(Δ²y) = 6σ², so σ ≈ 1.4826·MAD(Δ²y)/√6 — the MAD ignores the few large
    differences at sharp boundaries. Falls back to std(Δ²y)/√6 when the MAD
    is zero (e.g. coarsely rounded data). NaN if there are too few samples.
    """
    d2 = np.diff(np.asarray(y, dtype=float), n=2)
    if d2.size < 3:
        return float("nan")
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    if mad > 0:
        return 1.4826 * mad / np.sqrt(6)
    return float(np.std(d2, ddof=1) / np.sqrt(6))


def noise_method_warning(scores: dict[str, dict]) -> str | None:
    """Message when frequencies in one ranking were scored with different noise estimators."""
    intra = [name for name, sc in scores.items() if sc["noise_method"] == NOISE_INTRA]
    if not intra or len(intra) == len(scores):
        return None
    return (
        f"Only one usable pass for {', '.join(map(str, intra))}: noise was estimated "
        "within that profile rather than between passes, so its score is not "
        "strictly comparable with the others."
    )


def rank_scores(scores: dict[str, dict]) -> list[tuple[str, dict]]:
    """Scores sorted best first; blank (NaN) scores go last."""
    return sorted(
        scores.items(),
        key=lambda kv: (np.isfinite(kv[1]["score"]), np.nan_to_num(kv[1]["score"])),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Plot helpers (return Figure objects, never touch global pyplot state)
# ---------------------------------------------------------------------------

def profile_label(mode: str, channel: str | None = None) -> str:
    """Y-axis label for mean profiles, with the GEM units."""
    if mode == "AUX" and channel is None:
        return "Mean value (units in legend)"
    return f"Mean {ctr.value_label(mode, channel)}"


def _draw_markers(ax: plt.Axes, markers: list[float] | None) -> None:
    """Dotted vertical line at each event-marker distance."""
    for i, m in enumerate(markers or []):
        ax.axvline(m, color="0.4", linestyle=":", linewidth=0.8,
                   label="Event marker" if i == 0 else None)


def make_overview_figure(
    output_data: dict[str, pd.DataFrame],
    scores: dict[str, dict],
    mode: str,
    file_name: str,
    opts: GraphOptions | None = None,
    show_scores: bool = True,
    markers: list[float] | None = None,
) -> plt.Figure:
    """All representative profiles on one axes; scores in the legend when *show_scores*."""
    if opts is None:
        opts = GraphOptions()

    visible = opts.selected_sheets if opts.selected_sheets else list(output_data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    y_label = profile_label(mode)

    for sheet_name in visible:
        interp_df = output_data.get(sheet_name)
        if interp_df is None:
            continue
        rep_prof = interp_df.mean(axis=1, skipna=True)
        common_dist = interp_df.index.values
        name = ctr.value_label(mode, sheet_name) if mode == "AUX" else sheet_name
        if show_scores:
            name = f"{name} (SNR {scores[sheet_name]['score']:.1f} dB)"
        if not np.all(np.isnan(rep_prof.values)):
            ax.plot(
                common_dist,
                rep_prof.values,
                label=name,
                linewidth=opts.line_width,
                linestyle=opts.line_style,
            )

    _draw_markers(ax, markers)
    ax.set_xlabel(opts.x_label or "Distance (m)")
    ax.set_ylabel(opts.y_label or y_label)
    ax.set_title(opts.plot_title or f"Representative profiles [{mode}] — {file_name}")
    ax.legend(fontsize=8)
    ax.grid(opts.show_grid)
    if opts.x_lim is not None:
        ax.set_xlim(opts.x_lim)
    if opts.y_lim is not None:
        ax.set_ylim(opts.y_lim)
    fig.tight_layout()
    return fig


def plot_axis_options(table: pd.DataFrame | None, mode: str = "") -> tuple[list[str], list[str]]:
    """
    Choices of the graph editor's X and Y drop-downs. X: distance along the
    line (the representative-profile view), the position, time and reading
    columns, then every data channel (for cross-plots). Y: the representative
    profiles, then every data channel, those of *mode* first. Only the
    profile view without a prepared GEM table.
    """
    xs, ys = [PROFILE_X], [PROFILE_Y]
    if table is None:
        return xs, ys
    lower = {str(c).strip().lower(): str(c) for c in table.columns}
    for name in ("x", "y", "lat", "latitude", "lon", "long", "longitude", "sample"):
        if name in lower:
            xs.append(lower[name])
    if gem_io.time_seconds(table) is not None:
        xs.append(TIME_X)
    found = gem_io.find_channels(table.columns)
    order = [mode] + [m for m in gem_io.GEM_MODES if m != mode] if mode in found else list(gem_io.GEM_MODES)
    data = [col for m in order for col in found[m].values()]
    return xs + data, ys + data


def _axis_values(table: pd.DataFrame, choice: str) -> np.ndarray:
    """Values plotted for one drop-down choice: distance, time since the first reading, or a column."""
    if choice == PROFILE_X:
        return pd.to_numeric(table[pipeline.DISTANCE_COL], errors="coerce").to_numpy(dtype=float)
    if choice == TIME_X:
        t = gem_io.time_seconds(table)
        return t - np.nanmin(t)
    return pd.to_numeric(table[choice], errors="coerce").to_numpy(dtype=float)


def make_readings_figure(table: pd.DataFrame, opts: GraphOptions, title: str) -> plt.Figure:
    """
    The readings of the prepared table: opts.y_column against opts.x_column,
    one colour per line. Readings are joined in reading order where X runs
    one way along the line (distance, time, position), and drawn as points
    otherwise (cross-plots of two channels).
    """
    x = _axis_values(table, opts.x_column)
    y = _axis_values(table, opts.y_column)
    lines = gem_io.line_labels(table["Line"]) if "Line" in table.columns else pd.Series("0", index=table.index)
    groups = list(pd.Series(np.arange(len(table))).groupby(lines.to_numpy(), sort=False))
    cmap = plt.get_cmap("viridis", max(len(groups), 2))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (label, idx) in enumerate(groups):
        xi, yi = x[idx.to_numpy()], y[idx.to_numpy()]
        ok = np.isfinite(xi) & np.isfinite(yi)
        step = np.diff(xi[ok])
        colour = cmap(k)
        if ok.sum() > 1 and (np.all(step >= 0) or np.all(step <= 0)):
            ax.plot(xi, yi, color=colour, linewidth=0.5 * opts.line_width, linestyle=opts.line_style,
                    label=f"Line {label}")
        else:
            ax.scatter(xi, yi, color=colour, s=4, label=f"Line {label}")
    if len(groups) <= 12:
        ax.legend(fontsize=8)
    ax.set_xlabel(opts.x_label or opts.x_column)
    ax.set_ylabel(opts.y_label or opts.y_column)
    ax.set_title(opts.plot_title or f"{title} ({len(groups)} line{'s' if len(groups) != 1 else ''})")
    ax.grid(opts.show_grid)
    if opts.x_lim is not None:
        ax.set_xlim(opts.x_lim)
    if opts.y_lim is not None:
        ax.set_ylim(opts.y_lim)
    fig.tight_layout()
    return fig


def make_sheet_figure(
    sheet_name: str,
    interp_df: pd.DataFrame,
    mode: str,
    opts: GraphOptions | None = None,
    markers: list[float] | None = None,
) -> plt.Figure:
    """Per-sheet plot: individual traces + mean +/- 1 sigma envelope."""
    if opts is None:
        opts = GraphOptions()

    fig, ax = plt.subplots(figsize=(10, 4))
    common_dist = interp_df.index.values
    rep_prof = interp_df.mean(axis=1, skipna=True).values
    # Same estimator as the score's σ (ddof=1); blank where only one pass exists
    std_prof = interp_df.std(axis=1, skipna=True, ddof=1).values

    # Individual traces (thin, semi-transparent)
    if opts.show_traces:
        for col in interp_df.columns:
            ax.plot(
                common_dist,
                interp_df[col].values,
                color="0.55",
                alpha=0.4,
                linewidth=0.7,
                linestyle=opts.line_style,
            )

    # Mean profile
    ax.plot(
        common_dist,
        rep_prof,
        color="C0",
        linewidth=opts.line_width,
        linestyle=opts.line_style,
        label="Mean",
    )

    # +/- 1 sigma envelope
    if opts.show_envelope:
        ax.fill_between(
            common_dist,
            rep_prof - std_prof,
            rep_prof + std_prof,
            alpha=0.2,
            color="C0",
            linewidth=0,
            label="±1σ",
        )

    _draw_markers(ax, markers)
    ax.set_xlabel(opts.x_label or "Distance (m)")
    ax.set_ylabel(opts.y_label or profile_label(mode, sheet_name))
    ax.set_title(f"{sheet_name} — individual traces & representative profile")
    ax.legend()
    ax.grid(opts.show_grid)
    if opts.x_lim is not None:
        ax.set_xlim(opts.x_lim)
    if opts.y_lim is not None:
        ax.set_ylim(opts.y_lim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def excel_sheet_title(name: str) -> str:
    """Excel-safe sheet title: [ ] : * ? / and backslash become '_', at most 31 characters."""
    return re.sub(r"[\[\]:*?/\\]", "_", str(name))[:EXCEL_SHEET_NAME_MAX]


def build_excel_download(output_data: dict[str, pd.DataFrame]) -> bytes:
    """Pack all interpolated sheets into a single Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, interp_df in output_data.items():
            rep_prof = interp_df.mean(axis=1, skipna=True)
            col_label = f"{sheet_name[:EXCEL_SHEET_NAME_MAX - 5]}_mean"
            out_df = pd.DataFrame(
                {
                    "Distance (m)": interp_df.index.values,
                    col_label: rep_prof.values,
                }
            )
            out_df.to_excel(writer, sheet_name=excel_sheet_title(sheet_name), index=False)
    return buf.getvalue()


def build_scores_csv(scores: dict[str, dict]) -> bytes:
    df = pd.DataFrame.from_dict(scores, orient="index")
    df.index.name = "sheet"
    return df.to_csv().encode()


def mean_profiles_table(output_data: dict[str, pd.DataFrame], label_len: int) -> pd.DataFrame:
    """
    One "Distance (m)" column plus one "{freq}_mean" column per frequency/sheet.

    Each frequency keeps its own distance grid: profiles are outer-joined on
    distance, so a frequency is blank where it has no data rather than being
    written against another frequency's distances.
    """
    profiles = {}
    for freq, interp_df in output_data.items():
        prof = interp_df.mean(axis=1, skipna=True)
        # Round so grids that coincide up to float noise share rows
        prof.index = np.round(prof.index.to_numpy(dtype=float), 9)
        profiles[f"{freq[:label_len]}_mean"] = prof
    table = pd.concat(profiles, axis=1).sort_index()
    table.index.name = "Distance (m)"
    return table.reset_index()


def build_batch_xlsx(
    all_results: list[dict],
    include_scores: bool = True,
) -> bytes:
    """
    Build a single xlsx containing all interpolated data and scores across every
    uploaded file, mode, and frequency.

    Parameters
    ----------
    all_results : list of dicts, each with keys:
        "stem"        : str  — file stem used for sheet naming
        "mode"        : str  — "EC" or "MS"
        "output_data" : dict[freq_or_sheet, interpolated_df]
        "scores"      : dict[freq_or_sheet, score_dict]

    Workbook layout
    ---------------
    Sheet "Scores"   : one row per (file, mode, frequency) with SNR (dB) / signal σ / noise σ,
                       noise method and number of traces (only if *include_scores*)
    Per (file, mode) : distance column + one mean-profile column per frequency
    """
    buf = io.BytesIO()
    score_rows: list[dict] = []

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for entry in all_results:
            stem = entry["stem"]
            mode = entry["mode"]
            output_data: dict[str, pd.DataFrame] = entry["output_data"]
            scores: dict[str, dict] = entry["scores"]

            # ── Scores accumulation ──────────────────────────────────────
            for freq, sc in (scores.items() if include_scores else ()):
                score_rows.append({
                    "File": stem,
                    "Mode": mode,
                    "Frequency / Sheet": freq,
                    "SNR (dB)": round(sc["score"], 4),
                    "Signal (σ)": round(sc["signal_std"], 6),
                    "Noise (σ)": round(sc["mean_std"], 6),
                    "Noise method": sc["noise_method"],
                    "Traces": sc["n_traces"],
                })

            # ── Interpolated data sheet ──────────────────────────────────
            if not output_data:
                continue

            data_df = mean_profiles_table(output_data, label_len=20)

            # Sheet name: "{stem}_{mode}", truncated to 31 chars
            raw_sheet = excel_sheet_title(f"{stem}_{mode}")
            sheet_name = raw_sheet[:EXCEL_SHEET_NAME_MAX]
            # Deduplicate sheet names (multiple files could share a stem)
            existing = writer.sheets.keys()
            suffix = 2
            candidate = sheet_name
            while candidate in existing:
                tag = f"_{suffix}"
                candidate = raw_sheet[: EXCEL_SHEET_NAME_MAX - len(tag)] + tag
                suffix += 1
            sheet_name = candidate

            data_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # ── Scores sheet (written last so it sorts first in openpyxl order) ──
        if score_rows:
            scores_df = pd.DataFrame(score_rows)
            scores_df.to_excel(writer, sheet_name="Scores", index=False)

    buf.seek(0)
    return buf.getvalue()


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Graph editor UI helper
# ---------------------------------------------------------------------------

def item_noun(mode: str) -> tuple[str, str]:
    """(singular, plural) for the items of a mode: AUX channels or frequencies."""
    if mode == "AUX":
        return "channel", "channels"
    return "frequency", "frequencies"


def render_graph_editor(
    output_data: dict[str, pd.DataFrame],
    file_key: str,
    table: pd.DataFrame | None = None,
    mode: str = "",
) -> GraphOptions:
    """
    Render the graph editor expander and return the current GraphOptions.

    Parameters
    ----------
    output_data : processed sheets for this file
    file_key    : unique string used to namespace widget keys per file
    table       : prepared GEM table (offers its columns in the X / Y drop-downs)
    mode        : the tab's mode, whose channels come first in the Y drop-down
    """
    all_sheet_names = list(output_data.keys())

    # Compute data extents for axis-limit defaults
    all_x = np.concatenate([df.index.values for df in output_data.values()])
    all_y_means = np.concatenate(
        [df.mean(axis=1, skipna=True).values for df in output_data.values()]
    )
    valid_y = all_y_means[np.isfinite(all_y_means)]
    x_data_min, x_data_max = float(all_x.min()), float(all_x.max())
    y_data_min = float(valid_y.min()) if len(valid_y) else 0.0
    y_data_max = float(valid_y.max()) if len(valid_y) else 1.0

    opts = GraphOptions()
    x_options, y_options = plot_axis_options(table, mode)

    with st.expander("Graph editor", expanded=False):
        st.markdown("**Plot**")
        pc1, pc2 = st.columns(2)
        opts.x_column = pc1.selectbox(
            "X axis", x_options, key=f"ge_xcol_{file_key}",
            help="Distance along line shows the representative profiles. Position, time or a "
                 "channel plots the individual readings, one colour per line (two channels give "
                 "a cross-plot).",
        )
        opts.y_column = pc2.selectbox(
            "Y axis", y_options, key=f"ge_ycol_{file_key}",
            help="Representative profiles: the mean of all passes of each channel on the common "
                 "distance grid. A channel: its readings after the corrections and filters.",
        )
        if opts.readings and table is not None:
            x_vals = _axis_values(table, opts.x_column)
            y_vals = _axis_values(table, opts.y_column) if opts.y_column != PROFILE_Y else x_vals
            x_ok, y_ok = x_vals[np.isfinite(x_vals)], y_vals[np.isfinite(y_vals)]
            if x_ok.size:
                x_data_min, x_data_max = float(x_ok.min()), float(x_ok.max())
            if y_ok.size and opts.y_column != PROFILE_Y:
                y_data_min, y_data_max = float(y_ok.min()), float(y_ok.max())
        col_sheets, col_style, col_axes = st.columns([2, 1, 2])

        with col_sheets:
            st.markdown(f"**Visible {item_noun(mode)[1]}**")
            selected = st.multiselect(
                "Visible items",
                options=all_sheet_names,
                default=all_sheet_names,
                key=f"ge_sheets_{file_key}",
                label_visibility="collapsed",
            )
            opts.selected_sheets = selected

            st.markdown("**Detail plots**")
            opts.show_traces = st.checkbox(
                "Show individual traces",
                value=True,
                key=f"ge_traces_{file_key}",
            )
            opts.show_envelope = st.checkbox(
                "Show ±1σ envelope",
                value=True,
                key=f"ge_envelope_{file_key}",
            )

        with col_style:
            st.markdown("**Style**")
            opts.show_grid = st.checkbox(
                "Grid",
                value=True,
                key=f"ge_grid_{file_key}",
            )
            opts.line_width = st.slider(
                "Line width",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key=f"ge_lw_{file_key}",
            )
            style_label = st.selectbox(
                "Line style",
                options=list(LINE_STYLES.keys()),
                index=0,
                key=f"ge_ls_{file_key}",
            )
            opts.line_style = LINE_STYLES[style_label]

        with col_axes:
            st.markdown("**X axis range**")
            x_auto = st.checkbox("Automatic", value=True, key=f"ge_xauto_{file_key}")
            if not x_auto:
                xc1, xc2 = st.columns(2)
                x_min = xc1.number_input(
                    "Min", value=x_data_min, key=f"ge_xmin_{file_key}", format="%.2f"
                )
                x_max = xc2.number_input(
                    "Max", value=x_data_max, key=f"ge_xmax_{file_key}", format="%.2f"
                )
                if x_min < x_max:
                    opts.x_lim = (x_min, x_max)
                else:
                    st.caption("X min must be < X max")

            st.markdown("**Y axis range**")
            y_auto = st.checkbox("Automatic", value=True, key=f"ge_yauto_{file_key}")
            if not y_auto:
                yc1, yc2 = st.columns(2)
                y_min = yc1.number_input(
                    "Min", value=y_data_min, key=f"ge_ymin_{file_key}", format="%.4g"
                )
                y_max = yc2.number_input(
                    "Max", value=y_data_max, key=f"ge_ymax_{file_key}", format="%.4g"
                )
                if y_min < y_max:
                    opts.y_lim = (y_min, y_max)
                else:
                    st.caption("Y min must be < Y max")

        st.divider()
        st.markdown("**Labels & title**")
        lc1, lc2, lc3 = st.columns(3)
        opts.plot_title = lc1.text_input(
            "Overview plot title",
            value="",
            placeholder="Leave blank for default",
            key=f"ge_title_{file_key}",
        )
        opts.x_label = lc2.text_input(
            "X axis label",
            value="",
            placeholder="Distance (m)",
            key=f"ge_xlabel_{file_key}",
        )
        opts.y_label = lc3.text_input(
            "Y axis label",
            value="",
            placeholder="Automatic (units from file)",
            key=f"ge_ylabel_{file_key}",
        )

    return opts


# ---------------------------------------------------------------------------
# Rendering of one mode
# ---------------------------------------------------------------------------

def _render_mode_section(
    output_data: dict[str, pd.DataFrame],
    scores: dict[str, dict],
    mode: str,
    file_name: str,
    file_key: str,
    scoring: bool = True,
    markers: list[float] | None = None,
    sensor: emphysics.Sensor | None = None,
    table: pd.DataFrame | None = None,
) -> None:
    """
    Render ranking table (or channel list), graph editor, plots, and downloads for one mode.

    Parameters
    ----------
    output_data : {freq_or_sheet_name: interpolated_df}
    scores      : {freq_or_sheet_name: score_dict}
    mode        : "EC" or "MS"
    file_name   : original uploaded filename (for plot titles)
    file_key    : unique string for Streamlit widget key namespacing
    scoring     : show the frequency ranking and scores (never for AUX channels)
    markers     : event-marker distances drawn on the profiles
    table       : prepared GEM table, for plots of chosen columns (graph editor)
    """
    noun, nouns = item_noun(mode)
    if not output_data:
        st.error(f"No usable {nouns} found.")
        return

    stem = Path(file_name).stem
    show_scores = scoring and mode != "AUX"
    if show_scores:
        _render_ranking(scores)
        if len(output_data) > 1:
            ui_tools.render_frequency_selection(output_data, scores, file_key)
    else:
        _render_channel_list(output_data, scores, mode)
    if mode == "EC":
        ui_tools.render_frequency_info(output_data, sensor or emphysics.GEM2, file_key)
    freqs = [ctr.parse_frequency(name) or 0.0 for name in output_data]
    if mode in ("MS", "I") and max(freqs) >= ui_tools.PERMITTIVITY_FREQUENCY:
        st.caption(ui_tools.PERMITTIVITY_CAPTION)

    # ── Graph editor ─────────────────────────────────────────────────────
    opts = render_graph_editor(output_data, file_key=file_key, table=table, mode=mode)
    readings = opts.readings and table is not None

    def overview() -> plt.Figure:
        if readings:
            return make_readings_figure(
                table, opts, f"{opts.y_column} against {opts.x_column} — {file_name}"
            )
        return make_overview_figure(output_data, scores, mode, file_name, opts, show_scores, markers)

    # ── Overview plot ────────────────────────────────────────────────────
    title = f"{opts.y_column} against {opts.x_column}" if readings else "Representative profiles"
    with st.expander(title, expanded=False):
        if not readings and opts.x_column != PROFILE_X:
            st.caption("Representative profiles are drawn against distance along the line; "
                       "choose a channel for Y to plot against another X.")
        ui_tools.show_figure(overview())

    # ── Per-sheet detail ─────────────────────────────────────────────────
    with st.expander(f"Per-{noun} detail plots", expanded=False):
        visible = opts.selected_sheets if opts.selected_sheets else list(output_data.keys())
        for sheet_name in visible:
            interp_df = output_data.get(sheet_name)
            if interp_df is None:
                continue
            sc = scores[sheet_name]
            if show_scores:
                st.markdown(
                    f"**{sheet_name}** — SNR {sc['score']:.1f} dB · "
                    f"signal σ = {sc['signal_std']:.4g} · noise σ = {sc['mean_std']:.4g} "
                    f"({sc['noise_method']}, {ui_tools.plural(sc['n_traces'], 'trace')})"
                )
            else:
                st.markdown(f"**{sheet_name}** — {ui_tools.plural(sc['n_traces'], 'trace')}")
            ui_tools.show_figure(make_sheet_figure(sheet_name, interp_df, mode, opts, markers))

    # ── Downloads ────────────────────────────────────────────────────────
    st.markdown("#### Downloads")
    download_fig = overview()
    png_bytes = fig_to_png(download_fig)
    plt.close(download_fig)
    with ui_tools.button_row():
        ui_tools.download_button(
            "Interpolated profiles (.xlsx)", build_excel_download(output_data),
            f"{stem}_{mode}_interpolated.xlsx", key=f"dl_xlsx_{file_key}",
        )
        if show_scores:
            ui_tools.download_button(
                "Scores (.csv)", build_scores_csv(scores), f"{stem}_{mode}_scores.csv",
                key=f"dl_csv_{file_key}",
            )
        ui_tools.download_button(
            "Overview plot (.png)", png_bytes, f"{stem}_{mode}_overview.png",
            key=f"dl_png_{file_key}",
        )


def _render_channel_list(
    output_data: dict[str, pd.DataFrame], scores: dict[str, dict], mode: str
) -> None:
    """Channels of one mode without scores: passes and covered distance."""
    st.markdown("#### Channels")
    rows = [
        {
            "Channel": name,
            "Units": ctr.value_label(mode, name),
            "Traces": scores[name]["n_traces"],
            "From (m)": round(float(df.index.min()), 3),
            "To (m)": round(float(df.index.max()), 3),
        }
        for name, df in output_data.items()
    ]
    ui_tools.show_table(pd.DataFrame(rows))


def _render_ranking(scores: dict[str, dict]) -> None:
    """Frequency ranking table and best-frequency metrics."""
    ranking = rank_scores(scores)
    if not ranking:
        st.error("No sheets could be scored.")
        return

    rank_df = pd.DataFrame(
        [
            {
                "Rank": i,
                "Frequency / Sheet": name,
                "SNR (dB)": round(m["score"], 2),
                "Signal (σ)": round(m["signal_std"], 6),
                "Noise (σ)": round(m["mean_std"], 6),
                "Noise method": m["noise_method"],
                "Traces": m["n_traces"],
            }
            for i, (name, m) in enumerate(ranking, 1)
        ]
    )

    col_table, col_best = st.columns([3, 1])
    with col_table:
        st.markdown("#### Frequency ranking")
        styled = rank_df.style.format(
            {"SNR (dB)": "{:.2f}", "Signal (σ)": "{:.4g}", "Noise (σ)": "{:.4g}"}, na_rep="—"
        )
        if rank_df["SNR (dB)"].notna().any():  # all-blank scores have nothing to colour
            styled = styled.background_gradient(subset=["SNR (dB)"], cmap="Blues")
        ui_tools.show_table(styled)
    with col_best:
        best_name, best_metrics = ranking[0]
        st.metric("Best frequency", best_name)
        st.metric(
            "SNR",
            f"{best_metrics['score']:.1f} dB",
            help=(
                "SNR = 10 log10(P_signal / P_noise). P_noise = σ²: the pooled sample variance "
                "across passes (ddof=1) where ≥ 2 passes overlap, or from second differences of "
                "the measured samples (robust MAD) for a single pass. P_signal = variance of the "
                "mean profile minus the noise left in it (σ² / passes averaged). Blank when σ is "
                "zero or undefined, or when no signal rises above the noise."
            ),
        )
        st.metric("Signal (σ)", f"{best_metrics['signal_std']:.4g}")
        st.metric("Noise (σ)", f"{best_metrics['mean_std']:.4g}")


# ---------------------------------------------------------------------------
# 2D contouring (area maps & pseudo-sections)
# ---------------------------------------------------------------------------

def gem_value_column(mode: str, freq_label: str) -> str:
    """Raw GEM column name for a mode and channel label such as '4525Hz'."""
    return gem_io.channel_column(mode, freq_label)


@st.cache_data(show_spinner=False)
def read_raw_table(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """First sheet (XLSX) or the whole table (CSV), unprocessed."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)


def grid_params(contour: ContourSettings) -> dict:
    """Settings that affect the gridded result (display-only fields excluded)."""
    return {
        "coord_mode": contour.coord_mode,
        "method": contour.method,
        "cell_size": contour.cell_size,
        "blank_distance": contour.blank_distance,
        "level": contour.level_lines,
        "smoothing": contour.smoothing,
        "variogram_model": contour.variogram_model,
        "projection": contour.projection,
        "xy_epsg": contour.xy_epsg,
        "options": dict(contour.method_options),
    }


@st.cache_data(show_spinner=False, max_entries=16)
def compute_area_map_cached(
    file_bytes: bytes,
    file_name: str,
    value_col: str,
    prep: pipeline.PrepSettings | None = None,
    **params,
) -> ctr.AreaMapResult:
    """Cached contouring.compute_area_map on the prepared table; *params* from grid_params()."""
    table, _ = prepared_table(file_bytes, file_name, prep)
    return ctr.compute_area_map(table, value_col, **params)


# ---------------------------------------------------------------------------
# Basic / advanced settings, reset and the summary of changed settings
# ---------------------------------------------------------------------------

ADVANCED_KEY = "advanced"
ADVANCED_HELP = (
    "Shows every setting. Basic mode keeps the expert ones (coil geometry, calibration "
    "files, variogram, projection, colour scaling, map filters …) at their defaults, so a hidden "
    "setting never changes the results."
)
ADVANCED_HINT = "More settings with **Show advanced settings** at the top of the sidebar."
RESET_KEY = "settings_reset"        # counts resets; a new count gives every setting a fresh widget


def show_advanced() -> bool:
    """True when the advanced settings are shown (toggle at the top of the sidebar)."""
    return bool(st.session_state.get(ADVANCED_KEY, False))


def wkey(name: str) -> str:
    """
    Widget key of a sidebar setting. After 'Reset all settings' the key gets
    the reset count, so Streamlit makes a new widget at its default value
    (including file uploaders, whose value cannot be set from code).
    """
    n = st.session_state.get(RESET_KEY, 0)
    return f"{name}~{n}" if n else name


def reset_settings() -> None:
    st.session_state[RESET_KEY] = st.session_state.get(RESET_KEY, 0) + 1


def set_setting(name: str, value) -> None:
    """Button callback: sets a sidebar setting (runs before the widgets of the next run)."""
    st.session_state[wkey(name)] = value


def next_steps(
    mode: str, n_items: int, scoring: bool, contour: ContourSettings | None, has_xy: bool,
) -> list[tuple[str, str, str]]:
    """(label, setting, icon) for the views a tab could show next but has switched off."""
    steps = []
    if not scoring and mode in gem_io.FREQUENCY_MODES and n_items > 1:
        steps.append(("Rank the frequencies", "scoring", ":material/leaderboard:"))
    if contour is not None and not contour.area_map and mode in AREA_MAP_MODES and has_xy:
        steps.append(("Map this survey", "ct_area", ":material/map:"))
    if contour is not None and not contour.pseudosection and mode in PSEUDOSECTION_MODES and n_items > 1:
        steps.append(("Show a pseudo-section", "ct_pseudo", ":material/stacked_line_chart:"))
    return steps


def render_step_hint(spacing: float, distance_step: float, key: str, failed: bool = False) -> None:
    """
    Suggests a distance step that suits the reading spacing, with a button that sets it.
    *failed*: no profile could be made, so the hint is shown whatever the step.
    """
    step = ui_tools.suggested_step(spacing)
    if step is None or step == distance_step or not (failed or ui_tools.step_mismatch(distance_step, spacing)):
        return
    why = ("fewer readings than grid points: most profile values are interpolated"
           if distance_step < spacing else "much coarser than the readings: detail is lost")
    if failed:
        why = "too coarse for these lines"
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.info(f"Readings are about **{spacing:.3g} m** apart along the lines, so a distance step of "
            f"**{distance_step:g} m** is {why}. A step of **{step:g} m** suits this file.",
            icon=":material/straighten:")
    c2.button(f"Use {step:g} m", key=f"step_hint_{key}", on_click=set_setting,
              args=("prof_step", step), width="stretch",
              help="Sets Distance step in the sidebar (it applies to every file).")


def render_next_steps(steps: list[tuple[str, str, str]], key: str) -> None:
    """A row of buttons that switch on the next views (the matching sidebar toggles)."""
    if not steps:
        return
    st.markdown("#### Next steps")
    with ui_tools.button_row():
        for label, setting, icon in steps:
            st.button(label, key=f"next_{setting}_{key}", icon=icon, on_click=set_setting,
                      args=(setting, True), help="Switches this on in the sidebar.")


def changed_settings(
    prep: pipeline.PrepSettings, distance_step: float, interp_kind: str
) -> list[str]:
    """Short descriptions of the settings that differ from the defaults and change the results."""
    default = pipeline.PrepSettings()
    out = []
    if distance_step != DEFAULT_DISTANCE_STEP:
        out.append(f"distance step {distance_step:g} m")
    if interp_kind != DEFAULT_INTERP_KIND:
        out.append(f"{interp_kind} interpolation")
    if not prep.drop_flagged:
        out.append("flagged readings kept")
    if prep.exclude_lines:
        out.append(f"{ui_tools.plural(len(prep.exclude_lines), 'line')} left out")
    if prep.distance_method != default.distance_method:
        out.append(f"distance: {gem_io.DISTANCE_METHODS[prep.distance_method].lower()}")
    if prep.sensor.height != default.sensor.height:
        out.append(f"sensor height {prep.sensor.height:g} m")
    if (prep.sensor.separation, prep.sensor.bucking) != (default.sensor.separation,
                                                          default.sensor.bucking):
        out.append("custom coil spacing")
    if prep.recompute_from_iq:
        out.append("EC / MS recomputed from I / Q")
    if prep.viscosity_pair:
        out.append("magnetic-viscosity channels")
    return out + prep.corrections.describe()


def render_settings_status(changed: list[str]) -> None:
    """Sidebar line naming the settings that change the results, with a reset button."""
    if changed:
        st.caption(f"**{ui_tools.plural(len(changed), 'setting')} changed:** " + " · ".join(changed))
    else:
        st.caption("All settings that change the results are at their defaults.")
    st.button("Reset all settings", key="settings_reset_btn", on_click=reset_settings,
              icon=":material/restart_alt:",
              help="Puts every sidebar setting back to its default and clears loaded "
                   "calibration files. Uploaded data files are kept.")


# ---------------------------------------------------------------------------
# Settings files (save / load every sidebar setting)
# ---------------------------------------------------------------------------

SETTINGS_PREFIXES = ("prof_", "prep_", "sen_", "cor_", "ct_")
SETTINGS_NAMES = ("scoring", ADVANCED_KEY)
SETTINGS_SKIPPED = ("cor_offsets", "cor_ref", "ct_freq_", "ct_cv_")   # files, per-file widgets, buttons
SETTINGS_FILE_KEY = "settings_file"
SETTINGS_MESSAGE_KEY = "settings_message"


def is_saved_setting(name: str) -> bool:
    """True for the sidebar settings a settings file holds (not uploads, buttons or per-file choices)."""
    return ((name in SETTINGS_NAMES or name.startswith(SETTINGS_PREFIXES))
            and not name.startswith(SETTINGS_SKIPPED))


def current_settings(state) -> dict:
    """{setting name: value} of the sidebar settings in *state* (st.session_state), JSON-ready."""
    n = state.get(RESET_KEY, 0)
    out = {}
    for key in list(state.keys()):
        name, _, generation = str(key).partition("~")
        if not is_saved_setting(name) or (name != ADVANCED_KEY and int(generation or 0) != n):
            continue
        value = state[key]
        out[name] = list(value) if isinstance(value, tuple) else value
    return dict(sorted(out.items()))


def settings_json(settings: dict) -> bytes:
    return json.dumps({"app": "GEM-2 Analysis Tool", "version": APP_VERSION, "settings": settings},
                      indent=2, ensure_ascii=False).encode()


def parse_settings(data: bytes) -> dict:
    """Settings of a saved file; raises ValueError when it is not one."""
    try:
        doc = json.loads(data.decode("utf-8-sig"))
        settings = doc["settings"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("This is not a settings file saved by the GEM-2 Analysis Tool.") from exc
    if not isinstance(settings, dict):
        raise ValueError("This is not a settings file saved by the GEM-2 Analysis Tool.")
    choices = setting_choices()
    return {k: tuple(v) if isinstance(v, list) else v for k, v in settings.items()
            if is_saved_setting(k) and (k not in choices or v in choices[k])}


def setting_choices() -> dict[str, tuple]:
    """Options of the sidebar choices: a saved value no longer offered keeps the default."""
    return {
        "prof_interp": tuple(INTERP_METHODS),
        "ct_method": tuple(ctr.METHODS),
    }


def apply_settings(state, settings: dict) -> None:
    """
    Puts *settings* into *state* (st.session_state) under a new reset count, so
    every sidebar widget starts afresh: settings missing from the file are defaults.
    """
    state[RESET_KEY] = n = state.get(RESET_KEY, 0) + 1
    for name, value in settings.items():
        state[name if name == ADVANCED_KEY else f"{name}~{n}"] = value


def load_settings_file() -> None:
    """Uploader callback: replaces the sidebar settings by those of the uploaded file."""
    upload = st.session_state.get(SETTINGS_FILE_KEY)
    if upload is None:
        return
    try:
        settings = parse_settings(upload.getvalue())
    except ValueError as exc:
        st.session_state[SETTINGS_MESSAGE_KEY] = ("error", str(exc))
        return
    apply_settings(st.session_state, settings)
    st.session_state[SETTINGS_MESSAGE_KEY] = (
        "success", f"Loaded {ui_tools.plural(len(settings), 'setting')} from {upload.name}.")


def render_settings_file() -> None:
    """Sidebar expander to save the settings to a small file or load them back."""
    with st.expander("Save or load settings", expanded=False):
        st.caption("A settings file repeats a run later or on another computer, or shares it "
                   "with a colleague. Calibration files and uploaded data are not included.")
        ui_tools.download_button(
            "Settings (.json)", settings_json(current_settings(st.session_state)),
            "gem2_settings.json", key="settings_save",
        )
        st.file_uploader("Load settings (.json)", type=["json"], key=SETTINGS_FILE_KEY,
                         on_change=load_settings_file)
        message = st.session_state.pop(SETTINGS_MESSAGE_KEY, None)
        if message:
            (st.success if message[0] == "success" else st.error)(message[1])


def render_data_sidebar() -> pipeline.PrepSettings:
    """Sidebar groups for GEM files: data preparation, sensor geometry, corrections & filters."""
    st.divider()
    st.subheader("GEM files")
    with st.expander("GEM data preparation", expanded=False):
        st.caption(PREP_APPLIES_TO)
        drop_flagged = True
        if show_advanced():
            drop_flagged = st.checkbox(
                "Drop readings with a Status flag", value=True, key=wkey("prep_status"),
                help="The GEM-2 sets Status to a non-zero value when a reading has a "
                     "problem such as ADC overload.",
            )
        exclude = st.text_input(
            "Lines to leave out (comma-separated)", key=wkey("prep_exclude"),
            help="For example calibration or test lines recorded in the same file.",
        )
        method = st.selectbox(
            "Distance along line", list(gem_io.DISTANCE_METHODS),
            format_func=gem_io.DISTANCE_METHODS.get, key=wkey("prep_distance"), help=DISTANCE_HELP,
        )
        spacing = 1.0
        if method in ("sample", "markers"):
            spacing = st.number_input(
                "Reading spacing (m)" if method == "sample" else "Marker spacing (m)",
                min_value=0.001, value=1.0, step=0.1, format="%.3f", key=wkey("prep_spacing"),
                help=SCIENCE_HELP["spacing"],
            )
    sensor, recompute, viscosity_pair = render_sensor_sidebar()
    return pipeline.PrepSettings(
        drop_flagged=drop_flagged, exclude_lines=_parse_labels(exclude),
        distance_method=method, distance_spacing=float(spacing),
        corrections=render_corrections_sidebar(),
        sensor=sensor, recompute_from_iq=recompute, viscosity_pair=viscosity_pair,
    )


def render_sensor_sidebar() -> tuple[emphysics.Sensor, bool, tuple[float, float] | None]:
    """Coil geometry and height for the physics tools; EC/MS recomputation; viscosity pair."""
    with st.expander("Sensor geometry", expanded=False):
        st.caption("Used by: forward model, frequency information, multi-height calibration, "
                   "layered-earth inversion and the EC / MS "
                   "recomputation below.")
        height = st.number_input("Sensor height (m)", 0.0, 50.0, 1.0, step=0.05, key=wkey("sen_h"),
                                 help=SCIENCE_HELP["height"])
        separation, bucking, recompute, visc_text = 1.66, 1.035, False, ""
        if show_advanced():
            st.caption("GEM-2 coil spacings (Won et al., 1996). Check them against your sensor's "
                       ".gem file.")
            separation = st.number_input("Tx–Rx separation (m)", 0.1, 10.0, 1.66, step=0.01,
                                         key=wkey("sen_sep"), help=SCIENCE_HELP["separation"])
            bucking = st.number_input("Tx–bucking coil (m, 0 = none)", 0.0, 10.0, 1.035, step=0.005,
                                      format="%.3f", key=wkey("sen_buck"), help=SCIENCE_HELP["bucking"])
            recompute = st.checkbox(
                "Recompute EC / MS from I / Q", key=wkey("sen_recompute"),
                help="Half-space conversion of each reading (Huang & Won, 2000) with this geometry "
                     "and height. Overwrites exported EC / MS columns; adds them to I/Q-only files.",
            )
            visc_text = st.text_input(
                "Magnetic viscosity from frequencies (low, high Hz)", key=wkey("sen_visc"),
                help="Two frequencies with I_ / Q_ columns, e.g. '1525, 5325'. Adds EC from the "
                     "quadrature difference and the quadrature susceptibility κ″ to the AUX "
                     "channels (Simon et al., 2015). Uncalibrated Q offsets shift κ″, so compare "
                     "contrasts unless Q is calibrated.",
            )
        else:
            st.caption(ADVANCED_HINT)
    try:
        sensor = emphysics.Sensor(float(separation), float(bucking) or None, float(height))
    except ValueError as exc:
        st.error(f"{exc} Using the GEM-2 coil spacings.")
        sensor = emphysics.Sensor(height=float(height))
    pair = _parse_pair(visc_text)
    if visc_text.strip() and pair is None:
        st.warning("Magnetic viscosity needs two different frequencies in Hz, e.g. '1525, 5325'.")
    return sensor, recompute, pair


def _parse_pair(text: str) -> tuple[float, float] | None:
    """'1525, 5325' -> (1525.0, 5325.0); None if blank or not two distinct numbers."""
    try:
        values = tuple(float(p) for p in text.replace(",", " ").split())
    except ValueError:
        return None
    return values if len(values) == 2 and values[0] != values[1] else None


def _parse_labels(text: str) -> tuple[str, ...]:
    """'B1, B2 ,7' -> ('B1', 'B2', '7')."""
    return tuple(p.strip() for p in text.split(",") if p.strip())


def render_corrections_sidebar() -> corrections.CorrectionSettings:
    """Corrections & filters applied to GEM tables before profiles and maps."""
    adv = show_advanced()
    window, threshold = 5, 4.0
    drift_model, temp_col, alt_col, alt_ref = "piecewise", "", "", None
    offsets_file = ref_file = None
    soil_t, ref_radius, ref_method = 0.0, 2.0, "regression"
    with st.expander("Corrections & filters", expanded=False):
        st.caption(PREP_APPLIES_TO + " Multi-height calibration uses the readings as recorded.")
        st.markdown("**Filters**")
        despike = st.checkbox("Despike (running median)", key=wkey("cor_despike"), help=SCIENCE_HELP["despike"])
        if despike and adv:
            window = st.slider("Despike window (readings)", 3, 21, 5, step=2, key=wkey("cor_dwin"),
                               help=SCIENCE_HELP["despike_window"])
            threshold = st.number_input("Despike threshold (× noise σ)", 1.0, 20.0, 4.0,
                                        step=0.5, key=wkey("cor_dthr"), help=SCIENCE_HELP["despike_threshold"])
        clip = st.checkbox("Clip to percentiles", key=wkey("cor_clip"), help=SCIENCE_HELP["clip"])
        clip_range = None
        if clip:
            lo, hi = st.slider("Keep percentiles", 0.0, 100.0, (1.0, 99.0), step=0.5, key=wkey("cor_crange"),
                               help=SCIENCE_HELP["clip_range"])
            clip_range = (float(lo), float(hi))

        st.markdown("**Drift and environment**")
        drift_text = st.text_input("Base-station lines (comma-separated Line labels)",
                                   key=wkey("cor_drift"),
                                   help="Lines recorded at a fixed base station during the "
                                        "survey. Needs a Time column. They are removed after "
                                        "the correction.")
        if adv:
            drift_model = st.selectbox("Drift model", ["piecewise", "linear"], key=wkey("cor_dmodel"),
                                       help=SCIENCE_HELP["drift_model"])
            temp_col = st.text_input("Temperature column (needs ≥ 3 base-station lines)",
                                     key=wkey("cor_temp"), help=SCIENCE_HELP["temperature"]).strip()
            alt_col = st.text_input("Sensor-height column (e.g. drone altitude above ground)",
                                    key=wkey("cor_alt"), help=SCIENCE_HELP["altitude"]).strip()
            if alt_col:
                alt_ref_value = st.number_input("Reference height (m, 0 = median)", 0.0, 100.0, 0.0,
                                                step=0.1, key=wkey("cor_altref"),
                                                help=SCIENCE_HELP["altitude_ref"])
                alt_ref = float(alt_ref_value) or None

        if adv:
            st.markdown("**Calibration**")
            offsets_file = st.file_uploader(
                "Calibration offsets (.csv)", type=["csv"], key=wkey("cor_offsets"),
                help="Columns 'column' and 'offset'; each offset is subtracted from that column. "
                     "Produced by 'Multi-height calibration' under a file. I / Q offsets reach "
                     "EC / MS only with 'Recompute EC / MS from I / Q' (Sensor geometry).",
            )
            ref_file = st.file_uploader("Reference values (.csv)", type=["csv"], key=wkey("cor_ref"),
                                        help="Same coordinate columns as the survey (X/Y or "
                                             "Lat/Lon) plus one column per channel to calibrate, "
                                             "named like the survey column, e.g. EC1525Hz[mS/m].")
            if ref_file is not None:
                ref_radius = st.number_input("Matching radius (m)", 0.1, 100.0, 2.0, key=wkey("cor_rrad"),
                                             help=SCIENCE_HELP["ref_radius"])
                ref_method = st.selectbox("Calibration fit", ["regression", "moments"], key=wkey("cor_rmeth"),
                                          help=SCIENCE_HELP["ref_method"])
            soil_t = st.number_input("Soil temperature for EC at 25 °C (°C, 0 = off)", 0.0, 50.0, 0.0,
                                     key=wkey("cor_soilt"), help=SCIENCE_HELP["ec25"])

        st.markdown("**Positioning**")
        lag = st.number_input("GPS lag (s)", -5.0, 5.0, 0.0, step=0.1, key=wkey("cor_lag"),
                              help="Positive: readings were logged after the position. Use "
                                   "'Estimate GPS lag' under a file to find it.")
        if not adv:
            st.caption(ADVANCED_HINT)

    return corrections.CorrectionSettings(
        despike=despike, despike_window=int(window), despike_threshold=float(threshold),
        clip_percentiles=clip_range,
        altitude_column=alt_col, altitude_reference=alt_ref,
        temperature_column=temp_col,
        drift_lines=_parse_labels(drift_text), drift_model=drift_model,
        iq_offsets=offsets_file.getvalue() if offsets_file is not None else None,
        reference=ref_file.getvalue() if ref_file is not None else None,
        reference_radius=float(ref_radius), reference_method=ref_method,
        soil_temperature=float(soil_t) or None,
        lag_seconds=float(lag),
    )


def render_lag_estimate(
    file_bytes: bytes, file_name: str, file_key: str, prep: pipeline.PrepSettings
) -> None:
    """Expander that estimates the GPS time lag from crossings of neighbouring lines."""
    with st.expander("Estimate GPS lag", expanded=False):
        no_lag = replace(prep, recompute_from_iq=False,
                         corrections=replace(prep.corrections, lag_seconds=0.0))
        try:
            table, _ = prepared_table(file_bytes, file_name, no_lag)
        except ValueError as exc:
            st.info(str(exc))
            return
        columns = corrections.channel_columns(table)
        if not columns:
            return
        column = st.selectbox("Channel", columns, key=f"lag_col_{file_key}", help=SCIENCE_HELP["lag_channel"])
        if st.button("Estimate", key=f"lag_btn_{file_key}"):
            try:
                with st.spinner("Searching lags from −2 s to +2 s…"):
                    lag, cost = corrections.estimate_lag(table, column)
            except (ValueError, ctr.ContouringError) as exc:
                st.info(str(exc))
                return
            st.write(f"Best lag **{lag:+.1f} s** — enter it as 'GPS lag' in the sidebar.")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(corrections.LAG_SEARCH, cost, marker=".")
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("Mean |difference|")
            fig.tight_layout()
            ui_tools.show_figure(fig, stretch=False)


def render_contouring_sidebar() -> ContourSettings:
    """Sidebar controls for 2D contouring; call inside `with st.sidebar`."""
    st.divider()
    st.subheader("2D contouring")
    area = st.toggle("Area map (plan view)", value=False, key=wkey("ct_area"), help=SCIENCE_HELP["area_map"])
    pseudo = st.toggle("Pseudo-section", value=False, key=wkey("ct_pseudo"), help=SCIENCE_HELP["pseudosection"])
    if not area:
        if not pseudo:
            return ContourSettings()
        with st.expander("Map colours", expanded=False):
            st.caption("Used by: pseudo-sections.")
            colours = render_colour_controls()
        return ContourSettings(pseudosection=True, n_levels=colours.n_levels, colours=colours)

    with st.expander("Area map gridding", expanded=False):
        st.caption("Used by: area maps and combined surveys.")
        grid = _render_gridding_controls()
    with st.expander("Map colours", expanded=False):
        st.caption("Used by: area maps, combined surveys"
                   + (" and pseudo-sections." if pseudo else "."))
        colours = render_colour_controls()
        show_points = st.checkbox("Show data points", value=True, key=wkey("ct_points"),
                                  help="Marks the readings used for gridding (one per grid cell).")
    return ContourSettings(
        area_map=True, pseudosection=pseudo, n_levels=colours.n_levels, colours=colours,
        show_points=show_points, **grid,
    )


def _render_gridding_controls() -> dict:
    """Area-map gridding inputs; returns the matching ContourSettings fields."""
    adv = show_advanced()
    coord_mode, smoothing, variogram_model = "auto", 0.0, "spherical"
    blank, projection, xy_epsg = 0.0, "local", 0
    if adv:
        coord_mode = st.selectbox(
            "Coordinates", ["auto", "metres", "degrees"],
            format_func=str.capitalize, key=wkey("ct_coords"),
            help="Auto: Lat/Lon columns are degrees; X/Y are degrees only if they "
                 "look like lon/lat spanning < 0.05°.",
        )
    method = st.selectbox(
        "Gridding method", list(ctr.METHODS), format_func=ctr.METHODS.get, key=wkey("ct_method"),
        help=SCIENCE_HELP["gridding"],
    )
    if adv and method == "spline":
        smoothing = st.number_input(
            "Spline smoothing", min_value=0.0, value=0.0, step=0.1, key=wkey("ct_smooth"),
            help="0 = exact interpolation; larger values trade fit for smoothness.",
        )
    if adv and method == "kriging":
        variogram_model = st.selectbox(
            "Variogram model", ctr.VARIOGRAM_MODELS, key=wkey("ct_vario"), help=SCIENCE_HELP["variogram"],
        )
    method_options = render_method_options(method)
    cell = st.number_input(
        "Cell size (m, 0 = auto)", min_value=0.0, value=0.0, step=0.1,
        format="%.2f", key=wkey("ct_cell"), help=SCIENCE_HELP["cell"],
    )
    level = st.checkbox(
        "Line levelling (per-line median)", value=False, key=wkey("ct_level"), help=LEVELLING_HELP,
    )
    if adv:
        blank = st.number_input(
            "Blanking distance (m, 0 = auto)", min_value=0.0, value=0.0, step=0.5,
            format="%.2f", key=wkey("ct_blank"),
            help="Grid nodes farther than this from any data point are left blank. "
                 "Auto keeps the gaps between survey lines filled: the larger of "
                 "2 × median point spacing and 1.5 × the 90th-percentile distance "
                 "from grid nodes inside the survey to the nearest data point.",
        )
        projection = st.selectbox(
            "Projection of Lat/Lon", list(ctr.PROJECTIONS), format_func=ctr.PROJECTIONS.get,
            key=wkey("ct_proj"), help="UTM gives georeferenced .asc / GeoTIFF exports.",
        )
        xy_epsg = st.number_input(
            "EPSG code of X/Y (0 = unknown)", 0, 99999, 0, key=wkey("ct_epsg"),
            help="For files whose X/Y are already projected, e.g. 32634 for UTM 34N.",
        )
    else:
        st.caption(ADVANCED_HINT)
    xy_epsg = int(xy_epsg) or None
    problem = gridtools.epsg_problem(xy_epsg) if xy_epsg else None
    if problem:
        st.warning(problem)
        xy_epsg = None
    return dict(
        coord_mode=coord_mode,
        method=method,
        smoothing=float(smoothing),
        variogram_model=variogram_model,
        cell_size=float(cell) or None,
        blank_distance=float(blank) or None,
        level_lines=level,
        projection=projection,
        xy_epsg=xy_epsg,
        method_options=method_options,
    )


def render_method_options(method: str) -> tuple:
    """
    Sidebar inputs for the settings of *method* (advanced settings); returns
    (name, value) pairs. In basic mode the settings keep contouring.METHOD_DEFAULTS.
    """
    defaults = ctr.METHOD_DEFAULTS.get(method, {})
    opts = {}
    if not show_advanced():
        return tuple(opts.items())
    if method == "idw":
        opts["power"] = st.number_input("Power", 0.5, 10.0, defaults["power"], step=0.5,
                                        key=wkey("ct_idw_power"), help=SCIENCE_HELP["idw_power"])
        opts["delta"] = st.number_input("Smoothing distance (m)", 0.0, value=defaults["delta"],
                                        step=0.1, key=wkey("ct_idw_delta"),
                                        help=SCIENCE_HELP["idw_smoothing"])
    elif method == "min_curvature":
        opts["tension"] = st.slider("Tension", 0.0, 0.95, defaults["tension"], step=0.05,
                                    key=wkey("ct_tension"), help=SCIENCE_HELP["min_curvature_tension"])
    return tuple(opts.items())


def render_colour_controls() -> ctr.ColourStyle:
    """Sidebar colour settings shared by area maps and pseudo-sections."""
    adv = show_advanced()
    reverse, display, scale, lines = False, "filled", "linear", False
    cmap = st.selectbox("Colour map", list(ctr.COLOUR_MAPS), format_func=ctr.COLOUR_MAPS.get,
                        key=wkey("ct_cmap"), help=SCIENCE_HELP["colour_map"])
    if adv:
        reverse = st.checkbox("Reverse colours", value=False, key=wkey("ct_cmap_reverse"))
        display = st.selectbox("Display", list(ctr.MAP_DISPLAYS), format_func=ctr.MAP_DISPLAYS.get,
                               key=wkey("ct_display"), help=SCIENCE_HELP["map_display"])
        scale = st.selectbox("Colour scale", list(ctr.COLOUR_SCALES),
                             format_func=ctr.COLOUR_SCALES.get, key=wkey("ct_scale"),
                             help=SCIENCE_HELP["colour_scale"])
    range_mode = st.selectbox("Colour range", list(ctr.COLOUR_RANGES),
                              format_func=ctr.COLOUR_RANGES.get, key=wkey("ct_range"),
                              help=SCIENCE_HELP["colour_range"])
    percentiles, vmin, vmax = (2.0, 98.0), None, None
    if range_mode == "percentile":
        percentiles = st.slider("Percentiles", 0.0, 100.0, (2.0, 98.0), step=0.5,
                                key=wkey("ct_percentiles"))
    elif range_mode == "fixed":
        vmin = st.number_input("Colour minimum (map units)", value=0.0, key=wkey("ct_vmin"))
        vmax = st.number_input("Colour maximum (map units)", value=100.0, key=wkey("ct_vmax"))
    n_levels = st.slider("Contour levels", 5, 50, 20, key=wkey("ct_levels"), help=SCIENCE_HELP["levels"])
    if adv:
        lines = st.checkbox("Contour lines", value=False, key=wkey("ct_lines"),
                            help=SCIENCE_HELP["colour_lines"])
    return ctr.ColourStyle(
        cmap=cmap, reverse=reverse, range_mode=range_mode,
        percentiles=(float(percentiles[0]), float(percentiles[1])),
        vmin=None if vmin is None else float(vmin), vmax=None if vmax is None else float(vmax),
        scale=scale, display=display, contour_lines=lines, n_levels=int(n_levels),
    )


def _render_pseudosection(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    table: pd.DataFrame | None = None,
    sensor: emphysics.Sensor | None = None,
) -> None:
    st.markdown("#### Pseudo-section")
    traces = sorted({c for df in output_data.values() for c in df.columns})
    columns = ui_tools.transect_columns(traces, table, f"ps_track_{file_key}")
    profiles = {
        name: (df if columns is None else df[[c for c in columns if c in df.columns]]).mean(axis=1, skipna=True)
        for name, df in output_data.items()
    }
    try:
        ps = ctr.build_pseudosection(profiles)
        fig = ctr.make_pseudosection_figure(
            ps, ctr.value_label(mode),
            f"Pseudo-section [{mode}] — {file_name}", contour.n_levels, contour.colours,
        )
    except ctr.ContouringError as exc:
        st.info(f"{exc} {ui_tools.doc_link('mapping', 'pseudo-section-distance--frequency')}")
        return
    except Exception as exc:  # last resort: never show a traceback
        st.error(f"Pseudo-section failed: {exc}")
        return
    png = fig_to_png(fig)
    ui_tools.show_figure(fig)
    st.caption(PSEUDOSECTION_CAPTION)
    if mode == "EC":
        depths = ui_tools.depth_caption(profiles, sensor or emphysics.GEM2)
        if depths:
            st.caption(depths)
    ui_tools.download_button(
        "Pseudo-section (.png)", png, f"{Path(file_name).stem}_{mode}_pseudosection.png",
        key=f"dl_ps_{file_key}",
    )


def _render_area_map(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    file_bytes: bytes,
    prep: pipeline.PrepSettings | None = None,
) -> None:
    st.markdown("#### Area map")

    freq = st.selectbox("Channel", list(output_data.keys()), key=f"ct_freq_{file_key}",
                        help="Channel to grid and map. Gridding settings are in the sidebar "
                             "under 2D contouring.")
    label = ctr.value_label(mode, freq)
    try:
        with st.spinner("Gridding…"):
            result = compute_area_map_cached(
                file_bytes, file_name, gem_value_column(mode, freq), prep,
                **grid_params(contour),
            )
        result, processing = _map_processing(result, file_key)
        fig = ctr.make_area_map_figure(
            result, label, f"{freq} [{mode}] — {ctr.method_title(result)}",
            contour.n_levels, show_points=contour.show_points, style=contour.colours,
        )
    except ctr.GridTooLargeError as exc:
        st.warning(str(exc))
        return
    except ctr.KrigingError as exc:
        st.error(str(exc))
        return
    except ctr.ContouringError as exc:
        st.info(f"{exc} {ui_tools.doc_link('troubleshooting', 'no-area-map')}")
        return
    except Exception as exc:  # last resort: never show a traceback
        st.error(f"Area map failed: {exc}")
        return
    png = fig_to_png(fig)
    ui_tools.show_figure(fig)
    for msg in processing:
        st.caption(msg)

    details = (
        f"{result.n_raw:,} readings → {len(result.bv):,} block medians · "
        f"cell {result.spec.cell:g} m · blanking {result.blank_distance:.2f} m"
    )
    if result.variogram is not None:
        vf = result.variogram
        details += (
            f" · variogram {vf.model}: partial sill {vf.psill:.4g}, "
            f"range {vf.range:.3g} m, nugget {vf.nugget:.4g}"
        )
    st.caption(details)

    if st.button("Cross-validate (5-fold)", key=f"ct_cv_{file_key}"):
        with st.spinner("Cross-validating…"):
            try:
                cv = ctr.cross_validate(
                    result.bx, result.by, result.bv, result.method,
                    contour.smoothing, result.variogram,
                    options=result.options, spec=result.spec,
                    groups=cv_groups(result, file_bytes, file_name, prep),
                )
            except ctr.ContouringError as exc:
                st.error(str(exc))
            else:
                folds = ("whole transects held out, so each is estimated from its neighbours"
                         if cv["folds"] == "groups" else
                         "random folds, optimistic for densely sampled lines")
                st.write(
                    f"RMSE **{cv['rmse']:.4g}**, MAE **{cv['mae']:.4g}** "
                    f"({label}, {cv['n']:,} held-out points; {folds})"
                )

    _render_map_statistics(result, label, file_key)
    stem = Path(file_name).stem
    _render_map_downloads(result, png, f"{stem}_{mode}_{freq}_{result.method}", file_key)


def cv_groups(
    result: ctr.AreaMapResult, file_bytes: bytes, file_name: str, prep: pipeline.PrepSettings | None
) -> np.ndarray | None:
    """
    Transect of each block point for cross-validation: its line, with repeat
    passes of one transect (gem_io.track_groups) merged so a held-out line
    is not estimated from its own repeat pass. None when lines are unknown.
    """
    if result.bline is None:
        return None
    try:
        tracks = gem_io.track_groups(prepared_table(file_bytes, file_name, prep)[0]) or {}
    except (ValueError, KeyError):
        tracks = {}
    return np.array([tracks.get(line, f"line {line}") for line in result.bline], dtype=object)


def _map_processing(result: ctr.AreaMapResult, key: str) -> tuple[ctr.AreaMapResult, list[str]]:
    """Optional grid filter, chosen in an expander (advanced settings)."""
    if not show_advanced():
        return result, []
    with st.expander("Map processing (filters)", expanded=False):
        kind = st.selectbox("Filter", gridtools.MAP_FILTERS, key=f"mp_kind_{key}",
                            help=SCIENCE_HELP["map_filter"])
        size = st.slider("Window (cells)", 3, 51, 15 if kind == "high-pass" else 3, step=2,
                         key=f"mp_size_{key}", help=SCIENCE_HELP["map_window"])
        threshold = 4.0
        if kind == "despike":
            threshold = st.number_input("Despike threshold (× robust σ)", 1.0, 20.0, 4.0,
                                        key=f"mp_thr_{key}", help=SCIENCE_HELP["map_despike"])
    if kind == "none":
        return result, []
    z, msgs = gridtools.process_map(result.z, kind, int(size), float(threshold))
    return replace(result, z=z), msgs


def _render_map_statistics(result: ctr.AreaMapResult, label: str, key: str) -> None:
    with st.expander("Histogram & statistics", expanded=False):
        try:
            stats = gridtools.summary_stats(result.z)
        except ValueError:
            st.info("No values to summarise.")
            return
        ui_tools.show_table(pd.DataFrame([stats]).round(4))
        values = result.z[np.isfinite(result.z)]
        fig, ax = plt.subplots(figsize=(6, 2.5))
        flat = np.ptp(values) <= 1e-9 * max(float(np.abs(values).max()), 1e-12)  # round-off only
        ax.hist(values, bins=1 if flat else 50, color="C0")
        ax.set_xlabel(label)
        ax.set_ylabel("Grid nodes")
        fig.tight_layout()
        ui_tools.show_figure(fig, stretch=False)


def _render_map_downloads(result: ctr.AreaMapResult, png: bytes, base: str, key: str) -> None:
    files = [
        ("Area map (.png)", png, "png"),
        ("Grid (.csv)", ctr.grid_to_csv(result), "csv"),
        ("Grid (.asc)", ctr.grid_to_asc(result), "asc"),
        ("GeoTIFF (.tif)", gridtools.geotiff_bytes(result.z, result.spec.x0, result.spec.y0,
                                                   result.spec.cell, result.epsg), "tif"),
    ]
    if result.epsg:
        files.append(("Projection (.prj)", gridtools.prj_wkt(result.epsg), "prj"))
    with ui_tools.button_row():
        for label, data, ext in files:
            ui_tools.download_button(label, data, f"{base}.{ext}", key=f"dl_am_{ext}_{key}")
    if result.origin is not None:
        st.warning(
            "Input was in degrees and projected to local metres: the .asc and GeoTIFF are not "
            "georeferenced. Choose UTM under 'Projection of Lat/Lon' (advanced settings), or use "
            "the lon/lat columns of the CSV."
        )


def render_contouring(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    file_bytes: bytes,
    prep: pipeline.PrepSettings | None = None,
) -> None:
    """Render the enabled 2D contouring sections for one file and mode."""
    if not output_data:
        return
    if contour.pseudosection and mode in PSEUDOSECTION_MODES:
        try:
            table, _ = prepared_table(file_bytes, file_name, prep)
        except ValueError:   # preparation problems are already listed in the warnings
            table = None
        _render_pseudosection(output_data, mode, file_name, file_key, contour, table,
                              (prep or pipeline.PrepSettings()).sensor)
    if contour.area_map and mode in AREA_MAP_MODES:
        _render_area_map(output_data, mode, file_name, file_key, contour, file_bytes, prep)


@st.cache_data(show_spinner=False, max_entries=8)
def _merged_map_cached(
    files: tuple[tuple[bytes, str], ...], prep: pipeline.PrepSettings, value_col: str,
    tolerance: float, match: bool, **params,
) -> tuple[ctr.AreaMapResult, list[float]]:
    tables = [prepared_table(b, n, prep)[0] for b, n in files]
    merged, offsets = gridtools.merge_tables(tables, value_col, tolerance, match)
    return ctr.compute_area_map(merged, value_col, **params), offsets


@st.cache_data(show_spinner=False, max_entries=8)
def _difference_map_cached(
    file_a: tuple[bytes, str], file_b: tuple[bytes, str], prep: pipeline.PrepSettings,
    value_col: str, **params,
) -> ctr.AreaMapResult:
    a = prepared_table(*file_a, prep)[0]
    b = prepared_table(*file_b, prep)[0]
    return ctr.compute_difference_map(a, b, value_col, **params)


def render_combined_maps(
    gem_files: list[tuple[bytes, str]], contour: ContourSettings, prep: pipeline.PrepSettings
) -> None:
    """Merged (edge-matched) area map of several surveys, or a time-lapse difference of two."""
    st.divider()
    st.subheader("Combined surveys")
    try:
        tables = [prepared_table(b, n, prep)[0] for b, n in gem_files]
    except ValueError as exc:
        st.info(str(exc))
        return
    common = set.intersection(*[set(corrections.channel_columns(t)) for t in tables])
    found = gem_io.find_channels(common)
    channels = {col: (m, label) for m in AREA_MAP_MODES for label, col in found[m].items()}
    if not channels:
        st.info("The files share no EC, MS or AUX channel.")
        return
    col = st.selectbox("Channel", sorted(channels), key="cm_col")
    mode, label = channels[col]
    unit = ctr.value_label(mode, label)
    names = [n for _, n in gem_files]
    kind = st.radio("Map", ["Merged (edge-matched)", "Difference (B − A)"], horizontal=True,
                    key="cm_kind", help=SCIENCE_HELP["combined"])
    try:
        if kind.startswith("Merged"):
            chosen = st.multiselect("Surveys", names, default=names, key="cm_files")
            tol = st.number_input("Edge-matching distance (m)", 0.01, 100.0, 1.0, key="cm_tol",
                                  help=SCIENCE_HELP["edge_distance"])
            match = st.checkbox("Edge-match levels", True, key="cm_match", help=SCIENCE_HELP["edge_match"])
            if len(chosen) < 2:
                st.info("Choose at least two surveys.")
                return
            chosen = [n for n in names if n in chosen]        # upload order, as merged
            files = tuple(f for f in gem_files if f[1] in chosen)
            with st.spinner("Gridding merged surveys…"):
                result, offsets = _merged_map_cached(files, prep, col, float(tol), match,
                                                     **grid_params(contour))
            st.dataframe(pd.DataFrame({"Survey": chosen, "Offset added": offsets}).round(4),
                         hide_index=True, width="content")
            fig = ctr.make_area_map_figure(result, unit,
                                           f"{col} — merged", contour.n_levels,
                                           show_points=contour.show_points,
                                           style=contour.colours)
            base = f"merged_{label}"
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("A (earlier)", names, index=0, key="cm_a")
            b = c2.selectbox("B (later)", names, index=1, key="cm_b")
            if a == b:
                st.info("Choose two different surveys.")
                return
            fa = next(f for f in gem_files if f[1] == a)
            fb = next(f for f in gem_files if f[1] == b)
            with st.spinner("Gridding both surveys on one grid…"):
                result = _difference_map_cached(fa, fb, prep, col, **grid_params(contour))
            fig = ctr.make_area_map_figure(result, f"Δ {unit}", f"{col}: {b} − {a}",
                                           contour.n_levels, show_points=False, symmetric=True,
                                           style=replace(contour.colours, cmap="RdBu_r",
                                                         reverse=False))
            base = f"difference_{label}"
    except ctr.ContouringError as exc:
        st.info(f"{exc} {ui_tools.doc_link('mapping', 'combined-surveys')}")
        return
    png = fig_to_png(fig)
    ui_tools.show_figure(fig)
    _render_map_downloads(result, png, base, "combined")


# ---------------------------------------------------------------------------
# Per-format rendering dispatchers
# ---------------------------------------------------------------------------

def unique_file_key(stem: str, taken: set[str], suffixes: tuple[str, ...] = ()) -> str:
    """
    Widget key for one upload: the file stem, or stem_2, stem_3, ... when it
    (or stem_<suffix> for the given suffixes, e.g. the GEM modes) is taken.
    Two uploads named site.csv and site.xlsx get site and site_2.
    """
    key, n = stem, 1
    while key in taken or any(f"{key}_{s}" in taken for s in suffixes):
        n += 1
        key = f"{stem}_{n}"
    taken.add(key)
    taken.update(f"{key}_{s}" for s in suffixes)
    return key


def render_file_warnings(warnings: list[str]) -> None:
    """Collapsed list of the reading and processing messages of one file."""
    if warnings:
        label = f"{ui_tools.plural(len(warnings), 'message')} about this file"
        with st.expander(label, expanded=False, icon=":material/warning:"):
            st.caption("Every step the app took with this file and anything it skipped. "
                       + ui_tools.doc_link("troubleshooting", text="Troubleshooting"))
            for w in warnings:
                st.warning(w)


def render_gem_results(
    file_bytes: bytes,
    file_name: str,
    distance_step: float,
    interp_kind: str,
    contour: ContourSettings | None = None,
    scoring: bool = True,
    prep: pipeline.PrepSettings | None = None,
    file_key: str | None = None,
) -> None:
    """Process and render a GEM instrument file (CSV or XLSX). file_key keeps widget keys unique."""
    stem = file_key or Path(file_name).stem

    with st.spinner("Processing GEM file…"):
        output_data, scores, warnings = process_gem_file(
            file_bytes, file_name,
            distance_step=distance_step,
            interp_kind=interp_kind,
            prep=prep,
        )

    render_file_warnings(warnings)

    try:
        table, _ = prepared_table(file_bytes, file_name, prep)
        markers = gem_io.marker_distances(table, table[pipeline.DISTANCE_COL].to_numpy())
    except (ValueError, KeyError):  # preparation problems are already listed in the warnings
        table, markers = None, []
    summary = ui_tools.data_summary(table) if table is not None else None

    # Determine which modes have data
    available_modes = [m for m in gem_io.GEM_MODES if output_data.get(m)]

    if not available_modes:
        st.error("No usable frequencies found in this GEM file. "
                 + ui_tools.doc_link("troubleshooting", "no-usable-frequencies-or-sheets-found",
                                     "What to check"))
        if summary is not None:
            render_step_hint(summary.spacing, distance_step, stem, failed=True)
        return

    note = "GEM-2 export — every channel in the file is shown below, one tab per mode."
    if summary is not None:
        ui_tools.render_data_summary(summary, note)
        render_step_hint(summary.spacing, distance_step, stem)
    else:
        st.caption(note)
    prep = prep or pipeline.PrepSettings()
    tabs = st.tabs([
        f"{m} ({ui_tools.plural(len(output_data[m]), *item_noun(m))})"
        for m in available_modes
    ])

    for tab, mode_key in zip(tabs, available_modes):
        with tab:
            _render_mode_section(
                output_data[mode_key],
                scores[mode_key],
                mode_key,
                file_name,
                file_key=f"{stem}_{mode_key}",
                scoring=scoring,
                markers=markers,
                sensor=prep.sensor,
                table=table,
            )
            if contour is not None:
                render_contouring(
                    output_data[mode_key], mode_key, file_name,
                    f"{stem}_{mode_key}", contour, file_bytes, prep=prep,
                )
            render_next_steps(
                next_steps(mode_key, len(output_data[mode_key]), scoring, contour,
                           summary is not None and summary.has_xy),
                f"{stem}_{mode_key}",
            )

    st.markdown("#### Calibration")
    st.caption("Values found here are entered in the sidebar under Corrections & filters "
               "(calibration offsets with the advanced settings shown).")
    render_lag_estimate(file_bytes, file_name, stem, prep)
    try:
        raw_prep = replace(prep, exclude_lines=(), corrections=corrections.CorrectionSettings(),
                           recompute_from_iq=False)       # calibration lines as recorded
        calib_table, _ = prepared_table(file_bytes, file_name, raw_prep)
    except ValueError:
        calib_table = None
    if calib_table is not None:
        ui_tools.render_multiheight(calib_table, stem, prep.sensor)

    try:
        table, _ = prepared_table(file_bytes, file_name, prep)
    except ValueError:
        table = None
    st.markdown("#### Further analysis")
    ui_tools.render_inversion(output_data, scores, table, stem, prep.sensor)
    if table is not None:
        ui_tools.render_anomaly_spectrum(table, stem)
    soil_page_link()


def soil_page_link() -> None:
    """Link to the soil-tools page (a caption where the app runs without its pages, e.g. in tests)."""
    if st.session_state.get(PAGES_KEY):
        st.page_link(SOIL_PAGE, label="Soil tools: sampling design, soil-property calibration, "
                     "management zones", icon=":material/grass:")
    else:
        st.caption("Soil tools (sampling design, calibration, management zones) are on their own "
                   "page, in the navigation at the top of the sidebar.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

SAMPLE_FILE = Path(__file__).resolve().parent / "sample_data" / "synthetic_gem2_area_map.csv"
SAMPLE_STATE_KEY = "use_sample"
SOIL_PAGE = "pages/soil_tools.py"
UPLOADER_KEY = "uploader"
UPLOADS_KEY = "kept_uploads"       # the uploaded files, kept while another page is shown
SURVEYS_KEY = "surveys"            # {name: (bytes, name, prep)} of the GEM files on the analysis page
PAGES_KEY = "pages"                # set when the app runs with its page navigation


@dataclass(frozen=True)
class SampleFile:
    """The bundled sample survey, shaped like a Streamlit upload (name + getvalue)."""
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data

    @classmethod
    def load(cls) -> "SampleFile":
        return cls(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes())


def keep_uploads() -> None:
    """
    Uploader callback: a copy of the uploaded files that outlives the page.
    Streamlit drops the uploader's files when another page is shown, and an
    uploader's value cannot be set back, so the analysis page reads this copy.
    """
    files = st.session_state.get(UPLOADER_KEY) or []
    st.session_state[UPLOADS_KEY] = [SampleFile(f.name, f.getvalue()) for f in files]


def keep_settings(state) -> None:
    """
    Keeps the sidebar settings while another page is shown: Streamlit drops
    the state of widgets a page does not draw, unless it was set in code.
    """
    for key in list(state.keys()):
        if is_saved_setting(str(key).partition("~")[0]):
            state[key] = state[key]


TOUR_STEPS = [                   # (icon, title, where it happens)
    (":material/upload_file:", "Upload",
     "Drop GEM-2 exports above, or try the sample survey below."),
    (":material/fact_check:", "Check the summary",
     "A card under each file gives its lines, frequencies and the tools it supports."),
    (":material/leaderboard:", "Rank frequencies",
     "Switch on **Frequency scoring** (sidebar → Profiles) to rank them by signal-to-noise ratio."),
    (":material/map:", "Map",
     "Switch on **Area map** or **Pseudo-section** (sidebar → 2D contouring)."),
    (":material/download:", "Export",
     "Download profiles, scores, maps and grids under each tab, or every file at the end."),
]


def render_welcome() -> None:
    """Empty state: the steps of a first analysis, a sample to try, and where the settings are."""
    st.markdown("#### How it works")
    for column, (k, (icon, title, text)) in zip(st.columns(len(TOUR_STEPS)), enumerate(TOUR_STEPS, 1)):
        with column:
            st.markdown(f"{icon} **{k} · {title}**")
            st.caption(text)
    st.caption(
        "Every setting has a **?** tooltip. The sidebar starts with the settings most surveys "
        "need; **Show advanced settings** at its top adds the rest. "
        + ui_tools.doc_link("getting-started", "a-first-pass-through-the-app", "Getting started")
    )
    if SAMPLE_FILE.exists() and st.button(
        "Try the sample survey", key="sample_load", type="primary", icon=":material/science:",
        help="A synthetic GEM-2 area survey with five frequencies, coordinates and I/Q.",
    ):
        st.session_state[SAMPLE_STATE_KEY] = True
        st.rerun()


def is_gem_upload(file_bytes: bytes, file_name: str) -> bool:
    """True when the first rows of an upload look like a GEM-2 export (see is_gem_format)."""
    try:
        if file_name.lower().endswith(".csv"):
            probe = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
        else:
            probe = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=5)
        return is_gem_format(probe)
    except Exception:
        return False


def main():
    st.title("GEM-2 Analysis Tool")
    st.caption(
        "Multi-frequency EMI survey analysis for the GEM-2 — profiles, frequency ranking, "
        f"corrections, sensor physics and 2D mapping · v{APP_VERSION}"
    )

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.toggle("Show advanced settings", key=ADVANCED_KEY, help=ADVANCED_HELP)
        status = st.container()             # filled once every setting is known
        st.divider()
        st.subheader("Profiles")
        with st.expander("Profile interpolation & scoring", expanded=True):
            st.caption("Used by: representative profiles, per-frequency plots, frequency "
                       "ranking, pseudo-sections, inversion and batch export.")
            scoring = st.toggle("Frequency scoring", value=False, key=wkey("scoring"), help=SCORING_HELP)

            distance_step = st.number_input(
                "Distance step (m)",
                min_value=0.01,
                max_value=100.0,
                value=DEFAULT_DISTANCE_STEP,
                step=0.1, key=wkey("prof_step"),
                format="%.2f", help=SCIENCE_HELP["distance_step"],
            )

            interp_kind = st.selectbox(
                "Interpolation method",
                INTERP_METHODS,
                index=0, key=wkey("prof_interp"),
                help=SCIENCE_HELP["interpolation"],
            )
            if scoring and interp_kind == "kriging":
                st.warning(KRIGING_SCORE_WARNING)

            if distance_step > 10.0:
                st.warning(
                    f"Distance step is {distance_step:.2f} m. "
                    "If your data spans less than this, all sheets will be skipped."
                )

        prep = render_data_sidebar()
        if scoring and prep.corrections.lowers_noise:
            st.warning(SMOOTHING_SCORE_WARNING)
        contour = render_contouring_sidebar()
        prep = replace(prep, coord_mode=contour.coord_mode)   # one coordinate choice for all
        changed = changed_settings(prep, distance_step, interp_kind)
        with status:
            render_settings_status(changed)

        st.divider()
        render_settings_file()
        with st.expander("About & methods", expanded=False):
            st.markdown(ABOUT_MD)
            st.markdown(about_references_markdown())

    # ── File uploader ───────────────────────────────────────────────────────
    st.file_uploader(
        "Upload one or more data files (.xlsx or .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=True, key=UPLOADER_KEY, on_change=keep_uploads,
        help="GEM-2 exports from WinGEM / EMExport (CSV or XLSX) or .gbf-converted tables.",
    )
    uploaded_files = list(st.session_state.get(UPLOADS_KEY) or [])
    if uploaded_files and not st.session_state.get(UPLOADER_KEY):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.caption("Showing the files uploaded before: "
                   + ", ".join(f.name for f in uploaded_files) + ". Upload others above to replace them.")
        if c2.button("Clear files", key="uploads_clear", width="stretch"):
            st.session_state.pop(UPLOADS_KEY, None)
            st.rerun()
    if uploaded_files:
        st.session_state.pop(SAMPLE_STATE_KEY, None)    # real data replaces the sample
    elif st.session_state.get(SAMPLE_STATE_KEY):
        uploaded_files = [SampleFile.load()]
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.info(f"Showing the synthetic sample survey `{SAMPLE_FILE.name}`. Start with its "
                "summary card below, then the **Next steps** buttons under each tab. Upload your "
                "own files above to replace it.")
        if c2.button("Close sample", key="sample_close", width="stretch"):
            st.session_state.pop(SAMPLE_STATE_KEY, None)
            st.rerun()

    st.session_state[SURVEYS_KEY] = {}
    if not uploaded_files:
        render_welcome()
        render_survey_planning(prep.sensor)
        return

    if changed:
        st.info(
            f"**Results below use {ui_tools.plural(len(changed), 'changed setting')}:** "
            + " · ".join(changed) + ". Change or reset them in the sidebar.",
            icon=":material/tune:",
        )

    # ── Only GEM-2 exports are read ────────────────────────────────────────
    others = [f.name for f in uploaded_files if not is_gem_upload(f.getvalue(), f.name)]
    if others:
        st.error(
            f"Not a GEM-2 export: {', '.join(others)}. The app reads WinGEM / EMExport tables "
            "and .gbf-converted tables, recognised by their column names. "
            + ui_tools.doc_link("input-data", "", "Input data")
        )
        uploaded_files = [f for f in uploaded_files if f.name not in others]
        if not uploaded_files:
            render_survey_planning(prep.sensor)
            return
    st.session_state[SURVEYS_KEY] = {f.name: (f.getvalue(), f.name, prep) for f in uploaded_files}

    # ── Process each file and accumulate results for batch export ──────────
    batch_results: list[dict] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        stem = Path(file_name).stem
        # Collect processed data (cached — no extra computation cost)
        gem_output, gem_scores, _ = process_gem_file(
            file_bytes, file_name, distance_step, interp_kind, prep
        )
        for mode_key in gem_io.GEM_MODES:
            if gem_output.get(mode_key):
                batch_results.append({
                    "stem": stem,
                    "mode": mode_key,
                    "output_data": gem_output[mode_key],
                    "scores": gem_scores[mode_key],
                })

    # ── Per-file detailed results: one file at a time ──────────────────────
    taken_keys: set[str] = set()
    file_keys = [unique_file_key(Path(f.name).stem, taken_keys, gem_io.GEM_MODES) for f in uploaded_files]
    shown = pick_file([f.name for f in uploaded_files])
    uploaded_file, file_key = uploaded_files[shown], file_keys[shown]
    st.divider()
    st.subheader(uploaded_file.name)
    render_gem_results(
        uploaded_file.getvalue(), uploaded_file.name, distance_step, interp_kind, contour,
        scoring, prep, file_key=file_key,
    )

    gem_files = [(f.getvalue(), f.name) for f in uploaded_files]
    if contour.area_map and len(gem_files) >= 2:
        render_combined_maps(gem_files, contour, prep)
    if batch_results:
        render_batch_export(batch_results, uploaded_files, interp_kind, scoring)
    render_survey_planning(prep.sensor)


def pick_file(names: list[str]) -> int:
    """Index of the file whose results are shown; a picker appears when there are several."""
    if len(names) == 1:
        return 0
    st.divider()
    choice = st.segmented_control if len(names) <= 4 else st.selectbox
    kwargs = {"default": 0} if choice is st.segmented_control else {"index": 0}
    shown = choice(
        f"Results for one of {len(names)} files", list(range(len(names))),
        key=f"file_pick_{len(names)}",
        format_func=lambda i: names[i], **kwargs,
        help="One file is shown at a time; exports for every file are at the end of the page.",
    )
    return 0 if shown is None else int(shown)


def render_batch_export(
    batch_results: list[dict], uploaded_files: list, interp_kind: str, scoring: bool,
) -> None:
    """Workbook with the profiles (and scores) of every uploaded file."""
    st.divider()
    st.subheader("Export all files")
    what = "Interpolated profiles & scores" if scoring else "Interpolated profiles"
    files = ui_tools.plural(len(uploaded_files), "file")
    st.caption(f"{what} ({interp_kind}) across {files}.")
    batch_bytes = build_batch_xlsx(batch_results, include_scores=scoring)
    ui_tools.download_button(
        f"All files — {interp_kind} (.xlsx)", batch_bytes, f"batch_{interp_kind}.xlsx",
        key="dl_batch",
    )


def render_survey_planning(sensor: emphysics.Sensor) -> None:
    """Tools that need no data file: the forward model for planning a survey."""
    st.divider()
    st.subheader("Survey planning")
    ui_tools.render_forward_model(sensor)


def run_app() -> None:
    """Entry point: the survey-analysis page and the soil-tools page."""
    st.set_page_config(page_title="GEM-2 Analysis Tool", page_icon=None, layout="wide")
    keep_settings(st.session_state)
    st.session_state[PAGES_KEY] = True
    st.navigation([
        st.Page(main, title="Survey analysis", icon=":material/insights:", default=True),
        st.Page(SOIL_PAGE, title="Soil tools", icon=":material/grass:"),
    ]).run()


if __name__ == "__main__":
    run_app()