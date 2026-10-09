"""
Representative Incision Tool — Streamlit UI
Refactored from RIs_v1.py

Run with:
    streamlit run RIs_v2.py
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    make_interp_spline,
)

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

ALL_INTERP_METHODS = ["linear", "cubic", "nearest", "quadratic", "pchip", "akima", "polynomial"]

# Methods that fit a smooth trend instead of passing through every sample.
# They remove short-wavelength variability from each trace, lowering σ and
# inflating the score, so their scores are not comparable with interpolants.
TREND_FIT_METHODS = {"polynomial"}
TREND_FIT_WARNING = (
    "`polynomial` is a least-squares trend fit (degree ≤ 5), not an interpolant: "
    "it smooths each trace, which lowers σ and inflates the score. Use it to view "
    "trends, not to compare scores with the other methods."
)

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
    "Rank frequencies by signal-to-noise score (amplitude / noise σ). Off: "
    "profiles, maps and exports are produced without scores."
)
SMOOTHING_SCORE_WARNING = (
    "Despiking, smoothing or PCA noise reduction lower the noise σ, so scores "
    "rise. Compare scores only between runs with the same filters."
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
        "Column with the sensor height of each reading. Readings are corrected to the "
        "reference height with a fitted height response."
    ),
    "altitude_ref": (
        "Height every reading is corrected to; 0 uses the median height."
    ),
    "area_map": (
        "Grids the readings over the survey area from their coordinates and draws a contour "
        "map of each channel."
    ),
    "background": (
        "Shifts each EC channel so its median equals this value, e.g. from a reference "
        "measurement. Removes a constant calibration offset."
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
        "Blanks the most extreme values of each channel, e.g. readings near metal."
    ),
    "clip_range": (
        "Values below the lower or above the upper percentile are blanked."
    ),
    "coil_axis": (
        "Direction of the coil axis (the walking direction) on the map; the footprint is "
        "longer along it."
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
    "deconv_reg": (
        "Damping of the deconvolution: smaller values sharpen more but amplify noise."
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
        "leaves bull's-eyes around readings. Radial basis function: spline-like surface from "
        "a chosen kernel. Natural neighbour: weights from the area each reading would give "
        "up; smooth and never beyond the data range. Nearest neighbour: value of the closest "
        "reading. Modified Shepard: inverse distance blending local quadratics, so it "
        "follows trends. Local polynomial: a weighted polynomial fitted around each node. "
        "Polynomial regression: one trend surface for the whole area (regional trend). "
        "Moving average: mean of the readings within a search radius. Data metrics: a "
        "statistic of the readings within a search radius, e.g. their number."
    ),
    "heading": (
        "Keeps only readings walked within a range of headings, e.g. to check for heading "
        "error on lines walked back and forth."
    ),
    "heading_dir": (
        "Walking direction to keep, clockwise from north (or from the +Y axis)."
    ),
    "heading_tol": (
        "Readings within this many degrees of the heading are kept."
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
        "How each pass is resampled onto the common grid. Linear and nearest are the most "
        "conservative; cubic, quadratic, PCHIP and Akima are smooth (PCHIP and Akima do not "
        "overshoot); polynomial fits a trend and lowers the noise."
    ),
    "lag_channel": (
        "Channel compared between neighbouring lines; one with clear anomalies works best."
    ),
    "levels": (
        "Number of colour bands between the lowest and highest mapped value."
    ),
    "local_power": (
        "How fast a reading's weight falls across the search: (1 − d / R)^power."
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
    "method_order": (
        "Order of the polynomial: 1 is a plane, 2 a quadratic surface, 3 a cubic one. Higher "
        "orders follow more detail but swing more where readings are sparse."
    ),
    "metric": (
        "Statistic of the readings within the search radius of each node. The number and "
        "density of readings show the survey coverage; the others summarise the values."
    ),
    "min_curvature_tension": (
        "0 gives the smoothest surface (minimum curvature), which may overshoot between "
        "readings; values towards 1 pull the surface tight between them, like a stretched "
        "membrane (Smith & Wessel, 1990). 0.25–0.35 suits steep anomalies."
    ),
    "pseudosection": (
        "Frequency against distance for each line. Lower frequencies see deeper on average, "
        "but the frequency axis is not a calibrated depth (McNeill, 1980)."
    ),
    "rbf_kernel": (
        "Basis function of the surface. Multiquadric (Surfer's default) gives a smooth "
        "surface that follows the data closely; inverse multiquadric is flatter away from "
        "readings; the natural cubic and thin-plate splines bend least."
    ),
    "ref_method": (
        "Regression: least-squares line through the matched pairs. Moments: gain and offset "
        "that match the mean and spread of the reference values."
    ),
    "ref_radius": (
        "Survey readings within this distance of a reference point are matched to it (their "
        "median)."
    ),
    "search_radius": (
        "Readings within this distance of a node are used. 0 = automatic: a circle that "
        "holds about 16 readings at the survey's mean density."
    ),
    "separation": (
        "Distance from the transmitter to the receiver coil (1.66 m on the GEM-2). Wider "
        "separation sees deeper."
    ),
    "smooth": (
        "Centred moving average along each line. Lowers noise but blurs small anomalies (and "
        "raises the scores)."
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
    y_label: str = ""      # empty → profile_label(mode, is_gem)
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
    pivoted.columns = [f"Line_{c}" for c in pivoted.columns]

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
                f"{int(dup.sum())} repeated reading(s) at the same distance within "
                f"{keyed.loc[dup, 'Line'].nunique()} line(s): kept the first reading "
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

    output_data: dict[str, dict[str, pd.DataFrame]] = _empty_modes()
    scores: dict[str, dict[str, dict]] = _empty_modes()

    for mode_key in gem_io.GEM_MODES:
        for freq_label, pivoted_df in gem_data[mode_key].items():
            interp_df, score_dict, error, col_warnings = process_sheet(
                pivoted_df, distance_step, interp_kind
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
    "nearest": 1,
    "pchip": 2,
    "quadratic": 3,
    "polynomial": 3,
    "akima": 5,
    "cubic": 4,
}


def _interpolate_with_method(
    xp: np.ndarray, yp: np.ndarray, target_x: np.ndarray, method: str
) -> np.ndarray:
    """Interpolate *yp* at *xp* onto *target_x* using the named method."""
    if method == "linear":
        return np.interp(target_x, xp, yp)

    if method == "nearest":
        idx = np.searchsorted(xp, target_x, side="left")
        idx_left = np.clip(idx - 1, 0, len(xp) - 1)
        idx_right = np.clip(idx, 0, len(xp) - 1)
        nearest = np.where(
            np.abs(target_x - xp[idx_left]) <= np.abs(target_x - xp[idx_right]),
            idx_left,
            idx_right,
        )
        return yp[nearest]

    if method == "pchip":
        return PchipInterpolator(xp, yp)(target_x)

    if method == "quadratic":
        return make_interp_spline(xp, yp, k=2)(target_x)

    if method == "polynomial":
        # Least-squares trend fit (exact only for ≤ 6 points). Polynomial.fit
        # maps x onto [-1, 1], keeping projected coordinates well-conditioned.
        degree = min(len(xp) - 1, 5)
        return np.polynomial.Polynomial.fit(xp, yp, degree)(target_x)

    if method == "akima":
        return Akima1DInterpolator(xp, yp)(target_x)

    if method == "cubic":
        return CubicSpline(xp, yp)(target_x)

    raise ValueError(f"Unknown interpolation method: {method!r}")


# ---------------------------------------------------------------------------
# Core processing (pure functions — no side effects, no sys.exit)
# ---------------------------------------------------------------------------

def process_sheet(
    df: pd.DataFrame,
    distance_step: float = DEFAULT_DISTANCE_STEP,
    interp_kind: str = DEFAULT_INTERP_KIND,
) -> tuple[pd.DataFrame | None, dict | None, str | None, list[str]]:
    """
    Process a single sheet.

    Returns
    -------
    interpolated_df : DataFrame indexed by common distance, columns = original trace
                      columns; NaN outside each trace's own measured range
    score_dict      : {"mean_std": σ_noise, "amplitude": float, "score": float,
                       "noise_method": NOISE_BETWEEN | NOISE_INTRA, "n_traces": int}
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

    interpolated_lines: dict[str, np.ndarray] = {}
    raw_samples: dict[str, np.ndarray] = {}

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
                f"column '{col}': {n_dup} repeated distance value(s), kept the first reading"
            )
            df_xy = df_xy.drop_duplicates(subset="d", keep="first")

        if df_xy.shape[0] < 2:
            continue

        xp = df_xy["d"].values
        yp = df_xy["y"].values

        try:
            min_pts = _METHOD_MIN_POINTS.get(interp_kind, 2)
            if len(xp) < min_pts:
                col_warnings.append(
                    f"column '{col}': need ≥{min_pts} points for {interp_kind} interpolation, skipped"
                )
                continue
            values = _interpolate_with_method(xp, yp, common_dist, interp_kind)
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

    rep_prof = interpolated_df.mean(axis=1, skipna=True)
    n_traces = interpolated_df.shape[1]

    if n_traces >= 2:
        # Between-trace noise, only where ≥ 2 passes were measured. The passes
        # are a finite sample of the measurement process, hence ddof=1; the
        # point-wise variances are pooled as sqrt(mean(var)).
        overlap = interpolated_df.notna().sum(axis=1) >= 2
        if not overlap.any():
            return None, None, "traces do not overlap in distance", col_warnings
        var_prof = interpolated_df[overlap].var(axis=1, skipna=True, ddof=1)
        noise = float(np.sqrt(np.nanmean(var_prof)))
        amp_prof = rep_prof[overlap]
        noise_method = NOISE_BETWEEN
    else:
        noise = _intra_profile_noise(next(iter(raw_samples.values())))
        amp_prof = rep_prof
        noise_method = NOISE_INTRA

    amplitude = float(np.nanmax(amp_prof) - np.nanmin(amp_prof))

    if np.isfinite(noise) and noise >= SCORE_EPSILON:
        score = amplitude / noise
    else:
        # Zero or undefined noise: a ratio would be meaningless, and falling
        # back to the amplitude would rank a data-unit value against ratios.
        score = np.nan
        col_warnings.append(
            f"{noise_method} noise is zero or undefined — score left blank"
        )

    score_dict = {
        "mean_std": noise,
        "amplitude": amplitude,
        "score": score,
        "noise_method": noise_method,
        "n_traces": n_traces,
    }
    return interpolated_df, score_dict, None, col_warnings


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


@st.cache_data(show_spinner=False)
def process_file(
    excel_bytes: bytes,
    mode: str,
    distance_step: float = DEFAULT_DISTANCE_STEP,
    interp_kind: str = DEFAULT_INTERP_KIND,
) -> tuple[dict, dict, list[str]]:
    """
    Process all sheets in an uploaded Excel file (legacy multi-sheet format).

    Returns
    -------
    output_data               : {sheet_name: interpolated_df}
    representativeness_scores : {sheet_name: score_dict}
    warnings                  : list of warning strings
    """
    output_data: dict[str, pd.DataFrame] = {}
    scores: dict[str, dict] = {}
    warnings: list[str] = []

    try:
        excel = pd.ExcelFile(io.BytesIO(excel_bytes))
    except Exception as exc:
        warnings.append(f"Could not open file: {exc}")
        return output_data, scores, warnings

    for sheet_name in excel.sheet_names:
        sheet_name = str(sheet_name)  # guard against integer sheet names (e.g. 9000)
        try:
            df = pd.read_excel(excel, sheet_name=sheet_name)
        except Exception as exc:
            warnings.append(f"Sheet '{sheet_name}': read error — {exc}")
            continue

        interp_df, score_dict, error, col_warnings = process_sheet(df, distance_step, interp_kind)

        for w in col_warnings:
            warnings.append(f"Sheet '{sheet_name}': {w}")

        if error:
            warnings.append(f"Sheet '{sheet_name}': skipped — {error}")
            continue

        output_data[sheet_name] = interp_df
        scores[sheet_name] = score_dict

    mixed = noise_method_warning(scores)
    if mixed:
        warnings.append(mixed)

    return output_data, scores, warnings


# ---------------------------------------------------------------------------
# Plot helpers (return Figure objects, never touch global pyplot state)
# ---------------------------------------------------------------------------

def profile_label(mode: str, is_gem: bool, channel: str | None = None) -> str:
    """Y-axis label for mean profiles: GEM units are known, legacy units are not."""
    if mode == "AUX" and channel is None:
        return "Mean value (units in legend)"
    return f"Mean {ctr.value_label(mode, is_gem, channel)}"


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
    is_gem: bool = False,
    show_scores: bool = True,
    markers: list[float] | None = None,
) -> plt.Figure:
    """All representative profiles on one axes; scores in the legend when *show_scores*."""
    if opts is None:
        opts = GraphOptions()

    visible = opts.selected_sheets if opts.selected_sheets else list(output_data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    y_label = profile_label(mode, is_gem)

    for sheet_name in visible:
        interp_df = output_data.get(sheet_name)
        if interp_df is None:
            continue
        rep_prof = interp_df.mean(axis=1, skipna=True)
        common_dist = interp_df.index.values
        name = ctr.value_label(mode, is_gem, sheet_name) if mode == "AUX" else sheet_name
        if show_scores:
            name = f"{name} (score={scores[sheet_name]['score']:.2f})"
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
    profile view without a prepared GEM table (legacy files).
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
    is_gem: bool = False,
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
    ax.set_ylabel(opts.y_label or profile_label(mode, is_gem, sheet_name))
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
    Sheet "Scores"   : one row per (file, mode, frequency) with Score/Amplitude/Noise,
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
                    "Score": round(sc["score"], 4),
                    "Amplitude": round(sc["amplitude"], 6),
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


def build_all_methods_batch_xlsx(
    uploaded_files: list,
    mode: str,
    distance_step: float,
    include_scores: bool = True,
    prep: pipeline.PrepSettings | None = None,
) -> bytes:
    """
    Run every interpolation method against every uploaded file and pack all
    interpolated profiles and scores into a single xlsx.

    Workbook layout
    ---------------
    Sheet "Scores"            : File | Mode | Frequency | Method | Exact interpolant | Score
                                | Amplitude | Noise(σ)
                                | Noise method | Traces
    "{stem}_{mode}_{method}"  : Distance (m) + one {freq}_mean column per frequency
    """
    buf = io.BytesIO()
    score_rows: list[dict] = []

    def _unique_sheet(name: str, existing: "KeysView[str]") -> str:
        """Excel-safe, at most 31 chars, deduplicated."""
        name = excel_sheet_title(name)
        candidate = name
        suffix = 2
        while candidate in existing:
            tag = f"_{suffix}"
            candidate = name[: EXCEL_SHEET_NAME_MAX - len(tag)] + tag
            suffix += 1
        return candidate

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name
            stem = Path(file_name).stem

            # Detect format once
            is_gem = False
            try:
                if file_name.lower().endswith(".csv"):
                    probe = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
                else:
                    probe = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=5)
                is_gem = is_gem_format(probe)
            except Exception:
                pass

            for method in ALL_INTERP_METHODS:
                # Retrieve processed data (uses @st.cache_data — free if already computed)
                if is_gem:
                    out, sc, _ = process_gem_file(file_bytes, file_name, distance_step, method, prep)
                    entries = [
                        (mode_key, out[mode_key], sc[mode_key])
                        for mode_key in gem_io.GEM_MODES
                        if out.get(mode_key)
                    ]
                else:
                    out, sc, _ = process_file(file_bytes, mode, distance_step, method)
                    entries = [(mode, out, sc)] if out else []

                for mode_key, output_data, scores in entries:
                    # Scores rows
                    for freq, metrics in (scores.items() if include_scores else ()):
                        score_rows.append({
                            "File": stem,
                            "Mode": mode_key,
                            "Frequency / Sheet": freq,
                            "Method": method,
                            "Exact interpolant": method not in TREND_FIT_METHODS,
                            "Score": round(metrics["score"], 4),
                            "Amplitude": round(metrics["amplitude"], 6),
                            "Noise (σ)": round(metrics["mean_std"], 6),
                            "Noise method": metrics["noise_method"],
                            "Traces": metrics["n_traces"],
                        })

                    # Data sheet: Distance + one mean column per frequency
                    if not output_data:
                        continue
                    raw = f"{stem[:10]}_{mode_key}_{method}"
                    sheet_name = _unique_sheet(raw, writer.sheets.keys())
                    mean_profiles_table(output_data, label_len=18).to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )

        # Scores summary — written last (openpyxl appends; reorder below)
        if score_rows:
            pd.DataFrame(score_rows).to_excel(writer, sheet_name="Scores", index=False)

    # Move "Scores" sheet to the first position
    wb = __import__("openpyxl").load_workbook(io.BytesIO(buf.getvalue()))
    if "Scores" in wb.sheetnames:
        wb.move_sheet("Scores", offset=-len(wb.sheetnames) + 1)
    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)
    return out_buf.getvalue()


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Graph editor UI helper
# ---------------------------------------------------------------------------

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
            st.markdown("**Frequencies / sheets**")
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
            x_auto = st.checkbox("Auto", value=True, key=f"ge_xauto_{file_key}")
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
            y_auto = st.checkbox("Auto", value=True, key=f"ge_yauto_{file_key}")
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
# Reusable rendering section (shared by legacy and GEM paths)
# ---------------------------------------------------------------------------

def _render_mode_section(
    output_data: dict[str, pd.DataFrame],
    scores: dict[str, dict],
    mode: str,
    file_name: str,
    file_key: str,
    is_gem: bool = False,
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
    is_gem      : True for GEM files (known units); False for legacy files
    scoring     : show the frequency ranking and scores (never for AUX channels)
    markers     : event-marker distances drawn on the profiles
    table       : prepared GEM table, for plots of chosen columns (graph editor)
    """
    if not output_data:
        st.error("No usable frequencies / sheets found.")
        return

    stem = Path(file_name).stem
    show_scores = scoring and mode != "AUX"
    if show_scores:
        _render_ranking(scores)
        if len(output_data) > 1:
            ui_tools.render_frequency_selection(output_data, scores, file_key)
    else:
        _render_channel_list(output_data, scores, mode, is_gem)
    if is_gem and mode == "EC":
        ui_tools.render_frequency_info(output_data, sensor or emphysics.GEM2, file_key)
    freqs = [ctr.parse_frequency(name) or 0.0 for name in output_data]
    if is_gem and mode in ("MS", "I") and max(freqs) >= ui_tools.PERMITTIVITY_FREQUENCY:
        st.caption(ui_tools.PERMITTIVITY_CAPTION)

    # ── Graph editor ─────────────────────────────────────────────────────
    opts = render_graph_editor(output_data, file_key=file_key, table=table, mode=mode)
    readings = opts.readings and table is not None

    def overview() -> plt.Figure:
        if readings:
            return make_readings_figure(
                table, opts, f"{opts.y_column} against {opts.x_column} — {file_name}"
            )
        return make_overview_figure(output_data, scores, mode, file_name, opts, is_gem, show_scores, markers)

    # ── Overview plot ────────────────────────────────────────────────────
    if readings:
        st.markdown(f"#### {opts.y_column} against {opts.x_column}")
    else:
        st.markdown("#### Representative profiles")
        if opts.x_column != PROFILE_X:
            st.caption("Representative profiles are drawn against distance along the line; "
                       "choose a channel for Y to plot against another X.")
    overview_fig = overview()
    st.pyplot(overview_fig, use_container_width=True)
    plt.close(overview_fig)

    # ── Per-sheet detail ─────────────────────────────────────────────────
    with st.expander("Per-frequency detail plots", expanded=False):
        visible = opts.selected_sheets if opts.selected_sheets else list(output_data.keys())
        for sheet_name in visible:
            interp_df = output_data.get(sheet_name)
            if interp_df is None:
                continue
            sc = scores[sheet_name]
            if show_scores:
                st.markdown(
                    f"**{sheet_name}** — score {sc['score']:.2f} · "
                    f"A = {sc['amplitude']:.4g} · σ = {sc['mean_std']:.4g} "
                    f"({sc['noise_method']}, {sc['n_traces']} trace(s))"
                )
            else:
                st.markdown(f"**{sheet_name}** — {sc['n_traces']} trace(s)")
            sheet_fig = make_sheet_figure(sheet_name, interp_df, mode, opts, is_gem, markers)
            st.pyplot(sheet_fig, use_container_width=True)
            plt.close(sheet_fig)

    # ── Downloads ────────────────────────────────────────────────────────
    st.markdown("#### Downloads")
    columns = st.columns(3 if show_scores else 2)

    with columns[0]:
        st.download_button(
            label="Interpolated profiles (.xlsx)",
            data=build_excel_download(output_data),
            file_name=f"{stem}_{mode}_interpolated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_xlsx_{file_key}",
        )

    if show_scores:
        with columns[1]:
            st.download_button(
                label="Scores (.csv)",
                data=build_scores_csv(scores),
                file_name=f"{stem}_{mode}_scores.csv",
                mime="text/csv",
                key=f"dl_csv_{file_key}",
            )

    with columns[-1]:
        download_fig = overview()
        png_bytes = fig_to_png(download_fig)
        plt.close(download_fig)
        st.download_button(
            label="Overview plot (.png)",
            data=png_bytes,
            file_name=f"{stem}_{mode}_overview.png",
            mime="image/png",
            key=f"dl_png_{file_key}",
        )


def _render_channel_list(
    output_data: dict[str, pd.DataFrame], scores: dict[str, dict], mode: str, is_gem: bool
) -> None:
    """Channels of one mode without scores: passes and covered distance."""
    st.markdown("#### Channels")
    rows = [
        {
            "Channel": name,
            "Units": ctr.value_label(mode, is_gem, name),
            "Traces": scores[name]["n_traces"],
            "From (m)": round(float(df.index.min()), 3),
            "To (m)": round(float(df.index.max()), 3),
        }
        for name, df in output_data.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
                "Score": round(m["score"], 2),
                "Amplitude": round(m["amplitude"], 6),
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
            {"Score": "{:.2f}", "Amplitude": "{:.4g}", "Noise (σ)": "{:.4g}"}, na_rep="—"
        )
        if rank_df["Score"].notna().any():  # all-blank scores have nothing to colour
            styled = styled.background_gradient(subset=["Score"], cmap="Blues")
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )
    with col_best:
        best_name, best_metrics = ranking[0]
        st.metric("Best frequency", best_name)
        st.metric(
            "Score (A / σ)",
            f"{best_metrics['score']:.2f}",
            help=(
                "Score = amplitude / noise (σ). "
                "Multi-trace: σ = pooled sample std across passes (ddof=1), "
                "where ≥ 2 passes overlap. "
                "Single-trace: σ from second differences of the measured "
                "samples (robust MAD). "
                "If σ is zero or undefined, the score is left blank. "
                "Higher score = cleaner, larger signal."
            ),
        )
        st.metric("Amplitude", f"{best_metrics['amplitude']:.4g}")
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


def render_data_sidebar() -> pipeline.PrepSettings:
    """Sidebar controls for quality flags and along-line distance (GEM files)."""
    st.divider()
    st.subheader("GEM data")
    drop_flagged = st.checkbox(
        "Drop readings with a Status flag", value=True, key="prep_status",
        help="The GEM-2 sets Status to a non-zero value when a reading has a "
             "problem such as ADC overload.",
    )
    exclude = st.text_input(
        "Lines to leave out (comma-separated)", key="prep_exclude",
        help="For example calibration or test lines recorded in the same file.",
    )
    method = st.selectbox(
        "Distance along line", list(gem_io.DISTANCE_METHODS),
        format_func=gem_io.DISTANCE_METHODS.get, key="prep_distance", help=DISTANCE_HELP,
    )
    spacing = 1.0
    if method in ("sample", "markers"):
        spacing = st.number_input(
            "Reading spacing (m)" if method == "sample" else "Marker spacing (m)",
            min_value=0.001, value=1.0, step=0.1, format="%.3f", key="prep_spacing",
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
        st.caption("GEM-2 defaults (Won et al., 1996). Check them against your sensor's .gem file.")
        separation = st.number_input("Tx–Rx separation (m)", 0.1, 10.0, 1.66, step=0.01, key="sen_sep",
                                     help=SCIENCE_HELP["separation"])
        bucking = st.number_input("Tx–bucking coil (m, 0 = none)", 0.0, 10.0, 1.035, step=0.005,
                                  format="%.3f", key="sen_buck", help=SCIENCE_HELP["bucking"])
        height = st.number_input("Sensor height (m)", 0.0, 50.0, 1.0, step=0.05, key="sen_h",
                                 help=SCIENCE_HELP["height"])
        recompute = st.checkbox(
            "Recompute EC / MS from I / Q", key="sen_recompute",
            help="Half-space conversion of each reading (Huang & Won, 2000) with this geometry "
                 "and height. Overwrites exported EC / MS columns; adds them to I/Q-only files.",
        )
        visc_text = st.text_input(
            "Magnetic viscosity from frequencies (low, high Hz)", key="sen_visc",
            help="Two frequencies with I_ / Q_ columns, e.g. '1525, 5325'. Adds EC from the "
                 "quadrature difference and the quadrature susceptibility κ″ to the AUX "
                 "channels (Simon et al., 2015). Uncalibrated Q offsets shift κ″, so compare "
                 "contrasts unless Q is calibrated.",
        )
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
    with st.expander("Corrections & filters", expanded=False):
        st.markdown("**Filters**")
        despike = st.checkbox("Despike (running median)", key="cor_despike", help=SCIENCE_HELP["despike"])
        window, threshold = 5, 4.0
        if despike:
            window = st.slider("Despike window (readings)", 3, 21, 5, step=2, key="cor_dwin",
                               help=SCIENCE_HELP["despike_window"])
            threshold = st.number_input("Despike threshold (× noise σ)", 1.0, 20.0, 4.0,
                                        step=0.5, key="cor_dthr", help=SCIENCE_HELP["despike_threshold"])
        clip = st.checkbox("Clip to percentiles", key="cor_clip", help=SCIENCE_HELP["clip"])
        clip_range = None
        if clip:
            lo, hi = st.slider("Keep percentiles", 0.0, 100.0, (1.0, 99.0), step=0.5, key="cor_crange",
                               help=SCIENCE_HELP["clip_range"])
            clip_range = (float(lo), float(hi))
        smooth = st.slider("Running mean (readings, 0 = off)", 0, 21, 0, key="cor_smooth",
                           help=SCIENCE_HELP["smooth"])
        pca = st.number_input("PCA components kept per mode (0 = off)", 0, 10, 0, key="cor_pca",
                              help="Minsley et al. (2010): channels of a mode are strongly "
                                   "correlated, trailing components are mostly noise.")

        st.markdown("**Drift and environment**")
        drift_text = st.text_input("Base-station lines (comma-separated Line labels)",
                                   key="cor_drift",
                                   help="Lines recorded at a fixed base station during the "
                                        "survey. Needs a Time column. They are removed after "
                                        "the correction.")
        drift_model = st.selectbox("Drift model", ["piecewise", "linear"], key="cor_dmodel",
                                   help=SCIENCE_HELP["drift_model"])
        temp_col = st.text_input("Temperature column (needs ≥ 3 base-station lines)",
                                 key="cor_temp", help=SCIENCE_HELP["temperature"]).strip()
        alt_col = st.text_input("Sensor-height column (e.g. drone altitude above ground)",
                                key="cor_alt", help=SCIENCE_HELP["altitude"]).strip()
        alt_ref = None
        if alt_col:
            alt_ref_value = st.number_input("Reference height (m, 0 = median)", 0.0, 100.0, 0.0,
                                            step=0.1, key="cor_altref", help=SCIENCE_HELP["altitude_ref"])
            alt_ref = float(alt_ref_value) or None

        st.markdown("**Calibration**")
        offsets_file = st.file_uploader(
            "Calibration offsets (.csv)", type=["csv"], key="cor_offsets",
            help="Columns 'column' and 'offset'; each offset is subtracted from that column. "
                 "Produced by 'Multi-height calibration' under a file.",
        )
        background = st.number_input("Known background EC (mS/m, 0 = off)", 0.0, 10_000.0, 0.0,
                                     key="cor_bg", help=SCIENCE_HELP["background"])
        ref_file = st.file_uploader("Reference values (.csv)", type=["csv"], key="cor_ref",
                                    help="Same coordinate columns as the survey (X/Y or "
                                         "Lat/Lon) plus one column per channel to calibrate, "
                                         "named like the survey column, e.g. EC1525Hz[mS/m].")
        ref_radius, ref_method = 2.0, "regression"
        if ref_file is not None:
            ref_radius = st.number_input("Matching radius (m)", 0.1, 100.0, 2.0, key="cor_rrad",
                                         help=SCIENCE_HELP["ref_radius"])
            ref_method = st.selectbox("Calibration fit", ["regression", "moments"], key="cor_rmeth",
                                      help=SCIENCE_HELP["ref_method"])
        soil_t = st.number_input("Soil temperature for EC at 25 °C (°C, 0 = off)", 0.0, 50.0, 0.0,
                                 key="cor_soilt", help=SCIENCE_HELP["ec25"])

        st.markdown("**Positioning**")
        lag = st.number_input("GPS lag (s)", -5.0, 5.0, 0.0, step=0.1, key="cor_lag",
                              help="Positive: readings were logged after the position. Use "
                                   "'Estimate GPS lag' under a file to find it.")
        heading = st.checkbox("Keep one walking direction", key="cor_head", help=SCIENCE_HELP["heading"])
        bearing_center, bearing_tol = None, 30.0
        if heading:
            bearing_center = float(st.number_input("Heading (° from north)", 0.0, 359.0, 0.0,
                                                    key="cor_hdir", help=SCIENCE_HELP["heading_dir"]))
            bearing_tol = float(st.number_input("± tolerance (°)", 1.0, 90.0, 30.0, key="cor_htol",
                                                help=SCIENCE_HELP["heading_tol"]))

    return corrections.CorrectionSettings(
        despike=despike, despike_window=int(window), despike_threshold=float(threshold),
        clip_percentiles=clip_range,
        altitude_column=alt_col, altitude_reference=alt_ref,
        temperature_column=temp_col,
        drift_lines=_parse_labels(drift_text), drift_model=drift_model,
        iq_offsets=offsets_file.getvalue() if offsets_file is not None else None,
        ec_background=float(background) or None,
        reference=ref_file.getvalue() if ref_file is not None else None,
        reference_radius=float(ref_radius), reference_method=ref_method,
        soil_temperature=float(soil_t) or None,
        pca_components=int(pca), smooth_window=int(smooth),
        lag_seconds=float(lag),
        bearing_center=bearing_center, bearing_tolerance=bearing_tol,
    )


def render_lag_estimate(
    file_bytes: bytes, file_name: str, file_key: str, prep: pipeline.PrepSettings
) -> None:
    """Expander that estimates the GPS time lag from crossings of neighbouring lines."""
    with st.expander("Estimate GPS lag", expanded=False):
        no_lag = replace(prep, recompute_from_iq=False,
                         corrections=replace(prep.corrections, lag_seconds=0.0, bearing_center=None))
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
            st.pyplot(fig)
            plt.close(fig)


def render_contouring_sidebar() -> ContourSettings:
    """Sidebar controls for 2D contouring; call inside `with st.sidebar`."""
    st.divider()
    st.subheader("2D contouring")
    area = st.toggle("Area map (plan view)", value=False, key="ct_area", help=SCIENCE_HELP["area_map"])
    pseudo = st.toggle("Pseudo-section", value=False, key="ct_pseudo", help=SCIENCE_HELP["pseudosection"])
    if not area:
        if not pseudo:
            return ContourSettings()
        colours = render_colour_controls()
        return ContourSettings(pseudosection=True, n_levels=colours.n_levels, colours=colours)

    coord_mode = st.selectbox(
        "Coordinates", ["auto", "metres", "degrees"],
        format_func=str.capitalize, key="ct_coords",
        help="Auto: Lat/Lon columns are degrees; X/Y are degrees only if they "
             "look like lon/lat spanning < 0.05°.",
    )
    method = st.selectbox(
        "Gridding method", list(ctr.METHODS), format_func=ctr.METHODS.get, key="ct_method",
        help=SCIENCE_HELP["gridding"],
    )
    smoothing = 0.0
    variogram_model = "spherical"
    if method == "spline":
        smoothing = st.number_input(
            "Spline smoothing", min_value=0.0, value=0.0, step=0.1, key="ct_smooth",
            help="0 = exact interpolation; larger values trade fit for smoothness.",
        )
    if method == "rbf":
        smoothing = st.number_input(
            "Smoothing", min_value=0.0, value=0.0, step=0.1, key="ct_rbf_smooth",
            help="0 = exact interpolation; larger values trade fit for smoothness.",
        )
    if method == "kriging":
        variogram_model = st.selectbox(
            "Variogram model", ctr.VARIOGRAM_MODELS, key="ct_vario", help=SCIENCE_HELP["variogram"],
        )
    method_options = render_method_options(method)
    cell = st.number_input(
        "Cell size (m, 0 = auto)", min_value=0.0, value=0.0, step=0.1,
        format="%.2f", key="ct_cell", help=SCIENCE_HELP["cell"],
    )
    blank = st.number_input(
        "Blanking distance (m, 0 = auto)", min_value=0.0, value=0.0, step=0.5,
        format="%.2f", key="ct_blank",
        help="Grid nodes farther than this from any data point are left blank. "
             "Auto keeps the gaps between survey lines filled: the larger of "
             "2 × median point spacing and 1.5 × the 90th-percentile distance "
             "from grid nodes inside the survey to the nearest data point.",
    )
    level = st.checkbox(
        "Line levelling (per-line median)", value=False, key="ct_level", help=LEVELLING_HELP,
    )
    projection = st.selectbox(
        "Projection of Lat/Lon", list(ctr.PROJECTIONS), format_func=ctr.PROJECTIONS.get,
        key="ct_proj", help="UTM gives georeferenced .asc / GeoTIFF exports.",
    )
    xy_epsg = st.number_input(
        "EPSG code of X/Y (0 = unknown)", 0, 99999, 0, key="ct_epsg",
        help="For files whose X/Y are already projected, e.g. 32634 for UTM 34N.",
    )
    xy_epsg = int(xy_epsg) or None
    problem = gridtools.epsg_problem(xy_epsg) if xy_epsg else None
    if problem:
        st.warning(problem)
        xy_epsg = None
    colours = render_colour_controls()
    show_points = st.checkbox("Show data points", value=True, key="ct_points",
                              help="Marks the readings used for gridding (one per grid cell).")
    return ContourSettings(
        area_map=True,
        pseudosection=pseudo,
        coord_mode=coord_mode,
        method=method,
        smoothing=float(smoothing),
        variogram_model=variogram_model,
        cell_size=float(cell) or None,
        blank_distance=float(blank) or None,
        level_lines=level,
        n_levels=colours.n_levels,
        projection=projection,
        xy_epsg=xy_epsg,
        method_options=method_options,
        colours=colours,
        show_points=show_points,
    )


def render_method_options(method: str) -> tuple:
    """Sidebar inputs for the settings of *method*; returns (name, value) pairs."""
    defaults = ctr.METHOD_DEFAULTS.get(method, {})
    opts = {}
    if method == "idw":
        opts["power"] = st.number_input("Power", 0.5, 10.0, defaults["power"], step=0.5,
                                        key="ct_idw_power", help=SCIENCE_HELP["idw_power"])
        opts["delta"] = st.number_input("Smoothing distance (m)", 0.0, value=defaults["delta"],
                                        step=0.1, key="ct_idw_delta",
                                        help=SCIENCE_HELP["idw_smoothing"])
    elif method == "min_curvature":
        opts["tension"] = st.slider("Tension", 0.0, 0.95, defaults["tension"], step=0.05,
                                    key="ct_tension", help=SCIENCE_HELP["min_curvature_tension"])
    elif method == "rbf":
        opts["kernel"] = st.selectbox("Basis function", list(ctr.RBF_KERNELS),
                                      format_func=ctr.RBF_KERNELS.get, key="ct_rbf_kernel",
                                      help=SCIENCE_HELP["rbf_kernel"])
    elif method in ("local_polynomial", "polynomial"):
        opts["order"] = st.selectbox("Polynomial order", [1, 2, 3], key=f"ct_{method}_order",
                                     help=SCIENCE_HELP["method_order"])
        if method == "local_polynomial":
            opts["power"] = st.number_input("Weighting power", 0.0, 10.0, defaults["power"],
                                            step=0.5, key="ct_local_power",
                                            help=SCIENCE_HELP["local_power"])
    elif method in ("moving_average", "metrics"):
        if method == "metrics":
            opts["statistic"] = st.selectbox("Statistic", list(ctr.METRICS),
                                             format_func=ctr.METRICS.get, key="ct_metric",
                                             help=SCIENCE_HELP["metric"])
        opts["radius"] = st.number_input("Search radius (m, 0 = auto)", 0.0, value=0.0,
                                         step=0.5, format="%.2f", key="ct_radius",
                                         help=SCIENCE_HELP["search_radius"])
    return tuple(opts.items())


def render_colour_controls() -> ctr.ColourStyle:
    """Sidebar colour settings shared by area maps and pseudo-sections."""
    st.markdown("**Colours**")
    cmap = st.selectbox("Colour map", list(ctr.COLOUR_MAPS), format_func=ctr.COLOUR_MAPS.get,
                        key="ct_cmap", help=SCIENCE_HELP["colour_map"])
    reverse = st.checkbox("Reverse colours", value=False, key="ct_cmap_reverse")
    display = st.selectbox("Display", list(ctr.MAP_DISPLAYS), format_func=ctr.MAP_DISPLAYS.get,
                           key="ct_display", help=SCIENCE_HELP["map_display"])
    scale = st.selectbox("Colour scale", list(ctr.COLOUR_SCALES),
                         format_func=ctr.COLOUR_SCALES.get, key="ct_scale",
                         help=SCIENCE_HELP["colour_scale"])
    range_mode = st.selectbox("Colour range", list(ctr.COLOUR_RANGES),
                              format_func=ctr.COLOUR_RANGES.get, key="ct_range",
                              help=SCIENCE_HELP["colour_range"])
    percentiles, vmin, vmax = (2.0, 98.0), None, None
    if range_mode == "percentile":
        percentiles = st.slider("Percentiles", 0.0, 100.0, (2.0, 98.0), step=0.5,
                                key="ct_percentiles")
    elif range_mode == "fixed":
        vmin = st.number_input("Colour minimum (map units)", value=0.0, key="ct_vmin")
        vmax = st.number_input("Colour maximum (map units)", value=100.0, key="ct_vmax")
    n_levels = st.slider("Contour levels", 5, 50, 20, key="ct_levels", help=SCIENCE_HELP["levels"])
    lines = st.checkbox("Contour lines", value=False, key="ct_lines",
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
    is_gem: bool,
) -> None:
    st.markdown("#### Pseudo-section")
    profiles = {name: df.mean(axis=1, skipna=True) for name, df in output_data.items()}
    try:
        ps = ctr.build_pseudosection(profiles)
        fig = ctr.make_pseudosection_figure(
            ps, ctr.value_label(mode, is_gem),
            f"Pseudo-section [{mode}] — {file_name}", contour.n_levels, contour.colours,
        )
    except ctr.ContouringError as exc:
        st.info(str(exc))
        return
    except Exception as exc:  # last resort: never show a traceback
        st.error(f"Pseudo-section failed: {exc}")
        return
    st.pyplot(fig, use_container_width=True)
    png = fig_to_png(fig)
    plt.close(fig)
    st.caption(PSEUDOSECTION_CAPTION)
    st.download_button(
        "Pseudo-section (.png)", data=png,
        file_name=f"{Path(file_name).stem}_{mode}_pseudosection.png",
        mime="image/png", key=f"dl_ps_{file_key}",
    )


def _render_area_map(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    file_bytes: bytes,
    is_gem: bool,
    prep: pipeline.PrepSettings | None = None,
) -> None:
    st.markdown("#### Area map")
    if not is_gem:
        st.info(
            "Area maps need a GEM file with X/Y or Lat/Lon coordinates; "
            "the legacy multi-sheet format has none."
        )
        return

    freq = st.selectbox("Channel", list(output_data.keys()), key=f"ct_freq_{file_key}")
    label = ctr.value_label(mode, is_gem, freq)
    try:
        with st.spinner("Gridding…"):
            result = compute_area_map_cached(
                file_bytes, file_name, gem_value_column(mode, freq), prep,
                **grid_params(contour),
            )
        try:
            axis = gridtools.line_axis_angle(prepared_table(file_bytes, file_name, prep)[0])
        except ValueError:
            axis = 0.0
        result, processing = _map_processing(
            result, file_key, (prep or pipeline.PrepSettings()).sensor, axis
        )
        label = ctr.result_label(result, label)
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
        st.info(str(exc))
        return
    except Exception as exc:  # last resort: never show a traceback
        st.error(f"Area map failed: {exc}")
        return
    st.pyplot(fig, use_container_width=True)
    png = fig_to_png(fig)
    plt.close(fig)
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
                )
            except ctr.ContouringError as exc:
                st.error(str(exc))
            else:
                st.write(
                    f"RMSE **{cv['rmse']:.4g}**, MAE **{cv['mae']:.4g}** "
                    f"({label}, {cv['n']:,} held-out points; random folds, "
                    "optimistic for densely sampled lines)"
                )

    _render_map_statistics(result, label, file_key)
    stem = Path(file_name).stem
    _render_map_downloads(result, png, f"{stem}_{mode}_{freq}_{result.method}", file_key)


def _map_processing(
    result: ctr.AreaMapResult, key: str, sensor: emphysics.Sensor, axis_angle: float
) -> tuple[ctr.AreaMapResult, list[str]]:
    """Optional footprint deconvolution and grid filter, chosen in an expander."""
    with st.expander("Map processing (filters, footprint deconvolution)", expanded=False):
        kind = st.selectbox("Filter", gridtools.MAP_FILTERS, key=f"mp_kind_{key}",
                            help=SCIENCE_HELP["map_filter"])
        size = st.slider("Window (cells)", 3, 51, 15 if kind == "high-pass" else 3, step=2,
                         key=f"mp_size_{key}", help=SCIENCE_HELP["map_window"])
        threshold = 4.0
        if kind == "despike":
            threshold = st.number_input("Despike threshold (× robust σ)", 1.0, 20.0, 4.0,
                                        key=f"mp_thr_{key}", help=SCIENCE_HELP["map_despike"])
        deconv = st.checkbox(
            "Deconvolve the sensor footprint", key=f"mp_dec_{key}",
            help="Lateral Tikhonov deconvolution of the low-induction-number footprint of the "
                 "coils (Rx minus bucking coil, sensor height from the sidebar). Sharpens "
                 "apparent-conductivity maps; it does not resolve depth (cf. Guillemoteau et "
                 "al., 2017, for multi-coil sensors).",
        )
        reg, angle = 1e-2, axis_angle
        if deconv:
            reg = st.select_slider("Regularisation", [1e-4, 1e-3, 1e-2, 1e-1, 1.0], value=1e-2,
                                   key=f"mp_reg_{key}", help=SCIENCE_HELP["deconv_reg"])
            angle = st.number_input("Coil axis (° from map x axis)", 0.0, 180.0,
                                    float(round(axis_angle, 1)), key=f"mp_ang_{key}",
                                    help=SCIENCE_HELP["coil_axis"])
    if kind == "none" and not deconv:
        return result, []
    z, msgs = gridtools.process_map(
        result.z, result.spec.cell, kind, int(size), float(threshold), deconv, float(reg),
        float(angle), sensor,
    )
    return replace(result, z=z), msgs


def _render_map_statistics(result: ctr.AreaMapResult, label: str, key: str) -> None:
    with st.expander("Histogram & statistics", expanded=False):
        try:
            stats = gridtools.summary_stats(result.z)
        except ValueError:
            st.info("No values to summarise.")
            return
        st.dataframe(pd.DataFrame([stats]).round(4), hide_index=True, use_container_width=True)
        values = result.z[np.isfinite(result.z)]
        fig, ax = plt.subplots(figsize=(6, 2.5))
        flat = np.ptp(values) <= 1e-9 * max(float(np.abs(values).max()), 1e-12)  # round-off only
        ax.hist(values, bins=1 if flat else 50, color="C0")
        ax.set_xlabel(label)
        ax.set_ylabel("Grid nodes")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_map_downloads(result: ctr.AreaMapResult, png: bytes, base: str, key: str) -> None:
    columns = st.columns(5 if result.epsg else 4)
    columns[0].download_button(
        "Area map (.png)", data=png, file_name=f"{base}.png",
        mime="image/png", key=f"dl_am_png_{key}",
    )
    columns[1].download_button(
        "Grid (.csv)", data=ctr.grid_to_csv(result), file_name=f"{base}.csv",
        mime="text/csv", key=f"dl_am_csv_{key}",
    )
    columns[2].download_button(
        "Grid (.asc)", data=ctr.grid_to_asc(result), file_name=f"{base}.asc",
        mime="text/plain", key=f"dl_am_asc_{key}",
    )
    columns[3].download_button(
        "GeoTIFF (.tif)",
        data=gridtools.geotiff_bytes(result.z, result.spec.x0, result.spec.y0, result.spec.cell,
                                     result.epsg),
        file_name=f"{base}.tif", mime="image/tiff", key=f"dl_am_tif_{key}",
    )
    if result.epsg:
        columns[4].download_button(
            "Projection (.prj)", data=gridtools.prj_wkt(result.epsg), file_name=f"{base}.prj",
            mime="text/plain", key=f"dl_am_prj_{key}",
        )
    if result.origin is not None:
        st.warning(
            "Input was in degrees and projected to local metres: the .asc and GeoTIFF are not "
            "georeferenced. Choose UTM under 'Projection of Lat/Lon', or use the lon/lat "
            "columns of the CSV."
        )


def render_contouring(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    file_bytes: bytes,
    is_gem: bool,
    prep: pipeline.PrepSettings | None = None,
) -> None:
    """Render the enabled 2D contouring sections for one file and mode."""
    if not output_data:
        return
    if contour.pseudosection and mode != "AUX":   # AUX channels are not frequencies
        _render_pseudosection(output_data, mode, file_name, file_key, contour, is_gem)
    if contour.area_map:
        _render_area_map(output_data, mode, file_name, file_key, contour, file_bytes, is_gem, prep)


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
    if not common:
        st.info("The files share no channel.")
        return
    col = st.selectbox("Channel", sorted(common), key="cm_col")
    found = {m: chans for m, chans in gem_io.find_channels([col]).items() if chans}
    mode, label = next((m, next(iter(ch))) for m, ch in found.items())
    unit = ctr.value_label(mode, True, label)
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
                         hide_index=True)
            fig = ctr.make_area_map_figure(result, ctr.result_label(result, unit),
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
        st.info(str(exc))
        return
    st.pyplot(fig, use_container_width=True)
    png = fig_to_png(fig)
    plt.close(fig)
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


def render_legacy_results(
    file_bytes: bytes,
    file_name: str,
    mode: str,
    distance_step: float,
    interp_kind: str,
    contour: ContourSettings | None = None,
    scoring: bool = True,
    file_key: str | None = None,
) -> None:
    """Process and render a legacy multi-sheet Excel file."""
    stem = file_key or Path(file_name).stem

    with st.spinner("Processing…"):
        output_data, scores, warnings = process_file(
            file_bytes, mode,
            distance_step=distance_step,
            interp_kind=interp_kind,
        )

    if warnings:
        with st.expander(f"{len(warnings)} warning(s)", expanded=False):
            for w in warnings:
                st.warning(w)

    if not output_data:
        st.error("No usable sheets found in this file.")
        if distance_step > 1.0:
            st.info("Try reducing the distance step in the sidebar.")
        return

    _render_mode_section(output_data, scores, mode, file_name, file_key=stem, scoring=scoring)
    if contour is not None:
        render_contouring(
            output_data, mode, file_name, stem, contour, file_bytes, is_gem=False
        )


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

    if warnings:
        with st.expander(f"{len(warnings)} warning(s)", expanded=False):
            for w in warnings:
                st.warning(w)

    # Determine which modes have data
    available_modes = [m for m in gem_io.GEM_MODES if output_data.get(m)]

    if not available_modes:
        st.error("No usable frequencies found in this GEM file.")
        if distance_step > 1.0:
            st.info("Try reducing the distance step in the sidebar.")
        return

    try:
        table, _ = prepared_table(file_bytes, file_name, prep)
        markers = gem_io.marker_distances(table, table[pipeline.DISTANCE_COL].to_numpy())
    except (ValueError, KeyError):  # preparation problems are already listed in the warnings
        table, markers = None, []

    st.caption("GEM format detected — showing every channel in the file")
    prep = prep or pipeline.PrepSettings()
    render_lag_estimate(file_bytes, file_name, stem, prep)
    try:
        raw_prep = replace(prep, exclude_lines=(), corrections=corrections.CorrectionSettings(),
                           recompute_from_iq=False)       # calibration lines as recorded
        calib_table, _ = prepared_table(file_bytes, file_name, raw_prep)
    except ValueError:
        calib_table = None
    if calib_table is not None:
        ui_tools.render_multiheight(calib_table, stem, prep.sensor)

    tabs = st.tabs([
        f"{m} ({len(output_data[m])} {'channels' if m == 'AUX' else 'frequencies'})"
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
                is_gem=True,
                scoring=scoring,
                markers=markers,
                sensor=prep.sensor,
                table=table,
            )
            if contour is not None:
                render_contouring(
                    output_data[mode_key], mode_key, file_name,
                    f"{stem}_{mode_key}", contour, file_bytes, is_gem=True, prep=prep,
                )

    try:
        table, _ = prepared_table(file_bytes, file_name, prep)
    except ValueError:
        table = None
    st.markdown("#### Further analysis")
    ui_tools.render_inversion(output_data, scores, table, stem, prep.sensor)
    if table is not None:
        ui_tools.render_anomaly_spectrum(table, stem)
        ui_tools.render_soil_tools(table, stem)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Representative Incision Tool",
        page_icon=None,
        layout="wide",
    )

    st.title("Representative Incision Tool")
    st.caption(
        "Multi-frequency EMI (GEM-2) survey analysis — representative profiles, "
        "frequency ranking and 2D mapping · v2.1"
    )

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Processing")

        scoring = st.toggle("Frequency scoring", value=False, key="scoring", help=SCORING_HELP)

        mode = st.radio(
            "Measurement mode",
            list(MODES.keys()),
            horizontal=True,
            help="For legacy multi-sheet files. GEM files show both EC and MS automatically.",
        )

        distance_step = st.number_input(
            "Distance step (m)",
            min_value=0.01,
            max_value=100.0,
            value=DEFAULT_DISTANCE_STEP,
            step=0.1,
            format="%.2f", help=SCIENCE_HELP["distance_step"],
        )

        interp_kind = st.selectbox(
            "Interpolation method",
            ALL_INTERP_METHODS,
            index=0,
            format_func=lambda m: f"{m} (trend fit)" if m in TREND_FIT_METHODS else m,
            help=SCIENCE_HELP["interpolation"],
        )
        if scoring and interp_kind in TREND_FIT_METHODS:
            st.warning(TREND_FIT_WARNING)

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

        st.divider()
        with st.expander("About & methods", expanded=False):
            st.markdown(
                """
**Representative Incision Tool** — v2.1

#### Geophysical background

##### How EMI instruments work

The GEM-2 (Won et al., 1996) is a **broadband frequency-domain
EMI sensor**. A transmitter coil generates a time-varying
primary magnetic field that induces eddy currents in conductive
subsurface materials. Those currents produce a secondary magnetic
field, which the receiver coil measures as a complex voltage
ratio relative to the primary.

Under the **low-induction-number (LIN) approximation**
(McNeill, 1980) — valid when the ratio of coil separation to
skin depth is much less than 1 — the two components of the
secondary field decouple cleanly:

| Component | Physical quantity | Unit |
|---|---|---|
| **Quadrature** (out-of-phase) | Apparent electrical conductivity (EC) | mS/m |
| **In-phase** | Apparent magnetic susceptibility (MS) | 10⁻³ SI (ppt, dimensionless) |

This separation means a single instrument pass simultaneously
maps two independent subsurface properties.

##### What EC tells you

**Electrical conductivity** reflects how easily electrical
current flows through the bulk soil. It is controlled by:

- **Moisture content** — water greatly increases EC
- **Clay content and mineralogy** — clays with high CEC
  (e.g. smectite) are strongly conductive
- **Salinity** — dissolved ions are the primary charge carriers
- **Soil texture** — fine-grained materials retain more water
  and conduct better than coarse sands or gravels

High EC values indicate fine-grained, moist, saline, or
clay-rich substrates. Low values indicate dry, sandy, or
gravelly soils (Reynolds, 2011).

##### What MS tells you

**Magnetic susceptibility** measures how strongly the soil is
magnetised by the primary field. Elevated MS is associated with:

- **Ferrimagnetic minerals** — especially maghemite (γ-Fe₂O₃)
  and magnetite (Fe₃O₄)
- **Pedogenic enhancement** — topsoil MS is often higher than
  subsoil due to bacterial reduction–oxidation cycles forming
  fine-grained magnetite
- **Burning and anthropogenic enrichment** — fired hearths,
  kilns, and iron-rich fills are classic high-MS targets in
  archaeological surveys
- **Mafic lithologies** — basaltic parent material produces
  naturally elevated background MS

##### Why frequency matters

The **skin depth** δ (in metres) is the depth at which the
primary field amplitude falls to 1/e ≈ 37 % of its surface
value:

$$\\delta = \\sqrt{\\frac{2}{\\omega \\mu \\sigma}}$$

where ω is angular frequency, μ is magnetic permeability, and
σ is electrical conductivity. Because δ ∝ f⁻¹/², **lower
frequencies penetrate deeper** while **higher frequencies are
more sensitive to the shallow subsurface** (Callegary et al.,
2007). At a given site the optimal frequency is therefore
site-specific: it depends on the depth of the target and the
background conductivity.

The LIN approximation holds when the induction number
B = s/δ ≪ 1, where s is the coil separation. At high
frequencies or in very conductive soils this condition breaks
down and the linear relationship between signal and subsurface
properties no longer holds. The tool's scoring is most reliable
when the LIN condition is satisfied across all tested
frequencies.

##### What a representative incision is

A **representative incision** is a carefully chosen survey
transect, typically 50–200 m long, that crosses the main
geophysical contrasts expected across the site — for example,
running from a known high-EC wetland fringe into a low-EC
sandy terrace. Repeated passes (at least two) along the same
line allow assessment of instrument drift, operator-induced
variability, and short-term environmental noise. This replicated
design is what makes the between-trace standard deviation a
meaningful noise estimate.

---

#### Scoring methodology

Each frequency receives a **representativeness score**:

$$\\text{Score} = \\frac{A}{\\sigma_{\\text{noise}}}$$

| Symbol | Meaning |
|---|---|
| **A** | Signal amplitude — the peak-to-trough range of the mean profile: max(p̄) − min(p̄), taken where at least two passes overlap. Captures the total geophysical contrast resolved at that frequency. |
| **σ_noise** | Noise — estimated from the data depending on how many passes were acquired (see below). |

A higher score means the frequency resolves large subsurface
contrasts clearly above the noise level. This is equivalent to a
high signal-to-noise ratio (SNR), the standard geophysical
criterion for data quality (Sheriff & Geldart, 1995).

**How noise is estimated:**

**With ≥ 2 passes (recommended):**
At each distance step where at least two passes were measured,
the sample variance across passes (ddof = 1) is computed; σ is
the square root of the mean of these variances. The passes are a
finite sample of the measurement process whose noise is being
estimated, so the sample formula applies. This σ captures all
sources of between-pass variability: instrument noise,
positioning uncertainty, and short-term drift. Each pass is used
only inside its own measured distance range — nothing is
extrapolated.

**With 1 pass only (single-trace fallback):**
Between-trace σ is undefined, so the tool estimates noise from
the *measured* samples of that pass (not the interpolated grid).
Second differences Δ²y cancel a locally linear geological trend;
for white noise var(Δ²y) = 6σ², so σ ≈ 1.4826 · MAD(Δ²y) / √6.
The median absolute deviation ignores the few large differences
at sharp boundaries, and the estimate does not depend on the
distance step or interpolation method. Single-trace scores are
less reliable — multiple passes are always preferable — and the
ranking warns when they are mixed with multi-pass scores.

**Zero or undefined noise:**
When σ < 10⁻⁸ or cannot be estimated, the score is left blank and
ranked last, with a warning.

---

#### Processing pipeline

| Step | What happens |
|---|---|
| **1. Ingest** | File is read (CSV or XLSX). GEM format is auto-detected from column names (`Line`, `Y`, `EC*Hz[mS/m]`, `MSusc*Hz[1/1000]`, `I_*Hz`, `Q_*Hz`). Readings with a non-zero `Status` are dropped (optional), the enabled **Corrections & filters** run (despike, height, drift, calibration, GPS lag, …), and the distance of each reading along its line is found (sidebar **Distance along line**). The flat GEM table is pivoted: each channel becomes a matrix with distance as rows and survey lines as columns. |
| **2. Interpolate** | All traces are resampled onto a common evenly-spaced distance grid (`np.linspace`). The interpolation method is chosen from the sidebar (see *Interpolation methods* below). At a repeated distance within a trace only the first reading is kept. Each trace is left blank outside its own measured range. |
| **3. Score** (optional) | With **Frequency scoring** on, the mean profile and noise metric are computed as above and frequencies are ranked by descending score. |
| **4. Visualise** | An overview plot shows all mean profiles together. Per-frequency plots show individual traces (thin, semi-transparent), the mean profile (bold), and the ±1σ envelope. |
| **5. Export** | Per-file downloads (interpolated profiles XLSX, scores CSV, overview PNG) and a **Batch export** that packages results from all uploaded files into a single XLSX. A second batch option runs all interpolation methods simultaneously and exports every result for direct comparison. |

> **Precision note:** GEM CSV exports round EC values to integers
> and MS to one decimal place, discarding the instrument's full
> precision (3+ decimal places available in the XLSX). For
> quantitative frequency comparison, always use the XLSX export.

---

#### Interpolation methods

Seven methods are available from the sidebar dropdown. Each is applied uniformly to all traces within the selected file.

| Method | Min. points | Characteristics |
|---|---|---|
| **linear** | 2 | Piecewise linear connection between data points. Conservative, no overshoot. Recommended for noisy or sparse data. |
| **nearest** | 1 | Assigns each grid point the value of the closest data point. Useful for step-like or categorical-style signals. |
| **quadratic** | 3 | Quadratic B-spline (`make_interp_spline`, k = 2). Smoother than linear with modest curvature. |
| **cubic** | 4 | Cubic spline with continuous second derivative (`CubicSpline`). Best for dense, smooth, low-noise profiles. May overshoot at sharp boundaries. |
| **pchip** | 2 | Piecewise Cubic Hermite Interpolating Polynomial. Shape-preserving and monotone within each interval — avoids the overshoot of cubic splines. Good default for near-monotone geophysical profiles. |
| **akima** | 5 | Akima (1970) local spline. Uses only neighbouring points to set slopes, making it robust to isolated outliers that would disturb a global cubic spline. |
| **polynomial** | 3 | **Trend fit, not an interpolant.** Global least-squares polynomial (degree = min(n − 1, 5)); exact only for ≤ 6 points. Smooths each trace, which lowers σ and inflates the score — do not compare its scores with the other methods. |

The **Batch export — all methods** option runs all seven methods in one step and writes a single XLSX whose `Scores` sheet lists every (file, mode, frequency, method) combination side-by-side for direct comparison.

---

#### 2D contouring

Switch on in the sidebar under **2D contouring**.

- **Pseudo-section** — mean profiles of all frequencies as one
  distance × frequency contour, aligned on their common distance
  range. The frequency axis is *not* a calibrated depth axis:
  under LIN the depth response is set by coil geometry (McNeill,
  1980; Callegary et al., 2007); at most, lower frequencies see
  somewhat deeper (Huang, 2005).
- **Area map** — plan-view grid of one frequency from the X/Y (or
  Lat/Lon) coordinates of all lines: optional per-line median
  levelling (a simple form; cf. Mauring & Kihle, 2006),
  block-median reduction, then
  thin-plate spline (Briggs, 1974; Sandwell, 1987), ordinary
  kriging with a fitted variogram and standard-deviation map
  (Corwin & Lesch, 2005; Oliver & Webster, 2014; for
  regression / cokriging alternatives see Lesch et al., 1995) or
  linear triangulation. Nodes far from data are blanked.
  Use **Cross-validate** to compare methods (Li & Heap, 2011).

---

#### Supported file formats

| Format | Notes |
|---|---|
| **GEM-2 `.xlsx`** | Full instrument precision — recommended for scoring |
| **GEM-2 `.csv`** | EC rounded to integers; may slightly affect scores |
| **GEM-2 I/Q** | `I_*Hz` / `Q_*Hz` in ppm, plus `PowerLn`, `QSum`, `TotalEC[mS/m]` when present |
| **Legacy `.xlsx`** | One sheet per frequency; column 0 = distance (m), columns 1+ = survey traces |

---

#### Glossary

| Term | Definition |
|---|---|
| **EC** | Apparent electrical conductivity (mS/m) — quadrature EMI response |
| **MS** | Apparent magnetic susceptibility (10⁻³ SI, ppt) — in-phase EMI response |
| **EMI** | Frequency-domain electromagnetic induction |
| **LIN** | Low induction number approximation — the condition under which EC and MS decouple linearly (McNeill, 1980) |
| **Skin depth (δ)** | Depth at which primary field amplitude falls to 1/e; decreases with frequency and conductivity |
| **GEM-2** | Multi-frequency broadband EMI sensor (Won et al., 1996) |
| **SNR** | Signal-to-noise ratio |
| **Score** | A / σ_noise — the representativeness metric used for frequency ranking |
| **Amplitude (A)** | max − min of the mean profile across all passes |
| **σ_noise** | Pooled sample std across passes (multi-pass) or second-difference MAD estimate on the measured samples (single-pass) |
| **ddof = 1** | Sample standard deviation; used because the passes are a finite sample of the measurement process |
| **Representative incision** | A transect designed to sample the full range of subsurface variability at a site |
| **PCHIP** | Piecewise Cubic Hermite Interpolating Polynomial — shape-preserving spline that avoids overshoot |
| **Akima spline** | Local spline that derives slopes from neighbouring points only, reducing sensitivity to outliers |
| **Batch export** | Single XLSX download packaging interpolated profiles and scores from all uploaded files |

---

#### References

- Won, I.J., Keiswetter, D.A., Fields, G.R.A. & Sutton, L.C.
  (1996). GEM-2: A new multifrequency electromagnetic sensor.
  *J. Environ. Eng. Geophys.*, **1**(2), 129–137.
  [doi:10.4133/JEEG1.2.129](https://doi.org/10.4133/JEEG1.2.129)

- McNeill, J.D. (1980). *Electromagnetic terrain conductivity
  measurement at low induction numbers*. Technical Note TN-6,
  Geonics Limited, Mississauga, Canada.

- Callegary, J.B., Ferré, T.P.A. & Groom, R.W. (2007).
  Vertical spatial sensitivity and exploration depth of
  low-induction-number electromagnetic-induction instruments.
  *Vadose Zone J.*, **6**(1), 158–167.
  [doi:10.2136/vzj2006.0120](https://doi.org/10.2136/vzj2006.0120)

- Delefortrie, S., Saey, T., Van De Vijver, E., De Smedt, P.,
  Missiaen, T., Demerre, I. & Van Meirvenne, M. (2014).
  Frequency domain electromagnetic induction survey in the
  intertidal zone: Limitations of low-induction-number and
  depth of exploration. *J. Appl. Geophys.*, **100**, 119–130.
  [doi:10.1016/j.jappgeo.2013.10.017](https://doi.org/10.1016/j.jappgeo.2013.10.017)

- De Smedt, P., Van Meirvenne, M., Herremans, D., De Reu, J.,
  Saey, T., Meerschman, E., Crombé, P. & De Clercq, W. (2013).
  The 3-D reconstruction of medieval wetland reclamation through
  electromagnetic induction survey. *Scientific Reports*,
  **3**, 1517.
  [doi:10.1038/srep01517](https://doi.org/10.1038/srep01517)

- Reynolds, J.M. (2011). *An Introduction to Applied and
  Environmental Geophysics* (2nd ed.). Wiley-Blackwell.

- Sheriff, R.E. & Geldart, L.P. (1995). *Exploration
  Seismology* (2nd ed.). Cambridge University Press.

- Bakulin, A., Silvestrov, I. & Protasov, M. (2022).
  Signal-to-noise ratio computation for challenging land data.
  *Geophys. Prospect.*, **70**, 629–638.
  [doi:10.1111/1365-2478.13183](https://doi.org/10.1111/1365-2478.13183)

- Akima, H. (1970). A new method of interpolation and smooth
  curve fitting based on local procedures. *J. ACM*, **17**(4),
  589–602.
  [doi:10.1145/321607.321609](https://doi.org/10.1145/321607.321609)

- Fritsch, F.N. & Carlson, R.E. (1980). Monotone piecewise
  cubic interpolation. *SIAM J. Numer. Anal.*, **17**(2),
  238–246.
  [doi:10.1137/0717021](https://doi.org/10.1137/0717021)

- Briggs, I.C. (1974). Machine contouring using minimum
  curvature. *Geophysics*, **39**(1), 39–48.
  [doi:10.1190/1.1440410](https://doi.org/10.1190/1.1440410)

- Sandwell, D.T. (1987). Biharmonic spline interpolation of
  GEOS-3 and SEASAT altimeter data. *Geophys. Res. Lett.*,
  **14**(2), 139–142.
  [doi:10.1029/GL014i002p00139](https://doi.org/10.1029/GL014i002p00139)

- Lesch, S.M., Strauss, D.J. & Rhoades, J.D. (1995). Spatial
  prediction of soil salinity using electromagnetic induction
  techniques: 1. Statistical prediction models: a comparison of
  multiple linear regression and cokriging. *Water Resour. Res.*,
  **31**(2), 373–386.
  [doi:10.1029/94WR02179](https://doi.org/10.1029/94WR02179)

- Corwin, D.L. & Lesch, S.M. (2005). Apparent soil electrical
  conductivity measurements in agriculture. *Comput. Electron.
  Agric.*, **46**, 11–43.
  [doi:10.1016/j.compag.2004.10.005](https://doi.org/10.1016/j.compag.2004.10.005)

- Oliver, M.A. & Webster, R. (2014). A tutorial guide to
  geostatistics: computing and modelling variograms and kriging.
  *Catena*, **113**, 56–69.
  [doi:10.1016/j.catena.2013.09.006](https://doi.org/10.1016/j.catena.2013.09.006)

- Li, J. & Heap, A.D. (2011). A review of comparative studies of
  spatial interpolation methods in environmental sciences.
  *Ecol. Inform.*, **6**, 228–241.
  [doi:10.1016/j.ecoinf.2010.12.003](https://doi.org/10.1016/j.ecoinf.2010.12.003)

- Mauring, E. & Kihle, O. (2006). Leveling aerogeophysical data
  using a moving differential median filter. *Geophysics*,
  **71**(1), L5–L11.
  [doi:10.1190/1.2163912](https://doi.org/10.1190/1.2163912)

- Huang, H. (2005). Depth of investigation for small broadband
  electromagnetic sensors. *Geophysics*, **70**(6), G135–G142.
  [doi:10.1190/1.2122412](https://doi.org/10.1190/1.2122412)
                """
            )

    ui_tools.render_forward_model(prep.sensor)

    # ── File uploader ───────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload one or more data files (.xlsx or .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one `.xlsx` or `.csv` file to get started.")
        return

    # ── Process each file and accumulate results for batch export ──────────
    batch_results: list[dict] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        stem = Path(file_name).stem

        # Probe for GEM format (read only 5 rows for speed)
        is_gem = False
        try:
            if file_name.lower().endswith(".csv"):
                probe = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
            else:
                probe = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=5)
            is_gem = is_gem_format(probe)
        except Exception:
            pass

        # Collect processed data (cached — no extra computation cost)
        if is_gem:
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
        else:
            leg_output, leg_scores, _ = process_file(
                file_bytes, mode, distance_step, interp_kind
            )
            if leg_output:
                batch_results.append({
                    "stem": stem,
                    "mode": mode,
                    "output_data": leg_output,
                    "scores": leg_scores,
                })

    # ── Batch download (shown once, above per-file results) ────────────────
    if batch_results:
        st.divider()
        st.subheader("Batch export")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            what = "Interpolated profiles & scores" if scoring else "Interpolated profiles"
            st.caption(
                f"{what} for the **selected method** "
                f"({interp_kind}) across {len(uploaded_files)} file(s)."
            )
            batch_bytes = build_batch_xlsx(batch_results, include_scores=scoring)
            st.download_button(
                label=f"Download — {interp_kind} method (.xlsx)",
                data=batch_bytes,
                file_name=f"batch_{interp_kind}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_batch",
            )

        with dl_col2:
            st.caption(
                f"{what} for **all {len(ALL_INTERP_METHODS)} methods** "
                f"across {len(uploaded_files)} file(s). May take a moment to generate."
            )
            if st.button("Generate all-methods export", key="btn_all_methods"):
                with st.spinner(
                    f"Running {len(ALL_INTERP_METHODS)} interpolation methods "
                    f"across {len(uploaded_files)} file(s)…"
                ):
                    all_methods_bytes = build_all_methods_batch_xlsx(
                        uploaded_files, mode, distance_step, scoring, prep
                    )
                st.download_button(
                    label="Download — all methods (.xlsx)",
                    data=all_methods_bytes,
                    file_name="batch_all_methods.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_all_methods",
                )

    # ── Per-file detailed results ───────────────────────────────────────────
    gem_files: list[tuple[bytes, str]] = []
    taken_keys: set[str] = set()
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        is_gem = False
        try:
            if file_name.lower().endswith(".csv"):
                probe = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
            else:
                probe = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=5)
            is_gem = is_gem_format(probe)
        except Exception:
            pass

        stem = Path(file_name).stem
        if is_gem:
            gem_files.append((file_bytes, file_name))
            render_gem_results(
                file_bytes, file_name, distance_step, interp_kind, contour, scoring, prep,
                file_key=unique_file_key(stem, taken_keys, gem_io.GEM_MODES),
            )
        else:
            render_legacy_results(
                file_bytes, file_name, mode, distance_step, interp_kind, contour, scoring,
                file_key=unique_file_key(stem, taken_keys)
            )

    if contour.area_map and len(gem_files) >= 2:
        render_combined_maps(gem_files, contour, prep)


if __name__ == "__main__":
    main()