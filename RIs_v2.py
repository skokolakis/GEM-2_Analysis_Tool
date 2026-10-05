"""
UltraRank Frequency — Streamlit UI
Refactored from RIs_v1.py

Run with:
    streamlit run RIs_v2.py
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
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
from numpy.polynomial.polynomial import polyfit, polyval

import contouring as ctr

# ---------------------------------------------------------------------------
# Configuration (all in one place, easily overridden via Streamlit widgets)
# ---------------------------------------------------------------------------
DEFAULT_DISTANCE_STEP = 0.5
DEFAULT_INTERP_KIND = "linear"
SCORE_EPSILON = 1e-8
EXCEL_SHEET_NAME_MAX = 31

MODES = {
    "EC": "Mean EC Response (S/m)",
    "MS": "Mean MS Response (10⁻⁵ SI)",
}

ALL_INTERP_METHODS = ["linear", "cubic", "nearest", "quadratic", "pchip", "akima", "polynomial"]

LINE_STYLES = {
    "Solid": "-",
    "Dashed": "--",
    "Dash-dot": "-.",
    "Dotted": ":",
}

# GEM instrument format detection
GEM_EC_PATTERN = re.compile(r'^EC(\d+)Hz\[mS/m\]$')
GEM_MS_PATTERN = re.compile(r'^MSusc(\d+)Hz\[1/1000\]$')
GEM_REQUIRED_COLS = {'Line', 'Y'}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph editor options
# ---------------------------------------------------------------------------

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
    y_label: str = ""      # empty → MODES[mode]


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


PSEUDOSECTION_CAPTION = (
    "Frequency axis shows instrument response per frequency, not depth "
    "(under LIN, depth sensitivity is set by coil geometry — Huang, 2005). "
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
    """Return True if *df* has the GEM instrument column structure."""
    cols = set(str(c) for c in df.columns)
    if not GEM_REQUIRED_COLS.issubset(cols):
        return False
    has_ec = any(GEM_EC_PATTERN.match(str(c)) for c in df.columns)
    has_ms = any(GEM_MS_PATTERN.match(str(c)) for c in df.columns)
    return has_ec or has_ms


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
    """
    subset = df[[distance_col, line_col, value_col]].copy()
    subset[distance_col] = pd.to_numeric(subset[distance_col], errors="coerce")
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=[distance_col, value_col])

    pivoted = subset.pivot_table(
        index=distance_col,
        columns=line_col,
        values=value_col,
        aggfunc="mean",
    )

    # Rename columns to "Line_0", "Line_1", etc.
    pivoted.columns = [f"Line_{int(c)}" for c in pivoted.columns]

    # Reset index so column 0 = distance (what process_sheet expects)
    pivoted = pivoted.reset_index()
    return pivoted


def parse_gem_dataframe(
    df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Parse a GEM-format DataFrame into pivoted DataFrames per mode and frequency.

    Returns
    -------
    {"EC": {"4525Hz": pivoted_df, ...}, "MS": {"4525Hz": pivoted_df, ...}}
    """
    result: dict[str, dict[str, pd.DataFrame]] = {"EC": {}, "MS": {}}

    for col in df.columns:
        col_str = str(col)

        ec_match = GEM_EC_PATTERN.match(col_str)
        if ec_match:
            freq = ec_match.group(1)
            label = f"{freq}Hz"
            result["EC"][label] = pivot_gem_frequency(df, value_col=col_str)
            continue

        ms_match = GEM_MS_PATTERN.match(col_str)
        if ms_match:
            freq = ms_match.group(1)
            label = f"{freq}Hz"
            result["MS"][label] = pivot_gem_frequency(df, value_col=col_str)

    return result


@st.cache_data(show_spinner=False)
def process_gem_file(
    file_bytes: bytes,
    file_name: str,
    distance_step: float = DEFAULT_DISTANCE_STEP,
    interp_kind: str = DEFAULT_INTERP_KIND,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, dict]], list[str]]:
    """
    Process a GEM-format file (CSV or XLSX).

    Returns
    -------
    output_data : {"EC": {freq: interp_df, ...}, "MS": {freq: interp_df, ...}}
    scores      : {"EC": {freq: score_dict, ...}, "MS": {freq: score_dict, ...}}
    warnings    : list of warning strings
    """
    warnings: list[str] = []

    # Read raw data based on extension
    if file_name.lower().endswith(".csv"):
        try:
            raw_df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            warnings.append(f"Could not read CSV: {exc}")
            return {"EC": {}, "MS": {}}, {"EC": {}, "MS": {}}, warnings
    else:
        try:
            raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
        except Exception as exc:
            warnings.append(f"Could not read Excel: {exc}")
            return {"EC": {}, "MS": {}}, {"EC": {}, "MS": {}}, warnings

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

    gem_data = parse_gem_dataframe(raw_df)

    output_data: dict[str, dict[str, pd.DataFrame]] = {"EC": {}, "MS": {}}
    scores: dict[str, dict[str, dict]] = {"EC": {}, "MS": {}}

    for mode_key in ("EC", "MS"):
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

    return output_data, scores, warnings


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
        degree = min(len(xp) - 1, 5)
        coeffs = polyfit(xp, yp, degree)
        return polyval(target_x, coeffs)

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
    interpolated_df : DataFrame indexed by common distance, columns = original trace columns
    score_dict      : {"mean_std": float, "amplitude": float, "score": float}
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

    for col in line_data.columns:
        y = line_data[col].values
        mask = ~np.isnan(distance) & ~np.isnan(y)

        if mask.sum() < 2:
            log.debug("  Column %s: fewer than 2 valid points, skipped.", col)
            continue

        df_xy = (
            pd.DataFrame({"d": distance[mask], "y": y[mask]})
            .groupby("d", as_index=False)
            .mean()
            .sort_values("d")
            .drop_duplicates(subset="d")
        )

        if not df_xy["d"].is_monotonic_increasing:
            col_warnings.append(f"column '{col}': distance not monotonic after dedup, skipped")
            continue

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
            interpolated_lines[col] = _interpolate_with_method(xp, yp, common_dist, interp_kind)
        except Exception as exc:
            col_warnings.append(f"column '{col}': interpolation failed — {exc}")
            continue

    if not interpolated_lines:
        return None, None, "no columns survived interpolation", col_warnings

    interpolated_df = pd.DataFrame(interpolated_lines, index=common_dist)
    interpolated_df.index.name = "Distance (m)"

    rep_prof = interpolated_df.mean(axis=1, skipna=True)
    # Population std (ddof=0): the traces ARE the full population of survey
    # passes, not a sample from a larger population.
    std_prof = interpolated_df.std(axis=1, skipna=True, ddof=0)

    mean_std = float(std_prof.mean())
    amplitude = float(np.nanmax(rep_prof) - np.nanmin(rep_prof))

    if np.isnan(mean_std) or mean_std < SCORE_EPSILON:
        # Single trace or near-identical traces: between-trace std is meaningless.
        # Fall back to intra-profile SNR: amplitude / within-profile noise.
        # Noise is estimated as the std of residuals from a rolling-mean smoother,
        # which separates low-frequency signal from high-frequency noise.
        window = max(5, len(rep_prof) // 10)
        smoothed = rep_prof.rolling(window=window, center=True, min_periods=1).mean()
        intra_noise = float((rep_prof - smoothed).std())
        mean_std = intra_noise  # store noise level so it appears in the UI
        if intra_noise < SCORE_EPSILON:
            score = amplitude  # perfectly smooth data: score is raw amplitude
        else:
            score = amplitude / intra_noise
    else:
        # Multi-trace: score = signal range / between-trace variability (SNR)
        score = amplitude / mean_std if mean_std > SCORE_EPSILON else amplitude

    score_dict = {"mean_std": mean_std, "amplitude": amplitude, "score": score}
    return interpolated_df, score_dict, None, col_warnings


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

    return output_data, scores, warnings


# ---------------------------------------------------------------------------
# Plot helpers (return Figure objects, never touch global pyplot state)
# ---------------------------------------------------------------------------

def make_overview_figure(
    output_data: dict[str, pd.DataFrame],
    scores: dict[str, dict],
    mode: str,
    file_name: str,
    opts: GraphOptions | None = None,
) -> plt.Figure:
    """All representative profiles on one axes."""
    if opts is None:
        opts = GraphOptions()

    visible = opts.selected_sheets if opts.selected_sheets else list(output_data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    y_label = MODES[mode]

    for sheet_name in visible:
        interp_df = output_data.get(sheet_name)
        if interp_df is None:
            continue
        rep_prof = interp_df.mean(axis=1, skipna=True)
        common_dist = interp_df.index.values
        sc = scores[sheet_name]["score"]
        if not np.all(np.isnan(rep_prof.values)):
            ax.plot(
                common_dist,
                rep_prof.values,
                label=f"{sheet_name} (score={sc:.2f})",
                linewidth=opts.line_width,
                linestyle=opts.line_style,
            )

    ax.set_xlabel(opts.x_label or "Distance (m)")
    ax.set_ylabel(opts.y_label or y_label)
    ax.set_title(opts.plot_title or f"Representative profiles [{mode}] — {file_name}")
    ax.legend(fontsize=8)
    ax.grid(opts.show_grid, alpha=0.4)
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
) -> plt.Figure:
    """Per-sheet plot: individual traces + mean +/- 1 sigma envelope."""
    if opts is None:
        opts = GraphOptions()

    fig, ax = plt.subplots(figsize=(10, 4))
    common_dist = interp_df.index.values
    rep_prof = interp_df.mean(axis=1, skipna=True).values
    std_prof = interp_df.std(axis=1, skipna=True).values

    # Individual traces (thin, semi-transparent)
    if opts.show_traces:
        for col in interp_df.columns:
            ax.plot(
                common_dist,
                interp_df[col].values,
                color="steelblue",
                alpha=0.25,
                linewidth=0.8,
                linestyle=opts.line_style,
            )

    # Mean profile
    ax.plot(
        common_dist,
        rep_prof,
        color="navy",
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
            color="navy",
            label="±1σ",
        )

    ax.set_xlabel(opts.x_label or "Distance (m)")
    ax.set_ylabel(opts.y_label or MODES[mode])
    ax.set_title(f"{sheet_name} — individual traces & representative profile")
    ax.legend()
    ax.grid(opts.show_grid, alpha=0.4)
    if opts.x_lim is not None:
        ax.set_xlim(opts.x_lim)
    if opts.y_lim is not None:
        ax.set_ylim(opts.y_lim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

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
            out_df.to_excel(writer, sheet_name=sheet_name[:EXCEL_SHEET_NAME_MAX], index=False)
    return buf.getvalue()


def build_scores_csv(scores: dict[str, dict]) -> bytes:
    df = pd.DataFrame.from_dict(scores, orient="index")
    df.index.name = "sheet"
    return df.to_csv().encode()


def build_batch_xlsx(
    all_results: list[dict],
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
    Sheet "Scores"   : one row per (file, mode, frequency) with Score/Amplitude/Noise
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
            for freq, sc in scores.items():
                score_rows.append({
                    "File": stem,
                    "Mode": mode,
                    "Frequency / Sheet": freq,
                    "Score": round(sc["score"], 4),
                    "Amplitude": round(sc["amplitude"], 6),
                    "Noise (σ)": round(sc["mean_std"], 6),
                })

            # ── Interpolated data sheet ──────────────────────────────────
            if not output_data:
                continue

            # All frequencies share the same distance grid within a file/mode
            first_df = next(iter(output_data.values()))
            dist_col = first_df.index.values

            data_sheet: dict[str, np.ndarray] = {"Distance (m)": dist_col}
            for freq, interp_df in output_data.items():
                rep_prof = interp_df.mean(axis=1, skipna=True).values
                col_label = f"{freq[:20]}_mean"
                data_sheet[col_label] = rep_prof

            data_df = pd.DataFrame(data_sheet)

            # Sheet name: "{stem}_{mode}", truncated to 31 chars
            raw_sheet = f"{stem}_{mode}"
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
) -> bytes:
    """
    Run every interpolation method against every uploaded file and pack all
    interpolated profiles and scores into a single xlsx.

    Workbook layout
    ---------------
    Sheet "Scores"            : File | Mode | Frequency | Method | Score | Amplitude | Noise(σ)
    "{stem}_{mode}_{method}"  : Distance (m) + one {freq}_mean column per frequency
    """
    buf = io.BytesIO()
    score_rows: list[dict] = []

    def _unique_sheet(name: str, existing: "KeysView[str]") -> str:
        """Truncate to 31 chars and deduplicate."""
        candidate = name[:EXCEL_SHEET_NAME_MAX]
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
                    out, sc, _ = process_gem_file(file_bytes, file_name, distance_step, method)
                    entries = [
                        (mode_key, out[mode_key], sc[mode_key])
                        for mode_key in ("EC", "MS")
                        if out.get(mode_key)
                    ]
                else:
                    out, sc, _ = process_file(file_bytes, mode, distance_step, method)
                    entries = [(mode, out, sc)] if out else []

                for mode_key, output_data, scores in entries:
                    # Scores rows
                    for freq, metrics in scores.items():
                        score_rows.append({
                            "File": stem,
                            "Mode": mode_key,
                            "Frequency / Sheet": freq,
                            "Method": method,
                            "Score": round(metrics["score"], 4),
                            "Amplitude": round(metrics["amplitude"], 6),
                            "Noise (σ)": round(metrics["mean_std"], 6),
                        })

                    # Data sheet: Distance + one mean column per frequency
                    if not output_data:
                        continue
                    first_df = next(iter(output_data.values()))
                    data: dict[str, np.ndarray] = {"Distance (m)": first_df.index.values}
                    for freq, interp_df in output_data.items():
                        data[f"{freq[:18]}_mean"] = interp_df.mean(axis=1, skipna=True).values

                    raw = f"{stem[:10]}_{mode_key}_{method}"
                    sheet_name = _unique_sheet(raw, writer.sheets.keys())
                    pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)

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
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Graph editor UI helper
# ---------------------------------------------------------------------------

def render_graph_editor(
    output_data: dict[str, pd.DataFrame],
    file_key: str,
) -> GraphOptions:
    """
    Render the graph editor expander and return the current GraphOptions.

    Parameters
    ----------
    output_data : processed sheets for this file
    file_key    : unique string used to namespace widget keys per file
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

    with st.expander("Graph Editor", expanded=False):
        col_sheets, col_style, col_axes = st.columns([2, 1, 2])

        with col_sheets:
            st.markdown("**Frequencies / Sheets**")
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
            st.markdown("**X axis**")
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

            st.markdown("**Y axis**")
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
        st.markdown("**Labels & Title**")
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
            placeholder=f"{next(iter(MODES.values()))} …",
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
) -> None:
    """
    Render ranking table, graph editor, plots, and downloads for one mode.

    Parameters
    ----------
    output_data : {freq_or_sheet_name: interpolated_df}
    scores      : {freq_or_sheet_name: score_dict}
    mode        : "EC" or "MS"
    file_name   : original uploaded filename (for plot titles)
    file_key    : unique string for Streamlit widget key namespacing
    """
    if not output_data:
        st.error("No usable frequencies / sheets found.")
        return

    # ── Ranking table ───────────────────────────────────────────────────
    ranking = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    if not ranking:
        st.error("No sheets could be scored.")
        return

    stem = Path(file_name).stem

    rank_df = pd.DataFrame(
        [
            {
                "Rank": i,
                "Frequency / Sheet": name,
                "Score": round(m["score"], 2),
                "Amplitude": round(m["amplitude"], 6),
                "Noise (σ)": round(m["mean_std"], 6),
            }
            for i, (name, m) in enumerate(ranking, 1)
        ]
    )

    col_table, col_best = st.columns([3, 1])
    with col_table:
        st.markdown("### Frequency Ranking")
        st.dataframe(
            rank_df.style.background_gradient(subset=["Score"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )
    with col_best:
        best_name, best_metrics = ranking[0]
        st.metric(
            "Best frequency",
            best_name,
            f"score {best_metrics['score']:.2f}",
            help=(
                "Score = amplitude / noise (σ). "
                "Multi-trace: noise = population std across traces (ddof=0). "
                "Single-trace: noise = residual std after removing the trend "
                "with a rolling-window smoother. "
                "If noise is zero, score equals the raw amplitude. "
                "Higher score = cleaner, larger signal."
            ),
        )
        st.metric("Amplitude", f"{best_metrics['amplitude']:.4g}")
        st.metric("Noise (σ)", f"{best_metrics['mean_std']:.4g}")

    # ── Graph editor ─────────────────────────────────────────────────────
    opts = render_graph_editor(output_data, file_key=file_key)

    # ── Overview plot ────────────────────────────────────────────────────
    st.markdown("### All representative profiles")
    overview_fig = make_overview_figure(output_data, scores, mode, file_name, opts)
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
            st.markdown(
                f"**{sheet_name}** — score `{sc['score']:.2f}` | "
                f"amp `{sc['amplitude']:.4g}` | std `{sc['mean_std']:.4g}`"
            )
            sheet_fig = make_sheet_figure(sheet_name, interp_df, mode, opts)
            st.pyplot(sheet_fig, use_container_width=True)
            plt.close(sheet_fig)

    # ── Downloads ────────────────────────────────────────────────────────
    st.markdown("### Downloads")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            label="Interpolated profiles (.xlsx)",
            data=build_excel_download(output_data),
            file_name=f"{stem}_{mode}_interpolated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_xlsx_{file_key}",
        )

    with dl2:
        st.download_button(
            label="Scores (.csv)",
            data=build_scores_csv(scores),
            file_name=f"{stem}_{mode}_scores.csv",
            mime="text/csv",
            key=f"dl_csv_{file_key}",
        )

    with dl3:
        download_fig = make_overview_figure(output_data, scores, mode, file_name, opts)
        png_bytes = fig_to_png(download_fig)
        plt.close(download_fig)
        st.download_button(
            label="Overview plot (.png)",
            data=png_bytes,
            file_name=f"{stem}_{mode}_overview.png",
            mime="image/png",
            key=f"dl_png_{file_key}",
        )


# ---------------------------------------------------------------------------
# 2D contouring (area maps & pseudo-sections)
# ---------------------------------------------------------------------------

def gem_value_column(mode: str, freq_label: str) -> str:
    """Raw GEM column name for a mode and frequency label such as '4525Hz'."""
    return f"EC{freq_label}[mS/m]" if mode == "EC" else f"MSusc{freq_label}[1/1000]"


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
    }


@st.cache_data(show_spinner=False)
def compute_area_map_cached(
    file_bytes: bytes,
    file_name: str,
    value_col: str,
    **params,
) -> ctr.AreaMapResult:
    """Cached wrapper around contouring.compute_area_map; *params* from grid_params()."""
    return ctr.compute_area_map(read_raw_table(file_bytes, file_name), value_col, **params)


def render_contouring_sidebar() -> ContourSettings:
    """Sidebar controls for 2D contouring; call inside `with st.sidebar`."""
    st.divider()
    st.subheader("2D contouring")
    area = st.toggle("Area map (plan view)", value=False, key="ct_area")
    pseudo = st.toggle("Pseudo-section", value=False, key="ct_pseudo")
    if not area:
        return ContourSettings(area_map=False, pseudosection=pseudo)

    coord_mode = st.selectbox(
        "Coordinates", ["auto", "metres", "degrees"],
        format_func=str.capitalize, key="ct_coords",
        help="Auto: Lat/Lon columns are degrees; X/Y are degrees only if they "
             "look like lon/lat spanning < 0.05°.",
    )
    method = st.selectbox(
        "Gridding method", list(ctr.METHODS), format_func=ctr.METHODS.get, key="ct_method",
    )
    smoothing = 0.0
    variogram_model = "spherical"
    if method == "spline":
        smoothing = st.number_input(
            "Spline smoothing", min_value=0.0, value=0.0, step=0.1, key="ct_smooth",
            help="0 = exact interpolation; larger values trade fit for smoothness.",
        )
    if method == "kriging":
        variogram_model = st.selectbox(
            "Variogram model", ctr.VARIOGRAM_MODELS, key="ct_vario",
        )
    cell = st.number_input(
        "Cell size (m, 0 = auto)", min_value=0.0, value=0.0, step=0.1,
        format="%.2f", key="ct_cell",
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
    n_levels = st.slider("Contour levels", 5, 50, 20, key="ct_levels")
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
        n_levels=int(n_levels),
    )


def _render_pseudosection(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    is_gem: bool,
) -> None:
    st.markdown("### Pseudo-section")
    profiles = {name: df.mean(axis=1, skipna=True) for name, df in output_data.items()}
    try:
        ps = ctr.build_pseudosection(profiles)
        fig = ctr.make_pseudosection_figure(
            ps, ctr.value_label(mode, is_gem),
            f"Pseudo-section [{mode}] — {file_name}", contour.n_levels,
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
) -> None:
    st.markdown("### Area map")
    if not is_gem:
        st.info(
            "Area maps need a GEM file with X/Y or Lat/Lon coordinates; "
            "the legacy multi-sheet format has none."
        )
        return

    freq = st.selectbox("Frequency", list(output_data.keys()), key=f"ct_freq_{file_key}")
    label = ctr.value_label(mode, is_gem)
    try:
        with st.spinner("Gridding…"):
            result = compute_area_map_cached(
                file_bytes, file_name, gem_value_column(mode, freq),
                **grid_params(contour),
            )
        fig = ctr.make_area_map_figure(
            result, label, f"{freq} [{mode}] — {ctr.METHODS[result.method]}",
            contour.n_levels,
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
                )
            except ctr.ContouringError as exc:
                st.error(str(exc))
            else:
                st.write(
                    f"RMSE **{cv['rmse']:.4g}**, MAE **{cv['mae']:.4g}** "
                    f"({label}, {cv['n']:,} held-out points; random folds, "
                    "optimistic for densely sampled lines)"
                )

    stem = Path(file_name).stem
    base = f"{stem}_{mode}_{freq}_{result.method}"
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Area map (.png)", data=png, file_name=f"{base}.png",
        mime="image/png", key=f"dl_am_png_{file_key}",
    )
    c2.download_button(
        "Grid (.csv)", data=ctr.grid_to_csv(result), file_name=f"{base}.csv",
        mime="text/csv", key=f"dl_am_csv_{file_key}",
    )
    c3.download_button(
        "Grid (.asc)", data=ctr.grid_to_asc(result), file_name=f"{base}.asc",
        mime="text/plain", key=f"dl_am_asc_{file_key}",
    )
    if result.origin is not None:
        st.warning(
            "Input was in degrees: the .asc grid is in local metres and is not "
            "georeferenced. Use the lon/lat columns of the CSV to place it."
        )


def render_contouring(
    output_data: dict[str, pd.DataFrame],
    mode: str,
    file_name: str,
    file_key: str,
    contour: ContourSettings,
    file_bytes: bytes,
    is_gem: bool,
) -> None:
    """Render the enabled 2D contouring sections for one file and mode."""
    if not output_data:
        return
    if contour.pseudosection:
        _render_pseudosection(output_data, mode, file_name, file_key, contour, is_gem)
    if contour.area_map:
        _render_area_map(output_data, mode, file_name, file_key, contour, file_bytes, is_gem)


# ---------------------------------------------------------------------------
# Per-format rendering dispatchers
# ---------------------------------------------------------------------------

def render_legacy_results(
    file_bytes: bytes,
    file_name: str,
    mode: str,
    distance_step: float,
    interp_kind: str,
    contour: ContourSettings | None = None,
) -> None:
    """Process and render a legacy multi-sheet Excel file."""
    stem = Path(file_name).stem

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

    _render_mode_section(output_data, scores, mode, file_name, file_key=stem)
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
) -> None:
    """Process and render a GEM instrument file (CSV or XLSX)."""
    stem = Path(file_name).stem

    with st.spinner("Processing GEM file…"):
        output_data, scores, warnings = process_gem_file(
            file_bytes, file_name,
            distance_step=distance_step,
            interp_kind=interp_kind,
        )

    if warnings:
        with st.expander(f"{len(warnings)} warning(s)", expanded=False):
            for w in warnings:
                st.warning(w)

    # Determine which modes have data
    available_modes = [m for m in ("EC", "MS") if output_data.get(m)]

    if not available_modes:
        st.error("No usable frequencies found in this GEM file.")
        if distance_step > 1.0:
            st.info("Try reducing the distance step in the sidebar.")
        return

    st.caption("GEM format detected — showing all frequencies for both EC and MS")

    tabs = st.tabs([f"{m} ({len(output_data[m])} frequencies)" for m in available_modes])

    for tab, mode_key in zip(tabs, available_modes):
        with tab:
            _render_mode_section(
                output_data[mode_key],
                scores[mode_key],
                mode_key,
                file_name,
                file_key=f"{stem}_{mode_key}",
            )
            if contour is not None:
                render_contouring(
                    output_data[mode_key], mode_key, file_name,
                    f"{stem}_{mode_key}", contour, file_bytes, is_gem=True,
                )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="UltraRank Frequency",
        page_icon=None,
        layout="wide",
    )

    st.title("Representative Incision Tool")
    st.caption("Geophysical representative profile builder — EC / MS modes")

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

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
            format="%.2f",
        )

        interp_kind = st.selectbox(
            "Interpolation method",
            ["linear", "cubic", "nearest", "quadratic", "pchip", "akima", "polynomial"],
            index=0,
        )

        if distance_step > 10.0:
            st.warning(
                f"Distance step is {distance_step:.2f} m. "
                "If your data spans less than this, all sheets will be skipped."
            )

        contour = render_contouring_sidebar()

        st.divider()
        st.markdown("**Output files are available for download after processing.**")

        st.divider()
        with st.expander("About", expanded=False):
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
| **In-phase** | Apparent magnetic susceptibility (MS) | ×10⁻⁵ SI (dimensionless) |

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
| **A** | Signal amplitude — the peak-to-trough range of the mean profile: max(p̄) − min(p̄). Captures the total geophysical contrast resolved at that frequency. |
| **σ_noise** | Noise — estimated from the data depending on how many passes were acquired (see below). |

A higher score means the frequency resolves large subsurface
contrasts clearly above the noise level. This is equivalent to a
high signal-to-noise ratio (SNR), the standard geophysical
criterion for data quality (Sheriff & Geldart, 1995).

**How noise is estimated:**

**With ≥ 2 passes (recommended):**
σ is the mean of the point-wise population standard deviation
computed across all traces at each distance step. Because the
traces represent the complete set of survey passes — not a
sample from a larger population — the population formula
(ddof = 0) is appropriate. This σ captures all sources of
between-pass variability: instrument noise, positioning
uncertainty, and short-term drift.

**With 1 pass only (single-trace fallback):**
Between-trace σ is undefined, so the tool estimates intra-profile
noise by subtracting a rolling-mean smoother (window =
max(5, N/10) samples) from the profile and computing the
standard deviation of the residuals. This decomposes the signal
into a slowly varying geological trend and a high-frequency
noise component, using only the latter as σ. This approach is
analogous to the residual-noise estimators used in single-channel
seismic quality assessment (Bakulin et al., 2022). Note that
single-trace scores are less reliable — multiple passes are
always preferable.

**Near-zero noise fallback:**
When σ < 10⁻⁸ (effectively zero — e.g. a perfectly flat synthetic
trace), the score collapses to the raw amplitude A to avoid
numerical instability. This edge case does not arise with real
field data.

---

#### Processing pipeline

| Step | What happens |
|---|---|
| **1. Ingest** | File is read (CSV or XLSX). GEM format is auto-detected from column names (`Line`, `Y`, `EC*Hz[mS/m]`, `MSusc*Hz[1/1000]`). The flat GEM table is pivoted: each frequency becomes a matrix with distance as rows and survey lines as columns. |
| **2. Interpolate** | All traces are resampled onto a common evenly-spaced distance grid (`np.linspace`). The interpolation method is chosen from the sidebar (see *Interpolation methods* below). Duplicate distance values are averaged before interpolation. |
| **3. Score** | The mean profile and noise metric are computed as above. Frequencies are ranked by descending score. |
| **4. Visualise** | An overview plot shows all mean profiles together. Per-frequency plots show individual traces (thin, semi-transparent), the mean profile (bold), and the ±1σ envelope. |
| **5. Export** | Per-file downloads (interpolated profiles XLSX, scores CSV, overview PNG) and a **Batch Export** that packages results from all uploaded files into a single XLSX. A second batch option runs all interpolation methods simultaneously and exports every result for direct comparison. |

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
| **polynomial** | 3 | Global least-squares polynomial fit (degree = min(n − 1, 5)). Suitable for very smooth, low-point-count profiles; avoid for long profiles where Runge oscillations can appear. |

The **Batch Export — all methods** option runs all seven methods in one step and writes a single XLSX whose `Scores` sheet lists every (file, mode, frequency, method) combination side-by-side for direct comparison.

---

#### 2D contouring

Switch on in the sidebar under **2D contouring**.

- **Pseudo-section** — mean profiles of all frequencies as one
  distance × frequency contour, aligned on their common distance
  range. The frequency axis is *not* depth: under LIN, depth
  sensitivity is set by coil geometry (Huang, 2005).
- **Area map** — plan-view grid of one frequency from the X/Y (or
  Lat/Lon) coordinates of all lines: optional per-line median
  levelling (Mauring & Kihle, 2006), block-median reduction, then
  thin-plate spline (Briggs, 1974; Sandwell, 1987), ordinary
  kriging with a fitted variogram and standard-deviation map
  (Lesch et al., 1995; Corwin & Lesch, 2005; Oliver & Webster,
  2014) or linear triangulation. Nodes far from data are blanked.
  Use **Cross-validate** to compare methods (Li & Heap, 2011).

---

#### Supported file formats

| Format | Notes |
|---|---|
| **GEM-2 `.xlsx`** | Full instrument precision — recommended for scoring |
| **GEM-2 `.csv`** | EC rounded to integers; may slightly affect scores |
| **Legacy `.xlsx`** | One sheet per frequency; column 0 = distance (m), columns 1+ = survey traces |

---

#### Glossary

| Term | Definition |
|---|---|
| **EC** | Apparent electrical conductivity (mS/m) — quadrature EMI response |
| **MS** | Apparent magnetic susceptibility (×10⁻⁵ SI) — in-phase EMI response |
| **EMI** | Frequency-domain electromagnetic induction |
| **LIN** | Low induction number approximation — the condition under which EC and MS decouple linearly (McNeill, 1980) |
| **Skin depth (δ)** | Depth at which primary field amplitude falls to 1/e; decreases with frequency and conductivity |
| **GEM-2** | Multi-frequency broadband EMI sensor (Won et al., 1996) |
| **SNR** | Signal-to-noise ratio |
| **Score** | A / σ_noise — the representativeness metric used for frequency ranking |
| **Amplitude (A)** | max − min of the mean profile across all passes |
| **σ_noise** | Population std across traces (multi-pass) or residual std from smoother (single-pass) |
| **ddof = 0** | Population standard deviation; used because the survey passes are the full measurement ensemble, not a sample |
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
  techniques: 1. *Water Resour. Res.*, **31**(2), 373–386.
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
                file_bytes, file_name, distance_step, interp_kind
            )
            for mode_key in ("EC", "MS"):
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
        st.markdown("### Batch Export")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.caption(
                f"Interpolated profiles & scores for the **selected method** "
                f"({interp_kind}) across {len(uploaded_files)} file(s)."
            )
            batch_bytes = build_batch_xlsx(batch_results)
            st.download_button(
                label=f"Download — {interp_kind} method (.xlsx)",
                data=batch_bytes,
                file_name=f"batch_{interp_kind}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_batch",
            )

        with dl_col2:
            st.caption(
                f"Interpolated profiles & scores for **all {len(ALL_INTERP_METHODS)} methods** "
                f"across {len(uploaded_files)} file(s). May take a moment to generate."
            )
            if st.button("Generate all-methods export", key="btn_all_methods"):
                with st.spinner(
                    f"Running {len(ALL_INTERP_METHODS)} interpolation methods "
                    f"across {len(uploaded_files)} file(s)…"
                ):
                    all_methods_bytes = build_all_methods_batch_xlsx(
                        uploaded_files, mode, distance_step
                    )
                st.download_button(
                    label="Download — all methods (.xlsx)",
                    data=all_methods_bytes,
                    file_name="batch_all_methods.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_all_methods",
                )

    # ── Per-file detailed results ───────────────────────────────────────────
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

        if is_gem:
            render_gem_results(file_bytes, file_name, distance_step, interp_kind, contour)
        else:
            render_legacy_results(
                file_bytes, file_name, mode, distance_step, interp_kind, contour
            )


if __name__ == "__main__":
    main()