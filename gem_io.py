"""
GEM-2 export format: channel columns, quality flags, event markers and
distance along each survey line.

Pure functions only (no Streamlit). Column names follow the WinGEM export
(GEM-2 Manual v3.8, 2004): Line Sample X Y Mark Status GPSStat Time[ms]
Time[hhmmss.sss] PowerLn I_<f>Hz Q_<f>Hz QSum EC<f>Hz[mS/m] TotalEC[mS/m]
MSusc<f>Hz[1/1000].
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import contouring as ctr

CHANNEL_PATTERNS = {
    "EC": re.compile(r"^EC(\d+(?:\.\d+)?)Hz\[mS/m\]$"),
    "MS": re.compile(r"^MSusc(\d+(?:\.\d+)?)Hz\[1/1000\]$"),
    "I": re.compile(r"^I_(\d+(?:\.\d+)?)Hz$"),
    "Q": re.compile(r"^Q_(\d+(?:\.\d+)?)Hz$"),
}
AUX_CHANNELS = tuple(ctr.AUX_LABELS)          # PowerLn, QSum, TotalEC[mS/m]
GEM_MODES = ("EC", "MS", "I", "Q", "AUX")
FREQUENCY_MODES = ("EC", "MS", "I", "Q")      # modes whose channels are frequencies
MODE_NAMES = {
    "EC": "Electrical conductivity",
    "MS": "Magnetic susceptibility",
    "I": "In-phase",
    "Q": "Quadrature",
    "AUX": "Auxiliary channels",
}

DISTANCE_COL = "Distance (m)"   # column added to prepared tables (pipeline.prepare_gem_table)

DISTANCE_METHODS = {
    "Y": "Y column",
    "projection": "Projection on the survey axis",
    "path": "Path length along each line",
    "sample": "Reading number × spacing",
    "markers": "Between event markers (dead reckoning)",
}


def find_channels(columns) -> dict[str, dict[str, str]]:
    """{mode: {label: column}} for every recognised channel column, e.g. {"EC": {"4525Hz": "EC4525Hz[mS/m]"}}."""
    out: dict[str, dict[str, str]] = {m: {} for m in GEM_MODES}
    for col in columns:
        name = str(col)
        for mode, pattern in CHANNEL_PATTERNS.items():
            match = pattern.match(name)
            if match:
                out[mode][f"{match.group(1)}Hz"] = name
                break
        else:
            if name in AUX_CHANNELS:
                out["AUX"][name] = name
    return out


def channel_column(mode: str, label: str) -> str:
    """Raw column name for a mode and channel label such as '4525Hz'."""
    if mode == "EC":
        return f"EC{label}[mS/m]"
    if mode == "MS":
        return f"MSusc{label}[1/1000]"
    if mode in ("I", "Q"):
        return f"{mode}_{label}"
    if mode == "AUX":
        return label
    raise ValueError(f"Unknown mode: {mode!r}")


def time_seconds(df: pd.DataFrame) -> np.ndarray | None:
    """
    Time of each reading in seconds since the first midnight, from Time[ms]
    (milliseconds of the day) or Time[hhmmss.sss]; None if neither column
    exists. Passing midnight adds a day so time keeps increasing.
    """
    if "Time[ms]" in df.columns:
        t = pd.to_numeric(df["Time[ms]"], errors="coerce").to_numpy(dtype=float) / 1000.0
    elif "Time[hhmmss.sss]" in df.columns:
        v = pd.to_numeric(df["Time[hhmmss.sss]"], errors="coerce").to_numpy(dtype=float)
        hours = np.floor(v / 10000.0)
        minutes = np.floor(v / 100.0) % 100.0
        t = hours * 3600.0 + minutes * 60.0 + (v - hours * 10000.0 - minutes * 100.0)
    else:
        return None
    steps = np.diff(pd.Series(t).ffill().bfill().to_numpy())   # NaN never fakes midnight
    days = np.concatenate([[0.0], np.cumsum(steps < -43200.0)])
    return t + 86400.0 * days


def drop_flagged_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes readings with a non-zero Status (e.g. ADC overload). Returns (table, number removed)."""
    if "Status" not in df.columns:
        return df, 0
    status = pd.to_numeric(df["Status"], errors="coerce").fillna(0)
    bad = (status != 0).to_numpy()
    return df.loc[~bad].reset_index(drop=True), int(bad.sum())


def marker_rows(df: pd.DataFrame, line_col: str = "Line") -> np.ndarray:
    """
    Boolean mask of event-marker readings: the Mark value differs from the
    previous reading of the same line and is not 0. Works whether Mark is a
    running counter or is set only on the flagged reading. A marker on the
    first reading of a line cannot be told apart from a running counter's
    starting value, so it is not detected.
    """
    if "Mark" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    mark = pd.to_numeric(df["Mark"], errors="coerce")
    previous = mark.groupby(df[line_col], sort=False).shift(1)
    return (mark.notna() & previous.notna() & (mark != previous) & (mark != 0)).to_numpy()


def marker_distances(df: pd.DataFrame, distance: np.ndarray, line_col: str = "Line") -> list[float]:
    """Sorted, de-duplicated distances (mm resolution) of event-marker readings."""
    d = np.asarray(distance, dtype=float)[marker_rows(df, line_col)]
    return sorted(set(np.round(d[np.isfinite(d)], 3).tolist()))


def _local_xy(df: pd.DataFrame, coord_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in metres; GPS no-fix rows (0, 0 in degrees) become NaN."""
    x_col, y_col, is_deg = ctr.find_coordinate_columns(df, coord_mode)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if is_deg:
        nofix = (x == 0) & (y == 0)
        x[nofix] = np.nan
        y[nofix] = np.nan
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            x[ok], y[ok], _ = ctr.project_to_local_metres(x[ok], y[ok])
    return x, y


def _line_axis(x: np.ndarray, y: np.ndarray, ok: np.ndarray, df: pd.DataFrame,
               line_col: str) -> np.ndarray | None:
    """
    Mean direction of the survey lines: the main axis of each line, flipped to
    agree with the lines before it and weighted by its number of readings.
    None when no line has two distinct positions.
    """
    total = None
    for idx in df.groupby(line_col, sort=False).indices.values():
        i = idx[ok[idx]]
        if len(i) < 2:
            continue
        pts = np.column_stack([x[i], y[i]])
        pts = pts - pts.mean(axis=0)
        if not np.any(pts):
            continue
        _, vecs = np.linalg.eigh(pts.T @ pts)
        v = vecs[:, -1] * len(i)
        if total is not None and v @ total < 0:
            v = -v
        total = v if total is None else total + v
    if total is None or not np.any(total):
        return None
    return total / np.linalg.norm(total)


def _marker_distance(df: pd.DataFrame, spacing: float, line_col: str) -> np.ndarray:
    """
    Dead reckoning: the k-th event marker of a line is at k * spacing and
    readings between markers are spaced evenly. Readings before the first or
    after the last marker of a line are left blank (NaN).
    """
    out = np.full(len(df), np.nan)
    marks = marker_rows(df, line_col)
    for idx in df.groupby(line_col, sort=False).indices.values():
        fid = idx[marks[idx]]
        if len(fid) < 2:
            continue
        pos = np.arange(len(idx))
        at = np.searchsorted(idx, fid)
        inside = (pos >= at[0]) & (pos <= at[-1])
        out[idx[inside]] = np.interp(pos[inside], at, np.arange(len(fid)) * spacing)
    return out


def along_track_distance(
    df: pd.DataFrame,
    method: str = "Y",
    spacing: float = 1.0,
    coord_mode: str = "auto",
    line_col: str = "Line",
) -> np.ndarray:
    """
    Distance of every reading along its survey line, in metres.

    Y          : the Y column as exported (WinGEM grid surveys).
    projection : coordinates projected on the mean direction of the survey
                 lines, so repeat passes walked in either direction share
                 distances.
    path       : cumulative path length from each line's first reading.
    sample     : reading number within the line x spacing.
    markers    : dead reckoning between event markers placed `spacing` apart.
    """
    if method == "Y":
        return pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=float)
    if method == "sample":
        return df.groupby(line_col, sort=False).cumcount().to_numpy(dtype=float) * spacing
    if method == "markers":
        return _marker_distance(df, spacing, line_col)
    if method not in ("projection", "path"):
        raise ValueError(f"Unknown distance method: {method!r}")

    x, y = _local_xy(df, coord_mode)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        raise ctr.ContouringError("Not enough coordinates to compute distances.")
    out = np.full(len(df), np.nan)
    if method == "projection":
        pts = np.column_stack([x[ok], y[ok]])
        centred = pts - pts.mean(axis=0)
        axis = _line_axis(x, y, ok, df, line_col)
        if axis is None:                          # no line has two positions
            _, vecs = np.linalg.eigh(np.cov(centred.T))
            axis = vecs[:, -1]
        if axis[np.argmax(np.abs(axis))] < 0:     # increase eastward / northward
            axis = -axis
        proj = centred @ axis
        out[ok] = proj - proj.min()
        return out
    for idx in df.groupby(line_col, sort=False).indices.values():
        i = idx[ok[idx]]
        if len(i) == 0:
            continue
        step = np.hypot(np.diff(x[i]), np.diff(y[i]))
        out[i] = np.concatenate([[0.0], np.cumsum(step)])
    return out


def line_labels(lines: pd.Series) -> pd.Series:
    """Line labels as text, whole-number floats without '.0' (7.0 -> '7'), as typed by users."""
    def label(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    return lines.map(label)
