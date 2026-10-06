"""
Corrections and filters applied to a GEM table before profiles and maps:
despiking, clipping, smoothing, sensor-height (altitude) correction,
base-station drift and temperature drift, background offset, calibration
against reference data, EC at 25 °C, PCA noise reduction, GPS time-lag
correction and heading filter.

Pure functions only (no Streamlit). apply_corrections() runs the enabled
steps in a fixed order and reports each one in plain language.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

import contouring as ctr
import gem_io

LAG_SEARCH = np.round(np.arange(-2.0, 2.0001, 0.1), 3)   # s, candidate GPS lags
LAG_MAX_POINTS = 5000                                    # subsample for the lag search
LAG_NEIGHBOURS = 64                                      # first neighbour search for another line
MIN_OCCUPATIONS = 2                                      # base-station visits for drift
MIN_TEMPERATURE_OCCUPATIONS = 3                          # ... for a temperature coefficient


@dataclass(frozen=True)
class CorrectionSettings:
    """Enabled corrections and their parameters (hashable: part of the cache key)."""
    despike: bool = False
    despike_window: int = 5                  # readings
    despike_threshold: float = 4.0           # x robust sigma
    clip_percentiles: tuple[float, float] | None = None
    altitude_column: str = ""                # "" = off
    altitude_reference: float | None = None  # m; None = median altitude
    temperature_column: str = ""             # "" = off (needs base-station lines)
    drift_lines: tuple[str, ...] = ()        # Line labels of base-station occupations
    drift_model: str = "piecewise"           # "piecewise" | "linear"
    iq_offsets: bytes | None = None          # CSV column,offset (e.g. multi-height calibration)
    ec_background: float | None = None       # mS/m; shift EC channels so their median matches
    reference: bytes | None = None           # CSV of reference values (calibration)
    reference_radius: float = 2.0            # m, matching distance for reference points
    reference_method: str = "regression"     # "regression" | "moments"
    soil_temperature: float | None = None    # °C; converts EC to EC at 25 °C
    pca_components: int = 0                  # 0 = off
    smooth_window: int = 0                   # readings; 0 = off
    lag_seconds: float = 0.0                 # GPS lag; positive = data recorded late
    bearing_center: float | None = None      # degrees from north; None = off
    bearing_tolerance: float = 30.0          # degrees either side

    @property
    def active(self) -> bool:
        return self != CorrectionSettings()

    @property
    def lowers_noise(self) -> bool:
        """True if a step smooths the data (lowers σ and inflates scores)."""
        return self.despike or self.smooth_window > 1 or self.pca_components > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def channel_columns(df: pd.DataFrame, modes=gem_io.FREQUENCY_MODES) -> list[str]:
    """Columns of the frequency channels (EC, MS, I, Q) present in the table."""
    found = gem_io.find_channels(df.columns)
    return [col for m in modes for col in found[m].values()]


def _by_line(df: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each line, in table order."""
    return list(df.groupby("Line", sort=False).indices.values())


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _require_time(df: pd.DataFrame, step: str) -> np.ndarray:
    t = gem_io.time_seconds(df)
    if t is None:
        raise ValueError(f"{step} needs a Time[ms] or Time[hhmmss.sss] column.")
    return t


# ---------------------------------------------------------------------------
# Single-series filters
# ---------------------------------------------------------------------------


def robust_noise(v: np.ndarray) -> float:
    """
    Noise sigma of a series from its second differences, which cancel a
    locally linear trend: 1.4826 x MAD(diff2) / sqrt(6) (the single-trace
    estimator of the scoring). NaN when there are fewer than 5 values.
    """
    d2 = np.diff(np.asarray(v, dtype=float)[np.isfinite(v)], n=2)
    if d2.size < 3:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6.0)


def despike_series(v: np.ndarray, window: int = 5, threshold: float = 4.0) -> tuple[np.ndarray, int]:
    """
    Blanks readings that differ from their running median (`window` readings)
    by more than threshold x the robust noise sigma of the series. Readings
    within half a window of either end are not tested (their window is
    one-sided, so a linear trend alone would look like a spike).
    Returns (values with spikes as NaN, number of spikes).
    """
    s = pd.Series(np.asarray(v, dtype=float))
    med = s.rolling(window, center=True, min_periods=window).median()
    dev = (s - med).abs()
    scale = robust_noise(s.to_numpy())
    if not np.isfinite(scale) or scale == 0:
        # Coarsely rounded data (e.g. integer EC in CSV exports): use the
        # rounding noise of the smallest step instead.
        steps = np.abs(np.diff(s.dropna().to_numpy()))
        steps = steps[steps > 0]
        if steps.size == 0:
            return s.to_numpy(), 0
        scale = float(steps.min()) / math.sqrt(12.0)
    spikes = (dev > threshold * scale).to_numpy()
    out = s.to_numpy().copy()
    out[spikes] = np.nan
    return out, int(spikes.sum())


def running_mean(v: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over `window` readings (shorter at the ends)."""
    return pd.Series(np.asarray(v, dtype=float)).rolling(
        window, center=True, min_periods=1
    ).mean().to_numpy()


def clip_to_percentiles(v: np.ndarray, low: float, high: float) -> tuple[np.ndarray, int]:
    """Blanks values outside the [low, high] percentiles. Returns (values, number blanked)."""
    v = np.asarray(v, dtype=float).copy()
    finite = np.isfinite(v)
    if not finite.any():
        return v, 0
    lo, hi = np.percentile(v[finite], [low, high])
    out = finite & ((v < lo) | (v > hi))
    v[out] = np.nan
    return v, int(out.sum())


def ec_at_25(ec: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apparent conductivity at 25 °C (Sheets & Hendrickx, 1995, as used by
    Corwin & Lesch, 2005): EC25 = EC x (0.4470 + 1.4034 exp(-T / 26.815)).
    """
    return np.asarray(ec, dtype=float) * (0.4470 + 1.4034 * math.exp(-temperature / 26.815))


# ---------------------------------------------------------------------------
# Sensor height
# ---------------------------------------------------------------------------


def _height_model(h, a, b, c):
    return a + b * np.exp(c * h)


def fit_height_response(h: np.ndarray, v: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Signal as a function of sensor height: a + b exp(c h) (Vilhelmsen &
    Døssing, 2022), falling back to a straight line when the fit fails.
    Returns (model name, parameters).
    """
    ok = np.isfinite(h) & np.isfinite(v)
    h, v = h[ok], v[ok]
    if h.size < 4 or np.ptp(h) == 0:
        raise ValueError("Not enough distinct sensor heights to fit a height correction.")
    slope, intercept = np.polyfit(h, v, 1)
    span = float(np.ptp(h))
    try:
        p0 = [float(np.median(v)), float(slope) * span, -1.0 / span]
        p, _ = curve_fit(_height_model, h, v, p0=p0, maxfev=20000)
        if np.all(np.isfinite(p)):
            return "exponential", np.asarray(p, dtype=float)
    except (RuntimeError, ValueError):
        pass
    return "linear", np.array([intercept, slope])


def height_response(model: str, params: np.ndarray, h) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    if model == "exponential":
        return _height_model(h, *params)
    return params[0] + params[1] * h


def correct_height(v: np.ndarray, h: np.ndarray, reference: float) -> tuple[np.ndarray, str]:
    """v - f(h) + f(reference), with f fitted by fit_height_response."""
    model, params = fit_height_response(np.asarray(h, float), np.asarray(v, float))
    corr = np.asarray(v, float) - height_response(model, params, h) + height_response(model, params, reference)
    return corr, model


# ---------------------------------------------------------------------------
# Base-station drift and temperature
# ---------------------------------------------------------------------------


def base_station_occupations(df: pd.DataFrame, lines: tuple[str, ...]) -> list[np.ndarray]:
    """Row positions of each base-station occupation (one per listed Line label present)."""
    labels = gem_io.line_labels(df["Line"])
    return [np.nonzero((labels == str(lb)).to_numpy())[0] for lb in lines
            if (labels == str(lb)).any()]


def drift_curve(t_occ: np.ndarray, v_occ: np.ndarray, t: np.ndarray, model: str) -> np.ndarray:
    """Drift at times t from occupation means: piecewise linear (flat beyond the ends) or a straight line."""
    if model == "linear":
        slope, intercept = np.polyfit(t_occ, v_occ, 1)
        return intercept + slope * t
    order = np.argsort(t_occ)
    return np.interp(t, t_occ[order], v_occ[order])


def correct_drift(
    df: pd.DataFrame, lines: tuple[str, ...], model: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes instrument drift measured by repeated base-station occupations
    (USGS GEM-2 practice: mean of each occupation, trend in time, subtract),
    relative to the first occupation, then drops the base-station lines.
    """
    t = _require_time(df, "Drift correction")
    occ = base_station_occupations(df, lines)
    if len(occ) < MIN_OCCUPATIONS:
        raise ValueError(
            f"Drift correction needs at least {MIN_OCCUPATIONS} base-station lines; "
            f"found {len(occ)} of {', '.join(lines)}."
        )
    out = df.copy()
    t_occ = np.array([np.nanmean(t[i]) for i in occ])
    first = int(np.argmin(t_occ))
    drifts = []
    for col in channel_columns(df):
        v = _numeric(df, col)
        v_occ = np.array([np.nanmean(v[i]) for i in occ])
        curve = drift_curve(t_occ, v_occ, t, model)
        out[col] = v - (curve - v_occ[first])
        drifts.append(f"{col}: {np.ptp(v_occ):.4g}")
    base = np.concatenate(occ)
    out = out.drop(index=out.index[base]).reset_index(drop=True)
    return out, [
        f"Drift corrected from {len(occ)} base-station occupation(s) ({model}); "
        f"base-station lines removed ({len(base)} readings). Drift range — " + "; ".join(drifts)
    ]


def correct_temperature(
    df: pd.DataFrame, column: str, lines: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes a linear temperature dependence estimated from base-station
    occupations: value = c0 + c1 x temperature at a fixed location, so
    c1 x (T - T_mean) is subtracted from every reading.
    """
    if column not in df.columns:
        raise ValueError(f"Temperature column '{column}' not found.")
    occ = base_station_occupations(df, lines)
    if len(occ) < MIN_TEMPERATURE_OCCUPATIONS:
        raise ValueError(
            f"Temperature correction needs at least {MIN_TEMPERATURE_OCCUPATIONS} "
            "base-station lines."
        )
    temp = _numeric(df, column)
    t_occ = np.array([np.nanmean(temp[i]) for i in occ])
    if np.ptp(t_occ) == 0:
        raise ValueError("Base-station temperatures do not vary; no coefficient can be fitted.")
    out = df.copy()
    coefs = []
    for col in channel_columns(df):
        v = _numeric(df, col)
        v_occ = np.array([np.nanmean(v[i]) for i in occ])
        c1 = float(np.polyfit(t_occ, v_occ, 1)[0])
        out[col] = v - c1 * (temp - t_occ.mean())
        coefs.append(f"{col}: {c1:.4g}/°C")
    return out, ["Temperature drift removed — " + "; ".join(coefs)]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def xy_metres(df: pd.DataFrame, origin=None):
    """(x, y, origin) in metres; degrees are projected about `origin` (or the data centroid)."""
    x_col, y_col, is_deg = ctr.find_coordinate_columns(df, "auto")
    x = _numeric(df, x_col)
    y = _numeric(df, y_col)
    if not is_deg:
        return x, y, None
    nofix = (x == 0) & (y == 0)
    x[nofix] = np.nan
    y[nofix] = np.nan
    px, py, origin = ctr.project_to_local_metres(x, y, origin)
    return px, py, origin


def fit_reference_calibration(
    df: pd.DataFrame, reference: pd.DataFrame, radius: float, method: str = "regression"
) -> list[dict]:
    """
    Gain and offset per channel so the measured values match reference
    values (Lavoué et al., 2010; Mester et al., 2011: apparent conductivity
    predicted from ERT; Dragonetti et al., 2018: TDR). The reference table
    has the same coordinate columns as the survey and one column per channel
    it calibrates (named like the survey column). Each reference point is
    matched to the median of the survey readings within `radius` metres.

    method "regression": least-squares line reference = gain x measured + offset.
    method "moments"   : gain and offset that match the mean and standard deviation.
    """
    x, y, origin = xy_metres(df)
    rx, ry, _ = xy_metres(reference, origin)
    ok = np.isfinite(x) & np.isfinite(y)
    tree = cKDTree(np.column_stack([x[ok], y[ok]]))
    rows = np.nonzero(ok)[0]
    near = tree.query_ball_point(np.column_stack([rx, ry]), r=radius)
    fits = []
    for col in channel_columns(df):
        if col not in reference.columns:
            continue
        v = _numeric(df, col)
        measured = np.array([np.nanmedian(v[rows[i]]) if i else np.nan for i in near])
        ref = _numeric(reference, col)
        pair = np.isfinite(measured) & np.isfinite(ref)
        if pair.sum() < 3:
            continue
        m, r = measured[pair], ref[pair]
        if np.std(m) == 0 or np.std(r) == 0:
            continue                      # constant values: no gain can be fitted
        if method == "moments":
            gain = float(np.std(r, ddof=1) / np.std(m, ddof=1))
            offset = float(np.mean(r) - gain * np.mean(m))
        else:
            gain, offset = (float(c) for c in np.polyfit(m, r, 1))
        resid = r - (gain * m + offset)
        r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((r - r.mean()) ** 2))
        fits.append({"column": col, "gain": gain, "offset": offset, "r2": r2, "n": int(pair.sum())})
    return fits


# ---------------------------------------------------------------------------
# PCA noise reduction
# ---------------------------------------------------------------------------


def pca_denoise(values: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
    """
    Keeps the first n principal components of standardised channels
    (Minsley et al., 2010: FDEM channels are strongly correlated, so the
    trailing components are mostly noise). Rows with any NaN are returned
    unchanged. Returns (values, fraction of variance kept).
    """
    v = np.asarray(values, dtype=float)
    out = v.copy()
    rows = np.all(np.isfinite(v), axis=1)
    if rows.sum() < 3 or n_components >= v.shape[1]:
        return out, 1.0
    a = v[rows]
    mean, std = a.mean(axis=0), a.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    z = (a - mean) / std
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    k = max(1, int(n_components))
    out[rows] = (u[:, :k] * s[:k]) @ vt[:k] * std + mean
    return out, float(np.sum(s[:k] ** 2) / np.sum(s ** 2))


# ---------------------------------------------------------------------------
# Positioning
# ---------------------------------------------------------------------------


def _coordinate_pairs(df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    """Every coordinate pair in the table as (x column, y column, is degrees): Lon/Lat, then X/Y."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    pairs = []
    lat = next((lower[n] for n in ctr.LAT_NAMES if n in lower), None)
    lon = next((lower[n] for n in ctr.LON_NAMES if n in lower), None)
    if lat is not None and lon is not None:
        pairs.append((lon, lat, True))
    if "x" in lower and "y" in lower:
        x, y = _numeric(df, lower["x"]), _numeric(df, lower["y"])
        pairs.append((lower["x"], lower["y"], ctr.looks_like_degrees(x, y)))
    return pairs


def _interp_extrapolate(t_new: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linear interpolation in sorted t that continues the first and last segments beyond the ends."""
    out = np.interp(t_new, t, v)
    if len(t) >= 2:
        before, after = t_new < t[0], t_new > t[-1]
        if before.any() and t[1] > t[0]:
            out[before] = v[0] + (t_new[before] - t[0]) * (v[1] - v[0]) / (t[1] - t[0])
        if after.any() and t[-1] > t[-2]:
            out[after] = v[-1] + (t_new[after] - t[-1]) * (v[-1] - v[-2]) / (t[-1] - t[-2])
    return out


def shift_positions(df: pd.DataFrame, t: np.ndarray, lag: float) -> pd.DataFrame:
    """
    Corrects a GPS/sensor time lag: a reading logged at time t was measured
    where the sensor was at t - lag, so its coordinates are replaced by the
    track position at t - lag (interpolated within each line, extrapolated
    along its end segments). Lat/Lon and X/Y are both shifted when present;
    GPS no-fix rows (0, 0 in degrees) are left as they are.
    """
    pairs = _coordinate_pairs(df)
    if not pairs:
        raise ctr.ContouringError("No coordinate columns (X/Y or Lat/Lon) found in this file.")
    out = df.copy()
    for x_col, y_col, is_deg in pairs:
        x, y = _numeric(df, x_col), _numeric(df, y_col)
        valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        if is_deg:
            valid &= ~((x == 0) & (y == 0))
        nx, ny = x.copy(), y.copy()
        for idx in _by_line(df):
            ok = idx[valid[idx]]
            if len(ok) < 2:
                continue
            order = ok[np.argsort(t[ok], kind="stable")]
            nx[ok] = _interp_extrapolate(t[ok] - lag, t[order], x[order])
            ny[ok] = _interp_extrapolate(t[ok] - lag, t[order], y[order])
        out[x_col] = nx
        out[y_col] = ny
    return out


def _cross_line_difference(
    x: np.ndarray, y: np.ndarray, v: np.ndarray, lines: np.ndarray, probe: np.ndarray
) -> float:
    """
    Mean |v - v_nearest| over probe readings, nearest = closest reading on
    another line. The neighbour search widens until such a reading is found,
    so dense sampling along a line cannot hide the neighbouring lines. Pairs
    farther apart than twice the median pair distance (line ends, gaps) are
    ignored. inf when no reading has a neighbour on another line.
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    idx = np.nonzero(ok)[0]
    pending = probe[ok[probe]]
    if len(idx) < 2 or len(pending) == 0:
        return float("inf")
    tree = cKDTree(np.column_stack([x[idx], y[idx]]))
    dists, diffs = [], []
    k = LAG_NEIGHBOURS
    while len(pending):
        k_eff = min(k, len(idx))
        d, j = tree.query(np.column_stack([x[pending], y[pending]]), k=k_eff)
        d, j = d.reshape(len(pending), k_eff), j.reshape(len(pending), k_eff)
        other = lines[idx[j]] != lines[pending][:, None]
        found = other.any(axis=1)
        first = np.argmax(other, axis=1)
        rows = np.nonzero(found)[0]
        dists.append(d[rows, first[rows]])
        diffs.append(np.abs(v[idx[j[rows, first[rows]]]] - v[pending[rows]]))
        pending = pending[~found]
        if k_eff == len(idx):
            break
        k *= 4
    d, diff = np.concatenate(dists), np.concatenate(diffs)
    if d.size == 0:
        return float("inf")
    return float(np.mean(diff[d <= 2.0 * np.median(d)]))


def estimate_lag(
    df: pd.DataFrame, column: str, lags: np.ndarray = LAG_SEARCH, seed: int = 0
) -> tuple[float, np.ndarray]:
    """
    GPS time lag that minimises the mean absolute difference between each
    reading and its nearest reading on another line (González Jiménez et
    al., 2022). Works best when neighbouring lines were walked in opposite
    directions. Returns (best lag, mean difference for every candidate).
    Raises ValueError when no reading has a neighbour on another line.
    """
    t = _require_time(df, "Lag estimation")
    v = _numeric(df, column)
    lines = df["Line"].astype(str).to_numpy()
    probe = np.arange(len(df))
    if len(probe) > LAG_MAX_POINTS:
        probe = np.sort(np.random.default_rng(seed).choice(probe, LAG_MAX_POINTS, replace=False))
    cost = []
    for lag in lags:
        x, y, _ = xy_metres(shift_positions(df, t, float(lag)))
        cost.append(_cross_line_difference(x, y, v, lines, probe))
    cost = np.asarray(cost, dtype=float)
    if not np.isfinite(cost).any():
        raise ValueError("No reading has a neighbour on another line, so the lag cannot be estimated.")
    return float(lags[int(np.argmin(cost))]), cost


def headings(df: pd.DataFrame) -> np.ndarray:
    """
    Walking direction of each line, from its first to its last reading, in
    degrees clockwise from north (or from the +Y axis), given to all of its
    readings. A per-line direction is not upset by GPS jitter or by readings
    logged while standing still; a line walked out and back has no single
    direction, so give each direction its own Line.
    """
    x, y, _ = xy_metres(df)
    out = np.full(len(df), np.nan)
    for idx in _by_line(df):
        ok = idx[np.isfinite(x[idx]) & np.isfinite(y[idx])]
        if len(ok) < 2:
            continue
        dx, dy = x[ok[-1]] - x[ok[0]], y[ok[-1]] - y[ok[0]]
        if dx == 0 and dy == 0:
            continue
        out[idx] = math.degrees(math.atan2(dx, dy)) % 360.0
    return out


def heading_mask(df: pd.DataFrame, center: float, tolerance: float) -> np.ndarray:
    """True for readings whose heading is within `tolerance` degrees of `center`."""
    h = headings(df)
    diff = np.abs((h - center + 180.0) % 360.0 - 180.0)
    return np.isfinite(h) & (diff <= tolerance)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _per_line(df: pd.DataFrame, col: str, func) -> tuple[np.ndarray, int]:
    v = _numeric(df, col)
    total = 0
    for idx in _by_line(df):
        v[idx], n = func(v[idx])
        total += n
    return v, total


def apply_corrections(df: pd.DataFrame, s: CorrectionSettings) -> tuple[pd.DataFrame, list[str]]:
    """
    Runs the enabled corrections in this order: despike, clip, sensor height,
    temperature, drift, I/Q offsets, background offset, reference
    calibration, EC at 25 °C, PCA, smoothing, GPS lag, heading filter.
    Returns (table, messages).
    Raises ValueError when an enabled step cannot run (missing columns).
    """
    out = df.copy()
    msgs: list[str] = []
    cols = channel_columns(out)

    if s.despike:
        total = 0
        for col in cols:
            out[col], n = _per_line(
                out, col, lambda v: despike_series(v, s.despike_window, s.despike_threshold)
            )
            total += n
        msgs.append(f"Despiking blanked {total} reading(s) across {len(cols)} channel(s).")

    if s.clip_percentiles:
        lo, hi = s.clip_percentiles
        total = 0
        for col in cols:
            out[col], n = clip_to_percentiles(_numeric(out, col), lo, hi)
            total += n
        msgs.append(f"Clipping to the {lo:g}–{hi:g} percentile range blanked {total} value(s).")

    if s.altitude_column:
        if s.altitude_column not in out.columns:
            raise ValueError(f"Altitude column '{s.altitude_column}' not found.")
        h = _numeric(out, s.altitude_column)
        ref = s.altitude_reference if s.altitude_reference is not None else float(np.nanmedian(h))
        models = []
        for col in cols:
            out[col], model = correct_height(_numeric(out, col), h, ref)
            models.append(model)
        msgs.append(
            f"Sensor-height correction to {ref:.3g} m from '{s.altitude_column}' "
            f"({models.count('exponential')} exponential, {models.count('linear')} linear fits)."
        )

    if s.temperature_column:
        out, m = correct_temperature(out, s.temperature_column, s.drift_lines)
        msgs += m

    if s.drift_lines:
        out, m = correct_drift(out, s.drift_lines, s.drift_model)
        msgs += m

    if s.iq_offsets:
        table = pd.read_csv(io.BytesIO(s.iq_offsets))
        if not {"column", "offset"}.issubset(table.columns):
            raise ValueError("The offsets file needs 'column' and 'offset' columns.")
        applied = []
        for col, off in zip(table["column"].astype(str), pd.to_numeric(table["offset"], errors="coerce")):
            if col in out.columns and np.isfinite(off):
                out[col] = _numeric(out, col) - off
                applied.append(f"{col}: {-off:+.4g}")
        msgs.append("Subtracted calibration offsets — " + ("; ".join(applied) or "no matching columns"))

    if s.ec_background is not None:
        shifts = []
        for col in channel_columns(out, ("EC",)):
            v = _numeric(out, col)
            shift = s.ec_background - float(np.nanmedian(v))
            out[col] = v + shift
            shifts.append(f"{col}: {shift:+.4g}")
        msgs.append("Shifted EC channels to the known background — " + "; ".join(shifts))

    if s.reference:
        ref = pd.read_csv(io.BytesIO(s.reference))
        fits = fit_reference_calibration(out, ref, s.reference_radius, s.reference_method)
        if not fits:
            msgs.append("Reference calibration: no channel had 3 or more matched, non-constant points; nothing applied.")
        for f in fits:
            out[f["column"]] = f["gain"] * _numeric(out, f["column"]) + f["offset"]
            msgs.append(
                f"Calibrated {f['column']}: gain {f['gain']:.4g}, offset {f['offset']:+.4g} "
                f"(R² {f['r2']:.3f}, {f['n']} reference points, {s.reference_method})."
            )

    if s.soil_temperature is not None:
        for col in channel_columns(out, ("EC",)):
            out[col] = ec_at_25(_numeric(out, col), s.soil_temperature)
        msgs.append(f"EC converted to 25 °C from a soil temperature of {s.soil_temperature:g} °C.")

    if s.pca_components > 0:
        for mode in gem_io.FREQUENCY_MODES:
            mcols = channel_columns(out, (mode,))
            if len(mcols) <= s.pca_components:
                if mcols:
                    msgs.append(
                        f"PCA noise reduction on {mode} skipped: {len(mcols)} channel(s) for "
                        f"{s.pca_components} component(s)."
                    )
                continue
            values, kept = pca_denoise(out[mcols].apply(pd.to_numeric, errors="coerce").to_numpy(),
                                       s.pca_components)
            out[mcols] = values
            msgs.append(
                f"PCA noise reduction on {mode}: kept {s.pca_components} of {len(mcols)} "
                f"components ({100 * kept:.1f} % of the variance)."
            )

    if s.smooth_window > 1:
        for col in cols:
            out[col], _ = _per_line(out, col, lambda v: (running_mean(v, s.smooth_window), 0))
        msgs.append(f"Running mean over {s.smooth_window} readings.")

    if s.lag_seconds:
        out = shift_positions(out, _require_time(out, "GPS lag correction"), s.lag_seconds)
        msgs.append(f"Positions shifted for a GPS lag of {s.lag_seconds:+.2f} s.")

    if s.bearing_center is not None:
        keep = heading_mask(out, s.bearing_center, s.bearing_tolerance)
        msgs.append(
            f"Heading filter kept {int(keep.sum())} of {len(out)} reading(s) "
            f"({s.bearing_center:g}° ± {s.bearing_tolerance:g}°)."
        )
        out = out.loc[keep].reset_index(drop=True)

    return out, msgs
