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
LAG_NEIGHBOURS = 64                                      # candidates searched for another line
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
        return s.to_numpy(), 0
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

