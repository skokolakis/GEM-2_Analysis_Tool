"""
2D contouring for GEM EMI data: plan-view area maps and distance x frequency
pseudo-sections.

Pure functions only (no Streamlit). See
docs/superpowers/specs/2026-10-05-2d-contouring-design.md for the design and
literature basis.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_008.8
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "long", "longitude")
DEGREE_SPAN_MAX = 0.05          # auto-detect X/Y as degrees only below this span
MIN_POINTS = 10                 # minimum block-reduced points to grid
COLLINEAR_RATIO = 0.05          # minor/major principal spread below this = transect
MAX_GRID_NODES = 4_000_000
LOCAL_NEIGHBOURS = 64           # neighbourhood size for local spline / kriging
GLOBAL_MAX_POINTS = 2000        # spline: above this, use local neighbourhoods
KRIGE_MAX_POINTS = 4000         # PyKrige builds all pairwise distances at set-up
VARIOGRAM_MAX_POINTS = 2000     # subsample size for the empirical variogram
VARIOGRAM_BINS = 15
NODATA = -9999.0

METHODS = {
    "spline": "Thin-plate spline",
    "kriging": "Ordinary kriging",
    "linear": "Linear (Delaunay)",
}
VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian"]


class ContouringError(ValueError):
    """Expected, user-facing condition (shown with st.info)."""


class GridTooLargeError(ContouringError):
    """Requested grid exceeds MAX_GRID_NODES (shown with st.warning)."""


class KrigingError(ContouringError):
    """Variogram fit or kriging solve failed (shown with st.error)."""


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def value_label(mode: str, is_gem: bool) -> str:
    """Axis / colour-bar label with correct GEM units."""
    if not is_gem:
        return f"{mode} (input units)"
    return "EC (mS/m)" if mode == "EC" else "MS (10⁻³ SI)"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def looks_like_degrees(x: np.ndarray, y: np.ndarray) -> bool:
    """True if X/Y are plausibly lon/lat degrees covering a site-scale area."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return False
    in_range = np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 90)
    small_span = np.ptp(x) < DEGREE_SPAN_MAX and np.ptp(y) < DEGREE_SPAN_MAX
    return bool(in_range and small_span)


def find_coordinate_columns(df: pd.DataFrame, mode: str = "auto") -> tuple[str, str, bool]:
    """
    Return (x_column, y_column, is_degrees).

    Lat/Lon columns (case-insensitive) with at least 2 numeric values take precedence
    and are always degrees. Otherwise X/Y are used; *mode* is "auto", "metres" or "degrees".
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    lat_col = next((lower[n] for n in LAT_NAMES if n in lower), None)
    lon_col = next((lower[n] for n in LON_NAMES if n in lower), None)
    if lat_col is not None and lon_col is not None:
        # Check if both columns have at least 2 finite numeric values
        lat_valid = pd.to_numeric(df[lat_col], errors="coerce").notna().sum() >= 2
        lon_valid = pd.to_numeric(df[lon_col], errors="coerce").notna().sum() >= 2
        if lat_valid and lon_valid:
            return lon_col, lat_col, True

    if "x" in lower and "y" in lower:
        x_col, y_col = lower["x"], lower["y"]
        if mode == "degrees":
            return x_col, y_col, True
        if mode == "metres":
            return x_col, y_col, False
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        return x_col, y_col, looks_like_degrees(x, y)

    raise ContouringError("No coordinate columns (X/Y or Lat/Lon) found in this file.")


def project_to_local_metres(
    lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Equirectangular projection about the centroid. Returns (x, y, (lon0, lat0))."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))
    x = EARTH_RADIUS_M * math.cos(math.radians(lat0)) * np.radians(lon - lon0)
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y, (lon0, lat0)


def local_metres_to_lonlat(
    x: np.ndarray, y: np.ndarray, origin: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of project_to_local_metres."""
    lon0, lat0 = origin
    lat = lat0 + np.degrees(np.asarray(y, dtype=float) / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(
        np.asarray(x, dtype=float) / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    )
    return lon, lat
