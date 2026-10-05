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
COLLINEAR_RATIO = 0.05          # minor/major principal spread below this = elongated
ACROSS_TRACK_TOL_M = 1.0        # line positions closer than this are one track
MIN_TRACKS = 3                  # elongated data needs this many tracks to be an area
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


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def level_lines(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Zero-order levelling: shift each line so its median equals the global median."""
    values = np.asarray(values, dtype=float)
    global_median = float(np.nanmedian(values))
    line_medians = (
        pd.Series(values).groupby(np.asarray(lines)).transform("median").to_numpy()
    )
    return values + (global_median - line_medians)


def median_nn_spacing(x: np.ndarray, y: np.ndarray) -> float:
    """Median distance from each point to its nearest distinct neighbour."""
    pts = np.unique(np.column_stack([x, y]), axis=0)
    if len(pts) < 2:
        raise ContouringError("Not enough distinct points to grid.")
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


def round_sig(value: float, sig: int = 2) -> float:
    if value <= 0 or not np.isfinite(value):
        raise ContouringError("Could not determine a positive cell size.")
    return float(f"{value:.{sig}g}")


def auto_cell_size(x: np.ndarray, y: np.ndarray) -> float:
    """
    sqrt(bounding-box area / n points), 2 significant figures.

    Gives roughly one grid node per raw reading — for line surveys, the
    geometric mean of along-line and across-line spacing.
    """
    area = float(np.ptp(x) * np.ptp(y))
    return round_sig(math.sqrt(area / len(x)))


def count_tracks(x: np.ndarray, y: np.ndarray, lines: np.ndarray) -> int:
    """Distinct across-track line positions; repeat passes of one transect count once."""
    pts = np.column_stack([x, y]) - [np.mean(x), np.mean(y)]
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    across = pts @ vecs[:, 0]  # coordinate along the minor principal axis
    medians = np.sort(pd.Series(across).groupby(np.asarray(lines)).median().to_numpy())
    return int(1 + np.sum(np.diff(medians) > ACROSS_TRACK_TOL_M))


def check_geometry(x: np.ndarray, y: np.ndarray, lines: np.ndarray | None = None) -> None:
    """
    Raise ContouringError if there are too few points or the points form a
    single transect: strongly elongated (minor/major spread < COLLINEAR_RATIO)
    and, when line labels are given, fewer than MIN_TRACKS distinct
    across-track line positions (repeat passes of one transect are not an area).
    """
    if len(x) < MIN_POINTS:
        raise ContouringError("Not enough points to grid.")
    eig = np.linalg.eigvalsh(np.cov(np.vstack([x, y])))  # ascending
    elongated = eig[1] <= 0 or math.sqrt(max(eig[0], 0.0) / eig[1]) < COLLINEAR_RATIO
    if elongated and (lines is None or count_tracks(x, y, lines) < MIN_TRACKS):
        raise ContouringError(
            "Coordinates look like a single transect — use the pseudo-section."
        )


@dataclass(frozen=True)
class GridSpec:
    """Regular grid; nodes at cell centres x0 + (i + 0.5) * cell."""
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    @property
    def ys(self) -> np.ndarray:
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    @property
    def n_nodes(self) -> int:
        return self.nx * self.ny


def make_grid(x: np.ndarray, y: np.ndarray, cell: float) -> GridSpec:
    """Grid aligned to multiples of *cell* covering the points' bounding box."""
    x0 = math.floor(np.min(x) / cell) * cell
    y0 = math.floor(np.min(y) / cell) * cell
    nx = int(math.floor((np.max(x) - x0) / cell)) + 1
    ny = int(math.floor((np.max(y) - y0) / cell)) + 1
    spec = GridSpec(x0=x0, y0=y0, cell=cell, nx=nx, ny=ny)
    if spec.n_nodes > MAX_GRID_NODES:
        raise GridTooLargeError(
            f"Grid would have {spec.n_nodes:,} nodes (limit {MAX_GRID_NODES:,}) — "
            "increase the cell size."
        )
    return spec


def block_median(
    x: np.ndarray, y: np.ndarray, v: np.ndarray, spec: GridSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per occupied cell: median x, median y, median value."""
    ix = np.floor((np.asarray(x) - spec.x0) / spec.cell).astype(np.int64)
    iy = np.floor((np.asarray(y) - spec.y0) / spec.cell).astype(np.int64)
    df = pd.DataFrame({"ix": ix, "iy": iy, "x": x, "y": y, "v": v})
    agg = df.groupby(["iy", "ix"], sort=True)[["x", "y", "v"]].median()
    return agg["x"].to_numpy(), agg["y"].to_numpy(), agg["v"].to_numpy()


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariogramFit:
    model: str
    psill: float
    range: float
    nugget: float
    lags: tuple[float, ...]
    gamma: tuple[float, ...]

    @property
    def parameters(self) -> list[float]:
        """PyKrige parameter order for spherical / exponential / gaussian."""
        return [self.psill, self.range, self.nugget]


def fit_variogram(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, model: str = "spherical", seed: int = 0
) -> VariogramFit:
    """
    Empirical semivariogram up to half the maximum pair distance, fitted by
    pair-count-weighted least squares (Oliver & Webster, 2014).

    PyKrige's built-in fit uses lags over the full distance range, where
    long-range pairs dominate and smooth fields collapse to pure nugget.
    """
    from pykrige import variogram_models as vm

    funcs = {
        "spherical": vm.spherical_variogram_model,
        "exponential": vm.exponential_variogram_model,
        "gaussian": vm.gaussian_variogram_model,
    }
    if model not in funcs:
        raise ValueError(f"Unknown variogram model: {model!r}")

    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    if len(pv) < 2:
        raise KrigingError("Not enough points to fit a variogram.")
    if len(pv) > VARIOGRAM_MAX_POINTS:
        idx = np.random.default_rng(seed).choice(len(pv), VARIOGRAM_MAX_POINTS, replace=False)
        px, py, pv = px[idx], py[idx], pv[idx]

    d = pdist(np.column_stack([px, py]))
    g = 0.5 * pdist(pv[:, None], "sqeuclidean")
    max_lag = 0.5 * float(d.max())
    keep = (d > 0) & (d <= max_lag)
    d, g = d[keep], g[keep]
    edges = np.linspace(0.0, max_lag, VARIOGRAM_BINS + 1)
    which = np.clip(np.digitize(d, edges) - 1, 0, VARIOGRAM_BINS - 1)
    counts = np.bincount(which, minlength=VARIOGRAM_BINS)
    occupied = counts > 0
    lags = np.bincount(which, weights=d, minlength=VARIOGRAM_BINS)[occupied] / counts[occupied]
    gamma = np.bincount(which, weights=g, minlength=VARIOGRAM_BINS)[occupied] / counts[occupied]
    counts = counts[occupied]
    if len(lags) < 3:
        raise KrigingError(
            "Too few distance classes to fit a variogram. Try the thin-plate spline method."
        )

    def f(h, psill, rng_, nugget):
        return funcs[model]([psill, rng_, nugget], h)

    p0 = [max(float(gamma.max() - gamma.min()), 1e-12), max_lag / 2, max(float(gamma.min()), 0.0)]
    try:
        popt, _ = curve_fit(
            f, lags, gamma, p0=p0, sigma=1.0 / np.sqrt(counts),
            bounds=([0.0, 1e-6 * max_lag, 0.0], [np.inf, 4.0 * max_lag, np.inf]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        raise KrigingError(
            f"Variogram fit failed ({exc}). Try the thin-plate spline method."
        ) from exc
    return VariogramFit(
        model=model, psill=float(popt[0]), range=float(popt[1]), nugget=float(popt[2]),
        lags=tuple(lags.tolist()), gamma=tuple(gamma.tolist()),
    )


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def predict(
    px: np.ndarray,
    py: np.ndarray,
    pv: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    method: str,
    smoothing: float = 0.0,
    variogram: VariogramFit | None = None,
    variogram_model: str = "spherical",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Interpolate scattered (px, py, pv) at query points (qx, qy).

    Returns (estimate, variance); variance is None except for kriging. For
    kriging, *variogram* is fitted from the data when not supplied.
    Coordinates are centred on the data centroid for numerical conditioning.
    """
    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    cx, cy = float(np.mean(px)), float(np.mean(py))
    p = np.column_stack([px - cx, py - cy])
    q = np.column_stack([np.ravel(qx) - cx, np.ravel(qy) - cy])

    if method == "spline":
        neighbours = LOCAL_NEIGHBOURS if len(pv) > GLOBAL_MAX_POINTS else None
        rbf = RBFInterpolator(
            p, pv, kernel="thin_plate_spline", smoothing=smoothing, neighbors=neighbours
        )
        return rbf(q), None

    if method == "linear":
        return griddata(p, pv, q, method="linear"), None

    if method == "kriging":
        if variogram is None:
            variogram = fit_variogram(p[:, 0], p[:, 1], pv, variogram_model)
        return _krige(p, pv, q, variogram)

    raise ValueError(f"Unknown gridding method: {method!r}")


def _krige(
    p: np.ndarray, pv: np.ndarray, q: np.ndarray, variogram: VariogramFit
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with a local neighbourhood of LOCAL_NEIGHBOURS points."""
    from pykrige.ok import OrdinaryKriging

    n = len(pv)
    if n > KRIGE_MAX_POINTS:
        raise KrigingError(
            f"Kriging is limited to {KRIGE_MAX_POINTS:,} block-reduced points "
            f"(this map has {n:,}) — increase the cell size or use the thin-plate spline."
        )
    try:
        ok = OrdinaryKriging(
            p[:, 0], p[:, 1], pv,
            variogram_model=variogram.model,
            variogram_parameters=variogram.parameters,
            pseudo_inv=True,  # stable for smooth (e.g. gaussian, zero-nugget) models
            enable_plotting=False,
            verbose=False,
        )
        z, ss = ok.execute(
            "points", q[:, 0], q[:, 1],
            backend="C", n_closest_points=min(LOCAL_NEIGHBOURS, n),
        )
    except Exception as exc:  # pykrige raises a variety of types
        raise KrigingError(
            f"Kriging failed ({exc}). Try the thin-plate spline method."
        ) from exc
    z = np.asarray(np.ma.filled(z, np.nan), dtype=float)
    ss = np.asarray(np.ma.filled(ss, np.nan), dtype=float)
    return z, np.clip(ss, 0.0, None)


def blank_far(
    grid: np.ndarray, spec: GridSpec, px: np.ndarray, py: np.ndarray, max_distance: float
) -> np.ndarray:
    """Set grid nodes farther than *max_distance* from any data point to NaN."""
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    d, _ = cKDTree(np.column_stack([px, py])).query(
        np.column_stack([xx.ravel(), yy.ravel()])
    )
    out = np.array(grid, dtype=float, copy=True)
    out[(d > max_distance).reshape(out.shape)] = np.nan
    return out


def contour_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    """n_levels + 1 evenly spaced boundaries spanning the 2nd-98th percentile."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ContouringError("Nothing to contour (all values blank).")
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        pad = max(abs(lo) * 1e-6, 1e-12)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_levels + 1)
