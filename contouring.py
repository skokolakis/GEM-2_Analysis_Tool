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
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay, QhullError, cKDTree
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
VARIOGRAM_BINS = 15             # log-spaced lag classes up to half the max distance
VARIOGRAM_MIN_PAIRS = 30        # lag classes with fewer pairs are ignored
VARIOGRAM_ITERATIONS = 3        # Cressie weights are re-evaluated on the fitted model
GAUSSIAN_NUGGET_FLOOR = 1e-3    # x total sill; keeps gaussian kriging well conditioned
BLANK_NN_FACTOR = 2.0           # default blanking >= 2 x median point spacing
BLANK_COVERAGE_FACTOR = 1.5     # ... and >= 1.5 x P90 node-to-data distance inside the hull
KRIGE_CELL_GROWTH = 1.25        # auto cell growth step to respect KRIGE_MAX_POINTS
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

UNIT_LABELS = {
    "EC": "EC (mS/m)",
    "MS": "MS (10⁻³ SI)",
    "I": "In-phase (ppm)",
    "Q": "Quadrature (ppm)",
}
AUX_LABELS = {
    "PowerLn": "Power-line noise (mG)",
    "QSum": "Quadrature sum (ppm)",
    "TotalEC[mS/m]": "Total EC (mS/m)",
}


def value_label(mode: str, is_gem: bool, channel: str | None = None) -> str:
    """Axis / colour-bar label with correct GEM units; AUX channels carry their own."""
    if not is_gem:
        return f"{mode} (input units)"
    if mode == "AUX":
        return AUX_LABELS.get(channel, str(channel) if channel else "Auxiliary channel")
    return UNIT_LABELS.get(mode, mode)


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
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        usable = lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))  # (0, 0) = no fix
        if usable.sum() >= MIN_POINTS:
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
        pd.Series(values)
        .groupby(np.asarray(lines), dropna=False)
        .transform("median")
        .to_numpy()
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
    def pykrige_parameters(self) -> dict[str, float]:
        """
        Explicit keys for PyKrige. A PyKrige parameter *list* is read as
        [full sill, range, nugget] (the nugget is subtracted), so a list of
        [psill, range, nugget] would silently lower the partial sill.
        """
        return {"psill": self.psill, "range": self.range, "nugget": self.nugget}


def fit_variogram(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, model: str = "spherical", seed: int = 0
) -> VariogramFit:
    """
    Empirical semivariogram in log-spaced lag classes up to half the maximum
    pair distance, fitted by weighted least squares with Cressie (1985)
    weights N(h) / gamma(h)^2, re-evaluated on the fitted model.

    Log-spaced classes give the short lags their own classes, and Cressie
    weights stop the far more numerous long-lag pairs from dominating the fit.
    With equal-width classes and pair-count weights the nugget went to zero
    on noisy data and gaussian kriging became unstable; a small nugget floor
    keeps the gaussian model well conditioned.
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
    if d.size == 0:
        raise KrigingError(
            "Too few distance classes to fit a variogram. Try the thin-plate spline method."
        )
    edges = np.geomspace(max(float(np.percentile(d, 0.5)), 1e-9 * max_lag), max_lag,
                         VARIOGRAM_BINS + 1)
    edges[0] = 0.0
    which = np.clip(np.digitize(d, edges) - 1, 0, VARIOGRAM_BINS - 1)
    counts = np.bincount(which, minlength=VARIOGRAM_BINS)
    occupied = counts >= VARIOGRAM_MIN_PAIRS
    lags = np.bincount(which, weights=d, minlength=VARIOGRAM_BINS)[occupied] / counts[occupied]
    gamma = np.bincount(which, weights=g, minlength=VARIOGRAM_BINS)[occupied] / counts[occupied]
    counts = counts[occupied]
    if len(lags) < 3:
        raise KrigingError(
            "Too few distance classes to fit a variogram. Try the thin-plate spline method."
        )

    def f(h, psill, rng_, nugget):
        return funcs[model]([psill, rng_, nugget], h)

    popt = [max(float(gamma.max() - gamma.min()), 1e-12), max_lag / 2, max(float(gamma.min()), 0.0)]
    gamma_floor = 1e-6 * max(float(gamma.max()), 1e-12)
    try:
        for _ in range(VARIOGRAM_ITERATIONS):
            model_gamma = np.maximum(f(lags, *popt), gamma_floor)
            popt, _ = curve_fit(
                f, lags, gamma, p0=popt, sigma=model_gamma / np.sqrt(counts),
                bounds=([0.0, 1e-6 * max_lag, 0.0], [np.inf, 4.0 * max_lag, np.inf]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError) as exc:
        raise KrigingError(
            f"Variogram fit failed ({exc}). Try the thin-plate spline method."
        ) from exc
    psill, rng_, nugget = (float(v) for v in popt)
    if model == "gaussian":
        nugget = max(nugget, GAUSSIAN_NUGGET_FLOOR * (psill + nugget))
    return VariogramFit(
        model=model, psill=psill, range=rng_, nugget=nugget,
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

    if method in ("spline", "linear"):
        try:
            if method == "spline":
                neighbours = LOCAL_NEIGHBOURS if len(pv) > GLOBAL_MAX_POINTS else None
                rbf = RBFInterpolator(
                    p, pv, kernel="thin_plate_spline", smoothing=smoothing,
                    neighbors=neighbours,
                )
                return rbf(q), None
            return griddata(p, pv, q, method="linear"), None
        except (np.linalg.LinAlgError, QhullError, ValueError) as exc:
            raise ContouringError(
                f"Gridding failed ({exc}). Try a larger cell size or another method."
            ) from exc

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
            variogram_parameters=variogram.pykrige_parameters,
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


def default_blank_distance(spec: GridSpec, px: np.ndarray, py: np.ndarray) -> float:
    """
    Blanking distance that keeps the gaps between survey lines filled:
    max(BLANK_NN_FACTOR x median point spacing,
        BLANK_COVERAGE_FACTOR x 90th percentile of node-to-nearest-point
        distance over nodes inside the data's convex hull).
    """
    pts = np.column_stack([px, py])
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    try:
        inside = Delaunay(pts).find_simplex(nodes) >= 0
    except QhullError as exc:
        raise ContouringError(f"Could not triangulate the data ({exc}).") from exc
    nn = BLANK_NN_FACTOR * median_nn_spacing(px, py)
    if not inside.any():
        return nn
    d, _ = cKDTree(pts).query(nodes[inside])
    return max(nn, BLANK_COVERAGE_FACTOR * float(np.percentile(d, 90)))


def contour_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    """n_levels + 1 evenly spaced boundaries spanning the 2nd-98th percentile."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ContouringError("Nothing to contour (all values blank).")
    lo, hi = np.percentile(finite, [2, 98])
    if hi - lo <= 1e-9 * max(abs(lo), abs(hi), 1e-12):  # flat up to round-off
        pad = max(abs(lo) * 1e-6, 1e-12)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_levels + 1)


# ---------------------------------------------------------------------------
# Area-map pipeline
# ---------------------------------------------------------------------------

@dataclass
class AreaMapResult:
    spec: GridSpec
    z: np.ndarray                       # (ny, nx), NaN where blanked
    variance: np.ndarray | None         # kriging only
    bx: np.ndarray                      # block-reduced points used for gridding
    by: np.ndarray
    bv: np.ndarray
    method: str
    variogram: VariogramFit | None      # kriging only
    blank_distance: float
    origin: tuple[float, float] | None  # (lon0, lat0) if input was degrees
    levelled: bool
    n_raw: int


def compute_area_map(
    df: pd.DataFrame,
    value_col: str,
    coord_mode: str = "auto",
    method: str = "spline",
    cell_size: float | None = None,
    blank_distance: float | None = None,
    level: bool = False,
    smoothing: float = 0.0,
    variogram_model: str = "spherical",
    line_col: str = "Line",
) -> AreaMapResult:
    """Full area-map pipeline for one value column of a raw GEM table."""
    x_col, y_col, is_deg = find_coordinate_columns(df, coord_mode)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[keep], y[keep], v[keep]

    origin = None
    if is_deg:
        fix = ~((x == 0) & (y == 0))  # GPS no-fix rows are logged as (0, 0)
        keep[keep] = fix
        x, y, v = x[fix], y[fix], v[fix]
        x, y, origin = project_to_local_metres(x, y)

    levelled = False
    if level and line_col in df.columns:
        v = level_lines(v, df[line_col].to_numpy()[keep])
        levelled = True

    lines = df[line_col].to_numpy()[keep] if line_col in df.columns else None
    check_geometry(x, y, lines)
    cell = cell_size if cell_size else auto_cell_size(x, y)
    spec = make_grid(x, y, cell)
    bx, by, bv = block_median(x, y, v, spec)
    while method == "kriging" and not cell_size and len(bv) > KRIGE_MAX_POINTS:
        cell = round_sig(cell * KRIGE_CELL_GROWTH)
        spec = make_grid(x, y, cell)
        bx, by, bv = block_median(x, y, v, spec)
    if len(bv) < MIN_POINTS:
        raise ContouringError("Not enough points to grid.")

    variogram = fit_variogram(bx, by, bv, variogram_model) if method == "kriging" else None
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    z, var = predict(bx, by, bv, xx, yy, method, smoothing, variogram)
    z = z.reshape(spec.ny, spec.nx)
    if var is not None:
        var = var.reshape(spec.ny, spec.nx)

    dist = blank_distance if blank_distance else default_blank_distance(spec, bx, by)
    z = blank_far(z, spec, bx, by, dist)
    if var is not None:
        var = blank_far(var, spec, bx, by, dist)

    return AreaMapResult(
        spec=spec, z=z, variance=var, bx=bx, by=by, bv=bv, method=method,
        variogram=variogram, blank_distance=dist, origin=origin,
        levelled=levelled, n_raw=int(keep.sum()),
    )


def cross_validate(
    px: np.ndarray,
    py: np.ndarray,
    pv: np.ndarray,
    method: str,
    smoothing: float = 0.0,
    variogram: VariogramFit | None = None,
    k: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """
    k-fold CV on the block-reduced points. Returns rmse, mae, n (predicted points).

    For kriging the variogram is held fixed across folds (pass the fit used
    for the map).
    """
    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    folds = np.random.default_rng(seed).permutation(len(pv)) % k
    errors = []
    for f in range(k):
        test = folds == f
        train = ~test
        est, _ = predict(
            px[train], py[train], pv[train], px[test], py[test],
            method, smoothing, variogram,
        )
        errors.append(est - pv[test])
    err = np.concatenate(errors)
    err = err[np.isfinite(err)]
    if err.size == 0:
        raise ContouringError("Cross-validation produced no predictions.")
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "n": int(err.size),
    }


# ---------------------------------------------------------------------------
# Pseudo-section
# ---------------------------------------------------------------------------

def parse_frequency(label: str) -> float | None:
    """
    Frequency in Hz from a label: a bare number ('9000') or a number followed
    by Hz or kHz ('4525Hz' -> 4525.0, 'EC 93.5 kHz' -> 93500.0). None
    otherwise, e.g. 'Sheet1', so such labels get a categorical axis.
    """
    text = str(label).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k?)Hz", text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)) * (1000.0 if m.group(2) else 1.0)


@dataclass
class PseudoSection:
    distance: np.ndarray             # (n_dist,)
    labels: list[str]                # row labels, in plotting order
    frequencies: np.ndarray | None   # (n_freq,) if every label parsed, else None
    values: np.ndarray               # (n_freq, n_dist)


def build_pseudosection(profiles: dict[str, pd.Series]) -> PseudoSection:
    """
    Stack mean profiles (index = distance) onto their common overlap.

    Each profile is linearly interpolated onto a shared grid spanning the
    overlap of all ranges, at the smallest step among them; nothing is
    extrapolated.
    """
    cleaned: dict[str, pd.Series] = {}
    for label, s in profiles.items():
        s = s.dropna().sort_index()
        s = s[~s.index.duplicated()]
        if len(s) >= 2:
            cleaned[str(label)] = s
    if len(cleaned) < 2:
        raise ContouringError("A pseudo-section needs at least 2 frequencies.")

    lo = max(float(s.index.min()) for s in cleaned.values())
    hi = min(float(s.index.max()) for s in cleaned.values())
    if hi <= lo:
        raise ContouringError("Profiles do not overlap in distance.")
    step = min(
        float(np.median(np.diff(s.index.to_numpy(dtype=float)))) for s in cleaned.values()
    )
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    if n < 2:
        raise ContouringError("Profiles overlap by less than one distance step.")
    distance = np.clip(lo + step * np.arange(n), lo, hi)  # no float overshoot past hi

    labels = list(cleaned.keys())
    freqs = [parse_frequency(lb) for lb in labels]
    if all(f is not None and f > 0 for f in freqs) and len(set(freqs)) == len(freqs):
        order = np.argsort(freqs, kind="stable")
        labels = [labels[i] for i in order]
        frequencies = np.array([freqs[i] for i in order], dtype=float)
    else:
        frequencies = None

    values = np.vstack([
        np.interp(
            distance,
            cleaned[lb].index.to_numpy(dtype=float),
            cleaned[lb].to_numpy(dtype=float),
            left=np.nan, right=np.nan,
        )
        for lb in labels
    ])
    return PseudoSection(distance=distance, labels=labels, frequencies=frequencies, values=values)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_area_map_figure(
    result: AreaMapResult,
    label: str,
    title: str,
    n_levels: int = 20,
    show_points: bool = True,
) -> plt.Figure:
    """Contour map; for kriging, adds a kriging standard-deviation panel."""
    spec = result.spec
    # Compute levels first: they may raise, and no figure should be left open.
    levels = contour_levels(result.z, n_levels)
    std = np.sqrt(result.variance) if result.variance is not None else None
    std_levels = contour_levels(std, n_levels) if std is not None else None
    panels = 2 if std is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 6), squeeze=False)
    xs, ys = spec.xs, spec.ys

    ax = axes[0, 0]
    cs = ax.contourf(
        xs, ys, np.ma.masked_invalid(result.z),
        levels=levels, cmap="viridis", extend="both",
    )
    fig.colorbar(cs, ax=ax, label=label)
    if show_points:
        ax.plot(result.bx, result.by, ",", color="k", alpha=0.4)
    ax.set_title(title + (" — line-levelled (per-line median)" if result.levelled else ""))

    if std is not None:
        ax2 = axes[0, 1]
        cs2 = ax2.contourf(
            xs, ys, np.ma.masked_invalid(std),
            levels=std_levels, cmap="magma", extend="both",
        )
        fig.colorbar(cs2, ax=ax2, label=f"Kriging std. dev. — {label}")
        ax2.set_title("Kriging standard deviation")

    xlab = "Easting (local m)" if result.origin is not None else "X (m)"
    ylab = "Northing (local m)" if result.origin is not None else "Y (m)"
    for a in axes[0]:
        a.ticklabel_format(useOffset=False, style="plain")  # full map coordinates
        a.set_aspect("equal")
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
    fig.tight_layout()
    return fig


def make_pseudosection_figure(
    ps: PseudoSection, label: str, title: str, n_levels: int = 20
) -> plt.Figure:
    """Distance x frequency contour with a white line at each measured frequency."""
    levels = contour_levels(ps.values, n_levels)  # may raise; before any figure exists
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if ps.frequencies is not None:
        # Contour against log10(f) so values between rows are interpolated on the log axis.
        rows = np.log10(ps.frequencies)
        ax.set_ylabel("Frequency (Hz, log scale)")
    else:
        rows = np.arange(len(ps.labels), dtype=float)
        ax.set_ylabel("Frequency / sheet")
    ax.set_yticks(rows, labels=ps.labels)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    cs = ax.contourf(
        ps.distance, rows, np.ma.masked_invalid(ps.values),
        levels=levels, cmap="viridis", extend="both",
    )
    for r in rows:
        ax.axhline(r, color="white", linewidth=0.6, alpha=0.8)
    fig.colorbar(cs, ax=ax, label=label)
    ax.set_xlabel("Distance (m)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def grid_to_csv(result: AreaMapResult) -> bytes:
    """x, y, value [, variance] [, lon, lat] for every non-blank node."""
    spec = result.spec
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    keep = np.isfinite(result.z)
    out = {"x": xx[keep], "y": yy[keep], "value": result.z[keep]}
    if result.variance is not None:
        out["variance"] = result.variance[keep]
    if result.origin is not None:
        out["lon"], out["lat"] = local_metres_to_lonlat(out["x"], out["y"], result.origin)
    return pd.DataFrame(out).to_csv(index=False).encode()


def grid_to_asc(result: AreaMapResult) -> bytes:
    """ESRI ASCII grid; first data row is the northernmost."""
    spec = result.spec
    z = np.where(np.isfinite(result.z), result.z, NODATA)
    buf = io.StringIO()
    buf.write(f"ncols {spec.nx}\n")
    buf.write(f"nrows {spec.ny}\n")
    buf.write(f"xllcorner {spec.x0:.6f}\n")
    buf.write(f"yllcorner {spec.y0:.6f}\n")
    buf.write(f"cellsize {spec.cell:.6f}\n")
    buf.write(f"NODATA_value {NODATA:g}\n")
    np.savetxt(buf, z[::-1], fmt="%.6g")
    return buf.getvalue().encode()
