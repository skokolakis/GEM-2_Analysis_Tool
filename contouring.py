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
import warnings
from dataclasses import dataclass, field, replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, FuncNorm, LinearSegmentedColormap, LogNorm, Normalize
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator, griddata
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

PROJECTIONS = {
    "local": "Local metres (about the survey centre)",
    "utm": "UTM (WGS 84)",
}

# The gridding methods of Surfer (Golden Software), in the order of its Grid Data dialog
# after the original three.
METHODS = {
    "spline": "Thin-plate spline",
    "kriging": "Ordinary kriging",
    "linear": "Linear (Delaunay)",
    "min_curvature": "Minimum curvature",
    "idw": "Inverse distance to a power",
    "rbf": "Radial basis function",
    "natural_neighbor": "Natural neighbour",
    "nearest": "Nearest neighbour",
    "shepard": "Modified Shepard's method",
    "local_polynomial": "Local polynomial",
    "polynomial": "Polynomial regression (trend surface)",
    "moving_average": "Moving average",
    "metrics": "Data metrics",
}
GRID_METHODS = ("min_curvature", "natural_neighbor")   # solved on the grid nodes themselves
METHOD_DEFAULTS = {                                      # Surfer's defaults where it has them
    "idw": {"power": 2.0, "delta": 0.0},
    "min_curvature": {"tension": 0.0},
    "rbf": {"kernel": "multiquadric"},
    "local_polynomial": {"order": 1, "power": 2.0},
    "polynomial": {"order": 1},
    "moving_average": {"radius": 0.0},
    "metrics": {"statistic": "count", "radius": 0.0},
}
RBF_KERNELS = {
    "multiquadric": "Multiquadric",
    "inverse_multiquadric": "Inverse multiquadric",
    "cubic": "Natural cubic spline (r³)",
    "thin_plate_spline": "Thin-plate spline",
}
METRICS = {
    "count": "Number of readings",
    "density": "Readings per m²",
    "median": "Median",
    "minimum": "Minimum",
    "maximum": "Maximum",
    "range": "Range",
    "std": "Standard deviation",
}
VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian"]
SEARCH_POINTS = 64              # readings used per estimate (Surfer's default maximum)
METRICS_MAX_POINTS = 256        # readings counted per node by the data metrics
MOVING_AVERAGE_POINTS = 16      # the automatic search radius holds about this many readings
MIN_CURVATURE_MAX_NODES = 300_000   # sparse direct solve; larger grids take too long
MIN_CURVATURE_DATA_WEIGHT = 1e3     # data misfit weight against the curvature energy
SHEPARD_QUADRATIC_POINTS = 13   # Franke & Nielson (1980) / Renka (1988) defaults, as in Surfer
SHEPARD_WEIGHT_POINTS = 19
NATURAL_NEIGHBOUR_MAX_WORK = 2e8    # node visits of the discrete Sibson scatter
NATURAL_NEIGHBOUR_SUPERSAMPLE = 4   # raster points per cell side for the stolen areas

COLOUR_MAPS = {
    "viridis": "Viridis (perceptually uniform)",
    "surfer_rainbow": "Rainbow (Surfer)",
    "turbo": "Turbo (smooth rainbow)",
    "jet": "Jet",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "magma": "Magma",
    "cividis": "Cividis (colour-blind safe)",
    "terrain": "Terrain",
    "Spectral_r": "Spectral",
    "RdYlBu_r": "Red–yellow–blue",
    "RdBu_r": "Red–blue (diverging)",
    "gray": "Greyscale",
}
SURFER_RAINBOW = ["#a87cf0", "#2020ff", "#00b4ff", "#00e000", "#ffff00", "#ff8000", "#ff0000"]
COLOUR_RANGES = {
    "percentile": "Percentiles",
    "minmax": "Full data range (min–max)",
    "fixed": "Fixed values",
}
COLOUR_SCALES = {
    "linear": "Linear",
    "log": "Logarithmic",
    "equalised": "Histogram-equalised",
}
MAP_DISPLAYS = {
    "filled": "Filled contours",
    "image": "Continuous image",
}


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
    "ECqdiff[mS/m]": "EC from quadrature difference (mS/m)",
    "MagViscosity[1/1000]": "Magnetic viscosity κ″ (10⁻³ SI)",
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
    lon: np.ndarray, lat: np.ndarray, origin: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Equirectangular projection about *origin* (lon0, lat0), by default the
    centroid. Returns (x, y, (lon0, lat0)).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if origin is None:
        lon0 = float(np.nanmean(lon))
        lat0 = float(np.nanmean(lat))
    else:
        lon0, lat0 = (float(v) for v in origin)
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

def method_options(method: str, options: dict | None = None) -> dict:
    """The settings of *method*: its METHOD_DEFAULTS updated by *options* (other keys ignored)."""
    out = dict(METHOD_DEFAULTS.get(method, {}))
    out.update({k: v for k, v in (options or {}).items() if k in out})
    return out


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
    options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Interpolate scattered (px, py, pv) at query points (qx, qy).

    Returns (estimate, variance); variance is None except for kriging. For
    kriging, *variogram* is fitted from the data when not supplied. *options*
    are the method's settings (see METHOD_DEFAULTS). GRID_METHODS estimate
    grid nodes only: use predict_grid for them.
    Coordinates are centred on the data centroid for numerical conditioning.
    """
    if method in GRID_METHODS:
        raise ValueError(f"{METHODS[method]} works on grid nodes: use predict_grid.")
    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    cx, cy = float(np.mean(px)), float(np.mean(py))
    p = np.column_stack([px - cx, py - cy])
    q = np.column_stack([np.ravel(qx) - cx, np.ravel(qy) - cy])
    opts = method_options(method, options)

    if method in ("spline", "rbf", "linear"):
        try:
            if method == "linear":
                return griddata(p, pv, q, method="linear"), None
            kernel = "thin_plate_spline" if method == "spline" else opts["kernel"]
            neighbours = LOCAL_NEIGHBOURS if len(pv) > GLOBAL_MAX_POINTS else None
            shape = {"epsilon": _rbf_epsilon(p)} if "multiquadric" in kernel else {}
            rbf = RBFInterpolator(
                p, pv, kernel=kernel, smoothing=smoothing, neighbors=neighbours, **shape,
            )
            return rbf(q), None
        except (np.linalg.LinAlgError, QhullError, ValueError) as exc:
            raise ContouringError(
                f"Gridding failed ({exc}). Try a larger cell size or another method."
            ) from exc

    if method == "kriging":
        if variogram is None:
            variogram = fit_variogram(p[:, 0], p[:, 1], pv, variogram_model)
        return _krige(p, pv, q, variogram)

    if method == "idw":
        return _idw(p, pv, q, float(opts["power"]), float(opts["delta"])), None
    if method == "nearest":
        return pv[cKDTree(p).query(q)[1]], None
    if method == "shepard":
        return _shepard(p, pv, q), None
    if method == "local_polynomial":
        return _local_polynomial(p, pv, q, int(opts["order"]), float(opts["power"])), None
    if method == "polynomial":
        return _polynomial(p, pv, q, int(opts["order"])), None
    if method == "moving_average":
        return _search_statistic(p, pv, q, float(opts["radius"]), "mean"), None
    if method == "metrics":
        return _search_statistic(p, pv, q, float(opts["radius"]), opts["statistic"]), None

    raise ValueError(f"Unknown gridding method: {method!r}")


def predict_grid(
    px: np.ndarray,
    py: np.ndarray,
    pv: np.ndarray,
    spec: GridSpec,
    method: str,
    smoothing: float = 0.0,
    variogram: VariogramFit | None = None,
    variogram_model: str = "spherical",
    options: dict | None = None,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Estimate (and kriging variance) at every node of *spec*, as (ny, nx)
    arrays. *max_distance* (the blanking distance) bounds the work of the
    natural-neighbour method; nodes farther from the data are blanked later.
    """
    opts = method_options(method, options)
    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    if method == "min_curvature":
        return _min_curvature(px, py, pv, spec, float(opts["tension"])), None
    if method == "natural_neighbor":
        return _natural_neighbour(px, py, pv, spec, max_distance), None
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    z, var = predict(px, py, pv, xx, yy, method, smoothing, variogram, variogram_model, options)
    shape = (spec.ny, spec.nx)
    return z.reshape(shape), (var.reshape(shape) if var is not None else None)


def _rbf_epsilon(p: np.ndarray) -> float:
    """
    Shape parameter of the multiquadric kernels: 1 / R with Surfer's default
    R² = (diagonal of the data extent)² / (25 n). The scale-free kernels
    (thin-plate spline, cubic) take none.
    """
    diagonal = float(np.hypot(*np.ptp(p, axis=0)))
    r = diagonal / (5.0 * math.sqrt(len(p)))
    return 1.0 / r if r > 0 else 1.0


def _neighbours(p: np.ndarray, q: np.ndarray, k: int, radius: float = np.inf):
    """(distance, index) of the k nearest points, always 2-D; inf / len(p) beyond *radius*."""
    k = min(k, len(p))
    d, i = cKDTree(p).query(q, k=k, distance_upper_bound=radius)
    return d.reshape(len(q), k), i.reshape(len(q), k)


def _idw(p: np.ndarray, pv: np.ndarray, q: np.ndarray, power: float, delta: float) -> np.ndarray:
    """
    Inverse distance to a power over the SEARCH_POINTS nearest readings:
    weights 1 / (d² + δ²)^(power / 2). With δ = 0 a node on a reading takes
    its value exactly; δ > 0 smooths (Surfer's smoothing parameter).
    """
    if power <= 0:
        raise ContouringError("The inverse-distance power must be above 0.")
    d, i = _neighbours(p, q, SEARCH_POINTS)
    d2 = d ** 2 + delta ** 2
    hit = d2[:, 0] == 0
    w = 1.0 / np.where(d2 == 0, 1.0, d2) ** (power / 2.0)
    z = np.sum(w * pv[i], axis=1) / np.sum(w, axis=1)
    z[hit] = pv[i[hit, 0]]
    return z


def _poly_design(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Columns x^a y^b for a + b <= order, constant first; shape x.shape + (n_terms,)."""
    if order not in (1, 2, 3):
        raise ContouringError("The polynomial order must be 1, 2 or 3.")
    terms = [x ** (t - b) * y ** b for t in range(order + 1) for b in range(t + 1)]
    return np.stack(terms, axis=-1)


def _polynomial(p: np.ndarray, pv: np.ndarray, q: np.ndarray, order: int) -> np.ndarray:
    """Least-squares trend surface of the given order over all the data."""
    scale = max(float(np.ptp(p, axis=0).max()), 1e-12)
    a = _poly_design(p[:, 0] / scale, p[:, 1] / scale, order)
    if len(pv) < a.shape[1]:
        raise ContouringError(f"A order-{order} trend surface needs at least {a.shape[1]} points.")
    coef, *_ = np.linalg.lstsq(a, pv, rcond=None)
    return _poly_design(q[:, 0] / scale, q[:, 1] / scale, order) @ coef


def _local_polynomial(
    p: np.ndarray, pv: np.ndarray, q: np.ndarray, order: int, power: float, chunk: int = 4096
) -> np.ndarray:
    """
    Weighted least-squares polynomial about each node over its SEARCH_POINTS
    nearest readings, weights (1 - d / R)^power with R the distance to the
    farthest of them (Surfer's local polynomial). The estimate is the fitted
    constant term.
    """
    n_terms = _poly_design(np.zeros(1), np.zeros(1), order).shape[-1]
    if len(pv) <= n_terms:
        raise ContouringError(f"A local order-{order} polynomial needs more than {n_terms} points.")
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        qs = q[s:s + chunk]
        d, i = _neighbours(p, qs, SEARCH_POINTS)
        r = d[:, -1:] * 1.0001 + 1e-12
        w = (1.0 - d / r) ** power
        a = _poly_design((p[i, 0] - qs[:, :1]) / r, (p[i, 1] - qs[:, 1:]) / r, order)
        aw = a * w[..., None]
        m = np.einsum("nki,nkj->nij", aw, a)
        rhs = np.einsum("nki,nk->ni", aw, pv[i])
        out[s:s + chunk] = np.einsum("nj,nj->n", np.linalg.pinv(m)[:, 0, :], rhs)
    return out


def _shepard(p: np.ndarray, pv: np.ndarray, q: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """
    Modified Shepard's method (Franke & Nielson, 1980): a quadratic is fitted
    about every reading to its SHEPARD_QUADRATIC_POINTS neighbours, and the
    nodes blend the quadratics with weights ((R - d)+ / (R d))², R being the
    radius of each reading's SHEPARD_WEIGHT_POINTS neighbours.
    """
    n = len(pv)
    if n < 6:
        raise ContouringError("Modified Shepard's method needs at least 6 points.")
    tree = cKDTree(p)
    nq = min(SHEPARD_QUADRATIC_POINTS, n - 1)
    dq, iq = tree.query(p, k=nq + 1)
    dq, iq = dq[:, 1:], iq[:, 1:]                      # drop the reading itself
    rq = dq[:, -1:] * 1.01 + 1e-12
    wq = (np.clip(rq - dq, 0, None) / (rq * np.where(dq == 0, 1.0, dq))) ** 2
    dx, dy = p[iq, 0] - p[:, None, 0], p[iq, 1] - p[:, None, 1]
    a = np.stack([dx, dy, dx * dx, dx * dy, dy * dy], axis=-1)
    aw = a * wq[..., None]
    coef = np.einsum(
        "nij,nj->ni", np.linalg.pinv(np.einsum("nki,nkj->nij", aw, a)),
        np.einsum("nki,nk->ni", aw, pv[iq] - pv[:, None]),
    )
    nw = min(SHEPARD_WEIGHT_POINTS, n - 1)
    rw = tree.query(p, k=nw + 1)[0][:, -1]

    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        qs = q[s:s + chunk]
        d, i = _neighbours(p, qs, 2 * nw)
        r = rw[i]
        safe = np.where(d == 0, 1.0, d)
        w = (np.clip(r - d, 0, None) / (r * safe)) ** 2
        none = w.sum(axis=1) == 0                     # outside every reading's radius
        w[none] = 1.0 / safe[none] ** 2
        ex, ey = qs[:, None, 0] - p[i, 0], qs[:, None, 1] - p[i, 1]
        c = coef[i]
        local = (pv[i] + c[..., 0] * ex + c[..., 1] * ey
                 + c[..., 2] * ex * ex + c[..., 3] * ex * ey + c[..., 4] * ey * ey)
        z = np.sum(w * local, axis=1) / np.sum(w, axis=1)
        hit = d[:, 0] == 0
        z[hit] = pv[i[hit, 0]]
        out[s:s + chunk] = z
    return out


def auto_search_radius(p: np.ndarray, points: int = MOVING_AVERAGE_POINTS) -> float:
    """Radius of a circle holding about *points* readings at the survey's mean density."""
    area = float(np.prod(np.ptp(p, axis=0)))
    if area <= 0:
        raise ContouringError("Could not determine a search radius (the data have no area).")
    return math.sqrt(points * area / (math.pi * len(p)))


def _search_statistic(
    p: np.ndarray, pv: np.ndarray, q: np.ndarray, radius: float, statistic: str
) -> np.ndarray:
    """
    A statistic of the readings within *radius* of each node (0 = automatic,
    see auto_search_radius): the mean for the moving average, else one of
    METRICS. Nodes with no reading in range are blank (count and density: 0).
    """
    if statistic != "mean" and statistic not in METRICS:
        raise ValueError(f"Unknown statistic: {statistic!r}")
    radius = radius if radius > 0 else auto_search_radius(p)
    k = SEARCH_POINTS if statistic == "mean" else METRICS_MAX_POINTS
    d, i = _neighbours(p, q, k, radius)
    inside = np.isfinite(d)
    count = inside.sum(axis=1).astype(float)
    if statistic == "count":
        return count
    if statistic == "density":
        return count / (math.pi * radius ** 2)
    v = np.where(inside, pv[np.where(inside, i, 0)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # nodes with no reading in range
        if statistic == "mean":
            return np.nanmean(v, axis=1)
        if statistic == "median":
            return np.nanmedian(v, axis=1)
        if statistic == "minimum":
            return np.nanmin(v, axis=1)
        if statistic == "maximum":
            return np.nanmax(v, axis=1)
        if statistic == "range":
            return np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
        return np.nanstd(v, axis=1, ddof=1)


def _min_curvature(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, spec: GridSpec, tension: float
) -> np.ndarray:
    """
    Minimum curvature (Briggs, 1974) with tension (Smith & Wessel, 1990): the
    grid that minimises (1 - T) × the thin-plate curvature energy
    (z_xx² + 2 z_xy² + z_yy²) + T × the gradient energy, while its bilinear
    interpolation passes through the readings (weighted least squares,
    MIN_CURVATURE_DATA_WEIGHT). Edges are free. Solved directly as one sparse
    system rather than by Surfer's iterations.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    if not 0.0 <= tension < 1.0:
        raise ContouringError("The tension must be at least 0 and below 1.")
    nx, ny = spec.nx, spec.ny
    n = nx * ny
    if n > MIN_CURVATURE_MAX_NODES:
        raise GridTooLargeError(
            f"Minimum curvature is limited to {MIN_CURVATURE_MAX_NODES:,} grid nodes "
            f"(this grid has {n:,}) — increase the cell size."
        )
    if nx < 3 or ny < 3:
        raise ContouringError("Minimum curvature needs a grid of at least 3 × 3 nodes.")
    idx = np.arange(n).reshape(ny, nx)

    def stencil(*terms):
        rows = np.arange(terms[0][0].size)
        return sparse.csr_matrix(
            (np.concatenate([np.full(rows.size, c, dtype=float) for _, c in terms]),
             (np.tile(rows, len(terms)), np.concatenate([ids.ravel() for ids, _ in terms]))),
            shape=(rows.size, n),
        )

    dxx = stencil((idx[:, :-2], 1), (idx[:, 1:-1], -2), (idx[:, 2:], 1))
    dyy = stencil((idx[:-2], 1), (idx[1:-1], -2), (idx[2:], 1))
    dxy = stencil((idx[:-1, :-1], 1), (idx[:-1, 1:], -1), (idx[1:, :-1], -1), (idx[1:, 1:], 1))
    dx = stencil((idx[:, :-1], -1), (idx[:, 1:], 1))
    dy = stencil((idx[:-1], -1), (idx[1:], 1))

    # readings up to half a cell outside the outer nodes extrapolate the edge cells linearly
    fx = (px - spec.xs[0]) / spec.cell
    fy = (py - spec.ys[0]) / spec.cell
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    tx, ty = fx - i0, fy - j0
    rows = np.arange(len(pv))
    b = sparse.csr_matrix(
        (np.concatenate([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]),
         (np.tile(rows, 4), np.concatenate([idx[j0, i0], idx[j0, i0 + 1],
                                            idx[j0 + 1, i0], idx[j0 + 1, i0 + 1]]))),
        shape=(len(pv), n),
    )
    w = MIN_CURVATURE_DATA_WEIGHT
    a = ((1.0 - tension) * (dxx.T @ dxx + dyy.T @ dyy + 2.0 * (dxy.T @ dxy))
         + tension * (dx.T @ dx + dy.T @ dy) + w * (b.T @ b))
    mean = float(np.mean(pv))
    try:
        z = spsolve(a.tocsc(), w * (b.T @ (pv - mean)))
    except RuntimeError as exc:          # singular: e.g. all readings on one line
        raise ContouringError(f"Minimum curvature failed ({exc}).") from exc
    if not np.all(np.isfinite(z)):
        raise ContouringError("Minimum curvature failed (the readings do not span an area).")
    return (z + mean).reshape(ny, nx)


def _natural_neighbour(
    px: np.ndarray, py: np.ndarray, pv: np.ndarray, spec: GridSpec,
    max_distance: float | None = None,
) -> np.ndarray:
    """
    Natural-neighbour (Sibson) interpolation in its discrete form (Park et
    al., 2006): every point of a raster NATURAL_NEIGHBOUR_SUPERSAMPLE times
    finer than the grid takes the value of its nearest reading and passes it
    to all grid nodes within that distance; each node is the mean of what it
    receives, i.e. readings weighted by the area each would lose to the node.
    Distances are capped at *max_distance* (the blanking distance).
    """
    s = NATURAL_NEIGHBOUR_SUPERSAMPLE
    h = spec.cell / s
    fx = spec.x0 + (np.arange(spec.nx * s) + 0.5) * h
    fy = spec.y0 + (np.arange(spec.ny * s) + 0.5) * h
    rx, ry = (a.ravel() for a in np.meshgrid(fx, fy))
    d, i = cKDTree(np.column_stack([px, py])).query(np.column_stack([rx, ry]))
    if max_distance:
        d = np.minimum(d, max_distance)
    if float(np.sum(np.pi * (d / spec.cell) ** 2 + 1.0)) > NATURAL_NEIGHBOUR_MAX_WORK:
        raise GridTooLargeError(
            "Natural neighbour would take too long on this grid — increase the cell size "
            "or set a smaller blanking distance."
        )
    values = pv[i]
    # grid node nearest each raster point, and the raster point's offset from it (cells)
    bi = np.minimum(np.floor((rx - spec.x0) / spec.cell).astype(np.int64), spec.nx - 1)
    bj = np.minimum(np.floor((ry - spec.y0) / spec.cell).astype(np.int64), spec.ny - 1)
    ox = (rx - spec.xs[bi]) / spec.cell
    oy = (ry - spec.ys[bj]) / spec.cell
    r = d / spec.cell
    by_radius = np.argsort(-r, kind="stable")
    r_ascending = np.sort(r)
    n = spec.n_nodes
    total = np.zeros(n)
    count = np.zeros(n)
    reach = int(math.ceil(float(r.max()) + 0.5))
    for dj in range(-reach, reach + 1):
        for di in range(-reach, reach + 1):
            nearest = max(math.hypot(di, dj) - math.sqrt(0.5), 0.0)   # closest a raster point can be
            src = by_radius[: r.size - np.searchsorted(r_ascending, nearest, side="left")]
            if src.size == 0:
                continue
            tx, ty = bi[src] + di, bj[src] + dj
            ok = ((tx >= 0) & (tx < spec.nx) & (ty >= 0) & (ty < spec.ny)
                  & (np.hypot(di - ox[src], dj - oy[src]) <= r[src]))
            target = ty[ok] * spec.nx + tx[ok]
            total += np.bincount(target, weights=values[src[ok]], minlength=n)
            count += np.bincount(target, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = total / count
    if np.isnan(z).any():   # a node no raster point reached: its own nearest reading
        miss = np.isnan(z)
        xx, yy = np.meshgrid(spec.xs, spec.ys)
        z[miss] = pv[cKDTree(np.column_stack([px, py])).query(
            np.column_stack([xx.ravel()[miss], yy.ravel()[miss]]))[1]]
    return z.reshape(spec.ny, spec.nx)


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


@dataclass(frozen=True)
class ColourStyle:
    """How a map or pseudo-section is coloured (keys of the COLOUR_* / MAP_DISPLAYS tables)."""
    cmap: str = "viridis"
    reverse: bool = False
    range_mode: str = "percentile"
    percentiles: tuple[float, float] = (2.0, 98.0)
    vmin: float | None = None             # range_mode "fixed"; None = data minimum
    vmax: float | None = None
    scale: str = "linear"
    display: str = "filled"
    contour_lines: bool = False
    n_levels: int = 20


def colour_map(name: str, reverse: bool = False):
    """Matplotlib colour map by COLOUR_MAPS key (or any Matplotlib name)."""
    if name == "surfer_rainbow":
        cmap = LinearSegmentedColormap.from_list("surfer_rainbow", SURFER_RAINBOW)
    else:
        cmap = matplotlib.colormaps[name]
    return cmap.reversed() if reverse else cmap


def colour_levels(values: np.ndarray, style: ColourStyle) -> np.ndarray:
    """
    style.n_levels + 1 increasing colour boundaries over the style's range:
    evenly spaced (linear), geometric (log) or at quantiles of the values
    (histogram-equalised, equal map area per colour).
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ContouringError("Nothing to contour (all values blank).")
    if style.range_mode == "minmax":
        lo, hi = float(finite.min()), float(finite.max())
    elif style.range_mode == "fixed":
        lo = float(finite.min()) if style.vmin is None else float(style.vmin)
        hi = float(finite.max()) if style.vmax is None else float(style.vmax)
        if hi <= lo:
            raise ContouringError("The colour maximum must be above the colour minimum.")
    else:
        p_lo, p_hi = style.percentiles
        if not 0.0 <= p_lo < p_hi <= 100.0:
            raise ContouringError("The colour percentiles must satisfy 0 ≤ lower < upper ≤ 100.")
        lo, hi = (float(v) for v in np.percentile(finite, [p_lo, p_hi]))
    if hi - lo <= 1e-9 * max(abs(lo), abs(hi), 1e-12):  # flat up to round-off
        pad = max(abs(lo) * 1e-6, 1e-12)
        lo, hi = lo - pad, hi + pad
    n = style.n_levels
    if style.scale == "log":
        if lo <= 0:
            raise ContouringError(
                f"A logarithmic colour scale needs positive values; this map goes down to "
                f"{lo:.4g}. Use a linear scale or a fixed minimum above 0."
            )
        return np.geomspace(lo, hi, n + 1)
    if style.scale == "equalised":
        inside = finite[(finite >= lo) & (finite <= hi)]
        levels = np.unique(np.quantile(inside, np.linspace(0.0, 1.0, n + 1)))
        if levels.size >= 3:
            levels[0], levels[-1] = lo, hi
            return levels
    return np.linspace(lo, hi, n + 1)


def contour_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    """n_levels + 1 evenly spaced boundaries spanning the 2nd-98th percentile."""
    return colour_levels(values, ColourStyle(n_levels=n_levels))


def _extend(style: ColourStyle) -> str:
    """Colour-bar arrows for values beyond the range; the full data range has none."""
    return "neither" if style.range_mode == "minmax" else "both"


def _colour_norm(levels: np.ndarray, style: ColourStyle, n_colours: int):
    """Norm placing the colours: None lets filled linear contours map levels evenly."""
    if style.display == "filled":
        if style.scale == "linear":
            return None
        return BoundaryNorm(levels, n_colours, extend=_extend(style))
    if style.scale == "log":
        return LogNorm(levels[0], levels[-1])
    if style.scale == "equalised":
        u = np.linspace(0.0, 1.0, len(levels))
        return FuncNorm(
            (lambda x: np.interp(x, levels, u), lambda y: np.interp(y, u, levels)),
            vmin=levels[0], vmax=levels[-1],
        )
    return Normalize(levels[0], levels[-1])


def _draw_field(ax, x, y, z, style: ColourStyle, levels: np.ndarray, edges=None):
    """
    Draws z (rows along y) as filled contours or a continuous image, with
    optional contour lines. *edges* (x, y cell edges) draws the image cells
    exactly; without them cells are centred on x, y. Returns the mappable.
    """
    cmap = colour_map(style.cmap, style.reverse)
    norm = _colour_norm(levels, style, cmap.N)
    zm = np.ma.masked_invalid(z)
    if style.display == "image":
        if edges is not None:
            mappable = ax.pcolormesh(edges[0], edges[1], zm, cmap=cmap, norm=norm, shading="flat")
        else:
            mappable = ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, shading="nearest")
    else:
        mappable = ax.contourf(x, y, zm, levels=levels, cmap=cmap, norm=norm,
                               extend=_extend(style))
    if style.contour_lines:
        ax.contour(x, y, zm, levels=levels, colors="k", linewidths=0.4, alpha=0.6)
    return mappable


def _colour_bar(fig, mappable, ax, label: str, levels: np.ndarray, style: ColourStyle):
    """Colour bar with arrows for values beyond the range; at most ~10 labelled ticks."""
    extend = {"extend": _extend(style)} if style.display == "image" else {}
    bar = fig.colorbar(mappable, ax=ax, label=label, **extend)
    if style.display == "filled" and style.scale != "linear":
        ticks = levels[:: max(1, math.ceil(len(levels) / 10))]
        bar.set_ticks(ticks, labels=[f"{t:.4g}" for t in ticks])
    return bar


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
    origin: tuple[float, float] | None  # (lon0, lat0) if input was degrees, local projection
    levelled: bool
    n_raw: int
    epsg: int | None = None             # CRS of x / y when known (UTM or user-given)
    options: dict = field(default_factory=dict)   # the method's settings (method_options)


def method_title(result: AreaMapResult) -> str:
    """Method name for titles, with the setting that changes what the map shows."""
    name = METHODS[result.method]
    if result.method == "rbf":
        return f"{name} ({RBF_KERNELS[result.options['kernel']].lower()})"
    if result.method == "metrics":
        return f"{name}: {METRICS[result.options['statistic']].lower()}"
    if result.method in ("polynomial", "local_polynomial"):
        return f"{name}, order {result.options['order']}"
    return name


def result_label(result: AreaMapResult, label: str) -> str:
    """Colour-bar label: data metrics other than value statistics are not in the data's units."""
    if result.method != "metrics":
        return label
    statistic = result.options["statistic"]
    if statistic == "count":
        return "Readings within the search radius"
    if statistic == "density":
        return "Readings per m²"
    return f"{METRICS[statistic]} of {label}"


def survey_xy(
    df: pd.DataFrame,
    value_col: str,
    coord_mode: str = "auto",
    projection: str = "local",
    xy_epsg: int | None = None,
    origin: tuple[float, float] | None = None,
    epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, int | None]:
    """
    Map coordinates in metres and values of the usable readings.

    Degrees are projected to local metres about *origin* (default: the
    centroid) or, with projection="utm", to UTM (zone of the centroid unless
    *epsg* is given). Metre coordinates carry *xy_epsg* when known.
    Returns (x, y, v, keep mask over df rows, origin, epsg).
    """
    x_col, y_col, is_deg = find_coordinate_columns(df, coord_mode)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[keep], y[keep], v[keep]
    if not is_deg:
        return x, y, v, keep, None, xy_epsg or None
    fix = ~((x == 0) & (y == 0))  # GPS no-fix rows are logged as (0, 0)
    keep[keep] = fix
    x, y, v = x[fix], y[fix], v[fix]
    if projection == "utm":
        import gridtools

        if x.size == 0:
            raise ContouringError("No usable coordinates.")
        epsg = epsg or gridtools.utm_epsg(float(np.mean(x)), float(np.mean(y)))
        x, y = gridtools.lonlat_to_epsg(x, y, epsg)
        return x, y, v, keep, None, epsg
    x, y, origin = project_to_local_metres(x, y, origin)
    return x, y, v, keep, origin, None


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
    projection: str = "local",
    xy_epsg: int | None = None,
    origin: tuple[float, float] | None = None,
    epsg: int | None = None,
    grid_spec: GridSpec | None = None,
    options: dict | None = None,
) -> AreaMapResult:
    """
    Full area-map pipeline for one value column of a raw GEM table.
    *options* are the method's settings (see METHOD_DEFAULTS).

    *origin*, *epsg* and *grid_spec* pin the projection and grid, so that two
    surveys can be gridded on the same nodes (difference maps).
    """
    x, y, v, keep, origin, epsg = survey_xy(
        df, value_col, coord_mode, projection, xy_epsg, origin, epsg
    )

    levelled = False
    if level and line_col in df.columns:
        v = level_lines(v, df[line_col].to_numpy()[keep])
        levelled = True

    lines = df[line_col].to_numpy()[keep] if line_col in df.columns else None
    check_geometry(x, y, lines)
    if grid_spec is not None:
        spec = grid_spec
    else:
        cell = cell_size if cell_size else auto_cell_size(x, y)
        spec = make_grid(x, y, cell)
    bx, by, bv = block_median(x, y, v, spec)
    while method == "kriging" and not cell_size and grid_spec is None and len(bv) > KRIGE_MAX_POINTS:
        cell = round_sig(cell * KRIGE_CELL_GROWTH)
        spec = make_grid(x, y, cell)
        bx, by, bv = block_median(x, y, v, spec)
    if len(bv) < MIN_POINTS:
        raise ContouringError("Not enough points to grid.")

    variogram = fit_variogram(bx, by, bv, variogram_model) if method == "kriging" else None
    dist = blank_distance if blank_distance else default_blank_distance(spec, bx, by)
    z, var = predict_grid(bx, by, bv, spec, method, smoothing, variogram,
                          options=options, max_distance=dist)
    z = blank_far(z, spec, bx, by, dist)
    if var is not None:
        var = blank_far(var, spec, bx, by, dist)

    return AreaMapResult(
        spec=spec, z=z, variance=var, bx=bx, by=by, bv=bv, method=method,
        variogram=variogram, blank_distance=dist, origin=origin,
        levelled=levelled, n_raw=int(keep.sum()), epsg=epsg,
        options=method_options(method, options),
    )


def compute_difference_map(
    df_a: pd.DataFrame, df_b: pd.DataFrame, value_col: str, **params
) -> AreaMapResult:
    """
    B minus A on one shared grid (time-lapse). Both surveys use the
    projection and origin of A; the grid covers both; blank where either is.
    *params* are those of compute_area_map (cell_size None = automatic from A).
    """
    coord = {k: params.get(k) for k in ("coord_mode", "projection", "xy_epsg")}
    coord = {k: v for k, v in coord.items() if v is not None}
    mode = coord.get("coord_mode", "auto")
    if find_coordinate_columns(df_a, mode)[2] != find_coordinate_columns(df_b, mode)[2]:
        raise ContouringError(
            "The two surveys use different coordinates (one in degrees, one in metres)."
        )
    xa, ya, _, _, origin, epsg = survey_xy(df_a, value_col, **coord)
    xb, yb, _, _, _, _ = survey_xy(df_b, value_col, origin=origin, epsg=epsg, **coord)
    if xa.size == 0 or xb.size == 0:
        raise ContouringError("Not enough points to grid.")
    cell = params.get("cell_size") or auto_cell_size(xa, ya)
    spec = make_grid(np.concatenate([xa, xb]), np.concatenate([ya, yb]), cell)
    pinned = {**params, "origin": origin, "epsg": epsg, "grid_spec": spec}
    a = compute_area_map(df_a, value_col, **pinned)
    b = compute_area_map(df_b, value_col, **pinned)
    z = b.z - a.z
    return AreaMapResult(
        spec=spec, z=z, variance=None,
        bx=np.concatenate([a.bx, b.bx]), by=np.concatenate([a.by, b.by]),
        bv=np.concatenate([-a.bv, b.bv]), method=a.method, variogram=None,
        blank_distance=max(a.blank_distance, b.blank_distance), origin=origin,
        levelled=a.levelled, n_raw=a.n_raw + b.n_raw, epsg=epsg, options=a.options,
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
    options: dict | None = None,
    spec: GridSpec | None = None,
) -> dict[str, float]:
    """
    k-fold CV on the block-reduced points. Returns rmse, mae, n (predicted points).

    For kriging the variogram is held fixed across folds (pass the fit used
    for the map). GRID_METHODS grid each training fold on *spec* (default:
    automatic cell size) and are read at the held-out points bilinearly.
    """
    if method == "metrics":
        raise ContouringError(
            "Cross-validation compares estimates with held-out readings; data metrics are "
            "summaries of the readings, not estimates."
        )
    px, py, pv = (np.asarray(a, dtype=float) for a in (px, py, pv))
    if method in GRID_METHODS and spec is None:
        spec = make_grid(px, py, auto_cell_size(px, py))
    folds = np.random.default_rng(seed).permutation(len(pv)) % k
    errors = []
    for f in range(k):
        test = folds == f
        train = ~test
        if method in GRID_METHODS:
            dist = default_blank_distance(spec, px[train], py[train])
            grid, _ = predict_grid(px[train], py[train], pv[train], spec, method,
                                   options=options, max_distance=dist)
            est = RegularGridInterpolator(
                (spec.ys, spec.xs), grid, bounds_error=False, fill_value=np.nan,
            )(np.column_stack([py[test], px[test]]))
        else:
            est, _ = predict(
                px[train], py[train], pv[train], px[test], py[test],
                method, smoothing, variogram, options=options,
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
    cmap: str = "viridis",
    symmetric: bool = False,
    style: ColourStyle | None = None,
) -> plt.Figure:
    """
    Contour map or image; for kriging, adds a kriging standard-deviation panel.
    *style* sets the colours (default: *cmap* with *n_levels* filled contours
    over the 2nd-98th percentile). *symmetric* centres a linear colour range
    on zero (difference maps).
    """
    spec = result.spec
    style = style or ColourStyle(cmap=cmap, n_levels=n_levels)
    if symmetric:
        style = replace(style, scale="linear")
    # Compute levels first: they may raise, and no figure should be left open.
    levels = colour_levels(result.z, style)
    if symmetric:
        m = float(np.max(np.abs(levels[[0, -1]])))
        levels = np.linspace(-m, m, style.n_levels + 1)
    std = np.sqrt(result.variance) if result.variance is not None else None
    std_style = ColourStyle(cmap="magma", n_levels=style.n_levels, display=style.display)
    std_levels = colour_levels(std, std_style) if std is not None else None
    panels = 2 if std is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 6), squeeze=False)
    xs, ys = spec.xs, spec.ys
    edges = (spec.x0 + spec.cell * np.arange(spec.nx + 1),
             spec.y0 + spec.cell * np.arange(spec.ny + 1))

    ax = axes[0, 0]
    mappable = _draw_field(ax, xs, ys, result.z, style, levels, edges)
    _colour_bar(fig, mappable, ax, label, levels, style)
    if show_points:
        ax.plot(result.bx, result.by, ",", color="k", alpha=0.4)
    ax.set_title(title + (" — line-levelled (per-line median)" if result.levelled else ""))

    if std is not None:
        ax2 = axes[0, 1]
        mappable2 = _draw_field(ax2, xs, ys, std, std_style, std_levels, edges)
        _colour_bar(fig, mappable2, ax2, f"Kriging std. dev. — {label}", std_levels, std_style)
        ax2.set_title("Kriging standard deviation")

    if result.epsg:
        xlab, ylab = f"Easting (m, EPSG:{result.epsg})", f"Northing (m, EPSG:{result.epsg})"
    elif result.origin is not None:
        xlab, ylab = "Easting (local m)", "Northing (local m)"
    else:
        xlab, ylab = "X (m)", "Y (m)"
    for a in axes[0]:
        a.ticklabel_format(useOffset=False, style="plain")  # full map coordinates
        a.set_xlim(edges[0][0], edges[0][-1])               # the grid, without margins
        a.set_ylim(edges[1][0], edges[1][-1])
        a.set_aspect("equal")
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
    fig.tight_layout()
    return fig


def make_pseudosection_figure(
    ps: PseudoSection, label: str, title: str, n_levels: int = 20,
    style: ColourStyle | None = None,
) -> plt.Figure:
    """Distance x frequency contour with a white line at each measured frequency."""
    style = style or ColourStyle(n_levels=n_levels)
    levels = colour_levels(ps.values, style)  # may raise; before any figure exists
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
    mappable = _draw_field(ax, ps.distance, rows, ps.values, style, levels)
    for r in rows:
        ax.axhline(r, color="white", linewidth=0.6, alpha=0.8)
    _colour_bar(fig, mappable, ax, label, levels, style)
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
    elif result.epsg:
        from pyproj import Transformer

        t = Transformer.from_crs(result.epsg, 4326, always_xy=True)
        out["lon"], out["lat"] = t.transform(out["x"], out["y"])
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
