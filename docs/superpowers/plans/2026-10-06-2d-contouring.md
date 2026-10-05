# 2D Contouring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two optional, independently togglable 2D views to the Streamlit app — plan-view area maps (spline / ordinary kriging / linear gridding of GEM X–Y data) and distance × frequency pseudo-sections of the representative profiles.

**Architecture:** All numerical work lives in a new pure module `contouring.py` (no Streamlit imports), built and unit-tested bottom-up. `RIs_v2.py` only gains sidebar controls, `st.cache_data` wrappers and rendering functions that call the module after the existing per-mode sections. Spec: `docs/superpowers/specs/2026-10-05-2d-contouring-design.md` (read it first, including the 2026-10-06 amendments).

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy (`RBFInterpolator`, `griddata`, `cKDTree`, `curve_fit`), PyKrige ≥ 1.7 (ordinary kriging, C backend), Matplotlib, Streamlit ≥ 1.26 (`st.toggle`), pytest, `streamlit.testing.v1.AppTest`.

**Repository rules:**
- Never add any AI/assistant attribution to commits (no `Co-Authored-By`, no "Generated with" lines). Commit messages are plain one-liners.
- Work in the existing branch; do not push unless asked.
- All commands run from the repository root. On Windows use Git Bash.

**Domain notes for the implementer (read once):**
- GEM files are flat tables: one row per reading, columns `Line`, `X`, `Y` (sometimes `Lat`/`Lon`), and one column per frequency such as `EC4525Hz[mS/m]` (conductivity, mS/m) and `MSusc4525Hz[1/1000]` (magnetic susceptibility, 10⁻³ SI).
- The existing app treats each `Line` as a repeat pass of one transect and `Y` as distance along it. The new **area map** instead treats the file as an area survey (many parallel lines) and grids it in plan view. The new **pseudo-section** reuses the existing per-frequency mean profiles.
- "Block-median reduction" = one median point per grid cell before gridding, so densely sampled lines do not dominate.
- A "variogram" describes how dissimilar values are as a function of separation distance; kriging needs a fitted one. Parameters: *partial sill* (structured variance), *range* (correlation distance), *nugget* (uncorrelated noise).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `contouring.py` | Create | Coordinates, levelling, gridding, variogram, kriging, blanking, pseudo-section, figures, exports. Pure functions. |
| `tests/test_contouring.py` | Create | Unit tests for `contouring.py` on synthetic data. |
| `tests/test_app_contouring.py` | Create | Headless UI smoke tests via `AppTest`. |
| `pytest.ini` | Create | `pythonpath = .` so tests import the root modules. |
| `requirements.txt` | Modify | Add `pykrige`. |
| `requirements-dev.txt` | Create | `-r requirements.txt` + `pytest`. |
| `RIs_v2.py` | Modify | `ContourSettings`, sidebar, cached wrappers, render functions, wiring. |
| `README.md` | Modify | "2D Contouring" section + references. |

---

### Task 0: Tooling and dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`, `pytest.ini`, `tests/` (directory)

- [ ] **Step 1: Add PyKrige to runtime requirements**

Replace the full contents of `requirements.txt` with:

```text
streamlit
matplotlib
numpy
scipy
pandas
openpyxl
pykrige
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```text
-r requirements.txt
pytest
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Install and check the PyKrige C extension**

Run:
```bash
python -m pip install -r requirements-dev.txt
python -c "import pykrige, pykrige.lib.cok; print(pykrige.__version__, 'C backend ok')"
```
Expected: a version ≥ 1.7 followed by `C backend ok`. (If the C extension is missing PyKrige silently falls back to a slow pure-Python loop; the app still works but kriging will be slow — report this.)

- [ ] **Step 5: Commit**

```bash
mkdir -p tests
git add requirements.txt requirements-dev.txt pytest.ini
git commit -m "Add PyKrige dependency and pytest configuration"
```

---

### Task 1: Module skeleton, units and coordinates

**Files:**
- Create: `contouring.py`
- Create: `tests/test_contouring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_contouring.py`:

```python
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import contouring as C  # noqa: E402

VALUE_COL = "EC4525Hz[mS/m]"


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def bump(x, y):
    """Smooth Gaussian anomaly, height 10, on a background of 20."""
    return 20.0 + 10.0 * np.exp(-((x - 25) ** 2 + (y - 25) ** 2) / (2 * 8.0 ** 2))


def line_survey(n_lines=11, spacing=5.0, along=0.5, length=50.0):
    """Parallel N-S lines over a 50 x 50 m area, sampled every *along* metres."""
    rows = []
    for i in range(n_lines):
        ys = np.arange(0.0, length + 1e-9, along)
        xs = np.full_like(ys, i * spacing)
        rows.append(pd.DataFrame({"Line": i, "X": xs, "Y": ys, VALUE_COL: bump(xs, ys)}))
    return pd.concat(rows, ignore_index=True)


def bump_rmse(res):
    xx, yy = np.meshgrid(res.spec.xs, res.spec.ys)
    ok = np.isfinite(res.z)
    return float(np.sqrt(np.mean((res.z[ok] - bump(xx[ok], yy[ok])) ** 2)))


def make_result(spec, z, origin=None):
    return C.AreaMapResult(
        spec=spec, z=z, variance=None, bx=np.array([]), by=np.array([]),
        bv=np.array([]), method="spline", variogram=None, blank_distance=1.0,
        origin=origin, levelled=False, n_raw=0,
    )


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_value_label_units():
    assert C.value_label("EC", True) == "EC (mS/m)"
    assert C.value_label("MS", True) == "MS (10⁻³ SI)"
    assert C.value_label("EC", False) == "EC (input units)"


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * C.EARTH_RADIUS_M * math.asin(math.sqrt(a))


def test_projection_matches_haversine_within_0_1_percent():
    lon = np.array([4.0000, 4.0200, 4.0050])
    lat = np.array([50.0000, 50.0100, 50.0150])
    x, y, _ = C.project_to_local_metres(lon, lat)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        planar = math.hypot(x[i] - x[j], y[i] - y[j])
        true = haversine(lon[i], lat[i], lon[j], lat[j])
        assert abs(planar - true) / true < 1e-3


def test_projection_round_trip():
    lon = np.array([4.0, 4.01, 4.02])
    lat = np.array([50.0, 50.005, 50.01])
    x, y, origin = C.project_to_local_metres(lon, lat)
    lon2, lat2 = C.local_metres_to_lonlat(x, y, origin)
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)


def test_find_coordinates_metres():
    df = pd.DataFrame({"X": [500000.0, 500050.0], "Y": [4500000.0, 4500050.0]})
    assert C.find_coordinate_columns(df) == ("X", "Y", False)


def test_find_coordinates_xy_degrees():
    df = pd.DataFrame({"X": [4.0, 4.001], "Y": [50.0, 50.002]})
    assert C.find_coordinate_columns(df) == ("X", "Y", True)


def test_find_coordinates_small_local_grid_is_metres():
    df = pd.DataFrame({"X": [0.0, 80.0], "Y": [0.0, 60.0]})
    assert C.find_coordinate_columns(df) == ("X", "Y", False)


def test_find_coordinates_override():
    df = pd.DataFrame({"X": [0.0, 0.01], "Y": [0.0, 0.01]})
    assert C.find_coordinate_columns(df, "auto")[2] is True
    assert C.find_coordinate_columns(df, "metres")[2] is False
    assert C.find_coordinate_columns(pd.DataFrame({"X": [0.0, 80.0], "Y": [0.0, 60.0]}),
                                     "degrees")[2] is True


def test_find_coordinates_lat_lon_columns_take_precedence():
    df = pd.DataFrame({"X": [0.0, 80.0], "Y": [0.0, 60.0],
                       "Latitude": [50.0, 50.001], "LON": [4.0, 4.001]})
    assert C.find_coordinate_columns(df) == ("LON", "Latitude", True)


def test_find_coordinates_missing_raises():
    with pytest.raises(C.ContouringError, match="No coordinate"):
        C.find_coordinate_columns(pd.DataFrame({"Line": [1], "Dist": [0.0]}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'contouring'`.

- [ ] **Step 3: Create `contouring.py` with skeleton, units and coordinates**

```python
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
LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "long", "longitude"}
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

    Lat/Lon columns (case-insensitive) take precedence and are always degrees.
    Otherwise X/Y are used; *mode* is "auto", "metres" or "degrees".
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    lat_col = next((lower[n] for n in LAT_NAMES if n in lower), None)
    lon_col = next((lower[n] for n in LON_NAMES if n in lower), None)
    if lat_col is not None and lon_col is not None:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add contouring module with unit labels and coordinate handling"
```

---

### Task 2: Pre-processing — levelling, cell size, geometry check, grid, block median

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests to `tests/test_contouring.py`**

```python
# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def test_level_lines_removes_constant_offsets():
    # Field varies only along the lines, so every line has the same true median.
    y = np.tile(np.arange(0.0, 50.0, 0.5), 5)
    lines = np.repeat(np.arange(5), 100)
    truth = 20.0 + 0.1 * y
    offsets = np.array([3.0, -2.0, 0.5, 7.0, -4.0])
    levelled = C.level_lines(truth + offsets[lines], lines)
    resid = pd.Series(levelled - truth).groupby(lines).mean().to_numpy()
    assert np.ptp(resid) < 1e-9  # every line shifted by one common constant


def test_level_lines_also_removes_real_cross_line_trend():
    # Documents the caveat shown in the UI: levelling cannot tell a real
    # cross-line gradient from a line offset.
    lines = np.repeat(np.arange(4), 10)
    levelled = C.level_lines(lines * 5.0, lines)
    assert np.ptp(levelled) < 1e-9


def test_auto_cell_size_is_geometric_mean_spacing():
    df = line_survey()  # 5 m line spacing, 0.5 m along-line
    cell = C.auto_cell_size(df["X"].to_numpy(), df["Y"].to_numpy())
    assert 1.0 < cell < 2.5  # sqrt(5 * 0.5) ~= 1.6, edge effects aside


def test_make_grid_alignment_and_size():
    spec = C.make_grid(np.array([0.3, 9.7]), np.array([1.2, 4.9]), 1.0)
    assert (spec.x0, spec.y0, spec.nx, spec.ny) == (0.0, 1.0, 10, 4)
    np.testing.assert_allclose(spec.xs[:2], [0.5, 1.5])


def test_make_grid_too_large():
    with pytest.raises(C.GridTooLargeError):
        C.make_grid(np.array([0.0, 10000.0]), np.array([0.0, 10000.0]), 1.0)


def test_block_median_hand_example():
    spec = C.GridSpec(x0=0.0, y0=0.0, cell=1.0, nx=2, ny=1)
    x = np.array([0.1, 0.2, 0.9, 1.5])
    y = np.array([0.5, 0.5, 0.5, 0.5])
    v = np.array([1.0, 2.0, 10.0, 7.0])
    bx, by, bv = C.block_median(x, y, v, spec)
    np.testing.assert_allclose(bx, [0.2, 1.5])
    np.testing.assert_allclose(bv, [2.0, 7.0])


def test_check_geometry_collinear_raises():
    x = np.linspace(0, 100, 200)
    y = 0.5 * x + 1e-3 * np.sin(x)
    with pytest.raises(C.ContouringError, match="single transect"):
        C.check_geometry(x, y)


def test_check_geometry_area_passes():
    df = line_survey()
    C.check_geometry(df["X"].to_numpy(), df["Y"].to_numpy())


def test_check_geometry_too_few_points():
    with pytest.raises(C.ContouringError, match="Not enough"):
        C.check_geometry(np.arange(5.0), np.arange(5.0) ** 2)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: 9 new failures with `AttributeError: module 'contouring' has no attribute 'level_lines'` (and similar).

- [ ] **Step 3: Append the implementation to `contouring.py`**

```python


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


def check_geometry(x: np.ndarray, y: np.ndarray) -> None:
    """Raise ContouringError if too few points or points lie along one line."""
    if len(x) < MIN_POINTS:
        raise ContouringError("Not enough points to grid.")
    eig = np.linalg.eigvalsh(np.cov(np.vstack([x, y])))  # ascending
    if eig[1] <= 0 or math.sqrt(max(eig[0], 0.0) / eig[1]) < COLLINEAR_RATIO:
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add line levelling, grid definition and block-median reduction"
```

---

### Task 3: Variogram fitting

Why this exists: PyKrige's built-in variogram fit uses lag bins across the *full* distance range; on smooth fields it collapses to pure nugget and kriging returns the mean between lines (prototype RMSE 2.3 vs 0.02). We fit to half the maximum distance, weighted by pair counts (Oliver & Webster, 2014), and pass the parameters to PyKrige.

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing test**

```python
# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------

def test_fit_variogram_recovers_structure():
    df = line_survey()
    spec = C.make_grid(df["X"].to_numpy(), df["Y"].to_numpy(), 1.0)
    bx, by, bv = C.block_median(df["X"].to_numpy(), df["Y"].to_numpy(),
                                df[VALUE_COL].to_numpy(), spec)
    vf = C.fit_variogram(bx, by, bv, "spherical")
    assert vf.psill > 10 * vf.nugget  # smooth field: structured, not pure nugget
    assert 5.0 < vf.range < 100.0
    assert vf.parameters == [vf.psill, vf.range, vf.nugget]


def test_fit_variogram_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown variogram model"):
        C.fit_variogram(np.arange(20.0), np.arange(20.0) ** 0.5, np.arange(20.0), "cubic")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q -k variogram`
Expected: FAIL with `AttributeError: module 'contouring' has no attribute 'fit_variogram'`.

- [ ] **Step 3: Append the implementation**

```python


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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `20 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add empirical variogram fitting for kriging"
```

---

### Task 4: Interpolation, blanking and contour levels

Design notes: kriging always uses a 64-point local neighbourhood with PyKrige's C backend (global kriging took 46 s for 51k nodes; local ~2 s); `pseudo_inv=True` keeps zero-nugget Gaussian models stable (without it the prototype gave RMSE 951); PyKrige computes all pairwise distances at set-up, so input is capped at 4000 points.

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _block_points(cell=1.0):
    df = line_survey()
    spec = C.make_grid(df["X"].to_numpy(), df["Y"].to_numpy(), cell)
    bx, by, bv = C.block_median(df["X"].to_numpy(), df["Y"].to_numpy(),
                                df[VALUE_COL].to_numpy(), spec)
    return spec, bx, by, bv


@pytest.mark.parametrize("method", ["spline", "kriging"])
def test_predict_reproduces_bump_at_grid_nodes(method):
    spec, bx, by, bv = _block_points()
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    est, var = C.predict(bx, by, bv, xx, yy, method)
    assert np.sqrt(np.mean((est - bump(xx.ravel(), yy.ravel())) ** 2)) < 0.5
    if method == "kriging":
        assert var is not None and np.nanmin(var) >= 0.0
    else:
        assert var is None


@pytest.mark.parametrize("model", C.VARIOGRAM_MODELS)
def test_kriging_all_variogram_models_stable(model):
    # gaussian with ~zero nugget is ill-conditioned without the pseudo-inverse
    spec, bx, by, bv = _block_points()
    xx, yy = np.meshgrid(spec.xs, spec.ys)
    est, _ = C.predict(bx, by, bv, xx, yy, "kriging", variogram_model=model)
    assert np.sqrt(np.mean((est - bump(xx.ravel(), yy.ravel())) ** 2)) < 0.5


def test_kriging_point_cap():
    n = C.KRIGE_MAX_POINTS + 1
    p = np.random.default_rng(0).uniform(0, 100, (n, 2))
    vf = C.VariogramFit("spherical", 1.0, 10.0, 0.0, (), ())
    with pytest.raises(C.KrigingError, match="limited"):
        C.predict(p[:, 0], p[:, 1], np.ones(n), np.array([1.0]), np.array([1.0]),
                  "kriging", variogram=vf)


def test_linear_is_nan_outside_convex_hull():
    px = np.array([0.0, 10.0, 0.0, 10.0, 5.0])
    py = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
    est, _ = C.predict(px, py, np.arange(5.0), np.array([5.0, 20.0]),
                       np.array([5.0, 20.0]), "linear")
    assert np.isfinite(est[0]) and np.isnan(est[1])


def test_predict_unknown_method():
    with pytest.raises(ValueError, match="Unknown gridding method"):
        C.predict(np.arange(3.0), np.arange(3.0), np.arange(3.0),
                  np.array([0.0]), np.array([0.0]), "idw")


def test_blank_far_masks_distant_nodes():
    spec = C.GridSpec(x0=0.0, y0=0.0, cell=1.0, nx=10, ny=1)
    out = C.blank_far(np.ones((1, 10)), spec, np.array([0.5]), np.array([0.5]), 2.0)
    assert np.isfinite(out[0, :3]).all()  # nodes at 0.5, 1.5, 2.5
    assert np.isnan(out[0, 3:]).all()


def test_contour_levels_percentile_span():
    v = np.r_[np.linspace(0, 1, 98), 1000.0, np.nan]
    lv = C.contour_levels(v, 10)
    assert len(lv) == 11 and lv[-1] < 1000.0


def test_contour_levels_all_blank_raises():
    with pytest.raises(C.ContouringError, match="Nothing to contour"):
        C.contour_levels(np.array([np.nan, np.nan]), 10)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: new tests FAIL with `AttributeError: module 'contouring' has no attribute 'predict'` (and `blank_far`, `contour_levels`).

- [ ] **Step 3: Append the implementation**

```python


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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `31 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add spline, kriging and linear gridding with blanking"
```

---

### Task 5: Area-map pipeline and cross-validation

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Area-map pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["spline", "kriging"])
def test_compute_area_map_reproduces_smooth_bump(method):
    res = C.compute_area_map(line_survey(), VALUE_COL, method=method, cell_size=1.0)
    assert bump_rmse(res) < 0.5  # 5 % of bump height (10)
    if method == "kriging":
        assert res.variance is not None and res.variogram is not None
        assert np.nanmin(res.variance) >= 0.0
    else:
        assert res.variance is None and res.variogram is None


def test_compute_area_map_defaults_and_bookkeeping():
    df = line_survey()
    res = C.compute_area_map(df, VALUE_COL)
    assert res.n_raw == len(df)
    assert 1.0 < res.spec.cell < 2.5           # auto cell size
    assert res.blank_distance > 0
    assert not res.levelled and res.origin is None


def test_compute_area_map_degrees_records_origin():
    df = line_survey()
    lon0, lat0 = 4.0, 50.0
    df["Lon"] = lon0 + np.degrees(df["X"] / (C.EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    df["Lat"] = lat0 + np.degrees(df["Y"] / C.EARTH_RADIUS_M)
    res = C.compute_area_map(df, VALUE_COL, cell_size=1.0)
    assert res.origin is not None
    assert res.spec.nx == pytest.approx(51, abs=2)


def test_compute_area_map_levelling_flag():
    res = C.compute_area_map(line_survey(), VALUE_COL, level=True)
    assert res.levelled


def test_compute_area_map_transect_raises():
    df = line_survey(n_lines=1)
    df["X"] = df["X"] + 1e-4 * np.sin(df["Y"])
    with pytest.raises(C.ContouringError, match="single transect"):
        C.compute_area_map(df, VALUE_COL)


@pytest.mark.parametrize("method", ["spline", "kriging", "linear"])
def test_cross_validate_returns_finite_metrics(method):
    res = C.compute_area_map(line_survey(), VALUE_COL, method=method, cell_size=1.0)
    cv = C.cross_validate(res.bx, res.by, res.bv, method, variogram=res.variogram)
    assert cv["n"] > 0 and 0 <= cv["mae"] <= cv["rmse"] < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: new tests FAIL with `AttributeError: module 'contouring' has no attribute 'compute_area_map'`.

- [ ] **Step 3: Append the implementation**

```python


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
        x, y, origin = project_to_local_metres(x, y)

    levelled = False
    if level and line_col in df.columns:
        v = level_lines(v, df[line_col].to_numpy()[keep])
        levelled = True

    check_geometry(x, y)
    cell = cell_size if cell_size else auto_cell_size(x, y)
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

    dist = blank_distance if blank_distance else 2.0 * median_nn_spacing(bx, by)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `40 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add area-map pipeline and k-fold cross-validation"
```

---

### Task 6: Pseudo-section builder

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Pseudo-section
# ---------------------------------------------------------------------------

def test_parse_frequency():
    assert C.parse_frequency("4525Hz") == 4525.0
    assert C.parse_frequency("EC 93.5 kHz") == 93.5
    assert C.parse_frequency("north") is None


def test_pseudosection_aligns_overlap_and_sorts():
    a = pd.Series(np.arange(0.0, 51.0), index=np.arange(0.0, 51.0))             # 0..50, step 1
    b = pd.Series(np.arange(10.0, 60.5, 0.5), index=np.arange(10.0, 60.5, 0.5))  # 10..60, step 0.5
    ps = C.build_pseudosection({"38025Hz": a, "4525Hz": b})
    assert ps.labels == ["4525Hz", "38025Hz"]
    np.testing.assert_allclose(ps.frequencies, [4525.0, 38025.0])
    assert ps.distance[0] == 10.0 and ps.distance[-1] == 50.0
    assert np.diff(ps.distance)[0] == 0.5
    assert np.isfinite(ps.values).all()
    np.testing.assert_allclose(ps.values[0], ps.distance)  # b(d) = d
    np.testing.assert_allclose(ps.values[1], ps.distance)  # a(d) = d


def test_pseudosection_unparsed_labels_keep_order():
    s = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
    ps = C.build_pseudosection({"north": s, "south": s})
    assert ps.labels == ["north", "south"] and ps.frequencies is None


def test_pseudosection_needs_two_frequencies():
    s = pd.Series([1.0, 2.0], index=[0.0, 1.0])
    with pytest.raises(C.ContouringError, match="at least 2"):
        C.build_pseudosection({"4525Hz": s})


def test_pseudosection_no_overlap():
    a = pd.Series([1.0, 2.0], index=[0.0, 1.0])
    b = pd.Series([1.0, 2.0], index=[5.0, 6.0])
    with pytest.raises(C.ContouringError, match="overlap"):
        C.build_pseudosection({"1Hz": a, "2Hz": b})
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q -k "frequency or pseudosection"`
Expected: FAIL with `AttributeError: module 'contouring' has no attribute 'parse_frequency'`.

- [ ] **Step 3: Append the implementation**

```python


# ---------------------------------------------------------------------------
# Pseudo-section
# ---------------------------------------------------------------------------

def parse_frequency(label: str) -> float | None:
    """First number in a label, e.g. '4525Hz' -> 4525.0; None if absent."""
    m = re.search(r"\d+(?:\.\d+)?", str(label))
    return float(m.group()) if m else None


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
    distance = lo + step * np.arange(n)

    labels = list(cleaned.keys())
    freqs = [parse_frequency(lb) for lb in labels]
    if all(f is not None for f in freqs):
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `45 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add distance x frequency pseudo-section builder"
```

---

### Task 7: Figures

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_area_map_figure_spline_single_panel():
    res = C.compute_area_map(line_survey(), VALUE_COL, cell_size=2.0)
    fig = C.make_area_map_figure(res, "EC (mS/m)", "test")
    assert len(fig.axes) == 2  # map + colour bar
    plt.close(fig)


def test_area_map_figure_kriging_has_two_panels():
    res = C.compute_area_map(line_survey(), VALUE_COL, method="kriging", cell_size=2.0)
    fig = C.make_area_map_figure(res, "EC (mS/m)", "test")
    assert len(fig.axes) == 4  # two panels + two colour bars
    plt.close(fig)


def test_pseudosection_figure_labels_measured_frequencies():
    s = pd.Series(np.arange(10.0), index=np.arange(10.0))
    ps = C.build_pseudosection({"1000Hz": s, "10000Hz": s + 1})
    fig = C.make_pseudosection_figure(ps, "EC (mS/m)", "t")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1000Hz", "10000Hz"]
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q -k figure`
Expected: FAIL with `AttributeError: module 'contouring' has no attribute 'make_area_map_figure'`.

- [ ] **Step 3: Append the implementation**

```python


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
    panels = 2 if result.variance is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 6), squeeze=False)
    xs, ys = spec.xs, spec.ys

    ax = axes[0, 0]
    cs = ax.contourf(
        xs, ys, np.ma.masked_invalid(result.z),
        levels=contour_levels(result.z, n_levels), cmap="viridis", extend="both",
    )
    fig.colorbar(cs, ax=ax, label=label)
    if show_points:
        ax.plot(result.bx, result.by, ",", color="k", alpha=0.4)
    ax.set_title(title + (" — line-levelled (per-line median)" if result.levelled else ""))

    if result.variance is not None:
        ax2 = axes[0, 1]
        std = np.sqrt(result.variance)
        cs2 = ax2.contourf(
            xs, ys, np.ma.masked_invalid(std),
            levels=contour_levels(std, n_levels), cmap="magma", extend="both",
        )
        fig.colorbar(cs2, ax=ax2, label=f"Kriging std. dev. — {label}")
        ax2.set_title("Kriging standard deviation")

    xlab = "Easting (local m)" if result.origin is not None else "X (m)"
    ylab = "Northing (local m)" if result.origin is not None else "Y (m)"
    for a in axes[0]:
        a.set_aspect("equal")
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
    fig.tight_layout()
    return fig


def make_pseudosection_figure(
    ps: PseudoSection, label: str, title: str, n_levels: int = 20
) -> plt.Figure:
    """Distance x frequency contour with a white line at each measured frequency."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if ps.frequencies is not None:
        rows = ps.frequencies
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("Frequency (Hz)")
    else:
        rows = np.arange(len(ps.labels), dtype=float)
        ax.set_ylabel("Frequency / sheet")
    ax.set_yticks(rows, labels=ps.labels)
    cs = ax.contourf(
        ps.distance, rows, np.ma.masked_invalid(ps.values),
        levels=contour_levels(ps.values, n_levels), cmap="viridis", extend="both",
    )
    for r in rows:
        ax.axhline(r, color="white", linewidth=0.6, alpha=0.8)
    fig.colorbar(cs, ax=ax, label=label)
    ax.set_xlabel("Distance (m)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `48 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add area-map and pseudo-section figures"
```

---

### Task 8: Grid exports (CSV and ESRI ASCII)

**Files:**
- Modify: `contouring.py` (append)
- Modify: `tests/test_contouring.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def test_grid_to_asc_header_and_row_order():
    spec = C.GridSpec(x0=100.0, y0=200.0, cell=2.0, nx=3, ny=2)
    z = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])  # row 0 = south
    lines = C.grid_to_asc(make_result(spec, z)).decode().splitlines()
    assert lines[0] == "ncols 3" and lines[1] == "nrows 2"
    assert lines[2].startswith("xllcorner 100") and lines[3].startswith("yllcorner 200")
    assert lines[4].startswith("cellsize 2") and lines[5] == "NODATA_value -9999"
    assert lines[6].split() == ["4", "-9999", "6"]  # north row first
    assert lines[7].split() == ["1", "2", "3"]


def test_grid_to_csv_omits_blank_and_adds_lonlat():
    spec = C.GridSpec(x0=0.0, y0=0.0, cell=1.0, nx=2, ny=1)
    res = make_result(spec, np.array([[1.0, np.nan]]), origin=(4.0, 50.0))
    df = pd.read_csv(io.BytesIO(C.grid_to_csv(res)))
    assert list(df.columns) == ["x", "y", "value", "lon", "lat"] and len(df) == 1


def test_grid_to_csv_kriging_has_variance_column():
    res = C.compute_area_map(line_survey(), VALUE_COL, method="kriging", cell_size=2.0)
    df = pd.read_csv(io.BytesIO(C.grid_to_csv(res)))
    assert list(df.columns) == ["x", "y", "value", "variance"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_contouring.py -q -k grid_to`
Expected: FAIL with `AttributeError: module 'contouring' has no attribute 'grid_to_asc'`.

- [ ] **Step 3: Append the implementation**

```python


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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_contouring.py -q`
Expected: `51 passed`.

- [ ] **Step 5: Commit**

```bash
git add contouring.py tests/test_contouring.py
git commit -m "Add CSV and ESRI ASCII grid exports"
```

---

### Task 9: Streamlit integration

**Files:**
- Modify: `RIs_v2.py` (imports ~line 26; after `GraphOptions` ~line 76; before the "Per-format rendering dispatchers" banner ~line 1047; `render_legacy_results` ~1050–1078; `render_gem_results` ~1081–1123; sidebar in `main()` ~1166–1170; per-file dispatch ~1558–1561)
- Create: `tests/test_app_contouring.py`

- [ ] **Step 1: Write the failing UI smoke tests**

Create `tests/test_app_contouring.py`:

```python
"""Headless smoke tests for the 2D contouring UI (streamlit.testing.AppTest)."""
from streamlit.testing.v1 import AppTest


def _gem_app(method: str = "spline", lines: int = 6, legacy: bool = False):
    """Script body run by AppTest; all imports must be local."""
    import io

    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows = []
    for i in range(lines):
        ys = np.arange(0.0, 40.0, 0.5)
        xs = np.full_like(ys, 4.0 * i)
        bump = np.exp(-((xs - 10) ** 2 + (ys - 20) ** 2) / 50.0)
        rows.append(pd.DataFrame({
            "Line": i, "X": xs, "Y": ys,
            "EC1525Hz[mS/m]": 20 + 10 * bump,
            "EC9825Hz[mS/m]": 22 + 8 * bump,
            "MSusc1525Hz[1/1000]": 0.5 + 0.2 * bump,
            "MSusc9825Hz[1/1000]": 0.6 + 0.2 * bump,
        }))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    contour = R.ContourSettings(area_map=True, pseudosection=True, method=method)
    if legacy:
        legacy_df = pd.DataFrame({"d": np.arange(0.0, 40.0, 0.5)})
        legacy_df["t1"] = legacy_df["d"] * 0.1
        legacy_df["t2"] = legacy_df["d"] * 0.1 + 0.05
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            legacy_df.to_excel(w, sheet_name="1000", index=False)
            legacy_df.to_excel(w, sheet_name="5000", index=False)
        R.render_legacy_results(buf.getvalue(), "legacy.xlsx", "EC", 0.5, "linear", contour)
    else:
        R.render_gem_results(data, "survey.csv", 0.5, "linear", contour)


def _texts(at):
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


def test_gem_area_map_and_pseudosection_render():
    at = AppTest.from_function(_gem_app, kwargs={"method": "spline"}, default_timeout=120)
    at.run()
    assert not at.exception
    texts = " ".join(_texts(at))
    assert "### Pseudo-section" in texts and "### Area map" in texts
    assert "block medians" in texts


def test_gem_kriging_shows_variogram():
    at = AppTest.from_function(_gem_app, kwargs={"method": "kriging"}, default_timeout=120)
    at.run()
    assert not at.exception
    assert "variogram spherical" in " ".join(_texts(at))


def test_single_line_shows_transect_message():
    at = AppTest.from_function(_gem_app, kwargs={"lines": 1}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("single transect" in i.value for i in at.info)


def test_legacy_file_area_map_message_and_pseudosection():
    at = AppTest.from_function(_gem_app, kwargs={"legacy": True}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("legacy multi-sheet format" in i.value for i in at.info)
    assert "### Pseudo-section" in " ".join(_texts(at))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_app_contouring.py -q`
Expected: 4 failures; each `at.exception` reports `AttributeError: module 'RIs_v2' has no attribute 'ContourSettings'`.

- [ ] **Step 3: Import the module in `RIs_v2.py`**

Find:
```python
from numpy.polynomial.polynomial import polyfit, polyval
```
Replace with:
```python
from numpy.polynomial.polynomial import polyfit, polyval

import contouring as ctr
```

- [ ] **Step 4: Add `ContourSettings` and UI text constants after `GraphOptions`**

Find (end of the `GraphOptions` dataclass):
```python
    x_label: str = ""      # empty → "Distance (m)"
    y_label: str = ""      # empty → MODES[mode]
```
Replace with:
```python
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
```

- [ ] **Step 5: Add the contouring UI functions before the dispatchers banner**

Find:
```python
# ---------------------------------------------------------------------------
# Per-format rendering dispatchers
# ---------------------------------------------------------------------------
```
Replace with:
```python
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


@st.cache_data(show_spinner=False)
def compute_area_map_cached(
    file_bytes: bytes,
    file_name: str,
    value_col: str,
    contour: ContourSettings,
) -> ctr.AreaMapResult:
    """Cached wrapper around contouring.compute_area_map."""
    return ctr.compute_area_map(
        read_raw_table(file_bytes, file_name),
        value_col,
        coord_mode=contour.coord_mode,
        method=contour.method,
        cell_size=contour.cell_size,
        blank_distance=contour.blank_distance,
        level=contour.level_lines,
        smoothing=contour.smoothing,
        variogram_model=contour.variogram_model,
    )


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
             "Auto = 2 × median spacing of the block-reduced points.",
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
    except ctr.ContouringError as exc:
        st.info(str(exc))
        return
    fig = ctr.make_pseudosection_figure(
        ps, ctr.value_label(mode, is_gem),
        f"Pseudo-section [{mode}] — {file_name}", contour.n_levels,
    )
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
    try:
        with st.spinner("Gridding…"):
            result = compute_area_map_cached(
                file_bytes, file_name, gem_value_column(mode, freq), contour
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

    label = ctr.value_label(mode, is_gem)
    fig = ctr.make_area_map_figure(
        result, label, f"{freq} [{mode}] — {ctr.METHODS[result.method]}",
        contour.n_levels,
    )
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
                    f"({label}, {cv['n']:,} held-out points)"
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
```

- [ ] **Step 6: Wire into `render_legacy_results`**

Find:
```python
    distance_step: float,
    interp_kind: str,
) -> None:
    """Process and render a legacy multi-sheet Excel file."""
```
Replace with:
```python
    distance_step: float,
    interp_kind: str,
    contour: ContourSettings | None = None,
) -> None:
    """Process and render a legacy multi-sheet Excel file."""
```

Find:
```python
    _render_mode_section(output_data, scores, mode, file_name, file_key=stem)
```
Replace with:
```python
    _render_mode_section(output_data, scores, mode, file_name, file_key=stem)
    if contour is not None:
        render_contouring(
            output_data, mode, file_name, stem, contour, file_bytes, is_gem=False
        )
```

- [ ] **Step 7: Wire into `render_gem_results`**

Find:
```python
    distance_step: float,
    interp_kind: str,
) -> None:
    """Process and render a GEM instrument file (CSV or XLSX)."""
```
Replace with:
```python
    distance_step: float,
    interp_kind: str,
    contour: ContourSettings | None = None,
) -> None:
    """Process and render a GEM instrument file (CSV or XLSX)."""
```

Find (inside the `for tab, mode_key in zip(tabs, available_modes):` loop):
```python
                file_name,
                file_key=f"{stem}_{mode_key}",
            )
```
Replace with:
```python
                file_name,
                file_key=f"{stem}_{mode_key}",
            )
            if contour is not None:
                render_contouring(
                    output_data[mode_key], mode_key, file_name,
                    f"{stem}_{mode_key}", contour, file_bytes, is_gem=True,
                )
```

- [ ] **Step 8: Add the sidebar and pass settings in `main()`**

Find:
```python
                "If your data spans less than this, all sheets will be skipped."
            )
```
Replace with:
```python
                "If your data spans less than this, all sheets will be skipped."
            )

        contour = render_contouring_sidebar()
```

Find:
```python
        if is_gem:
            render_gem_results(file_bytes, file_name, distance_step, interp_kind)
        else:
            render_legacy_results(file_bytes, file_name, mode, distance_step, interp_kind)
```
Replace with:
```python
        if is_gem:
            render_gem_results(file_bytes, file_name, distance_step, interp_kind, contour)
        else:
            render_legacy_results(
                file_bytes, file_name, mode, distance_step, interp_kind, contour
            )
```

- [ ] **Step 9: Run the UI tests and the full suite**

Run: `python -m pytest -q`
Expected: `55 passed` (51 unit + 4 UI). The UI tests take ~15 s.

- [ ] **Step 10: Commit**

```bash
git add RIs_v2.py tests/test_app_contouring.py
git commit -m "Add 2D contouring controls and views to the app"
```

---

### Task 10: Documentation

**Files:**
- Modify: `README.md` (dependencies line ~25; new section before `## Output Files` ~line 140; references list ~line 252)
- Modify: `RIs_v2.py` (About text: before `#### Supported file formats`; end of the About `#### References` list)

- [ ] **Step 1: README dependencies**

Find:
```markdown
- Dependencies: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`
```
Replace with:
```markdown
- Dependencies: `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`, `pykrige`
- Tests: `pip install -r requirements-dev.txt`, then `python -m pytest`
```

- [ ] **Step 2: README section**

Find:
```markdown
## Output Files
```
Replace with:
```markdown
## 2D Contouring

Two optional views, switched on in the sidebar under **2D contouring**. Both are off by default.

### Pseudo-section (distance × frequency)

Stacks the representative (mean) profiles of all frequencies into one contour plot: distance on x, frequency (log scale) on y. Profiles are aligned on the distance range they all cover — nothing is extrapolated. White lines mark the measured frequencies; colours between them are interpolated.

> The frequency axis is **not depth**. Under the low-induction-number condition the depth sensitivity of a loop–loop sensor is set by coil geometry rather than frequency (McNeill, 1980; Huang, 2005).

### Area map (plan view)

For GEM files covering an area (several lines with X/Y or Lat/Lon coordinates), one frequency at a time:

1. **Coordinates** — `Lat`/`Latitude` + `Lon`/`Long`/`Longitude` columns are used as degrees; otherwise `X`/`Y`. Degrees are projected to local metres (equirectangular about the survey centroid; < 0.1 % error at site scale). Override auto-detection in the sidebar if needed.
2. **Line levelling** (optional) — shifts each line to the survey median to remove line-to-line offsets (striping). It also removes any real gradient across lines, so compare with levelling off. For more advanced levelling see Mauring & Kihle (2006).
3. **Block-median reduction** — one median point per grid cell so densely sampled lines do not dominate. Default cell size = √(bounding-box area / number of readings), about one node per reading.
4. **Gridding**

   | Method | Notes |
   |---|---|
   | Thin-plate spline | Minimum-curvature (biharmonic) surface (Briggs, 1974; Sandwell, 1987). Smoothing 0 interpolates exactly. |
   | Ordinary kriging | Standard for EMI / apparent-conductivity mapping (Lesch et al., 1995; Corwin & Lesch, 2005). The variogram (spherical / exponential / gaussian) is fitted to lags up to half the maximum distance, weighted by pair counts (Oliver & Webster, 2014); kriging uses the 64 nearest points. A second panel maps the kriging standard deviation. Limited to 4,000 block medians. |
   | Linear | Delaunay triangulation; blank outside the data hull. |

5. **Blanking** — grid nodes farther than the blanking distance from any block median are left blank (default 2 × median spacing).
6. **Cross-validation** — 5-fold RMSE / MAE on the block medians, to compare methods on your data (Li & Heap, 2011).

**Downloads:** PNG; grid CSV (`x, y, value` [, `variance`] [, `lon, lat`]); ESRI ASCII grid (`.asc`) for QGIS / ArcGIS. With degree input the `.asc` is in local metres and is not georeferenced — use the CSV `lon`/`lat` columns.

---

## Output Files
```

- [ ] **Step 3: README references**

Find:
```markdown
10. Fritsch, F.N. & Carlson, R.E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, **17**(2), 238–246. https://doi.org/10.1137/0717021
```
Replace with:
```markdown
10. Fritsch, F.N. & Carlson, R.E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, **17**(2), 238–246. https://doi.org/10.1137/0717021

11. Briggs, I.C. (1974). Machine contouring using minimum curvature. *Geophysics*, **39**(1), 39–48. https://doi.org/10.1190/1.1440410

12. Sandwell, D.T. (1987). Biharmonic spline interpolation of GEOS-3 and SEASAT altimeter data. *Geophysical Research Letters*, **14**(2), 139–142. https://doi.org/10.1029/GL014i002p00139

13. Lesch, S.M., Strauss, D.J. & Rhoades, J.D. (1995). Spatial prediction of soil salinity using electromagnetic induction techniques: 1. Statistical prediction models. *Water Resources Research*, **31**(2), 373–386. https://doi.org/10.1029/94WR02179

14. Corwin, D.L. & Lesch, S.M. (2005). Apparent soil electrical conductivity measurements in agriculture. *Computers and Electronics in Agriculture*, **46**, 11–43. https://doi.org/10.1016/j.compag.2004.10.005

15. Oliver, M.A. & Webster, R. (2014). A tutorial guide to geostatistics: Computing and modelling variograms and kriging. *Catena*, **113**, 56–69. https://doi.org/10.1016/j.catena.2013.09.006

16. Li, J. & Heap, A.D. (2011). A review of comparative studies of spatial interpolation methods in environmental sciences: Performance and impact factors. *Ecological Informatics*, **6**, 228–241. https://doi.org/10.1016/j.ecoinf.2010.12.003

17. Mauring, E. & Kihle, O. (2006). Leveling aerogeophysical data using a moving differential median filter. *Geophysics*, **71**(1), L5–L11. https://doi.org/10.1190/1.2163912

18. Huang, H. (2005). Depth of investigation for small broadband electromagnetic sensors. *Geophysics*, **70**(6), G135–G142. https://doi.org/10.1190/1.2122412
```

- [ ] **Step 4: About text in `RIs_v2.py` — new section**

Find (inside the About `st.markdown("""...""")` string):
```markdown
#### Supported file formats
```
Replace with:
```markdown
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
```

- [ ] **Step 5: About text in `RIs_v2.py` — references**

Find (last entry of the About references, inside the same string):
```markdown
- Fritsch, F.N. & Carlson, R.E. (1980). Monotone piecewise
  cubic interpolation. *SIAM J. Numer. Anal.*, **17**(2),
  238–246.
  [doi:10.1137/0717021](https://doi.org/10.1137/0717021)
```
Replace with:
```markdown
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
```

- [ ] **Step 6: Verify the app still imports and tests pass**

Run: `python -m pytest -q`
Expected: `55 passed`.

- [ ] **Step 7: Commit**

```bash
git add README.md RIs_v2.py
git commit -m "Document 2D contouring and add gridding references"
```

---

### Task 11: Manual verification in the running app

**Files:** none committed (sample data goes to a temporary directory).

- [ ] **Step 1: Generate synthetic GEM files**

Run:
```bash
python - <<'PY'
import numpy as np, pandas as pd, tempfile, os
out = tempfile.mkdtemp()
rng = np.random.default_rng(0)
rows = []
for i in range(21):                                   # 21 lines, 2 m apart
    y = np.arange(0.0, 60.0, 0.2)
    x = np.full_like(y, 2.0 * i) + rng.normal(0, 0.05, y.size)
    bump = np.exp(-((x - 20) ** 2 + (y - 30) ** 2) / 80.0)
    row = {"Line": i, "X": x + 500000, "Y": y + 4500000}
    for f, a in [(1525, 6), (5325, 8), (18325, 10), (47025, 12), (93025, 13)]:
        row[f"EC{f}Hz[mS/m]"] = 20 + a * bump + rng.normal(0, 0.3, y.size) + 0.8 * (i % 2)
        row[f"MSusc{f}Hz[1/1000]"] = 0.5 + 0.2 * bump + rng.normal(0, 0.02, y.size)
    rows.append(pd.DataFrame(row))
area = pd.concat(rows, ignore_index=True)
area.to_csv(os.path.join(out, "area_utm.csv"), index=False)
deg = area.copy()
deg["Lon"] = 4.0 + (deg["X"] - 500000) / (6371008.8 * np.cos(np.radians(50))) * 180 / np.pi
deg["Lat"] = 50.0 + (deg["Y"] - 4500000) / 6371008.8 * 180 / np.pi
deg.to_csv(os.path.join(out, "area_lonlat.csv"), index=False)
area[area["Line"] == 0].to_csv(os.path.join(out, "transect.csv"), index=False)
print(out)
PY
```
Expected: prints a temp directory containing `area_utm.csv` (~6,300 rows, alternate lines offset by 0.8 mS/m to create striping), `area_lonlat.csv`, `transect.csv`.

- [ ] **Step 2: Run the app**

Run: `streamlit run RIs_v2.py` and open `http://localhost:8501`.

- [ ] **Step 3: Check each behaviour (upload the files from Step 1)**

| Action | Expected |
|---|---|
| Upload `area_utm.csv`, both toggles off | App looks exactly as before; no new sections. |
| Toggle **Pseudo-section** | Under each EC/MS tab: contour with y ticks 1525Hz…93025Hz on a log axis, white lines at each, caption about "not depth", PNG download. |
| Toggle **Area map**, method spline | Frequency picker; round anomaly centred near X=500020, Y=4500030; visible N–S striping; caption "… readings → … block medians · cell … m". |
| Enable **Line levelling** | Striping largely gone; title ends "— line-levelled (per-line median)". |
| Method **Ordinary kriging** | Second panel (kriging std. dev., low on lines, higher between); caption shows variogram parameters. If it reports the 4,000-point limit, set cell size to 1.0 m. |
| **Cross-validate (5-fold)** | RMSE / MAE in mS/m appear, close to the 0.3 mS/m noise level (prototype: 0.26–0.30 at 18325 Hz). |
| Download **Grid (.asc)**, open in QGIS | Raster sits at the UTM coordinates; NoData outside blanked area. |
| Upload `area_lonlat.csv` with Area map on | Axes labelled "Easting/Northing (local m)"; warning that the .asc is not georeferenced; CSV has lon/lat columns. |
| Upload `transect.csv` with Area map on | Info: "Coordinates look like a single transect — use the pseudo-section." |
| Set cell size 0.01 m | Warning about grid size limit; no crash. |

- [ ] **Step 4: Final full test run**

Run: `python -m pytest -q`
Expected: `55 passed`.

---

## Self-review notes (completed while writing this plan)

- **Spec coverage:** toggles/UI → Task 9; coordinates → 1; levelling, block median, grid, collinearity, size cap → 2; variogram (amended) → 3; spline/kriging/linear, blanking, percentile levels → 4; pipeline + CV → 5; pseudo-section → 6; figures (incl. kriging std panel, measured-frequency lines, caption) → 7, 9; exports → 8; error table → 2, 4, 6, 9 (tests in 9 cover transect + legacy messages); units → 1, 9; docs/references → 10; dependencies → 0.
- **Verified:** all module and UI code in this plan was run in a prototype before writing (51 unit + 4 UI tests passing; PyKrige 1.7.3, SciPy 1.17, Streamlit 1.54, Matplotlib 3.10).
- **Out of scope:** issues #5–#18 (scientific-fidelity fixes to the existing scoring) — separate work.
