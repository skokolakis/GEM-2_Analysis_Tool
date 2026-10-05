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


def test_find_coordinates_blank_lat_lon_falls_back_to_xy():
    df = pd.DataFrame({"X": [0.0, 80.0], "Y": [0.0, 60.0],
                       "Lat": [np.nan, np.nan], "Lon": [np.nan, np.nan]})
    assert C.find_coordinate_columns(df) == ("X", "Y", False)


def test_find_coordinates_missing_raises():
    with pytest.raises(C.ContouringError, match="No coordinate"):
        C.find_coordinate_columns(pd.DataFrame({"Line": [1], "Dist": [0.0]}))
