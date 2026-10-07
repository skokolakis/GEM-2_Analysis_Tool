"""Georeferencing, difference maps and the map-processing / combined-survey UI."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import contouring as C  # noqa: E402


def _grid_survey(lat0=37.9, lon0=23.7, shift=0.0, lines=8, degrees=True):
    rows = []
    for k in range(lines):
        t = np.linspace(0, 40, 81)
        x, y = 5.0 * k, t
        v = 20 + 10 * np.exp(-((x - 15) ** 2 + (y - 20) ** 2) / 60) + shift
        if degrees:
            lat = lat0 + y / 111_195.0
            lon = lon0 + x / (111_195.0 * np.cos(np.radians(lat0)))
            rows.append(pd.DataFrame({"Line": k, "Lat": lat, "Lon": lon, "EC1525Hz[mS/m]": v}))
        else:
            rows.append(pd.DataFrame({"Line": k, "X": x, "Y": y, "EC1525Hz[mS/m]": v}))
    return pd.concat(rows, ignore_index=True)


def test_utm_projection_sets_epsg_and_csv_lonlat():
    res = C.compute_area_map(_grid_survey(), "EC1525Hz[mS/m]", projection="utm")
    assert res.epsg == 32634 and res.origin is None
    assert 7.3e5 < res.spec.x0 < 7.5e5                     # easting of Athens in zone 34
    df = pd.read_csv(io.BytesIO(C.grid_to_csv(res)))
    assert {"lon", "lat"} <= set(df.columns)
    assert df["lat"].between(37.89, 37.91).all()


def test_metre_coordinates_carry_user_epsg():
    res = C.compute_area_map(_grid_survey(degrees=False), "EC1525Hz[mS/m]", xy_epsg=32634)
    assert res.epsg == 32634


def test_difference_map_recovers_constant_change():
    a = _grid_survey()
    b = _grid_survey(shift=3.0)
    diff = C.compute_difference_map(a, b, "EC1525Hz[mS/m]", projection="local")
    finite = diff.z[np.isfinite(diff.z)]
    assert finite.size > 100
    np.testing.assert_allclose(finite, 3.0, atol=0.05)


def test_symmetric_levels_centre_on_zero():
    res = C.compute_difference_map(_grid_survey(), _grid_survey(shift=1.0), "EC1525Hz[mS/m]")
    fig = C.make_area_map_figure(res, "Δ", "t", 10, cmap="RdBu_r", symmetric=True)
    lo, hi = fig.axes[0].collections[0].get_clim()
    plt.close(fig)
    assert lo == pytest.approx(-hi)


