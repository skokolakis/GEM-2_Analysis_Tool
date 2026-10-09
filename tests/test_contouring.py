import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

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
    n = C.MIN_POINTS
    df = pd.DataFrame({"X": np.linspace(0.0, 80.0, n), "Y": np.linspace(0.0, 60.0, n),
                       "Latitude": np.linspace(50.0, 50.001, n),
                       "LON": np.linspace(4.0, 4.001, n)})
    assert C.find_coordinate_columns(df) == ("LON", "Latitude", True)


def test_find_coordinates_blank_lat_lon_falls_back_to_xy():
    df = pd.DataFrame({"X": [0.0, 80.0], "Y": [0.0, 60.0],
                       "Lat": [np.nan, np.nan], "Lon": [np.nan, np.nan]})
    assert C.find_coordinate_columns(df) == ("X", "Y", False)


def test_find_coordinates_missing_raises():
    with pytest.raises(C.ContouringError, match="No coordinate"):
        C.find_coordinate_columns(pd.DataFrame({"Line": [1], "Dist": [0.0]}))


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
    assert vf.pykrige_parameters == {"psill": vf.psill, "range": vf.range,
                                     "nugget": vf.nugget}


def test_fit_variogram_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown variogram model"):
        C.fit_variogram(np.arange(20.0), np.arange(20.0) ** 0.5, np.arange(20.0), "cubic")


def _lines_df(offsets, length=1000.0):
    rows = []
    for i, off in enumerate(offsets):
        y = np.arange(0.0, length, 1.0)
        rows.append(pd.DataFrame({"Line": i, "X": np.full_like(y, off), "Y": y}))
    return pd.concat(rows, ignore_index=True)


def test_check_geometry_narrow_corridor_survey_passes():
    df = _lines_df([0.0, 5.0, 10.0, 15.0, 20.0])  # 20 m x 1000 m corridor
    C.check_geometry(df["X"].to_numpy(), df["Y"].to_numpy(), df["Line"].to_numpy())


def test_check_geometry_repeat_passes_of_one_transect_raise():
    df = _lines_df([0.0, 0.3, -0.4])  # three passes, GPS offsets < 1 m
    with pytest.raises(C.ContouringError, match="single transect"):
        C.check_geometry(df["X"].to_numpy(), df["Y"].to_numpy(), df["Line"].to_numpy())


def test_check_geometry_corridor_without_line_labels_raises():
    df = _lines_df([0.0, 5.0, 10.0, 15.0, 20.0])
    with pytest.raises(C.ContouringError, match="single transect"):
        C.check_geometry(df["X"].to_numpy(), df["Y"].to_numpy())


def test_fit_variogram_too_few_points():
    with pytest.raises(C.KrigingError, match="Not enough points"):
        C.fit_variogram(np.array([0.0]), np.array([0.0]), np.array([1.0]))


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
                  np.array([0.0]), np.array([0.0]), "bogus")


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


# ---------------------------------------------------------------------------
# Review fixes: blanking, NaN line labels, kriging cell growth, error wrapping
# ---------------------------------------------------------------------------

def test_default_blanking_keeps_gaps_between_wide_lines():
    df = line_survey(n_lines=6, spacing=40.0, along=1.0, length=200.0)
    res = C.compute_area_map(df, VALUE_COL)
    xx, yy = np.meshgrid(res.spec.xs, res.spec.ys)
    inside = (xx <= 200.0) & (yy <= 200.0)  # nodes within the surveyed block
    assert np.isfinite(res.z[inside]).mean() > 0.98


def test_explicit_blank_distance_is_respected():
    df = line_survey(n_lines=6, spacing=40.0, along=1.0, length=200.0)
    res = C.compute_area_map(df, VALUE_COL, blank_distance=5.0)
    assert res.blank_distance == 5.0
    assert np.isnan(res.z).mean() > 0.3  # strips between 40 m lines are blanked


def test_levelling_with_blank_line_labels_keeps_map():
    df = line_survey()
    df.loc[df.index[:20], "Line"] = np.nan
    res = C.compute_area_map(df, VALUE_COL, level=True, cell_size=1.0)
    assert np.isfinite(res.bv).all()
    assert np.isfinite(res.z).any()


def test_kriging_auto_cell_grows_to_respect_point_cap():
    rng = np.random.default_rng(0)
    n = 12000
    df = pd.DataFrame({"Line": rng.integers(0, 50, n),
                       "X": rng.uniform(0, 100, n), "Y": rng.uniform(0, 100, n)})
    df[VALUE_COL] = bump(df["X"] / 2, df["Y"] / 2)
    res = C.compute_area_map(df, VALUE_COL, method="kriging")
    assert len(res.bv) <= C.KRIGE_MAX_POINTS
    assert res.spec.cell > C.auto_cell_size(df["X"].to_numpy(), df["Y"].to_numpy())


def test_predict_wraps_degenerate_geometry_errors():
    px = np.arange(10.0)
    py = 2.0 * px  # perfectly collinear
    with pytest.raises(C.ContouringError, match="Gridding failed"):
        C.predict(px, py, np.ones(10), np.array([1.0]), np.array([5.0]), "spline")
    with pytest.raises(C.ContouringError, match="Gridding failed"):
        C.predict(px, py, np.ones(10), np.array([1.0]), np.array([5.0]), "linear")


# ---------------------------------------------------------------------------
# Pseudo-section
# ---------------------------------------------------------------------------

def test_parse_frequency():
    assert C.parse_frequency("4525Hz") == 4525.0
    assert C.parse_frequency("EC 93.5 kHz") == 93500.0
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
    assert "log" in ax.get_ylabel()
    np.testing.assert_allclose(ax.get_yticks(), [3.0, 4.0])  # log10 of 1000, 10000
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1000Hz", "10000Hz"]
    plt.close(fig)


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


# ---------------------------------------------------------------------------
# Review fixes: pseudo-section edge cases
# ---------------------------------------------------------------------------

def test_pseudosection_float_endpoint_has_no_blank_column():
    idx = np.linspace(0.0, 3.0, 31)  # step 0.1 with float noise
    a = pd.Series(np.ones(31), index=idx)
    b = pd.Series(np.ones(4), index=[0.0, 0.1, 0.2, 0.30000000000000004])
    c = pd.Series(np.ones(4), index=[0.0, 0.1, 0.2, 0.3])
    ps = C.build_pseudosection({"1Hz": a, "2Hz": b, "3Hz": c})
    assert np.isfinite(ps.values).all()
    assert ps.distance[-1] <= 0.3


def test_pseudosection_overlap_below_one_step_raises():
    a = pd.Series([1.0, 2.0, 3.0], index=[0.0, 5.0, 10.0])
    b = pd.Series([1.0, 2.0, 3.0], index=[9.5, 14.5, 19.5])
    with pytest.raises(C.ContouringError, match="less than one distance step"):
        C.build_pseudosection({"1Hz": a, "2Hz": b})


def test_pseudosection_duplicate_distances_tolerated():
    a = pd.Series([1.0, 1.0, 2.0, 3.0], index=[0.0, 0.0, 1.0, 2.0])
    b = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
    ps = C.build_pseudosection({"1Hz": a, "2Hz": b})
    assert np.isfinite(ps.values).all()


def test_pseudosection_zero_or_duplicate_frequency_uses_categorical_axis():
    s = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
    assert C.build_pseudosection({"0Hz": s, "10Hz": s}).frequencies is None
    assert C.build_pseudosection({"EC 10Hz": s, "MS 10Hz": s}).frequencies is None


# ---------------------------------------------------------------------------
# Review fixes: near-flat contour levels, frequency label parsing
# ---------------------------------------------------------------------------

def test_contour_levels_increasing_for_round_off_flat_field():
    lv = C.contour_levels(0.5 + np.array([0.0, 1e-16, 2e-16, 1e-16]), 10)
    assert np.all(np.diff(lv) > 0)


def test_parse_frequency_bare_numbers_and_hz_only():
    assert C.parse_frequency("9000") == 9000.0
    assert C.parse_frequency("Sheet1") is None
    assert C.parse_frequency("Line 3") is None


# ---------------------------------------------------------------------------
# Final fix round: variogram on noisy data, kHz labels, figures, GPS no-fix rows
# ---------------------------------------------------------------------------

def _noisy_survey():
    df = line_survey()
    df[VALUE_COL] = df[VALUE_COL] + np.random.default_rng(1).normal(0, 0.3, len(df))
    return df


def test_gaussian_kriging_is_stable_on_noisy_data():
    # Before the fix the fitted nugget was ~0 and cross-validation RMSE was ~1600.
    res = C.compute_area_map(_noisy_survey(), VALUE_COL, method="kriging",
                             variogram_model="gaussian", cell_size=1.0)
    assert res.variogram.nugget > 0
    cv = C.cross_validate(res.bx, res.by, res.bv, "kriging", variogram=res.variogram)
    assert cv["rmse"] < 0.5


def test_gaussian_nugget_floor_on_noise_free_data():
    res = C.compute_area_map(line_survey(), VALUE_COL, method="kriging",
                             variogram_model="gaussian", cell_size=1.0)
    vf = res.variogram
    assert vf.nugget >= C.GAUSSIAN_NUGGET_FLOOR * (vf.psill + vf.nugget) * 0.999
    assert bump_rmse(res) < 0.1


def test_parse_frequency_converts_khz_and_orders_rows():
    assert C.parse_frequency("1.5kHz") == 1500.0
    s = pd.Series([1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0])
    ps = C.build_pseudosection({lb: s for lb in ["5kHz", "475Hz", "1.5kHz"]})
    assert ps.labels == ["475Hz", "1.5kHz", "5kHz"]
    np.testing.assert_allclose(ps.frequencies, [475.0, 1500.0, 5000.0])


def test_failed_area_map_figure_does_not_leak_a_figure():
    spec = C.GridSpec(x0=0.0, y0=0.0, cell=1.0, nx=2, ny=2)
    res = make_result(spec, np.full((2, 2), np.nan))
    before = len(plt.get_fignums())
    with pytest.raises(C.ContouringError):
        C.make_area_map_figure(res, "EC (mS/m)", "t")
    assert len(plt.get_fignums()) == before


def test_area_map_axes_show_full_utm_coordinates():
    df = line_survey()
    df["X"] += 500000.0
    df["Y"] += 4500000.0
    res = C.compute_area_map(df, VALUE_COL, cell_size=2.0)
    fig = C.make_area_map_figure(res, "EC (mS/m)", "t")
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.xaxis.get_offset_text().get_text() == ""
    assert ax.yaxis.get_offset_text().get_text() == ""
    plt.close(fig)


def test_gps_no_fix_zero_rows_are_dropped():
    df = line_survey()
    lon0, lat0 = 4.0, 50.0
    df["Lon"] = lon0 + np.degrees(df["X"] / (C.EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    df["Lat"] = lat0 + np.degrees(df["Y"] / C.EARTH_RADIUS_M)
    df.loc[df.index[:30], ["Lon", "Lat"]] = 0.0
    res = C.compute_area_map(df, VALUE_COL, cell_size=1.0)
    assert res.n_raw == len(df) - 30
    assert res.spec.nx == pytest.approx(51, abs=2)


def test_too_few_gps_fixes_fall_back_to_xy():
    df = line_survey()
    df["Lat"] = np.nan
    df["Lon"] = np.nan
    df.loc[df.index[:3], ["Lat", "Lon"]] = [50.0, 4.0]
    assert C.find_coordinate_columns(df) == ("X", "Y", False)


# ---------------------------------------------------------------------------
# PyKrige parameter semantics (partial sill vs full sill)
# ---------------------------------------------------------------------------

def test_pykrige_receives_partial_sill_range_and_nugget():
    from pykrige.ok import OrdinaryKriging
    vf = C.VariogramFit("spherical", 2.0, 4.0, 0.5, (), ())
    p = np.random.default_rng(0).uniform(0, 10, (30, 2))
    ok = OrdinaryKriging(p[:, 0], p[:, 1], np.zeros(30), variogram_model=vf.model,
                         variogram_parameters=vf.pykrige_parameters)
    np.testing.assert_allclose(ok.variogram_model_parameters, [2.0, 4.0, 0.5])


@pytest.mark.parametrize("model", C.VARIOGRAM_MODELS)
def test_kriging_pure_noise_stays_within_noise(model):
    df = line_survey()
    df[VALUE_COL] = np.random.default_rng(3).normal(0.0, 1.0, len(df))
    res = C.compute_area_map(df, VALUE_COL, method="kriging",
                             variogram_model=model, cell_size=1.0)
    cv = C.cross_validate(res.bx, res.by, res.bv, "kriging", variogram=res.variogram)
    assert cv["rmse"] < 1.5          # block medians of 2 readings: sigma ~0.7
    assert np.nanmax(np.abs(res.z)) < 5.0


# ---------------------------------------------------------------------------
# Surfer gridding methods
# ---------------------------------------------------------------------------

BUMP_RMSE = {          # cell 1 m, lines 5 m apart; smoothing / non-exact methods get more room
    "spline": 0.05, "kriging": 0.1, "linear": 0.2, "min_curvature": 0.05, "idw": 0.5,
    "rbf": 0.1, "natural_neighbor": 0.2, "nearest": 0.6, "shepard": 0.1,
    "local_polynomial": 0.5, "moving_average": 0.5,
}


@pytest.mark.parametrize("method", list(BUMP_RMSE))
def test_every_method_maps_the_bump(method):
    res = C.compute_area_map(line_survey(), VALUE_COL, method=method, cell_size=1.0)
    assert bump_rmse(res) < BUMP_RMSE[method]
    assert res.options == C.method_options(method)
    cv = C.cross_validate(res.bx, res.by, res.bv, method, variogram=res.variogram,
                          options=res.options, spec=res.spec)
    assert cv["n"] > 0 and np.isfinite(cv["rmse"])


@pytest.mark.parametrize("method,options,tol", [
    ("min_curvature", {"tension": 0.0}, 1e-6),
    ("local_polynomial", {"order": 1}, 1e-6),
    ("polynomial", {"order": 1}, 1e-6),
    ("natural_neighbor", {}, 0.2),
])
def test_methods_reproduce_a_plane_inside_the_survey(method, options, tol):
    df = line_survey()
    df[VALUE_COL] = 2.0 * df["X"] + 3.0 * df["Y"]
    res = C.compute_area_map(df, VALUE_COL, method=method, cell_size=1.0, options=options)
    xx, yy = np.meshgrid(res.spec.xs, res.spec.ys)
    inner = np.isfinite(res.z) & (xx > 3) & (xx < 47) & (yy > 3) & (yy < 47)
    assert np.max(np.abs(res.z - (2.0 * xx + 3.0 * yy))[inner]) < tol


@pytest.mark.parametrize("tension", [0.0, 0.5, 0.9])
def test_min_curvature_honours_the_readings_at_any_tension(tension):
    res = C.compute_area_map(line_survey(), VALUE_COL, method="min_curvature", cell_size=1.0,
                             options={"tension": tension})
    at_points = RegularGridInterpolator((res.spec.ys, res.spec.xs), res.z)(
        np.column_stack([res.by, res.bx]).clip([res.spec.ys[0], res.spec.xs[0]],
                                               [res.spec.ys[-1], res.spec.xs[-1]]))
    inside = (res.bx > res.spec.xs[0]) & (res.bx < res.spec.xs[-1])
    assert np.max(np.abs(at_points - res.bv)[inside]) < 0.01


def test_polynomial_orders_fit_their_own_surfaces():
    rng = np.random.default_rng(1)
    px, py = rng.uniform(0, 10, 200), rng.uniform(0, 10, 200)
    f = 1 + px - 2 * py + 0.3 * px * py - 0.1 * py ** 2 + 0.02 * px ** 3
    q = np.array([2.0, 7.5]), np.array([3.0, 6.0])
    exact = 1 + q[0] - 2 * q[1] + 0.3 * q[0] * q[1] - 0.1 * q[1] ** 2 + 0.02 * q[0] ** 3
    est, _ = C.predict(px, py, f, *q, "polynomial", options={"order": 3})
    assert np.allclose(est, exact)
    est, _ = C.predict(px, py, f, *q, "polynomial", options={"order": 1})
    assert not np.allclose(est, exact)
    with pytest.raises(C.ContouringError, match="order"):
        C.predict(px, py, f, *q, "polynomial", options={"order": 4})


def test_exact_methods_pass_through_readings():
    rng = np.random.default_rng(2)
    px, py, pv = rng.uniform(0, 10, 50), rng.uniform(0, 10, 50), rng.normal(0, 1, 50)
    for method in ("idw", "nearest", "shepard", "rbf"):
        est, _ = C.predict(px, py, pv, px[:5], py[:5], method)
        assert np.allclose(est, pv[:5], atol=1e-6), method


def test_idw_power_and_smoothing():
    px, py, pv = np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 10.0])
    q = np.array([2.5]), np.array([0.0])
    low, _ = C.predict(px, py, pv, *q, "idw", options={"power": 1.0})
    high, _ = C.predict(px, py, pv, *q, "idw", options={"power": 4.0})
    assert high[0] < low[0] < 5.0                     # higher power: the near reading dominates
    on, _ = C.predict(px, py, pv, np.array([0.0]), np.array([0.0]), "idw",
                      options={"delta": 5.0})
    assert on[0] > 0.0                                # smoothing no longer honours the reading
    with pytest.raises(C.ContouringError, match="power"):
        C.predict(px, py, pv, *q, "idw", options={"power": 0.0})


def test_moving_average_and_data_metrics():
    px = np.array([0.0, 1.0, 0.0, 1.0, 10.0])
    py = np.array([0.0, 0.0, 1.0, 1.0, 10.0])
    pv = np.array([1.0, 2.0, 3.0, 6.0, 100.0])
    q = np.array([0.5, 20.0]), np.array([0.5, 20.0])
    mean, _ = C.predict(px, py, pv, *q, "moving_average", options={"radius": 2.0})
    assert mean[0] == pytest.approx(3.0) and np.isnan(mean[1])

    def metric(stat):
        return C.predict(px, py, pv, *q, "metrics", options={"statistic": stat, "radius": 2.0})[0]

    assert metric("count").tolist() == [4.0, 0.0]
    assert metric("density")[0] == pytest.approx(4 / (math.pi * 4.0))
    assert metric("median")[0] == pytest.approx(2.5)
    assert metric("minimum")[0] == 1.0 and metric("maximum")[0] == 6.0
    assert metric("range")[0] == 5.0
    assert metric("std")[0] == pytest.approx(np.std([1, 2, 3, 6], ddof=1))


def test_rbf_kernels_and_titles():
    for kernel in C.RBF_KERNELS:
        res = C.compute_area_map(line_survey(), VALUE_COL, method="rbf", cell_size=2.0,
                                 options={"kernel": kernel})
        assert bump_rmse(res) < 0.5, kernel           # inverse multiquadric is the flattest
        assert C.RBF_KERNELS[kernel].lower() in C.method_title(res)


def test_metrics_label_title_and_no_cross_validation():
    res = C.compute_area_map(line_survey(), VALUE_COL, method="metrics", cell_size=2.0,
                             options={"statistic": "count"})
    assert C.result_label(res, "EC (mS/m)") == "Readings within the search radius"
    assert C.method_title(res) == "Data metrics: number of readings"
    res = C.compute_area_map(line_survey(), VALUE_COL, method="metrics", cell_size=2.0,
                             options={"statistic": "median"})
    assert C.result_label(res, "EC (mS/m)") == "Median of EC (mS/m)"
    with pytest.raises(C.ContouringError, match="summaries"):
        C.cross_validate(res.bx, res.by, res.bv, "metrics", options=res.options)


def test_grid_methods_need_the_grid_and_limit_its_size():
    with pytest.raises(ValueError, match="predict_grid"):
        C.predict(np.arange(3.0), np.arange(3.0), np.arange(3.0),
                  np.array([0.0]), np.array([0.0]), "min_curvature")
    spec = C.GridSpec(x0=0.0, y0=0.0, cell=1.0, nx=1000, ny=1000)
    with pytest.raises(C.GridTooLargeError, match="Minimum curvature"):
        C.predict_grid(np.arange(3.0), np.arange(3.0), np.arange(3.0), spec, "min_curvature")
    with pytest.raises(C.ContouringError, match="tension"):
        C.compute_area_map(line_survey(), VALUE_COL, method="min_curvature", cell_size=2.0,
                           options={"tension": 1.0})


def test_method_options_fill_defaults_and_ignore_other_keys():
    assert C.method_options("idw", {"power": 3.0, "kernel": "cubic"}) == {"power": 3.0, "delta": 0.0}
    assert C.method_options("spline", {"power": 3.0}) == {}


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_colour_ranges():
    v = np.arange(101.0)
    assert C.colour_levels(v, C.ColourStyle(n_levels=4)).tolist() == [2, 26, 50, 74, 98]
    lv = C.colour_levels(v, C.ColourStyle(n_levels=4, range_mode="minmax"))
    assert lv[0] == 0 and lv[-1] == 100
    lv = C.colour_levels(v, C.ColourStyle(n_levels=4, percentiles=(10, 90)))
    assert lv[0] == 10 and lv[-1] == 90
    lv = C.colour_levels(v, C.ColourStyle(n_levels=4, range_mode="fixed", vmin=20, vmax=40))
    assert lv.tolist() == [20, 25, 30, 35, 40]
    with pytest.raises(C.ContouringError, match="maximum"):
        C.colour_levels(v, C.ColourStyle(range_mode="fixed", vmin=40, vmax=20))
    with pytest.raises(C.ContouringError, match="percentiles"):
        C.colour_levels(v, C.ColourStyle(percentiles=(90, 10)))


def test_colour_scales():
    v = np.exp(np.linspace(0, 5, 500))                   # skewed, like EC
    lv = C.colour_levels(v, C.ColourStyle(n_levels=5, scale="log", range_mode="minmax"))
    assert np.allclose(np.diff(np.log(lv)), np.log(lv[1] / lv[0]))
    eq = C.colour_levels(v, C.ColourStyle(n_levels=5, scale="equalised", range_mode="minmax"))
    counts = np.histogram(v, eq)[0]
    assert counts.max() - counts.min() <= 2              # equal share of values per colour
    with pytest.raises(C.ContouringError, match="positive"):
        C.colour_levels(v - 10, C.ColourStyle(scale="log"))
    flat = C.colour_levels(np.full(50, 3.0), C.ColourStyle(scale="equalised"))
    assert np.all(np.diff(flat) > 0)


def test_colour_maps_including_surfer_rainbow():
    for name in C.COLOUR_MAPS:
        assert C.colour_map(name).N > 0
    rainbow = C.colour_map("surfer_rainbow")
    assert rainbow(0.0)[2] > rainbow(0.0)[1] and rainbow(1.0)[0] == 1.0   # purple-blue to red
    assert C.colour_map("surfer_rainbow", reverse=True)(0.0) == rainbow(1.0)


@pytest.mark.parametrize("display", list(C.MAP_DISPLAYS))
@pytest.mark.parametrize("scale", list(C.COLOUR_SCALES))
def test_area_map_figure_displays_and_scales(display, scale):
    res = C.compute_area_map(line_survey(), VALUE_COL, method="min_curvature", cell_size=2.0)
    style = C.ColourStyle(cmap="surfer_rainbow", display=display, scale=scale,
                          range_mode="minmax", contour_lines=True, n_levels=12)
    fig = C.make_area_map_figure(res, "EC (mS/m)", "t", style=style, show_points=False)
    ax = fig.axes[0]
    assert ax.get_xlim() == (res.spec.x0, res.spec.x0 + res.spec.nx * res.spec.cell)
    assert len(fig.axes) == 2
    fig.canvas.draw()                                     # norms and colour bar render
    plt.close(fig)


def test_pseudosection_figure_takes_a_colour_style():
    s = pd.Series(np.arange(1.0, 11.0), index=np.arange(10.0))
    ps = C.build_pseudosection({"1000Hz": s, "10000Hz": s + 1})
    fig = C.make_pseudosection_figure(ps, "EC", "t", style=C.ColourStyle(
        display="image", scale="log", cmap="turbo", reverse=True))
    fig.canvas.draw()
    plt.close(fig)
