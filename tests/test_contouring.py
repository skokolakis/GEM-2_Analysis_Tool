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
    assert vf.parameters == [vf.psill, vf.range, vf.nugget]


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
