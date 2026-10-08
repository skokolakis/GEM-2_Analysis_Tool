import io
import numpy as np
import pandas as pd
import pytest

import emphysics as E
import gridtools as G


def _bump(n=60, cell=0.25):
    ax = np.arange(n) * cell
    xx, yy = np.meshgrid(ax, ax)
    return 20.0 + 10.0 * ((np.abs(xx - 7.5) < 2) & (np.abs(yy - 7.5) < 2)), cell


def test_lowpass_keeps_constant_and_blanks():
    z = np.full((10, 10), 5.0)
    z[3, 3] = np.nan
    out = G.lowpass(z, 3)
    assert np.isnan(out[3, 3])
    np.testing.assert_allclose(out[np.isfinite(out)], 5.0)


def test_highpass_removes_plane_in_the_interior():
    yy, xx = np.mgrid[0:40, 0:40].astype(float)
    out = G.highpass(2 * xx + yy, 5)
    np.testing.assert_allclose(out[5:-5, 5:-5], 0.0, atol=1e-9)


def test_despike_replaces_only_spikes():
    rng = np.random.default_rng(0)
    z = 10 + rng.normal(0, 0.1, (30, 30))
    z[10, 10] = 50.0
    out, n = G.despike(z, 3, 6.0)
    assert n == 1
    assert out[10, 10] == pytest.approx(10, abs=0.5)
    assert np.array_equal(np.delete(out.ravel(), 10 * 30 + 10), np.delete(z.ravel(), 10 * 30 + 10))


def test_despike_does_not_use_far_values_at_blank_edges():
    rng = np.random.default_rng(1)
    z = 10 + rng.normal(0, 0.1, (20, 20))
    z[5:15, 12:15] += 20.0                    # a real block against the blank area
    z[:, 15:] = np.nan
    out, _ = G.despike(z, 3, 6.0)
    assert out[5, 14] == pytest.approx(z[5, 14])


def test_summary_stats():
    s = G.summary_stats(np.array([1.0, 2.0, 3.0, np.nan]))
    assert s["n"] == 3 and s["mean"] == 2.0 and s["median"] == 2.0 and s["max"] == 3.0


def test_edge_match_and_merge():
    a = pd.DataFrame({"Line": 1, "X": np.arange(10.0), "Y": 0.0, "v": 10.0})
    b = pd.DataFrame({"Line": 1, "X": np.arange(10.0) + 5, "Y": 0.0, "v": 13.0})
    merged, offsets = G.merge_tables([a, b], "v", tolerance=0.1)
    assert offsets == [0.0, -3.0]
    assert np.allclose(merged["v"], 10.0)
    assert set(merged["Line"]) == {"0:1", "1:1"}
    _, none = G.merge_tables([a, b.assign(X=b["X"] + 100)], "v", tolerance=0.1)
    assert none == [0.0, 0.0]


def test_merge_rejects_mixed_coordinates():
    import contouring as ctr

    a = pd.DataFrame({"Line": 1, "X": np.arange(12.0), "Y": 0.0, "v": 1.0})
    b = pd.DataFrame({"Line": 1, "Lat": 37.0 + 1e-5 * np.arange(12), "Lon": 23.0, "v": 1.0})
    with pytest.raises(ctr.ContouringError, match="different coordinate"):
        G.merge_tables([a, b], "v", 1.0)


def test_edge_match_needs_enough_pairs():
    a = pd.DataFrame({"Line": 1, "X": np.arange(10.0), "Y": 0.0, "v": 10.0})
    b = pd.DataFrame({"Line": 1, "X": np.arange(10.0) + 8, "Y": 0.0, "v": 13.0})   # 2 pairs
    assert G.merge_tables([a, b], "v", tolerance=0.1)[1] == [0.0, 0.0]
    assert G.edge_match_offset(np.zeros((0, 2)), np.zeros(0), np.zeros((3, 2)), np.ones(3), 1.0) == (0.0, 0)


def test_line_axis_angle():
    rows = [pd.DataFrame({"Line": k, "X": np.arange(20.0) * np.cos(np.radians(30)) + 3 * k,
                          "Y": np.arange(20.0) * np.sin(np.radians(30))}) for k in range(4)]
    assert G.line_axis_angle(pd.concat(rows, ignore_index=True)) == pytest.approx(30.0, abs=1e-6)


def test_process_map_filters_and_messages():
    z, cell = _bump()
    out, msgs = G.process_map(z, cell, "low-pass", 3)
    assert out.shape == z.shape and "Low-pass" in msgs[0]
    out, msgs = G.process_map(z, cell, "none", deconvolve_footprint=True)
    assert "Deconvolved" in msgs[0]
    with pytest.raises(ValueError):
        G.process_map(z, cell, "median")


@pytest.mark.parametrize("depth", [0.25, 0.5, 2.0, 4.0])
def test_sensitivity_integrates_to_mcneill_depth_function(depth):
    """Over a horizontal plane at depth d the kernel is proportional to d / (4 d^2 + s^2)^1.5."""
    s = 1.66

    def plane(d):
        ax = np.linspace(-40 * max(d, s), 40 * max(d, s), 1201)
        xx, yy = np.meshgrid(ax + s / 2, ax)
        return G.lin_sensitivity(xx, yy, d, s, 0.0).sum() * (ax[1] - ax[0]) ** 2

    phi = lambda d: d / (4 * d * d + s * s) ** 1.5
    assert plane(depth) / plane(1.0) == pytest.approx(phi(depth) / phi(1.0), rel=2e-3)


def test_footprint_sums_to_one_and_follows_angle():
    k0 = G.footprint(0.2, angle_deg=0)
    k90 = G.footprint(0.2, angle_deg=90)
    assert k0.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(k90, np.rot90(k0, -1), atol=1e-12)
    # symmetric across the coil axis; negative lobes between the coils are physical
    np.testing.assert_allclose(k0, k0[::-1, :], atol=1e-15)
    assert k0.min() < 0


def test_deconvolution_sharpens_a_blurred_anomaly():
    truth, cell = _bump()
    k = G.footprint(cell, E.Sensor(height=0.5))
    blurred = G.convolve(truth, k)
    sharp = G.deconvolve(blurred, k, regularisation=1e-4)
    err = lambda m: np.sqrt(np.mean((m - truth) ** 2))
    assert err(sharp) < 0.6 * err(blurred)


def test_deconvolution_keeps_blanks():
    truth, cell = _bump()
    truth[:5, :5] = np.nan
    out = G.deconvolve(truth, G.footprint(cell), 1e-2)
    assert np.isnan(out[:5, :5]).all() and np.isfinite(out[10:, 10:]).all()


@pytest.mark.parametrize("cell", [0.5, 2.0, 5.0])
def test_broad_anomaly_passes_deconvolution_at_any_cell_size(cell):
    # The footprint is averaged over each cell, so coarse grids do not alias it.
    x = np.arange(-150.0, 150.0 + cell / 2, cell)
    gx, gy = np.meshgrid(x, x)
    broad = 10.0 * np.exp(-(gx ** 2 + gy ** 2) / (2 * 60.0 ** 2))
    out = G.deconvolve(broad, G.footprint(cell), 1e-2)
    mid = len(x) // 2
    assert out[mid, mid] == pytest.approx(10.0, rel=0.05)


def test_deconvolution_keeps_a_regional_trend():
    rows, cols = np.indices((80, 120))
    plane = 20.0 + 40.0 * cols / 119 + 5.0 * rows / 79
    out = G.deconvolve(plane, G.footprint(0.5), 1e-4)
    np.testing.assert_allclose(out, plane, atol=1e-6)


def test_deconvolution_inverts_convolve_for_an_asymmetric_kernel():
    m = np.random.default_rng(0).normal(size=(64, 64))
    k = np.zeros((5, 5))
    k[2, 2], k[2, 4] = 0.7, 0.3
    blurred = G.convolve(m, k)
    back = G.deconvolve(blurred, k, 1e-6)
    inner = (slice(8, -8), slice(8, -8))
    err = lambda a: np.sqrt(np.mean((a[inner] - m[inner]) ** 2))
    assert err(back) < 0.1 * err(blurred)


def test_utm_zone():
    assert G.utm_epsg(23.7, 37.9) == 32634          # Athens
    assert G.utm_epsg(-70.6, -33.4) == 32719        # Santiago


def test_projection_and_prj():
    x, y = G.lonlat_to_epsg([21.0], [0.0], 32634)
    assert x[0] == pytest.approx(500000.0, abs=1e-3)
    assert b"UTM_Zone_34N" in G.prj_wkt(32634)


def test_geotiff_round_trip_values_and_tags():
    from PIL import Image

    z = np.arange(12, dtype=float).reshape(3, 4)
    z[0, 0] = np.nan
    data = G.geotiff_bytes(z, 500000.0, 4000000.0, 2.0, 32634)
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    assert arr.shape == (3, 4)
    assert arr[0, 1] == 9.0                     # first stored row = northernmost (z row 2)
    assert arr[2, 0] == G.GEOTIFF_NODATA
    tags = img.tag_v2
    assert tuple(tags[33550])[:2] == (2.0, 2.0)
    assert tuple(tags[33922])[3:5] == (500000.0, 4000006.0)
    assert 32634 in tuple(tags[34735])


def test_epsg_problems():
    assert G.epsg_problem(32634) is None
    assert "not a projected" in G.epsg_problem(4326)
    assert "not a known" in G.epsg_problem(12345)
