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


