import io
import time

import numpy as np
import pandas as pd
import pytest

import emphysics as E
import inversion as V

FREQS = [475.0, 1525.0, 5325.0, 18325.0, 63025.0]


def test_layer_grid_sums_to_max_depth_and_grows():
    g = V.make_layer_grid(max_depth=6.0, n_layers=15, first=0.1)
    assert g.n_layers == 15
    assert g.thickness.sum() == pytest.approx(6.0, rel=1e-6)
    assert g.thickness[0] == pytest.approx(0.1)
    assert np.all(np.diff(g.thickness) > 0)


def test_halfspace_data_recovers_halfspace():
    g = V.make_layer_grid(6.0, 12)
    d = E.forward_ppm(FREQS, [0.03]).imag[None, :]
    res = V.invert(FREQS, d, V.data_errors(d, None, 0.01, 0.1), g, lateral_weight=0)
    assert res.chi2 <= 1.0
    top = res.log_sigma[0, :6]                      # well-resolved shallow layers
    np.testing.assert_allclose(10 ** top, 0.03, rtol=0.15)


def test_two_layer_contrast_is_recovered_in_shape():
    g = V.make_layer_grid(6.0, 15)
    sigma = np.where(g.depth_top < 1.0, 0.005, 0.1)
    d = E.forward_ppm(FREQS, sigma, thickness=g.thickness).imag[None, :]
    # small errors: the smoothest model that fits them is sharp enough to show the contrast
    res = V.invert(FREQS, d, V.data_errors(d, None, 0.003, 0.05), g, lateral_weight=0)
    shallow = res.log_sigma[0, g.depth_top < 0.5].mean()
    deep = res.log_sigma[0, (g.depth_top > 1.5) & (g.depth_top < 3)].mean()
    assert deep - shallow > 0.5                     # at least a 3x increase with depth
    assert res.chi2 < 2.0


def test_lateral_constraint_smooths_noisy_neighbours():
    g = V.make_layer_grid(6.0, 10)
    rng = np.random.default_rng(0)
    clean = E.forward_ppm(FREQS, [0.02]).imag
    d = clean[None, :] * (1 + 0.05 * rng.standard_normal((8, len(FREQS))))
    err = V.data_errors(d, None, 0.05, 0.5)
    free = V.invert(FREQS, d, err, g, lateral_weight=0)
    lci = V.invert(FREQS, d, err, g, lateral_weight=5.0)
    spread = lambda r: np.std(r.log_sigma[:, 0])
    assert spread(lci) < spread(free)


def test_inversion_runtime_is_interactive():
    g = V.make_layer_grid(6.0, 15)
    d = np.repeat(E.forward_ppm(FREQS, [0.02]).imag[None, :], 40, axis=0)
    t = time.perf_counter()
    V.invert(FREQS, d * 1.02, V.data_errors(d, None), g)
    assert time.perf_counter() - t < 30


def test_missing_data_are_ignored():
    g = V.make_layer_grid(6.0, 10)
    d = E.forward_ppm(FREQS, [0.02]).imag[None, :].copy()
    d[0, 2] = np.nan
    res = V.invert(FREQS, d, V.data_errors(np.nan_to_num(d), None, 0.01, 0.1), g, lateral_weight=0)
    assert np.isfinite(res.chi2)


def test_errors_take_the_larger_of_noise_and_model():
    d = np.array([[100.0, 1000.0]])
    err = V.data_errors(d, np.array([10.0, 1.0]), relative=0.03, floor=1.0)
    np.testing.assert_allclose(err, [[10.0, 31.0]])


def test_ec_to_quadrature_round_trips_halfspace_conversion():
    q = E.forward_ppm(FREQS, [0.04]).imag
    back = V.ec_to_quadrature(FREQS, np.full((1, 5), 40.0))
    np.testing.assert_allclose(back[0], q, rtol=1e-9)


def test_emagpy_csv_columns():
    csv = V.emagpy_csv([0, 1], [0, 0], [1525.0, 5325.0], [[10, 11], [12, 13]],
                       errors_ms_m=[0.5, 0.7])
    df = pd.read_csv(io.BytesIO(csv))
    assert list(df.columns) == [
        "x", "y", "elevation", "HCP1.66f1525h1", "HCP1.66f5325h1",
        "HCP1.66f1525h1_err", "HCP1.66f5325h1_err",
    ]
    assert df["HCP1.66f5325h1_err"].tolist() == [0.7, 0.7]


def test_layer_grid_reaches_deep_targets_and_rejects_impossible_ones():
    assert V.make_layer_grid(50.0, 5, 0.01).thickness.sum() == pytest.approx(50.0, rel=1e-6)
    with pytest.raises(ValueError, match="deeper than"):
        V.make_layer_grid(0.5, 15, 0.1)


def test_missing_noise_and_errors_at_gaps_do_not_block():
    g = V.make_layer_grid(6.0, 10)
    d = E.forward_ppm(FREQS, [0.02]).imag[None, :].copy()
    err = V.data_errors(d, np.array([1.0, np.nan, 1.0, np.nan, 1.0]), 0.01, 0.1)
    assert np.isfinite(err).all()
    d[0, 2] = np.nan
    err[0, 2] = 0.0                                  # no datum there, so no error is needed
    assert np.isfinite(V.invert(FREQS, d, err, g, lateral_weight=0).chi2)
    with pytest.raises(ValueError, match="No finite data"):
        V.invert(FREQS, np.full((1, 5), np.nan), np.ones((1, 5)), g)


def test_regularisation_does_not_depend_on_its_start():
    g = V.make_layer_grid(6.0, 15)
    sigma = np.where(g.depth_top < 1.0, 0.005, 0.1)
    rng = np.random.default_rng(0)
    clean = E.forward_ppm(FREQS, sigma, thickness=g.thickness).imag
    d = (clean * (1 + 0.03 * rng.standard_normal(len(FREQS))))[None, :]
    err = V.data_errors(d, None, 0.03, 0.0001)
    low, high = (V.invert(FREQS, d, err, g, alpha=a, lateral_weight=0) for a in (10.0, 1000.0))
    rough = [np.sum(np.diff(r.log_sigma[0]) ** 2) for r in (low, high)]
    assert 0.5 < low.chi2 <= 1.0 and 0.5 < high.chi2 <= 1.0     # fits, but not the noise
    assert rough[0] == pytest.approx(rough[1], rel=0.3)


def test_unreachable_target_stops_early_with_regularisation_left():
    g = V.make_layer_grid(6.0, 10)
    rng = np.random.default_rng(1)
    d = E.forward_ppm(FREQS, [0.02]).imag[None, :] * (1 + 0.1 * rng.standard_normal(len(FREQS)))
    res = V.invert(FREQS, d, V.data_errors(d, None, 0.001, 0.0001), g, lateral_weight=0)
    assert res.chi2 > 1.0
    assert res.iterations < V.MAX_ITERATIONS and res.alpha >= V.ALPHA_MIN


def test_jacobian_is_computed_in_chunks(monkeypatch):
    sizes = []
    real = E.forward_ppm_batch

    def counting(f, sigma, *args, **kwargs):
        sizes.append(len(sigma))
        return real(f, sigma, *args, **kwargs)

    monkeypatch.setattr(E, "forward_ppm_batch", counting)
    V._jacobian(FREQS, np.full((50, 10), -2.0), V.make_layer_grid(6.0, 10), E.GEM2)
    assert max(sizes) <= V.FORWARD_CHUNK and sum(sizes) == 50 * 11
