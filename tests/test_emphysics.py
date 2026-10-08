import numpy as np
import pytest
from scipy.special import j0

import emphysics as E

FREQS = [475.0, 1525.0, 5325.0, 18325.0, 63025.0]


def _reference_ratio(r, h, f, sigma, kappa=0.0):
    """Brute-force trapezoid Hankel integral for a half-space (needs h > 0)."""
    lam = np.linspace(0, 200 / max(h, 0.05), 400_001)
    w = 2 * np.pi * f
    mu1 = E.MU0 * (1 + kappa)
    u = np.sqrt(lam ** 2 + 1j * w * mu1 * sigma)
    R0 = (lam / E.MU0 - u / mu1) / (lam / E.MU0 + u / mu1)
    integrand = lam ** 2 * R0 * np.exp(-2 * lam * h) * j0(lam * r)
    return -r ** 3 * np.trapezoid(integrand, lam)


@pytest.mark.parametrize("sigma", [0.001, 0.05, 1.0])
@pytest.mark.parametrize("f", [1525.0, 63025.0])
def test_halfspace_matches_brute_force_integral(sigma, f):
    s = E.Sensor(height=0.3)
    ref = E.PPM * (_reference_ratio(1.66, 0.3, f, sigma) - _reference_ratio(1.035, 0.3, f, sigma))
    got = E.forward_ppm(f, [sigma], sensor=s)[0]
    assert got.imag == pytest.approx(ref.imag, rel=1e-4)
    assert got.real == pytest.approx(ref.real, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("h", [0.0, 0.5, 1.5])
def test_low_induction_number_limit(h):
    s = E.Sensor(height=h)
    q = E.forward_ppm(475.0, [1e-6], sensor=s).imag[0]
    assert q == pytest.approx(E.lin_ppm_per_sigma(475.0, s) * 1e-6, rel=5e-4)


@pytest.mark.parametrize("h", [0.0, 0.7])
def test_static_magnetic_halfspace_is_analytic(h):
    s = E.Sensor(height=h)
    kappa = 0.01
    a = 2 * h

    def z(r):
        return -r ** 3 * (kappa / (2 + kappa)) * (2 * a * a - r * r) / (a * a + r * r) ** 2.5

    expected = E.PPM * (z(1.66) - z(1.035))
    got = E.forward_ppm(1000.0, [0.0], [kappa], sensor=s)[0]
    assert got.real == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert abs(got.imag) < 1e-9


def test_bucking_cancels_static_susceptibility_at_ground_level():
    # At h = 0 both coils see kappa / (2 + kappa); the bucking coil removes it.
    assert abs(E.forward_ppm(1000.0, [0.0], [0.01], sensor=E.Sensor(height=0.0))[0]) < 1e-6


def test_identical_layers_equal_halfspace():
    one = E.forward_ppm(FREQS, [0.03])
    three = E.forward_ppm(FREQS, [0.03, 0.03, 0.03], thickness=[0.4, 2.0])
    np.testing.assert_allclose(three, one, rtol=1e-6)


def test_sensor_and_forward_reject_impossible_values():
    with pytest.raises(ValueError, match="bucking coil"):
        E.Sensor(separation=1.66, bucking=1.66)
    with pytest.raises(ValueError, match="height"):
        E.Sensor(height=-0.1)
    with pytest.raises(ValueError, match="> -1"):
        E.forward_ppm(1525.0, [0.01], [-1.0])


def test_deep_layer_matters_less_at_high_frequency_never_negative_sensitivity():
    c = E.cumulative_sensitivity(5325.0, 0.02, [0.0, 0.5, 1, 2, 4, 8, 16])
    assert c[0] == 1.0
    assert np.all(np.diff(c) < 0)
    assert 0 < c[-1] < 0.1


def test_ppm_round_trip_recovers_halfspace():
    for f in FREQS:
        for sigma, kappa in [(0.005, 0.0), (0.05, 2e-3), (0.5, 1e-4)]:
            z = E.forward_ppm(f, [sigma], [kappa])[0]
            s_est, k_est = E.halfspace_from_ppm(f, z.real, z.imag)
            assert s_est == pytest.approx(sigma, rel=1e-4)
            assert k_est == pytest.approx(kappa, abs=1e-6)


def test_skin_depth_and_induction_number():
    assert E.skin_depth(0.01, 10_000.0) == pytest.approx(50.33, rel=1e-3)
    assert E.induction_number(0.01, 10_000.0) == pytest.approx(1.66 / 50.33, rel=1e-3)
    assert E.skin_depth(0.0, 10_000.0) == float("inf")


def test_depth_of_investigation_is_finite_and_shallower_at_higher_height_no():
    d = E.depth_of_investigation(5325.0, 0.02)
    assert 0.5 < d < 5.0


def test_batch_conversion_matches_scalar_and_is_fast():
    import time

    rng = np.random.default_rng(0)
    sig = 10 ** rng.uniform(-3, -0.3, 2000)
    kap = rng.uniform(0, 3e-3, 2000)
    for f in (1525.0, 63025.0):
        z = E.forward_ppm_batch([f], sig[:, None], kap[:, None])[:, 0]
        t = time.perf_counter()
        s_est, k_est = E.halfspace_from_ppm_batch(f, z.real, z.imag)
        assert time.perf_counter() - t < 20
        np.testing.assert_allclose(s_est, sig, rtol=1e-4)
        np.testing.assert_allclose(k_est, kap, atol=1e-7)


def test_batch_conversion_blanks_non_positive_quadrature():
    s, k = E.halfspace_from_ppm_batch(1525.0, np.array([10.0, 10.0, np.nan]), np.array([-1.0, 0.0, 5.0]))
    assert np.isnan(s).all() and np.isnan(k).all()


@pytest.mark.parametrize("h", [0.0, 0.8212])
def test_conversion_where_in_phase_ignores_susceptibility(h):
    # Bucked GEM-2: no static susceptibility response at the ground and near 0.82 m.
    s = E.Sensor(height=h)
    assert not E.susceptibility_resolved(s)
    z = E.forward_ppm(1525.0, [0.02], [1e-3], sensor=s)[0]
    sigma, kappa = E.halfspace_from_ppm(1525.0, z.real - 500.0, z.imag, s)
    assert sigma == pytest.approx(0.02, rel=2e-3) and np.isnan(kappa)


def test_conversion_of_highly_conductive_ground():
    # The quadrature peaks near 2 S/m at 63 kHz; the in-phase picks the branch.
    for f, sigma in [(63025.0, 1.0), (63025.0, 3.0), (18325.0, 5.0)]:
        z = E.forward_ppm(f, [sigma], [1e-3])[0]
        s_est, k_est = E.halfspace_from_ppm(f, z.real, z.imag)
        assert s_est == pytest.approx(sigma, rel=1e-4)
        assert k_est == pytest.approx(1e-3, abs=1e-6)


def test_conversion_blanks_readings_no_halfspace_explains():
    s, k = E.halfspace_from_ppm_batch(63025.0, np.array([2e5, 10.0]), np.array([3e4, -1.0]))
    assert np.isnan(s).all() and np.isnan(k).all()
    assert np.isnan(E.halfspace_from_ppm(1525.0, 10.0, -1.0)).all()


def test_frequency_table_columns_and_trend():
    t = E.frequency_table([475.0, 63025.0], [0.02, 0.02])
    assert list(t.columns[:4]) == ["Frequency (Hz)", "EC (mS/m)", "Skin depth (m)", "Induction number"]
    assert t["Skin depth (m)"].iloc[0] > t["Skin depth (m)"].iloc[1]
    assert t["Induction number"].iloc[1] > t["Induction number"].iloc[0]


def test_multiheight_bias_is_recovered():
    f = np.array([1525.0, 5325.0, 18325.0, 63025.0])
    h = np.array([0.3, 0.8, 1.4, 2.0])
    bias_i = np.array([30.0, -20.0, 50.0, 100.0])
    bias_q = np.array([-15.0, 10.0, 40.0, -60.0])
    ii, qq = [], []
    for hk in h:
        z = E.forward_ppm(f, [0.04], [1e-3], sensor=E.Sensor(height=hk))
        ii.append(z.real + bias_i)
        qq.append(z.imag + bias_q)
    fit = E.fit_multiheight_bias(f, h, np.array(ii), np.array(qq))
    assert fit["sigma"] == pytest.approx(0.04, rel=1e-3)
    np.testing.assert_allclose(fit["bias_i"], bias_i, atol=0.5)
    np.testing.assert_allclose(fit["bias_q"], bias_q, atol=0.5)
    assert fit["rms"] < 1e-3


def test_multiheight_needs_two_heights():
    with pytest.raises(ValueError, match="2 or more heights"):
        E.fit_multiheight_bias([1525.0], [1.0, 1.0], [[1.0], [1.0]], [[1.0], [1.0]])


def test_multiheight_bias_from_ground_level():
    f = np.array([1525.0, 18325.0, 63025.0])
    h = np.array([0.0, 0.5, 1.0, 1.5])
    bias_i, bias_q = np.array([300.0, -200.0, 150.0]), np.array([-150.0, 100.0, -80.0])
    ii, qq = [], []
    for hk in h:
        z = E.forward_ppm(f, [0.03], [1e-3], sensor=E.Sensor(height=hk))
        ii.append(z.real + bias_i)
        qq.append(z.imag + bias_q)
    fit = E.fit_multiheight_bias(f, h, np.array(ii), np.array(qq))
    assert fit["sigma"] == pytest.approx(0.03, rel=1e-3)
    np.testing.assert_allclose(fit["bias_i"], bias_i, atol=0.5)
    np.testing.assert_allclose(fit["bias_q"], bias_q, atol=0.5)
