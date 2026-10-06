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

