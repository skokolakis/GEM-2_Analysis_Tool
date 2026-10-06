"""
Frequency-domain EM physics for the GEM-2: layered-earth forward model,
conversion of in-phase / quadrature (ppm) to apparent conductivity and
susceptibility, skin depth, induction number and depth sensitivity.

Pure functions only (no Streamlit).

Sign convention: e^{+iwt}; the secondary/primary ratio of a horizontal
co-planar (HCP) loop pair over a conductive half-space has a positive
quadrature, ~ i*w*mu0*sigma*s^2/4 at low induction number (McNeill, 1980).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0

MU0 = 4e-7 * np.pi
PPM = 1e6

# Gauss-Legendre nodes per half-period of J0(lambda * r)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)
LAMBDA_MAX_TIMES_R = 60.0       # integrate lambda * r up to this (see tests)
LAMBDA_MIN_TIMES_R = 1e-5       # first geometric panel edge, lambda * r
SMALL_PANELS = 24               # geometric panels below one half-period of J0
THIN_LAYER_FACTOR = 8.0         # ... and lambda up to this / thinnest layer


@dataclass(frozen=True)
class Sensor:
    """
    Loop-loop geometry. The GEM-2 uses a transmitter, a receiver at
    `separation` and a bucking coil at `bucking` that cancels the primary
    field at the receiver (Won et al., 1996). Check both values against the
    sensor configuration file (*.gem) of your instrument.
    """
    separation: float = 1.66      # Tx-Rx, m
    bucking: float | None = 1.035  # Tx-bucking coil, m; None for a plain pair
    height: float = 1.0           # sensor above ground, m


GEM2 = Sensor()


def _quadrature_nodes(r: float, lam_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite Gauss-Legendre nodes on [0, lam_max]: geometric panels up to
    one half-period of J0 (they resolve the induction scale sqrt(w mu sigma),
    which is far below 1 / r at low induction number), then one panel per
    half-period of J0(lam r).
    """
    width = np.pi / r
    small = np.geomspace(LAMBDA_MIN_TIMES_R / r, width, SMALL_PANELS + 1)
    edges = np.concatenate([[0.0], small, np.arange(2, int(np.ceil(lam_max / width)) + 1) * width])
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * np.diff(edges)[:, None]
    lam = (mid + half * _GL_NODES[None, :]).ravel()
    w = (half * _GL_WEIGHTS[None, :]).ravel()
    return lam, w


def _surface_reflection(
    lam: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    thickness: np.ndarray,
) -> np.ndarray:
    """
    TE reflection coefficient at the surface, quasi-static.

    lam: (L,), omega: (F,), sigma/kappa: (M, N) for M models of N layers
    (last = half-space), thickness: (N-1,). Returns (M, F, L).
    """
    mu = MU0 * (1.0 + kappa)[:, :, None, None]          # (M, N, 1, 1)
    sig = sigma[:, :, None, None]
    lam = lam[None, None, :]                            # (1, 1, L)
    w = omega[None, :, None]                            # (1, F, 1)
    n = sigma.shape[1]
    u = np.sqrt(lam ** 2 + 1j * w * mu[:, n - 1] * sig[:, n - 1])
    y_hat = u / mu[:, n - 1]
    for k in range(n - 2, -1, -1):
        u = np.sqrt(lam ** 2 + 1j * w * mu[:, k] * sig[:, k])
        y = u / mu[:, k]
        e = np.exp(-2.0 * u * thickness[k])   # Re(u) > 0: no overflow, unlike tanh
        t = (1.0 - e) / (1.0 + e)
        y_hat = y * (y_hat + y * t) / (y + y_hat * t)
    y0 = lam / MU0
    return (y0 - y_hat) / (y0 + y_hat)


def _hcp_ratio(
    r: float,
    height: float,
    omega: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    thickness: np.ndarray,
) -> np.ndarray:
    """
    Secondary/primary field ratio Hs/Hp, shape (M, F), for vertical magnetic
    dipoles at separation r and height h above M layered earths:

        Z = -r^3 * Int lam^2 R0(lam) exp(-2 lam h) J0(lam r) dlam

    The large-lam limit lam^2 R0 -> A lam^2 + B (A, B from the top layer) is
    integrated analytically, so the numerical integrand decays like lam^-2
    even when h = 0.
    """
    a = 2.0 * height
    mur = 1.0 + kappa[:, :1]                                        # (M, 1)
    A = (mur - 1.0) / (mur + 1.0)                                   # (M, 1)
    B = -1j * omega[None, :] * MU0 * mur ** 2 * sigma[:, :1] / (mur + 1.0) ** 2   # (M, F)
    lam_max = LAMBDA_MAX_TIMES_R / r
    if len(thickness):
        lam_max = max(lam_max, THIN_LAYER_FACTOR / float(np.min(thickness)))
    lam, w = _quadrature_nodes(r, lam_max)
    R0 = _surface_reflection(lam, omega, sigma, kappa, thickness)    # (M, F, L)
    rest = lam ** 2 * R0 - A[:, :, None] * lam ** 2 - B[:, :, None]
    integral = (rest * (np.exp(-a * lam) * j0(lam * r))) @ w         # (M, F)
    d2 = a * a + r * r
    analytic = A * (2 * a * a - r * r) / d2 ** 2.5 + B / np.sqrt(d2)
    return -r ** 3 * (integral + analytic)


def forward_ppm_batch(
    frequencies,
    sigma,
    kappa=None,
    thickness=None,
    sensor: Sensor = GEM2,
) -> np.ndarray:
    """
    Complex response in ppm, shape (M, F), for M layered models that share
    the layer thicknesses. sigma, kappa: (M, N) in S/m and SI; thickness:
    (N-1,) in m. Real part = in-phase, imaginary part = quadrature.

    With a bucking coil the measured ratio is Z(Rx) - Z(bucking): the bucking
    coil is wound so that its primary cancels the receiver's, which makes its
    secondary contribute with weight 1 relative to the receiver's primary.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    kappa = np.zeros_like(sigma) if kappa is None else np.atleast_2d(np.asarray(kappa, float))
    thickness = np.zeros(0) if thickness is None else np.atleast_1d(np.asarray(thickness, float))
    if kappa.shape != sigma.shape or sigma.shape[1] != len(thickness) + 1:
        raise ValueError("need kappa.shape == sigma.shape == (M, len(thickness) + 1)")
    if np.any(sigma < 0) or np.any(thickness <= 0) or sensor.height < 0:
        raise ValueError("conductivities and height must be >= 0, thicknesses > 0")
    omega = 2 * np.pi * f
    z = _hcp_ratio(sensor.separation, sensor.height, omega, sigma, kappa, thickness)
    if sensor.bucking:
        z = z - _hcp_ratio(sensor.bucking, sensor.height, omega, sigma, kappa, thickness)
    return PPM * z


def forward_ppm(
    frequencies,
    sigma,
    kappa=None,
    thickness=None,
    sensor: Sensor = GEM2,
) -> np.ndarray:
    """
    Complex response in ppm of one layered earth, one value per frequency.

    sigma: layer conductivities in S/m (last entry = half-space);
    kappa: layer susceptibilities in SI (default 0);
    thickness: thicknesses of all but the last layer, m.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    kappa = None if kappa is None else np.atleast_1d(np.asarray(kappa, dtype=float))[None, :]
    if kappa is not None and kappa.shape[1] != sigma.size:
        raise ValueError("need len(kappa) == len(sigma)")
    return forward_ppm_batch(frequencies, sigma[None, :], kappa, thickness, sensor)[0]


def lin_ppm_per_sigma(frequency: float, sensor: Sensor = GEM2) -> float:
    """
    Quadrature ppm per S/m at low induction number, including sensor height
    (McNeill's cumulative response 1 / sqrt(4 z^2 + 1), z = h / r).
    """
    w = 2 * np.pi * float(frequency)

    def term(r):
        return r * r / np.sqrt(4 * (sensor.height / r) ** 2 + 1)

    total = term(sensor.separation) - (term(sensor.bucking) if sensor.bucking else 0.0)
    return PPM * w * MU0 * total / 4.0

