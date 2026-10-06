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


def halfspace_from_ppm(
    frequency: float,
    inphase: float,
    quadrature: float,
    sensor: Sensor = GEM2,
) -> tuple[float, float]:
    """
    Apparent conductivity (S/m) and susceptibility (SI) of the homogeneous
    half-space that reproduces one (in-phase, quadrature) pair in ppm
    (Huang & Won, 2000). Starts from the low-induction-number estimate.
    """
    q_per_sigma = lin_ppm_per_sigma(frequency, sensor)
    i_per_kappa = forward_ppm(frequency, [0.0], [1e-3], sensor=sensor).real[0] / 1e-3
    s0 = max(quadrature / q_per_sigma, 1e-5)
    k0 = inphase / i_per_kappa if i_per_kappa else 0.0
    scale = np.array([abs(quadrature) + 1.0, abs(inphase) + 1.0])

    def residual(p):
        z = forward_ppm(frequency, [np.exp(p[0])], [p[1]], sensor=sensor)[0]
        return (np.array([z.imag, z.real]) - [quadrature, inphase]) / scale

    sol = least_squares(residual, [np.log(s0), k0], method="lm", xtol=1e-12, ftol=1e-12)
    return float(np.exp(sol.x[0])), float(sol.x[1])


def skin_depth(sigma: float, frequency: float) -> float:
    """Plane-wave skin depth sqrt(2 / (w mu0 sigma)) in m."""
    if sigma <= 0:
        return float("inf")
    return float(np.sqrt(2.0 / (2 * np.pi * frequency * MU0 * sigma)))


def induction_number(sigma: float, frequency: float, sensor: Sensor = GEM2) -> float:
    """B = separation / skin depth; LIN holds for B << 1."""
    return float(sensor.separation / skin_depth(sigma, frequency))


def cumulative_sensitivity(
    frequency: float, sigma: float, depths, sensor: Sensor = GEM2
) -> np.ndarray:
    """
    Fraction of the half-space quadrature response that comes from below each
    depth: Q(resistive layer of thickness z over sigma) / Q(half-space sigma).
    """
    full = forward_ppm(frequency, [sigma], sensor=sensor).imag[0]
    out = []
    for z in np.atleast_1d(depths):
        if z <= 0:
            out.append(1.0)
            continue
        q = forward_ppm(frequency, [1e-8, sigma], thickness=[z], sensor=sensor).imag[0]
        out.append(q / full)
    return np.asarray(out, dtype=float)


def depth_of_investigation(
    frequency: float, sigma: float, sensor: Sensor = GEM2, fraction: float = 0.3
) -> float:
    """Depth below which only `fraction` of the quadrature response originates."""
    depths = np.geomspace(0.01, 100.0, 200)
    c = cumulative_sensitivity(frequency, sigma, depths, sensor)
    below = np.nonzero(c <= fraction)[0]
    if below.size == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return float(depths[0])
    # log-linear interpolation between the bracketing depths
    x0, x1 = np.log(depths[i - 1]), np.log(depths[i])
    t = (c[i - 1] - fraction) / (c[i - 1] - c[i])
    return float(np.exp(x0 + t * (x1 - x0)))


# ---------------------------------------------------------------------------
# Conversion of many readings, frequency information, multi-height calibration
# ---------------------------------------------------------------------------

CONVERSION_CHUNK = 4000        # readings per batched forward call
CONVERSION_ITERATIONS = 15
LOG_SIGMA_STEP = 1e-4          # finite-difference steps for the 2 x 2 Jacobian
KAPPA_STEP = 1e-6


def _halfspace_batch(frequency: float, log_sigma: np.ndarray, kappa: np.ndarray, sensor: Sensor):
    z = forward_ppm_batch([frequency], np.exp(log_sigma)[:, None], kappa[:, None], sensor=sensor)
    return z[:, 0]


def halfspace_from_ppm_batch(
    frequency: float,
    inphase: np.ndarray,
    quadrature: np.ndarray,
    sensor: Sensor = GEM2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised halfspace_from_ppm for many readings at one frequency:
    damped Gauss-Newton on (ln sigma, kappa) with a 2 x 2 finite-difference
    Jacobian, started from the low-induction-number estimates. Readings with
    a quadrature <= 0 or missing values return NaN. Returns (sigma S/m, kappa SI).
    """
    i_all = np.asarray(inphase, dtype=float)
    q_all = np.asarray(quadrature, dtype=float)
    sigma = np.full(i_all.shape, np.nan)
    kappa = np.full(i_all.shape, np.nan)
    ok = np.isfinite(i_all) & np.isfinite(q_all) & (q_all > 0)
    q_per_sigma = lin_ppm_per_sigma(frequency, sensor)
    i_per_kappa = forward_ppm(frequency, [0.0], [1e-3], sensor=sensor).real[0] / 1e-3
    for start in range(0, int(ok.sum()), CONVERSION_CHUNK):
        rows = np.nonzero(ok)[0][start: start + CONVERSION_CHUNK]
        i_obs, q_obs = i_all[rows], q_all[rows]
        scale_q, scale_i = np.abs(q_obs) + 1.0, np.abs(i_obs) + 1.0
        ls = np.log(np.maximum(q_obs / q_per_sigma, 1e-6))
        k = i_obs / i_per_kappa if i_per_kappa else np.zeros_like(i_obs)

        def misfit(z):
            return np.hypot((z.imag - q_obs) / scale_q, (z.real - i_obs) / scale_i)

        z = _halfspace_batch(frequency, ls, k, sensor)
        for _ in range(CONVERSION_ITERATIONS):
            zs = _halfspace_batch(frequency, ls + LOG_SIGMA_STEP, k, sensor)
            zk = _halfspace_batch(frequency, ls, k + KAPPA_STEP, sensor)
            # rows: (quadrature, in-phase) residuals; columns: (ln sigma, kappa)
            a = (zs.imag - z.imag) / LOG_SIGMA_STEP / scale_q
            b = (zk.imag - z.imag) / KAPPA_STEP / scale_q
            c = (zs.real - z.real) / LOG_SIGMA_STEP / scale_i
            d = (zk.real - z.real) / KAPPA_STEP / scale_i
            rq = (q_obs - z.imag) / scale_q
            ri = (i_obs - z.real) / scale_i
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            d_ls = (d * rq - b * ri) / det
            d_k = (a * ri - c * rq) / det
            current = misfit(z)
            step = np.ones_like(ls)
            accepted = np.zeros(ls.shape, dtype=bool)
            for _ in range(6):                      # halve the step until the misfit drops
                trial_ls = ls + step * np.clip(d_ls, -2.0, 2.0)
                trial_k = k + step * d_k
                zt = _halfspace_batch(frequency, trial_ls, trial_k, sensor)
                better = (misfit(zt) < current) & ~accepted
                ls = np.where(better, trial_ls, ls)
                k = np.where(better, trial_k, k)
                z = np.where(better, zt, z)
                accepted |= better
                step = np.where(accepted, step, 0.5 * step)
                if accepted.all():
                    break
            if np.max(np.abs(d_ls)) < 1e-9 and np.max(np.abs(d_k)) < 1e-12:
                break
        sigma[rows] = np.exp(ls)
        kappa[rows] = k
    return sigma, kappa


def frequency_table(frequencies, sigma_s_m, sensor: Sensor = GEM2, fraction: float = 0.3):
    """
    Per frequency: skin depth, induction number and depth of investigation
    (depth below which only `fraction` of the quadrature response originates)
    for the given apparent conductivities (S/m, one per frequency).
    """
    import pandas as pd

    doi_col = f"Depth of investigation (m, {100 * (1 - fraction):.0f} % above)"
    rows = []
    for f, s in zip(np.atleast_1d(frequencies), np.atleast_1d(sigma_s_m)):
        f, s = float(f), float(s)
        ok = np.isfinite(s) and s > 0
        rows.append({
            "Frequency (Hz)": f,
            "EC (mS/m)": s * 1000 if ok else np.nan,
            "Skin depth (m)": skin_depth(s, f) if ok else np.nan,
            "Induction number": induction_number(s, f, sensor) if ok else np.nan,
            doi_col: depth_of_investigation(f, s, sensor, fraction) if ok else np.nan,
        })
    return pd.DataFrame(rows)


def fit_multiheight_bias(
    frequencies,
    heights,
    inphase: np.ndarray,
    quadrature: np.ndarray,
    sensor: Sensor = GEM2,
) -> dict:
    """
    Multi-elevation calibration (after Minsley et al., 2014): readings over one
    spot at several sensor heights are fitted by a homogeneous half-space plus
    an additive bias per frequency and component. inphase / quadrature have
    shape (n_heights, n_freq) in ppm. Needs at least 2 heights.

    Returns {"sigma": S/m, "kappa": SI, "bias_i": (F,), "bias_q": (F,), "rms": ppm}.
    The biases are the values to subtract from the survey's I and Q.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    h = np.atleast_1d(np.asarray(heights, dtype=float))
    i_obs = np.atleast_2d(np.asarray(inphase, dtype=float))
    q_obs = np.atleast_2d(np.asarray(quadrature, dtype=float))
    if len(np.unique(h)) < 2:
        raise ValueError("Multi-height calibration needs readings at 2 or more heights.")
    n_f = len(f)

    def model(p):
        s, kap = np.exp(p[0]), p[1]
        out = np.array([
            forward_ppm(f, [s], [kap], sensor=Sensor(sensor.separation, sensor.bucking, hk))
            for hk in h
        ])
        return out.real + p[2: 2 + n_f], out.imag + p[2 + n_f:]

    def residual(p):
        mi, mq = model(p)
        return np.concatenate([(mi - i_obs).ravel(), (mq - q_obs).ravel()])

    lowest = int(np.argmin(h))
    s0, k0 = halfspace_from_ppm(f[0], i_obs[lowest, 0], q_obs[lowest, 0],
                                Sensor(sensor.separation, sensor.bucking, h[lowest]))
    p0 = np.concatenate([[np.log(max(s0, 1e-5)), k0], np.zeros(2 * n_f)])
    sol = least_squares(residual, p0, x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=2000)
    return {
        "sigma": float(np.exp(sol.x[0])),
        "kappa": float(sol.x[1]),
        "bias_i": sol.x[2: 2 + n_f].copy(),
        "bias_q": sol.x[2 + n_f:].copy(),
        "rms": float(np.sqrt(np.mean(sol.fun ** 2))),
    }

