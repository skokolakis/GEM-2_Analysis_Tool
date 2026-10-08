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

    def __post_init__(self):
        if not self.separation > 0:
            raise ValueError("The Tx–Rx separation must be positive.")
        if self.bucking is not None and not 0 < self.bucking < self.separation:
            raise ValueError("The bucking coil must lie between the transmitter and the receiver.")
        if not self.height >= 0:
            raise ValueError("The sensor height must not be negative.")


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
    if np.any(kappa <= -1):
        raise ValueError("susceptibilities must be > -1 SI")
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
    (Huang & Won, 2000). NaN where halfspace_from_ppm_batch gives NaN.
    """
    sigma, kappa = halfspace_from_ppm_batch(frequency, [inphase], [quadrature], sensor)
    return float(sigma[0]), float(kappa[0])


CONVERSION_CHUNK = 4000        # readings per batched forward call
CONVERSION_ITERATIONS = 25
LOG_SIGMA_STEP = 1e-4          # finite-difference steps for the 2 x 2 Jacobian
KAPPA_STEP = 1e-6
SIGMA_RANGE = (1e-5, 10.0)     # S/m, apparent conductivities a conversion can return
KAPPA_RANGE = (-0.1, 1.0)      # SI; beyond these the in-phase is not a susceptibility
MIN_PPM_PER_KAPPA = 5000.0     # static in-phase ppm per SI below which MS is not resolved
CONVERSION_TOLERANCE = 1e-3    # normalised misfit above which no half-space fits a reading
_GRID_LOG_SIGMA = np.log(np.geomspace(*SIGMA_RANGE, 61))
_GRID_KAPPA = np.array([-0.1, -0.03, -0.01, -0.003, -0.001, 0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])


def _halfspace_batch(frequency: float, log_sigma: np.ndarray, kappa: np.ndarray, sensor: Sensor):
    z = forward_ppm_batch([frequency], np.exp(log_sigma)[:, None], kappa[:, None], sensor=sensor)
    return z[:, 0]


def kappa_sensitivity(sensor: Sensor = GEM2) -> float:
    """Static in-phase response in ppm per SI of susceptibility (frequency independent)."""
    return float(forward_ppm(1000.0, [0.0], [1e-3], sensor=sensor).real[0] / 1e-3)


def susceptibility_resolved(sensor: Sensor = GEM2) -> bool:
    """
    False where the in-phase barely depends on susceptibility: with a bucking
    coil, at ground level (both coils see the same image) and near the height
    where the bucked static response changes sign (about 0.82 m for the GEM-2).
    """
    return abs(kappa_sensitivity(sensor)) >= MIN_PPM_PER_KAPPA


def halfspace_from_ppm_batch(
    frequency: float,
    inphase: np.ndarray,
    quadrature: np.ndarray,
    sensor: Sensor = GEM2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apparent conductivity and susceptibility of many readings at one
    frequency (Huang & Won, 2000). Each reading starts from the closest
    model of a (sigma, kappa) grid, which picks the right branch where the
    quadrature is no longer monotonic in sigma (high induction number); a
    damped Gauss-Newton on (ln sigma, kappa) with a 2 x 2 finite-difference
    Jacobian then refines it within SIGMA_RANGE and KAPPA_RANGE. Converged
    readings drop out of the iterations.

    Where the in-phase is insensitive to susceptibility (see
    susceptibility_resolved) kappa is fixed at 0, sigma comes from the
    quadrature alone (below its maximum) and kappa is returned as NaN.
    Readings with missing values, a quadrature <= 0 or no half-space within
    the ranges that reproduces them return NaN. Returns (sigma S/m, kappa SI).
    """
    i_all = np.atleast_1d(np.asarray(inphase, dtype=float))
    q_all = np.atleast_1d(np.asarray(quadrature, dtype=float))
    sigma = np.full(i_all.shape, np.nan)
    kappa = np.full(i_all.shape, np.nan)
    free = susceptibility_resolved(sensor)
    if free:
        gs, gk = (g.ravel() for g in np.meshgrid(_GRID_LOG_SIGMA, _GRID_KAPPA, indexing="ij"))
    else:
        gs, gk = _GRID_LOG_SIGMA, np.zeros_like(_GRID_LOG_SIGMA)
    gz = _halfspace_batch(frequency, gs, gk, sensor)
    lo_s, hi_s = np.log(SIGMA_RANGE)
    lo_k, hi_k = KAPPA_RANGE if free else (0.0, 0.0)
    if not free:                      # quadrature only: keep the branch below its maximum
        peak = int(np.argmax(gz.imag))
        gs, gk, gz, hi_s = gs[: peak + 1], gk[: peak + 1], gz[: peak + 1], gs[peak]
    w_i = 1.0 if free else 0.0
    ok = np.isfinite(i_all) & np.isfinite(q_all) & (q_all > 0)
    rows_ok = np.nonzero(ok)[0]
    for start in range(0, len(rows_ok), CONVERSION_CHUNK):
        rows = rows_ok[start: start + CONVERSION_CHUNK]
        i_obs, q_obs = i_all[rows], q_all[rows]
        scale_q = scale_i = np.hypot(i_obs, q_obs) + 1.0     # one scale: |reading| + 1 ppm

        def misfit(z, sel):
            return np.hypot((z.imag - q_obs[sel]) / scale_q[sel], w_i * (z.real - i_obs[sel]) / scale_i[sel])

        dist = np.hypot((gz.imag[None, :] - q_obs[:, None]) / scale_q[:, None],
                        w_i * (gz.real[None, :] - i_obs[:, None]) / scale_i[:, None])
        best = np.argmin(dist, axis=1)
        ls, k = gs[best].copy(), gk[best].copy()
        z = gz[best].copy()
        active = np.ones(len(rows), dtype=bool)
        for _ in range(CONVERSION_ITERATIONS):
            a_rows = np.nonzero(active)[0]
            if a_rows.size == 0:
                break
            lsa, ka, za = ls[a_rows], k[a_rows], z[a_rows]
            sq, si = scale_q[a_rows], scale_i[a_rows]
            zs = _halfspace_batch(frequency, lsa + LOG_SIGMA_STEP, ka, sensor)
            # rows: (quadrature, in-phase) residuals; columns: (ln sigma, kappa)
            a = (zs.imag - za.imag) / LOG_SIGMA_STEP / sq
            c = (zs.real - za.real) / LOG_SIGMA_STEP / si
            rq = (q_obs[a_rows] - za.imag) / sq
            ri = (i_obs[a_rows] - za.real) / si
            if free:
                zk = _halfspace_batch(frequency, lsa, ka + KAPPA_STEP, sensor)
                b = (zk.imag - za.imag) / KAPPA_STEP / sq
                d = (zk.real - za.real) / KAPPA_STEP / si
                det = a * d - b * c
                det = np.where(np.abs(det) < 1e-30, 1e-30, det)
                d_ls = (d * rq - b * ri) / det
                d_k = (a * ri - c * rq) / det
            else:
                d_ls = np.where(np.abs(a) > 1e-30, rq / np.where(np.abs(a) > 1e-30, a, 1.0), 0.0)
                d_k = np.zeros_like(d_ls)
            shrink = np.minimum(1.0, 2.0 / np.maximum(np.abs(d_ls), 1e-300))   # |d ln sigma| <= 2
            d_ls, d_k = d_ls * shrink, d_k * shrink
            current = misfit(za, a_rows)
            step = np.ones_like(lsa)
            accepted = np.zeros(lsa.shape, dtype=bool)
            for _ in range(8):                      # halve the step until the misfit drops
                trial_ls = np.clip(lsa + step * d_ls, lo_s, hi_s)
                trial_k = np.clip(ka + step * d_k, lo_k, hi_k)
                zt = _halfspace_batch(frequency, trial_ls, trial_k, sensor)
                better = (misfit(zt, a_rows) < current) & ~accepted
                lsa = np.where(better, trial_ls, lsa)
                ka = np.where(better, trial_k, ka)
                za = np.where(better, zt, za)
                accepted |= better
                step = np.where(accepted, step, 0.5 * step)
                if accepted.all():
                    break
            ls[a_rows], k[a_rows], z[a_rows] = lsa, ka, za
            done = ~accepted | ((np.abs(d_ls) < 1e-10) & (np.abs(d_k) < 1e-12))
            active[a_rows[done]] = False
        fits = misfit(z, slice(None)) <= CONVERSION_TOLERANCE
        sigma[rows[fits]] = np.exp(ls[fits])
        kappa[rows[fits]] = k[fits] if free else np.nan
    return sigma, kappa


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
# Frequency information and multi-height calibration
# ---------------------------------------------------------------------------


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

    For a given half-space the best biases are the mean differences between
    the readings and the model over the heights, so only (sigma, kappa) are
    searched: first on a grid, then by bounded least squares. This works
    from any starting heights, including the sensor on the ground.

    Returns {"sigma": S/m, "kappa": SI, "bias_i": (F,), "bias_q": (F,), "rms": ppm}.
    The biases are the values to subtract from the survey's I and Q.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    h = np.atleast_1d(np.asarray(heights, dtype=float))
    i_obs = np.atleast_2d(np.asarray(inphase, dtype=float))
    q_obs = np.atleast_2d(np.asarray(quadrature, dtype=float))
    if len(np.unique(h)) < 2:
        raise ValueError("Multi-height calibration needs readings at 2 or more heights.")
    obs = i_obs + 1j * q_obs                                  # (K, F)
    sensors = [Sensor(sensor.separation, sensor.bucking, hk) for hk in h]

    def response(log_s, kap):                                # (M,) -> (M, K, F)
        return np.stack([
            forward_ppm_batch(f, np.exp(log_s)[:, None], kap[:, None], sensor=sk) for sk in sensors
        ], axis=1)

    def residuals(z):                                         # after the best biases
        r = obs[None] - z
        r = r - r.mean(axis=1, keepdims=True)
        return np.concatenate([r.real.reshape(len(z), -1), r.imag.reshape(len(z), -1)], axis=1)

    gs, gk = (g.ravel() for g in np.meshgrid(_GRID_LOG_SIGMA, _GRID_KAPPA, indexing="ij"))
    best = int(np.argmin(np.sum(residuals(response(gs, gk)) ** 2, axis=1)))
    lower = [np.log(SIGMA_RANGE[0]), KAPPA_RANGE[0]]
    upper = [np.log(SIGMA_RANGE[1]), KAPPA_RANGE[1]]
    sol = least_squares(
        lambda p: residuals(response(p[:1], p[1:]))[0], [gs[best], gk[best]],
        bounds=(lower, upper), x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=2000,
    )
    z = response(sol.x[:1], sol.x[1:])[0]
    bias = (obs - z).mean(axis=0)
    return {
        "sigma": float(np.exp(sol.x[0])),
        "kappa": float(sol.x[1]),
        "bias_i": bias.real.copy(),
        "bias_q": bias.imag.copy(),
        "rms": float(np.sqrt(np.mean(sol.fun ** 2))),
    }
