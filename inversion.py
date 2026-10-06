"""
Smooth 1D and laterally constrained (quasi-2D) inversion of GEM-2
quadrature data for layered conductivity, plus EMagPy-ready export.

Pure functions only (no Streamlit). Model parameters are log10(sigma) of
fixed layers; regularisation penalises vertical and lateral roughness
(Constable et al., 1987 "Occam"; Auken & Christiansen, 2004 "LCI").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

import contouring as ctr
import corrections
import emphysics as E
import gem_io
import pipeline

MIN_LOG_SIGMA = -5.0          # 0.01 mS/m
MAX_LOG_SIGMA = 1.0           # 10 S/m
JACOBIAN_STEP = 0.01          # log10 units
MAX_ITERATIONS = 15
ALPHA_DECREASE = 0.5          # regularisation cooling per iteration
STEP_HALVINGS = 6


@dataclass(frozen=True)
class LayerGrid:
    """Fixed layer thicknesses; the last layer is a half-space."""
    thickness: np.ndarray        # (N-1,)

    @property
    def n_layers(self) -> int:
        return len(self.thickness) + 1

    @property
    def depth_top(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.thickness)])


def make_layer_grid(max_depth: float = 6.0, n_layers: int = 15, first: float = 0.1) -> LayerGrid:
    """n_layers - 1 thicknesses growing geometrically from `first` so they sum to max_depth."""
    n = n_layers - 1
    if n < 1 or first <= 0 or max_depth <= first:
        raise ValueError("need n_layers >= 2 and 0 < first < max_depth")
    # Solve first * (q^n - 1) / (q - 1) = max_depth for the growth factor q.
    lo, hi = 1.0 + 1e-9, 10.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        total = first * n if abs(q - 1) < 1e-12 else first * (q ** n - 1) / (q - 1)
        lo, hi = (q, hi) if total < max_depth else (lo, q)
    return LayerGrid(thickness=first * q ** np.arange(n))


@dataclass
class InversionResult:
    log_sigma: np.ndarray        # (n_stations, n_layers)
    grid: LayerGrid
    predicted: np.ndarray        # (n_stations, n_freq) quadrature ppm
    chi2: float                  # misfit / number of data
    alpha: float
    iterations: int
    history: list[float]


def _forward_q(frequencies, log_sigma: np.ndarray, grid: LayerGrid, sensor: E.Sensor) -> np.ndarray:
    """Quadrature ppm for M models: log_sigma (M, N) -> (M, F)."""
    return E.forward_ppm_batch(frequencies, 10.0 ** log_sigma, None, grid.thickness, sensor).imag


def _jacobian(frequencies, log_sigma: np.ndarray, grid: LayerGrid, sensor: E.Sensor):
    """Forward response (S, F) and per-station Jacobians (S, F, N) by forward differences."""
    s, n = log_sigma.shape
    perturbed = np.repeat(log_sigma[:, None, :], n + 1, axis=1)          # (S, N+1, N)
    idx = np.arange(n)
    perturbed[:, idx + 1, idx] += JACOBIAN_STEP
    d = _forward_q(frequencies, perturbed.reshape(-1, n), grid, sensor).reshape(s, n + 1, -1)
    base = d[:, 0, :]                                                     # (S, F)
    jac = (d[:, 1:, :] - base[:, None, :]) / JACOBIAN_STEP                # (S, N, F)
    return base, np.transpose(jac, (0, 2, 1))                             # (S, F, N)


def _roughness(n_stations: int, n_layers: int, lateral_weight: float) -> sparse.csr_matrix:
    """Stacked first-difference operators: vertical within, lateral between stations."""
    dz = sparse.diags([-1.0, 1.0], [0, 1], shape=(n_layers - 1, n_layers))
    rz = sparse.kron(sparse.identity(n_stations), dz)
    blocks = [rz]
    if n_stations > 1 and lateral_weight > 0:
        dx = sparse.diags([-1.0, 1.0], [0, 1], shape=(n_stations - 1, n_stations))
        blocks.append(lateral_weight * sparse.kron(dx, sparse.identity(n_layers)))
    return sparse.vstack(blocks).tocsr()


def invert(
    frequencies,
    data_q: np.ndarray,
    errors: np.ndarray,
    grid: LayerGrid,
    sensor: E.Sensor = E.GEM2,
    alpha: float = 10.0,
    lateral_weight: float = 1.0,
    start_log_sigma: np.ndarray | None = None,
    target_chi2: float = 1.0,
) -> InversionResult:
    """
    Gauss-Newton inversion of quadrature data (ppm) for log10(sigma).

    data_q, errors: (n_stations, n_freq); errors are 1-sigma data errors in
    ppm. Stations are inverted jointly with lateral smoothing weighted by
    `lateral_weight` (0 = independent 1D inversions). The regularisation
    weight starts at `alpha` and is halved each iteration while the misfit
    stays above `target_chi2` (chi-squared per datum).
    """
    d = np.atleast_2d(np.asarray(data_q, dtype=float))
    e = np.atleast_2d(np.asarray(errors, dtype=float))
    if d.shape != e.shape or np.any(~np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("errors must be positive, finite and the same shape as the data")
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n_st, n_l = d.shape[0], grid.n_layers
    finite = np.isfinite(d)
    w = np.where(finite, 1.0 / e, 0.0)
    d0 = np.where(finite, d, 0.0)

    if start_log_sigma is None:
        # Start from the mean low-induction-number apparent conductivity of each station.
        lin = np.array([E.lin_ppm_per_sigma(fi, sensor) for fi in f])
        sa = np.nanmedian(np.where(finite, d / lin, np.nan), axis=1)
        sa = np.where(np.isfinite(sa) & (sa > 0), sa, 1e-2)
        m = np.repeat(np.log10(sa)[:, None], n_l, axis=1)
    else:
        m = np.array(start_log_sigma, dtype=float).reshape(n_st, n_l)
    m = np.clip(m, MIN_LOG_SIGMA, MAX_LOG_SIGMA)

    rough = _roughness(n_st, n_l, lateral_weight)
    n_data = int(finite.sum())

    def misfit(pred):
        return float(np.sum((w * (d0 - pred)) ** 2)) / n_data

    def objective(pred, model, a):
        return misfit(pred) * n_data + a * float(np.sum((rough @ model.ravel()) ** 2))

    pred = _forward_q(f, m, grid, sensor)
    history = [misfit(pred)]
    it = 0
    for it in range(1, MAX_ITERATIONS + 1):
        if history[-1] <= target_chi2:
            break
        base, jac = _jacobian(f, m, grid, sensor)
        wj = sparse.block_diag([w[i][:, None] * jac[i] for i in range(n_st)], format="csr")
        r = (w * (d0 - base)).ravel()
        lhs = sparse.vstack([wj, np.sqrt(alpha) * rough]).tocsr()
        rhs = np.concatenate([r, -np.sqrt(alpha) * (rough @ m.ravel())])
        step = lsqr(lhs, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)[0].reshape(n_st, n_l)
        current = objective(base, m, alpha)
        for _ in range(STEP_HALVINGS):
            trial = np.clip(m + step, MIN_LOG_SIGMA, MAX_LOG_SIGMA)
            trial_pred = _forward_q(f, trial, grid, sensor)
            if objective(trial_pred, trial, alpha) < current:
                m, pred = trial, trial_pred
                break
            step *= 0.5
        history.append(misfit(pred))
        alpha *= ALPHA_DECREASE
    return InversionResult(
        log_sigma=m, grid=grid, predicted=pred, chi2=history[-1],
        alpha=alpha, iterations=it, history=history,
    )


def data_errors(
    data: np.ndarray, noise: np.ndarray | None, relative: float = 0.03, floor: float = 1.0
) -> np.ndarray:
    """
    1-sigma errors in ppm: the larger of the measured noise (per frequency,
    e.g. the between-pass sigma) and relative * |data| + floor.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    model_err = relative * np.abs(data) + floor
    if noise is None:
        return model_err
    return np.maximum(model_err, np.broadcast_to(np.asarray(noise, dtype=float), data.shape))


def ec_to_quadrature(frequencies, ec_ms_m: np.ndarray, sensor: E.Sensor = E.GEM2) -> np.ndarray:
    """
    Quadrature ppm of the half-space whose conductivity is the exported
    apparent conductivity (mS/m): undoes a half-space conversion so that
    EC-only exports can be inverted. Shape (n_stations, n_freq).
    """
    ec = np.atleast_2d(np.asarray(ec_ms_m, dtype=float)) / 1000.0
    out = np.full(ec.shape, np.nan)
    for j, fj in enumerate(np.atleast_1d(frequencies)):
        col = ec[:, j]
        ok = np.isfinite(col) & (col >= 0)
        if ok.any():
            out[ok, j] = E.forward_ppm_batch([fj], col[ok][:, None], sensor=sensor).imag[:, 0]
    return out


def emagpy_csv(
    x: np.ndarray,
    y: np.ndarray,
    frequencies,
    ec_ms_m: np.ndarray,
    sensor: E.Sensor = E.GEM2,
    errors_ms_m: np.ndarray | None = None,
) -> bytes:
    """
    Survey file for EMagPy: columns x, y, elevation, then one apparent
    conductivity column per frequency named HCP{separation}f{freq}h{height}
    (EMagPy coil-name convention; values in mS/m), optional *_err columns.
    EMagPy models a plain loop pair: the GEM-2 bucking coil is not modelled.
    """
    ec = np.atleast_2d(np.asarray(ec_ms_m, dtype=float))
    out = {"x": np.asarray(x, float), "y": np.asarray(y, float), "elevation": np.zeros(len(x))}
    names = []
    for j, fj in enumerate(np.atleast_1d(frequencies)):
        name = f"HCP{sensor.separation:g}f{float(fj):g}h{sensor.height:g}"
        names.append(name)
        out[name] = ec[:, j]
    if errors_ms_m is not None:
        err = np.broadcast_to(np.asarray(errors_ms_m, dtype=float), ec.shape)
        for j, name in enumerate(names):
            out[f"{name}_err"] = err[:, j]
    return pd.DataFrame(out).to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Profiles -> stations, results -> tables and figures
# ---------------------------------------------------------------------------


@dataclass
class StationData:
    distance: np.ndarray         # (S,) m
    frequencies: np.ndarray      # (F,) Hz
    quadrature: np.ndarray       # (S, F) ppm
    noise: np.ndarray | None     # (F,) ppm, measured between passes; None if unknown
    source: str                  # "Q" or "EC"


def _mean_profile_table(output_data: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Mean profiles of all channels on their joint distance grid: (distance, labels, values (D, F))."""
    profiles = {}
    for label, df in output_data.items():
        prof = df.mean(axis=1, skipna=True)
        prof.index = np.round(prof.index.to_numpy(dtype=float), 9)
        profiles[label] = prof
    table = pd.concat(profiles, axis=1).sort_index()
    return table.index.to_numpy(dtype=float), list(table.columns), table.to_numpy(dtype=float)


def station_data(
    output_data: dict[str, dict[str, pd.DataFrame]],
    scores: dict[str, dict[str, dict]],
    station_step: float,
    sensor: E.Sensor = E.GEM2,
) -> StationData:
    """
    Stations every `station_step` metres along the mean profiles. Uses the
    quadrature channels when present, otherwise EC converted to quadrature
    (ec_to_quadrature). The measured noise is the between-pass sigma of each
    frequency (None for a frequency scored from a single pass).
    """
    source = "Q" if output_data.get("Q") else "EC"
    channels = output_data.get(source) or {}
    if not channels:
        raise ValueError("Inversion needs Q (quadrature) or EC channels.")
    distance, labels, values = _mean_profile_table(channels)
    freqs = np.array([float(lb[:-2]) for lb in labels])
    order = np.argsort(freqs)
    freqs, values = freqs[order], values[:, order]
    labels = [labels[i] for i in order]
    targets = np.arange(distance[0], distance[-1] + 1e-9, station_step)
    keep = np.unique(np.abs(distance[:, None] - targets[None, :]).argmin(axis=0))
    distance, values = distance[keep], values[keep]

    noise = np.array([
        scores.get(source, {}).get(lb, {}).get("mean_std", np.nan)
        if scores.get(source, {}).get(lb, {}).get("noise_method") == "between-trace" else np.nan
        for lb in labels
    ])
    if source == "EC":
        q = ec_to_quadrature(freqs, values, sensor)
        # noise in mS/m -> ppm with the local slope dQ/dEC at the median EC
        med = np.nanmedian(values, axis=0, keepdims=True)
        slope = (ec_to_quadrature(freqs, med * 1.01, sensor) - ec_to_quadrature(freqs, med, sensor)) / (0.01 * med)
        noise = noise * np.abs(slope[0])
        values = q
    return StationData(
        distance=distance, frequencies=freqs, quadrature=values,
        noise=noise if np.isfinite(noise).any() else None, source=source,
    )


def model_table(result: InversionResult, distance: np.ndarray) -> pd.DataFrame:
    """One row per station and layer: distance, depth from/to (m), EC (mS/m)."""
    top = result.grid.depth_top
    bottom = np.concatenate([top[1:], [np.inf]])
    rows = []
    for d, model in zip(distance, result.log_sigma):
        for z0, z1, ls in zip(top, bottom, model):
            rows.append({"Distance (m)": d, "Depth from (m)": z0, "Depth to (m)": z1,
                         "EC (mS/m)": 1000.0 * 10.0 ** ls})
    return pd.DataFrame(rows)


def make_section_figure(result: InversionResult, distance: np.ndarray, title: str):
    """Distance x depth section of log10 EC (mS/m); the half-space is drawn 20 % below the last layer."""
    import matplotlib.pyplot as plt

    top = result.grid.depth_top
    edges_z = np.concatenate([top, [top[-1] * 1.2 if top[-1] > 0 else 1.0]])
    if len(distance) > 1:
        mid = 0.5 * (distance[1:] + distance[:-1])
        edges_x = np.concatenate([[2 * distance[0] - mid[0]], mid, [2 * distance[-1] - mid[-1]]])
    else:
        edges_x = np.array([distance[0] - 0.5, distance[0] + 0.5])
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(edges_x, edges_z, (result.log_sigma + 3.0).T, cmap="viridis", shading="flat")
    ax.invert_yaxis()
    fig.colorbar(mesh, ax=ax, label="log₁₀ EC (mS/m)")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{title} — χ² {result.chi2:.2f}, {result.iterations} iterations")
    fig.tight_layout()
    return fig


def emagpy_from_table(
    table: pd.DataFrame, sensor: E.Sensor = E.GEM2, errors_ms_m: dict[str, float] | None = None
) -> bytes:
    """
    EMagPy survey file from a prepared GEM table: every reading with its
    coordinates in metres (projected about the survey centroid when the file
    has degrees; along-line distance and y = 0 when it has no coordinates)
    and the EC channels in frequency order. errors_ms_m maps EC column names
    to 1-sigma errors.
    """
    found = gem_io.find_channels(table.columns)["EC"]
    if not found:
        raise ValueError("The EMagPy export needs EC channels.")
    labels = sorted(found, key=lambda lb: float(lb[:-2]))
    try:
        x, y, _ = corrections.xy_metres(table)
    except ctr.ContouringError:
        x = pd.to_numeric(table[pipeline.DISTANCE_COL], errors="coerce").to_numpy(dtype=float)
        y = np.zeros(len(table))
    ec = table[[found[lb] for lb in labels]].apply(pd.to_numeric, errors="coerce").to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    err = None
    if errors_ms_m:
        err = np.array([errors_ms_m.get(found[lb], np.nan) for lb in labels])
        if not np.isfinite(err).all():
            err = None
    return emagpy_csv(x[keep], y[keep], [float(lb[:-2]) for lb in labels], ec[keep], sensor, err)
