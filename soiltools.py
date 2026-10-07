"""
Soil-science tools driven by the EMI survey: response-surface sampling
design, calibration of soil properties against apparent conductivity, and
management zones by fuzzy c-means.

Pure functions only (no Streamlit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

import corrections

DESIGN_POOL = 20                 # candidates (nearest in feature space) per design target
INNER_RING, OUTER_RING = 1.0, 1.75   # design radii in standardised principal-component units


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def standardise(features: np.ndarray) -> np.ndarray:
    """Columns scaled to mean 0, standard deviation 1 (constant columns become 0)."""
    f = np.asarray(features, dtype=float)
    std = f.std(axis=0, ddof=1)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (f - f.mean(axis=0)) / std


def principal_scores(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Standardised principal-component scores (each component has unit variance)."""
    z = standardise(features)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    k = min(n_components, z.shape[1])
    scores = u[:, :k] * s[:k]
    sd = scores.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return scores / sd


# ---------------------------------------------------------------------------
# Sampling design
# ---------------------------------------------------------------------------


def design_targets(n_sites: int, n_components: int) -> np.ndarray:
    """
    Target points in standardised principal-component space: normal quantiles
    for one component; for two, the centre plus an inner ring (radius 1) and
    an outer ring (radius 1.75) at staggered angles, so the design spans the
    centre and the extremes of the response surface.
    """
    if n_sites < 1:
        raise ValueError("Need at least one sampling site.")
    if n_components == 1:
        return norm.ppf((np.arange(n_sites) + 0.5) / n_sites)[:, None]
    inner = (n_sites - 1) // 2
    outer = n_sites - 1 - inner
    pts = [(0.0, 0.0)]
    for k in range(inner):
        a = 2 * math.pi * k / max(inner, 1)
        pts.append((INNER_RING * math.cos(a), INNER_RING * math.sin(a)))
    for k in range(outer):
        a = 2 * math.pi * (k + 0.5) / max(outer, 1)
        pts.append((OUTER_RING * math.cos(a), OUTER_RING * math.sin(a)))
    return np.array(pts[:n_sites])


def sampling_design(xy: np.ndarray, features: np.ndarray, n_sites: int = 12) -> pd.DataFrame:
    """
    Sensor-directed sampling sites in the spirit of response-surface sampling
    (Lesch, 2005; ESAP-RSSD): for every design target the candidates closest
    to it in principal-component space form a pool, and the one farthest
    from the sites already chosen is taken, which spreads the sites in space.
    Returns one row per site: row (index into xy), x, y, pc1 [, pc2], target.
    """
    xy = np.asarray(xy, dtype=float)
    f = np.asarray(features, dtype=float)
    ok = np.all(np.isfinite(f), axis=1) & np.all(np.isfinite(xy), axis=1)
    rows = np.nonzero(ok)[0]
    if len(rows) < n_sites:
        raise ValueError(f"Only {len(rows)} usable readings for {n_sites} sites.")
    pcs = principal_scores(f[rows], n_components=2 if f.shape[1] > 1 else 1)
    targets = design_targets(n_sites, pcs.shape[1])
    tree = cKDTree(pcs)
    chosen: list[int] = []
    for t in targets:
        _, pool = tree.query(t, k=min(len(rows), DESIGN_POOL + len(chosen)))
        pool = [int(p) for p in np.atleast_1d(pool) if int(p) not in chosen][:DESIGN_POOL]
        if chosen:
            spread = cKDTree(xy[rows[chosen]]).query(xy[rows[pool]])[0]
            pick = pool[int(np.argmax(spread))]
        else:
            pick = pool[0]
        chosen.append(pick)
    out = pd.DataFrame({"row": rows[chosen], "x": xy[rows[chosen], 0], "y": xy[rows[chosen], 1]})
    for j in range(pcs.shape[1]):
        out[f"pc{j + 1}"] = pcs[chosen, j]
    out["target"] = [", ".join(f"{v:+.2f}" for v in t) for t in targets]
    return out


# ---------------------------------------------------------------------------
# Calibration of a soil property against apparent conductivity
# ---------------------------------------------------------------------------


@dataclass
class PropertyModel:
    """Least-squares model property ~ conductivity channels (+ trend surface)."""
    channels: list[str]
    log_property: bool
    log_channels: bool
    trend: bool
    coefficients: pd.Series      # intercept, channels, [x, y]
    r2: float
    rmse: float                  # on the fitted (possibly log) scale
    n: int
    observed: np.ndarray
    fitted: np.ndarray


def _design_matrix(values: np.ndarray, xy: np.ndarray | None, log_channels: bool, trend: bool) -> np.ndarray:
    v = np.log(values) if log_channels else values
    cols = [np.ones(len(v)), *v.T]
    if trend:
        cols += [xy[:, 0], xy[:, 1]]
    return np.column_stack(cols)


def fit_property_model(
    table: pd.DataFrame,
    samples: pd.DataFrame,
    property_col: str,
    channels: list[str],
    radius: float,
    log_property: bool = True,
    trend: bool = False,
) -> PropertyModel:
    """
    Regression of a soil property measured at sampling sites on the apparent
    conductivity there (Lesch et al., 1995; ESAP-Calibrate): each sample takes
    the median of the survey readings within `radius` metres. Conductivities
    enter as logarithms when all are positive; with log_property the property
    is fitted on a log scale; trend adds x and y terms.
    """
    x, y, origin = corrections.xy_metres(table)
    sx, sy, _ = corrections.xy_metres(samples, origin)
    ok = np.isfinite(x) & np.isfinite(y)
    rows = np.nonzero(ok)[0]
    near = cKDTree(np.column_stack([x[ok], y[ok]])).query_ball_point(np.column_stack([sx, sy]), r=radius)
    values = np.array([
        [np.nanmedian(pd.to_numeric(table[c], errors="coerce").to_numpy()[rows[i]]) if i else np.nan
         for c in channels]
        for i in near
    ], dtype=float).reshape(len(samples), len(channels))
    prop = pd.to_numeric(samples[property_col], errors="coerce").to_numpy(dtype=float)
    use = np.all(np.isfinite(values), axis=1) & np.isfinite(prop)
    if log_property:
        use &= prop > 0
    n_terms = 1 + len(channels) + (2 if trend else 0)
    if use.sum() < n_terms + 2:
        raise ValueError(f"Only {int(use.sum())} samples matched readings; need at least {n_terms + 2}.")
    log_channels = bool(np.all(values[use] > 0))
    a = _design_matrix(values[use], np.column_stack([sx, sy])[use], log_channels, trend)
    target = np.log(prop[use]) if log_property else prop[use]
    coef, *_ = np.linalg.lstsq(a, target, rcond=None)
    fitted = a @ coef
    resid = target - fitted
    r2 = 1.0 - float(resid @ resid) / float(((target - target.mean()) ** 2).sum())
    names = ["intercept", *[f"ln {c}" if log_channels else c for c in channels]] + (["x", "y"] if trend else [])
    return PropertyModel(
        channels=list(channels), log_property=log_property, log_channels=log_channels, trend=trend,
        coefficients=pd.Series(coef, index=names), r2=r2,
        rmse=float(np.sqrt(np.mean(resid ** 2))), n=int(use.sum()), observed=target, fitted=fitted,
    )


def predict_property(model: PropertyModel, table: pd.DataFrame) -> np.ndarray:
    """Predicted property at every reading (back-transformed from the log scale when fitted on it)."""
    values = table[model.channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xy = None
    if model.trend:
        x, y, _ = corrections.xy_metres(table)
        xy = np.column_stack([x, y])
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = _design_matrix(values, xy, model.log_channels, model.trend) @ model.coefficients.to_numpy()
    return np.exp(pred) if model.log_property else pred


# ---------------------------------------------------------------------------
# Management zones (fuzzy c-means)
# ---------------------------------------------------------------------------


def fuzzy_cmeans(
    data: np.ndarray, c: int, m: float = 1.3, max_iter: int = 300, tol: float = 1e-6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy c-means (Bezdek, 1981) with Euclidean distance on the given
    (already standardised) data. Returns (memberships (n, c), centres (c, k)).
    """
    x = np.asarray(data, dtype=float)
    if c < 2 or c >= len(x):
        raise ValueError("Need 2 <= number of zones < number of points.")
    rng = np.random.default_rng(seed)
    u = rng.dirichlet(np.ones(c), size=len(x))
    for _ in range(max_iter):
        w = u ** m
        centres = (w.T @ x) / w.sum(axis=0)[:, None]
        d = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        d = np.fmax(d, 1e-12)
        ratio = (d[:, :, None] / d[:, None, :]) ** (2.0 / (m - 1.0))
        new_u = 1.0 / ratio.sum(axis=2)
        if np.max(np.abs(new_u - u)) < tol:
            u = new_u
            break
        u = new_u
    order = np.argsort(centres[:, 0])              # zone 1 = lowest first feature
    return u[:, order], centres[order]


def fuzziness_performance_index(u: np.ndarray) -> float:
    """
    FPI = c / (c - 1) x (1 - sum(u^2) / n) (as used by Fridgen et al., 2004):
    0 for distinct (crisp) zones, 1 when memberships are shared equally.
    """
    n, c = u.shape
    return float(c / (c - 1.0) * (1.0 - np.sum(u ** 2) / n))


def modified_partition_entropy(u: np.ndarray) -> float:
    """MPE = -sum(u log u) / (n log c) (Boydell & McBratney, 2002); 0 = organised, 1 = disorganised."""
    n, c = u.shape
    uu = np.clip(u, 1e-300, 1.0)
    return float(-np.sum(uu * np.log(uu)) / (n * math.log(c)))


def zone_indices(data: np.ndarray, c_range=range(2, 7), m: float = 1.3) -> pd.DataFrame:
    """FPI and MPE for each number of zones; the lowest values suggest the number to use."""
    rows = []
    for c in c_range:
        if c >= len(data):
            break
        u, _ = fuzzy_cmeans(data, c, m)
        rows.append({"Zones": c, "FPI": fuzziness_performance_index(u),
                     "MPE": modified_partition_entropy(u)})
    return pd.DataFrame(rows)
