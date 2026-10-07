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


