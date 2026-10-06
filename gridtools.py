"""
Operations on gridded maps: NaN-aware filters, summary statistics, survey
merging with edge matching, difference maps, lateral deconvolution of the
GEM-2 footprint, and georeferenced exports (UTM, .prj, GeoTIFF).

Pure functions only (no Streamlit).
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

import contouring as ctr
import corrections
import emphysics as E

# ---------------------------------------------------------------------------
# NaN-aware filters (normalised convolution: blanks neither spread nor bias)
# ---------------------------------------------------------------------------


def _nan_uniform(z: np.ndarray, size: int) -> np.ndarray:
    """Mean of the finite values in a size x size window; NaN where z is NaN."""
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(z)
    num = ndimage.uniform_filter(np.where(ok, z, 0.0), size=size, mode="nearest")
    den = ndimage.uniform_filter(ok.astype(float), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(ok, out, np.nan)


def lowpass(z: np.ndarray, size: int = 3) -> np.ndarray:
    """Moving-average smoothing over size x size cells."""
    return _nan_uniform(z, size)


def highpass(z: np.ndarray, size: int = 15) -> np.ndarray:
    """Removes the regional trend: z minus its size x size moving average."""
    return np.asarray(z, dtype=float) - _nan_uniform(z, size)


def despike(z: np.ndarray, size: int = 3, threshold: float = 3.0) -> tuple[np.ndarray, int]:
    """
    Replaces cells that differ from the local median by more than
    threshold x 1.4826 x (median absolute deviation of all such differences)
    with that local median. Returns (grid, number of cells replaced).
    """
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(z)
    filled = np.where(ok, z, np.nanmedian(z))
    med = ndimage.median_filter(filled, size=size, mode="nearest")
    dev = np.abs(z - med)
    scale = 1.4826 * np.nanmedian(dev[ok])
    if not np.isfinite(scale) or scale == 0:
        return z.copy(), 0
    spikes = ok & (dev > threshold * scale)
    out = np.where(spikes, med, z)
    return out, int(spikes.sum())


def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Count, mean, standard deviation, min, 2nd/50th/98th percentile and max of finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values")
    p2, p50, p98 = np.percentile(v, [2, 50, 98])
    return {
        "n": int(v.size), "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()), "p2": float(p2), "median": float(p50),
        "p98": float(p98), "max": float(v.max()),
    }


