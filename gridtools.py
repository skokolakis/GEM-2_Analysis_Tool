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


# ---------------------------------------------------------------------------
# Merging surveys and difference maps
# ---------------------------------------------------------------------------


def edge_match_offset(
    ref_xy: np.ndarray, ref_v: np.ndarray, xy: np.ndarray, v: np.ndarray, tolerance: float
) -> tuple[float, int]:
    """
    Constant to add to survey (xy, v) so it matches the reference where they
    overlap: median of reference - survey over pairs closer than `tolerance`.
    Returns (offset, number of pairs); (0, 0) when they do not overlap.
    """
    d, i = cKDTree(ref_xy).query(xy, distance_upper_bound=tolerance)
    pair = np.isfinite(d)
    if not pair.any():
        return 0.0, 0
    return float(np.median(ref_v[i[pair]] - v[pair])), int(pair.sum())


def merge_tables(
    tables: list[pd.DataFrame], value_col: str, tolerance: float, match: bool = True
) -> tuple[pd.DataFrame, list[float]]:
    """
    Concatenates GEM tables into one survey. Line labels become
    "<survey>:<line>" so lines stay distinct. With match=True each table after
    the first is shifted by its edge-match offset (median difference to the
    tables before it where readings are closer than `tolerance` metres).
    All tables must use the same coordinate columns. Returns (table, offsets).
    """
    kinds = {ctr.find_coordinate_columns(t, "auto") for t in tables}
    if len(kinds) > 1:
        raise ctr.ContouringError("The surveys use different coordinate columns.")
    origin = None
    parts, offsets, ref_xy, ref_v = [], [], [], []
    for k, table in enumerate(tables):
        x, y, o = corrections.xy_metres(table, origin)
        origin = origin or o
        part = table.copy()
        part["Line"] = f"{k}:" + part["Line"].astype(str)
        v = pd.to_numeric(part[value_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        offset = 0.0
        if match and ref_xy:
            offset, _ = edge_match_offset(
                np.vstack(ref_xy), np.concatenate(ref_v),
                np.column_stack([x[ok], y[ok]]), v[ok], tolerance,
            )
            part[value_col] = v + offset
        ref_xy.append(np.column_stack([x[ok], y[ok]]))
        ref_v.append(v[ok] + offset)
        offsets.append(offset)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), offsets


