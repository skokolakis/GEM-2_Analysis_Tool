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


# ---------------------------------------------------------------------------
# Lateral footprint of the GEM-2 and deconvolution
# ---------------------------------------------------------------------------


def lin_sensitivity(x, y, z, r: float, height: float) -> np.ndarray:
    """
    Unnormalised low-induction-number sensitivity of a horizontal co-planar
    pair (Tx at the origin, Rx at (r, 0), both `height` above ground) to
    conductivity at (x, y, depth z): the dot product of the two dipoles'
    quasi-static electric fields, which circle each vertical dipole
    (reciprocity / Born approximation).
    """
    zz = np.asarray(z, dtype=float) + height
    rt = np.sqrt(x ** 2 + y ** 2 + zz ** 2)
    rr = np.sqrt((x - r) ** 2 + y ** 2 + zz ** 2)
    return (x * (x - r) + y * y) / (rt ** 3 * rr ** 3)


def _depth_nodes(height: float, max_depth: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Depth samples (log-spaced, finer near the surface) and trapezoid weights."""
    z = np.concatenate([[0.0], np.geomspace(1e-3, max_depth, n)])
    w = np.zeros_like(z)
    dz = np.diff(z)
    w[:-1] += dz / 2
    w[1:] += dz / 2
    return z, w


def footprint(
    cell: float, sensor: E.Sensor = E.GEM2, angle_deg: float = 0.0, half_width: float | None = None
) -> np.ndarray:
    """
    Lateral footprint of the sensor on a map grid with spacing `cell`:
    depth-integrated sensitivity of Rx minus bucking coil, each scaled to its
    low-induction-number total, centred on the Tx-Rx midpoint with the coil
    axis at `angle_deg` from the grid x axis. Sums to 1.
    """
    s, h = sensor.separation, sensor.height
    if half_width is None:
        half_width = 3.0 * s + 4.0 * h
    n = int(math.ceil(half_width / cell))
    ax = np.arange(-n, n + 1) * cell
    gx, gy = np.meshgrid(ax, ax)
    a = math.radians(angle_deg)
    # grid -> sensor frame (coil axis along +x', Tx at x' = -s/2)
    xs = gx * math.cos(a) + gy * math.sin(a) + s / 2
    ys = -gx * math.sin(a) + gy * math.cos(a)
    z, wz = _depth_nodes(h, max_depth=10.0 * s + 10.0 * h)

    def radius_term(r):
        k = sum(w * lin_sensitivity(xs, ys, zi, r, h) for zi, w in zip(z, wz))
        total = r * r / (4.0 * math.sqrt(4.0 * (h / r) ** 2 + 1.0))
        return k / k.sum() * total

    k = radius_term(s)
    if sensor.bucking:
        k = k - radius_term(sensor.bucking)
    return k / k.sum()


def line_axis_angle(df: pd.DataFrame) -> float:
    """Survey-line direction, degrees counter-clockwise from the map x axis in [0, 180)."""
    x, y, _ = corrections.xy_metres(df)
    angles = []
    for idx in df.groupby("Line", sort=False).indices.values():
        ok = idx[np.isfinite(x[idx]) & np.isfinite(y[idx])]
        if len(ok) >= 2:
            angles.append(math.degrees(math.atan2(y[ok[-1]] - y[ok[0]], x[ok[-1]] - x[ok[0]])) % 180.0)
    if not angles:
        return 0.0
    # median of axial angles via doubled-angle vectors
    a = np.radians(2 * np.asarray(angles))
    return float(math.degrees(math.atan2(np.median(np.sin(a)), np.median(np.cos(a)))) / 2 % 180.0)


MAP_FILTERS = ("none", "despike", "low-pass", "high-pass")


def process_map(
    z: np.ndarray,
    cell: float,
    kind: str = "none",
    size: int = 3,
    threshold: float = 4.0,
    deconvolve_footprint: bool = False,
    regularisation: float = 1e-2,
    angle_deg: float = 0.0,
    sensor: E.Sensor = E.GEM2,
) -> tuple[np.ndarray, list[str]]:
    """Optional footprint deconvolution, then one grid filter. Returns (grid, messages)."""
    msgs = []
    out = np.asarray(z, dtype=float)
    if deconvolve_footprint:
        out = deconvolve(out, footprint(cell, sensor, angle_deg), regularisation)
        msgs.append(
            f"Deconvolved the sensor footprint (coil axis {angle_deg:.0f}° from x, "
            f"regularisation {regularisation:g})."
        )
    if kind == "despike":
        out, n = despike(out, size, threshold)
        msgs.append(f"Despiked {n} cell(s).")
    elif kind == "low-pass":
        out = lowpass(out, size)
        msgs.append(f"Low-pass: moving average over {size} × {size} cells.")
    elif kind == "high-pass":
        out = highpass(out, size)
        msgs.append(f"High-pass: regional trend over {size} × {size} cells removed.")
    elif kind != "none":
        raise ValueError(f"Unknown map filter: {kind!r}")
    return out, msgs


def _fill_nan(z: np.ndarray) -> np.ndarray:
    """Fills NaN cells with the value of the nearest finite cell."""
    ok = np.isfinite(z)
    if ok.all():
        return z
    idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def deconvolve(z: np.ndarray, kernel: np.ndarray, regularisation: float = 1e-2) -> np.ndarray:
    """
    Tikhonov (Wiener-type) deconvolution of a map by a footprint kernel:
    Z * conj(K) / (|K|^2 + regularisation * max|K|^2) in the Fourier domain.
    Blank cells are filled from their nearest neighbour for the transform and
    blanked again; edges are mirror-padded by the kernel half-width.
    """
    z = np.asarray(z, dtype=float)
    blank = ~np.isfinite(z)
    pad = kernel.shape[0] // 2
    work = np.pad(_fill_nan(z), pad, mode="reflect")
    mean = work.mean()
    shape = work.shape
    kpad = np.zeros(shape)
    kpad[: kernel.shape[0], : kernel.shape[1]] = kernel
    kpad = np.roll(kpad, (-(kernel.shape[0] // 2), -(kernel.shape[1] // 2)), axis=(0, 1))
    K = np.fft.rfft2(kpad)
    Z = np.fft.rfft2(work - mean)
    lam = regularisation * float(np.max(np.abs(K)) ** 2)
    out = np.fft.irfft2(Z * np.conj(K) / (np.abs(K) ** 2 + lam), s=shape) + mean
    out = out[pad: pad + z.shape[0], pad: pad + z.shape[1]]
    return np.where(blank, np.nan, out)


def convolve(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Forward model of deconvolve (same padding), for tests and previews."""
    pad = kernel.shape[0] // 2
    work = np.pad(np.asarray(z, dtype=float), pad, mode="reflect")
    return ndimage.convolve(work, kernel[::-1, ::-1], mode="nearest")[
        pad: pad + z.shape[0], pad: pad + z.shape[1]
    ]


