"""
Operations on gridded maps: NaN-aware filters, summary statistics, survey
merging with edge matching, difference maps and georeferenced exports (UTM,
.prj, GeoTIFF).

Pure functions only (no Streamlit).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

import contouring as ctr
import corrections

MIN_EDGE_PAIRS = 5             # overlapping readings needed before an edge-match offset is applied

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


def _fill_nan(z: np.ndarray) -> np.ndarray:
    """Fills NaN cells with the value of the nearest finite cell."""
    ok = np.isfinite(z)
    if ok.all():
        return z
    idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def despike(z: np.ndarray, size: int = 3, threshold: float = 3.0) -> tuple[np.ndarray, int]:
    """
    Replaces cells that differ from the local median by more than
    threshold x 1.4826 x (median absolute deviation of all such differences)
    with that local median. Returns (grid, number of cells replaced).
    """
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(z)
    if not ok.any():
        return z.copy(), 0
    filled = _fill_nan(z)                   # nearest values, so blanks do not pull edge medians
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
    Returns (offset, number of pairs); the offset is 0 with fewer than
    MIN_EDGE_PAIRS pairs.
    """
    if len(ref_xy) == 0 or len(xy) == 0:
        return 0.0, 0
    d, i = cKDTree(ref_xy).query(xy, distance_upper_bound=tolerance)
    pair = np.isfinite(d)
    if pair.sum() < MIN_EDGE_PAIRS:
        return 0.0, int(pair.sum())
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


MAP_FILTERS = ("none", "despike", "low-pass", "high-pass")


def process_map(
    z: np.ndarray,
    kind: str = "none",
    size: int = 3,
    threshold: float = 4.0,
) -> tuple[np.ndarray, list[str]]:
    """One grid filter. Returns (grid, messages)."""
    msgs = []
    out = np.asarray(z, dtype=float)
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


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 / UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def lonlat_to_epsg(lon: np.ndarray, lat: np.ndarray, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    """Projects WGS 84 lon/lat to the CRS `epsg` (metres)."""
    from pyproj import Transformer

    t = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def prj_wkt(epsg: int) -> bytes:
    """ESRI .prj contents (WKT1) for `epsg`, to sit next to an .asc grid."""
    from pyproj import CRS
    from pyproj.enums import WktVersion

    return CRS.from_epsg(epsg).to_wkt(WktVersion.WKT1_ESRI).encode()


GEOTIFF_NODATA = -9999.0


def geotiff_bytes(z: np.ndarray, x0: float, y0: float, cell: float, epsg: int | None) -> bytes:
    """
    Single-band float32 GeoTIFF. (x0, y0) is the lower-left corner of the
    lower-left cell, row 0 of `z` is the southernmost row (as in GridSpec);
    the file stores the northernmost row first. With epsg=None the file has
    pixel scale and tie point but no CRS.
    """
    from PIL import Image, TiffImagePlugin, TiffTags

    z = np.asarray(z, dtype=np.float32)
    ny, _ = z.shape
    data = np.where(np.isfinite(z), z, GEOTIFF_NODATA).astype(np.float32)[::-1]
    img = Image.fromarray(data, mode="F")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(cell), float(cell), 0.0)                       # ModelPixelScale
    ifd.tagtype[33550] = TiffTags.DOUBLE
    ifd[33922] = (0.0, 0.0, 0.0, float(x0), float(y0 + ny * cell), 0.0)  # ModelTiepoint
    ifd.tagtype[33922] = TiffTags.DOUBLE
    keys = [1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]                  # projected, PixelIsArea
    if epsg:
        keys[3] = 3
        keys += [3072, 0, 1, int(epsg)]                                # ProjectedCSTypeGeoKey
    ifd[34735] = tuple(keys)                                           # GeoKeyDirectory
    ifd.tagtype[34735] = TiffTags.SHORT
    ifd[42113] = f"{GEOTIFF_NODATA:g}"                                 # GDAL_NODATA
    ifd.tagtype[42113] = TiffTags.ASCII
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    return buf.getvalue()


def epsg_problem(epsg: int) -> str | None:
    """Why `epsg` cannot label metre X/Y (unknown, or not a projected CRS); None when it can."""
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    try:
        crs = CRS.from_epsg(int(epsg))
    except CRSError:
        return f"EPSG:{epsg} is not a known coordinate system; X/Y are treated as unreferenced."
    if not crs.is_projected:
        return (f"EPSG:{epsg} ({crs.name}) is not a projected system in metres; "
                "X/Y are treated as unreferenced.")
    return None
