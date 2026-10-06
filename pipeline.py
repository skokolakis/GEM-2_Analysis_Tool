"""
Preparation of a raw GEM table before profiles and maps are built: quality
flags, corrections and filters, apparent conductivity and susceptibility
from in-phase / quadrature, and distance along each line.

Pure functions only (no Streamlit). Every step reports what it changed in
plain-language messages that the app shows as warnings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import corrections
import emphysics
import gem_io

DISTANCE_COL = gem_io.DISTANCE_COL   # column added by prepare_gem_table
READING_ORDER_METHODS = ("sample", "markers")   # distances that count every logged reading


@dataclass(frozen=True)
class PrepSettings:
    """Options that change the prepared table (hashable: used as a cache key)."""
    drop_flagged: bool = True
    exclude_lines: tuple[str, ...] = ()  # Line labels left out (e.g. calibration lines)
    distance_method: str = "Y"        # key of gem_io.DISTANCE_METHODS
    distance_spacing: float = 1.0     # m, for "sample" and "markers"
    coord_mode: str = "auto"          # "auto" | "metres" | "degrees"
    corrections: corrections.CorrectionSettings = corrections.CorrectionSettings()
    sensor: emphysics.Sensor = emphysics.GEM2
    recompute_from_iq: bool = False   # EC / MS from I / Q with the sensor geometry above


def recompute_ec_ms(df: pd.DataFrame, sensor: emphysics.Sensor) -> tuple[pd.DataFrame, list[str]]:
    """
    Writes EC{f}Hz[mS/m] and MSusc{f}Hz[1/1000] for every frequency with both
    I_{f}Hz and Q_{f}Hz columns: the homogeneous half-space that reproduces
    each reading (Huang & Won, 2000), for the given coil geometry and height.
    """
    out = df.copy()
    found = gem_io.find_channels(df.columns)
    done, blank = [], 0
    for label, q_col in found["Q"].items():
        i_col = found["I"].get(label)
        if i_col is None:
            continue
        f = float(label[:-2])
        sigma, kappa = emphysics.halfspace_from_ppm_batch(
            f,
            pd.to_numeric(df[i_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float),
            sensor,
        )
        out[gem_io.channel_column("EC", label)] = sigma * 1000.0
        out[gem_io.channel_column("MS", label)] = kappa * 1000.0
        done.append(label)
        blank += int(np.isnan(sigma).sum())
    if not done:
        return out, ["Recompute from I/Q: no frequency has both I_ and Q_ columns."]
    sep = f"{sensor.separation:g} m" + (f", bucking {sensor.bucking:g} m" if sensor.bucking else "")
    msgs = [f"EC and MS recomputed from I/Q for {', '.join(done)} (coils {sep}, height {sensor.height:g} m)."]
    if blank:
        msgs.append(f"{blank} reading(s) with quadrature ≤ 0 have no EC / MS.")
    return out, msgs


def prepare_gem_table(raw: pd.DataFrame, prep: PrepSettings | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (prepared copy of the table, messages). The prepared table has
    an extra DISTANCE_COL column with the distance of each reading along its
    line. Raises ValueError (ContouringError) when distances cannot be built.
    """
    prep = prep or PrepSettings()
    messages: list[str] = []
    df = raw.copy()

    # Reading-order distances count every logged reading, so they are computed
    # before flagged, filtered or excluded readings are dropped.
    order_based = prep.distance_method in READING_ORDER_METHODS
    if order_based:
        df[DISTANCE_COL] = gem_io.along_track_distance(
            df, prep.distance_method, prep.distance_spacing, prep.coord_mode
        )

    if prep.drop_flagged:
        df, n_flagged = gem_io.drop_flagged_rows(df)
        if n_flagged:
            messages.append(
                f"Dropped {n_flagged} reading(s) with a non-zero Status flag "
                "(the instrument reports a problem such as ADC overload)."
            )

    if prep.exclude_lines:
        drop = gem_io.line_labels(df["Line"]).isin(prep.exclude_lines).to_numpy()
        df = df.loc[~drop].reset_index(drop=True)
        messages.append(f"Left out {int(drop.sum())} reading(s) of line(s) {', '.join(prep.exclude_lines)}.")

    if prep.corrections.active:
        df, steps = corrections.apply_corrections(df, prep.corrections)
        messages.extend(steps)

    if prep.recompute_from_iq:
        df, steps = recompute_ec_ms(df, prep.sensor)
        messages.extend(steps)

    if order_based:
        distance = df[DISTANCE_COL].to_numpy(dtype=float)
    else:
        distance = gem_io.along_track_distance(
            df, prep.distance_method, prep.distance_spacing, prep.coord_mode
        )
    if prep.distance_method != "Y":
        n_blank = int(np.isnan(distance).sum())
        if n_blank:
            messages.append(
                f"{n_blank} reading(s) have no distance with the "
                f"'{gem_io.DISTANCE_METHODS[prep.distance_method]}' method and are not used "
                "in profiles."
            )
    df[DISTANCE_COL] = distance
    return df, messages
