"""
Preparation of a raw GEM table before profiles and maps are built: quality
flags, corrections and filters, apparent conductivity and susceptibility
from in-phase / quadrature, and distance along each line.

Pure functions only (no Streamlit). Every step reports what it changed in
plain-language messages that the app shows as warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import corrections
import emphysics
import gem_io
import interpret

DISTANCE_COL = gem_io.DISTANCE_COL   # column added by prepare_gem_table
READING_ORDER_METHODS = ("sample", "markers")   # distances that count every logged reading
HEIGHT_STEP = 0.02                                # m; readings within a step share one conversion
OFFSETS_NOT_IN_EC = (
    "The calibration offsets were subtracted from the I / Q channels only: the EC and MS "
    "columns of this file are still as exported, so EC / MS profiles, scores and maps are not "
    "calibrated. Turn on 'Recompute EC / MS from I / Q' (Sensor geometry, advanced settings) "
    "to calibrate them."
)


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
    viscosity_pair: tuple[float, float] | None = None   # Hz; adds magnetic-viscosity AUX channels


def _halfspace(f: float, i: np.ndarray, q: np.ndarray, sensor: emphysics.Sensor,
               heights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """halfspace_from_ppm_batch with the sensor height of each reading (HEIGHT_STEP bins) when given."""
    if heights is None:
        return emphysics.halfspace_from_ppm_batch(f, i, q, sensor)
    sigma, kappa = np.full(i.shape, np.nan), np.full(i.shape, np.nan)
    level = np.round(heights / HEIGHT_STEP)
    for lv in np.unique(level[np.isfinite(level) & (level >= 0)]):
        rows = level == lv
        sensor_h = replace(sensor, height=float(lv * HEIGHT_STEP))
        sigma[rows], kappa[rows] = emphysics.halfspace_from_ppm_batch(f, i[rows], q[rows], sensor_h)
    return sigma, kappa


def recompute_ec_ms(
    df: pd.DataFrame, sensor: emphysics.Sensor, height_column: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    """
    Writes EC{f}Hz[mS/m] and MSusc{f}Hz[1/1000] for every frequency with both
    I_{f}Hz and Q_{f}Hz columns: the homogeneous half-space that reproduces
    each reading (Huang & Won, 2000), for the given coil geometry and height.
    With a height column, each reading is converted at its own sensor height
    (to HEIGHT_STEP), which removes the height effect from EC / MS by the
    physics rather than by a fitted curve; readings without a height are
    left blank. Where the file already has EC / MS, the messages compare them
    with the recomputed values: a median ratio far from 1, or MS running the
    opposite way, points to a different geometry, height or I/Q sign convention.
    """
    out = df.copy()
    found = gem_io.find_channels(df.columns)
    heights = None
    if height_column:
        if height_column not in df.columns:
            raise ValueError(f"Altitude column '{height_column}' not found.")
        heights = pd.to_numeric(df[height_column], errors="coerce").to_numpy(dtype=float)
    done, blank, ratios, flipped = [], 0, [], []
    for label, q_col in found["Q"].items():
        i_col = found["I"].get(label)
        if i_col is None:
            continue
        f = float(label[:-2])
        sigma, kappa = _halfspace(
            f,
            pd.to_numeric(df[i_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float),
            sensor, heights,
        )
        ec_col, ms_col = gem_io.channel_column("EC", label), gem_io.channel_column("MS", label)
        ec, ms = sigma * 1000.0, kappa * 1000.0
        if ec_col in df.columns:
            old = pd.to_numeric(df[ec_col], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(old) & np.isfinite(ec) & (old > 0)
            if both.sum() >= 3:
                ratios.append(f"{label} ×{np.median(ec[both] / old[both]):.3g}")
        if ms_col in df.columns:
            old = pd.to_numeric(df[ms_col], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(old) & np.isfinite(ms)
            if (both.sum() >= 3 and np.std(old[both]) > 0 and np.std(ms[both]) > 0
                    and np.corrcoef(old[both], ms[both])[0, 1] < 0):
                flipped.append(label)
        out[ec_col] = ec
        out[ms_col] = ms
        done.append(label)
        blank += int(np.isnan(sigma).sum())
    if not done:
        return out, ["Recompute from I/Q: no frequency has both I_ and Q_ columns."]
    sep = f"{sensor.separation:g} m" + (f", bucking {sensor.bucking:g} m" if sensor.bucking else "")
    where = (f"each reading at its own height from '{height_column}'" if heights is not None
             else f"height {sensor.height:g} m")
    msgs = [f"EC and MS recomputed from I/Q for {', '.join(done)} (coils {sep}, {where})."]
    if heights is not None:
        msgs.append("The height effect is removed by the conversion itself, so the fitted "
                    "sensor-height correction is not applied.")
    elif not emphysics.susceptibility_resolved(sensor):
        msgs.append(
            f"At {sensor.height:g} m the in-phase of this coil geometry barely depends on "
            "susceptibility, so EC comes from the quadrature alone and MS is left blank."
        )
    if blank:
        msgs.append(f"{blank} reading(s) have no EC / MS: quadrature ≤ 0, or no half-space reproduces them.")
    if ratios:
        msgs.append(
            "Recomputed / exported EC (median): " + ", ".join(ratios) + ". A ratio far from 1 points "
            "to a different coil geometry, height or calibration than the instrument's."
        )
    if flipped:
        msgs.append(
            "Recomputed MS runs opposite to the exported MS for " + ", ".join(flipped)
            + ": check the sign convention of the in-phase."
        )
    return out, msgs


def prepare_gem_table(raw: pd.DataFrame, prep: PrepSettings | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (prepared copy of the table, messages). The prepared table has
    an extra DISTANCE_COL column with the distance of each reading along its
    line. Raises ValueError (ContouringError) when distances cannot be built.
    """
    prep = prep or PrepSettings()
    df, messages = gem_io.normalise_columns(raw)     # other export layouts -> WinGEM columns
    df = df.copy()

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
        # base-station lines are needed by the drift correction, which removes them itself
        left_out = set(prep.exclude_lines) - set(prep.corrections.drift_lines)
        drop = gem_io.line_labels(df["Line"]).isin(left_out).to_numpy()
        df = df.loc[~drop].reset_index(drop=True)
        messages.append(f"Left out {int(drop.sum())} reading(s) of line(s) {', '.join(prep.exclude_lines)}.")

    if prep.recompute_from_iq:
        # I/Q corrections, then EC / MS from I / Q (at each reading's height when there is a
        # height column, instead of the fitted height correction), then the corrections of EC / MS
        height_column = prep.corrections.altitude_column
        readings = replace(prep.corrections, altitude_column="")
        if readings.active:
            df, steps = corrections.apply_corrections(df, readings, stage="readings")
            messages.extend(steps)
        df, steps = recompute_ec_ms(df, prep.sensor, height_column)
        messages.extend(steps)
        if prep.corrections.active:
            df, steps = corrections.apply_corrections(df, prep.corrections, stage="derived")
            messages.extend(steps)
    elif prep.corrections.active:
        df, steps = corrections.apply_corrections(df, prep.corrections)
        messages.extend(steps)
        if prep.corrections.iq_offsets and corrections.channel_columns(df, ("EC", "MS")):
            messages.append(OFFSETS_NOT_IN_EC)

    if prep.viscosity_pair:
        try:
            df, steps = interpret.add_viscosity_columns(df, prep.viscosity_pair, prep.sensor)
        except ValueError as exc:                # the rest of the file is still usable
            steps = [f"Magnetic viscosity skipped: {exc}"]
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
