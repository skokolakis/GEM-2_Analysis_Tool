"""
Preparation of a raw GEM table before profiles and maps are built: quality
flags, corrections and filters, and distance along each line.

Pure functions only (no Streamlit). Every step reports what it changed in
plain-language messages that the app shows as warnings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import corrections
import gem_io

DISTANCE_COL = gem_io.DISTANCE_COL   # column added by prepare_gem_table
READING_ORDER_METHODS = ("sample", "markers")   # distances that count every logged reading


@dataclass(frozen=True)
class PrepSettings:
    """Options that change the prepared table (hashable: used as a cache key)."""
    drop_flagged: bool = True
    distance_method: str = "Y"        # key of gem_io.DISTANCE_METHODS
    distance_spacing: float = 1.0     # m, for "sample" and "markers"
    coord_mode: str = "auto"          # "auto" | "metres" | "degrees"
    corrections: corrections.CorrectionSettings = corrections.CorrectionSettings()


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

    if prep.corrections.active:
        df, steps = corrections.apply_corrections(df, prep.corrections)
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
