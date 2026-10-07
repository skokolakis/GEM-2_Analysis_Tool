"""
Interpretation aids: redundancy-aware frequency selection, magnetic
viscosity from two frequencies, and EMI anomaly spectra.

Pure functions only (no Streamlit).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import corrections
import emphysics as E
import gem_io

MAX_FREQUENCIES = 3            # Vilhelmsen & Døssing (2022): more frequencies, more noise each
REDUNDANT_CORRELATION = 0.95   # |r| above which two frequencies carry the same information

VISCOSITY_COLUMN = "MagViscosity[1/1000]"
QDIFF_EC_COLUMN = "ECqdiff[mS/m]"


# ---------------------------------------------------------------------------
# Frequency selection
# ---------------------------------------------------------------------------


def profile_correlation(output_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pearson correlation between the mean profiles of every pair of channels (pairwise-complete)."""
    profiles = {}
    for label, df in output_data.items():
        prof = df.mean(axis=1, skipna=True)
        prof.index = np.round(prof.index.to_numpy(dtype=float), 9)
        profiles[label] = prof
    return pd.concat(profiles, axis=1).sort_index().corr()


def select_frequencies(
    scores: dict[str, dict],
    correlation: pd.DataFrame,
    max_count: int = MAX_FREQUENCIES,
    max_correlation: float = REDUNDANT_CORRELATION,
) -> pd.DataFrame:
    """
    Greedy selection: frequencies in descending score order are kept unless
    their mean profile correlates (|r| > max_correlation) with one already
    kept, up to max_count. Blank scores are never selected.
    Returns one row per frequency: Frequency, Score, Selected, Reason.
    """
    ranked = sorted(
        scores.items(),
        key=lambda kv: (np.isfinite(kv[1]["score"]), np.nan_to_num(kv[1]["score"])),
        reverse=True,
    )
    chosen: list[str] = []
    rows = []
    for name, sc in ranked:
        if not np.isfinite(sc["score"]):
            rows.append((name, sc["score"], False, "no score"))
            continue
        twin = max(
            ((other, abs(correlation.loc[name, other])) for other in chosen
             if np.isfinite(correlation.loc[name, other])),
            key=lambda t: t[1], default=None,
        )
        if twin is not None and twin[1] > max_correlation:
            rows.append((name, sc["score"], False, f"redundant with {twin[0]} (|r| = {twin[1]:.3f})"))
        elif len(chosen) >= max_count:
            rows.append((name, sc["score"], False, f"beyond the {max_count} best"))
        else:
            chosen.append(name)
            rows.append((name, sc["score"], True, "selected"))
    return pd.DataFrame(rows, columns=["Frequency", "Score", "Selected", "Reason"])


