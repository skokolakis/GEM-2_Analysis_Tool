"""
Streamlit panels for the physics tools: forward modelling for survey
planning, frequency information (skin depth, induction number, depth of
investigation) and multi-height calibration of I/Q offsets.

Numerical work lives in emphysics.py; the small table helpers here are pure
and tested on their own.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import contouring as ctr
import emphysics
import gem_io

DEFAULT_FREQUENCIES = "475, 1525, 5325, 18325, 63025"
PERMITTIVITY_FREQUENCY = 40_000.0     # Hz; above this, permittivity can reach the in-phase
PERMITTIVITY_CAPTION = (
    "Above about 40 kHz the in-phase response can include a dielectric-permittivity "
    "contribution, not only magnetic susceptibility (Benech et al., 2016)."
)
HEIGHT_BIN = 0.05                     # m, readings within a bin share one height level
LAYER_COLUMNS = ["Thickness (m)", "EC (mS/m)", "MS (10⁻³ SI)"]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def parse_frequencies(text: str) -> list[float]:
    """'475, 1525 5325' -> [475.0, 1525.0, 5325.0]; raises ValueError on bad entries."""
    parts = [p for p in text.replace(",", " ").split() if p]
    freqs = [float(p) for p in parts]
    if not freqs or any(f <= 0 for f in freqs):
        raise ValueError("Enter one or more positive frequencies in Hz.")
    return freqs


def layers_from_table(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer table (rows top to bottom; the last row is the half-space and its
    thickness is ignored) -> (sigma S/m, kappa SI, thicknesses m).
    """
    t = table[LAYER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    t = t.dropna(subset=["EC (mS/m)"]).reset_index(drop=True)
    if t.empty:
        raise ValueError("Enter at least one layer with an EC value.")
    thickness = t["Thickness (m)"].to_numpy(dtype=float)[:-1]
    if np.any(~np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("Every layer above the half-space needs a positive thickness.")
    sigma = t["EC (mS/m)"].to_numpy(dtype=float) / 1000.0
    kappa = t["MS (10⁻³ SI)"].fillna(0.0).to_numpy(dtype=float) / 1000.0
    if np.any(sigma < 0):
        raise ValueError("EC must not be negative.")
    return sigma, kappa, thickness


def model_response(freqs, table: pd.DataFrame, sensor: emphysics.Sensor) -> pd.DataFrame:
    """In-phase, quadrature and half-space apparent EC / MS of a layer table at each frequency."""
    sigma, kappa, thickness = layers_from_table(table)
    z = emphysics.forward_ppm(freqs, sigma, kappa, thickness, sensor)
    rows = []
    for f, zi in zip(freqs, z):
        sa, ka = emphysics.halfspace_from_ppm(f, zi.real, zi.imag, sensor) if zi.imag > 0 else (np.nan, np.nan)
        rows.append({"Frequency (Hz)": f, "I (ppm)": zi.real, "Q (ppm)": zi.imag,
                     "EC apparent (mS/m)": sa * 1000, "MS apparent (10⁻³ SI)": ka * 1000})
    return pd.DataFrame(rows)


def height_levels(
    rows: pd.DataFrame, height_col: str, labels: list[str], bin_size: float = HEIGHT_BIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean I and Q per height level for the given frequency labels.
    Returns (heights (K,), inphase (K, F), quadrature (K, F)).
    """
    h = pd.to_numeric(rows[height_col], errors="coerce")
    level = (h / bin_size).round() * bin_size
    cols = {}
    for lb in labels:
        cols[f"I|{lb}"] = pd.to_numeric(rows[gem_io.channel_column("I", lb)], errors="coerce")
        cols[f"Q|{lb}"] = pd.to_numeric(rows[gem_io.channel_column("Q", lb)], errors="coerce")
    means = pd.DataFrame(cols).groupby(level).mean().dropna()
    i = means[[f"I|{lb}" for lb in labels]].to_numpy()
    q = means[[f"Q|{lb}" for lb in labels]].to_numpy()
    return means.index.to_numpy(dtype=float), i, q

