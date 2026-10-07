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


# ---------------------------------------------------------------------------
# Magnetic viscosity
# ---------------------------------------------------------------------------


def viscosity_two_frequencies(
    f_low: float,
    f_high: float,
    i_low: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    sensor: E.Sensor = E.GEM2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conductivity, in-phase susceptibility and quadrature susceptibility
    (magnetic viscosity) from two frequencies, after Simon et al. (2015).

    At low induction number the conductive quadrature grows in proportion to
    frequency, while a viscous (frequency-independent quadrature)
    susceptibility kappa'' adds the same quadrature at both:
        Q(f) = sigma * L(f) - kappa'' * K,
    with L(f) the LIN quadrature per S/m and K the static in-phase per unit
    susceptibility for this geometry (e^{+iwt}: kappa = kappa' - i kappa'').
    The conductive part of the low-frequency in-phase is then removed with the
    half-space model before kappa' = I / K.
    Returns (sigma S/m, kappa' SI, kappa'' SI) per reading.
    """
    l_low = E.lin_ppm_per_sigma(f_low, sensor)
    l_high = E.lin_ppm_per_sigma(f_high, sensor)
    k = E.forward_ppm(f_low, [0.0], [1e-3], sensor=sensor).real[0] / 1e-3
    if k == 0:
        raise ValueError("The sensor has no in-phase susceptibility response at this height.")
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    sigma = (q_high - q_low) / (l_high - l_low)
    kappa_q = (sigma * l_low - q_low) / k
    ok = np.isfinite(sigma) & (sigma > 0)
    i_sigma = np.full(sigma.shape, np.nan)
    if ok.any():
        i_sigma[ok] = E.forward_ppm_batch([f_low], sigma[ok][:, None], sensor=sensor).real[:, 0]
    kappa_i = (np.asarray(i_low, dtype=float) - i_sigma) / k
    return sigma, kappa_i, kappa_q


def add_viscosity_columns(
    df: pd.DataFrame, pair: tuple[float, float], sensor: E.Sensor
) -> tuple[pd.DataFrame, list[str]]:
    """Adds ECqdiff[mS/m] and MagViscosity[1/1000] (kappa'' x 1000) from two I/Q frequencies."""
    f_low, f_high = sorted(float(f) for f in pair)
    cols = {}
    for f, comp in ((f_low, "I"), (f_low, "Q"), (f_high, "Q")):
        col = gem_io.channel_column(comp, f"{f:g}Hz")
        if col not in df.columns:
            raise ValueError(f"Magnetic viscosity needs the column {col}.")
        cols[(f, comp)] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    sigma, _, kappa_q = viscosity_two_frequencies(
        f_low, f_high, cols[(f_low, "I")], cols[(f_low, "Q")], cols[(f_high, "Q")], sensor
    )
    out = df.copy()
    out[QDIFF_EC_COLUMN] = sigma * 1000.0
    out[VISCOSITY_COLUMN] = kappa_q * 1000.0
    return out, [
        f"Magnetic viscosity from {f_low:g} and {f_high:g} Hz (Simon et al., 2015): added "
        f"'{QDIFF_EC_COLUMN}' and '{VISCOSITY_COLUMN}' to the AUX channels."
    ]


def viscosity_per_decade(kappa_q: np.ndarray) -> np.ndarray:
    """Susceptibility drop per decade of frequency for a log-uniform relaxation spectrum: (2 ln 10 / pi) kappa''."""
    return 2.0 * math.log(10.0) / math.pi * np.asarray(kappa_q, dtype=float)


# ---------------------------------------------------------------------------
# Anomaly spectra
# ---------------------------------------------------------------------------


def anomaly_spectrum(
    table: pd.DataFrame,
    centre: tuple[float, float] | float,
    radius: float,
    background: str = "annulus",
) -> pd.DataFrame:
    """
    In-phase and quadrature anomaly per frequency around one point: mean of
    the readings within `radius` minus the background, which is the median of
    the readings between radius and 2 x radius ("annulus") or of the whole
    survey ("survey"). `centre` is (x, y) in map metres, or a distance along
    the line when the table has no coordinates.
    Returns columns Frequency (Hz), I anomaly (ppm), Q anomaly (ppm), n.
    """
    found = gem_io.find_channels(table.columns)
    labels = sorted((lb for lb in found["Q"] if lb in found["I"]), key=lambda lb: float(lb[:-2]))
    if not labels:
        raise ValueError("Anomaly spectra need I_ and Q_ columns.")
    if isinstance(centre, tuple):
        x, y, _ = corrections.xy_metres(table)
        d = np.hypot(x - centre[0], y - centre[1])
    else:
        d = np.abs(pd.to_numeric(table[gem_io.DISTANCE_COL], errors="coerce").to_numpy() - centre)
    inside = d <= radius
    if not inside.any():
        raise ValueError("No readings within the radius of that point.")
    ring = (d > radius) & (d <= 2 * radius) if background == "annulus" else np.isfinite(d) & ~inside
    if not ring.any():
        raise ValueError("No background readings around that point.")
    rows = []
    for lb in labels:
        i = pd.to_numeric(table[found["I"][lb]], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(table[found["Q"][lb]], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "Frequency (Hz)": float(lb[:-2]),
            "I anomaly (ppm)": float(np.nanmean(i[inside]) - np.nanmedian(i[ring])),
            "Q anomaly (ppm)": float(np.nanmean(q[inside]) - np.nanmedian(q[ring])),
            "n": int(inside.sum()),
        })
    return pd.DataFrame(rows)


def make_spectrum_figure(spectrum: pd.DataFrame, title: str):
    """In-phase and quadrature anomaly against frequency, and the Argand diagram (Q against I)."""
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    f = spectrum["Frequency (Hz)"]
    a1.semilogx(f, spectrum["I anomaly (ppm)"], "o-", label="In-phase")
    a1.semilogx(f, spectrum["Q anomaly (ppm)"], "s--", label="Quadrature")
    a1.axhline(0, color="0.5", linewidth=0.8)
    a1.set_xlabel("Frequency (Hz)")
    a1.set_ylabel("Anomaly (ppm)")
    a1.legend()
    a1.grid(True, alpha=0.4)
    a2.plot(spectrum["I anomaly (ppm)"], spectrum["Q anomaly (ppm)"], "o-")
    for _, row in spectrum.iterrows():
        a2.annotate(f"{row['Frequency (Hz)']:g}", (row["I anomaly (ppm)"], row["Q anomaly (ppm)"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel("In-phase anomaly (ppm)")
    a2.set_ylabel("Quadrature anomaly (ppm)")
    a2.set_title("Argand diagram")
    a2.grid(True, alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
