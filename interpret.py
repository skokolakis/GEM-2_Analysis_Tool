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
MIN_OVERLAP = 10               # common readings needed for a correlation

VISCOSITY_COLUMN = "MagViscosity[1/1000]"
QDIFF_EC_COLUMN = "ECqdiff[mS/m]"


# ---------------------------------------------------------------------------
# Frequency selection
# ---------------------------------------------------------------------------


def profile_correlation(output_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pearson correlation between every pair of channels over the readings of
    all passes, aligned on distance and pass (pairwise-complete; NaN with
    fewer than MIN_OVERLAP common values). Correlating the passes rather than
    their mean keeps the spatial signal that averaging lines walked in
    different places or directions would cancel.
    """
    values = {}
    for label, df in output_data.items():
        index = pd.MultiIndex.from_product([np.round(df.index.to_numpy(dtype=float), 9), df.columns])
        values[label] = pd.Series(df.to_numpy(dtype=float).ravel(), index=index)
    return pd.concat(values, axis=1).corr(min_periods=MIN_OVERLAP)


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


VISCOSITY_SIGMAS = np.geomspace(1e-5, 10.0, 2000)   # S/m, half-space table for separating viscosity


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

    A viscous (frequency-independent quadrature) susceptibility kappa'' adds
    the same quadrature -kappa'' K at both frequencies (K: static in-phase
    per unit susceptibility of this geometry; e^{+iwt}: kappa = kappa' -
    i kappa''), while the conductive quadrature Qc(f, sigma) grows with
    frequency. So sigma follows from Q(f_high) - Q(f_low) = Qc(f_high, sigma)
    - Qc(f_low, sigma), read from a half-space table of the full forward model
    (the low-induction-number line would leak conductivity into kappa''),
    then kappa'' = (Qc(f_low, sigma) - Q(f_low)) / K and kappa' = (I(f_low) -
    Ic(f_low, sigma)) / K. NaN where the difference is outside the table.
    Raises ValueError where the in-phase hardly responds to susceptibility.
    Returns (sigma S/m, kappa' SI, kappa'' SI) per reading.
    """
    if not E.susceptibility_resolved(sensor):
        raise ValueError(
            f"At {sensor.height:g} m this coil geometry has almost no in-phase susceptibility "
            "response, so magnetic viscosity cannot be separated from conductivity."
        )
    k = E.kappa_sensitivity(sensor)
    z = E.forward_ppm_batch([f_low, f_high], VISCOSITY_SIGMAS[:, None], sensor=sensor)
    dq = z[:, 1].imag - z[:, 0].imag
    rising = np.concatenate([[True], np.diff(dq) > 0]).cumprod().astype(bool) & (dq > 0)
    log_s, z_low, log_dq = np.log(VISCOSITY_SIGMAS[rising]), z[rising, 0], np.log(dq[rising])
    i_low, q_low, q_high = (np.asarray(v, dtype=float) for v in (i_low, q_low, q_high))
    diff = q_high - q_low
    sigma, kappa_i, kappa_q = (np.full(diff.shape, np.nan) for _ in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(diff) & (diff > 0) & (np.log(diff) >= log_dq[0]) & (np.log(diff) <= log_dq[-1])
    ls = np.interp(np.log(diff[ok]), log_dq, log_s)
    sigma[ok] = np.exp(ls)
    q_c = np.exp(np.interp(ls, log_s, np.log(z_low.imag)))
    i_c = np.interp(ls, log_s, z_low.real)
    kappa_q[ok] = (q_c - q_low[ok]) / k
    kappa_i[ok] = (i_low[ok] - i_c) / k
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
            found = gem_io.find_channels(df.columns)
            have = sorted((lb for lb in found["Q"] if lb in found["I"]), key=lambda lb: float(lb[:-2]))
            raise ValueError(
                f"Magnetic viscosity needs the column {col}; this file has I/Q at "
                f"{', '.join(have) or 'no frequency'}."
            )
        cols[(f, comp)] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    sigma, _, kappa_q = viscosity_two_frequencies(
        f_low, f_high, cols[(f_low, "I")], cols[(f_low, "Q")], cols[(f_high, "Q")], sensor
    )
    out = df.copy()
    out[QDIFF_EC_COLUMN] = sigma * 1000.0
    out[VISCOSITY_COLUMN] = kappa_q * 1000.0
    blank = int(np.isnan(sigma).sum())
    return out, [
        f"Magnetic viscosity from {f_low:g} and {f_high:g} Hz (Simon et al., 2015): added "
        f"'{QDIFF_EC_COLUMN}' and '{VISCOSITY_COLUMN}' to the AUX channels"
        + (f" ({blank} reading(s) blank)" if blank else "") + ". Uncalibrated quadrature "
        "offsets shift the viscosity values, so compare contrasts unless Q is calibrated."
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
    Returns columns Frequency (Hz), I anomaly (ppm), Q anomaly (ppm), n
    (readings with a quadrature inside the radius).
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
            "n": int(np.isfinite(q[inside]).sum()),
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
    a1.grid(True)
    a2.plot(spectrum["I anomaly (ppm)"], spectrum["Q anomaly (ppm)"], "o-")
    for _, row in spectrum.iterrows():
        a2.annotate(f"{row['Frequency (Hz)']:g}", (row["I anomaly (ppm)"], row["Q anomaly (ppm)"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel("In-phase anomaly (ppm)")
    a2.set_ylabel("Quadrature anomaly (ppm)")
    a2.set_title("Argand diagram")
    a2.grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
