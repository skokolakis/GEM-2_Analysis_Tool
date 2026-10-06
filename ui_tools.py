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


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _default_layers(top_ec: float, bottom_ec: float) -> pd.DataFrame:
    return pd.DataFrame({
        "Thickness (m)": [1.0, np.nan], "EC (mS/m)": [top_ec, bottom_ec], "MS (10⁻³ SI)": [0.0, 0.0],
    })


def render_forward_model(sensor: emphysics.Sensor) -> None:
    """Survey planning: responses of a background and a target layered model."""
    with st.expander("Forward model (survey planning)", expanded=False):
        st.caption(
            "Layered-earth response of the sensor in the sidebar geometry. Rows go from the "
            "top down; the last row is the half-space (its thickness is ignored). A difference "
            "larger than 3 × the noise level is marked detectable."
        )
        freq_text = st.text_input("Frequencies (Hz)", DEFAULT_FREQUENCIES, key="fm_freqs")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Background**")
            background = st.data_editor(_default_layers(10.0, 30.0), num_rows="dynamic", key="fm_bg")
        with c2:
            st.markdown("**Target**")
            target = st.data_editor(_default_layers(10.0, 100.0), num_rows="dynamic", key="fm_tg")
        noise = st.number_input("Noise level (ppm)", 0.0, 10_000.0, 5.0, key="fm_noise",
                                help="The GEM-2 Manual quotes about 5 ppm ambient noise.")
        try:
            freqs = parse_frequencies(freq_text)
            rb = model_response(freqs, background, sensor)
            rt = model_response(freqs, target, sensor)
        except ValueError as exc:
            st.info(str(exc))
            return
        table = pd.DataFrame({
            "Frequency (Hz)": freqs,
            "Q background (ppm)": rb["Q (ppm)"], "Q target (ppm)": rt["Q (ppm)"],
            "I background (ppm)": rb["I (ppm)"], "I target (ppm)": rt["I (ppm)"],
            "EC apparent background (mS/m)": rb["EC apparent (mS/m)"],
            "EC apparent target (mS/m)": rt["EC apparent (mS/m)"],
        })
        dq = (rt["Q (ppm)"] - rb["Q (ppm)"]).abs()
        di = (rt["I (ppm)"] - rb["I (ppm)"]).abs()
        table["Detectable"] = np.maximum(dq, di) > 3 * noise
        st.dataframe(table.round(4), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.semilogx(freqs, rb["EC apparent (mS/m)"], "o-", label="Background")
        ax.semilogx(freqs, rt["EC apparent (mS/m)"], "s-", label="Target")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Apparent EC (mS/m)")
        ax.grid(True, alpha=0.4)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_frequency_info(
    output_data: dict[str, pd.DataFrame], sensor: emphysics.Sensor, file_key: str
) -> None:
    """Skin depth, induction number and depth of investigation for the EC frequencies."""
    with st.expander("Frequency information (skin depth, depth of investigation)", expanded=False):
        labels, freqs, sigmas = [], [], []
        for label, df in output_data.items():
            f = ctr.parse_frequency(label)
            ec = float(np.nanmedian(df.mean(axis=1, skipna=True)))
            if f is None or not np.isfinite(ec):
                continue
            labels.append(label)
            freqs.append(f)
            sigmas.append(ec / 1000.0)
        if not freqs:
            st.info("No frequency labels to analyse.")
            return
        table = emphysics.frequency_table(freqs, sigmas, sensor)
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        st.caption(
            "For a half-space with the median EC of each frequency, sensor at "
            f"{sensor.height:g} m. Induction number ≪ 1: low-induction-number conditions hold "
            "(McNeill, 1980). Depth of investigation: 70 % of the quadrature response comes from "
            "above it (cumulative sensitivity of the full forward model)."
        )
        depths = np.geomspace(0.05, 20.0, 40)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, f, s in zip(labels, freqs, sigmas):
            if s > 0:
                ax.plot(emphysics.cumulative_sensitivity(f, s, depths, sensor), depths, label=label)
        ax.invert_yaxis()
        ax.set_yscale("log")
        ax.set_xlabel("Fraction of response from below depth")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_multiheight(table: pd.DataFrame, file_key: str, sensor: emphysics.Sensor) -> None:
    """Fits I/Q offsets from readings over one spot at several sensor heights."""
    with st.expander("Multi-height calibration (I/Q offsets)", expanded=False):
        st.caption(
            "Hold the sensor over one spot at several heights and record it as its own line(s) "
            "with a height column. A half-space plus one offset per frequency and component is "
            "fitted (after Minsley et al., 2014). Download the offsets and load them as "
            "'Calibration offsets' in the sidebar."
        )
        found = gem_io.find_channels(table.columns)
        labels = [lb for lb in found["Q"] if lb in found["I"]]
        if not labels:
            st.info("Needs I_ and Q_ columns.")
            return
        height_col = st.text_input("Height column", "Height", key=f"mh_col_{file_key}")
        if height_col not in table.columns:
            st.info(f"No column named '{height_col}'.")
            return
        lines = st.multiselect(
            "Calibration line(s)", sorted(gem_io.line_labels(table["Line"]).unique()), key=f"mh_lines_{file_key}"
        )
        if not lines or not st.button("Fit offsets", key=f"mh_fit_{file_key}"):
            return
        rows = table[gem_io.line_labels(table["Line"]).isin(lines)]
        heights, i, q = height_levels(rows, height_col, labels)
        try:
            fit = emphysics.fit_multiheight_bias(
                [float(lb[:-2]) for lb in labels], heights, i, q, sensor
            )
        except ValueError as exc:
            st.info(str(exc))
            return
        st.write(
            f"Half-space EC **{fit['sigma'] * 1000:.4g} mS/m**, MS **{fit['kappa'] * 1000:.4g}** "
            f"× 10⁻³ SI, RMS misfit {fit['rms']:.3g} ppm from {len(heights)} height levels."
        )
        offsets = pd.DataFrame({
            "column": [gem_io.channel_column("I", lb) for lb in labels]
            + [gem_io.channel_column("Q", lb) for lb in labels],
            "offset": np.concatenate([fit["bias_i"], fit["bias_q"]]),
        })
        st.dataframe(offsets.round(3), hide_index=True)
        st.download_button(
            "Offsets (.csv)", offsets.to_csv(index=False).encode(),
            file_name=f"{file_key}_iq_offsets.csv", mime="text/csv", key=f"mh_dl_{file_key}",
        )
