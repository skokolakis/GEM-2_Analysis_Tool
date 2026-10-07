"""
Streamlit panels for the physics tools: forward modelling for survey
planning, frequency information (skin depth, induction number, depth of
investigation), multi-height calibration of I/Q offsets, and layered-earth
inversion with EMagPy export.

Numerical work lives in emphysics.py; the small table helpers here are pure
and tested on their own.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import contouring as ctr
import corrections
import emphysics
import gem_io
import interpret
import inversion
import soiltools

DEFAULT_FREQUENCIES = "475, 1525, 5325, 18325, 63025"
PERMITTIVITY_FREQUENCY = 40_000.0     # Hz; above this, permittivity can reach the in-phase
PERMITTIVITY_CAPTION = (
    "Above about 40 kHz the in-phase response can include a dielectric-permittivity "
    "contribution, not only magnetic susceptibility (Benech et al., 2016)."
)
HEIGHT_BIN = 0.05                     # m, readings within a bin share one height level
MULTIHEIGHT_MAX_RMS = 50.0            # ppm, about 10 x the GEM-2 noise; worse fits give no offsets
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


@st.cache_data(show_spinner=False)
def _frequency_info(freqs: tuple, sigmas: tuple, sensor: emphysics.Sensor):
    """Frequency table and cumulative-sensitivity curves (cached: expanders run on every rerun)."""
    table = emphysics.frequency_table(freqs, sigmas, sensor)
    depths = np.geomspace(0.05, 20.0, 40)
    curves = [emphysics.cumulative_sensitivity(f, s, depths, sensor) if s > 0 else None
              for f, s in zip(freqs, sigmas)]
    return table, depths, curves


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
        table, depths, curves = _frequency_info(tuple(freqs), tuple(sigmas), sensor)
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        st.caption(
            "For a half-space with the median EC of each frequency, sensor at "
            f"{sensor.height:g} m. Induction number ≪ 1: low-induction-number conditions hold "
            "(McNeill, 1980). Depth of investigation: 70 % of the quadrature response comes from "
            "above it (cumulative sensitivity of the full forward model)."
        )
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, curve in zip(labels, curves):
            if curve is not None:
                ax.plot(curve, depths, label=label)
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
        if not lines:
            return
        state_key = f"mh_result_{file_key}"
        if st.button("Fit offsets", key=f"mh_fit_{file_key}"):
            rows = table[gem_io.line_labels(table["Line"]).isin(lines)]
            heights, i, q = height_levels(rows, height_col, labels)
            try:
                fit = emphysics.fit_multiheight_bias(
                    [float(lb[:-2]) for lb in labels], heights, i, q, sensor
                )
            except ValueError as exc:
                st.session_state.pop(state_key, None)
                st.info(str(exc))
                return
            st.session_state[state_key] = (tuple(lines), tuple(labels), len(heights), fit)
        stored = st.session_state.get(state_key)
        if stored is None or stored[:2] != (tuple(lines), tuple(labels)):
            return
        n_levels, fit = stored[2:]
        st.write(
            f"Half-space EC **{fit['sigma'] * 1000:.4g} mS/m**, MS **{fit['kappa'] * 1000:.4g}** "
            f"× 10⁻³ SI, RMS misfit {fit['rms']:.3g} ppm from {n_levels} height levels."
        )
        if fit["rms"] > MULTIHEIGHT_MAX_RMS:
            st.error(
                f"A half-space fits the calibration readings poorly (RMS {fit['rms']:.3g} ppm > "
                f"{MULTIHEIGHT_MAX_RMS:g} ppm): the ground under the spot is not uniform, the heights "
                "are wrong or the readings are noisy. No offsets are offered."
            )
            return
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


# ---------------------------------------------------------------------------
# Inversion and export
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def run_inversion(
    frequencies: np.ndarray,
    data_q: np.ndarray,
    errors: np.ndarray,
    n_layers: int,
    max_depth: float,
    first: float,
    alpha: float,
    lateral: float,
    sensor: emphysics.Sensor,
) -> inversion.InversionResult:
    """Cached inversion.invert on a fresh layer grid."""
    grid = inversion.make_layer_grid(max_depth, n_layers, first)
    return inversion.invert(frequencies, data_q, errors, grid, sensor, alpha, lateral)


def render_inversion(
    output_data: dict[str, dict[str, pd.DataFrame]],
    scores: dict[str, dict[str, dict]],
    table: pd.DataFrame | None,
    file_key: str,
    sensor: emphysics.Sensor,
) -> None:
    """Smooth 1D / laterally constrained inversion of the mean profiles, and EMagPy export."""
    with st.expander("Layered-earth inversion (1D / laterally constrained)", expanded=False):
        st.caption(
            "Inverts the quadrature of the mean profiles for a smooth layered conductivity "
            "model at stations along the line (Occam-style Gauss–Newton, Constable et al., "
            "1987; lateral constraints as in Auken & Christiansen, 2004). EC-only files are "
            "converted to quadrature with the half-space model. Calibrate first: offsets "
            "in I/Q bias the result (Minsley et al., 2014)."
        )
        c1, c2, c3 = st.columns(3)
        n_layers = c1.slider("Layers", 5, 30, 15, key=f"inv_n_{file_key}")
        max_depth = c1.number_input("Depth to half-space (m)", 0.5, 50.0, 6.0, key=f"inv_d_{file_key}")
        first = c1.number_input("First layer (m)", 0.01, 5.0, 0.1, key=f"inv_f_{file_key}")
        alpha = c2.number_input("Regularisation (start)", 0.001, 10_000.0, 10.0, format="%.3g",
                                key=f"inv_a_{file_key}")
        lateral = c2.number_input("Lateral constraint (0 = independent 1D)", 0.0, 100.0, 1.0,
                                  key=f"inv_l_{file_key}")
        step = c2.number_input("Station spacing (m)", 0.1, 100.0, 1.0, key=f"inv_s_{file_key}")
        rel = c3.number_input("Relative error (%)", 0.1, 50.0, 3.0, key=f"inv_r_{file_key}")
        floor = c3.number_input("Error floor (ppm)", 0.01, 1000.0, 1.0, key=f"inv_fl_{file_key}")
        use_noise = c3.checkbox("Use measured noise (between passes)", True, key=f"inv_n2_{file_key}")
        try:
            stations = inversion.station_data(output_data, scores, step, sensor)
        except ValueError as exc:
            st.info(str(exc))
            return
        st.caption(
            f"{len(stations.distance)} stations × {len(stations.frequencies)} frequencies "
            f"from the {stations.source} channels."
        )
        errors = inversion.data_errors(
            np.nan_to_num(stations.quadrature, nan=0.0),
            stations.noise if use_noise else None, rel / 100.0, floor,
        )
        run_key = f"inv_run_{file_key}"
        params = (int(n_layers), float(max_depth), float(first), float(alpha), float(lateral),
                  float(step), float(rel), float(floor), bool(use_noise), sensor,
                  hash(stations.quadrature.tobytes()))
        too_many = len(stations.distance) > inversion.MAX_STATIONS
        if too_many:
            length = float(stations.distance[-1] - stations.distance[0])
            st.info(
                f"More than {inversion.MAX_STATIONS} stations: set the station spacing to at least "
                f"{length / inversion.MAX_STATIONS:.2g} m to invert."
            )
        elif st.button("Invert", key=f"inv_btn_{file_key}"):
            st.session_state[run_key] = params          # the inputs this inversion is for
        ran = st.session_state.get(run_key)
        if ran is not None and ran != params:
            st.caption("The inputs changed since the last inversion: press Invert to update it.")
        if ran == params and not too_many:
            try:
                with st.spinner("Inverting…"):
                    result = run_inversion(
                        stations.frequencies, stations.quadrature, errors,
                        int(n_layers), float(max_depth), float(first), float(alpha),
                        float(lateral), sensor,
                    )
            except ValueError as exc:
                st.info(str(exc))
                return
            fig = inversion.make_section_figure(result, stations.distance, f"Inverted EC — {file_key}")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.download_button(
                "Model (.csv)", inversion.model_table(result, stations.distance).to_csv(index=False).encode(),
                file_name=f"{file_key}_inversion.csv", mime="text/csv", key=f"inv_dl_{file_key}",
            )
        if table is not None and output_data.get("EC"):
            errors_ec = {
                gem_io.channel_column("EC", lb): sc["mean_std"]
                for lb, sc in scores.get("EC", {}).items() if sc["noise_method"] == "between-trace"
            }
            try:
                data = inversion.emagpy_from_table(table, sensor, errors_ec or None)
            except ValueError as exc:
                st.info(str(exc))
                return
            st.download_button(
                "EMagPy survey (.csv)", data, file_name=f"{file_key}_emagpy.csv",
                mime="text/csv", key=f"inv_emagpy_{file_key}",
                help="Coil columns HCP{separation}f{frequency}h{height} in mS/m with *_err "
                     "columns from the between-pass noise. EMagPy models a plain loop pair: "
                     "the GEM-2 bucking coil is not modelled there.",
            )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def render_frequency_selection(
    output_data: dict[str, pd.DataFrame], scores: dict[str, dict], file_key: str
) -> None:
    """Redundancy-aware choice of a few frequencies from the scores and profile correlations."""
    with st.expander("Frequency selection (redundancy-aware)", expanded=False):
        st.caption(
            "The GEM-2 shares its transmitter power among all frequencies, so fewer "
            "frequencies give less noise on each; drone-towed GEM-2 tests recommend at most "
            "three (Vilhelmsen & Døssing, 2022). Strongly correlated frequencies carry the "
            "same information (Minsley et al., 2010)."
        )
        c1, c2 = st.columns(2)
        count = c1.slider("Frequencies to keep", 1, max(1, len(scores)), min(3, len(scores)),
                          key=f"sel_n_{file_key}")
        limit = c2.slider("Redundant above |r|", 0.5, 1.0, 0.95, step=0.01, key=f"sel_r_{file_key}")
        corr = interpret.profile_correlation(output_data)
        table = interpret.select_frequencies(scores, corr, count, limit)
        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="Correlation of mean profiles")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_anomaly_spectrum(table: pd.DataFrame, file_key: str) -> None:
    """In-phase / quadrature anomaly against frequency around a chosen point."""
    with st.expander("Anomaly spectrum (EMI spectroscopy)", expanded=False):
        st.caption(
            "Mean I and Q within the radius minus the background, for every frequency with "
            "I_ and Q_ columns. The shape of the spectrum and of the Argand diagram helps tell "
            "metal targets from soil and geology (Huang & Won, 2003)."
        )
        found = gem_io.find_channels(table.columns)
        if not any(lb in found["I"] for lb in found["Q"]):
            st.info("Needs I_ and Q_ columns.")
            return
        try:
            x, y, _ = corrections.xy_metres(table)
            has_xy = bool(np.isfinite(x).any())
        except ctr.ContouringError:
            has_xy = False
        c1, c2, c3 = st.columns(3)
        if has_xy:
            cx = c1.number_input("X (m)", value=float(np.nanmedian(x)), key=f"sp_x_{file_key}")
            cy = c2.number_input("Y (m)", value=float(np.nanmedian(y)), key=f"sp_y_{file_key}")
            centre = (float(cx), float(cy))
            st.caption("Map metres: the projected coordinates the area map uses for Lat/Lon.")
        else:                                    # distances are along each line: pick one
            labels = gem_io.line_labels(table["Line"])
            line = c2.selectbox("Line", sorted(labels.unique()), key=f"sp_line_{file_key}")
            table = table[(labels == line).to_numpy()]
            d = pd.to_numeric(table[gem_io.DISTANCE_COL], errors="coerce")
            centre = float(c1.number_input("Distance (m)", value=float(d.median()), key=f"sp_d_{file_key}"))
        radius = c3.number_input("Radius (m)", 0.01, 1000.0, 1.0, key=f"sp_r_{file_key}")
        background = st.radio("Background", ["annulus", "survey"], horizontal=True,
                              format_func={"annulus": "Ring from radius to 2 × radius",
                                           "survey": "Whole survey"}.get,
                              key=f"sp_bg_{file_key}")
        try:
            spectrum = interpret.anomaly_spectrum(table, centre, float(radius), background)
        except ValueError as exc:
            st.info(str(exc))
            return
        st.dataframe(spectrum.round(3), hide_index=True)
        fig = interpret.make_spectrum_figure(spectrum, f"Anomaly at {centre}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Soil tools
# ---------------------------------------------------------------------------

ZONE_SAMPLE = 5000          # readings used to compare zone numbers (random subset)


@st.cache_data(show_spinner=False, max_entries=8)
def _property_map(table: pd.DataFrame, column: str) -> ctr.AreaMapResult:
    return ctr.compute_area_map(table, column)


def _lonlat_columns(x: np.ndarray, y: np.ndarray, origin) -> dict:
    if origin is None:
        return {}
    lon, lat = ctr.local_metres_to_lonlat(x, y, origin)
    return {"lon": lon, "lat": lat}


def render_soil_tools(table: pd.DataFrame, file_key: str) -> None:
    """Sampling design, calibration of a soil property, and management zones from EC channels."""
    with st.expander("Soil tools (sampling design, calibration, management zones)", expanded=False):
        ec_cols = list(gem_io.find_channels(table.columns)["EC"].values())
        if not ec_cols:
            st.info("Needs EC channels.")
            return
        try:
            x, y, origin = corrections.xy_metres(table)
        except ctr.ContouringError:
            st.info("Needs X/Y or Lat/Lon coordinates.")
            return
        chans = st.multiselect("EC channels", ec_cols, default=ec_cols[:2], key=f"soil_ch_{file_key}")
        if not chans:
            return
        features = table[chans].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        t_design, t_cal, t_zones = st.tabs(["Sampling design", "Calibration", "Management zones"])

        with t_design:
            st.caption(
                "Sampling sites that span the conductivity range and are spread over the field "
                "(response-surface design after Lesch, 2005; ESAP-RSSD)."
            )
            n_sites = st.number_input("Sites", 4, 60, 12, key=f"soil_n_{file_key}")
            try:
                sites = soiltools.sampling_design(np.column_stack([x, y]), features, int(n_sites))
            except ValueError as exc:
                st.info(str(exc))
            else:
                extra = _lonlat_columns(sites["x"].to_numpy(), sites["y"].to_numpy(), origin)
                sites = sites.assign(**extra)
                st.dataframe(sites.round(4), hide_index=True, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(x, y, s=2, c="0.7")
                ax.scatter(sites["x"], sites["y"], c="red", s=25)
                for k, (sx, sy) in enumerate(zip(sites["x"], sites["y"]), 1):
                    ax.annotate(str(k), (sx, sy), fontsize=8, xytext=(3, 3), textcoords="offset points")
                ax.set_aspect("equal")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.download_button("Sites (.csv)", sites.to_csv(index=False).encode(),
                                   file_name=f"{file_key}_sampling_sites.csv", mime="text/csv",
                                   key=f"soil_dl_sites_{file_key}")

        with t_cal:
            st.caption(
                "Regression of a measured soil property on the conductivity at the sampling "
                "sites (Lesch et al., 1995; ESAP-Calibrate). The samples file has the survey's "
                "coordinate columns and one column per measured property."
            )
            upload = st.file_uploader("Soil samples (.csv)", type=["csv"], key=f"soil_up_{file_key}")
            if upload is not None:
                samples = pd.read_csv(upload)
                skip = {c.lower() for c in ("x", "y", "lat", "latitude", "lon", "long", "longitude")}
                props = [c for c in samples.columns if str(c).lower() not in skip
                         and pd.to_numeric(samples[c], errors="coerce").notna().any()]
                if not props:
                    st.info("No numeric property column in the samples file.")
                else:
                    c1, c2 = st.columns(2)
                    prop = c1.selectbox("Property", props, key=f"soil_prop_{file_key}")
                    radius = c1.number_input("Matching radius (m)", 0.1, 100.0, 2.0, key=f"soil_rad_{file_key}")
                    log_p = c2.checkbox("Fit the property on a log scale", True, key=f"soil_log_{file_key}")
                    trend = c2.checkbox("Add a trend surface (x, y)", False, key=f"soil_tr_{file_key}")
                    try:
                        model = soiltools.fit_property_model(table, samples, prop, chans, radius, log_p, trend)
                    except (ValueError, ctr.ContouringError) as exc:
                        st.info(str(exc))
                    else:
                        st.write(f"R² **{model.r2:.3f}**, RMSE {model.rmse:.4g} "
                                 f"({'log scale' if log_p else 'property units'}), {model.n} samples.")
                        st.dataframe(model.coefficients.rename("coefficient").to_frame().round(5))
                        column = f"Predicted {prop}"
                        predicted = table.assign(**{column: soiltools.predict_property(model, table)})
                        try:
                            result = _property_map(predicted, column)
                            fig = ctr.make_area_map_figure(result, column, column)
                        except ctr.ContouringError as exc:
                            st.info(str(exc))
                        else:
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        out = pd.DataFrame({"x": x, "y": y, column: predicted[column]})
                        st.download_button("Predictions (.csv)", out.to_csv(index=False).encode(),
                                           file_name=f"{file_key}_{prop}_predicted.csv",
                                           mime="text/csv", key=f"soil_dl_pred_{file_key}")

        with t_zones:
            st.caption(
                "Fuzzy c-means on the standardised channels (Bezdek, 1981), as in Management "
                "Zone Analyst (Fridgen et al., 2004). The lowest FPI and MPE (modified partition "
                "entropy, Boydell & McBratney, 2002) suggest the number of zones."
            )
            ok = np.all(np.isfinite(features), axis=1) & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                st.info("Not enough complete readings.")
                return
            data = soiltools.standardise(features[ok])
            m = st.slider("Fuzziness exponent", 1.1, 3.0, 1.3, step=0.05, key=f"soil_m_{file_key}")
            rng = np.random.default_rng(0)
            subset = data if len(data) <= ZONE_SAMPLE else data[rng.choice(len(data), ZONE_SAMPLE, replace=False)]
            indices = soiltools.zone_indices(subset, range(2, 7), m)
            st.dataframe(indices.round(4), hide_index=True)
            best = int(indices.loc[indices["FPI"].idxmin(), "Zones"])
            c = st.slider("Zones", 2, 6, best, key=f"soil_c_{file_key}")
            u, _ = soiltools.fuzzy_cmeans(data, int(c), m)
            zone = u.argmax(axis=1) + 1
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(x[ok], y[ok], c=zone, s=4, cmap="tab10", vmin=0.5, vmax=10.5)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fig.colorbar(sc, ax=ax, ticks=range(1, int(c) + 1), label="Zone")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            out = pd.DataFrame({"x": x[ok], "y": y[ok], "zone": zone,
                                **{f"membership_{k + 1}": u[:, k] for k in range(int(c))}})
            st.download_button("Zones (.csv)", out.to_csv(index=False).encode(),
                               file_name=f"{file_key}_zones.csv", mime="text/csv",
                               key=f"soil_dl_zones_{file_key}")
