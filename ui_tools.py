"""
Streamlit panels for the physics tools: forward modelling for survey
planning, frequency information (skin depth, induction number, depth of
investigation), multi-height calibration of I/Q offsets, and layered-earth
inversion with EMagPy export.

Numerical work lives in emphysics.py; the small table helpers here are pure
and tested on their own.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

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

DOCS_URL = "https://github.com/skokolakis/GEM-2_Analysis_Tool/blob/main/docs"
DEFAULT_FREQUENCIES = "475, 1525, 5325, 18325, 63025"
PERMITTIVITY_FREQUENCY = 40_000.0     # Hz; above this, permittivity can reach the in-phase
PERMITTIVITY_CAPTION = (
    "Above about 40 kHz the in-phase response can include a dielectric-permittivity "
    "contribution, not only magnetic susceptibility (Benech et al., 2016)."
)
HEIGHT_BIN = 0.05                     # m, readings within a bin share one height level
MULTIHEIGHT_MAX_RMS = 50.0            # ppm, about 10 x the GEM-2 noise; worse fits give no offsets
LAYER_COLUMNS = ["Thickness (m)", "EC (mS/m)", "MS (10⁻³ SI)"]
NEEDS_IQ = (
    "This tool needs in-phase and quadrature columns (I_…Hz and Q_…Hz, in ppm); this file "
    "has none. Export them from WinGEM / EMExport to use it."
)
NEEDS_XY = "This tool needs X/Y or Lat/Lon coordinates; this file has none."


SCIENCE_HELP = {                 # tooltips (the "?" icon) of the science choices
    "fm_freqs": (
        "Frequencies to model, comma-separated, e.g. those your GEM-2 records."
    ),
    "inv_alpha": (
        "Starting weight of the smoothness constraint. Each iteration adjusts it (Occam): the "
        "smoothest model that fits the data within their errors."
    ),
    "inv_depth": (
        "Depth of the top of the half-space. Keep it near the depth of investigation of the "
        "lowest frequency; deeper layers are not resolved."
    ),
    "inv_first": (
        "Thickness of the top layer; the layers below thicken geometrically down to the "
        "half-space."
    ),
    "inv_floor": (
        "Smallest data error in ppm, added to the relative error."
    ),
    "inv_lateral": (
        "How strongly neighbouring stations are tied together (Auken & Christiansen, 2004). 0 "
        "inverts each station on its own."
    ),
    "inv_layers": (
        "Number of model layers; the last is a half-space. More layers give a smoother depth "
        "profile but take longer."
    ),
    "inv_noise": (
        "Uses the measured noise of each frequency (between repeat passes, or within the line "
        "for a single pass), divided by √passes, where it is larger than the relative error "
        "plus floor."
    ),
    "transect": (
        "The lines of this file lie at different positions, so their mean is not a profile of "
        "one place. Choose the transect to use; lines walked over the same transect (repeat "
        "passes) are averaged."
    ),
    "inv_rel": (
        "Data error as a percentage of each quadrature value."
    ),
    "inv_step": (
        "Distance between inverted stations along the line (at most 200 stations)."
    ),
    "mh_height": (
        "Column with the sensor height of each calibration reading."
    ),
    "mh_lines": (
        "Lines recorded over one spot at several heights."
    ),
    "sel_corr": (
        "Two frequencies whose readings correlate more strongly than this carry the same "
        "information; the lower-scoring one is skipped."
    ),
    "sel_count": (
        "How many of the best-scoring frequencies to keep. The GEM-2 shares its transmitter "
        "power among its frequencies, so fewer give less noise each; at most three are "
        "recommended (Vilhelmsen & Døssing, 2022)."
    ),
    "soil_channels": (
        "Channels used for the sampling design and the management zones."
    ),
    "soil_log": (
        "Fits ln(property): suits positive, skewed properties such as salinity."
    ),
    "soil_m": (
        "How soft the zone boundaries are (m in fuzzy c-means): 1.1–1.5 is usual for soils; "
        "larger values share readings between zones."
    ),
    "soil_property": (
        "Measured soil property to calibrate against EC."
    ),
    "soil_radius": (
        "Survey readings within this distance of a sample are matched to it (their median)."
    ),
    "soil_sites": (
        "Number of sampling sites. They cover the range of the selected channels and are "
        "spread over the field (Lesch, 2005)."
    ),
    "soil_trend": (
        "Adds x and y terms for a gradual change across the field that conductivity does not "
        "explain."
    ),
    "soil_zones": (
        "Number of management zones; the lowest FPI and MPE in the table suggest it."
    ),
    "sp_background": (
        "Subtracted from the anomaly: the median of a ring from the radius to twice the "
        "radius around the point, or of the whole survey."
    ),
    "sp_line": (
        "The file has no coordinates, so distances are along each line: the spectrum is taken "
        "on one line."
    ),
    "sp_radius": (
        "Readings within this distance of the point make up the anomaly."
    ),
}


# ---------------------------------------------------------------------------
# Shared display helpers (one look for every plot, table and download)
# ---------------------------------------------------------------------------

DOWNLOAD_ICON = ":material/download:"
MIME_TYPES = {
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".png": "image/png",
    ".asc": "text/plain",
    ".prj": "text/plain",
    ".tif": "image/tiff",
}


def doc_link(page: str, anchor: str = "", text: str = "") -> str:
    """Markdown link to a documentation page (and section), e.g. doc_link("troubleshooting", "no-area-map")."""
    url = f"{DOCS_URL}/{page}.md" + (f"#{anchor}" if anchor else "")
    return f"[{text or 'More in the documentation'}]({url})"


NEEDS_ANCHOR = "a-tool-needs-columns-the-file-lacks"     # troubleshooting section


def show_needs(message: str) -> None:
    """Info box for a tool the file cannot use, linked to the troubleshooting section."""
    st.info(f"{message} {doc_link('troubleshooting', NEEDS_ANCHOR)}")


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 file', '3 files'; *many* for irregular plurals ('frequency' -> 'frequencies')."""
    return f"{n:,} {word if n == 1 else many or word + 's'}"


def show_figure(fig, stretch: bool = True) -> None:
    """
    Shows and closes a matplotlib figure. Wide plots (profiles, maps,
    sections) fill the page; small diagnostic plots (stretch=False) keep their
    own size, since stretched to the page width their text grows oversized.
    """
    st.pyplot(fig, width="stretch" if stretch else "content")
    plt.close(fig)


def show_table(df, **kwargs) -> None:
    """A dataframe across the page width, without the index."""
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)


def button_row():
    """Container that lays buttons side by side at their own width, wrapping on narrow screens."""
    return st.container(horizontal=True, gap="small")


def download_button(label: str, data, file_name: str, key: str, help: str | None = None) -> None:
    """Download button with the shared icon; the MIME type follows the file extension."""
    suffix = file_name[file_name.rfind("."):].lower()
    st.download_button(
        label, data=data, file_name=file_name, key=key, help=help, icon=DOWNLOAD_ICON,
        mime=MIME_TYPES.get(suffix, "application/octet-stream"),
    )


# ---------------------------------------------------------------------------
# Data summary (what a file holds and which tools it supports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSummary:
    lines: int
    readings: int
    line_length: float                # m, distance covered summed over the lines (NaN if unknown)
    duration: float | None            # s, first to last reading; None without a time column
    frequencies: tuple[str, ...]      # channel labels such as '1525Hz', low to high
    has_xy: bool                      # X/Y or Lat/Lon coordinates
    has_ec: bool
    has_quadrature: bool
    has_iq: bool                      # I_ and Q_ of at least one frequency
    has_time: bool
    spacing: float = float("nan")     # m, median distance between neighbouring readings of a line


STEP_RANGE = (0.5, 5.0)               # a distance step within these multiples of the spacing suits the data
NICE_STEPS = (1.0, 2.0, 5.0)          # steps are suggested as 1, 2 or 5 × a power of ten


def median_spacing(distance: pd.Series, lines: pd.Series) -> float:
    """Median gap between neighbouring readings along each line (repeats ignored); NaN if none."""
    d = pd.DataFrame({"line": lines.to_numpy(), "d": pd.to_numeric(distance, errors="coerce").to_numpy()})
    d = d.dropna().sort_values(["line", "d"], kind="stable")
    gaps = d.groupby("line")["d"].diff().to_numpy()
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    return float(np.median(gaps)) if gaps.size else float("nan")


def suggested_step(spacing: float, minimum: float = 0.01) -> float | None:
    """The 1-2-5 step nearest to the reading spacing (on a log scale); None if unknown."""
    if not np.isfinite(spacing) or spacing <= 0:
        return None
    exponent = np.floor(np.log10(spacing))
    candidates = [m * 10.0 ** e for e in (exponent - 1, exponent, exponent + 1) for m in NICE_STEPS]
    best = min(candidates, key=lambda c: abs(np.log(c / spacing)))
    return float(max(round(best, 10), minimum))


def step_mismatch(step: float, spacing: float) -> bool:
    """True when the distance step is far from the reading spacing (see STEP_RANGE)."""
    if not np.isfinite(spacing) or spacing <= 0:
        return False
    return not STEP_RANGE[0] * spacing <= step <= STEP_RANGE[1] * spacing


def data_summary(table: pd.DataFrame) -> DataSummary:
    """Size, extent and contents of a prepared GEM table."""
    found = gem_io.find_channels(table.columns)
    labels = {lb for m in gem_io.FREQUENCY_MODES for lb in found[m]}
    lines = (gem_io.line_labels(table["Line"]) if "Line" in table.columns
             else pd.Series("0", index=table.index))
    length = spacing = float("nan")
    if gem_io.DISTANCE_COL in table.columns:
        d = pd.to_numeric(table[gem_io.DISTANCE_COL], errors="coerce")
        length = float(d.groupby(lines.to_numpy()).agg(lambda v: v.max() - v.min()).sum())
        spacing = median_spacing(d, lines)
    t = gem_io.time_seconds(table)
    duration = None
    if t is not None and np.isfinite(t).any():
        duration = float(np.nanmax(t) - np.nanmin(t))
    try:
        x, y, _ = corrections.xy_metres(table)
        has_xy = bool((np.isfinite(x) & np.isfinite(y)).any())
    except (ctr.ContouringError, KeyError):
        has_xy = False
    return DataSummary(
        lines=int(lines.nunique()), readings=len(table), line_length=length, duration=duration,
        frequencies=tuple(sorted(labels, key=lambda lb: ctr.parse_frequency(lb) or 0.0)),
        has_xy=has_xy, has_ec=bool(found["EC"]), has_quadrature=bool(found["Q"]),
        has_iq=any(lb in found["I"] for lb in found["Q"]), has_time=duration is not None,
        spacing=spacing,
    )


def tool_availability(s: DataSummary) -> list[tuple[str, bool, str]]:
    """(tool, available, what it needs) for the tools that depend on the file's contents."""
    return [
        ("Area maps", s.has_xy, "needs X/Y or Lat/Lon coordinates"),
        ("Pseudo-sections", len(s.frequencies) >= 2, "needs two or more frequencies"),
        ("Layered-earth inversion", s.has_ec or s.has_quadrature, "needs EC or Q_ channels"),
        ("Anomaly spectrum", s.has_iq, "needs I_ and Q_ columns"),
        ("Multi-height calibration", s.has_iq, "needs I_ and Q_ columns"),
        ("Base-station drift", s.has_time, "needs a Time column"),
        ("GPS lag estimate", s.has_time and s.has_xy, "needs a Time column and coordinates"),
        ("Soil tools", s.has_ec and s.has_xy, "need EC channels and coordinates"),
    ]


def format_duration(seconds: float | None) -> str:
    """'45 s', '12 min', '2 h 05 min'; '—' when unknown."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = int(round(seconds / 60))
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def summary_line(s: DataSummary) -> str:
    """'**31** lines · **24,806** readings · **2,480 m** along the lines · **47 min** · ...'."""
    parts = [f"**{plural(s.lines, 'line')}**", f"**{plural(s.readings, 'reading')}**"]
    if np.isfinite(s.line_length):
        parts.append(f"**{s.line_length:,.0f} m** along the lines")
    if np.isfinite(s.spacing):
        parts.append(f"readings **{s.spacing:.3g} m** apart")
    if s.duration is not None:
        parts.append(f"**{format_duration(s.duration)}** of recording")
    if s.frequencies:
        hz = ", ".join(lb.removesuffix("Hz") for lb in s.frequencies)
        parts.append(f"**{plural(len(s.frequencies), 'frequency', 'frequencies')}** ({hz} Hz)")
    return " · ".join(parts)


def render_data_summary(s: DataSummary, note: str = "") -> None:
    """Bordered card: size of the survey, its frequencies, and which tools it supports."""
    with st.container(border=True):
        if note:
            st.caption(note)
        st.markdown(summary_line(s))
        tools = tool_availability(s)
        ready = [name for name, ok, _ in tools if ok]
        missing = [f"{name} ({why})" for name, ok, why in tools if not ok]
        if ready:
            st.caption(":green[:material/check_circle:] **Available:** " + " · ".join(ready))
        if missing:
            st.caption(":gray[:material/block:] **Not available:** " + " · ".join(missing) + ". "
                       + doc_link("troubleshooting", NEEDS_ANCHOR, "Why?"))


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
    with st.expander("Forward model (layered-earth response)", expanded=False):
        st.caption(
            "Layered-earth response of the sensor in the sidebar geometry. Rows go from the "
            "top down; the last row is the half-space (its thickness is ignored). A difference "
            "larger than 3 × the noise level is marked detectable."
        )
        freq_text = st.text_input("Frequencies (Hz)", DEFAULT_FREQUENCIES, key="fm_freqs",
                                  help=SCIENCE_HELP["fm_freqs"])
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
        show_table(table.round(4))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.semilogx(freqs, rb["EC apparent (mS/m)"], "o-", label="Background")
        ax.semilogx(freqs, rt["EC apparent (mS/m)"], "s-", label="Target")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Apparent EC (mS/m)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        show_figure(fig, stretch=False)


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
            st.info("No channel here is labelled with a frequency in Hz, so there is nothing to analyse.")
            return
        table, depths, curves = _frequency_info(tuple(freqs), tuple(sigmas), sensor)
        show_table(table.round(3))
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
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        show_figure(fig, stretch=False)


def render_multiheight(table: pd.DataFrame, file_key: str, sensor: emphysics.Sensor) -> None:
    """Fits I/Q offsets from readings over one spot at several sensor heights."""
    with st.expander("Multi-height calibration (I/Q offsets)", expanded=False):
        st.caption(
            "Hold the sensor over one spot at several heights and record it as its own line(s) "
            "with a height column. A half-space plus one offset per frequency and component is "
            "fitted (after Minsley et al., 2014). Download the offsets and load them as "
            "'Calibration offsets' in the sidebar (advanced settings)."
        )
        found = gem_io.find_channels(table.columns)
        labels = [lb for lb in found["Q"] if lb in found["I"]]
        if not labels:
            show_needs(NEEDS_IQ)
            return
        height_col = st.text_input("Height column", "Height", key=f"mh_col_{file_key}",
                                   help=SCIENCE_HELP["mh_height"])
        if height_col not in table.columns:
            st.info(f"The file has no column named '{height_col}'. Enter the name of the column "
                    "that holds the sensor height of each reading.")
            return
        lines = st.multiselect(
            "Calibration line(s)", sorted(gem_io.line_labels(table["Line"]).unique()),
            key=f"mh_lines_{file_key}", help=SCIENCE_HELP["mh_lines"],
        )
        if not lines:
            st.caption("Choose the calibration line(s) to fit the offsets.")
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
        show_table(offsets.round(3))
        download_button(
            "Offsets (.csv)", offsets.to_csv(index=False).encode(),
            f"{file_key}_iq_offsets.csv", key=f"mh_dl_{file_key}",
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


def transect_columns(
    traces, table: pd.DataFrame | None, key: str, label: str = "Transect"
) -> list[str] | None:
    """
    Trace columns of the transect the user picks when the lines of *table*
    lie at different positions (gem_io.track_groups); None when they are all
    repeat passes of one transect, or their positions are unknown.
    """
    try:
        tracks = gem_io.track_groups(table) if table is not None else None
    except (KeyError, ValueError):
        tracks = None
    if not tracks or len(set(tracks.values())) < 2:
        return None
    by_track: dict[int, list[str]] = {}
    for line, track in tracks.items():
        col = gem_io.trace_column(line)
        if col in traces:
            by_track.setdefault(track, []).append(col)
    if not by_track:
        return None
    names = {t: "Line " + ", ".join(gem_io.line_labels(pd.Series([c[len("Line_"):] for c in cols])))
             for t, cols in by_track.items()}
    track = st.selectbox(label, list(by_track), format_func=names.get, key=key,
                         help=SCIENCE_HELP["transect"])
    return by_track[track]


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
        n_layers = c1.slider("Layers", 5, 30, 15, key=f"inv_n_{file_key}", help=SCIENCE_HELP["inv_layers"])
        max_depth = c1.number_input("Depth to half-space (m)", 0.5, 50.0, 6.0, key=f"inv_d_{file_key}",
                                    help=SCIENCE_HELP["inv_depth"])
        first = c1.number_input("First layer (m)", 0.01, 5.0, 0.1, key=f"inv_f_{file_key}",
                                help=SCIENCE_HELP["inv_first"])
        alpha = c2.number_input("Regularisation (start)", 0.001, 10_000.0, 10.0, format="%.3g",
                                key=f"inv_a_{file_key}", help=SCIENCE_HELP["inv_alpha"])
        lateral = c2.number_input("Lateral constraint (0 = independent 1D)", 0.0, 100.0, 1.0,
                                  key=f"inv_l_{file_key}", help=SCIENCE_HELP["inv_lateral"])
        step = c2.number_input("Station spacing (m)", 0.1, 100.0, 1.0, key=f"inv_s_{file_key}",
                               help=SCIENCE_HELP["inv_step"])
        rel = c3.number_input("Relative error (%)", 0.1, 50.0, 3.0, key=f"inv_r_{file_key}",
                              help=SCIENCE_HELP["inv_rel"])
        floor = c3.number_input("Error floor (ppm)", 0.01, 1000.0, 1.0, key=f"inv_fl_{file_key}",
                                help=SCIENCE_HELP["inv_floor"])
        use_noise = c3.checkbox("Use measured noise (between passes)", True, key=f"inv_n2_{file_key}",
                                help=SCIENCE_HELP["inv_noise"])
        source = output_data.get("Q") or output_data.get("EC") or {}
        traces = sorted({c for df in source.values() for c in df.columns})
        columns = transect_columns(traces, table, f"inv_track_{file_key}")
        try:
            stations = inversion.station_data(output_data, scores, step, sensor, columns)
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
                  hash(stations.quadrature.tobytes()), tuple(columns or ()))
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
            show_figure(fig)
            download_button(
                "Model (.csv)", inversion.model_table(result, stations.distance).to_csv(index=False).encode(),
                f"{file_key}_inversion.csv", key=f"inv_dl_{file_key}",
            )
        if table is not None and output_data.get("EC"):
            errors_ec = {
                gem_io.channel_column("EC", lb): sc["mean_std"]
                for lb, sc in scores.get("EC", {}).items() if np.isfinite(sc["mean_std"])
            }
            try:
                data = inversion.emagpy_from_table(table, sensor, errors_ec or None)
            except ValueError as exc:
                st.info(str(exc))
                return
            download_button(
                "EMagPy survey (.csv)", data, f"{file_key}_emagpy.csv", key=f"inv_emagpy_{file_key}",
                help="Coil columns HCP{separation}f{frequency}h{height} in mS/m with *_err "
                     "columns from the measured noise of each frequency. EMagPy models a plain loop pair: "
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
                          key=f"sel_n_{file_key}", help=SCIENCE_HELP["sel_count"])
        limit = c2.slider("Redundant above |r|", 0.5, 1.0, 0.95, step=0.01, key=f"sel_r_{file_key}",
                          help=SCIENCE_HELP["sel_corr"])
        corr = interpret.profile_correlation(output_data)
        table = interpret.select_frequencies(scores, corr, count, limit)
        show_table(table.round(3))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="Correlation of mean profiles")
        fig.tight_layout()
        show_figure(fig, stretch=False)


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
            show_needs(NEEDS_IQ)
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
            line = c2.selectbox("Line", sorted(labels.unique()), key=f"sp_line_{file_key}",
                                help=SCIENCE_HELP["sp_line"])
            table = table[(labels == line).to_numpy()]
            d = pd.to_numeric(table[gem_io.DISTANCE_COL], errors="coerce")
            centre = float(c1.number_input("Distance (m)", value=float(d.median()), key=f"sp_d_{file_key}"))
        radius = c3.number_input("Radius (m)", 0.01, 1000.0, 1.0, key=f"sp_r_{file_key}",
                                 help=SCIENCE_HELP["sp_radius"])
        background = st.radio("Background", ["annulus", "survey"], horizontal=True,
                              format_func={"annulus": "Ring from radius to 2 × radius",
                                           "survey": "Whole survey"}.get,
                              key=f"sp_bg_{file_key}", help=SCIENCE_HELP["sp_background"])
        try:
            spectrum = interpret.anomaly_spectrum(table, centre, float(radius), background)
        except ValueError as exc:
            st.info(str(exc))
            return
        show_table(spectrum.round(3))
        fig = interpret.make_spectrum_figure(spectrum, f"Anomaly at {centre}")
        show_figure(fig)


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


def _read_samples(upload) -> pd.DataFrame:
    """Samples CSV with any common separator and encoding (UTF-8, Greek or Western Excel)."""
    data = upload.getvalue()
    for encoding in ("utf-8-sig", "cp1253", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except csv.Error as exc:                  # e.g. empty file: no separator to detect
            raise ValueError(f"The samples file could not be read: {exc}.") from exc
    raise ValueError("The samples file could not be read as text.")


@st.cache_data(show_spinner=False, max_entries=8)
def _zone_table(data: np.ndarray, m: float) -> pd.DataFrame:
    return soiltools.zone_indices(data, range(2, 7), m)


@st.cache_data(show_spinner=False, max_entries=8)
def _zone_fit(data: np.ndarray, c: int, m: float) -> np.ndarray:
    return soiltools.fuzzy_cmeans(data, c, m)[0]


PLAN_VIEW_WIDTH = 7.0       # in; plan-view maps of the soil tools (height follows the survey's shape)


def plan_view(x: np.ndarray, y: np.ndarray, width: float = PLAN_VIEW_WIDTH):
    """
    Figure and axes for an equal-axis map of the readings, sized to the
    survey's shape (height 3-9 in) so the map fills the figure instead of
    sitting as a strip in a fixed landscape frame.
    """
    dx = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    dy = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    fig, ax = plt.subplots(figsize=(width, float(np.clip(0.85 * width * dy / dx, 3.0, 9.0))))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return fig, ax


def zone_figure(x: np.ndarray, y: np.ndarray, zone: np.ndarray, n_zones: int):
    """Map of the management zones with one colour, and one colour-bar entry, per zone used."""
    from matplotlib.colors import ListedColormap

    fig, ax = plan_view(x, y)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:n_zones])
    sc = ax.scatter(x, y, c=zone, s=4, cmap=cmap, vmin=0.5, vmax=n_zones + 0.5)
    fig.colorbar(sc, ax=ax, ticks=range(1, n_zones + 1), label="Zone")
    fig.tight_layout()
    return fig


def show_plan_view(fig) -> None:
    """Shows a plan-view map at its own size: stretched to the page width it looks zoomed in."""
    show_figure(fig, stretch=False)


def render_soil_tools(table: pd.DataFrame, file_key: str) -> None:
    """Sampling design, calibration of a soil property, and management zones from EC channels."""
    with st.expander("Soil tools (sampling design, calibration, management zones)", expanded=False):
        ec_cols = list(gem_io.find_channels(table.columns)["EC"].values())
        if not ec_cols:
            show_needs("The soil tools work on EC channels (EC…Hz[mS/m]); this file has none. "
                    "Turn on 'Recompute EC / MS from I / Q' under Sensor geometry (advanced "
                    "settings) to add them.")
            return
        try:
            x, y, origin = corrections.xy_metres(table)
        except ctr.ContouringError:
            show_needs(NEEDS_XY)
            return
        chans = st.multiselect("EC channels", ec_cols, default=ec_cols[:2], key=f"soil_ch_{file_key}",
                               help=SCIENCE_HELP["soil_channels"])
        if not chans:
            st.caption("Choose at least one EC channel.")
            return
        features = table[chans].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        t_design, t_cal, t_zones = st.tabs(["Sampling design", "Calibration", "Management zones"])

        with t_design:
            st.caption(
                "Sampling sites that span the conductivity range and are spread over the field "
                "(response-surface design after Lesch, 2005; ESAP-RSSD)."
            )
            n_sites = st.number_input("Sites", 4, 60, 12, key=f"soil_n_{file_key}",
                                      help=SCIENCE_HELP["soil_sites"])
            try:
                sites = soiltools.sampling_design(np.column_stack([x, y]), features, int(n_sites))
            except ValueError as exc:
                st.info(str(exc))
            else:
                extra = _lonlat_columns(sites["x"].to_numpy(), sites["y"].to_numpy(), origin)
                sites = sites.assign(**extra)
                show_table(sites.round(4))
                fig, ax = plan_view(x, y)
                ax.scatter(x, y, s=2, c="0.7")
                ax.scatter(sites["x"], sites["y"], c="red", s=25)
                for k, (sx, sy) in enumerate(zip(sites["x"], sites["y"]), 1):
                    ax.annotate(str(k), (sx, sy), fontsize=8, xytext=(3, 3), textcoords="offset points")
                fig.tight_layout()
                show_plan_view(fig)
                download_button("Sites (.csv)", sites.to_csv(index=False).encode(),
                                f"{file_key}_sampling_sites.csv", key=f"soil_dl_sites_{file_key}")

        with t_cal:
            st.caption(
                "Regression of a measured soil property on the conductivity at the sampling "
                "sites (Lesch et al., 1995; ESAP-Calibrate). The samples file has the survey's "
                "coordinate columns and one column per measured property."
            )
            upload = st.file_uploader("Soil samples (.csv)", type=["csv"], key=f"soil_up_{file_key}")
            if upload is not None:
                try:
                    samples = _read_samples(upload)
                except ValueError as exc:            # includes pandas' ParserError
                    st.info(f"Could not read the samples file: {exc}")
                    samples = pd.DataFrame()
                skip = {"x", "y", "lat", "latitude", "lon", "long", "longitude", "row", "pc1", "pc2",
                        "target", "line", "id", "zone"}
                props = [c for c in samples.columns if str(c).lower() not in skip
                         and pd.to_numeric(samples[c], errors="coerce").notna().any()]
                if not props:
                    st.info("No numeric property column in the samples file.")
                else:
                    c1, c2 = st.columns(2)
                    prop = c1.selectbox("Property", props, key=f"soil_prop_{file_key}",
                                        help=SCIENCE_HELP["soil_property"])
                    radius = c1.number_input("Matching radius (m)", 0.1, 100.0, 2.0, key=f"soil_rad_{file_key}",
                                             help=SCIENCE_HELP["soil_radius"])
                    log_p = c2.checkbox("Fit the property on a log scale", True, key=f"soil_log_{file_key}",
                                        help=SCIENCE_HELP["soil_log"])
                    trend = c2.checkbox("Add a trend surface (x, y)", False, key=f"soil_tr_{file_key}",
                                        help=SCIENCE_HELP["soil_trend"])
                    try:
                        model = soiltools.fit_property_model(table, samples, prop, chans, radius, log_p, trend)
                    except (ValueError, ctr.ContouringError) as exc:
                        st.info(str(exc))
                    else:
                        st.write(f"R² **{model.r2:.3f}**, RMSE {model.rmse:.4g} "
                                 f"({'log scale' if log_p else 'property units'}), {model.n} samples.")
                        st.dataframe(model.coefficients.rename("coefficient").to_frame().map("{:.6g}".format),
                                     width="content")
                        if not model.log_channels:
                            st.caption("Some samples have EC ≤ 0, so EC enters linearly, not as ln(EC).")
                        column = f"Predicted {prop}"
                        predicted = table.assign(**{column: soiltools.predict_property(model, table)})
                        try:
                            result = _property_map(predicted, column)
                            fig = ctr.make_area_map_figure(result, column, column)
                        except ctr.ContouringError as exc:
                            st.info(str(exc))
                        else:
                            show_figure(fig)
                        out = pd.DataFrame({"x": x, "y": y, **_lonlat_columns(x, y, origin),
                                            column: predicted[column]})
                        download_button("Predictions (.csv)", out.to_csv(index=False).encode(),
                                        f"{file_key}_{prop}_predicted.csv", key=f"soil_dl_pred_{file_key}")

        with t_zones:
            st.caption(
                "Fuzzy c-means on the standardised channels (clipped to their 1st–99th "
                "percentiles; Bezdek, 1981), as in Management "
                "Zone Analyst (Fridgen et al., 2004). The lowest FPI and MPE (modified partition "
                "entropy, Boydell & McBratney, 2002) suggest the number of zones."
            )
            ok = np.all(np.isfinite(features), axis=1) & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                st.info("Fewer than 10 readings have every chosen channel and coordinates: "
                        "too few to form zones.")
                return
            data = soiltools.robust_standardise(features[ok])
            m = st.slider("Fuzziness exponent", 1.1, 3.0, 1.3, step=0.05, key=f"soil_m_{file_key}",
                          help=SCIENCE_HELP["soil_m"])
            rng = np.random.default_rng(0)
            subset = data if len(data) <= ZONE_SAMPLE else data[rng.choice(len(data), ZONE_SAMPLE, replace=False)]
            indices = _zone_table(subset, float(m))
            show_table(indices.round(4))
            best = int(indices.loc[indices["FPI"].idxmin(), "Zones"])
            c = st.slider("Zones", 2, 6, best, key=f"soil_c_{file_key}", help=SCIENCE_HELP["soil_zones"])
            u = _zone_fit(data, int(c), float(m))
            zone = u.argmax(axis=1) + 1
            if (u.sum(axis=0) < 0.01 * len(u)).any():
                st.warning("A zone holds less than 1 % of the readings: try fewer zones or a larger exponent.")
            show_plan_view(zone_figure(x[ok], y[ok], zone, int(c)))
            out = pd.DataFrame({"x": x[ok], "y": y[ok], **_lonlat_columns(x[ok], y[ok], origin), "zone": zone,
                                **{f"membership_{k + 1}": u[:, k] for k in range(int(c))}})
            download_button("Zones (.csv)", out.to_csv(index=False).encode(),
                            f"{file_key}_zones.csv", key=f"soil_dl_zones_{file_key}")
