"""Empty state, sample survey, page layout and consistent wording, headless (AppTest)."""
import matplotlib
matplotlib.use("Agg")
from streamlit.testing.v1 import AppTest  # noqa: E402

import RIs_v2 as R  # noqa: E402
import ui_tools  # noqa: E402
from test_app_gem_export import _full_export  # noqa: E402


def test_plural():
    assert ui_tools.plural(1, "file") == "1 file"
    assert ui_tools.plural(3, "file") == "3 files"
    assert ui_tools.plural(2, "frequency", "frequencies") == "2 frequencies"
    assert ui_tools.plural(1_250, "reading") == "1,250 readings"


def test_item_noun_follows_the_data():
    assert R.item_noun("EC", True) == ("frequency", "frequencies")
    assert R.item_noun("AUX", True) == ("channel", "channels")
    assert R.item_noun("EC", False) == ("sheet", "sheets")


def test_empty_page_offers_the_sample_and_survey_planning():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Upload a GEM-2 export to begin" in i.value for i in at.info)
    assert at.button(key="sample_load")
    assert "Survey planning" in [h.value for h in at.subheader]


def test_sample_survey_loads_and_closes():
    at = AppTest.from_file("RIs_v2.py", default_timeout=300)
    at.run()
    at.button(key="sample_load").click().run()
    assert not at.exception
    assert R.SAMPLE_FILE.name in [h.value for h in at.subheader]
    assert "AUX (2 channels)" in [t.label for t in at.tabs]
    at.button(key="sample_close").click().run()
    assert not at.exception
    assert R.SAMPLE_FILE.name not in [h.value for h in at.subheader]


def test_results_come_before_calibration_and_further_analysis():
    at = AppTest.from_function(_full_export, default_timeout=120)
    at.run()
    assert not at.exception
    headings = [m.value for m in at.markdown if m.value.startswith("####")]
    assert headings.index("#### Downloads") < headings.index("#### Calibration")
    assert headings.index("#### Calibration") < headings.index("#### Further analysis")


def test_aux_detail_plots_are_per_channel():
    at = AppTest.from_function(_full_export, default_timeout=120)
    at.run()
    labels = [e.label for e in at.expander]
    assert labels.count("Per-frequency detail plots") == 4     # EC, MS, I, Q
    assert labels.count("Per-channel detail plots") == 1       # AUX


def test_downloads_share_one_icon():
    at = AppTest.from_function(_full_export, kwargs={"scoring": True}, default_timeout=120)
    at.run()
    buttons = at.get("download_button")
    assert buttons and all(b.proto.icon == ui_tools.DOWNLOAD_ICON for b in buttons)


# ── Data summary ─────────────────────────────────────────────────────────────

def _table(**extra):
    import pandas as pd
    import gem_io
    n = 20
    base = {"Line": [0] * 10 + [1] * 10, gem_io.DISTANCE_COL: list(range(10)) * 2,
            "EC1525Hz[mS/m]": [20.0] * n, "EC5325Hz[mS/m]": [21.0] * n}
    return pd.DataFrame({**base, **extra})


def test_data_summary_counts_and_tools():
    s = ui_tools.data_summary(_table())
    assert (s.lines, s.readings, s.line_length) == (2, 20, 18.0)
    assert s.frequencies == ("1525Hz", "5325Hz") and s.duration is None
    tools = {name: ok for name, ok, _ in ui_tools.tool_availability(s)}
    assert tools["Pseudo-sections"] and tools["Layered-earth inversion"]
    assert not tools["Area maps"] and not tools["Base-station drift"] and not tools["Soil tools"]


def test_data_summary_with_coordinates_time_and_iq():
    s = ui_tools.data_summary(_table(
        X=[float(i % 10) for i in range(20)], Y=[float(i // 10) for i in range(20)],
        **{"Time[ms]": [36_000_000.0 + 1000 * i for i in range(20)],
           "I_1525Hz": [1.0] * 20, "Q_1525Hz": [2.0] * 20},
    ))
    assert s.has_xy and s.has_iq and s.duration == 19.0
    assert all(ok for _, ok, _ in ui_tools.tool_availability(s))


def test_format_duration():
    assert ui_tools.format_duration(None) == "—"
    assert ui_tools.format_duration(45) == "45 s"
    assert ui_tools.format_duration(720) == "12 min"
    assert ui_tools.format_duration(7500) == "2 h 05 min"


def test_sample_survey_shows_the_summary_card():
    at = AppTest.from_file("RIs_v2.py", default_timeout=300)
    at.run()
    at.button(key="sample_load").click().run()
    assert any("**31 lines**" in m.value and "**2,480 m** along the lines" in m.value
               for m in at.markdown)
    assert any("Available:" in c.value and "Area maps" in c.value for c in at.caption)


def test_summary_line():
    s = ui_tools.data_summary(_table())
    assert ui_tools.summary_line(s) == (
        "**2 lines** · **20 readings** · **18 m** along the lines · "
        "**2 frequencies** (1525, 5325 Hz)"
    )


# ── Changed settings and reset ───────────────────────────────────────────────

def test_corrections_describe_their_enabled_steps():
    import corrections
    assert corrections.CorrectionSettings().describe() == []
    c = corrections.CorrectionSettings(despike=True, smooth_window=5, lag_seconds=-0.3)
    assert c.describe() == ["despike", "GPS lag -0.3 s", "running mean (5 readings)"]


def test_changed_settings_lists_only_what_differs():
    import corrections
    import emphysics
    import pipeline
    assert R.changed_settings(pipeline.PrepSettings(), R.DEFAULT_DISTANCE_STEP, "linear") == []
    prep = pipeline.PrepSettings(exclude_lines=("B1",), sensor=emphysics.Sensor(height=0.4),
                                 corrections=corrections.CorrectionSettings(despike=True))
    assert R.changed_settings(prep, 1.0, "pchip") == [
        "distance step 1 m", "pchip interpolation", "1 line left out", "sensor height 0.4 m",
        "despike",
    ]


def test_changed_settings_are_shown_and_reset():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert any("at their defaults" in c.value for c in at.sidebar.caption)
    at.slider(key="cor_smooth").set_value(5).run()
    assert any("running mean (5 readings)" in c.value for c in at.sidebar.caption)
    at.button(key="settings_reset_btn").click().run()
    assert not at.exception
    assert any("at their defaults" in c.value for c in at.sidebar.caption)
    assert all(s.value == 0 for s in at.slider if s.label.startswith("Running mean"))


# ── Basic and advanced settings ──────────────────────────────────────────────

def test_basic_mode_hides_expert_settings():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    keys = {w.key for w in at.number_input}
    assert "sen_h" in keys and "sen_sep" not in keys and "cor_pca" not in keys
    at.toggle(key=R.ADVANCED_KEY).set_value(True).run()
    keys = {w.key for w in at.number_input}
    assert {"sen_sep", "cor_pca", "cor_bg"} <= keys


def test_hidden_settings_return_to_their_defaults():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    at.toggle(key=R.ADVANCED_KEY).set_value(True).run()
    at.number_input(key="cor_pca").set_value(2).run()
    assert any("PCA (2 components)" in c.value for c in at.sidebar.caption)
    at.toggle(key=R.ADVANCED_KEY).set_value(False).run()
    assert any("at their defaults" in c.value for c in at.sidebar.caption)
