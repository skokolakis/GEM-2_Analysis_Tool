"""Next-step buttons, distance-step hints, doc links, file picker, settings files, tour."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import RIs_v2 as R  # noqa: E402
import ui_tools  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _gem_app(distance_step: float = 0.5, scoring: bool = False):
    """Script body run by AppTest: the three-line survey of test_app_gem_export, all maps off."""
    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows = []
    for line in range(3):
        ys = np.arange(0.0, 30.0, 0.5)
        bump = np.exp(-((ys - 15) ** 2) / 10)
        rows.append(pd.DataFrame({
            "Line": line, "X": 0.1 * line, "Y": ys, "Status": 0,
            "I_1525Hz": 300 + 20 * bump, "Q_1525Hz": 150 + 40 * bump,
            "I_9825Hz": 320 + 20 * bump, "Q_9825Hz": 600 + 90 * bump,
            "EC1525Hz[mS/m]": 20 + 10 * bump, "EC9825Hz[mS/m]": 22 + 8 * bump,
        }))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    R.render_gem_results(data, "survey.csv", distance_step, "linear", R.ContourSettings(),
                         scoring, None, file_key="survey")


# ── Next steps ───────────────────────────────────────────────────────────────

def test_next_steps_offer_what_is_switched_off():
    off = R.ContourSettings()
    assert [s for _, s, _ in R.next_steps("EC", 2, False, off, True)] == ["scoring", "ct_area", "ct_pseudo"]
    assert [s for _, s, _ in R.next_steps("AUX", 2, False, off, True)] == ["ct_area"]
    assert [s for _, s, _ in R.next_steps("EC", 2, True, off, False)] == ["ct_pseudo"]
    on = R.ContourSettings(area_map=True, pseudosection=True)
    assert R.next_steps("EC", 2, True, on, True) == []
    assert [s for _, s, _ in R.next_steps("EC", 2, False, None, True)] == ["scoring"]


def test_next_step_button_switches_the_setting_on():
    at = AppTest.from_function(_gem_app, default_timeout=120)
    at.run()
    assert not at.exception
    at.button(key="next_ct_area_survey_EC").click().run()
    assert at.session_state["ct_area"] is True


# ── Distance step ────────────────────────────────────────────────────────────

def test_median_spacing_and_suggested_step():
    d = pd.Series([0.0, 0.1, 0.2, 0.2, 0.3, 5.0, 5.25, 5.5])
    lines = pd.Series([0, 0, 0, 0, 0, 1, 1, 1])
    assert ui_tools.median_spacing(d, lines) == pytest.approx(0.1)
    assert ui_tools.suggested_step(0.13) == 0.1
    assert ui_tools.suggested_step(0.33) == 0.5
    assert ui_tools.suggested_step(2.4) == 2.0
    assert ui_tools.suggested_step(0.001) == 0.01      # the distance-step minimum
    assert ui_tools.suggested_step(float("nan")) is None
    assert not ui_tools.step_mismatch(0.5, 0.1)
    assert ui_tools.step_mismatch(0.6, 0.1) and ui_tools.step_mismatch(0.04, 0.1)


def test_mismatched_step_is_flagged_and_fixed_in_one_click():
    at = AppTest.from_function(_gem_app, kwargs={"distance_step": 0.05}, default_timeout=120)
    at.run()
    assert any("**0.5 m** apart" in i.value for i in at.info)
    at.button(key="step_hint_survey").click().run()
    assert at.session_state["prof_step"] == 0.5


def test_too_coarse_step_explains_the_failure():
    at = AppTest.from_function(_gem_app, kwargs={"distance_step": 50.0}, default_timeout=120)
    at.run()
    assert any("No usable frequencies" in e.value and "troubleshooting" in e.value for e in at.error)
    assert at.button(key="step_hint_survey").label == "Use 0.5 m"


# ── Links to the documentation ───────────────────────────────────────────────

def _anchors(page: str) -> set[str]:
    """GitHub's anchors of the headings of a docs page."""
    text = (DOCS / f"{page}.md").read_text(encoding="utf-8")
    heads = re.findall(r"^#+ (.+)$", text, flags=re.M)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in heads}


def test_every_doc_link_points_at_a_real_section():
    source = Path(R.__file__).read_text(encoding="utf-8") + Path(ui_tools.__file__).read_text(encoding="utf-8")
    links = re.findall(r"doc_link\(['\"]([\w-]+)['\"],\s*['\"]([\w-]+)['\"]", source)
    links += [("troubleshooting", ui_tools.NEEDS_ANCHOR)]
    assert len(links) >= 6
    for page, anchor in links:
        assert anchor in _anchors(page), (page, anchor)


def test_doc_link():
    assert ui_tools.doc_link("mapping", "combined-surveys", "Maps") == (
        f"[Maps]({ui_tools.DOCS_URL}/mapping.md#combined-surveys)")


# ── File picker ──────────────────────────────────────────────────────────────

def _picker_app():
    import streamlit as st

    import RIs_v2 as R

    st.write(f"shown={R.pick_file(['a.csv', 'b.csv', 'c.xlsx'])}")


def test_file_picker_shows_one_file_at_a_time():
    at = AppTest.from_function(_picker_app, default_timeout=60)
    at.run()
    assert "shown=0" in at.markdown[-1].value
    picker = at.get("button_group")[0]
    picker.set_value([2]).run()             # the test harness takes a list for one choice
    assert "shown=2" in at.markdown[-1].value


# ── Settings files ───────────────────────────────────────────────────────────

def test_settings_round_trip():
    state = {"cor_lag": 0.5, "ct_area": True, "ct_percentiles": (2.0, 98.0), "advanced": True,
             "ct_freq_site_EC": "1525Hz", "cor_offsets": None, "sample_load": False}
    saved = R.current_settings(state)
    assert saved == {"advanced": True, "cor_lag": 0.5, "ct_area": True, "ct_percentiles": [2.0, 98.0]}
    loaded = R.parse_settings(R.settings_json(saved))
    assert loaded["ct_percentiles"] == (2.0, 98.0)
    target = {}
    R.apply_settings(target, loaded)
    assert target[R.RESET_KEY] == 1 and target["cor_lag~1"] == 0.5 and target["advanced"] is True


def test_settings_after_a_reset_use_the_current_widgets():
    state = {R.RESET_KEY: 2, "cor_lag~1": 0.3, "cor_lag~2": 0.7}
    assert R.current_settings(state) == {"cor_lag": 0.7}


def test_saved_choices_no_longer_offered_keep_the_default():
    saved = {"prof_interp": "akima", "ct_method": "kriging", "prof_step": 0.5}
    loaded = R.parse_settings(R.settings_json(saved))
    assert loaded == {"ct_method": "kriging", "prof_step": 0.5}


def test_foreign_files_are_refused():
    with pytest.raises(ValueError, match="not a settings file"):
        R.parse_settings(b'{"something": 1}')
    with pytest.raises(ValueError, match="not a settings file"):
        R.parse_settings(b"not json")


def test_settings_expander_is_in_the_sidebar():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert "Save or load settings" in [e.label for e in at.sidebar.expander]
    assert any(b.label == "Settings (.json)" for b in at.sidebar.get("download_button"))
