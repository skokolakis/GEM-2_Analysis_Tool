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
