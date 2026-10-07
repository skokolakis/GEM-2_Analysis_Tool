"""Scoring toggle and full GEM-2 export handling, headless (streamlit.testing.AppTest)."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import RIs_v2 as R  # noqa: E402


def _full_export(scoring: bool = False, distance_method: str = "Y", status_bad: int = 0):
    """Script body run by AppTest; all imports must be local."""
    import numpy as np
    import pandas as pd

    import pipeline
    import RIs_v2 as R

    rows = []
    for line in range(3):
        ys = np.arange(0.0, 30.0, 0.5)
        bump = np.exp(-((ys - 15) ** 2) / 10)
        marks = np.where(ys % 10 == 0, 1 + ys // 10, 0)
        rows.append(pd.DataFrame({
            "Line": line, "Sample": np.arange(ys.size), "X": 0.1 * line, "Y": ys,
            "Mark": marks, "Status": 0, "PowerLn": 0.2 + 0.01 * ys,
            "I_1525Hz": 300 + 20 * bump, "Q_1525Hz": 150 + 40 * bump,
            "I_9825Hz": 320 + 20 * bump, "Q_9825Hz": 600 + 90 * bump, "QSum": 750 + 130 * bump,
            "EC1525Hz[mS/m]": 20 + 10 * bump, "EC9825Hz[mS/m]": 22 + 8 * bump,
            "MSusc1525Hz[1/1000]": 0.5 + 0.2 * bump, "MSusc9825Hz[1/1000]": 0.6 + 0.2 * bump,
        }))
    table = pd.concat(rows, ignore_index=True)
    table.loc[: status_bad - 1, "Status"] = 3
    data = table.to_csv(index=False).encode()
    prep = pipeline.PrepSettings(distance_method=distance_method)
    R.render_gem_results(data, "survey.csv", 0.5, "linear", None, scoring, prep)


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_is_gem_format_accepts_iq_only_export():
    df = pd.DataFrame({"Line": [0], "Y": [0.0], "I_1525Hz": [1.0], "Q_1525Hz": [2.0]})
    assert R.is_gem_format(df)
    assert not R.is_gem_format(pd.DataFrame({"Line": [0], "Y": [0.0], "PowerLn": [1.0]}))


def test_overview_legend_without_scores_and_markers():
    interp = pd.DataFrame({"a": [1.0, 2.0], "b": [1.5, 2.5]}, index=[0.0, 1.0])
    fig = R.make_overview_figure(
        {"f": interp}, {"f": {"score": 3.0}}, "EC", "x", is_gem=True,
        show_scores=False, markers=[0.5],
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["f", "Event marker"]
    plt.close(fig)


def test_batch_without_scores_has_no_scores_sheet():
    interp = pd.DataFrame({"a": [1.0, 2.0]}, index=[0.0, 1.0])
    entry = {"stem": "f", "mode": "EC", "output_data": {"x": interp},
             "scores": {"x": {"score": 1.0, "amplitude": 1.0, "mean_std": 1.0,
                              "noise_method": "between-trace", "n_traces": 1}}}
    xlsx = R.build_batch_xlsx([entry], include_scores=False)
    assert pd.ExcelFile(io.BytesIO(xlsx)).sheet_names == ["f_EC"]


def test_excel_download_accepts_bracketed_channel_names():
    interp = pd.DataFrame({"a": [1.0, 2.0]}, index=[0.0, 1.0])
    xlsx = R.build_excel_download({"TotalEC[mS/m]": interp})
    assert pd.ExcelFile(io.BytesIO(xlsx)).sheet_names == ["TotalEC_mS_m_"]


def test_batch_sheet_names_are_sanitised():
    interp = pd.DataFrame({"a": [1.0, 2.0]}, index=[0.0, 1.0])
    entry = {"stem": "site[1]", "mode": "EC", "output_data": {"x": interp}, "scores": {}}
    xlsx = R.build_batch_xlsx([entry], include_scores=False)
    assert pd.ExcelFile(io.BytesIO(xlsx)).sheet_names == ["site_1__EC"]


def test_scoring_off_shows_channels_not_ranking():
    at = AppTest.from_function(_full_export, kwargs={"scoring": False}, default_timeout=120)
    at.run()
    assert not at.exception
    texts = _texts(at)
    assert "### Channels" in texts and "Frequency Ranking" not in texts
    assert [t.label for t in at.tabs][:5] == [
        "EC (2 frequencies)", "MS (2 frequencies)", "I (2 frequencies)",
        "Q (2 frequencies)", "AUX (2 channels)",
    ]


def test_scoring_on_ranks_frequency_modes_only():
    at = AppTest.from_function(_full_export, kwargs={"scoring": True}, default_timeout=120)
    at.run()
    assert not at.exception
    texts = _texts(at)
    assert texts.count("Frequency Ranking") == 4      # EC, MS, I, Q — not AUX
    assert "### Channels" in texts                    # AUX channels are listed


def test_flagged_readings_are_reported():
    at = AppTest.from_function(_full_export, kwargs={"status_bad": 4}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Dropped 4 reading(s)" in w.value for w in at.warning)


def test_projection_distance_runs():
    at = AppTest.from_function(
        _full_export, kwargs={"distance_method": "projection"}, default_timeout=120
    )
    at.run()
    assert not at.exception


def test_main_page_defaults_to_scoring_off():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert at.toggle(key="scoring").value is False


def _small_gem_csv():
    import numpy as np

    rows = [pd.DataFrame({"Line": k, "X": 2.0 * k, "Y": np.arange(0.0, 20.0, 0.5),
                          "EC1525Hz[mS/m]": 10 + k + np.sin(np.arange(40) / 5)}) for k in range(3)]
    return pd.concat(rows, ignore_index=True).to_csv(index=False).encode()


def _two_uploads_app():
    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows = [pd.DataFrame({"Line": k, "X": 2.0 * k, "Y": np.arange(0.0, 20.0, 0.5),
                          "EC1525Hz[mS/m]": 10 + k + np.sin(np.arange(40) / 5)}) for k in range(3)]
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    R.render_gem_results(data, "site.csv", 0.5, "linear", None, False, None, file_key="site")
    R.render_gem_results(data, "site.csv", 0.5, "linear", None, False, None, file_key="site_2")


def test_unique_file_keys_avoid_widget_collisions():
    taken = set()
    assert R.unique_file_key("site", taken, ("EC",)) == "site"
    assert R.unique_file_key("site", taken, ("EC",)) == "site_2"
    assert R.unique_file_key("site_EC", taken) == "site_EC_2"      # a legacy file named like a GEM mode key
    at = AppTest.from_function(_two_uploads_app, default_timeout=180)
    at.run()
    assert not at.exception


def test_gem_file_is_prepared_once(monkeypatch):
    import pipeline

    calls = []
    real = pipeline.prepare_gem_table
    monkeypatch.setattr(pipeline, "prepare_gem_table", lambda raw, prep=None: calls.append(1) or real(raw, prep))
    R.process_gem_file.clear()
    R.prepared_table.clear()
    prep = pipeline.PrepSettings(drop_flagged=False)
    R.process_gem_file(_small_gem_csv(), "once.csv", 0.5, "linear", prep)
    R.prepared_table(_small_gem_csv(), "once.csv", prep)
    assert len(calls) == 1
