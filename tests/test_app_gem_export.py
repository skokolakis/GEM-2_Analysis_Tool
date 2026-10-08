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
    assert "#### Channels" in texts and "Frequency ranking" not in texts
    assert [t.label for t in at.tabs][:5] == [
        "EC (2 frequencies)", "MS (2 frequencies)", "I (2 frequencies)",
        "Q (2 frequencies)", "AUX (2 channels)",
    ]


def test_scoring_on_ranks_frequency_modes_only():
    at = AppTest.from_function(_full_export, kwargs={"scoring": True}, default_timeout=120)
    at.run()
    assert not at.exception
    texts = _texts(at)
    assert texts.count("Frequency ranking") == 4      # EC, MS, I, Q — not AUX
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


def _gbf_csv():
    import numpy as np

    rows, t0 = [], 0.0
    for k in range(3):
        s = np.linspace(0, 10, 41)
        t = t0 + 100.0 * np.arange(41)
        t0 = t[-1] + 5000.0
        rows.append(pd.DataFrame({"X": float(k), "Y": s, "Mark": 0, "Status": 0, "Time[ms]": t,
                                  "Ip_15270Hz": 1300.0 + k + np.sin(s), "Qd_15270Hz": 1500.0 + s,
                                  "EC15270Hz[mS/m]": 37.0 + s + 0.1 * k,
                                  "MSusc15270Hz[1/1000]": -12.0 + np.cos(s)}))
    return pd.concat(rows, ignore_index=True).to_csv(index=False).encode()


def test_gbf_converted_export_loads_with_lines_and_iq():
    import pipeline

    data = _gbf_csv()
    assert R.is_gem_format(pd.read_csv(io.BytesIO(data), nrows=5))
    out, scores, warnings = R.process_gem_file(data, "gbf.csv", 0.5, "linear", pipeline.PrepSettings())
    assert {m for m, v in out.items() if v} == {"EC", "MS", "I", "Q"}
    assert out["Q"]["15270Hz"].shape[1] == 3                        # three lines
    assert any("constant X" in w for w in warnings)


def test_graph_editor_axis_choices_and_readings_figure():
    import pipeline

    table, _ = pipeline.prepare_gem_table(pd.read_csv(io.BytesIO(_gbf_csv())))
    xs, ys = R.plot_axis_options(table, "Q")
    assert xs[0] == R.PROFILE_X and ys[0] == R.PROFILE_Y
    assert {"X", "Y", R.TIME_X} <= set(xs) and "EC15270Hz[mS/m]" in xs
    assert ys[1] == "Q_15270Hz" and "EC15270Hz[mS/m]" in ys        # the tab's own channels first
    assert R.plot_axis_options(None, "EC") == ([R.PROFILE_X], [R.PROFILE_Y])
    opts = R.GraphOptions(x_column=R.TIME_X, y_column="EC15270Hz[mS/m]")
    fig = R.make_readings_figure(table, opts, "t")
    assert len(fig.axes[0].lines) == 3 and fig.axes[0].get_xlabel() == R.TIME_X
    plt.close(fig)
    cross = R.make_readings_figure(table, R.GraphOptions(x_column="I_15270Hz", y_column="Q_15270Hz"), "t")
    assert len(cross.axes[0].collections) == 3                       # not monotonic: points, not lines
    plt.close(cross)


def _gbf_app():
    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows, t0 = [], 0.0
    for k in range(3):
        s = np.linspace(0, 10, 41)
        t = t0 + 100.0 * np.arange(41)
        t0 = t[-1] + 5000.0
        rows.append(pd.DataFrame({"X": float(k), "Y": s, "Mark": 0, "Status": 0, "Time[ms]": t,
                                  "Ip_15270Hz": 1300.0 + k + np.sin(s), "Qd_15270Hz": 1500.0 + s,
                                  "EC15270Hz[mS/m]": 37.0 + s + 0.1 * k,
                                  "MSusc15270Hz[1/1000]": -12.0 + np.cos(s)}))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    R.render_gem_results(data, "gbf.csv", 0.5, "linear", None, False, None)


def test_graph_editor_plots_chosen_columns_in_the_app():
    at = AppTest.from_function(_gbf_app, default_timeout=180)
    at.run()
    assert not at.exception
    at.selectbox(key="ge_xcol_gbf_EC").set_value(R.TIME_X).run()
    at.selectbox(key="ge_ycol_gbf_EC").set_value("EC15270Hz[mS/m]").run()
    assert not at.exception
    assert any("EC15270Hz[mS/m] against Time (s)" in m.value for m in at.markdown)


def test_science_choices_have_help_tooltips():
    import inspect

    import ui_tools

    for module in (R, ui_tools):
        source = inspect.getsource(module)
        unused = [k for k in module.SCIENCE_HELP if f'SCIENCE_HELP["{k}"]' not in source]
        assert not unused
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert at.toggle(key="ct_area").help == R.SCIENCE_HELP["area_map"]
    assert at.number_input(key="sen_sep").help == R.SCIENCE_HELP["separation"]
    assert at.checkbox(key="cor_despike").help == R.SCIENCE_HELP["despike"]
