"""Corrections sidebar and GPS-lag panel, headless (streamlit.testing.AppTest)."""
from streamlit.testing.v1 import AppTest


def _serpentine_app(smooth: int = 0):
    """Script body run by AppTest; all imports must be local."""
    import numpy as np
    import pandas as pd

    import corrections
    import pipeline
    import RIs_v2 as R

    rows, t0 = [], 36_000.0
    for line in range(6):
        ys = np.linspace(0.0, 40.0, 81)
        if line % 2:
            ys = ys[::-1]
        t = t0 + 0.25 * np.arange(ys.size)
        t0 = t[-1] + 5.0
        rows.append(pd.DataFrame({
            "Line": line, "X": 2.0 * line, "Y": ys, "Time[ms]": 1000 * t,
            "EC1525Hz[mS/m]": 20 + 5 * np.sin(ys / 6.0),
            "EC9825Hz[mS/m]": 22 + 5 * np.sin(ys / 6.0),
        }))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    prep = pipeline.PrepSettings(
        distance_method="projection",
        corrections=corrections.CorrectionSettings(smooth_window=smooth),
    )
    R.render_gem_results(data, "serp.csv", 0.5, "linear", None, False, prep)


def test_correction_messages_are_listed():
    at = AppTest.from_function(_serpentine_app, kwargs={"smooth": 5}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Running mean over 5 readings" in w.value for w in at.warning)


def test_lag_estimate_button():
    at = AppTest.from_function(_serpentine_app, default_timeout=120)
    at.run()
    at.button(key="lag_btn_serp").click().run()
    assert not at.exception
    assert any("Best lag" in m.value for m in at.markdown)


def test_smoothing_with_scoring_warns_in_sidebar():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    at.toggle(key="scoring").set_value(True)
    at.slider(key="cor_smooth").set_value(5)
    at.run()
    assert not at.exception
    assert any("lower the noise σ" in w.value for w in at.sidebar.warning)
