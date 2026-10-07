"""Frequency selection, magnetic viscosity and anomaly spectra in the app (AppTest)."""
from streamlit.testing.v1 import AppTest


def _app(scoring: bool = True, viscosity: bool = False):
    import numpy as np
    import pandas as pd

    import pipeline
    import RIs_v2 as R

    rows = []
    for k in range(5):
        ys = np.arange(0.0, 20.0, 0.5)
        xs = np.full_like(ys, float(k))
        r = np.hypot(xs - 2, ys - 10)
        bump = np.exp(-r ** 2)
        part = pd.DataFrame({"Line": k, "X": xs, "Y": ys})
        for f, s in ((1525.0, 1.0), (5325.0, 1.02), (18325.0, -0.5)):
            part[f"I_{f:g}Hz"] = 100 + 10 * s * bump + 0.1 * np.sin(ys * f)
            part[f"Q_{f:g}Hz"] = 50 * f / 1525 + 20 * s * bump + 0.1 * np.cos(ys * f)
        rows.append(part)
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    prep = pipeline.PrepSettings(viscosity_pair=(1525.0, 5325.0) if viscosity else None)
    R.render_gem_results(data, "interp.csv", 0.5, "linear", None, scoring, prep)


def test_frequency_selection_table_under_ranking():
    at = AppTest.from_function(_app, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Reason" in d.value.columns for d in at.dataframe)


def test_viscosity_channels_appear_in_aux():
    at = AppTest.from_function(_app, kwargs={"scoring": False, "viscosity": True}, default_timeout=180)
    at.run()
    assert not at.exception
    assert at.tabs[-1].label == "AUX (2 channels)"
    assert any("Magnetic viscosity from 1525 and 5325 Hz" in w.value for w in at.warning)


def test_anomaly_spectrum_panel():
    at = AppTest.from_function(_app, kwargs={"scoring": False}, default_timeout=180)
    at.run()
    at.number_input(key="sp_x_interp").set_value(2.0)
    at.number_input(key="sp_y_interp").set_value(10.0)
    at.number_input(key="sp_r_interp").set_value(0.6)
    at.run()
    assert not at.exception
    spec = next(d.value for d in at.dataframe if "Q anomaly (ppm)" in d.value.columns)
    assert len(spec) == 3 and (spec["Q anomaly (ppm)"].iloc[:2] > 5).all()
