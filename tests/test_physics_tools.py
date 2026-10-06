"""Physics tools: table helpers and panels (AppTest)."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import emphysics as E
import ui_tools as U

FREQS = [1525.0, 18325.0]


def test_parse_frequencies():
    assert U.parse_frequencies("475, 1525 5325") == [475.0, 1525.0, 5325.0]
    with pytest.raises(ValueError):
        U.parse_frequencies("abc")
    with pytest.raises(ValueError):
        U.parse_frequencies("-5")


def test_layers_from_table_and_model_response():
    table = pd.DataFrame({"Thickness (m)": [1.0, None], "EC (mS/m)": [20.0, 20.0],
                          "MS (10⁻³ SI)": [0.0, 0.0]})
    sigma, kappa, thickness = U.layers_from_table(table)
    np.testing.assert_allclose(sigma, [0.02, 0.02])
    assert thickness.tolist() == [1.0]
    resp = U.model_response(FREQS, table, E.GEM2)
    np.testing.assert_allclose(resp["EC apparent (mS/m)"], 20.0, rtol=1e-4)
    with pytest.raises(ValueError, match="positive thickness"):
        U.layers_from_table(table.assign(**{"Thickness (m)": [0.0, None]}))


def test_height_levels_bins_and_averages():
    rows = pd.DataFrame({"Height": [0.49, 0.51, 1.0, 1.02], "I_1525Hz": [1, 3, 5, 7.0],
                         "Q_1525Hz": [2, 4, 6, 8.0]})
    h, i, q = U.height_levels(rows, "Height", ["1525Hz"])
    np.testing.assert_allclose(h, [0.5, 1.0])
    np.testing.assert_allclose(i[:, 0], [2, 6])
    np.testing.assert_allclose(q[:, 0], [3, 7])


def _app_with_heights():
    import numpy as np
    import pandas as pd

    import emphysics as E
    import pipeline
    import RIs_v2 as R

    freqs = [1525.0, 18325.0]
    rows = []
    for line in range(3):
        ys = np.arange(0.0, 20.0, 0.5)
        z = E.forward_ppm(freqs, [0.03], [1e-3])
        part = pd.DataFrame({"Line": line, "X": 0.0, "Y": ys, "Height": 1.0})
        for f, zi in zip(freqs, z):
            part[f"I_{f:g}Hz"] = zi.real + 0.5 * np.sin(ys)
            part[f"Q_{f:g}Hz"] = zi.imag + np.cos(ys)
        rows.append(part)
    cal = []
    for h in (0.5, 1.0, 1.5, 2.0):
        z = E.forward_ppm(freqs, [0.03], [1e-3], sensor=E.Sensor(height=h))
        part = pd.DataFrame({"Line": "CAL", "X": 0.0, "Y": [0.0, 0.1], "Height": h})
        for f, zi in zip(freqs, z):
            part[f"I_{f:g}Hz"] = zi.real + 25.0
            part[f"Q_{f:g}Hz"] = zi.imag - 10.0
        cal.append(part)
    data = pd.concat(rows + cal, ignore_index=True).to_csv(index=False).encode()
    prep = pipeline.PrepSettings(recompute_from_iq=True, exclude_lines=("CAL",))
    R.render_gem_results(data, "heights.csv", 0.5, "linear", None, False, prep)


def test_iq_only_file_gets_ec_tab_frequency_info_and_multiheight_fit():
    at = AppTest.from_function(_app_with_heights, default_timeout=180)
    at.run()
    assert not at.exception
    assert at.tabs[0].label.startswith("EC")
    at.multiselect(key="mh_lines_heights").set_value(["CAL"]).run()
    at.button(key="mh_fit_heights").click().run()
    assert not at.exception
    offsets = next(d.value for d in at.dataframe if "offset" in d.value.columns)
    assert offsets.loc[offsets["column"] == "I_1525Hz", "offset"].iloc[0] == pytest.approx(25.0, abs=0.5)
    assert offsets.loc[offsets["column"] == "Q_18325Hz", "offset"].iloc[0] == pytest.approx(-10.0, abs=0.5)


def test_forward_model_panel_on_main_page():
    at = AppTest.from_file("RIs_v2.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Detectable" in str(df.value.columns.tolist()) for df in at.dataframe)
