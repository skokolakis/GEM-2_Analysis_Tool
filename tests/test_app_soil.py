"""Soil tools page (AppTest). File upload is not simulated: calibration is unit-tested."""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import pipeline
import RIs_v2 as R


def _field() -> bytes:
    rows = []
    for k in range(12):
        ys = np.arange(0.0, 60.0, 1.0)
        xs = np.full_like(ys, 5.0 * k)
        rows.append(pd.DataFrame({
            "Line": k, "X": xs, "Y": ys,
            "EC1525Hz[mS/m]": np.where(xs < 30, 10.0, 40.0) + np.sin(ys / 5),
            "EC9825Hz[mS/m]": np.where(xs < 30, 12.0, 38.0) + np.cos(ys / 7),
        }))
    return pd.concat(rows, ignore_index=True).to_csv(index=False).encode()


def test_soil_page_design_and_zones():
    at = AppTest.from_file(R.SOIL_PAGE, default_timeout=180)
    at.session_state[R.SURVEYS_KEY] = {"field.csv": (_field(), "field.csv", pipeline.PrepSettings())}
    at.run()
    assert not at.exception
    sites = next(d.value for d in at.dataframe if "pc1" in d.value.columns)
    assert len(sites) == 12
    indices = next(d.value for d in at.dataframe if "FPI" in d.value.columns)
    assert indices.loc[indices["FPI"].idxmin(), "Zones"] == 2      # two EC domains
    labels = [b.label for b in at.get("download_button")]
    assert "Sites (.csv)" in labels and "Zones (.csv)" in labels


def test_soil_page_without_a_survey_offers_the_sample():
    at = AppTest.from_file(R.SOIL_PAGE, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Survey analysis" in i.value for i in at.info)
    at.button(key="soil_sample").click().run()
    assert not at.exception
    assert any("pc1" in d.value.columns for d in at.dataframe)


def test_switching_pages_keeps_the_files_and_settings():
    at = AppTest.from_file("RIs_v2.py", default_timeout=180)
    at.run()
    at.button(key="sample_load").click().run()
    at.toggle(key="scoring").set_value(True).run()
    at.switch_page(R.SOIL_PAGE).run()
    assert not at.exception and [t.value for t in at.title] == ["Soil tools"]
    at.switch_page("RIs_v2.py").run()
    assert not at.exception
    assert at.toggle(key="scoring").value is True
    assert len(at.tabs) == 5                                        # the sample is still shown
