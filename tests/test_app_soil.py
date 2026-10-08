"""Soil tools panel in the app (AppTest). File upload is not simulated: calibration is unit-tested."""
from streamlit.testing.v1 import AppTest


def _soil_app():
    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows = []
    for k in range(12):
        ys = np.arange(0.0, 60.0, 1.0)
        xs = np.full_like(ys, 5.0 * k)
        rows.append(pd.DataFrame({
            "Line": k, "X": xs, "Y": ys,
            "EC1525Hz[mS/m]": np.where(xs < 30, 10.0, 40.0) + np.sin(ys / 5),
            "EC9825Hz[mS/m]": np.where(xs < 30, 12.0, 38.0) + np.cos(ys / 7),
        }))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    R.render_gem_results(data, "field.csv", 1.0, "linear", None, False, None)


def test_soil_tools_design_and_zones():
    at = AppTest.from_function(_soil_app, default_timeout=180)
    at.run()
    assert not at.exception
    sites = next(d.value for d in at.dataframe if "pc1" in d.value.columns)
    assert len(sites) == 12
    indices = next(d.value for d in at.dataframe if "FPI" in d.value.columns)
    assert indices.loc[indices["FPI"].idxmin(), "Zones"] == 2      # two EC domains
    labels = [b.label for b in at.get("download_button")]
    assert "Sites (.csv)" in labels and "Zones (.csv)" in labels
