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


