"""Unit tests for interpret.py (selection, viscosity, anomaly spectra)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import emphysics as E  # noqa: E402
import interpret as I  # noqa: E402
import pipeline  # noqa: E402


def _profiles():
    x = np.arange(0, 50.0, 0.5)
    a = np.sin(x / 5)
    rng = np.random.default_rng(0)
    data = {
        "475Hz": a + rng.normal(0, 0.01, x.size),
        "1525Hz": a * 1.01 + rng.normal(0, 0.01, x.size),        # twin of 475 Hz
        "18325Hz": np.cos(x / 7),
        "63025Hz": np.cos(x / 7) * 0.5 + np.sin(x / 3),
    }
    return {k: pd.DataFrame({"L1": v, "L2": v}, index=x) for k, v in data.items()}


def test_profile_correlation_is_symmetric_with_unit_diagonal():
    c = I.profile_correlation(_profiles())
    np.testing.assert_allclose(np.diag(c), 1.0)
    assert c.loc["475Hz", "1525Hz"] > 0.99


def test_selection_skips_redundant_and_caps_count():
    scores = {"475Hz": {"score": 10.0}, "1525Hz": {"score": 9.0},
              "18325Hz": {"score": 5.0}, "63025Hz": {"score": 1.0}}
    sel = I.select_frequencies(scores, I.profile_correlation(_profiles()), max_count=2)
    picked = sel.loc[sel["Selected"], "Frequency"].tolist()
    assert picked == ["475Hz", "18325Hz"]
    reasons = dict(zip(sel["Frequency"], sel["Reason"]))
    assert reasons["1525Hz"].startswith("redundant with 475Hz")
    assert reasons["63025Hz"] == "beyond the 2 best"


