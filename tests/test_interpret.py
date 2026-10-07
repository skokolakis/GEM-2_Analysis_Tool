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


def test_viscosity_recovers_synthetic_parameters():
    f1, f2 = 1525.0, 5325.0
    sigma, kq = np.array([0.01, 0.05]), np.array([2e-4, 5e-5])
    l1, l2 = E.lin_ppm_per_sigma(f1), E.lin_ppm_per_sigma(f2)
    k = E.forward_ppm(f1, [0.0], [1e-3]).real[0] / 1e-3
    q1 = sigma * l1 - kq * k
    q2 = sigma * l2 - kq * k
    i1 = E.forward_ppm_batch([f1], sigma[:, None]).real[:, 0] + 1e-3 * k
    s, ki, kqe = I.viscosity_two_frequencies(f1, f2, i1, q1, q2)
    np.testing.assert_allclose(s, sigma, rtol=1e-9)
    np.testing.assert_allclose(kqe, kq, rtol=1e-9)
    np.testing.assert_allclose(ki, 1e-3, rtol=1e-6)


def test_viscosity_columns_and_missing_columns():
    df = pd.DataFrame({"I_1525Hz": [10.0], "Q_1525Hz": [20.0], "Q_5325Hz": [70.0]})
    out, msgs = I.add_viscosity_columns(df, (5325.0, 1525.0), E.GEM2)
    assert {I.VISCOSITY_COLUMN, I.QDIFF_EC_COLUMN} <= set(out.columns)
    with pytest.raises(ValueError, match="I_5325Hz|Q_9825Hz"):
        I.add_viscosity_columns(df, (1525.0, 9825.0), E.GEM2)


def test_viscosity_per_decade():
    assert I.viscosity_per_decade(np.array([1.0]))[0] == pytest.approx(1.4658, rel=1e-3)


def test_viscosity_pipeline_adds_aux_channels():
    df = pd.DataFrame({"Line": 0, "Y": np.arange(5.0), "I_1525Hz": 10.0, "Q_1525Hz": 20.0,
                       "Q_5325Hz": 70.0})
    out, msgs = pipeline.prepare_gem_table(df, pipeline.PrepSettings(viscosity_pair=(1525.0, 5325.0)))
    import gem_io
    aux = gem_io.find_channels(out.columns)["AUX"]
    assert I.VISCOSITY_COLUMN in aux and I.QDIFF_EC_COLUMN in aux


