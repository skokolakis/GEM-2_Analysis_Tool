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


def test_correlation_needs_enough_common_readings():
    x = np.arange(3.0)
    out = {"a": pd.DataFrame({"L1": [1.0, 2.0, 3.0]}, index=x),
           "b": pd.DataFrame({"L1": [2.0, 1.0, 5.0]}, index=x)}
    assert np.isnan(I.profile_correlation(out).loc["a", "b"])


def test_viscosity_recovers_forward_model_data_without_conductivity_bias():
    sigma = np.array([0.001, 0.02, 0.05, 0.1])
    k = E.kappa_sensitivity()
    for f1, f2 in ((1525.0, 5325.0), (5325.0, 18325.0)):
        z = E.forward_ppm_batch([f1, f2], sigma[:, None])
        for kq in (0.0, 5e-5):
            s, ki, kqe = I.viscosity_two_frequencies(
                f1, f2, z[:, 0].real + 1e-3 * k, z[:, 0].imag - kq * k, z[:, 1].imag - kq * k
            )
            np.testing.assert_allclose(s, sigma, rtol=1e-4)
            np.testing.assert_allclose(kqe, kq, atol=5e-7)
            np.testing.assert_allclose(ki, 1e-3, atol=1e-6)


def test_viscosity_needs_an_in_phase_susceptibility_response():
    with pytest.raises(ValueError, match="almost no in-phase"):
        I.viscosity_two_frequencies(1525.0, 5325.0, [0.0], [20.0], [70.0], E.Sensor(height=0.0))


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


def test_viscosity_pair_missing_from_file_is_reported_not_fatal():
    df = pd.DataFrame({"Line": 0, "Y": np.arange(5.0), "I_1525Hz": 10.0, "Q_1525Hz": 20.0,
                       "EC1525Hz[mS/m]": 5.0})
    out, msgs = pipeline.prepare_gem_table(df, pipeline.PrepSettings(viscosity_pair=(1525.0, 5325.0)))
    assert I.VISCOSITY_COLUMN not in out.columns and "EC1525Hz[mS/m]" in out.columns
    assert any(m.startswith("Magnetic viscosity skipped") and "1525Hz" in m for m in msgs)


def _anomaly_table():
    rows = []
    for k in range(9):
        xs = np.full(41, float(k))
        ys = np.linspace(0, 8, 41)
        r = np.hypot(xs - 4, ys - 4)
        bump = (r < 1.0).astype(float)
        part = pd.DataFrame({"Line": k, "X": xs, "Y": ys})
        for f, ai, aq in ((1525.0, 5.0, 20.0), (18325.0, 15.0, 8.0)):
            part[f"I_{f:g}Hz"] = 100 + ai * bump
            part[f"Q_{f:g}Hz"] = 50 + aq * bump
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_anomaly_spectrum_with_annulus_background():
    s = I.anomaly_spectrum(_anomaly_table(), (4.0, 4.0), 0.9)
    np.testing.assert_allclose(s["I anomaly (ppm)"], [5.0, 15.0])
    np.testing.assert_allclose(s["Q anomaly (ppm)"], [20.0, 8.0])
    fig = I.make_spectrum_figure(s, "t")
    assert len(fig.axes) == 2
    plt.close(fig)


def test_anomaly_spectrum_errors():
    with pytest.raises(ValueError, match="No readings"):
        I.anomaly_spectrum(_anomaly_table(), (100.0, 100.0), 0.5)
    with pytest.raises(ValueError, match="I_ and Q_"):
        I.anomaly_spectrum(pd.DataFrame({"X": [0.0], "Y": [0.0]}), (0.0, 0.0), 1.0)
