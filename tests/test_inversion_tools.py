"""Stations from profiles, result tables, EMagPy export and the inversion panel."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import emphysics as E  # noqa: E402
import inversion as V  # noqa: E402
import pipeline  # noqa: E402

FREQS = [1525.0, 5325.0, 18325.0]


def _profiles(values_by_label, distance=np.arange(0, 10.5, 0.5), passes=2):
    return {lb: pd.DataFrame({f"Line_{p}": v for p in range(passes)}, index=distance)
            for lb, v in values_by_label.items()}


def test_station_data_from_quadrature():
    q = E.forward_ppm(FREQS, [0.02]).imag
    out = {"Q": _profiles({f"{f:g}Hz": np.full(21, qi) for f, qi in zip(FREQS[::-1], q[::-1])})}
    scores = {"Q": {f"{f:g}Hz": {"mean_std": 2.0, "noise_method": "between-trace"} for f in FREQS}}
    s = V.station_data(out, scores, station_step=2.0)
    assert s.source == "Q"
    np.testing.assert_allclose(s.frequencies, FREQS)            # sorted
    np.testing.assert_allclose(s.distance, [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(s.quadrature[0], q)
    np.testing.assert_allclose(s.noise, 2.0)


def test_station_data_from_ec_converts_values_and_noise():
    out = {"EC": _profiles({f"{f:g}Hz": np.full(21, 20.0) for f in FREQS})}
    scores = {"EC": {f"{f:g}Hz": {"mean_std": 0.5, "noise_method": "between-trace"} for f in FREQS}}
    s = V.station_data(out, scores, 1.0)
    assert s.source == "EC"
    np.testing.assert_allclose(s.quadrature[0], E.forward_ppm(FREQS, [0.02]).imag, rtol=1e-9)
    slope = (E.forward_ppm(FREQS, [0.0202]).imag - E.forward_ppm(FREQS, [0.02]).imag) / 0.2
    np.testing.assert_allclose(s.noise, 0.5 * slope, rtol=1e-6)        # ppm per mS/m


def test_single_pass_noise_is_not_used():
    out = {"Q": _profiles({"1525Hz": np.ones(21)}, passes=1)}
    scores = {"Q": {"1525Hz": {"mean_std": 2.0, "noise_method": "intra-profile"}}}
    assert V.station_data(out, scores, 1.0).noise is None


def test_model_table_and_section_figure():
    g = V.make_layer_grid(4.0, 5)
    res = V.InversionResult(log_sigma=np.full((3, 5), -2.0), grid=g, predicted=np.zeros((3, 2)),
                            chi2=1.0, alpha=1.0, iterations=2, history=[1.0])
    table = V.model_table(res, np.array([0.0, 1.0, 2.0]))
    assert len(table) == 15 and np.allclose(table["EC (mS/m)"], 10.0)
    assert np.isinf(table["Depth to (m)"].iloc[4])
    fig = V.make_section_figure(res, np.array([0.0, 1.0, 2.0]), "t")
    assert fig.axes[0].get_ylabel() == "Depth (m)"
    plt.close(fig)


def test_emagpy_from_table_with_and_without_coordinates():
    table = pd.DataFrame({"Line": 0, "X": [0.0, 1.0], "Y": [0.0, 0.0],
                          "EC5325Hz[mS/m]": [11.0, 12.0], "EC1525Hz[mS/m]": [10.0, 10.5],
                          pipeline.DISTANCE_COL: [0.0, 1.0]})
    df = pd.read_csv(io.BytesIO(V.emagpy_from_table(table, errors_ms_m={
        "EC1525Hz[mS/m]": 0.2, "EC5325Hz[mS/m]": 0.3})))
    assert list(df.columns) == ["x", "y", "elevation", "HCP1.66f1525h1", "HCP1.66f5325h1",
                                "HCP1.66f1525h1_err", "HCP1.66f5325h1_err"]
    no_xy = table.drop(columns=["X", "Y"])
    df = pd.read_csv(io.BytesIO(V.emagpy_from_table(no_xy)))
    assert df["x"].tolist() == [0.0, 1.0] and df["y"].tolist() == [0.0, 0.0]


def test_emagpy_reads_the_export():
    emagpy = pytest.importorskip("emagpy")
    table = pd.DataFrame({"Line": 0, "X": np.arange(5.0), "Y": 0.0,
                          "EC1525Hz[mS/m]": 10.0, "EC5325Hz[mS/m]": 11.0,
                          pipeline.DISTANCE_COL: np.arange(5.0)})
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.csv")
        with open(path, "wb") as fh:
            fh.write(V.emagpy_from_table(table))
        k = emagpy.Problem()
        k.createSurvey(path)
        assert k.surveys[0].freqs == [1525.0, 5325.0]


def test_station_noise_is_the_noise_of_the_mean_profile():
    out = {"Q": _profiles({"1525Hz": np.ones(21)}, passes=4)}
    scores = {"Q": {"1525Hz": {"mean_std": 2.0, "noise_method": "between-trace", "n_traces": 4}}}
    np.testing.assert_allclose(V.station_data(out, scores, 1.0).noise, 1.0)


def test_emagpy_skips_readings_without_ec():
    table = pd.DataFrame({"Line": 0, "X": [0.0, 1.0, 2.0], "Y": 0.0,
                          "EC1525Hz[mS/m]": [10.0, np.nan, 11.0], pipeline.DISTANCE_COL: [0.0, 1.0, 2.0]})
    df = pd.read_csv(io.BytesIO(V.emagpy_from_table(table)))
    assert df["x"].tolist() == [0.0, 2.0]


def _inversion_app():
    import numpy as np
    import pandas as pd

    import emphysics as E
    import RIs_v2 as R

    freqs = [1525.0, 5325.0, 18325.0, 63025.0]
    rows = []
    for line in range(2):
        ys = np.arange(0.0, 10.0, 0.25)
        part = pd.DataFrame({"Line": line, "X": 0.0, "Y": ys})
        for f in freqs:
            q = E.forward_ppm([f], [0.01], [0.0]).imag[0]
            part[f"I_{f:g}Hz"] = 0.0
            part[f"Q_{f:g}Hz"] = q * (1 + 0.01 * np.sin(ys + line))
        rows.append(part)
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    R.render_gem_results(data, "inv.csv", 0.25, "linear", None, False, None)


def test_inversion_panel_runs_and_offers_model_download():
    at = AppTest.from_function(_inversion_app, default_timeout=300)
    at.run()
    at.number_input(key="inv_s_inv").set_value(2.0).run()
    at.button(key="inv_btn_inv").click().run()
    assert not at.exception
    assert any("stations" in c.value for c in at.caption)


def test_inversion_reruns_only_on_the_button():
    at = AppTest.from_function(_inversion_app, default_timeout=300)
    at.run()
    at.number_input(key="inv_s_inv").set_value(2.0).run()
    at.button(key="inv_btn_inv").click().run()
    at.number_input(key="inv_a_inv").set_value(100.0).run()
    assert not at.exception
    assert any("inputs changed" in c.value for c in at.caption)
