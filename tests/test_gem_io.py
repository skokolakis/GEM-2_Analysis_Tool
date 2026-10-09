"""Unit tests for gem_io.py (GEM-2 export handling)."""
import numpy as np
import pandas as pd
import pytest

import contouring as ctr
import gem_io as G


def _export(lines=2, n=11, mark=None, status=None):
    rows = []
    for line in range(lines):
        ys = np.arange(n, dtype=float)
        rows.append(pd.DataFrame({
            "Line": line, "Sample": np.arange(n), "X": 5.0 * line, "Y": ys,
            "Mark": 0, "Status": 0, "PowerLn": 0.1,
            "I_1525Hz": 100 + ys, "Q_1525Hz": 50 + ys,
            "EC1525Hz[mS/m]": 20 + ys, "MSusc1525Hz[1/1000]": 0.5 + 0 * ys,
            "QSum": 50 + ys,
        }))
    df = pd.concat(rows, ignore_index=True)
    if mark is not None:
        df["Mark"] = mark
    if status is not None:
        df["Status"] = status
    return df


def test_value_labels_for_new_modes():
    assert ctr.value_label("I", True) == "In-phase (ppm)"
    assert ctr.value_label("Q", True) == "Quadrature (ppm)"
    assert ctr.value_label("AUX", True, "PowerLn") == "Power-line noise (mG)"
    assert ctr.value_label("EC", True) == "EC (mS/m)"


def test_find_channels_recognises_every_mode():
    ch = G.find_channels(_export().columns)
    assert ch["EC"] == {"1525Hz": "EC1525Hz[mS/m]"}
    assert ch["MS"] == {"1525Hz": "MSusc1525Hz[1/1000]"}
    assert ch["I"] == {"1525Hz": "I_1525Hz"}
    assert ch["Q"] == {"1525Hz": "Q_1525Hz"}
    assert ch["AUX"] == {"PowerLn": "PowerLn", "QSum": "QSum"}


def test_find_channels_accepts_decimal_frequencies():
    assert G.find_channels(["EC93.5Hz[mS/m]"])["EC"] == {"93.5Hz": "EC93.5Hz[mS/m]"}


@pytest.mark.parametrize("mode", ["EC", "MS", "I", "Q", "AUX"])
def test_channel_column_inverts_find_channels(mode):
    for label, col in G.find_channels(_export().columns)[mode].items():
        assert G.channel_column(mode, label) == col


def test_drop_flagged_rows():
    status = np.zeros(22, dtype=int)
    status[[3, 15]] = 2
    df, n = G.drop_flagged_rows(_export(status=status))
    assert n == 2 and len(df) == 20


def test_drop_flagged_rows_without_status_column():
    df, n = G.drop_flagged_rows(_export().drop(columns="Status"))
    assert n == 0 and len(df) == 22


@pytest.mark.parametrize("marks", [
    [3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],      # running counter
    [0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0],      # set only on the flagged reading
])
def test_marker_rows_for_both_mark_styles(marks):
    df = _export(lines=1, mark=marks)
    assert np.nonzero(G.marker_rows(df))[0].tolist() == [2, 5, 8]


def test_marker_rows_ignore_missing_marks():
    df = _export(lines=1, n=6, mark=[0, 0, np.nan, 0, 5, 5])
    assert np.nonzero(G.marker_rows(df))[0].tolist() == [4]


def test_marker_distances_deduplicated():
    df = _export(lines=2, mark=[0, 0, 7, 0, 0, 8, 0, 0, 0, 9, 0] * 2)
    assert G.marker_distances(df, df["Y"].to_numpy()) == [2.0, 5.0, 9.0]


def test_marker_distance_dead_reckoning():
    df = _export(lines=1, mark=[0, 0, 7, 0, 0, 8, 0, 0, 0, 9, 0])
    d = G.along_track_distance(df, "markers", spacing=10.0)
    assert np.isnan(d[:2]).all() and np.isnan(d[10])
    np.testing.assert_allclose(d[2:10], [0, 10 / 3, 20 / 3, 10, 12.5, 15, 17.5, 20])


def test_projection_distance_shares_axis_for_reverse_passes():
    t = np.linspace(0, 30, 31)
    ang = np.radians(30)
    a = pd.DataFrame({"Line": 0, "X": t * np.cos(ang), "Y": t * np.sin(ang)})
    b = pd.DataFrame({"Line": 1, "X": t[::-1] * np.cos(ang) + 0.2, "Y": t[::-1] * np.sin(ang)})
    d = G.along_track_distance(pd.concat([a, b], ignore_index=True), "projection", coord_mode="metres")
    np.testing.assert_allclose(d[:31], t, atol=0.2)
    np.testing.assert_allclose(d[31:], t[::-1], atol=0.2)


def test_projection_follows_line_direction_on_wide_grids():
    rows = [pd.DataFrame({"Line": k, "X": float(k), "Y": np.linspace(0, 20, 41)}) for k in range(50)]
    d = G.along_track_distance(pd.concat(rows, ignore_index=True), "projection", coord_mode="metres")
    np.testing.assert_allclose(d.reshape(50, 41), np.tile(np.linspace(0, 20, 41), (50, 1)), atol=1e-9)


def test_path_distance_restarts_on_each_line():
    df = pd.DataFrame({"Line": [0, 0, 0, 1, 1], "X": [0, 3, 3, 10, 10], "Y": [0, 4, 8, 0, 1]})
    d = G.along_track_distance(df, "path", coord_mode="metres")
    np.testing.assert_allclose(d, [0, 5, 9, 0, 1])


def test_sample_distance():
    d = G.along_track_distance(_export(lines=2, n=3), "sample", spacing=0.5)
    np.testing.assert_allclose(d, [0, 0.5, 1.0, 0, 0.5, 1.0])


def test_projection_without_coordinates_raises():
    df = pd.DataFrame({"Line": [0, 1], "Y": [0.0, 1.0]})
    with pytest.raises(ctr.ContouringError):
        G.along_track_distance(df, "projection")


def test_time_seconds_from_ms_and_hhmmss_with_midnight():
    df = pd.DataFrame({"Time[hhmmss.sss]": [235959.5, 0.5, 1.0]})
    np.testing.assert_allclose(G.time_seconds(df), [86399.5, 86400.5, 86401.0])
    df = pd.DataFrame({"Time[ms]": [1000.0, 2500.0]})
    np.testing.assert_allclose(G.time_seconds(df), [1.0, 2.5])
    assert G.time_seconds(pd.DataFrame({"x": [1]})) is None


def test_time_seconds_ignores_missing_values():
    df = pd.DataFrame({"Time[ms]": [5e7, np.nan, 50_001_000.0, 50_002_000.0]})
    np.testing.assert_allclose(G.time_seconds(df), [50000, np.nan, 50001, 50002])


def _gbf_export(lines=3, n=21, along="Y"):
    """A GEM-2 export converted from a .gbf file: no Line column, Ip_/Qd_ for in-phase and quadrature."""
    rows, t0 = [], 0.0
    for k in range(lines):
        s = np.linspace(0, 10, n)
        x, y = (np.full(n, float(k)), s) if along == "Y" else (s, np.full(n, float(k)))
        t = t0 + 100.0 * np.arange(n)
        t0 = t[-1] + 5000.0
        rows.append(pd.DataFrame({"X": x, "Y": y, "Mark": 0, "Status": 0, "Time[ms]": t,
                                  "Ip_15270Hz": 1300.0 + k, "Qd_15270Hz": 1500.0 + s,
                                  "EC15270Hz[mS/m]": 37.0 + s, "MSusc15270Hz[1/1000]": -12.0}))
    return pd.concat(rows, ignore_index=True)


def test_normalise_renames_ip_qd_and_takes_lines_from_constant_x():
    df, msgs = G.normalise_columns(_gbf_export())
    assert {"I_15270Hz", "Q_15270Hz"} <= set(df.columns) and "Ip_15270Hz" not in df.columns
    assert df["Line"].tolist() == [0.0] * 21 + [1.0] * 21 + [2.0] * 21
    assert any("constant X" in m and "3 lines" in m for m in msgs)
    assert G.find_channels(df.columns)["I"] == {"15270Hz": "I_15270Hz"}


def test_lines_from_constant_y_from_time_gaps_or_one_line():
    df, _ = G.normalise_columns(_gbf_export(along="X"))
    assert df["Line"].nunique() == 3
    rng = np.random.default_rng(0)
    gps = _gbf_export()
    gps["X"] += rng.normal(0, 0.01, len(gps))      # GPS positions: no coordinate stays constant
    gps["Y"] += rng.normal(0, 0.01, len(gps))
    df, msgs = G.normalise_columns(gps)
    assert df["Line"].nunique() == 3 and any("time" in m for m in msgs)
    df, msgs = G.normalise_columns(gps.drop(columns="Time[ms]"))
    assert df["Line"].nunique() == 1 and any("one line" in m for m in msgs)


def test_lines_split_where_y_restarts():
    # X never changes: only Y jumping back to the start of the line marks a new line
    same_x = _gbf_export(lines=4)
    same_x["X"] = 0.0
    df, msgs = G.normalise_columns(same_x)
    assert df["Line"].tolist() == [float(k) for k in range(4) for _ in range(21)]
    assert any("Y restarts" in m and "4 lines" in m for m in msgs)
    df, msgs = G.normalise_columns(same_x.drop(columns=["X", "Time[ms]"]))   # no X column at all
    assert df["Line"].nunique() == 4 and any("Y restarts" in m for m in msgs)
    two_passes = pd.concat([_gbf_export(lines=2), _gbf_export(lines=2)], ignore_index=True)
    df, _ = G.normalise_columns(two_passes)                 # each X walked twice
    assert df["Line"].nunique() == 4
    serpentine = _gbf_export(lines=3)
    serpentine["X"] = 0.0
    serpentine.loc[21:41, "Y"] = serpentine.loc[21:41, "Y"].to_numpy()[::-1]   # walked back
    df, _ = G.normalise_columns(serpentine)
    assert df["Line"].nunique() == 1                         # a turn is not a restart


def test_normalise_leaves_wingem_exports_alone():
    df = _export()
    out, msgs = G.normalise_columns(df)
    assert out.equals(df) and msgs == []
