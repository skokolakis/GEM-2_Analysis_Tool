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


def test_marker_distances_deduplicated():
    df = _export(lines=2, mark=[0, 0, 7, 0, 0, 8, 0, 0, 0, 9, 0] * 2)
    assert G.marker_distances(df, df["Y"].to_numpy()) == [2.0, 5.0, 9.0]


