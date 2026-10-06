"""Unit tests for pipeline.py (preparation of GEM tables)."""
import numpy as np
import pandas as pd

import pipeline as P


def _export(lines=2, n=11, status=None):
    rows = []
    for line in range(lines):
        ys = np.arange(n, dtype=float)
        rows.append(pd.DataFrame({
            "Line": line, "Sample": np.arange(n), "X": 5.0 * line, "Y": ys,
            "Status": 0, "EC1525Hz[mS/m]": 20 + ys,
        }))
    df = pd.concat(rows, ignore_index=True)
    if status is not None:
        df["Status"] = status
    return df


def test_prepare_drops_flagged_and_adds_distance():
    status = np.zeros(22, dtype=int)
    status[0] = 1
    df, msgs = P.prepare_gem_table(_export(status=status), P.PrepSettings(distance_method="sample"))
    assert len(df) == 21 and P.DISTANCE_COL in df.columns
    assert any("Status" in m for m in msgs)
    assert df[P.DISTANCE_COL].iloc[0] == 1.0          # the dropped reading still counts


def test_reading_order_distances_count_dropped_readings():
    df = pd.DataFrame({"Line": 0, "Y": np.arange(11.0), "Status": 0, "EC1525Hz[mS/m]": 1.0,
                       "Mark": [0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0]})
    df.loc[5, "Status"] = 1                         # the flagged reading carries a marker
    prep = P.PrepSettings(distance_method="markers", distance_spacing=10.0)
    out, _ = P.prepare_gem_table(df, prep)
    np.testing.assert_allclose(out[P.DISTANCE_COL].dropna(), [0, 10 / 3, 20 / 3, 40 / 3, 50 / 3, 20])


def test_prepare_keeps_flagged_when_asked():
    status = np.zeros(22, dtype=int)
    status[0] = 1
    df, msgs = P.prepare_gem_table(_export(status=status), P.PrepSettings(drop_flagged=False))
    assert len(df) == 22 and not msgs
