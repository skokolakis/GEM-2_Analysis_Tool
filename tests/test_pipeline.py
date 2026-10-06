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
    assert df[P.DISTANCE_COL].iloc[0] == 0.0          # line 0 restarts after the dropped reading


def test_prepare_keeps_flagged_when_asked():
    status = np.zeros(22, dtype=int)
    status[0] = 1
    df, msgs = P.prepare_gem_table(_export(status=status), P.PrepSettings(drop_flagged=False))
    assert len(df) == 22 and not msgs
