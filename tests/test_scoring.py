"""Regression tests for GEM ingestion, scoring and exports in RIs_v2."""
import io

import numpy as np
import pandas as pd
import pytest

import RIs_v2 as R


# ---------------------------------------------------------------------------
# GEM pivot (#18)
# ---------------------------------------------------------------------------

def _gem_table(lines) -> pd.DataFrame:
    rows = []
    for line in lines:
        ys = np.arange(0.0, 10.0, 1.0)
        rows.append(pd.DataFrame({"Line": line, "Y": ys, "EC1525Hz[mS/m]": ys + 1.0}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("labels", [["L1", "L2"], [1.5, 2.5], [0, 1]])
def test_pivot_accepts_any_line_label(labels):
    pivoted = R.pivot_gem_frequency(_gem_table(labels), "EC1525Hz[mS/m]")
    assert list(pivoted.columns[1:]) == [f"Line_{c}" for c in labels]


def test_pivot_keeps_first_reading_and_warns_once():
    table = _gem_table([0, 1])
    repeat = pd.DataFrame({"Line": [0], "Y": [3.0], "EC1525Hz[mS/m]": [99.0]})
    table = pd.concat([table, repeat], ignore_index=True)
    table["MSusc1525Hz[1/1000]"] = 0.5

    warnings: list[str] = []
    parsed = R.parse_gem_dataframe(table, warnings)

    pivoted = parsed["EC"]["1525Hz"].set_index("Y")
    assert pivoted.loc[3.0, "Line_0"] == 4.0          # first reading, not mean(4, 99)
    assert len(warnings) == 1 and "1 repeated reading" in warnings[0]


def test_process_sheet_keeps_first_reading_at_repeated_distance():
    df = pd.DataFrame({"d": [0.0, 1.0, 1.0, 2.0], "a": [0.0, 1.0, 9.0, 2.0]})
    interp, _, error, col_warnings = R.process_sheet(df, 1.0, "linear")
    assert error is None
    assert interp.loc[1.0, "a"] == 1.0
    assert any("repeated distance" in w for w in col_warnings)
