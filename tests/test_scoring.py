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


# ---------------------------------------------------------------------------
# Units (#8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode, is_gem, expected",
    [
        ("EC", True, "Mean EC (mS/m)"),
        ("MS", True, "Mean MS (10⁻³ SI)"),
        ("EC", False, "Mean EC (input units)"),
    ],
)
def test_plot_axes_carry_correct_units(mode, is_gem, expected):
    import matplotlib.pyplot as plt

    interp = pd.DataFrame({"a": [1.0, 2.0], "b": [1.5, 2.5]}, index=[0.0, 1.0])
    scores = {"f": {"score": 1.0}}
    fig = R.make_overview_figure({"f": interp}, scores, mode, "x", is_gem=is_gem)
    assert fig.axes[0].get_ylabel() == expected
    plt.close(fig)
    fig = R.make_sheet_figure("f", interp, mode, is_gem=is_gem)
    assert fig.axes[0].get_ylabel() == expected
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch export registration (#9)
# ---------------------------------------------------------------------------

def _offset_sheets():
    a = pd.DataFrame({"d": np.arange(0, 50.5, 0.5), "L": np.arange(101.0)})
    b = pd.DataFrame({"d": np.arange(10, 60.5, 0.5), "L": np.arange(101.0)})
    oa, sa, *_ = R.process_sheet(a, 0.5, "linear")
    ob, sb, *_ = R.process_sheet(b, 0.5, "linear")
    return {"s1": oa, "s2": ob}, {"s1": sa, "s2": sb}


def test_batch_export_keeps_each_frequency_on_its_own_distances():
    out, sc = _offset_sheets()
    xlsx = R.build_batch_xlsx([{"stem": "f", "mode": "EC", "output_data": out, "scores": sc}])
    sheet = pd.read_excel(io.BytesIO(xlsx), sheet_name="f_EC").set_index("Distance (m)")
    assert sheet.index.min() == 0 and sheet.index.max() == 60
    assert sheet.loc[10.0, "s2_mean"] == 0.0         # s2 starts at 10 m
    assert np.isnan(sheet.loc[5.0, "s2_mean"])
    assert np.isnan(sheet.loc[55.0, "s1_mean"])


def test_batch_export_handles_grids_of_different_length():
    out, sc = _offset_sheets()
    short = pd.DataFrame({"d": np.arange(0, 20.5, 0.5), "L": np.arange(41.0)})
    out["s3"], sc["s3"], *_ = R.process_sheet(short, 0.5, "linear")
    xlsx = R.build_batch_xlsx([{"stem": "f", "mode": "EC", "output_data": out, "scores": sc}])
    sheet = pd.read_excel(io.BytesIO(xlsx), sheet_name="f_EC")
    assert len(sheet) == 121 and sheet["s3_mean"].notna().sum() == 41
