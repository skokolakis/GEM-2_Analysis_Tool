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


# ---------------------------------------------------------------------------
# Scoring: extrapolation (#5), ddof (#10), single-trace noise (#11),
# mixed estimators (#7)
# ---------------------------------------------------------------------------

X = np.arange(0, 100.5, 0.5)
STEP = 20 + 10 * np.tanh((X - 50) / 5)      # true amplitude ≈ 20


def _trace(rng, lo=0.0, hi=100.0, noise=0.5):
    m = (X >= lo) & (X <= hi)
    s = np.full_like(X, np.nan)
    s[m] = STEP[m] + rng.normal(0, noise, m.sum())
    return s


@pytest.mark.parametrize("method", [m for m in R.ALL_INTERP_METHODS if m != "polynomial"])
def test_traces_are_not_extrapolated(method):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"d": X, "L1": _trace(rng, 0, 100), "L2": _trace(rng, 3, 97),
                       "L3": _trace(rng, 0, 95)})
    interp, sc, error, _ = R.process_sheet(df, 0.5, method)
    assert error is None
    assert interp.loc[interp.index < 3, "L2"].isna().all()
    assert interp.loc[interp.index > 95, "L3"].isna().all()
    assert sc["amplitude"] < 22                      # was up to 294 for cubic
    assert 0.4 < sc["mean_std"] < 0.6


def test_between_trace_noise_uses_sample_std_and_matches_plot_envelope():
    rng = np.random.default_rng(1)
    x = np.arange(0, 1000.0, 0.5)
    df = pd.DataFrame({"d": x, "a": rng.normal(0, 1, x.size), "b": rng.normal(0, 1, x.size)})
    interp, sc, *_ = R.process_sheet(df, 0.5, "linear")
    assert sc["mean_std"] == pytest.approx(1.0, abs=0.05)   # ddof=0 gave ≈ 0.71
    envelope = interp.std(axis=1, ddof=1)
    assert np.sqrt(np.nanmean(envelope ** 2)) == pytest.approx(sc["mean_std"])


def test_single_trace_noise_independent_of_distance_step():
    rng = np.random.default_rng(0)
    xs = np.arange(0, 101, 1.0)
    df = pd.DataFrame({"d": xs, "L1": 20 + 10 * np.tanh((xs - 50) / 5) + rng.normal(0, .5, xs.size)})
    noises = [R.process_sheet(df, step, "linear")[1]["mean_std"] for step in (0.1, 0.5, 1.0, 2.0)]
    assert max(noises) == min(noises)


def test_single_trace_noise_is_unbiased_on_a_step_profile():
    xs = np.arange(0, 101, 1.0)
    trend = 20 + 10 * np.tanh((xs - 50) / 5)
    est = [R._intra_profile_noise(trend + np.random.default_rng(s).normal(0, .5, xs.size))
           for s in range(200)]
    assert np.mean(est) == pytest.approx(0.5, rel=0.03)


def test_single_and_multi_trace_scores_agree_and_are_labelled():
    ratios = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        a = pd.DataFrame({"d": X, **{f"L{i}": STEP + rng.normal(0, 0.5, X.size) for i in range(3)}})
        b = a.copy()
        b["L1"] = np.nan
        b["L2"] = np.nan
        sa = R.process_sheet(a, 0.5, "linear")[1]
        sb = R.process_sheet(b, 0.5, "linear")[1]
        ratios.append(sb["score"] / sa["score"])
    assert (sa["noise_method"], sa["n_traces"]) == (R.NOISE_BETWEEN, 3)
    assert (sb["noise_method"], sb["n_traces"]) == (R.NOISE_INTRA, 1)
    # Same signal and noise: the estimators agree on average (was 38 vs 64)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.1)


def test_mixed_noise_methods_warn_in_legacy_file():
    rng = np.random.default_rng(0)
    multi = pd.DataFrame({"d": X, "a": _trace(rng), "b": _trace(rng)})
    single = pd.DataFrame({"d": X, "a": _trace(rng)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        multi.to_excel(w, sheet_name="1000", index=False)
        single.to_excel(w, sheet_name="5000", index=False)
    _, scores, warnings = R.process_file.__wrapped__(buf.getvalue(), "EC", 0.5, "linear")
    assert scores["5000"]["noise_method"] == R.NOISE_INTRA
    assert any("Only one usable pass for 5000" in w for w in warnings)


def test_zero_noise_gives_blank_score_ranked_last():
    flat = pd.DataFrame({"d": X, "a": np.ones_like(X), "b": np.ones_like(X)})
    _, sc, error, col_warnings = R.process_sheet(flat, 0.5, "linear")
    assert error is None and np.isnan(sc["score"])
    assert any("score left blank" in w for w in col_warnings)
    ranked = R.rank_scores({"flat": sc, "ok": {"score": 3.0}, "better": {"score": 5.0}})
    assert [name for name, _ in ranked] == ["better", "ok", "flat"]


def test_non_overlapping_traces_are_reported():
    df = pd.DataFrame({"d": X, "a": np.where(X < 40, 1.0, np.nan), "b": np.where(X > 60, 2.0, np.nan)})
    assert R.process_sheet(df, 0.5, "linear")[2] == "traces do not overlap in distance"
