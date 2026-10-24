"""Regression tests for GEM ingestion, scoring and exports in RIs_v2."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import RIs_v2 as R  # noqa: E402


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
# Lines at different positions are not repeat passes (#15)
# ---------------------------------------------------------------------------

def test_parallel_lines_are_scored_within_each_line():
    rng = np.random.default_rng(0)
    d = np.arange(0.0, 40.0, 0.1)
    # three transects over different ground (offsets of 50), noise 1
    df = pd.DataFrame({"d": d, **{f"Line_{k}": 50.0 * k + np.sin(d / 5) * 10 + rng.normal(0, 1, d.size)
                                  for k in range(3)}})
    tracks = {f"Line_{k}": k for k in range(3)}
    _, sc, error, _ = R.process_sheet(df, 0.5, "linear", tracks)
    assert error is None and sc["noise_method"] == R.NOISE_INTRA and sc["passes"] == 1
    assert 0.8 < sc["mean_std"] < 1.2                          # not the 50-unit spread between lines
    _, lumped, *_ = R.process_sheet(df, 0.5, "linear")
    assert lumped["mean_std"] > 40                             # all lines taken as repeat passes


def test_repeat_passes_are_pooled_within_their_transect():
    rng = np.random.default_rng(1)
    d = np.arange(0.0, 40.0, 0.5)
    ground = {0: 0.0, 1: 100.0}
    cols = {f"Line_{k}": ground[k // 2] + np.sin(d / 5) * 10 + rng.normal(0, 1, d.size) for k in range(4)}
    df = pd.DataFrame({"d": d, **cols})
    _, sc, *_ = R.process_sheet(df, 0.5, "linear", {f"Line_{k}": k // 2 for k in range(4)})
    assert sc["noise_method"] == R.NOISE_BETWEEN and sc["passes"] == 2
    assert 0.7 < sc["mean_std"] < 1.4
    assert 6 < sc["signal_std"] < 8.5                          # 10 sin within a transect (≈ 7.1), not the 100 step


@pytest.mark.parametrize("passes", [2, 5])
def test_snr_is_signal_power_over_noise_power_whatever_the_passes(passes):
    d = np.arange(0.0, 200.0, 0.5)
    signal = 3.0 * np.sin(d / 7.0)                              # power 4.5
    est = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"d": d, **{f"p{k}": signal + rng.normal(0, 1.0, d.size) for k in range(passes)}})
        _, sc, *_ = R.process_sheet(df, 0.5, "linear")
        est.append(sc["snr"])
        assert sc["score"] == pytest.approx(10 * np.log10(sc["snr"]))
    # var(mean profile) holds 1/passes of the noise; it is subtracted, so the SNR is unbiased
    assert np.mean(est) == pytest.approx(4.5, rel=0.05)


def test_snr_of_a_single_pass():
    d = np.arange(0.0, 200.0, 0.5)
    est = [R.process_sheet(pd.DataFrame({"d": d, "a": 3.0 * np.sin(d / 7.0)
                                         + np.random.default_rng(s).normal(0, 1.0, d.size)}),
                           0.5, "linear")[1]["snr"] for s in range(40)]
    assert np.mean(est) == pytest.approx(4.5, rel=0.1)


def test_pure_noise_has_no_signal_and_a_blank_score():
    rng = np.random.default_rng(3)
    d = np.arange(0.0, 100.0, 0.5)
    blank = 0
    for _ in range(20):
        df = pd.DataFrame({"d": d, "a": rng.normal(0, 1, d.size), "b": rng.normal(0, 1, d.size)})
        _, sc, _, warnings = R.process_sheet(df, 0.5, "linear")
        assert sc["snr"] < 0.15 or np.isnan(sc["score"])          # no signal beyond chance
        if np.isnan(sc["score"]):
            blank += 1
            assert any("no signal rises above the noise" in w for w in warnings)
    assert blank > 0


def test_area_survey_file_warns_about_separate_lines():
    rows = []
    for k in range(4):
        y = np.arange(0.0, 20.0, 0.5)
        rows.append(pd.DataFrame({"Line": k, "X": 3.0 * k, "Y": y, "EC1525Hz[mS/m]": 10.0 * k + y}))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    _, scores, warnings = R.process_gem_file(data, "area.csv", 0.5, "linear")
    assert any("4 different positions" in w for w in warnings)
    assert scores["EC"]["1525Hz"]["noise_method"] == R.NOISE_INTRA


# ---------------------------------------------------------------------------
# Units (#8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode, expected",
    [
        ("EC", "Mean EC (mS/m)"),
        ("MS", "Mean MS (10⁻³ SI)"),
    ],
)
def test_plot_axes_carry_correct_units(mode, expected):
    interp = pd.DataFrame({"a": [1.0, 2.0], "b": [1.5, 2.5]}, index=[0.0, 1.0])
    scores = {"f": {"score": 1.0}}
    fig = R.make_overview_figure({"f": interp}, scores, mode, "x")
    assert fig.axes[0].get_ylabel() == expected
    plt.close(fig)
    fig = R.make_sheet_figure("f", interp, mode)
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
STEP = 20 + 10 * np.tanh((X - 50) / 5)      # true signal σ ≈ 9.5


def _trace(rng, lo=0.0, hi=100.0, noise=0.5):
    m = (X >= lo) & (X <= hi)
    s = np.full_like(X, np.nan)
    s[m] = STEP[m] + rng.normal(0, noise, m.sum())
    return s


@pytest.mark.parametrize("method", R.INTERP_METHODS)
def test_traces_are_not_extrapolated(method):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"d": X, "L1": _trace(rng, 0, 100), "L2": _trace(rng, 3, 97),
                       "L3": _trace(rng, 0, 95)})
    interp, sc, error, _ = R.process_sheet(df, 0.5, method)
    assert error is None
    assert interp.loc[interp.index < 3, "L2"].isna().all()
    assert interp.loc[interp.index > 95, "L3"].isna().all()
    assert sc["signal_std"] < 11                     # extrapolated ends would inflate it
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
        ratios.append(sb["snr"] / sa["snr"])
    assert (sa["noise_method"], sa["n_traces"]) == (R.NOISE_BETWEEN, 3)
    assert (sb["noise_method"], sb["n_traces"]) == (R.NOISE_INTRA, 1)
    # Same signal and noise: the estimators agree on average (was 38 vs 64)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.1)


def test_mixed_noise_methods_warn():
    rng = np.random.default_rng(0)
    multi = pd.DataFrame({"d": X, "a": _trace(rng), "b": _trace(rng)})
    single = pd.DataFrame({"d": X, "a": _trace(rng)})
    scores = {"1000Hz": R.process_sheet(multi, 0.5, "linear")[1],
              "5000Hz": R.process_sheet(single, 0.5, "linear")[1]}
    assert scores["5000Hz"]["noise_method"] == R.NOISE_INTRA
    assert "Only one usable pass for 5000Hz" in R.noise_method_warning(scores)


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
