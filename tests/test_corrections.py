"""Unit tests for corrections.py (filters, drift, calibration, positioning)."""
import math

import numpy as np
import pandas as pd
import pytest

import corrections as C
import gem_io
import pipeline


def _survey(n_lines=6, n=81, seed=0, serpentine=True, dt=0.2):
    """Parallel lines along Y, 2 m apart; EC = smooth field; Time[ms] increases."""
    rng = np.random.default_rng(seed)
    rows, t0 = [], 36_000.0
    for line in range(n_lines):
        ys = np.linspace(0.0, 40.0, n)
        if serpentine and line % 2:
            ys = ys[::-1]
        t = t0 + dt * np.arange(n)
        t0 = t[-1] + 5.0
        field = 20 + 5 * np.sin(ys / 6.0) + 0.5 * line
        rows.append(pd.DataFrame({
            "Line": line, "X": 2.0 * line, "Y": ys, "Time[ms]": 1000 * t,
            "EC1525Hz[mS/m]": field + rng.normal(0, 0.05, n),
            "EC9825Hz[mS/m]": 1.1 * field + rng.normal(0, 0.05, n),
            "MSusc1525Hz[1/1000]": 0.5 + 0 * ys,
        }))
    return pd.concat(rows, ignore_index=True)


# ── filters ──────────────────────────────────────────────────────────────────

def test_despike_series_blanks_spike_only():
    v = np.sin(np.linspace(0, 3, 50)) + np.random.default_rng(0).normal(0, 0.01, 50)
    v[20] += 5
    out, n = C.despike_series(v, 5, 6.0)
    assert n == 1 and np.isnan(out[20])


def test_despike_handles_integer_rounded_data():
    v = np.round(20 + 2 * np.sin(np.linspace(0, 3, 60)))
    v[30] += 20
    out, n = C.despike_series(v, 5, 4.0)
    assert n == 1 and np.isnan(out[30])


def test_clip_and_running_mean():
    v, n = C.clip_to_percentiles(np.arange(100.0), 5, 95)
    assert n == 10 and np.isnan(v[0]) and v[50] == 50
    np.testing.assert_allclose(C.running_mean(np.array([0, 3, 6.0]), 3), [1.5, 3, 4.5])


def test_ec_at_25_is_identity_at_25_and_lowers_warm_soil():
    assert C.ec_at_25(np.array([10.0]), 25.0)[0] == pytest.approx(10.0, abs=0.01)
    assert C.ec_at_25(np.array([10.0]), 35.0)[0] < 10.0


# ── sensor height ────────────────────────────────────────────────────────────

def test_height_correction_removes_exponential_trend():
    rng = np.random.default_rng(1)
    h = rng.uniform(0.5, 1.5, 400)
    geology = rng.normal(0, 0.2, 400)
    v = 100 + 300 * np.exp(-2.0 * h) + geology
    corr, model = C.correct_height(v, h, reference=1.0)
    assert model == "exponential"
    assert np.std(corr - geology) < 0.05
    assert np.mean(corr) == pytest.approx(100 + 300 * math.exp(-2.0), abs=0.1)


# ── drift and temperature ────────────────────────────────────────────────────

def _with_base_station(drift_per_hour=3.0, temp=None):
    df = _survey()
    t = gem_io.time_seconds(df)
    df["EC1525Hz[mS/m]"] += drift_per_hour * (t - t[0]) / 3600.0
    base = []
    for k, tb in enumerate([t[0] - 60, t[len(t) // 2], t[-1] + 60]):
        b = pd.DataFrame({"Line": f"B{k}", "X": 50.0, "Y": 0.0, "Time[ms]": 1000 * (tb + np.arange(5)),
                          "EC1525Hz[mS/m]": 30 + drift_per_hour * (tb - t[0]) / 3600.0,
                          "EC9825Hz[mS/m]": 33.0, "MSusc1525Hz[1/1000]": 0.5})
        if temp is not None:
            b["Temp"] = temp[k]
        base.append(b)
    if temp is not None:
        df["Temp"] = np.interp(t, [t[0], t[len(t) // 2], t[-1]], temp)
    return pd.concat([base[0], df, base[1], base[2]], ignore_index=True), df


def test_drift_correction_flattens_and_drops_base_lines():
    table, survey = _with_base_station()
    out, msgs = C.correct_drift(table, ("B0", "B1", "B2"), "piecewise")
    assert not out["Line"].astype(str).str.startswith("B").any()
    clean = _survey()
    resid = out["EC1525Hz[mS/m]"].to_numpy() - clean["EC1525Hz[mS/m]"].to_numpy()
    assert np.ptp(resid) < 0.2                        # drift was ~0.1 mS/m per minute
    assert "3 base-station occupation" in msgs[0]


def test_drift_needs_two_occupations_and_time():
    table, _ = _with_base_station()
    with pytest.raises(ValueError, match="at least 2"):
        C.correct_drift(table, ("B0",), "linear")
    with pytest.raises(ValueError, match="Time"):
        C.correct_drift(table.drop(columns="Time[ms]"), ("B0", "B1"), "linear")


def test_base_station_lines_match_float_labels():
    table, _ = _with_base_station()
    numbers = {"B0": 100.0, "B1": 101.0, "B2": 102.0}
    table["Line"] = [numbers[v] if v in numbers else float(v) for v in table["Line"]]
    _, msgs = C.correct_drift(table, ("100", "101", "102"), "piecewise")
    assert "3 base-station occupation" in msgs[0]


def test_temperature_coefficient_from_base_station():
    table, _ = _with_base_station(drift_per_hour=0.0, temp=[10.0, 20.0, 15.0])
    table["EC9825Hz[mS/m]"] += 0.4 * (table["Temp"] - 15.0)
    out, msgs = C.correct_temperature(table, "Temp", ("B0", "B1", "B2"))
    base = out[out["Line"].astype(str).str.startswith("B")]
    assert np.ptp(base["EC9825Hz[mS/m]"]) < 1e-9
    assert "0.4/°C" in msgs[0]


# ── calibration ──────────────────────────────────────────────────────────────

def test_reference_calibration_recovers_gain_and_offset():
    df = _survey(serpentine=False)
    pts = df.iloc[::20][["X", "Y", "EC1525Hz[mS/m]"]].copy()
    pts["EC1525Hz[mS/m]"] = 1.5 * pts["EC1525Hz[mS/m]"] - 4.0
    fits = C.fit_reference_calibration(df, pts, radius=0.3)
    assert len(fits) == 1
    assert fits[0]["gain"] == pytest.approx(1.5, rel=0.02)
    assert fits[0]["offset"] == pytest.approx(-4.0, abs=0.5)
    moments = C.fit_reference_calibration(df, pts, radius=0.3, method="moments")
    assert moments[0]["gain"] == pytest.approx(1.5, rel=0.05)


def test_reference_calibration_skips_constant_channels():
    df = _survey(serpentine=False)
    pts = df.iloc[::20][["X", "Y"]].copy()
    pts["EC1525Hz[mS/m]"] = 25.0
    assert C.fit_reference_calibration(df, pts, radius=0.3) == []


# ── PCA ──────────────────────────────────────────────────────────────────────

def test_pca_denoise_reduces_independent_noise():
    rng = np.random.default_rng(2)
    signal = rng.normal(0, 1, (500, 1)) * np.array([[1.0, 0.9, 0.8, 0.7]])
    noisy = signal + rng.normal(0, 0.1, signal.shape)
    out, kept = C.pca_denoise(noisy, 1)
    assert np.std(out - signal) < np.std(noisy - signal)
    assert kept > 0.95


# ── positioning ──────────────────────────────────────────────────────────────

def test_estimate_lag_recovers_injected_lag():
    df = _survey(n=161, dt=0.25)
    t = gem_io.time_seconds(df)
    # Data recorded 0.5 s late: each value belongs to where the sensor was 0.5 s earlier.
    true_y = np.empty(len(df))
    for idx in df.groupby("Line").indices.values():
        true_y[idx] = np.interp(t[idx] - 0.5, t[idx], df["Y"].to_numpy()[idx])
    df["EC1525Hz[mS/m]"] = 20 + 5 * np.sin(true_y / 6.0)
    lag, cost = C.estimate_lag(df, "EC1525Hz[mS/m]")
    assert lag == pytest.approx(0.5, abs=0.11)
    assert cost.argmin() == np.argmin(np.abs(C.LAG_SEARCH - lag))


def test_heading_filter_keeps_one_direction():
    df = _survey()                                     # even lines north, odd lines south
    keep = C.heading_mask(df, 0.0, 30.0)
    assert set(df.loc[keep, "Line"]) == {0, 2, 4}


def test_estimate_lag_with_dense_sampling():
    df = _survey(n=2001, dt=0.04)                      # 25 Hz at 0.5 m/s, lines 2 m apart
    t = gem_io.time_seconds(df)
    true_y = np.empty(len(df))
    for idx in df.groupby("Line").indices.values():
        true_y[idx] = np.interp(t[idx] - 0.5, t[idx], df["Y"].to_numpy()[idx])
    df["EC1525Hz[mS/m]"] = 20 + 5 * np.sin(true_y / 6.0)
    lag, _ = C.estimate_lag(df, "EC1525Hz[mS/m]")
    assert lag == pytest.approx(0.5, abs=0.11)


def test_estimate_lag_without_neighbouring_lines_raises():
    with pytest.raises(ValueError, match="neighbour on another line"):
        C.estimate_lag(_survey(n_lines=1), "EC1525Hz[mS/m]")


def test_shift_positions_keeps_gps_no_fix_rows_and_extrapolates():
    n = 20
    df = pd.DataFrame({"Line": 0, "Lat": 37.9 + 1e-5 * np.arange(n), "Lon": 23.7,
                       "X": 0.0, "Y": np.arange(n, dtype=float), "Time[ms]": 1000.0 * np.arange(n)})
    df.loc[10, ["Lat", "Lon"]] = 0.0
    out = C.shift_positions(df, gem_io.time_seconds(df), 1.0)
    assert (out.loc[10, ["Lat", "Lon"]] == 0.0).all()
    assert out["Lat"].drop(index=10).between(37.89, 37.91).all()
    assert out["Lat"].iloc[5] == pytest.approx(37.9 + 4e-5)
    assert out["Y"].iloc[5] == pytest.approx(4.0)           # X/Y shifted too
    assert out["Y"].iloc[0] == pytest.approx(-1.0)          # extrapolated, not clamped


def test_headings_use_whole_line_direction():
    rng = np.random.default_rng(3)
    y = np.concatenate([[0.0, 0.0, 0.0], np.arange(0, 20, 0.1)])  # stands still, then walks north
    df = pd.DataFrame({"Line": 0, "X": rng.normal(0, 0.3, y.size), "Y": y + rng.normal(0, 0.3, y.size)})
    assert C.heading_mask(df, 0.0, 30.0).all()


def test_estimate_lag_on_one_long_line_fails_fast():
    import time

    df = _survey(n_lines=1, n=4001, dt=0.04)
    start = time.perf_counter()
    with pytest.raises(ValueError, match="neighbour on another line"):
        C.estimate_lag(df, "EC1525Hz[mS/m]")
    assert time.perf_counter() - start < 3


def test_shift_positions_extrapolates_with_end_velocity():
    t = 0.9 + 0.1 * np.arange(100)                 # 10 Hz logging, GPS updated once a second
    df = pd.DataFrame({"Line": 0, "X": 0.0, "Y": np.floor(t), "Time[ms]": 1000 * t})
    out = C.shift_positions(df, gem_io.time_seconds(df), 1.0)
    assert -1.6 < out["Y"].iloc[0] < -0.5           # about 1 m/s, not the 10 m/s of the first two readings


# ── pipeline ─────────────────────────────────────────────────────────────────

def test_background_offset_and_soil_temperature_in_pipeline():
    df = _survey()
    s = C.CorrectionSettings(ec_background=30.0)
    out, msgs = C.apply_corrections(df, s)
    assert np.median(out["EC1525Hz[mS/m]"]) == pytest.approx(30.0)
    assert out["MSusc1525Hz[1/1000]"].equals(df["MSusc1525Hz[1/1000]"])


def test_corrections_reported_and_scores_flagged_through_pipeline():
    prep = pipeline.PrepSettings(corrections=C.CorrectionSettings(smooth_window=5, despike=True))
    out, msgs = pipeline.prepare_gem_table(_survey(), prep)
    assert any("Running mean over 5" in m for m in msgs)
    assert any("Despiking blanked" in m for m in msgs)
    assert prep.corrections.lowers_noise


def test_inactive_settings_change_nothing():
    df = _survey()
    out, msgs = pipeline.prepare_gem_table(df, pipeline.PrepSettings())
    assert not msgs
    pd.testing.assert_frame_equal(out.drop(columns=pipeline.DISTANCE_COL), df)
