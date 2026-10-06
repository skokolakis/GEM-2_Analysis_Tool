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



