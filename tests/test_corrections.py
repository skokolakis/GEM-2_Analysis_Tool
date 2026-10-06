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

