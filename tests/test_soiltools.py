"""Unit tests for soiltools.py (sampling design, property calibration, fuzzy zones)."""
import numpy as np
import pandas as pd
import pytest

import soiltools as S


def _field(n=40, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 100, n), np.linspace(0, 60, n))
    x, y = xs.ravel(), ys.ravel()
    ec1 = 10 + 30 * (x / 100) + 5 * np.sin(y / 10) + rng.normal(0, 0.5, x.size)
    ec2 = 12 + 25 * (x / 100) + 3 * np.cos(y / 8) + rng.normal(0, 0.5, x.size)
    return pd.DataFrame({"Line": np.repeat(np.arange(n), n), "X": x, "Y": y,
                         "EC1525Hz[mS/m]": ec1, "EC9825Hz[mS/m]": ec2})


def test_principal_scores_are_standardised():
    f = _field()[["EC1525Hz[mS/m]", "EC9825Hz[mS/m]"]].to_numpy()
    s = S.principal_scores(f)
    np.testing.assert_allclose(s.std(axis=0, ddof=1), 1.0)
    np.testing.assert_allclose(s.mean(axis=0), 0.0, atol=1e-12)


def test_design_targets_shapes():
    assert S.design_targets(6, 1).shape == (6, 1)
    t = S.design_targets(12, 2)
    assert t.shape == (12, 2) and tuple(t[0]) == (0.0, 0.0)
    radii = np.hypot(t[1:, 0], t[1:, 1])
    assert set(np.round(radii, 6)) == {1.0, 1.75}


def test_sampling_design_spans_feature_space_and_spreads_sites():
    df = _field()
    f = df[["EC1525Hz[mS/m]", "EC9825Hz[mS/m]"]].to_numpy()
    sites = S.sampling_design(df[["X", "Y"]].to_numpy(), f, 12)
    assert len(sites) == 12 and sites["row"].is_unique
    assert sites["pc1"].min() < -1.0 and sites["pc1"].max() > 1.0
    d = np.hypot(*(sites[["x", "y"]].to_numpy()[:, None, :] - sites[["x", "y"]].to_numpy()[None]).transpose(2, 0, 1))
    assert np.min(d[np.triu_indices(12, 1)]) > 2.0


def test_sampling_design_needs_enough_points():
    with pytest.raises(ValueError, match="usable readings"):
        S.sampling_design(np.zeros((3, 2)), np.ones((3, 1)), 12)


