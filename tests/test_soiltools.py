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


def test_property_model_recovers_log_linear_relation():
    df = _field()
    sites = df.iloc[::37].copy()
    sites["Clay"] = np.exp(0.5 + 0.8 * np.log(sites["EC1525Hz[mS/m]"]))
    model = S.fit_property_model(df, sites[["X", "Y", "Clay"]], "Clay", ["EC1525Hz[mS/m]"], radius=0.5)
    assert model.r2 > 0.999
    assert model.coefficients["ln EC1525Hz[mS/m]"] == pytest.approx(0.8, rel=1e-3)
    pred = S.predict_property(model, df)
    np.testing.assert_allclose(pred, np.exp(0.5 + 0.8 * np.log(df["EC1525Hz[mS/m]"])), rtol=1e-6)


def test_property_model_with_trend_and_too_few_samples():
    df = _field()
    sites = df.iloc[::37].copy()
    sites["pH"] = 6 + 0.01 * sites["X"] + 0.02 * sites["EC9825Hz[mS/m]"]
    model = S.fit_property_model(df, sites[["X", "Y", "pH"]], "pH", ["EC9825Hz[mS/m]"], 0.5,
                                 log_property=False, trend=True)
    assert model.r2 > 0.99 and "x" in model.coefficients.index
    with pytest.raises(ValueError, match="matched readings"):
        S.fit_property_model(df, sites[["X", "Y", "pH"]].iloc[:3], "pH", ["EC9825Hz[mS/m]"], 0.5)


def test_fuzzy_cmeans_separates_two_blobs():
    rng = np.random.default_rng(1)
    a = rng.normal(-3, 0.3, (100, 2))
    b = rng.normal(3, 0.3, (100, 2))
    u, centres = S.fuzzy_cmeans(np.vstack([a, b]), 2)
    np.testing.assert_allclose(u.sum(axis=1), 1.0)
    assert (u[:100, 0] > 0.9).all() and (u[100:, 1] > 0.9).all()
    assert centres[0, 0] < 0 < centres[1, 0]


def test_indices_prefer_the_true_number_of_zones():
    rng = np.random.default_rng(2)
    data = np.vstack([rng.normal(c, 0.3, (80, 2)) for c in (-4, 0, 4)])
    table = S.zone_indices(data, range(2, 6))
    assert table.loc[table["FPI"].idxmin(), "Zones"] == 3
    assert table.loc[table["MPE"].idxmin(), "Zones"] == 3


def test_index_limits():
    crisp = np.eye(3)[np.arange(30) % 3]
    assert S.fuzziness_performance_index(crisp) == pytest.approx(0.0)
    assert S.modified_partition_entropy(crisp) == pytest.approx(0.0, abs=1e-12)
    flat = np.full((30, 3), 1 / 3)
    assert S.fuzziness_performance_index(flat) == pytest.approx(1.0)
    assert S.modified_partition_entropy(flat) == pytest.approx(1.0)


def _latlon_field():
    df = _field()
    lat = 37.9 + df["Y"] / 111_195.0
    lon = 23.7 + df["X"] / (111_195.0 * np.cos(np.radians(37.9)))
    return df.drop(columns=["X", "Y"]).assign(Lat=lat, Lon=lon)


def test_property_model_with_few_latlon_samples_and_missing_coordinates():
    df = _latlon_field()
    sites = df.iloc[::200].copy()                                   # 8 samples
    sites["Clay"] = np.exp(0.5 + 0.8 * np.log(sites["EC1525Hz[mS/m]"]))
    sites.iloc[0, sites.columns.get_loc("Lat")] = np.nan            # no GPS fix
    model = S.fit_property_model(df, sites[["Lat", "Lon", "Clay"]], "Clay", ["EC1525Hz[mS/m]"], 0.5)
    assert model.n == 7 and model.r2 > 0.999


def test_property_model_rejects_a_constant_property_and_masks_bad_predictions():
    df = _field()
    sites = df.iloc[::37].copy()
    with pytest.raises(ValueError, match="does not vary"):
        S.fit_property_model(df, sites.assign(pH=7.0), "pH", ["EC1525Hz[mS/m]"], 0.5, log_property=False)
    sites["Clay"] = np.exp(0.5 + 0.8 * np.log(sites["EC1525Hz[mS/m]"]))
    model = S.fit_property_model(df, sites, "Clay", ["EC1525Hz[mS/m]"], 0.5)
    pred = S.predict_property(model, df.assign(**{"EC1525Hz[mS/m]": 0.0}))
    assert np.isnan(pred).all()


def test_robust_scaling_keeps_a_spike_from_owning_a_zone():
    rng = np.random.default_rng(0)
    ec = rng.lognormal(3.0, 0.3, (3000, 2))
    ec[0] = 1e4
    u, _ = S.fuzzy_cmeans(S.robust_standardise(ec), 6, 1.1)
    assert u.sum(axis=0).min() > 0.01 * len(u)


def test_principal_scores_drop_collinear_noise():
    a = np.random.default_rng(0).normal(size=200)
    s = S.principal_scores(np.column_stack([a, 2 * a]))
    assert s.shape == (200, 1) and np.corrcoef(s[:, 0], a)[0, 1] > 0.999


def test_unreadable_samples_file_is_a_value_error():
    import ui_tools as U

    class Upload:
        def getvalue(self):
            return b""

    with pytest.raises(ValueError, match="could not be read"):
        U._read_samples(Upload())


def test_zone_map_is_sized_to_the_survey_and_lists_only_the_zones_used():
    import matplotlib.pyplot as plt

    import ui_tools as U

    x, y = np.meshgrid(np.arange(61.0), np.linspace(0, 77, 50))     # taller than wide, like a 61-line field
    zone = np.where(x < 30, 1, 3).ravel()
    fig = U.zone_figure(x.ravel(), y.ravel(), zone, 3)
    w, h = fig.get_size_inches()
    assert h > w                                    # portrait field -> portrait figure
    assert list(fig.axes[1].get_yticks()) == [1, 2, 3]
    plt.close(fig)
    fig, _ = U.plan_view(np.array([0.0, 500.0]), np.array([0.0, 1.0]))
    assert fig.get_size_inches()[1] == 3.0         # very flat surveys keep a usable height
    plt.close(fig)
