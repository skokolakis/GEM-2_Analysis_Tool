"""Unit tests for pipeline.py (preparation of GEM tables)."""
import numpy as np
import pandas as pd

import emphysics as E
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
    assert df[P.DISTANCE_COL].iloc[0] == 1.0          # the dropped reading still counts


def test_reading_order_distances_count_dropped_readings():
    df = pd.DataFrame({"Line": 0, "Y": np.arange(11.0), "Status": 0, "EC1525Hz[mS/m]": 1.0,
                       "Mark": [0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0]})
    df.loc[5, "Status"] = 1                         # the flagged reading carries a marker
    prep = P.PrepSettings(distance_method="markers", distance_spacing=10.0)
    out, _ = P.prepare_gem_table(df, prep)
    np.testing.assert_allclose(out[P.DISTANCE_COL].dropna(), [0, 10 / 3, 20 / 3, 40 / 3, 50 / 3, 20])


def test_prepare_keeps_flagged_when_asked():
    status = np.zeros(22, dtype=int)
    status[0] = 1
    df, msgs = P.prepare_gem_table(_export(status=status), P.PrepSettings(drop_flagged=False))
    assert len(df) == 22 and not msgs


FREQS = [1525.0, 18325.0]


def _iq_table(sigma=0.03, kappa=1e-3, n=30):
    z = E.forward_ppm(FREQS, [sigma], [kappa])
    df = pd.DataFrame({"Line": 0, "Y": np.arange(n, dtype=float)})
    for f, zi in zip(FREQS, z):
        df[f"I_{f:g}Hz"] = zi.real
        df[f"Q_{f:g}Hz"] = zi.imag
    return df


def test_recompute_adds_ec_and_ms_columns():
    out, msgs = P.recompute_ec_ms(_iq_table(), E.GEM2)
    np.testing.assert_allclose(out["EC1525Hz[mS/m]"], 30.0, rtol=1e-4)
    np.testing.assert_allclose(out["MSusc18325Hz[1/1000]"], 1.0, atol=1e-3)
    assert "1525Hz, 18325Hz" in msgs[0]


def test_prepare_with_recompute_and_excluded_lines():
    df = pd.concat([_iq_table(), _iq_table().assign(Line=9)], ignore_index=True)
    prep = P.PrepSettings(recompute_from_iq=True, exclude_lines=("9",))
    out, msgs = P.prepare_gem_table(df, prep)
    assert set(out["Line"]) == {0} and "EC1525Hz[mS/m]" in out.columns
    assert any("Left out 30 reading(s)" in m for m in msgs)


def test_iq_offsets_are_subtracted():
    import corrections

    offsets = b"column,offset\nI_1525Hz,10.0\nQ_18325Hz,-5.0\nI_99Hz,1.0\n"
    df = _iq_table()
    prep = P.PrepSettings(corrections=corrections.CorrectionSettings(iq_offsets=offsets))
    out, msgs = P.prepare_gem_table(df, prep)
    np.testing.assert_allclose(out["I_1525Hz"], df["I_1525Hz"] - 10.0)
    np.testing.assert_allclose(out["Q_18325Hz"], df["Q_18325Hz"] + 5.0)
    assert any("Subtracted calibration offsets" in m for m in msgs)


def test_ec_corrections_survive_recompute():
    import corrections

    prep = P.PrepSettings(recompute_from_iq=True,
                          corrections=corrections.CorrectionSettings(ec_background=50.0))
    out, msgs = P.prepare_gem_table(_iq_table(), prep)
    np.testing.assert_allclose(out["EC1525Hz[mS/m]"], 50.0, rtol=1e-6)
    first = [m.split()[0] for m in msgs]
    assert first.index("EC") < first.index("Shifted")


def test_recompute_compares_with_exported_ec_and_blanks_ms_at_ground_level():
    sensor = E.Sensor(height=0.0)
    z = E.forward_ppm([1525.0], [0.03], [1e-3], sensor=sensor)[0]
    df = pd.DataFrame({"Line": 0, "Y": np.arange(10.0), "I_1525Hz": z.real, "Q_1525Hz": z.imag,
                       "EC1525Hz[mS/m]": 15.0})
    out, msgs = P.recompute_ec_ms(df, sensor)
    assert out["MSusc1525Hz[1/1000]"].isna().all()
    assert any("barely depends on susceptibility" in m for m in msgs)
    assert any("1525Hz ×2" in m for m in msgs)
