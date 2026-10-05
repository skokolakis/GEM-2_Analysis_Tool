"""Headless smoke tests for the 2D contouring UI (streamlit.testing.AppTest)."""
from streamlit.testing.v1 import AppTest


def _gem_app(method: str = "spline", lines: int = 6, legacy: bool = False):
    """Script body run by AppTest; all imports must be local."""
    import io

    import numpy as np
    import pandas as pd

    import RIs_v2 as R

    rows = []
    for i in range(lines):
        ys = np.arange(0.0, 40.0, 0.5)
        xs = np.full_like(ys, 4.0 * i)
        bump = np.exp(-((xs - 10) ** 2 + (ys - 20) ** 2) / 50.0)
        rows.append(pd.DataFrame({
            "Line": i, "X": xs, "Y": ys,
            "EC1525Hz[mS/m]": 20 + 10 * bump,
            "EC9825Hz[mS/m]": 22 + 8 * bump,
            "MSusc1525Hz[1/1000]": 0.5 + 0.2 * bump,
            "MSusc9825Hz[1/1000]": 0.6 + 0.2 * bump,
        }))
    data = pd.concat(rows, ignore_index=True).to_csv(index=False).encode()
    contour = R.ContourSettings(area_map=True, pseudosection=True, method=method)
    if legacy:
        legacy_df = pd.DataFrame({"d": np.arange(0.0, 40.0, 0.5)})
        legacy_df["t1"] = legacy_df["d"] * 0.1
        legacy_df["t2"] = legacy_df["d"] * 0.1 + 0.05
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            legacy_df.to_excel(w, sheet_name="1000", index=False)
            legacy_df.to_excel(w, sheet_name="5000", index=False)
        R.render_legacy_results(buf.getvalue(), "legacy.xlsx", "EC", 0.5, "linear", contour)
    else:
        R.render_gem_results(data, "survey.csv", 0.5, "linear", contour)


def _texts(at):
    return [m.value for m in at.markdown] + [c.value for c in at.caption]


def test_gem_area_map_and_pseudosection_render():
    at = AppTest.from_function(_gem_app, kwargs={"method": "spline"}, default_timeout=120)
    at.run()
    assert not at.exception
    texts = " ".join(_texts(at))
    assert "### Pseudo-section" in texts and "### Area map" in texts
    assert "block medians" in texts


def test_gem_kriging_shows_variogram():
    at = AppTest.from_function(_gem_app, kwargs={"method": "kriging"}, default_timeout=120)
    at.run()
    assert not at.exception
    assert "variogram spherical" in " ".join(_texts(at))


def test_single_line_shows_transect_message():
    at = AppTest.from_function(_gem_app, kwargs={"lines": 1}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("single transect" in i.value for i in at.info)


def test_legacy_file_area_map_message_and_pseudosection():
    at = AppTest.from_function(_gem_app, kwargs={"legacy": True}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("legacy multi-sheet format" in i.value for i in at.info)
    assert "### Pseudo-section" in " ".join(_texts(at))
