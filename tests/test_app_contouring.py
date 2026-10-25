"""Headless smoke tests for the 2D contouring UI (streamlit.testing.AppTest)."""
import pytest
from streamlit.testing.v1 import AppTest

import contouring as ctr


def _gem_app(
    method: str = "spline",
    lines: int = 6,
    blank_distance=None,
    constant_ms: bool = False,
    options: tuple = (),
    colours: dict | None = None,
):
    """Script body run by AppTest; all imports must be local."""
    import numpy as np
    import pandas as pd

    import contouring as ctr
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
    table = pd.concat(rows, ignore_index=True)
    if constant_ms:
        table["MSusc1525Hz[1/1000]"] = 0.5
        table["MSusc9825Hz[1/1000]"] = 0.5
    data = table.to_csv(index=False).encode()
    contour = R.ContourSettings(
        area_map=True, pseudosection=True, method=method, blank_distance=blank_distance,
        method_options=options, colours=ctr.ColourStyle(**(colours or {})),
    )
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


def test_tiny_blanking_distance_shows_message_not_traceback():
    at = AppTest.from_function(_gem_app, kwargs={"blank_distance": 0.01}, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Nothing to contour" in i.value for i in at.info)


def test_constant_column_does_not_crash():
    at = AppTest.from_function(_gem_app, kwargs={"constant_ms": True}, default_timeout=120)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("method", [m for m in ctr.METHODS if m not in ("spline", "kriging")])
def test_every_gridding_method_renders(method):
    at = AppTest.from_function(_gem_app, kwargs={"method": method}, default_timeout=120)
    at.run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert "block medians" in " ".join(_texts(at))


def test_colour_options_render_map_and_pseudosection():
    colours = {"cmap": "surfer_rainbow", "display": "image", "scale": "equalised",
               "range_mode": "minmax", "contour_lines": True}
    at = AppTest.from_function(
        _gem_app, kwargs={"method": "min_curvature", "options": (("tension", 0.25),),
                          "colours": colours},
        default_timeout=120,
    )
    at.run()
    assert not at.exception and not at.error
    texts = " ".join(_texts(at))
    assert "### Pseudo-section" in texts and "block medians" in texts


def test_log_colours_on_negative_values_show_a_message():
    at = AppTest.from_function(
        _gem_app, kwargs={"colours": {"scale": "log", "range_mode": "fixed", "vmin": -1.0}},
        default_timeout=120,
    )
    at.run()
    assert not at.exception
    assert any("positive values" in i.value for i in at.info)


def _sidebar_app():
    """Script body run by AppTest: the contouring sidebar, with its result written out."""
    import streamlit as st

    import RIs_v2 as R

    with st.sidebar:
        contour = R.render_contouring_sidebar()
    st.write(repr(contour))


def test_sidebar_offers_method_settings_and_colours():
    at = AppTest.from_function(_sidebar_app, default_timeout=60)
    at.session_state["advanced"] = True
    at.run()
    at.sidebar.toggle(key="ct_area").set_value(True).run()
    at.sidebar.selectbox(key="ct_method").set_value("idw").run()
    at.sidebar.number_input(key="ct_idw_power").set_value(3.0).run()
    at.sidebar.selectbox(key="ct_cmap").set_value("surfer_rainbow").run()
    at.sidebar.selectbox(key="ct_range").set_value("fixed").run()
    at.sidebar.number_input(key="ct_vmax").set_value(80.0).run()
    assert not at.exception
    assert [e.label for e in at.sidebar.expander] == ["Area map gridding", "Map colours"]
    shown = at.main.markdown[-1].value
    assert "method='idw'" in shown and "('power', 3.0)" in shown
    assert "cmap='surfer_rainbow'" in shown and "vmax=80.0" in shown


def test_sidebar_colours_apply_to_the_pseudosection_alone():
    at = AppTest.from_function(_sidebar_app, default_timeout=60)
    at.session_state["advanced"] = True
    at.run()
    at.sidebar.toggle(key="ct_pseudo").set_value(True).run()
    at.sidebar.selectbox(key="ct_display").set_value("image").run()
    shown = at.main.markdown[-1].value
    assert "area_map=False" in shown and "pseudosection=True" in shown
    assert "display='image'" in shown


def test_bulk_export_zips_every_channel():
    at = AppTest.from_function(_gem_app, default_timeout=180)
    at.run()
    at.checkbox(key="bulk_grids_survey_EC").check()
    at.button(key="bulk_btn_survey_EC").click().run()
    assert not at.exception
    button = next(b for b in at.get("download_button") if b.label == "All EC maps (.zip)")
    assert button.proto.url        # the archive was handed to the browser


def test_bulk_export_needs_a_new_zip_after_a_settings_change():
    at = AppTest.from_function(_gem_app, default_timeout=180)
    at.run()
    at.button(key="bulk_btn_survey_EC").click().run()
    at.checkbox(key="bulk_grids_survey_EC").check().run()
    assert not at.exception
    assert any("press Prepare ZIP" in c.value for c in at.caption)
    assert not any(b.label == "All EC maps (.zip)" for b in at.get("download_button"))


def _channel_table(empty_ms: bool = False) -> bytes:
    import numpy as np
    import pandas as pd

    rows = []
    for i in range(6):
        ys = np.arange(0.0, 40.0, 0.5)
        xs = np.full_like(ys, 4.0 * i)
        bump = np.exp(-((xs - 10) ** 2 + (ys - 20) ** 2) / 50.0)
        rows.append(pd.DataFrame({
            "Line": i, "X": xs, "Y": ys,
            "EC1525Hz[mS/m]": 20 + 10 * bump,
            "EC9825Hz[mS/m]": 22 + 8 * bump,
            "MSusc1525Hz[1/1000]": 0.5 + 0.2 * bump,
            "MSusc9825Hz[1/1000]": np.nan if empty_ms else 0.6 + 0.2 * bump,
        }))
    return pd.concat(rows, ignore_index=True).to_csv(index=False).encode()


def test_build_area_map_zip_holds_one_map_per_channel():
    import io
    import zipfile

    import RIs_v2 as R

    calls = []
    data, skipped = R.build_area_map_zip(
        _channel_table(), "survey.csv", "EC", ["1525Hz", "9825Hz"], R.ContourSettings(area_map=True),
        include_grids=True, progress=lambda done, total: calls.append((done, total)),
    )
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert skipped == []
    assert calls == [(1, 2), (2, 2)]
    for freq in ("1525Hz", "9825Hz"):
        for ext in ("png", "csv", "asc", "tif"):
            assert f"survey_EC_{freq}_spline.{ext}" in names


def test_build_area_map_zip_leaves_out_a_channel_that_cannot_be_mapped():
    import io
    import zipfile

    import RIs_v2 as R

    data, skipped = R.build_area_map_zip(
        _channel_table(empty_ms=True), "survey.csv", "MS", ["1525Hz", "9825Hz"],
        R.ContourSettings(area_map=True),
    )
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == ["survey_MS_1525Hz_spline.png"]
    assert len(skipped) == 1 and skipped[0].startswith("9825Hz:")
