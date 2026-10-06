"""Physics tools: table helpers and panels (AppTest)."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import emphysics as E
import ui_tools as U

FREQS = [1525.0, 18325.0]


def test_parse_frequencies():
    assert U.parse_frequencies("475, 1525 5325") == [475.0, 1525.0, 5325.0]
    with pytest.raises(ValueError):
        U.parse_frequencies("abc")
    with pytest.raises(ValueError):
        U.parse_frequencies("-5")


def test_layers_from_table_and_model_response():
    table = pd.DataFrame({"Thickness (m)": [1.0, None], "EC (mS/m)": [20.0, 20.0],
                          "MS (10⁻³ SI)": [0.0, 0.0]})
    sigma, kappa, thickness = U.layers_from_table(table)
    np.testing.assert_allclose(sigma, [0.02, 0.02])
    assert thickness.tolist() == [1.0]
    resp = U.model_response(FREQS, table, E.GEM2)
    np.testing.assert_allclose(resp["EC apparent (mS/m)"], 20.0, rtol=1e-4)
    with pytest.raises(ValueError, match="positive thickness"):
        U.layers_from_table(table.assign(**{"Thickness (m)": [0.0, None]}))


def test_height_levels_bins_and_averages():
    rows = pd.DataFrame({"Height": [0.49, 0.51, 1.0, 1.02], "I_1525Hz": [1, 3, 5, 7.0],
                         "Q_1525Hz": [2, 4, 6, 8.0]})
    h, i, q = U.height_levels(rows, "Height", ["1525Hz"])
    np.testing.assert_allclose(h, [0.5, 1.0])
    np.testing.assert_allclose(i[:, 0], [2, 6])
    np.testing.assert_allclose(q[:, 0], [3, 7])

