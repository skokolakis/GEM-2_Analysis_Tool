# Getting started

[← Documentation index](README.md)

## Use it online

The app runs on Streamlit Community Cloud: **https://gemris.streamlit.app** — nothing to install.

## Install locally

Requirements: Python 3.9 or newer.

```bash
git clone https://github.com/skokolakis/GEM-2_Analysis_Tool.git
cd GEM-2_Analysis_Tool
pip install -r requirements.txt
```

Dependencies (from [`requirements.txt`](../requirements.txt)): `streamlit` (≥ 1.26), `matplotlib` (≥ 3.5), `numpy`, `scipy` (≥ 1.10), `pandas` (≥ 2.1), `openpyxl`, `pykrige` (≥ 1.7), `pyproj` (≥ 3.3), `Pillow`.

## Run

```bash
streamlit run RIs_v2.py
```

The app opens in your browser at `http://localhost:8501`. On Windows you can also double-click [`launch.bat`](../launch.bat).

The light, publication-style theme comes from [`.streamlit/config.toml`](../.streamlit/config.toml).

## A first pass through the app

1. **Upload** one or more `.xlsx` or `.csv` files. GEM-2 exports are recognised from their column names; anything else is read as a legacy multi-sheet file — see [Input data](input-data.md).
2. **Sidebar → Profiles → Profile interpolation & scoring**: choose the distance step and interpolation method, and switch on **Frequency scoring** if you want frequencies ranked — see [Profiles, scoring & interpolation](profiles-and-scoring.md).
3. **Sidebar → GEM files**: **GEM data preparation** (Status flags, lines to leave out, how distance along each line is measured), **Sensor geometry** and [**Corrections & filters**](corrections.md). Each sidebar group is named after the tool it sets and starts with a line saying which tools use it.
4. **Sidebar → 2D contouring**: switch on area maps or pseudo-sections, then set them under **Area map gridding** and **Map colours** — see [2D mapping](mapping.md).
5. Each file gets one tab per mode found (**EC**, **MS**, **I**, **Q**, **AUX**) with profiles, the graph editor and downloads. Under **Further analysis** are the [sensor physics](sensor-physics.md), [interpretation](interpretation.md) and [soil](soil-tools.md) tools.
6. The **Forward model** on the main page works without an upload — use it to choose frequencies before a survey.

All processing is local and deterministic: no external APIs or databases.

## Sample data

[`sample_data/synthetic_gem2_area_map.csv`](../sample_data/synthetic_gem2_area_map.csv) is a synthetic GEM-2 survey for trying the area maps: a 60 × 80 m grid of 31 lines, 2 m apart, walked back and forth at 1 m/s with 10 readings a second, over a two-layer ground with known targets. Its I / Q come from the app's own layered-earth forward model, and EC / MS are their half-space conversion, as WinGEM exports them. Regenerate it with:

```bash
python sample_data/make_synthetic_gem2.py sample_data/synthetic_gem2_area_map.csv
```

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover the numerical modules and drive the Streamlit app headlessly (`streamlit.testing.v1.AppTest`). See [Architecture](architecture.md#tests).
