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

Dependencies (from [`requirements.txt`](../requirements.txt)): `streamlit` (≥ 1.50), `matplotlib` (≥ 3.5), `numpy`, `scipy` (≥ 1.10), `pandas` (≥ 2.1), `openpyxl`, `pykrige` (≥ 1.7), `pyproj` (≥ 3.3), `Pillow`.

## Run

```bash
streamlit run RIs_v2.py
```

The app opens in your browser at `http://localhost:8501`. On Windows you can also double-click [`launch.bat`](../launch.bat).

The light, publication-style theme comes from [`.streamlit/config.toml`](../.streamlit/config.toml).

## A first pass through the app

The empty page shows these steps in short (**How it works**).

1. **Upload** one or more `.xlsx` or `.csv` files — or press **Try the sample survey** to load the [sample data](#sample-data). With several files, a picker above the results shows one file at a time. GEM-2 exports are recognised from their column names; other files are refused with a message — see [Input data](input-data.md).
2. **Sidebar → Profiles → Profile interpolation & scoring**: choose the distance step and interpolation method, and switch on **Frequency scoring** if you want frequencies ranked — see [Profiles, scoring & interpolation](profiles-and-scoring.md).
3. **Sidebar → GEM files**: **GEM data preparation** (Status flags, lines to leave out, how distance along each line is measured), **Sensor geometry** and [**Corrections & filters**](corrections.md). Each sidebar group is named after the tool it sets and starts with a line saying which tools use it.
4. **Sidebar → 2D contouring**: switch on area maps or pseudo-sections, then set them under **Area map gridding** and **Map colours** — see [2D mapping](mapping.md).
5. Each GEM-2 file opens with a **summary card**: number of lines and readings, line length, reading spacing, survey duration, the frequencies found, and which tools the file supports — for example *Area maps* need coordinates and *Base-station drift* needs a Time column. When the distance step does not suit the reading spacing (under half of it, or more than five times it), a note under the card suggests one, and **Use … m** sets it.
6. Each file gets one tab per mode found (**EC**, **MS**, **I**, **Q**, **AUX**) with profiles, the graph editor and downloads. Below the tabs, **Calibration** holds the GPS-lag estimate and the multi-height calibration, whose results go into the sidebar; **Further analysis** holds the [sensor physics](sensor-physics.md) and [interpretation](interpretation.md) tools; the [soil tools](soil-tools.md) have their own page, in the navigation at the top of the sidebar. **Next steps** at the end of a tab switches on what it could show next — frequency ranking, the area map, the pseudo-section — without a trip to the sidebar.
7. **Export all files**, near the end of the page, packs every file's profiles (and scores) into one workbook — see [Outputs](outputs.md#batch-export).
8. The **Forward model** under **Survey planning**, at the end of the page, works without an upload — use it to choose frequencies before a survey.

### Basic and advanced settings

The sidebar starts in basic mode: the settings most surveys need. **Show advanced settings** at its top adds the expert ones — coil spacings, EC / MS recomputation and magnetic viscosity, despike window and threshold, drift model, temperature and sensor-height columns, calibration files, EC at 25 °C, the gridding method's own settings, blanking, projection and EPSG code, colour scale and display, contour lines, and the map filters under each area map. A hidden setting is always at its default, so switching back to basic mode puts any advanced setting you changed back to its default.

Under the toggle, a line names every setting that differs from its default and changes the results (distance step, interpolation, data preparation, sensor geometry, corrections and filters); the same list is shown above the results. **Reset all settings** puts every sidebar setting back to its default and clears loaded calibration files; uploaded data files are kept.

### Saving settings

**Save or load settings**, near the end of the sidebar, downloads every sidebar setting as a small `.json` file, and loads one back: to repeat a run later, after the page was refreshed, or on another computer, or to share it with a colleague. Settings missing from a loaded file go back to their defaults. Calibration files and uploaded data are not included.

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
