# GEM-2 Analysis Tool

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18302817.svg)](https://doi.org/10.5281/zenodo.18302817)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Streamlit web app for processing, analysing and mapping multi-frequency electromagnetic induction (EMI) surveys from the **GEM-2** broadband sensor (Won et al., 1996): from raw WinGEM exports to corrected profiles, frequency rankings, layered-earth inversions and georeferenced maps.

**Try it online:** https://gemris.streamlit.app

---

## Quick start

```bash
git clone https://github.com/skokolakis/GEM-2_Analysis_Tool.git
cd GEM-2_Analysis_Tool
pip install -r requirements.txt
streamlit run RIs_v2.py
```

Then upload a GEM-2 export (`.csv` / `.xlsx`) — or the [synthetic sample survey](sample_data/synthetic_gem2_area_map.csv). Details: [Getting started](docs/getting-started.md).

---

## What it does

| | | Read more |
|---|---|---|
| **Read** | WinGEM / EMExport tables and `.gbf`-converted exports with every GEM-2 channel (EC, MS, I, Q, power-line noise, QSum, total EC), or legacy multi-sheet workbooks. Status flags, event markers and five ways to measure distance along a line | [Input data](docs/input-data.md) |
| **Correct** | Despike, clip, sensor-height, temperature and base-station drift, I/Q calibration offsets, GPS lag, background EC, reference calibration (ERT / TDR), EC at 25 °C, PCA noise reduction, running mean, heading filter | [Corrections & filters](docs/corrections.md) |
| **Profile & rank** | Representative (mean) profiles on a common distance grid with seven interpolation methods, and an optional signal-to-noise score — amplitude ÷ between-pass noise — to rank frequencies | [Profiles, scoring & interpolation](docs/profiles-and-scoring.md) |
| **Model** | Layered-earth forward model with the bucking coil, EC / MS recomputed from I / Q, skin depth and depth of investigation, multi-height calibration, smooth 1D / laterally constrained inversion with EMagPy export | [Sensor physics & inversion](docs/sensor-physics.md) |
| **Interpret** | Redundancy-aware frequency selection, magnetic viscosity from two frequencies, anomaly spectra with Argand diagrams, power-line noise maps | [Interpretation aids](docs/interpretation.md) |
| **Map** | Distance × frequency pseudo-sections; plan-view area maps with Surfer's gridding methods (kriging, minimum curvature, natural neighbour, RBF, …), cross-validation, Surfer-style colours, map filters, footprint deconvolution, merged and time-lapse surveys, GeoTIFF / ASCII-grid export | [2D mapping](docs/mapping.md) |
| **Soil** | Response-surface sampling design, soil-property calibration against EC, fuzzy c-means management zones | [Soil tools](docs/soil-tools.md) |
| **Export** | Interpolated profiles, scores, plots, batch workbooks for one or all interpolation methods, grids, inversion models | [Outputs](docs/outputs.md) |

Every scientific choice is explained in the app (sidebar help and **About & methods**) and in the pages above, with citations collected in [References](docs/references.md).

---

## Documentation

Start with the **[documentation index](docs/README.md)**, or jump to:

- [Getting started](docs/getting-started.md) — install, run, sample data, tests
- [Input data](docs/input-data.md) · [Corrections & filters](docs/corrections.md) · [Profiles, scoring & interpolation](docs/profiles-and-scoring.md)
- [Sensor physics & inversion](docs/sensor-physics.md) · [Interpretation aids](docs/interpretation.md) · [2D mapping](docs/mapping.md) · [Soil tools](docs/soil-tools.md)
- [Outputs](docs/outputs.md) · [Troubleshooting](docs/troubleshooting.md) · [References](docs/references.md)
- For developers: [Architecture](docs/architecture.md)

---

## Repository layout

```
RIs_v2.py          Streamlit app (entry point)
gem_io.py          GEM-2 export reading, lines, distance
pipeline.py        preparation of the raw table
corrections.py     corrections & filters
emphysics.py       forward model and EC / MS conversion
inversion.py       layered-earth inversion, EMagPy export
interpret.py       interpretation aids
soiltools.py       soil tools
contouring.py      area maps and pseudo-sections
gridtools.py       map filters, merging, georeferenced exports
ui_tools.py        Streamlit panels for the physics and soil tools
sample_data/       synthetic GEM-2 survey and its generator
tests/             pytest suite
docs/              documentation
```

---

## Citation

If you use this software, please cite it — see [`CITATION.cff`](CITATION.cff) (DOI [10.5281/zenodo.18302817](https://doi.org/10.5281/zenodo.18302817)).

## License

MIT — see [`LICENSE`](LICENSE).
