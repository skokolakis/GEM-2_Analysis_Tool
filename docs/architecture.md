# Architecture

[← Documentation index](README.md)

The Streamlit entry point is [`RIs_v2.py`](../RIs_v2.py) (the file name predates the rename to *GEM-2 Analysis Tool* and is kept so existing deployments keep working). The numerical work lives in modules of pure functions with no Streamlit dependency, each tested on its own.

## Modules

| Module | Responsibility |
|---|---|
| [`gem_io.py`](../gem_io.py) | GEM-2 export channels, `.gbf`-converted column names, Status flags, event markers, line detection, distance along line |
| [`pipeline.py`](../pipeline.py) | `PrepSettings` and `prepare_gem_table()`: the raw table before profiles and maps (flags, corrections, EC / MS from I / Q, distance) |
| [`corrections.py`](../corrections.py) | Despike, clip, height, drift, temperature, calibration, EC25, PCA, GPS lag, heading |
| [`emphysics.py`](../emphysics.py) | Layered-earth forward model, I/Q ↔ EC/MS conversion, skin depth, sensitivity, multi-height fit |
| [`inversion.py`](../inversion.py) | Smooth 1D / laterally constrained inversion, stations, EMagPy export |
| [`interpret.py`](../interpret.py) | Redundancy-aware frequency selection, magnetic viscosity, anomaly spectra |
| [`soiltools.py`](../soiltools.py) | Sampling design, soil-property calibration, fuzzy c-means management zones |
| [`contouring.py`](../contouring.py) | Area maps (all gridding methods, cross-validation, colour styles), pseudo-sections, unit labels |
| [`gridtools.py`](../gridtools.py) | Grid filters, statistics, survey merging, footprint deconvolution, UTM, GeoTIFF, `.prj` |
| [`ui_tools.py`](../ui_tools.py) | Streamlit panels for the physics, inversion, interpretation and soil tools |
| [`RIs_v2.py`](../RIs_v2.py) | The app: sidebar, file handling, profiles, scoring, plots, exports, map rendering |

## Inside `RIs_v2.py`

```
RIs_v2.py
├── Configuration, constants and in-app help text
├── GEM format detection
│   ├── is_gem_format()
│   ├── pivot_gem_frequency()
│   └── parse_gem_dataframe()
├── Data processing (cached)
│   ├── process_sheet()             — core interpolation & scoring
│   ├── process_file()              — legacy multi-sheet XLSX
│   ├── process_gem_file()          — GEM CSV / XLSX
│   └── rank_scores()
├── Interpolation helpers
│   └── _interpolate_with_method()
├── Plotting
│   ├── make_overview_figure()
│   ├── make_readings_figure()
│   └── make_sheet_figure()
├── Export helpers
│   ├── build_excel_download()
│   ├── build_scores_csv()
│   ├── build_batch_xlsx()
│   ├── build_all_methods_batch_xlsx()
│   └── fig_to_png()
├── UI components
│   ├── render_graph_editor()
│   ├── _render_mode_section()
│   ├── render_data_sidebar() / render_sensor_sidebar() / render_corrections_sidebar()
│   └── render_contouring_sidebar() / render_contouring() / render_combined_maps()
├── Dispatch
│   ├── render_legacy_results()
│   └── render_gem_results()
└── Streamlit app — main()
```

No external APIs or databases — all processing is local and deterministic.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

[`tests/`](../tests/) has one file per module (`test_gem_io.py`, `test_corrections.py`, `test_emphysics.py`, …) and `test_app_*.py` files that drive the app headlessly with `streamlit.testing.v1.AppTest`. The physics tests check the forward model against a brute-force integral, the LIN limit and the static susceptible half-space, and the footprint kernel against McNeill's HCP depth response.

## Sample data

[`sample_data/make_synthetic_gem2.py`](../sample_data/make_synthetic_gem2.py) builds a synthetic GEM-2 area survey from the forward model — see [Getting started](getting-started.md#sample-data).

## Design notes

- [2D contouring — design](superpowers/specs/2026-10-05-2d-contouring-design.md)
- [2D contouring — implementation plan](superpowers/plans/2026-10-06-2d-contouring.md)
