# 2D Contouring — Design

Date: 2026-10-05
Status: approved (design), pending implementation plan

## Goal

Add two independently togglable 2D visualisations to the Representative Incision Tool:

1. **Area map** — plan-view (X–Y) gridded contour map of EC / MS per frequency, built from the raw survey points of a GEM file covering an area.
2. **Pseudo-section** — distance × frequency contour of the representative (mean) profiles of an incision, built from data the app already computes.

Both are off by default; existing behaviour is unchanged when they are off.

## Literature basis

| Topic | Reference |
|---|---|
| Minimum-curvature / biharmonic spline gridding | Briggs, I.C. (1974). Machine contouring using minimum curvature. *Geophysics* 39(1), 39–48. https://doi.org/10.1190/1.1440410 |
| | Sandwell, D.T. (1987). Biharmonic spline interpolation of GEOS-3 and SEASAT altimeter data. *Geophys. Res. Lett.* 14(2), 139–142. https://doi.org/10.1029/GL014i002p00139 |
| | Smith, W.H.F. & Wessel, P. (1990). Gridding with continuous curvature splines in tension. *Geophysics* 55(3), 293–305. https://doi.org/10.1190/1.1442837 |
| Kriging of EMI / ECa data | Lesch, S.M., Strauss, D.J. & Rhoades, J.D. (1995). Spatial prediction of soil salinity using electromagnetic induction techniques: 1. *Water Resour. Res.* 31(2), 373–386. https://doi.org/10.1029/94WR02179 |
| | Corwin, D.L. & Lesch, S.M. (2005). Apparent soil electrical conductivity measurements in agriculture. *Comput. Electron. Agric.* 46, 11–43. https://doi.org/10.1016/j.compag.2004.10.005 |
| | Oliver, M.A. & Webster, R. (2014). A tutorial guide to geostatistics: computing and modelling variograms and kriging. *Catena* 113, 56–69. https://doi.org/10.1016/j.catena.2013.09.006 |
| Choosing a method by cross-validation | Li, J. & Heap, A.D. (2011). A review of comparative studies of spatial interpolation methods in environmental sciences. *Ecol. Inform.* 6, 228–241. https://doi.org/10.1016/j.ecoinf.2010.12.003 |
| Line levelling | Mauring, E. & Kihle, O. (2006). Leveling aerogeophysical data using a moving differential median filter. *Geophysics* 71(1), L5–L11. https://doi.org/10.1190/1.2163912 |
| Frequency is not depth under LIN | Huang, H. (2005). Depth of investigation for small broadband electromagnetic sensors. *Geophysics* 70(6), G135–G142. https://doi.org/10.1190/1.2122412 |
| Context (pre-processing, out of scope here) | Delefortrie, S. et al. (2015). Evaluating corrections for a horizontal offset between sensor and position data for surveys on land. *Precis. Agric.* 17, 349–364. https://doi.org/10.1007/s11119-015-9423-8 |
| | Minsley, B.J., Smith, B.D., Hammack, R. et al. (2012). Calibration and filtering strategies for frequency domain electromagnetic data. *J. Appl. Geophys.* 80, 56–66. https://doi.org/10.1016/j.jappgeo.2012.01.008 |
| Software | Müller, S., Schüler, L., Zech, A. & Heße, F. (2022). GSTools v1.3. *Geosci. Model Dev.* 15, 3161–3182. https://doi.org/10.5194/gmd-15-3161-2022 (considered; PyKrige chosen) |

All DOIs verified against Crossref on 2026-10-05.

## Architecture

- **New module `contouring.py`** — pure functions, no Streamlit imports:
  - coordinate detection and projection
  - area pipeline: levelling → block-median reduction → gridding → blanking
  - pseudo-section matrix builder
  - figure builders (`make_area_map_figure`, `make_pseudosection_figure`) returning `plt.Figure`, mirroring the existing `make_*_figure` style
  - export helpers (grid CSV, ESRI ASCII grid)
- **`RIs_v2.py`** — sidebar controls, cached wrappers (`st.cache_data`) around `contouring` calls keyed on file bytes + parameters, and rendering inside each file's EC / MS tab.
- **Dependencies** — `pykrige` added to `requirements.txt`; `pytest` added to a new `requirements-dev.txt` (keeps the hosted app's install lean).

## UI

Sidebar section **"2D contouring"**:
- Toggle: Area map (plan view) — default off
- Toggle: Pseudo-section — default off
- Area-map settings (visible only when the area map is on):
  - Coordinates: Auto / Metres / Degrees (default Auto)
  - Method: Thin-plate spline / Ordinary kriging / Linear (default Thin-plate spline)
  - Spline smoothing `s` ≥ 0 (default 0; spline only)
  - Variogram model: spherical / exponential / gaussian (default spherical; kriging only)
  - Cell size (m): default auto = median nearest-neighbour spacing of the raw points, rounded to 2 significant figures
  - Blanking distance (m): default auto = 2 × median nearest-neighbour spacing of the block-reduced points
  - Line levelling: on/off (default off)
  - Contour levels: integer, default 20

Inside each file's mode tab, after existing sections:
- **Pseudo-section** (if toggled): one figure per file × mode; PNG download.
- **Area map** (if toggled and file has coordinates): frequency selectbox; contour map with optional data-point overlay, equal aspect; for kriging, a second panel with the kriging standard deviation (√variance); button **"Cross-validate"** running 5-fold CV and showing RMSE and MAE; downloads: PNG, grid CSV, `.asc`.

The legacy multi-sheet format has no coordinates, so only the pseudo-section applies to it.

## Processing — area map

Per file × mode × frequency:

1. **Input** — raw GEM table (not the pivoted transect form). Keep rows where both coordinates and the value are finite.
2. **Coordinates**
   - Latitude/longitude columns are detected by case-insensitive name: latitude ∈ {`lat`, `latitude`}, longitude ∈ {`lon`, `long`, `longitude`}. If present they are used, in degrees.
   - Otherwise `X` / `Y` are used. In Auto mode they are treated as degrees only if all |Y| ≤ 90, all |X| ≤ 180 **and** both spans < 0.05; otherwise metres. The sidebar override replaces auto-detection.
   - Degrees → local metres by equirectangular projection about the centroid (lat₀, lon₀): x = R·cos(lat₀)·Δlon, y = R·Δlat, with R = 6 371 008.8 m and angles in radians.
3. **Line levelling** (optional) — for each `Line`, add (global median − line median) to its values. When on, the figure caption says "line-levelled (per-line median)".
4. **Block-median reduction** — bin points into square cells of the chosen size aligned to the grid; each occupied cell becomes one point at the median x, median y and median value of its members.
5. **Grid** — regular nodes at cell centres covering the bounding box of the block-reduced points.
6. **Gridding**
   - Thin-plate spline: `scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline", smoothing=s)`, with `neighbors=64` when there are more than 2000 points.
   - Ordinary kriging: `pykrige.ok.OrdinaryKriging(variogram_model=…, enable_plotting=False)`; `execute("grid", …)` with `backend="vectorized"` for ≤ 2000 points, otherwise `backend="loop", n_closest_points=64`. Returns estimate and variance.
   - Linear: `scipy.interpolate.griddata(method="linear")`; nodes outside the convex hull of the points are NaN.
7. **Blanking** — `scipy.spatial.cKDTree` over the block-reduced points; nodes whose nearest-point distance exceeds the blanking distance are set to NaN, in both the estimate and the variance.
8. **Colour scale** — contour levels span the 2nd–98th percentiles of the finite gridded values; `extend="both"`.
9. **Cross-validation** (on demand) — 5-fold random split (fixed seed 0) of the block-reduced points; same method and parameters; report RMSE and MAE in data units.

## Processing — pseudo-section

Per file × mode:

1. For each frequency, take the mean profile (row mean of the interpolated traces) with its own distance index.
2. Common distance grid = the **overlap** of all profiles' ranges, using the smallest step found among them. Each profile is linearly interpolated onto it; points outside a profile's own range are NaN (never extrapolated).
3. Frequency value is parsed from the label (first number in the name, e.g. `4525Hz` → 4525). If every label parses, rows are sorted ascending and the y-axis is log-scaled. Otherwise the original order is kept on a categorical axis.
4. `contourf` over (distance, frequency) with the same 2nd–98th percentile level scheme; a thin white horizontal line marks each measured frequency, making clear that values between rows are interpolated.
5. Caption: "Frequency axis shows instrument response per frequency, not depth (under LIN, depth sensitivity is set by coil geometry — Huang, 2005)."

## Units

New plots use the correct GEM units: EC in mS/m, MS in 10⁻³ SI (ppt). For legacy files, the label is "EC (input units)" / "MS (input units)". (The wrong units in existing plots are tracked separately in issue #8.)

## Exports

- PNG (dpi 200) for the area map and the pseudo-section.
- Grid CSV: columns `x`, `y`, `value`, plus `variance` (kriging) and `lon`, `lat` (inverse-projected when the input was in degrees). Blanked nodes are omitted.
- ESRI ASCII grid (`.asc`): written with `xllcorner`/`yllcorner` = the lower-left cell corner, `cellsize`, and `NODATA_value -9999`. For degree input it is written in local metres and the UI shows a warning that it is not georeferenced and that the CSV lon/lat columns should be used to place it.

## Error handling

| Condition | Behaviour |
|---|---|
| < 10 points after block reduction | `st.info("Not enough points to grid")`, no map |
| Minor/major principal-axis spread ratio < 0.05 | `st.info("Coordinates look like a single transect — use the pseudo-section")`, no map |
| No coordinate columns | Area-map section shows a note for that file |
| Kriging variogram fit or solve raises | `st.error` with the exception text and a suggestion to use the thin-plate spline |
| Grid > 4 000 000 nodes | `st.warning` asking for a larger cell size, no map |
| Pseudo-section with < 2 frequencies | `st.info`, no plot |
| Profiles with no distance overlap | `st.info("Profiles do not overlap in distance")`, no plot |

## Testing

`tests/test_contouring.py` (pytest), using synthetic data only:

- Projection: equirectangular distance vs haversine within 0.1 % over 2 km at latitude 50°.
- Coordinate auto-detection: metres → metres; degrees → degrees; a 0–80 m local grid → metres.
- Levelling removes known per-line constant offsets (residual offset < 1e-9).
- Block reduction: cell medians and counts match a hand-computed example.
- Spline and kriging reproduce a smooth Gaussian bump sampled on lines: RMSE at held-out nodes < 5 % of bump height; kriging variance ≥ 0 everywhere.
- Linear gridding yields NaN outside the convex hull.
- Blanking sets nodes farther than the distance threshold to NaN.
- Collinearity detection flags a single straight line and passes a 10-line grid.
- Pseudo-section aligns two profiles with different grids onto their overlap, with NaN only where expected.
- `.asc` writer: header values and the row order (north row first) are correct.

## Out of scope

Lag / sensor–GPS offset correction, GeoTIFF export, inversion, and the scientific-fidelity fixes tracked in issues #5–#18.
