# 2D mapping

[← Documentation index](README.md)

Two optional views, switched on in the sidebar under **2D contouring**; both are off by default. Their settings sit below the switches in **Area map gridding** and **Map colours**. Pseudo-sections are drawn on the **EC** and **MS** tabs; area maps on the **EC**, **MS** and **AUX** tabs. The raw **I** and **Q** tabs have neither: in-phase and quadrature (ppm) grow with frequency, so a pseudo-section of them mostly shows that trend rather than the ground. Implementation: [`contouring.py`](../contouring.py) (gridding, pseudo-sections) and [`gridtools.py`](../gridtools.py) (map filters, merging, georeferenced exports). Design notes: [`superpowers/specs/2026-10-05-2d-contouring-design.md`](superpowers/specs/2026-10-05-2d-contouring-design.md).

- [Pseudo-section](#pseudo-section-distance--frequency)
- [Area map](#area-map-plan-view) — [gridding methods](#gridding-methods), [colours](#colours), [projection](#projection), [map processing](#map-processing), [downloads](#downloads)
- [Combined surveys](#combined-surveys)

## Pseudo-section (distance × frequency)

Stacks the line-averaged profiles of all frequencies (the mean across survey lines at each distance — meaningful for repeat passes of one transect) into one contour plot: distance on x, frequency on y.

- The y-axis is logarithmic when every label is a distinct positive frequency, otherwise categorical. Labels in kHz are converted to Hz.
- Profiles are aligned on the distance range they all cover — nothing is extrapolated.
- White lines mark the measured frequencies; colours between them are interpolated.

> **The frequency axis is not a calibrated depth axis.** Under the LIN approximation the depth response of a loop–loop sensor is set by coil geometry, not frequency (McNeill, 1980; Callegary et al., 2007). Analyses of small broadband sensors nevertheless find that the practical depth of investigation grows roughly with the square root of the skin depth, so lower frequencies see somewhat deeper (Huang, 2005). Read the axis as a qualitative trend at most.

## Area map (plan view)

For GEM files covering an area (several lines with X/Y or Lat/Lon coordinates), one frequency at a time. Narrow corridor surveys are accepted when they have at least three distinct line positions; repeat passes along one transect (lines < 1 m apart) are treated as a transect — use the pseudo-section for those. Try it with the [sample survey](getting-started.md#sample-data).

1. **Coordinates** — `Lat`/`Latitude` + `Lon`/`Long`/`Longitude` columns are used as degrees when they hold at least 10 GPS fixes (rows logged as 0, 0 are treated as "no fix" and dropped); otherwise `X`/`Y`. The override under **Area map gridding** (auto / metres / degrees) applies to `X`/`Y` only. Degrees are projected as described under [Projection](#projection).
2. **Line levelling** (optional) — shifts each line to the survey median to remove line-to-line offsets (striping). It also removes any real gradient across lines, so compare with levelling off. For more advanced levelling see Mauring & Kihle (2006).
3. **Block-median reduction** — one median point per grid cell, so densely sampled lines do not dominate. Default cell size = √(bounding-box area / number of readings), about one node per reading.
4. **Gridding** — see the [table below](#gridding-methods).
5. **Blanking** — grid nodes farther than the blanking distance from any block median are left blank. The default keeps the gaps between survey lines filled: the larger of 2 × median point spacing and 1.5 × the 90th-percentile distance from grid nodes inside the survey to the nearest data point.
6. **Cross-validation** (button **Cross-validate (5-fold)**) — RMSE / MAE on the block medians, to compare methods on your data (Li & Heap, 2011). Folds are random, so along densely sampled lines the errors are optimistic; use them to rank methods rather than as absolute accuracy.

### Gridding methods

The gridding methods of Golden Software's Surfer:

| Method | Notes |
|---|---|
| Thin-plate spline | Minimum-curvature (biharmonic) surface (Briggs, 1974; Sandwell, 1987). Smoothing 0 interpolates exactly. |
| Ordinary kriging | Widely used for mapping apparent electrical conductivity (Corwin & Lesch, 2005); for regression / cokriging alternatives that use calibration samples see Lesch et al. (1995). One omnidirectional variogram (spherical / exponential / gaussian) is fitted to log-spaced lag classes up to half the maximum distance with Cressie (1985) weights, which lets the nugget pick up measurement noise where the model shape allows (spherical and exponential models, linear near the origin, can absorb noise into their slope; check the fitted nugget shown under the map) (Oliver & Webster, 2014); a small nugget floor keeps the gaussian model numerically stable. Needs roughly 30 or more block medians; uses the 64 nearest points. A second panel maps the kriging standard deviation. At most 4,000 block medians: with the automatic cell size the cell grows until that holds; with a manual cell size, choose a larger cell if the limit is hit. |
| Linear (Delaunay) | Triangulation; blank outside the data hull. |
| Minimum curvature | The smoothest surface through the data (Briggs, 1974), the usual choice for geophysical maps. Tension 0–0.95 (Smith & Wessel, 1990): 0 is pure minimum curvature, which can overshoot between readings; higher tension pulls the surface tight between them. Solved directly as one sparse least-squares system on the grid (edges free), so it is limited to 300,000 grid nodes. |
| Inverse distance to a power | Weighted mean of the 64 nearest block medians, weights 1 / (d² + δ²)^(power/2). Power 2 by default; a smoothing distance δ > 0 stops the surface passing exactly through readings and softens the bull's-eyes this method leaves around them. |
| Radial basis function | Multiquadric (Surfer's default), inverse multiquadric, natural cubic spline (r³) or thin-plate spline, with the 64 nearest points above 2,000 block medians. The multiquadric shape follows Surfer: R² = (diagonal of the data extent)² / (25 n). |
| Natural neighbour | Sibson weights — the area each reading would give up to a new point — computed in the discrete form of Park et al. (2006) on a raster four times finer than the grid. Smooth, never beyond the data range, and blank-free inside the data. |
| Nearest neighbour | Value of the closest block median: no smoothing, useful for dense regular grids. |
| Modified Shepard's method | Inverse-distance blend of local quadratics fitted around every reading (Franke & Nielson, 1980), with 13 points per quadratic and 19 for the weights as in Surfer, so it follows trends that plain inverse distance flattens. |
| Local polynomial | A polynomial of order 1–3 fitted around each node to its 64 nearest block medians, weights (1 − d / R)^power. |
| Polynomial regression | One trend surface (order 1–3) for the whole area, to show or remove a regional trend. |
| Moving average | Mean of the block medians within a search radius (automatic: a circle that holds about 16 readings at the survey's mean density). |
| Data metrics | A statistic of the block medians within the search radius: number, density, median, minimum, maximum, range or standard deviation. Number and density show the survey coverage. Not cross-validated, since these are summaries rather than estimates. |

Minimum curvature and natural neighbour are computed on the grid itself; cross-validation grids each training fold and reads it at the held-out points.

### Colours

Shared by the area map and the pseudo-section (sidebar → **Map colours**):

| Option | Choices |
|---|---|
| Colour map | Viridis (default), Rainbow (Surfer), Turbo, Jet, Plasma, Inferno, Magma, Cividis, Terrain, Spectral, Red–yellow–blue, Red–blue, Greyscale; each can be reversed. Viridis and cividis change evenly in brightness; rainbow scales show more detail at a glance but make some value steps look like edges. |
| Display | Filled contours, or a continuous image that colours every grid cell by its own value (Surfer's image map). Contour lines can be drawn on top of either. |
| Colour range | Percentiles (default 2nd–98th), the full data range (min–max, as Surfer), or fixed values so several maps share one scale. Values beyond the range take the end colours (arrows on the colour bar). |
| Colour scale | Linear; logarithmic (positive values only), for skewed data such as EC; or histogram-equalised, where each colour covers the same share of the map. |

To reproduce a Surfer image map: Rainbow (Surfer), continuous image, full data range, linear scale, with minimum curvature or kriging gridding.

### Projection

`Lat`/`Lon` are projected to local metres about the survey centre (default: equirectangular projection of a spherical Earth about the centroid, good to a few tenths of a percent at site scale) or to **UTM (WGS 84)** in the zone of the survey centre (sidebar → **Area map gridding** → *Projection of Lat/Lon*). For files whose `X`/`Y` are already projected, enter their **EPSG code** (e.g. 32634 for UTM 34N) so the exports carry it.

### Map processing

Expander **Map processing (filters, footprint deconvolution)** under each map, applied to the gridded values:

| Option | What it does |
|---|---|
| Despike | Replaces cells that differ from their window median by more than *threshold* × 1.4826 × the median absolute difference |
| Low-pass | Moving average over *window* × *window* cells (blank cells neither spread nor bias it) |
| High-pass | Removes the regional trend: the value minus its *window* × *window* moving average |
| Deconvolve the sensor footprint | Lateral Tikhonov deconvolution of the coils' low-induction-number footprint (receiver minus bucking coil, sensor height from *Sensor geometry*; the coil axis defaults to the survey-line direction). Sharpens apparent-conductivity maps. It does not resolve depth: the 3D multichannel deconvolution of Guillemoteau et al. (2017) needs several coil geometries, while a GEM-2 has one geometry at several frequencies, and at low induction number the frequencies share the same footprint |

The footprint kernel is the dot product of the transmitter's and receiver's quasi-static electric fields (Born approximation). The tests check that its depth profile matches McNeill's (1980) HCP depth response.

**Histogram & statistics** — count, mean, standard deviation, minimum, 2nd / 50th / 98th percentile and maximum of the gridded values, with a histogram.

### Downloads

PNG; grid CSV (`x, y, value` [, `variance`] [, `lon, lat`]); ESRI ASCII grid (`.asc`); single-band float32 **GeoTIFF** (`.tif`, no-data −9999); and, when the CRS is known (UTM or a user EPSG code), an ESRI **`.prj`** file to sit next to the `.asc`. With local-metre projection the `.asc` and GeoTIFF are not georeferenced — choose UTM, or use the CSV `lon`/`lat` columns.

## Combined surveys

With the area map on and two or more GEM files uploaded, **Combined surveys** (below the files) grids a channel the files share:

- **Merged (edge-matched)** — the surveys become one; line labels are prefixed with the survey number. With *Edge-match levels* each survey after the first is shifted by the median difference to the surveys before it, over readings closer than the edge-matching distance (as in edge matching of adjacent grids in archaeological-prospection software). The offsets are listed.
- **Difference (B − A)** — both surveys are gridded on one grid in the projection of A, and A is subtracted from B (time-lapse). Nodes blank in either survey stay blank; colours are centred on zero.

Full citations: [References](references.md).
