# Sensor physics & inversion

[← Documentation index](README.md)

## How the GEM-2 measures

The GEM-2 (Won et al., 1996) is a broadband frequency-domain EMI sensor. A transmitter coil drives a time-varying primary magnetic field that induces eddy currents in the ground; their secondary field is measured at the receiver as in-phase (I) and quadrature (Q) parts, in ppm of the primary. A bucking coil cancels the primary at the receiver.

Under the low-induction-number (LIN) approximation (McNeill, 1980) the two parts decouple:

| Component | Physical quantity | Unit |
|---|---|---|
| **Quadrature** | Apparent electrical conductivity (EC) | mS/m |
| **In-phase** | Apparent magnetic susceptibility (MS) | 10⁻³ SI |

**EC** rises with moisture, clay content and mineralogy, salinity and fine texture; low values indicate dry, sandy or gravelly ground (Reynolds, 2011). **MS** is raised by ferrimagnetic minerals (maghemite, magnetite), pedogenic enhancement of topsoil, burning and anthropogenic enrichment (hearths, kilns, iron-rich fills) and mafic parent material.

> Above about 40 kHz the in-phase response can include a dielectric-permittivity contribution (Benech et al., 2016); the MS and I tabs say so when such frequencies are present.

## Forward model

[`emphysics.py`](../emphysics.py) implements a layered-earth forward model of the GEM-2: vertical magnetic dipoles (horizontal co-planar coils) over horizontal layers, quasi-static, with the bucking coil subtracted (Won et al., 1996; Ward & Hohmann, 1988). The Hankel integral is evaluated by Gauss–Legendre quadrature with its large-wavenumber limit integrated analytically. The tests check it against a brute-force integral, McNeill's (1980) low-induction-number limit, and the analytic static response of a susceptible half-space.

**Sensor geometry** (sidebar → GEM data): Tx–Rx separation (default 1.66 m), Tx–bucking-coil distance (1.035 m; 0 = none) and sensor height (1.0 m). Check the coil distances against your sensor's `.gem` configuration.

## Tools

| Tool | Where | What it does |
|---|---|---|
| Recompute EC / MS from I / Q | Sensor geometry → checkbox | For every frequency with `I_` and `Q_` columns, finds the homogeneous half-space (conductivity and susceptibility) that reproduces each reading (Huang & Won, 2000), for your geometry and height. Overwrites exported EC / MS; adds them to I/Q-only exports. Readings with quadrature ≤ 0 are left blank |
| Frequency information | EC tab | Skin depth, induction number and depth of investigation (depth above which 70 % of the quadrature response originates, from the forward model) for the median EC of each frequency, with cumulative-sensitivity curves |
| Forward model (survey planning) | Main page | Background and target layered models → in-phase, quadrature and apparent EC per frequency; a difference over 3 × the noise level is marked detectable. Works without an upload — use it to choose frequencies before a survey |
| Multi-height calibration | Under each GEM file | See below |
| Lines to leave out | GEM data | Removes calibration or test lines from profiles and maps |

### Multi-height calibration

Readings over one spot at several heights (their own line(s), with a `Height` column) are fitted with a half-space plus one additive offset per frequency and component (after Minsley et al., 2014). Download the offsets and load them as **Calibration offsets** under [Corrections & filters](corrections.md).

## Layered-earth inversion

Under each GEM file, **Layered-earth inversion** ([`inversion.py`](../inversion.py)) inverts the quadrature of the mean profiles for a smooth layered conductivity model at stations along the line:

- Fixed layers growing geometrically from the first thickness to the depth of the half-space; parameters are log₁₀ conductivity.
- Gauss–Newton with vertical smoothing (Occam-style, Constable et al., 1987) and, between neighbouring stations, lateral smoothing (laterally constrained inversion, Auken & Christiansen, 2004). Lateral constraint 0 gives independent 1D inversions.
- The regularisation weight starts at the chosen value and is halved each iteration until χ² per datum reaches 1.
- Data errors: the larger of *relative error × |data| + floor* and, optionally, the **measured between-pass noise** of each frequency — the same σ the [scoring](profiles-and-scoring.md#scoring) uses (converted to ppm for EC data).
- EC-only files are converted to quadrature with the half-space model, which undoes a half-space conversion like the one in WinGEM.

**Downloads:** the model as CSV (distance, depth from / to, EC) and an **EMagPy survey file** (McLachlan et al., 2021): coordinates in metres, one column per frequency named `HCP{separation}f{frequency}h{height}` in mS/m, with `_err` columns from the between-pass noise. EMagPy models a plain loop pair, so the GEM-2 bucking coil is not modelled there.

> **Calibrate first** (multi-height offsets, reference calibration): offsets in I / Q bias inverted models (Minsley et al., 2014).

Full citations: [References](references.md).
