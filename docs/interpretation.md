# Interpretation aids

[← Documentation index](README.md)

Implementation: [`interpret.py`](../interpret.py); panels in [`ui_tools.py`](../ui_tools.py).

## Frequency selection

*Where:* under the frequency ranking (scoring on), **Frequency selection (redundancy-aware)**.

Keeps up to *n* frequencies in descending score order, skipping any whose mean profile correlates above |r| with one already kept, and shows the correlation matrix. The GEM-2 shares its transmitter power among the frequencies, and drone-towed GEM-2 tests recommend at most three (Vilhelmsen & Døssing, 2022); strongly correlated frequencies carry the same information (Minsley et al., 2010).

## Magnetic viscosity

*Where:* sidebar → Sensor geometry → **Magnetic viscosity from frequencies** (two frequencies with I / Q columns, e.g. `1525, 5325`).

After Simon et al. (2015): at low induction number the conductive quadrature grows with frequency, while a viscous susceptibility κ″ adds the same quadrature at both frequencies, so Q(f) = σ·L(f) − κ″·K. The app adds `ECqdiff[mS/m]` and `MagViscosity[1/1000]` (κ″ × 1000) to the **AUX** tab, where they can be profiled and mapped. For a log-uniform relaxation spectrum the drop in susceptibility per decade of frequency is (2 ln 10 / π)·κ″.

Uncalibrated Q offsets shift κ″, so compare contrasts unless Q is calibrated ([multi-height calibration](sensor-physics.md#multi-height-calibration)).

## Anomaly spectrum

*Where:* under each GEM file, **Anomaly spectrum (EMI spectroscopy)**.

Mean I and Q within a radius of a point minus the background (a ring from the radius to 2 × radius, or the whole survey), for every frequency, with the Argand diagram (Q against I). The shape of the spectrum helps tell metal targets from soil (Huang & Won, 2003).

## Power-line noise map

*Where:* AUX tab → area map of `PowerLn`.

The GEM-2 logs power-line noise in mG; mapping it shows where noise contaminates the survey (GEM-2 Manual). See [2D mapping](mapping.md).

Full citations: [References](references.md).
