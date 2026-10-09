# Soil tools

[← Documentation index](README.md)

*Where:* under each GEM file with coordinates, **Soil tools (sampling design, calibration, management zones)**. It works on the EC channels you choose. Implementation: [`soiltools.py`](../soiltools.py).

## Sampling design

Picks sampling sites that span the conductivity range and are spread over the field (response-surface sampling after Lesch, 2005, as in ESAP-RSSD):

1. Standardised principal-component scores of the chosen channels.
2. Design targets at the centre, on an inner ring (radius 1) and an outer ring (radius 1.75) — normal quantiles for a single channel.
3. For each target, of the 20 readings closest to it in component space, the one farthest from the sites already chosen.

Download the sites as CSV (map metres, plus lon / lat for degree files).

## Calibration

Upload measured soil properties with the same coordinate columns as the survey. Each sample takes the median conductivity of the readings within the matching radius, and the property is regressed on ln(EC) — on a log scale if chosen, optionally with an x, y trend surface (Lesch et al., 1995; ESAP-Calibrate). The tab shows R², RMSE and the coefficients, maps the predicted property and exports it.

## Management zones

Fuzzy c-means on the standardised channels (Bezdek, 1981), as in Management Zone Analyst (Fridgen et al., 2004). For 2–6 zones the fuzziness performance index (FPI) and modified partition entropy (MPE; Boydell & McBratney, 2002) are listed; the lowest values suggest the number of zones. Readings are mapped by zone and exported with their memberships.

Full citations: [References](references.md).
