"""
Synthetic GEM-2 survey for testing the area maps: a 60 x 80 m grid of 31
lines (2 m apart) walked back and forth at 1 m/s with 10 readings a second.
I/Q come from the app's layered-earth forward model (emphysics) of a
two-layer ground with known targets; EC / MS are the half-space conversion
of I/Q, as WinGEM exports them.

    python sample_data/make_synthetic_gem2.py sample_data/synthetic_gem2_area_map.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the app's modules
import contouring as ctr  # noqa: E402
import emphysics as E  # noqa: E402

OUT = sys.argv[1]
FREQS = [475.0, 1525.0, 5325.0, 18325.0, 63025.0]
LINE_SPACING, LENGTH, SPEED, RATE = 2.0, 80.0, 1.0, 10.0     # m, m, m/s, Hz
N_LINES = 31
TOP = 0.6                                                     # m, top-layer thickness
rng = np.random.default_rng(42)

# ── survey geometry: serpentine lines along Y ────────────────────────────
rows, t = [], 36_000.0                                        # 10:00:00
for k in range(N_LINES):
    n = int(LENGTH * RATE / SPEED) + 1
    y = np.linspace(0.0, LENGTH, n)
    if k % 2:
        y = y[::-1]
    x = k * LINE_SPACING + rng.normal(0, 0.05, n)             # a little wobble off the line
    times = t + np.arange(n) / RATE
    t = times[-1] + 12.0                                      # turn at the end of the line
    rows.append(pd.DataFrame({"Line": k + 1, "Sample": np.arange(1, n + 1), "X": x, "Y": y, "t": times}))
df = pd.concat(rows, ignore_index=True)
x, y = df["X"].to_numpy(), df["Y"].to_numpy()

# ── ground model ─────────────────────────────────────────────────────────
top = np.full(len(df), 0.015)                                 # S/m, 15 mS/m topsoil
bottom = np.full(len(df), 0.025)                              # 25 mS/m subsoil
kappa = np.zeros(len(df))
# 1. conductive clay lens (round, deep): subsoil 120 mS/m
lens = np.hypot(x - 18, y - 55) < 9
bottom[lens] = 0.120
# 2. buried metal-free drainage channel (linear, shallow): topsoil 70 mS/m, 3 m wide, diagonal
d_channel = np.abs((y - 10) - 0.9 * (x - 5)) / np.hypot(1, 0.9)
top[d_channel < 1.5] = 0.070
# 3. saline patch (gradual): both layers rise towards the north-east corner
bump = np.exp(-((x - 52) ** 2 + (y - 18) ** 2) / (2 * 7.0 ** 2))
top += 0.060 * bump
bottom += 0.080 * bump
# 4. resistive gravel bar: 4 mS/m across the south-west
gravel = (x < 22) & (y < 20) & (np.hypot(x - 8, y - 8) < 11)
top[gravel] = 0.004
bottom[gravel] = 0.008
# 5. magnetic object (e.g. a kiln or basalt block), susceptibility 8e-3 SI, from the surface down
mag = np.hypot(x - 40, y - 62) < 2.0
kappa[mag] = 8e-3

sigma = np.column_stack([top, bottom])
kap = np.column_stack([kappa, kappa])                         # the object reaches below the topsoil
z = np.empty((len(df), len(FREQS)), dtype=complex)
for a in range(0, len(df), 2000):
    z[a: a + 2000] = E.forward_ppm_batch(FREQS, sigma[a: a + 2000], kap[a: a + 2000], [TOP])

# ── instrument: noise, slow drift, power line ────────────────────────────
noise_ppm = np.array([8.0, 5.0, 4.0, 4.0, 6.0])               # GEM-2 manual: a few ppm
drift = 3.0 * (df["t"].to_numpy() - 36_000.0) / 3600.0       # 3 ppm per hour on Q
inphase = z.real + rng.normal(0, 1, z.shape) * noise_ppm
quad = z.imag + rng.normal(0, 1, z.shape) * noise_ppm + drift[:, None]

out = df[["Line", "Sample", "X", "Y"]].copy()
lon, lat = ctr.local_metres_to_lonlat(x, y, (23.7000, 37.9000))
out["Lat"], out["Lon"] = lat, lon
out["Mark"] = (np.floor(np.where(df["Line"] % 2 == 1, y, LENGTH - y) / 10.0)).astype(int)  # every 10 m
out["Status"] = 0
out.loc[rng.choice(len(out), 25, replace=False), "Status"] = 1                       # a few ADC flags
secs = df["t"].to_numpy()
out["Time[ms]"] = np.round(secs * 1000.0, 1)
hh, mm = np.floor(secs / 3600), np.floor(secs % 3600 / 60)
out["Time[hhmmss.sss]"] = [f"{int(h):02d}{int(m):02d}{s:06.3f}" for h, m, s in zip(hh, mm, secs % 60)]
out["PowerLn"] = np.round(30 + 2 * np.sin(secs / 50) + rng.normal(0, 0.3, len(out)), 2)
ec_cols = []
for j, f in enumerate(FREQS):
    out[f"I_{f:g}Hz"] = np.round(inphase[:, j], 3)
    out[f"Q_{f:g}Hz"] = np.round(quad[:, j], 3)
for j, f in enumerate(FREQS):
    s, k = E.halfspace_from_ppm_batch(f, inphase[:, j], quad[:, j])
    out[f"EC{f:g}Hz[mS/m]"] = np.round(s * 1000.0, 4)
    ec_cols.append(f"EC{f:g}Hz[mS/m]")
    out[f"MSusc{f:g}Hz[1/1000]"] = np.round(k * 1000.0, 4)
out.insert(out.columns.get_loc(f"MSusc{FREQS[0]:g}Hz[1/1000]"), "TotalEC[mS/m]",
           np.round(out[ec_cols].mean(axis=1), 4))
out.to_csv(OUT, index=False)
print(f"{len(out)} readings, {N_LINES} lines -> {OUT}")
print(out.iloc[:3, :12].to_string())
