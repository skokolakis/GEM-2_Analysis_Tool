"""
GEM-2 export format: channel columns, quality flags, event markers and
distance along each survey line.

Pure functions only (no Streamlit). Column names follow the WinGEM export
(GEM-2 Manual v3.8, 2004): Line Sample X Y Mark Status GPSStat Time[ms]
Time[hhmmss.sss] PowerLn I_<f>Hz Q_<f>Hz QSum EC<f>Hz[mS/m] TotalEC[mS/m]
MSusc<f>Hz[1/1000].
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import contouring as ctr

CHANNEL_PATTERNS = {
    "EC": re.compile(r"^EC(\d+(?:\.\d+)?)Hz\[mS/m\]$"),
    "MS": re.compile(r"^MSusc(\d+(?:\.\d+)?)Hz\[1/1000\]$"),
    "I": re.compile(r"^I_(\d+(?:\.\d+)?)Hz$"),
    "Q": re.compile(r"^Q_(\d+(?:\.\d+)?)Hz$"),
}
AUX_CHANNELS = tuple(ctr.AUX_LABELS)          # PowerLn, QSum, TotalEC[mS/m]
GEM_MODES = ("EC", "MS", "I", "Q", "AUX")
FREQUENCY_MODES = ("EC", "MS", "I", "Q")      # modes whose channels are frequencies
MODE_NAMES = {
    "EC": "Electrical conductivity",
    "MS": "Magnetic susceptibility",
    "I": "In-phase",
    "Q": "Quadrature",
    "AUX": "Auxiliary channels",
}

DISTANCE_COL = "Distance (m)"   # column added to prepared tables (pipeline.prepare_gem_table)

DISTANCE_METHODS = {
    "Y": "Y column",
    "projection": "Projection on the survey axis",
    "path": "Path length along each line",
    "sample": "Reading number × spacing",
    "markers": "Between event markers (dead reckoning)",
}


def find_channels(columns) -> dict[str, dict[str, str]]:
    """{mode: {label: column}} for every recognised channel column, e.g. {"EC": {"4525Hz": "EC4525Hz[mS/m]"}}."""
    out: dict[str, dict[str, str]] = {m: {} for m in GEM_MODES}
    for col in columns:
        name = str(col)
        for mode, pattern in CHANNEL_PATTERNS.items():
            match = pattern.match(name)
            if match:
                out[mode][f"{match.group(1)}Hz"] = name
                break
        else:
            if name in AUX_CHANNELS:
                out["AUX"][name] = name
    return out


def channel_column(mode: str, label: str) -> str:
    """Raw column name for a mode and channel label such as '4525Hz'."""
    if mode == "EC":
        return f"EC{label}[mS/m]"
    if mode == "MS":
        return f"MSusc{label}[1/1000]"
    if mode in ("I", "Q"):
        return f"{mode}_{label}"
    if mode == "AUX":
        return label
    raise ValueError(f"Unknown mode: {mode!r}")


