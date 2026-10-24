"""Shared test setup."""
import matplotlib

# A GUI backend (TkAgg on Windows) breaks when the app tests draw figures from
# AppTest's script thread: Tk aborts the interpreter ("async handler deleted
# by the wrong thread"). Figures are only ever saved, never shown.
matplotlib.use("Agg")
