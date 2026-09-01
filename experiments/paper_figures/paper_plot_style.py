"""Shared publication style for all paper figures."""
import matplotlib
from pathlib import Path as pathlib_Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300
FIG_DIR = str(pathlib_Path(__file__).resolve().parents[2] / "paper" / "figures")

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

# Okabe-Ito, colorblind-safe; consistent with every artifact figure in the repo.
GREEN = "#009E73"
VERMILLION = "#D55E00"
BLUE = "#0072B2"
GREY = "#B0BEC5"
DARK = "#37474F"


def save_fig(fig, name):
    for fmt in ("pdf", "png"):
        path = f"{FIG_DIR}/{name}.{fmt}"
        fig.savefig(path)
        print(f"Saved: {path}")
