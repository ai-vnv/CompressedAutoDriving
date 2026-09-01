"""fig:matrix — 9 members x 5 curricula verdict matrix, read from artifacts.

The unmodified reference A0 is omitted (its row is all-reference by construction).

Verdicts come from the results CSVs, never hardcoded. Single-column figure.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from paper_plot_style import BLUE, GREEN, VERMILLION, save_fig

ROOT = Path(__file__).resolve().parents[2]
F17 = ROOT / "artifacts/f17_optimization_method_order_v1/results/pathway_results.csv"
F18 = ROOT / "artifacts/f18_fp16_control_v1/results/pathway_results.csv"

CUR = ["C0", "C1", "C2", "C3", "C4"]
ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
LABEL = {
    "A0": "A0  original (ref.)", "A1": "A1  prune",
    "A2": "A2  prune+KD(C4)", "A3": "A3  prune+KD(bal)",
    "A4": "A4  PTQ only", "A5": "A5  prune+PTQ",
    "A6": "A6  KD(bal)+PTQ", "A7": "A7  KD(C4)+PTQ+QAT",
    "A8": "A8  KD(bal)+QAT", "A9": "A9  KD(bal)+FP16",
}


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


r17, r18 = rows(F17), rows(F18)


def status(pid, cur):
    src = r18 if pid == "A9" else r17
    key = "F16H" if pid == "A9" else pid  # artifact ID for the FP16 member
    return next(x["status"] for x in src if x["pathway_id"] == key and x["curriculum"] == cur)


VAL = {"FAIL": 0, "PASS": 1, "REFERENCE": 2}
grid = np.array([[VAL[status(p, c)] for c in CUR] for p in ORDER])

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.imshow(grid, cmap=ListedColormap([VERMILLION, GREEN, BLUE]), vmin=-0.5, vmax=2.5,
          aspect="auto")
WORD = {0: "fail", 1: "pass", 2: "ref."}
for i, p in enumerate(ORDER):
    for j, c in enumerate(CUR):
        ax.text(j, i, WORD[grid[i, j]], ha="center", va="center", color="white",
                fontsize=6.4, fontweight="bold")
ax.set_xticks(range(5), CUR, fontsize=7)
ax.set_yticks(range(len(ORDER)), [LABEL[p] for p in ORDER], fontsize=6.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
# thin separators between the FP32, INT8, and FP16 blocks
for y in (2.5, 7.5):
    ax.axhline(y, color="white", lw=2.2)
fig.subplots_adjust(left=0.34, right=0.995, top=0.995, bottom=0.06)
save_fig(fig, "fig_matrix")
