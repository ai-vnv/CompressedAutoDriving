"""fig:precision — FP32 vs FP16 vs INT8 on the same checkpoint (3 panels, wide).

All values are read from the F18 artifacts. The FP16 member is named A9 in the
paper; its artifact ID is F16H.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from paper_plot_style import BLUE, DARK, GREEN, VERMILLION, save_fig

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts/f18_fp16_control_v1/results"
CUR = ["C0", "C1", "C2", "C3", "C4"]
MEMBERS = [("A3", "A3 FP32"), ("F16H", "A9 FP16"), ("A6", "A6 INT8")]
COLORS = {"A3": BLUE, "F16H": GREEN, "A6": VERMILLION}


def rows(name):
    with open(A / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


res, fid = rows("pathway_results.csv"), rows("same_state_fidelity.csv")
bench = json.loads((A / "precision_benchmark.json").read_text())["rows"]


def get(rws, pid, cur, key):
    return next(x[key] for x in rws if x["pathway_id"] == pid and x["curriculum"] == cur)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7.16, 1.9),
                                    gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

# ---- (a) retention grid ------------------------------------------------------
grid = np.array([[1 if get(res, pid, c, "status") == "PASS" else 0 for c in CUR]
                 for pid, _ in MEMBERS])
ax0.imshow(grid, cmap=ListedColormap([VERMILLION, GREEN]), vmin=-0.5, vmax=1.5,
           aspect="auto")
for i in range(3):
    for j in range(5):
        ax0.text(j, i, "pass" if grid[i, j] else "fail", ha="center", va="center",
                 color="white", fontsize=6.6, fontweight="bold")
ax0.set_xticks(range(5), CUR, fontsize=7)
ax0.set_yticks(range(3), [lab for _, lab in MEMBERS], fontsize=7)
ax0.tick_params(length=0)
for s in ax0.spines.values():
    s.set_visible(False)
ax0.set_title("(a) task verdicts", fontsize=8)

# ---- (b) omega Spearman vs the acceptance threshold --------------------------
x = np.arange(5)
# A3 and A9 overlap near 1.0; distinct line styles and markers keep the A3
# reference visible through the gaps of the dashed A9 line.
STYLES = {
    "A3": dict(ls="-", marker="o", lw=1.6, ms=4.2, mfc="none", zorder=2),
    "F16H": dict(ls=(0, (3, 2)), marker="^", lw=1.0, ms=3, zorder=3),
    "A6": dict(ls="-", marker="o", lw=1.0, ms=3, zorder=2),
}
for pid, lab in MEMBERS:
    y = [float(get(fid, pid, c, "omega_spearman")) for c in CUR]
    ax1.plot(x, y, color=COLORS[pid], label=lab, **STYLES[pid])
ax1.axhline(0.970, color="#555555", lw=0.8, ls="--")
ax1.text(4.1, 0.9705, "threshold", fontsize=5.8, color="#555555", va="bottom",
         ha="right")
ax1.set_xticks(x, CUR, fontsize=7)
ax1.set_ylim(0.915, 1.004)
ax1.tick_params(labelsize=6.5)
ax1.set_ylabel("$\\omega$ Spearman", fontsize=7)
ax1.legend(fontsize=5.6, frameon=False, loc="lower left")
ax1.set_title("(b) action fidelity", fontsize=8)

# ---- (c) latency and parameter memory ----------------------------------------
# A6 memory is absent from the benchmark json (the traced graph is opaque to
# the original benchmark); compute_int8_memory.py unpacks it (1-byte weights,
# fp32 biases, per-channel scale+zero point) into int8_parameter_memory.json.
a6_mem = json.loads(
    (Path(__file__).with_name("int8_parameter_memory.json")).read_text()
)["logical_parameter_memory_bytes"]
ids = [pid for pid, _ in MEMBERS]
lat = [bench[p]["latency_median_us"] for p in ids]
mem = [(bench[p]["logical_parameter_memory_bytes"] or a6_mem) / 1024 for p in ids]
xb = np.arange(3)
b1 = ax2.bar(xb - 0.19, lat, 0.38, color=DARK)
ax2.set_ylabel("latency ($\\mu$s)", fontsize=7)
ax2.set_xticks(xb, [lab.split()[1] for _, lab in MEMBERS], fontsize=7)
ax2.tick_params(labelsize=6.5)
tw = ax2.twinx()
tw.bar(xb + 0.19, mem, 0.38, color="#B0BEC5")
tw.set_ylabel("param. memory (KiB)", fontsize=7)
tw.tick_params(labelsize=6.5)
tw.spines["top"].set_visible(False)
for b, v in zip(b1, lat):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center",
             fontsize=5.8)
ax2.set_title("(c) actor cost", fontsize=8)

fig.subplots_adjust(left=0.075, right=0.94, top=0.88, bottom=0.10, wspace=0.42)
save_fig(fig, "fig_precision")
