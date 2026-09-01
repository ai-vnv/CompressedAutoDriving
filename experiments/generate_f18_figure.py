#!/usr/bin/env python3
"""Figure 12: FP32 vs FP16 vs INT8 on the same fixed checkpoint and the same block."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/f18_fp16_control_v1"
CUR = ["C0", "C1", "C2", "C3", "C4"]
ROWS = [("A3", "FP32 anchor\n(prune + balanced KD)"),
        ("F16H", "FP16\n(same checkpoint, cast)"),
        ("A6", "INT8 PTQ\n(same checkpoint, quantized)")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    results = list(csv.DictReader(open(A / "results/pathway_results.csv", newline="")))
    fidelity = list(csv.DictReader(open(A / "results/same_state_fidelity.csv", newline="")))
    bench = json.loads((A / "results/precision_benchmark.json").read_text())["rows"]

    def status(pid, cur):
        return next((r["status"] for r in results
                     if r["pathway_id"] == pid and r["curriculum"] == cur), "-")

    def fid(pid, cur):
        return next((r for r in fidelity
                     if r["pathway_id"] == pid and r["curriculum"] == cur), None)

    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.32)

    # --- retention grid ---
    ax = fig.add_subplot(gs[0, :])
    vals = {"FAIL": 0, "PASS": 1}
    grid = np.array([[vals.get(status(p, c), 0) for c in CUR] for p, _ in ROWS])
    ax.imshow(grid, cmap=ListedColormap(["#D55E00", "#009E73"]), vmin=-0.5, vmax=1.5,
              aspect="auto")
    for i, (pid, _) in enumerate(ROWS):
        for j, c in enumerate(CUR):
            ax.text(j, i, status(pid, c), ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(range(5), CUR)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title("Precision Reduction on One Fixed Recovered Checkpoint: FP16 Retains, "
                 "Tested INT8 Does Not", pad=12)
    ax.text(0, -0.30, "Same anchor (SHA 64c84cd0…), same seeds 180201–208, same deterministic "
            "backend, same frozen gates. No retraining anywhere.",
            transform=ax.transAxes, fontsize=7, color="#333333", va="top")

    # --- omega MAE ---
    ax1 = fig.add_subplot(gs[1, 0])
    width = 0.26
    x = np.arange(5)
    for k, (pid, _) in enumerate(ROWS):
        ax1.bar(x + (k - 1) * width,
                [float(fid(pid, c)["omega_mae_rad_s"]) for c in CUR], width,
                label=pid, color=["#0072B2", "#009E73", "#D55E00"][k])
    ax1.set_xticks(x, CUR, fontsize=8)
    ax1.set_ylabel("ω MAE (rad/s)", fontsize=9)
    ax1.set_title("Same-state action error", fontsize=9.5)
    ax1.legend(fontsize=7, frameon=False)

    # --- spearman with gate line ---
    ax2 = fig.add_subplot(gs[1, 1])
    for k, (pid, _) in enumerate(ROWS):
        ax2.plot(x, [float(fid(pid, c)["omega_spearman"]) for c in CUR], "o-",
                 color=["#0072B2", "#009E73", "#D55E00"][k], label=pid, markersize=4)
    ax2.axhline(0.970, ls="--", lw=1, color="#555555")
    ax2.text(4.05, 0.970, " gate", fontsize=6.5, color="#555555", va="center")
    ax2.set_xticks(x, CUR, fontsize=8)
    ax2.set_ylabel("ω Spearman", fontsize=9)
    ax2.set_title("Fidelity gate component", fontsize=9.5)
    ax2.legend(fontsize=7, frameon=False, loc="lower left")

    # --- size and latency ---
    ax3 = fig.add_subplot(gs[1, 2])
    ids = [p for p, _ in ROWS]
    lat = [bench[p]["latency_median_us"] for p in ids]
    mem = [(bench[p]["logical_parameter_memory_bytes"] or 0) / 1024 for p in ids]
    bars = ax3.bar(np.arange(3) - 0.2, lat, 0.4, color="#37474F", label="latency median (µs)")
    ax3.set_ylabel("latency median (µs)", fontsize=8.5)
    ax3.set_xticks(range(3), ids, fontsize=8)
    twin = ax3.twinx()
    twin.bar(np.arange(3) + 0.2, mem, 0.4, color="#B0BEC5", label="param memory (KiB)")
    twin.set_ylabel("logical param memory (KiB)", fontsize=8.5)
    twin.spines["top"].set_visible(False)
    for b, v in zip(bars, lat):
        ax3.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=6.5)
    ax3.set_title("Cost (actor-only, 1 thread)", fontsize=9.5)
    ax3.text(0, -0.30, "INT8 param memory is not directly comparable (traced graph).",
             transform=ax3.transAxes, fontsize=6, color="#333333", va="top")

    (A / "figures").mkdir(exist_ok=True)
    fig.savefig(A / "figures/12_precision_control_fp32_fp16_int8.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(A / "figures/12_precision_control_fp32_fp16_int8.pdf", bbox_inches="tight")
    print("wrote 12_precision_control_fp32_fp16_int8.png/.pdf")


if __name__ == "__main__":
    main()
