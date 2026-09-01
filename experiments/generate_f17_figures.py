#!/usr/bin/env python3
"""Publication-ready English figures for F17 (optimization-method order).

Builds only figures whose inputs exist; skips the rest without inventing placeholders.
Captions use the licensed questions frozen in
docs/F17_COMPARISON_INTERPRETATION_AMENDMENT.md.
"""
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
A = ROOT / "artifacts/f17_optimization_method_order_v1"
FIG = A / "figures"
CUR = ["C0", "C1", "C2", "C3", "C4"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08, "axes.spines.top": False, "axes.spines.right": False,
})

STATUS_V = {"FAIL": 0, "UNRESOLVED": 1, "PENDING": 1, "PASS": 2, "REFERENCE": 3}
STATUS_C = ["#D55E00", "#B0BEC5", "#009E73", "#0072B2"]
ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def heat(ax, statuses, rows, cols, title, note=None, note_color="#333333"):
    matrix = np.asarray([[STATUS_V.get(v, 1) for v in r] for r in statuses])
    ax.imshow(matrix, cmap=ListedColormap(STATUS_C), vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    for i, row in enumerate(statuses):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center", fontsize=6.6, fontweight="bold",
                    color="white" if value in {"FAIL", "PASS", "REFERENCE"} else "#333333")
    ax.set_title(title, pad=10)
    ax.tick_params(length=0)
    if note:
        ax.text(0, -0.14, note, transform=ax.transAxes, fontsize=6.8, color=note_color,
                va="top")


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ready(rows, pid):
    return all(not str(status_of(rows, pid, c)).startswith(("PENDING", "PARTIAL")) for c in CUR)


def status_of(rows, pid, cur):
    return next((r["status"] for r in rows
                 if r["pathway_id"] == pid and r["curriculum"] == cur), "PENDING")


def figure_06(rows):
    """All pathways, shown as branches with their construction strings."""
    summary = load_json(A / "results/pathway_summary.json")
    if not summary:
        return
    present = [p for p in ORDER if p in summary["pathways"]]
    if not present:
        return
    grid = [[summary["pathways"][p]["statuses"].get(c, "PENDING") for c in CUR] for p in present]
    labels = [f"{p}  {summary['pathways'][p]['optimization_method_order']}"
              f"  [{summary['pathways'][p]['precision']}, w{summary['pathways'][p]['target_width']}]"
              for p in present]
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(present) + 1.8))
    heat(ax, grid, labels, CUR,
         "Cross-Curriculum Retention Across Optimization Pathways",
         "Separate constructed branches; not a linear chain. Deterministic evaluation, "
         "matched block 180201-208, gates relative to the deterministic A0 reference.")
    save(fig, "06_optimization_method_pathways")


def figure_07(rows):
    """A1 vs A2 vs A3: does rehearsal coverage determine recoverability?"""
    need = ["A1", "A2", "A3"]
    if not all(ready(rows, p) for p in need):
        return
    labels = {
        "A1": "Pruning only (no distillation)",
        "A2": "Pruning + C4-focused KD",
        "A3": "Pruning + balanced C0-C4 KD (anchor)",
    }
    grid = [[status_of(rows, p, c) for c in CUR] for p in need]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    heat(ax, grid, [labels[p] for p in need], CUR,
         "Curriculum Coverage Determines Recoverability After Pruning",
         "Same pruned parent, same teacher, same loss and budget; the manipulated factor "
         "across A2 vs A3 is the rehearsal coverage of the distillation data.")
    save(fig, "07_distillation_coverage_recovery")


def figure_08(rows):
    """A3 vs A6: PRIMARY QUANTIZATION — fixed FP32 parent through the frozen PTQ."""
    if not (ready(rows, "A3") and ready(rows, "A6")):
        return
    labels = {"A3": "Recovered FP32 anchor (fixed checkpoint)",
              "A6": "Same checkpoint after frozen PTQ (INT8)"}
    grid = [[status_of(rows, p, c) for c in CUR] for p in ("A3", "A6")]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    heat(ax, grid, [labels["A3"], labels["A6"]], CUR,
         "Does PTQ Reintroduce Cross-Curriculum Failure After FP32 Recovery?",
         "Fixed parent checkpoint, identical seeds, identical gates: any PASS-to-FAIL "
         "transition is associated with the frozen PTQ procedure for this checkpoint. "
         "No training occurs anywhere in this comparison.")
    save(fig, "08_fp32_to_ptq_transition")


def figure_09(rows):
    """A6 vs A8: alternative quantization routes from the same FP32 parent."""
    if not (ready(rows, "A6") and ready(rows, "A8")):
        return
    labels = {"A6": "PTQ route:  anchor -> PTQ",
              "A8": "QAT route:  anchor -> balanced QAT+KD -> INT8"}
    grid = [[status_of(rows, p, c) for c in CUR] for p in ("A6", "A8")]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    heat(ax, grid, [labels["A6"], labels["A8"]], CUR,
         "Can a QAT+KD Route Preserve Retention Relative to the PTQ Route?",
         "Both routes branch from the same recovered FP32 parent. A8 does not retrain the "
         "A6 INT8 graph; this compares alternative quantization routes, not a repair.")
    save(fig, "09_qat_recovery")


def figure_10(rows):
    """A5 vs A6: placement comparison — balanced KD inserted before PTQ."""
    if not (ready(rows, "A5") and ready(rows, "A6")):
        return
    labels = {"A5": "prune -> PTQ  (no distillation)",
              "A6": "prune -> balanced KD -> PTQ"}
    grid = [[status_of(rows, p, c) for c in CUR] for p in ("A5", "A6")]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    heat(ax, grid, [labels["A5"], labels["A6"]], CUR,
         "Does Distillation Before Quantization Improve Curriculum Retention?",
         "Placement comparison: what differs is the presence of balanced distillation "
         "before PTQ, not a reordering of identical stages. Not a factorial proof of "
         "optimization order.")
    save(fig, "10_method_order_comparison")


def main():
    rows = read_rows(A / "results/pathway_results.csv")
    for builder in (figure_06, figure_07, figure_08, figure_09, figure_10):
        try:
            builder(rows)
        except Exception as error:
            print(f"  skipped {builder.__name__}: {type(error).__name__}: {error}")
    pending = [s for s in ("06_optimization_method_pathways", "07_distillation_coverage_recovery",
                           "08_fp32_to_ptq_transition", "09_qat_recovery",
                           "10_method_order_comparison")
               if not (FIG / f"{s}.png").exists()]
    if pending:
        print("\nnot yet buildable:")
        for s in pending:
            print(f"  {s}")


if __name__ == "__main__":
    main()
