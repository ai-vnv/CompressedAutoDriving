#!/usr/bin/env python3
"""Publication-ready English figures for F16.

Builds every figure the current evidence supports and skips the rest rather than
inventing placeholder content. Figures needing INT8 results or an opened holdout are
emitted only once those artifacts exist.
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
A = ROOT / "artifacts/f16_sequence_int8_recovery_v1"
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


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def heat(ax, statuses, rows, cols, title, note=None):
    matrix = np.asarray([[STATUS_V.get(v, 1) for v in r] for r in statuses])
    ax.imshow(matrix, cmap=ListedColormap(STATUS_C), vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    for i, row in enumerate(statuses):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="white" if value in {"FAIL", "PASS", "REFERENCE"} else "#333333")
    ax.set_title(title, pad=10)
    ax.tick_params(length=0)
    if note:
        ax.text(0, -0.16, note, transform=ax.transAxes, fontsize=6.8, color="#B00020",
                va="top", fontweight="bold")


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def figure_01():
    rows = read_rows(A / "results/collapse_map.csv")
    if not rows:
        return
    order = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    label = {
        "A0": "Original Policy", "A1": "Pruning Only",
        "A2": "Pruning + historical C4-focused KD", "A3": "PTQ of Original",
        "A4": "QAT + KD (unpruned)", "A5": "Pruning + PTQ",
        "A6": "Pruning + KD + PTQ", "A7": "Historical final INT8",
    }
    block = "180001-180008"
    grid, labels = [], []
    for model in order:
        grid.append([next((x["status"] for x in rows
                           if x["model_id"] == model and x["curriculum"] == c
                           and x["evaluation_seed_block"] == block), "PENDING") for c in CUR])
        labels.append(label[model])
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    heat(ax, grid, labels, CUR,
         "Where Does Cross-Curriculum Competence First Collapse?",
         f"F15 localization block {block}, non-deterministic backend. Parallel branches are "
         "shown as constructed, not as a linear history.")
    ax.set_xlabel("Curriculum")
    save(fig, "01_cross_curriculum_collapse_map")


def figure_02():
    rows = read_rows(A / "results/collapse_map.csv")
    if not rows:
        return
    picks = [
        ("A2", "180001-180008", "Pruning + historical C4-focused KD\nwidth 64, F15 localization block"),
        ("R64", "180201-180208", "Pruning + balanced C0-C4 KD\nwidth 64, F15 recovery block"),
    ]
    grid, labels = [], []
    for model, block, text in picks:
        grid.append([next((x["status"] for x in rows if x["model_id"] == model
                           and x["curriculum"] == c and x["evaluation_seed_block"] == block),
                          "PENDING") for c in CUR])
        labels.append(text)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    heat(ax, grid, labels, CUR,
         "Recovery with Curriculum-Balanced Knowledge Distillation",
         "Rows come from different evaluation blocks and are not bitwise comparable. The "
         "manipulated factor is rehearsal coverage; teacher, survivors, loss, optimizer and "
         "budget were held fixed.")
    save(fig, "02_historical_vs_multicurriculum_kd")


def figure_03():
    rows = read_rows(A / "results/training_realization_results.csv")
    if not rows:
        return
    reals = sorted({r["realization"] for r in rows})
    widths = sorted({int(r["target_width"]) for r in rows})
    fig, axes = plt.subplots(1, len(reals), figsize=(3.3 * len(reals) + 1.4, 3.5), squeeze=False)
    for index, real in enumerate(reals):
        grid, labels = [], []
        for width in widths:
            for sequence in ("Direct", "Progressive"):
                cells = [next((x["status"] for x in rows if int(x["target_width"]) == width
                               and x["sequence"] == sequence and x["realization"] == real
                               and x["curriculum"] == c), "PENDING") for c in CUR]
                grid.append(cells)
                labels.append(f"{sequence[0]}-{width}")
        ax = axes[0][index]
        heat(ax, grid, labels, CUR, f"Training realization {real}")
        if index:
            ax.set_yticklabels([])
    fig.suptitle("Does Pruning Schedule Affect FP32 Retention?",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.01, -0.05,
             "Matched endpoints: identical surviving original neurons, identical evaluation "
             "seeds, identical frozen gates, deterministic backend. Direct and Progressive share "
             "the training seed within each realization.",
             fontsize=6.8, color="#333333")
    save(fig, "03_direct_vs_progressive_fp32")


def figure_04():
    classification = load_json(A / "results/sequence_classification.json")
    widths_csv = read_rows(A / "results/width_results.csv")
    if not (classification and widths_csv):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.7),
                                   gridspec_kw={"width_ratios": [1.0, 1.4]})
    reals = ["S1", "S2", "S3"]
    grid, labels = [], []
    for row in sorted(widths_csv, key=lambda x: (int(x["target_width"]), x["sequence"])):
        per = dict(kv.split(":") for kv in row["all_five_pass_by_realization"].split("|") if ":" in kv)
        grid.append([per.get(s, "PENDING") for s in reals])
        labels.append(f"{row['sequence'][0]}-{row['target_width']}")
    heat(ax1, grid, labels, reals, "All-five-curricula verdict")
    ax1.set_xlabel("Training realization")

    legend = {
        "NO MATERIAL SEQUENCE EFFECT DETECTED": ("none", "#009E73"),
        "TRAINING-SEED SENSITIVE / INCONCLUSIVE": ("unstable", "#D55E00"),
        "SEQUENCE EFFECT SUPPORTED": ("supported", "#0072B2"),
        "PROVISIONAL_AWAITING_REPLICATION": ("provisional", "#B0BEC5"),
        "CONCORDANT_SINGLE_REALIZATION": ("n=1", "#B0BEC5"),
        "NOT_YET_REPLICATED": ("-", "#EEEEEE"),
    }
    widths = sorted(classification["sequence_classification_by_width"], key=int)
    for i, width in enumerate(widths):
        for j, curriculum in enumerate(CUR):
            status = classification["sequence_classification_by_width"][width][curriculum]["classification"]
            short, colour = legend.get(status, ("?", "#EEEEEE"))
            ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=colour, alpha=0.85))
            ax2.text(j, i, short, ha="center", va="center", fontsize=7, fontweight="bold",
                     color="white" if colour != "#EEEEEE" else "#333333")
    ax2.set_xticks(range(len(CUR)), CUR)
    ax2.set_yticks(range(len(widths)), [f"width {w}" for w in widths])
    ax2.set_xlim(-0.5, len(CUR) - 0.5)
    ax2.set_ylim(len(widths) - 0.5, -0.5)
    ax2.set_title("Pruning-Schedule Effect Classification", pad=10)
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle("Retention Across Independent Training Realizations",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.text(0.01, -0.06,
             "unstable = the Direct-versus-Progressive pruning-schedule direction changed between training seeds. "
             "No p-value is claimed from a small number of training realizations.",
             fontsize=6.8, color="#333333")
    save(fig, "04_training_realization_stability")


def figure_05():
    rows = read_rows(A / "results/training_realization_results.csv")
    if not rows:
        return
    widths = sorted({int(r["target_width"]) for r in rows})
    reals = sorted({r["realization"] for r in rows})
    grid, labels = [], []
    for width in widths:
        for real in reals:
            for sequence in ("Direct", "Progressive"):
                cells = [next((x["status"] for x in rows if int(x["target_width"]) == width
                               and x["sequence"] == sequence and x["realization"] == real
                               and x["curriculum"] == c), None) for c in CUR]
                if all(v is None for v in cells):
                    continue
                grid.append([v or "PENDING" for v in cells])
                labels.append(f"{width} {sequence[0]} {real}")
    if not grid:
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(grid) + 1.7))
    heat(ax, grid, labels, CUR, "Cross-Curriculum Retention Across Actor Widths",
         "Rows are width / sequence / training realization. Non-monotonic width behaviour is "
         "shown as measured; no capacity threshold is asserted.")
    save(fig, "05_width_retention_matrix")


def figure_08():
    rows = read_rows(A / "results/same_state_fidelity.csv")
    if not rows:
        return
    candidates = sorted({r["candidate_id"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(CUR))
    for candidate in candidates:
        values = []
        for curriculum in CUR:
            hit = next((r["omega_mae_rad_s"] for r in rows
                        if r["candidate_id"] == candidate and r["curriculum"] == curriculum), None)
            values.append(float(hit) if hit else np.nan)
        ax.plot(x, values, marker="o", ms=4, lw=1.3, label=candidate)
    gate = 0.200
    ax.axhline(gate, color="#D55E00", ls="--", lw=1.2)
    ax.text(len(CUR) - 1, gate * 1.05, f"frozen gate {gate} rad/s", ha="right",
            fontsize=7, color="#D55E00")
    ax.set_xticks(x, CUR)
    ax.set_xlabel("Curriculum")
    ax.set_ylabel("omega MAE versus Original (rad/s)")
    ax.set_title("Same-State Action Fidelity Across Optimization Stages")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(alpha=0.2)
    fig.text(0.01, -0.04,
             "Offline replay of the Original trajectories' normalized public 29D rows through each "
             "candidate. Every FP32 candidate passes the fidelity gate, including candidates that "
             "fail closed-loop behaviour.",
             fontsize=6.8, color="#333333")
    save(fig, "08_same_state_action_fidelity")


def main():
    for builder in (figure_01, figure_02, figure_03, figure_04, figure_05, figure_08):
        try:
            builder()
        except Exception as error:  # a missing input must not abort the rest
            print(f"  skipped {builder.__name__}: {type(error).__name__}: {error}")
    pending = [stem for stem in (
        "06_fp32_to_int8_transition", "07_width_sequence_precision_matrix",
        "09_failure_timeline", "10_final_candidate_performance",
        "11_efficiency_tradeoff", "12_final_holdout",
    ) if not (FIG / f"{stem}.png").exists()]
    print()
    print("not yet buildable (awaiting INT8 / candidate / holdout evidence):")
    for stem in pending:
        print(f"  {stem}")


if __name__ == "__main__":
    main()
