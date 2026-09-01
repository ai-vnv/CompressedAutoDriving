#!/usr/bin/env python3
"""Generate publication-ready English F15 figures from frozen artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
FIGURES = ARTIFACTS / "figures"
CURRICULA = ("C0", "C1", "C2", "C3", "C4")
MODEL_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
MODEL_NAMES = {
    "A0": "Original Policy", "A1": "Pruning Only", "A2": "Pruning + KD",
    "A3": "PTQ", "A4": "QAT + Distillation", "A5": "Pruning + PTQ",
    "A6": "Pruning + KD + PTQ", "A7": "Final INT8 Policy",
}
STATUS_VALUE = {"FAIL": 0, "UNRESOLVED": 1, "PASS": 2, "REFERENCE": 3, "MISSING": 1}
STATUS_COLORS = ["#D55E00", "#B0BEC5", "#009E73", "#0072B2"]


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def status_heatmap(ax, statuses, rows, cols, title):
    matrix = np.asarray([[STATUS_VALUE.get(value, 1) for value in row] for row in statuses])
    ax.imshow(matrix, cmap=ListedColormap(STATUS_COLORS), vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    for i, row in enumerate(statuses):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center", fontsize=7,
                    color="white" if value in {"FAIL", "PASS", "REFERENCE"} else "#333333",
                    fontweight="bold")
    ax.set_title(title, pad=10)
    ax.tick_params(length=0)


def figure_01(matrix):
    statuses = [[matrix["decisions"][model][curriculum.lower()]["status"] for curriculum in CURRICULA] for model in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    status_heatmap(ax, statuses, [MODEL_NAMES[m] for m in MODEL_ORDER], CURRICULA,
                   "Cross-Curriculum Competence Across Compression Stages")
    ax.set_xlabel("Curriculum")
    ax.text(0, -0.16, "PASS requires absolute curriculum competence and no unacceptable regression versus the Original Policy on paired seeds.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    save(fig, "01_cross_curriculum_competence_across_compression_stages")


def figure_02(decision):
    rows, labels = [], []
    for curriculum in CURRICULA:
        item = decision["first_collapse_by_curriculum"][curriculum.lower()]
        rows.append(item["statuses"])
        collapse = item["first_collapse"]
        labels.append("No observed collapse" if collapse is None else f"First failure after {collapse['after']}")
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    status_heatmap(ax, rows, [f"{c} — {label}" for c, label in zip(CURRICULA, labels)],
                   ["Original", "Pruning", "Pruning + KD", "Final INT8"],
                   "Where Does Cross-Curriculum Competence First Collapse?")
    ax.set_xlabel("Actual final-path stage")
    save(fig, "02_first_collapse_stage_by_curriculum")


def figure_03(pruning):
    model_ids = ["P192", "P128", "P96", "P64", "PD192", "PD128", "PD96", "PD64"]
    labels = ["192×192 — Pruning", "128×128 — Pruning", "96×96 — Pruning", "64×64 — Pruning",
              "192×192 — Pruning + KD", "128×128 — Pruning + KD", "96×96 — Pruning + KD", "64×64 — Pruning + KD"]
    statuses = [[pruning["decisions"][model][c.lower()]["status"] for c in CURRICULA] for model in model_ids]
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    status_heatmap(ax, statuses, labels, CURRICULA, "Curriculum Retention Across Pruning Widths")
    ax.axhline(3.5, color="white", linewidth=3)
    ax.set_xlabel("Curriculum")
    save(fig, "03_pruning_width_vs_curriculum_retention")


def figure_04(fidelity):
    v = np.asarray([[fidelity["results"][c.lower()][m]["metrics"]["v_cmd_mps"]["mae"] for c in CURRICULA] for m in MODEL_ORDER])
    omega = np.asarray([[fidelity["results"][c.lower()][m]["metrics"]["omega_cmd_rad_s"]["mae"] for c in CURRICULA] for m in MODEL_ORDER])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), constrained_layout=True)
    for ax, matrix, title, unit in zip(axes, (v, omega), ("Linear velocity", "Angular velocity / steering"), ("m/s", "rad/s")):
        image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(np.arange(5), CURRICULA)
        ax.set_yticks(np.arange(8), [MODEL_NAMES[m] for m in MODEL_ORDER])
        for i in range(8):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[i,j] > 0.6 * np.max(matrix) else "#222222")
        ax.set_title(f"{title} MAE ({unit})")
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle("Same-State Action Fidelity Across Compression Stages", fontweight="bold", fontsize=11)
    save(fig, "04_action_fidelity_by_curriculum_and_stage")


def figure_05(decision):
    events = [row for row in decision.get("failure_events", []) if row["family"] == "matrix"]
    preferred = next((row for row in events if row["model_id"] == "A7"), events[0] if events else None)
    if preferred is None or preferred["event_step"] is None:
        return
    model, curriculum, seed, event = preferred["model_id"], preferred["curriculum"], int(preferred["seed"]), int(preferred["event_step"])
    compressed_path = Path(preferred["trace_path"])
    with (ARTIFACTS / "localization/matrix_episodes.csv").open(newline="", encoding="utf-8") as stream:
        episode_rows = list(csv.DictReader(stream))
    original_row = next(
        row for row in episode_rows
        if row["model_id"] == "A0" and row["curriculum"] == curriculum and int(row["seed"]) == seed
    )
    original_path = Path(original_row["trace_path"])
    with np.load(original_path, allow_pickle=False) as a0, np.load(compressed_path, allow_pickle=False) as compressed:
        start, stop = max(0, event - 90), event + 46
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.8), sharex=True)
        for archive, label, color, style in ((a0, "Original Policy", "#0072B2", "-"), (compressed, MODEL_NAMES.get(model, model), "#D55E00", "--")):
            end = min(stop, len(archive["progress_m"]))
            x = np.arange(start, end)
            axes[0].plot(x, archive["progress_m"][start:end], label=label, color=color, linestyle=style)
            axes[1].plot(x, archive["physical_action"][start:end, 0], color=color, linestyle=style)
            axes[2].plot(x, archive["physical_action"][start:end, 1], color=color, linestyle=style)
        for ax in axes:
            ax.axvline(event, color="#C44E52", linewidth=1.2, label="First objective failure" if ax is axes[0] else None)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Progress (m)"); axes[1].set_ylabel("v_cmd (m/s)"); axes[2].set_ylabel("omega_cmd (rad/s)")
        axes[2].set_xlabel("Environment step")
        axes[0].legend(ncol=3, loc="best")
        axes[0].set_title(f"Original vs Compressed Policy at the First Objective Failure\n{curriculum.upper()}, seed {seed}: {preferred['event_labels']}")
    save(fig, "05_original_vs_compressed_failure_timeline")


def _load_recovery_results():
    rows = []
    for path in sorted((ARTIFACTS / "recovery").glob("**/selection_result.json")):
        payload = load_json(path)
        entry = payload["entry"]
        rows.append((entry["name"], [payload["behavior"]["decisions"][entry["variant"]][c.lower()]["status"] for c in CURRICULA], payload["fidelity"]["all_curricula_pass"], payload["eligible"]))
    return rows


def figure_06():
    rows = _load_recovery_results()
    fp32 = [row for row in rows if "INT8" not in row[0]]
    if not fp32:
        return
    fig, ax = plt.subplots(figsize=(7.0, max(2.3, 0.45 * len(fp32) + 1.5)))
    status_heatmap(ax, [row[1] for row in fp32], [row[0] for row in fp32], CURRICULA,
                   "Recovery with Multi-Curriculum Knowledge Distillation")
    save(fig, "06_multi_curriculum_distillation_recovery")


def figure_07():
    rows = _load_recovery_results()
    quantized = [row for row in rows if "INT8" in row[0]]
    if not quantized:
        return
    fig, ax = plt.subplots(figsize=(7.0, max(2.3, 0.45 * len(quantized) + 1.5)))
    status_heatmap(ax, [row[1] for row in quantized], [row[0] for row in quantized], CURRICULA,
                   "Does Quantization Preserve the Recovered Curriculum?")
    save(fig, "07_quantization_after_recovery")


def figure_08(final):
    entry = final["candidate"]
    statuses = [["REFERENCE"] * 5, [final["behavior"]["decisions"][entry["variant"]][c.lower()]["status"] for c in CURRICULA]]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    status_heatmap(ax, statuses, ["Original Policy", entry["name"]], CURRICULA, "Final Cross-Curriculum Performance")
    ax.text(0, -0.25, f"Once-only holdout classification: {final['classification']}", transform=ax.transAxes, fontsize=8, fontweight="bold")
    save(fig, "08_final_cross_curriculum_performance")


def figure_08_without_holdout():
    """Final cross-curriculum standing when the once-only holdout was never opened.

    F15 stopped before the holdout because no INT8 candidate satisfied the frozen gates.
    The figure therefore reports the recovery-selection split and says so on its face; it
    must not be read as a holdout result.
    """
    rows = _load_recovery_results()
    if not rows:
        return
    labels = ["Original Policy"] + [row[0] for row in rows]
    statuses = [["REFERENCE"] * 5] + [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.45 * len(labels) + 1.4)))
    status_heatmap(ax, statuses, labels, CURRICULA, "Final Cross-Curriculum Performance")
    ax.text(0, -0.30,
            "Recovery-selection seeds 180201-180208. The once-only holdout (180301-180308) was NOT opened:\n"
            "no INT8 candidate satisfied the frozen gates, so no final candidate could be frozen.",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold", color="#B00020")
    save(fig, "08_final_cross_curriculum_performance")


def figure_09(efficiency):
    """Compression against retained curricula, annotated with actor-only latency.

    Reads the F15 recovery efficiency schema. Models the script did not evaluate on a
    curriculum split (the Original reference and the historical A7 endpoint) are placed
    using their known F15 localization standing and marked as such.
    """
    localization = load_json(ARTIFACTS / "localization/matrix_results.json")

    def passed_from_localization(model_id):
        decisions = localization["decisions"].get(model_id)
        if decisions is None:
            return None
        return sum(decisions[c.lower()]["status"] in {"PASS", "REFERENCE"} for c in CURRICULA)

    # Several candidates share an identical (parameters, curricula-passed) coordinate, so
    # annotations are stacked deterministically instead of being drawn on top of each other.
    points = []
    for row in efficiency["models"]:
        passed = row.get("curricula_behavior_passed")
        split = "recovery selection"
        if passed is None:
            passed = passed_from_localization(row["model_id"])
            split = "localization"
        if passed is None:
            continue
        points.append((row, passed, split))

    occupancy: dict[tuple[int, int], int] = {}
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for row, passed, split in points:
        key = (int(row["parameter_count"]), int(passed))
        rank = occupancy.get(key, 0)
        occupancy[key] = rank + 1
        eligible = row.get("eligible")
        if eligible is True:
            color, marker = "#009E73", "o"
        elif eligible is False:
            color, marker = "#D55E00", "X"
        else:
            color, marker = "#0072B2", "s"
        ax.scatter(row["parameter_count"], passed, s=95, marker=marker, color=color,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(f'{row["label"]}\n{row["batch1_latency_us_median"]:.1f} us | {split}',
                    (row["parameter_count"], passed),
                    xytext=(9, -4 - 26 * rank), textcoords="offset points",
                    fontsize=6.4, va="top", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=0, shrinkB=2) if rank else None)
    ax.set_xscale("log")
    ax.set_ylim(-0.3, 5.6)
    ax.set_yticks(range(6))
    ax.set_xlabel("Actor parameters (log scale)")
    ax.set_ylabel("Curricula passed (of 5)")
    ax.set_title("Compression–Retention Trade-off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#009E73", label="eligible (all frozen gates)"),
        plt.Line2D([], [], marker="X", linestyle="", color="#D55E00", label="not eligible"),
        plt.Line2D([], [], marker="s", linestyle="", color="#0072B2", label="reference / historical"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7)
    ax.grid(alpha=0.2)
    save(fig, "09_compression_retention_tradeoff")


def main() -> None:
    matrix = load_json(ARTIFACTS / "localization/matrix_results.json")
    pruning = load_json(ARTIFACTS / "localization/pruning_results.json")
    fidelity = load_json(ARTIFACTS / "localization/open_loop_fidelity_by_curriculum.json")
    decision = load_json(ARTIFACTS / "localization/failure_localization_decision.json")
    figure_01(matrix); figure_02(decision); figure_03(pruning); figure_04(fidelity); figure_05(decision)
    figure_06(); figure_07()
    final_path = ARTIFACTS / "final/final_holdout.json"
    if final_path.exists():
        figure_08(load_json(final_path))
    else:
        figure_08_without_holdout()
    efficiency_path = ARTIFACTS / "final/efficiency_summary.json"
    if efficiency_path.exists(): figure_09(load_json(efficiency_path))


if __name__ == "__main__":
    main()
