#!/usr/bin/env python3
"""Generate the human-readable English F14/F12 figure package.

The script is synthesis-only: it reads frozen machine-readable artifacts and
does not invoke a simulator, policy training, attribution, or evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
F14 = ROOT / "artifacts/f14_explainability_aware_compression_v1"
F12 = ROOT / "artifacts/f12_belief_ppo_compression_v1"
OUT = F14 / "figures_explained_id"
SCRIPT_REL = "experiments/generate_f14_explained_figures_id.py"

GROUPS = ["Lane", "Ego", "StopLine", "Pedestrian", "Stop", "PreviousAction"]
GROUP_LABELS = ["Lane", "Ego", "Stop Line", "Pedestrian", "Stop", "Previous Action"]
PHASES = ["nominal", "lane_curve", "pedestrian_relevant", "stop_required", "stop_satisfied"]
PHASE_LABELS = ["Nominal driving", "Lane curve", "Pedestrian relevant", "Stop required", "Stop satisfied"]
ACTIONS = ["v_cmd_mps", "omega_cmd_rad_s"]

MODEL_NAMES = {
    "A0": "Original Policy",
    "A1": "Pruning Only",
    "A2": "Pruning + Knowledge Distillation",
    "A3": "Post-Training Quantization (PTQ)",
    "A4": "Quantization-Aware Training + Distillation",
    "A5": "Pruning + PTQ",
    "A6": "Pruning + Distillation + PTQ",
    "A7": "Final INT8: Pruning + Distillation + QAT",
}
SHORT_NAMES = {
    "A0": "Original",
    "A1": "Pruning Only",
    "A2": "Pruning + KD",
    "A3": "PTQ",
    "A4": "QAT + KD",
    "A5": "Pruning + PTQ",
    "A6": "Pruning + KD + PTQ",
    "A7": "Final INT8",
}

COLORS = {
    "ink": "#24323D",
    "muted": "#637381",
    "line": "#CBD5E1",
    "pale": "#F6F8FA",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "gold": "#E69F00",
    "orange": "#D55E00",
    "rose": "#CC79A7",
    "red": "#B23A48",
    "gray": "#8C8C8C",
}
GROUP_COLORS = [COLORS["blue"], COLORS["sky"], COLORS["gold"], COLORS["orange"], COLORS["green"], COLORS["rose"]]
STATUS_COLORS = {
    "REFERENCE": "#D9E7F5",
    "PRESERVED": "#CFE8DD",
    "PARTIAL": "#FBE7B2",
    "SHIFTED": "#F4C7C3",
    "PASS": "#CFE8DD",
    "FAIL": "#F4C7C3",
    "FAILED": "#F4C7C3",
    "NOT PRESERVED": "#E7B3B0",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_pair(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def title(fig: plt.Figure, main: str, subtitle: str | None = None) -> None:
    fig.suptitle(main, x=0.04, ha="left", y=0.985, fontsize=15, fontweight="bold", color=COLORS["ink"])
    if subtitle:
        fig.text(0.04, 0.94, subtitle, ha="left", va="top", fontsize=9.5, color=COLORS["muted"])


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, *,
            fc: str = "white", ec: str = COLORS["line"], fontsize: float = 9,
            weight: str = "normal", text_color: str = COLORS["ink"], radius: float = 0.02) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, linespacing=1.25)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = COLORS["muted"]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color))


def heatmap(ax: plt.Axes, matrix: np.ndarray, *, title_text: str, annotate: bool = True,
            cmap: str = "YlGnBu", vmin: float = 0.0, vmax: float = 1.0) -> None:
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if annotate:
                value = matrix[i, j]
                color = "white" if value > (vmin + 0.58 * (vmax - vmin)) else COLORS["ink"]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(len(GROUPS)), GROUP_LABELS, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(PHASES)), PHASE_LABELS)
    ax.set_title(title_text, pad=9)
    ax.set_xticks(np.arange(-0.5, len(GROUPS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PHASES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def group_matrix(rows: Iterable[dict[str, Any]], variant: str, action: str) -> np.ndarray:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["variant"] == variant and row["phase"] in PHASES and row["action"] == action:
            lookup[(row["phase"], row["group"])] = float(row["absolute_share"])
    return np.asarray([[lookup[(phase, group)] for group in GROUPS] for phase in PHASES])


def classification_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ablation = read_json("artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json")
    hierarchy = read_json("artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json")
    benchmark = read_json("artifacts/f12_belief_ppo_compression_v1/benchmarks/actor_benchmarks.json")
    return ablation["classification_axes"], hierarchy["classification"], benchmark["results"]


def status_cards(variants: list[str], main: str, subtitle: str, callout: str, extra_lines: dict[str, list[str]] | None = None) -> plt.Figure:
    axes, hierarchy, _ = classification_data()
    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, main, subtitle)
    rows = [
        ("Semantic attribution preservation", "semantic"),
        ("Counterfactual response preservation", "counterfactual"),
        ("Action fidelity to Original Policy", "action_fidelity"),
        ("C4 task completion", "closed_loop"),
    ]
    left = 0.27
    card_w = (0.69 - 0.03 * (len(variants) - 1)) / len(variants)
    xs = [left + i * (card_w + 0.03) for i in range(len(variants))]
    for row_i, (label, _) in enumerate(rows):
        y = 0.72 - row_i * 0.125
        ax.text(0.03, y + 0.035, label, ha="left", va="center", fontsize=9.5, color=COLORS["ink"], fontweight="bold")
        if row_i == 0:
            ax.text(0.03, y - 0.005, "preserved phase–action cells (out of 10)", ha="left", va="center", fontsize=7.5, color=COLORS["muted"])
        elif row_i == 1:
            ax.text(0.03, y - 0.005, "preserved primary tests (out of 3)", ha="left", va="center", fontsize=7.5, color=COLORS["muted"])
        elif row_i == 2:
            ax.text(0.03, y - 0.005, "same 29D input; v_cmd and omega_cmd", ha="left", va="center", fontsize=7.5, color=COLORS["muted"])
    for x, variant in zip(xs, variants):
        heading = SHORT_NAMES[variant]
        if variant == "A2":
            heading = "Pruning +\nKnowledge Distillation"
        elif variant == "A4":
            heading = "QAT +\nKnowledge Distillation"
        elif variant == "A6":
            heading = "Pruning + Distillation\n+ PTQ"
        elif variant == "A7":
            heading = "Final INT8\nPruning + Distillation + QAT"
        ax.text(x + card_w / 2, 0.835, heading, ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLORS["ink"], linespacing=1.15)
        sem = axes[variant]["semantic_attribution"]
        cf = axes[variant]["counterfactual_functional_sensitivity"]
        values = [
            f"{sem['preserved_phase_action_cells']}/{sem['total_phase_action_cells']}\n{sem['classification']}",
            f"{cf['preserved_primary_cells']}/{cf['total_primary_cells']}\n{cf['classification']}",
            "REFERENCE" if variant == "A0" else ("PASS" if hierarchy[variant]["action_fidelity"] == "PRESERVED" else "FAIL"),
            "REFERENCE" if variant == "A0" else ("PRESERVED" if hierarchy[variant]["closed_loop"] == "PRESERVED" else "NOT PRESERVED"),
        ]
        categories = [sem["classification"], cf["classification"], values[2], values[3]]
        for row_i, (value, category) in enumerate(zip(values, categories)):
            y = 0.72 - row_i * 0.125
            add_box(ax, (x, y), (card_w, 0.082), value, fc=STATUS_COLORS.get(category, "#ECEFF1"), ec="white", fontsize=8.5, weight="bold")
        if extra_lines and variant in extra_lines:
            ax.text(x + card_w / 2, 0.205, "\n".join(extra_lines[variant]), ha="center", va="top", fontsize=7.6, color=COLORS["muted"])
    add_box(ax, (0.055, 0.025), (0.89, 0.105), textwrap.fill(callout, width=125), fc="#EEF4F8", ec="#AFC4D3", fontsize=8.8, weight="bold")
    return fig


def figure_01() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "From Front Camera to Physical Action", "The deployed Belief-PPO pipeline and its information boundary")
    centers = [0.065, 0.215, 0.405, 0.605, 0.775, 0.92]
    widths = [0.105, 0.145, 0.165, 0.165, 0.115, 0.115]
    labels = [
        "DUCKIETOWN\nSIMULATOR",
        "INPUT\nFront RGB\n+ measured ego motion",
        "PERCEPTION\nMobileNetV3-small → lane\nYOLO11n → Duckie / stop sign",
        "BELIEF ESTIMATION\nLane EKF\nPedestrian EKF + existence\nStop-state logic",
        "29D SEMANTIC\nPOLICY INPUT",
        "PPO ACTOR → OUTPUT\n29 → 256 → 256 → 2\nv_cmd (m/s)\nomega_cmd (rad/s)",
    ]
    fills = ["#F1F5F9", "#E8F1F8", "#E8F4F0", "#FFF4DF", "#F2EBFA", "#FDEBE8"]
    for i, (cx, width, label, fill) in enumerate(zip(centers, widths, labels, fills)):
        add_box(ax, (cx - width / 2, 0.35), (width, 0.34), label, fc=fill, fontsize=8.1 if i not in (2, 3) else 7.7, weight="bold" if i in (0, 4, 5) else "normal")
        if i < len(centers) - 1:
            arrow(ax, (cx + width / 2 + 0.005, 0.52), (centers[i + 1] - widths[i + 1] / 2 - 0.005, 0.52))
    ax.text(0.50, 0.255, "INTERMEDIATE: measurement → belief → normalized 29D representation", ha="center", fontsize=9, color=COLORS["muted"])
    add_box(ax, (0.23, 0.08), (0.54, 0.09), "PPO does not receive RGB, detector boxes, or simulator world state directly.", fc="#FFF8E7", ec="#D9B44A", fontsize=10, weight="bold")
    return fig


def write_figure_spec_01() -> Path:
    spec_dir = OUT / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    labels = ["Simulator", "Front RGB + ego motion", "Perception", "Measurements", "Belief estimation", "29D representation", "PPO actor", "v_cmd, omega_cmd"]
    xs = [75, 205, 340, 465, 590, 715, 825, 925]
    for index, (label, x) in enumerate(zip(labels, xs)):
        nodes.append({"id": f"n{index}", "label": label, "x": x, "y": 175, "width": 105, "height": 64, "shape": "rounded", "font_size": 12})
    spec = {
        "title": "From Front Camera to Physical Action",
        "canvas": {"width": 1000, "height": 350},
        "style": {"font_family": "DejaVu Sans", "font_size": 12, "bg_color": "#FFFFFF", "palette": ["#E8F1F8", "#E8F4F0", "#FFF4DF", "#F2EBFA"]},
        "nodes": nodes,
        "edges": [{"from": f"n{i}", "to": f"n{i+1}", "color": "#637381"} for i in range(len(nodes) - 1)],
        "labels": [{"text": "PPO does not receive RGB directly.", "x": 500, "y": 290, "font_size": 14, "color": "#B23A48", "anchor": "middle"}],
    }
    path = spec_dir / "01_project_pipeline_from_rgb_to_action.figurespec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def figure_02() -> plt.Figure:
    # The integrity artifact binds the exact feature partition; display examples
    # are plain-language summaries of those frozen field names.
    contract = read_json("artifacts/f14_explainability_aware_compression_v1/integrity/actor_registry_verified.json")
    groups = contract["group_indices_zero_based"]
    sources = {
        "Lane": "MobileNet lane measurement → Lane EKF",
        "Ego": "Measured ego motion",
        "StopLine": "Public route stop-line observer",
        "Pedestrian": "YOLO Duckie → projection → EKF/existence",
        "Stop": "YOLO stop sign + stop-state logic",
        "PreviousAction": "Previous physical PPO command",
    }
    examples = {
        "Lane": "validity, lateral/heading error, curvature, uncertainty",
        "Ego": "actual linear velocity and yaw rate",
        "StopLine": "signed distance to the stopping point",
        "Pedestrian": "existence, range, bearing, rates, uncertainty",
        "Stop": "existence, range, bearing, uncertainty, mode",
        "PreviousAction": "previous v_cmd and omega_cmd",
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "What Is Inside the 29D Policy Input?", "Six complete, non-overlapping semantic groups; every dimension appears exactly once")
    positions = [(0.05, 0.56), (0.37, 0.56), (0.69, 0.56), (0.05, 0.20), (0.37, 0.20), (0.69, 0.20)]
    for index, (group, pos) in enumerate(zip(GROUPS, positions)):
        fields = groups[group]
        body = textwrap.fill(examples[group], width=40)
        source = textwrap.fill(f"Source: {sources[group]}", width=42)
        label = f"{GROUP_LABELS[index].upper()} — {len(fields)} dimensions\n\n{body}\n\n{source}"
        add_box(ax, pos, (0.27, 0.25), label, fc=mpl_colors.to_rgba(GROUP_COLORS[index], 0.10), ec=GROUP_COLORS[index], fontsize=8.0, weight="normal")
    ax.text(0.5, 0.105, "29D = belief-conditioned semantic policy representation", ha="center", fontsize=11, fontweight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.065, "It is not RGB, not full simulator state, and not a pure probability distribution.", ha="center", fontsize=9.2, color=COLORS["muted"])
    return fig


def figure_03(dev_rows: list[dict[str, str]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.3))
    title(fig, "What Information Contributes to Original Policy Decisions?", "Group Shapley relative contribution on the frozen 500-state development diagnostic set")
    matrices = [group_matrix(dev_rows, "A0", action) for action in ACTIONS]
    for ax, matrix, panel in zip(axes, matrices, ["Forward speed command (v_cmd)", "Steering command (omega_cmd)"]):
        im = heatmap(ax, matrix, title_text=panel)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.09, right=0.90, wspace=0.28)
    cax = fig.add_axes([0.925, 0.25, 0.014, 0.43])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean absolute Group Shapley share")
    fig.text(0.5, 0.01, "Larger values indicate larger relative contribution—not physical causality.", ha="center", fontsize=9, color=COLORS["muted"])
    return fig


def figure_04() -> plt.Figure:
    return status_cards(
        ["A0", "A1", "A2"],
        "Pruning Failure and Knowledge-Distillation Recovery",
        "Development diagnosis: the same 500 public 29D states for every actor",
        "Distillation recovered counterfactual response, action fidelity, and C4 behavior—without restoring the Original attribution pattern.",
    )


def figure_05() -> plt.Figure:
    return status_cards(
        ["A0", "A3", "A4"],
        "Post-Training Quantization and QAT-Guided Recovery",
        "PTQ converts a trained FP32 actor directly to INT8; QAT simulates quantization during recovery before conversion",
        "An actor can miss the numerical equivalence threshold while still completing the tested C4 scenario.",
    )


def figure_06() -> plt.Figure:
    return status_cards(
        ["A0", "A1", "A5"],
        "Why Pruning + PTQ Failed",
        "Paired diagnosis of the failed pruning branch before and after INT8 PTQ",
        "Adding PTQ did not rescue the degradation already present after pruning. This is pruning-dominated failure evidence, not proof that PTQ caused the original failure.",
    )


def figure_07() -> plt.Figure:
    _, _, benchmark = classification_data()
    p_reduction = 100.0 * (1.0 - benchmark["A7"]["dense_parameter_count"] / benchmark["A0"]["dense_parameter_count"])
    file_reduction = 100.0 * (1.0 - benchmark["A7"]["actor_checkpoint_size_bytes"] / benchmark["A0"]["actor_checkpoint_size_bytes"])
    speedup = benchmark["A0"]["batch1_latency_us_median"] / benchmark["A7"]["batch1_latency_us_median"]
    extras = {
        "A2": [f"FP32 · {benchmark['A2']['dense_parameter_count']:,} params", f"CPU median {benchmark['A2']['batch1_latency_us_median']:.2f} µs"],
        "A6": [f"INT8 · {benchmark['A6']['actor_checkpoint_size_bytes']:,} bytes", f"CPU median {benchmark['A6']['batch1_latency_us_median']:.2f} µs"],
        "A7": [f"{p_reduction:.2f}% fewer parameters", f"{file_reduction:.2f}% smaller actor file", f"{speedup:.2f}× actor-only CPU speedup"],
    }
    return status_cards(
        ["A2", "A6", "A7"],
        "Successful Deployment Pathway",
        "Recovered pruned actor → INT8 PTQ → final INT8 actor with QAT-guided distillation",
        "Deployment success does not imply semantic equivalence: the final actor passes action fidelity and C4 behavior while attribution and intervention sensitivity remain shifted.",
        extra_lines=extras,
    )


def figure_08() -> plt.Figure:
    _, hierarchy, _ = classification_data()
    variants = list(MODEL_NAMES)
    columns = ["Semantic Attribution", "Counterfactual Response", "Action Fidelity", "C4 Behavior"]
    keys = ["semantic", "counterfactual", "action_fidelity", "closed_loop"]
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    table_x = 3.0
    ax.set_xlim(0, len(columns) + table_x)
    ax.set_ylim(-0.9, len(variants) + 0.9)
    ax.axis("off")
    title(fig, "Optimization-Stage Failure-Mode Diagnostic Matrix", "Categories describe co-occurring diagnostic changes; columns do not assert a causal sequence")
    for j, label in enumerate(columns):
        header = label.replace(" ", "\n", 1)
        ax.text(table_x + j + 0.5, len(variants) + 0.28, header, ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"], linespacing=1.1)
    for i, variant in enumerate(variants):
        y = len(variants) - 1 - i
        ax.text(0.05, y + 0.5, SHORT_NAMES[variant], ha="left", va="center", fontsize=9, fontweight="bold" if variant in ("A0", "A7") else "normal", color=COLORS["ink"])
        for j, key in enumerate(keys):
            raw = hierarchy[variant][key]
            display = {"DRIFTED": "FAIL", "PRESERVED": "PRESERVED", "NOT PRESERVED": "NOT PRESERVED", "SHIFTED": "SHIFTED", "PARTIAL": "PARTIAL"}.get(raw, raw)
            if variant == "A0" and key in ("action_fidelity", "closed_loop"):
                display = "REFERENCE"
            x = table_x + j
            ax.add_patch(Rectangle((x, y + 0.08), 0.96, 0.84, facecolor=STATUS_COLORS.get(display, "#ECEFF1"), edgecolor="white"))
            ax.text(x + 0.48, y + 0.5, display, ha="center", va="center", fontsize=8.2, fontweight="bold", color=COLORS["ink"])
    ax.text(table_x, -0.45, "PARTIAL is distinct from SHIFTED; unavailable evidence would be shown as UNRESOLVED, never as zero drift.", ha="left", fontsize=8.5, color=COLORS["muted"])
    return fig


def figure_09(final: dict[str, Any]) -> plt.Figure:
    summaries = final["group_summaries"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    title(fig, "Original Policy vs Final INT8 Policy", "Final comparison: same 4,400 policy states and identical reference assignments")
    for row_i, variant in enumerate(["A0", "A7"]):
        for col_i, action in enumerate(ACTIONS):
            matrix = group_matrix(summaries[variant], variant if "variant" in summaries[variant][0] else "", action) if False else np.asarray([
                [next(float(x["absolute_share"]) for x in summaries[variant] if x["phase"] == phase and x["action"] == action and x["group"] == group) for group in GROUPS]
                for phase in PHASES
            ])
            ax = axes[row_i, col_i]
            im = heatmap(ax, matrix, title_text=f"{MODEL_NAMES[variant]} — {'v_cmd' if action == ACTIONS[0] else 'omega_cmd'}")
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.90, hspace=0.43, wspace=0.28)
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.48])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean absolute Group Shapley share")
    fig.text(0.5, 0.005, "Observed redistribution: steering attribution becomes relatively more concentrated on Previous Action after compression; this is not a causal claim.", ha="center", fontsize=8.8, color=COLORS["muted"])
    return fig


def figure_10(final: dict[str, Any]) -> plt.Figure:
    primary_keys = [
        ("pedestrian_absent", "pedestrian_relevant", "v_cmd_mps", "Pedestrian removed", "Speed response"),
        ("stop_absent", "stop_required", "v_cmd_mps", "Stop requirement removed", "Speed response"),
        ("lane_centered", "lane_curve", "omega_cmd_rad_s", "Lane centered", "Steering response"),
    ]
    comparisons = {(x["intervention"], x["phase"], x["action"]): x for x in final["counterfactual_comparison"]}
    original = [comparisons[k]["reference_mean"] for k, *_ in [(x[:3],) + x[3:] for x in primary_keys]] if False else []
    original, compressed = [], []
    for intervention, phase, action, _, _ in primary_keys:
        row = comparisons[(intervention, phase, action)]
        original.append(float(row["reference_mean"]))
        compressed.append(float(row["candidate_mean"]))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.7))
    title(fig, "Semantic Intervention Responses: Original vs Final INT8", "A counterfactual changes one public semantic concept and measures the actor-output change")
    for idx, (ax, item, a0, a7) in enumerate(zip(axes, primary_keys, original, compressed)):
        _, _, action, intervention_label, response_label = item
        bars = ax.bar([0, 1], [a0, a7], color=[COLORS["gray"], COLORS["orange"]], width=0.62, edgecolor="white")
        ax.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
        ax.set_xticks([0, 1], ["Original\nPolicy", "Final INT8\nPolicy"])
        ax.set_title(f"{intervention_label}\n{response_label}", fontsize=10)
        unit = "m/s" if action == "v_cmd_mps" else "rad/s"
        ax.set_ylabel(f"Mean change in action ({unit})")
        ax.grid(axis="y", alpha=0.18)
        ax.margins(y=0.20)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, [a0, a7]):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 0.52, f"{value:+.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    fig.text(0.5, 0.005, "Semantic policy-input interventions test functional sensitivity, not real-world causality.", ha="center", fontsize=8.8, color=COLORS["muted"])
    fig.subplots_adjust(top=0.76, bottom=0.20, left=0.07, right=0.98, wspace=0.33)
    return fig


def figure_11() -> plt.Figure:
    terms = [
        ("POMDP", "Decision-making when the true world state is only partly observable."),
        ("Observation", "Information available to the system at one time step."),
        ("Measurement", "A one-frame estimate extracted from sensor data."),
        ("Belief", "A state estimate combining past information, new measurements, and uncertainty."),
        ("EKF", "A recursive estimator for nonlinear measurements with uncertainty."),
        ("PPO", "A reinforcement-learning algorithm used to train the policy."),
        ("Attribution", "Relative contribution assigned to an input group for one actor output."),
        ("Group Shapley", "Exact six-group contribution analysis across all 64 coalitions."),
        ("Counterfactual", "A controlled change to one semantic input concept."),
        ("Pruning", "Removing hidden neurons to build a smaller dense actor."),
        ("Knowledge Distillation", "Training a smaller actor to reproduce a frozen Original Policy."),
        ("PTQ", "Converting a trained FP32 model to INT8 after training."),
        ("QAT", "Simulating quantization during recovery before INT8 conversion."),
        ("INT8", "Eight-bit integer representation used for compact deployment."),
        ("Fidelity", "Numerical similarity of optimized and Original actions on the same input."),
        ("Closed-loop", "The policy acts repeatedly while the environment changes in response."),
        ("Nominal phase", "Ordinary driving outside the registered curve, pedestrian, or active-stop phases."),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "Glossary for Readers New to AI and Robotics", "Plain-language meanings of the technical terms used in the figure package")
    cols = 2
    rows = int(np.ceil(len(terms) / cols))
    for idx, (term, definition) in enumerate(terms):
        col = idx // rows
        row = idx % rows
        x = 0.04 + col * 0.49
        y = 0.84 - row * 0.088
        add_box(ax, (x, y), (0.43, 0.065), f"{term}\n{definition}", fc="#F7F9FB", ec="#D9E2EA", fontsize=7.8, weight="normal")
    return fig


def build_manifest(outputs: dict[str, dict[str, Any]]) -> None:
    actors = read_json("artifacts/f14_explainability_aware_compression_v1/integrity/actor_registry_verified.json")
    model_hashes = {}
    raw_actors = actors.get("actors") or actors.get("verified_actors") or actors
    if isinstance(raw_actors, dict):
        for key, value in raw_actors.items():
            if isinstance(value, dict) and "sha256" in value:
                model_hashes[key] = value["sha256"]
    for entry in outputs.values():
        entry["figure_sha256"] = {Path(path).name: sha256(Path(path)) for path in entry["outputs"]}
        entry["outputs"] = [str(Path(path).relative_to(ROOT)) for path in entry["outputs"]]
    manifest = {
        "schema_version": 1,
        "language": "English",
        "purpose": "human-readable synthesis of immutable F12/F14 evidence",
        "generator": SCRIPT_REL,
        "generator_sha256": sha256(ROOT / SCRIPT_REL),
        "frozen_original_checkpoint_sha256": "02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250",
        "frozen_final_int8_sha256": "f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e",
        "actor_hashes": model_hashes,
        "development_dataset": {"states": 500, "source": "artifacts/f12_belief_ppo_compression_v1/datasets/development_public_actor_states.npz"},
        "final_dataset": {"states": 4400, "source": "artifacts/f11_ppo_explanation_v2/r004/locked_public_trace.npz"},
        "figures": outputs,
    }
    (OUT / "figure_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_figure_spec_01()
    dev_rows = read_csv("artifacts/f14_explainability_aware_compression_v1/ablation_group_summary.csv")
    final = read_json("artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json")
    jobs = [
        ("01_project_pipeline_from_rgb_to_action", figure_01, ["FORMULATION.md", "src/duckie_pomdp/control/ppo_environment.py", "configs/f10_ppo_visual_objects_v30.toml"], "architecture contract"),
        ("02_policy_input_29d_explained", figure_02, ["artifacts/f14_explainability_aware_compression_v1/coalition_schema.json", "configs/f14_explainability_aware_compression_v1.toml"], "frozen 29D contract"),
        ("03_original_policy_semantic_attribution", lambda: figure_03(dev_rows), ["artifacts/f14_explainability_aware_compression_v1/ablation_group_summary.csv"], "F14 development diagnostic: 500 states"),
        ("04_pruning_failure_and_distillation_recovery", figure_04, ["artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json", "artifacts/f14_explainability_aware_compression_v1/integrated_historical_metrics.json"], "F14 development + frozen F12 selection evidence"),
        ("05_ptq_and_qat_explained", figure_05, ["artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json", "artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv"], "F14 development + frozen F12 selection evidence"),
        ("06_pruning_plus_ptq_failure", figure_06, ["artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json", "artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json"], "F14 development + frozen F12 selection evidence"),
        ("07_successful_deployment_pathway", figure_07, ["artifacts/f14_explainability_aware_compression_v1/ablation_comparison_metrics.json", "artifacts/f12_belief_ppo_compression_v1/benchmarks/actor_benchmarks.json", "artifacts/f12_belief_ppo_compression_v1/final/model_selection.json"], "F14 development + frozen F12 deployment evidence"),
        ("08_failure_mode_diagnostic_matrix", figure_08, ["artifacts/f14_explainability_aware_compression_v1/failure_modes/failure_hierarchy.json"], "F14 development diagnostic: 500 states"),
        ("09_final_original_vs_int8_shapley", lambda: figure_09(final), ["artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json", "artifacts/f14_explainability_aware_compression_v1/final_a0_a7_shapley.csv"], "F14 final comparison: 4,400 frozen R004 states"),
        ("10_final_counterfactual_response", lambda: figure_10(final), ["artifacts/f14_explainability_aware_compression_v1/final_comparison_metrics.json", "artifacts/f14_explainability_aware_compression_v1/final_a0_a7_counterfactuals.csv"], "F14 final comparison: 4,400 frozen R004 states"),
        ("11_glossary_for_non_ai_readers", figure_11, ["FORMULATION.md", "docs/F12_COMPRESSION_PROTOCOL.md", "docs/F14_PROTOCOL.md"], "terminology synthesis; no new scientific measurements"),
    ]
    outputs: dict[str, dict[str, Any]] = {}
    for stem, builder, sources, dataset in jobs:
        paths = save_pair(builder(), stem)
        outputs[stem] = {"source_data_files": sources, "dataset": dataset, "script": SCRIPT_REL, "outputs": [str(path) for path in paths]}
    build_manifest(outputs)
    print(f"Generated {len(jobs)} figures in {OUT}")


if __name__ == "__main__":
    main()
