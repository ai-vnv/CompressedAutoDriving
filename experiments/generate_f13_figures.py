#!/usr/bin/env python3
"""Generate publication figures for the frozen F13 diagnostic evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.compressed_policy_analysis import actor_physical, file_sha256
from duckie_pomdp.explain.development_protocol import apply_semantic_intervention, normalize_physical
from duckie_pomdp.optimization.actor_compression import extract_original_actor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"
GROUPS = ("Lane", "Ego", "StopLine", "Pedestrian", "Stop", "PreviousAction")
PHASES = ("nominal", "lane_curve", "pedestrian_relevant", "stop_required", "stop_satisfied")
COLORS = {
    "Lane": "#0072B2", "Ego": "#56B4E9", "StopLine": "#009E73",
    "Pedestrian": "#D55E00", "Stop": "#E69F00", "PreviousAction": "#CC79A7",
}


def main() -> None:
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    root = resolve(config["artifacts"]["directory"])
    integrity = json.loads((root / "integrity/surrogate_equivalence.json").read_text())
    if integrity["gradient_attribution_authorized"]:
        raise RuntimeError("this generator implements the preregistered blocked-attribution branch")
    counterfactual = json.loads((root / "counterfactual/counterfactual_metrics.json").read_text())
    stress = json.loads((root / "failure_modes/exploratory/summary.json").read_text())
    figures = root / "figures"
    if figures.exists():
        raise FileExistsError("F13 figure directory already exists")
    figures.mkdir(parents=True)
    (root / "final").mkdir(exist_ok=True)
    configure_style()
    group_rows = read_group_rows(config)
    overall(group_rows, figures)
    heatmap_original(group_rows, figures, "v_cmd_mps", "v")
    heatmap_original(group_rows, figures, "omega_cmd_rad_s", "omega")
    for kind in ("v", "omega"):
        blocked_heatmap(figures / f"a7_phase_heatmap_{kind}", "A7 Distributional IG — UNRESOLVED")
        blocked_heatmap(figures / f"attribution_drift_heatmap_{kind}", "Attribution drift — UNRESOLVED")
    preservation(group_rows, figures)
    counterfactual_figure(counterfactual, figures)
    representative_panels(config, group_rows, figures, root)
    failure_figure(stress, figures)
    manifest = {
        "schema_version": 1,
        "config_sha256": file_sha256(CONFIG),
        "original_attribution_source": str(resolve(config["frozen"]["f11"]["r004_group_summary"])),
        "original_attribution_source_sha256": config["frozen"]["f11"]["r004_group_summary_sha256"],
        "a7_attribution_status": "BLOCKED",
        "a7_attribution_reason": integrity["reason"],
        "counterfactual_source": "counterfactual/counterfactual_metrics.json",
        "stress_source": "failure_modes/exploratory/summary.json",
        "bev_kind": "ego-centric public-belief schematic; no world pose or privileged GT",
        "figures": {path.name: file_sha256(path) for path in sorted(figures.iterdir())},
    }
    write_json(root / "final/figure_data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15,
    })


def read_group_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve(config["frozen"]["f11"]["r004_group_summary"])
    if file_sha256(path) != config["frozen"]["f11"]["r004_group_summary_sha256"]:
        raise RuntimeError("frozen R004 group summary hash mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def overall(rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, target, title in zip(axes, ("v_cmd_mps", "omega_cmd_rad_s"), (r"$v_{cmd}$", r"$\omega_{cmd}$")):
        selected = {r["group"]: r for r in rows if r["target"] == target and r["public_phase"] == "all"}
        values = np.asarray([float(selected[g]["mean_absolute_group_share"]) for g in GROUPS])
        low = np.asarray([float(selected[g]["share_ci_low"]) for g in GROUPS])
        high = np.asarray([float(selected[g]["share_ci_high"]) for g in GROUPS])
        x = np.arange(len(GROUPS))
        ax.bar(x - .18, values, .36, color=[COLORS[g] for g in GROUPS], yerr=np.vstack((values-low, high-values)), capsize=2, label="Original")
        ax.bar(x + .18, np.zeros_like(values), .36, color="#D9D9D9", edgecolor="#666", hatch="//", label="A7: unresolved")
        ax.set_title(title)
        ax.set_xticks(x, [short(g) for g in GROUPS], rotation=35, ha="right")
        ax.set_ylim(0, .62)
        ax.set_ylabel("Mean absolute attribution share")
        ax.legend(loc="upper right")
    fig.suptitle("Original vs A7 semantic attribution (A7 IG blocked)", fontweight="bold")
    fig.text(.5, -.03, "A7 has no exact frozen pre-conversion QAT state; no attribution value is imputed.", ha="center", fontsize=8)
    save(fig, out / "original_vs_a7_overall_attribution")


def matrix(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    lookup = {(r["public_phase"], r["group"]): float(r["mean_absolute_group_share"]) for r in rows if r["target"] == target}
    return np.asarray([[lookup[(phase, group)] for group in GROUPS] for phase in PHASES])


def heatmap_original(rows: list[dict[str, Any]], out: Path, target: str, suffix: str) -> None:
    values = matrix(rows, target)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    annotate(ax, values)
    heat_axes(ax)
    ax.set_title(f"Original phase-conditioned attribution — {target}")
    fig.colorbar(im, ax=ax, label="Mean absolute share", shrink=.78)
    save(fig, out / f"original_phase_heatmap_{suffix}")


def blocked_heatmap(path: Path, title: str) -> None:
    values = np.full((len(PHASES), len(GROUPS)), np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.imshow(np.zeros_like(values), cmap="Greys", vmin=0, vmax=1, alpha=.16, aspect="auto")
    heat_axes(ax)
    for i in range(len(PHASES)):
        for j in range(len(GROUPS)):
            ax.text(j, i, "—", ha="center", va="center", color="#666")
    ax.text(2.5, 2, "BLOCKED\nexact A7-QAT state not persisted", ha="center", va="center", fontsize=11, fontweight="bold", bbox={"facecolor":"white","edgecolor":"#555","alpha":.94})
    ax.set_title(title)
    save(fig, path)


def preservation(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    ax.axis("off")
    entries = []
    for phase in PHASES:
        for target in ("v_cmd_mps", "omega_cmd_rad_s"):
            values = {r["group"]: float(r["mean_absolute_group_share"]) for r in rows if r["target"] == target and r["public_phase"] == phase}
            entries.append([phase, "v" if target.startswith("v_") else "omega", max(values, key=values.get), "UNRESOLVED", "—", "—"])
    table = ax.table(cellText=entries, colLabels=("Phase", "Action", "Original top", "A7 top", "Spearman", "L1"), loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor("#DCEAF5"); cell.set_text_props(weight="bold")
        elif col >= 3: cell.set_facecolor("#EEEEEE")
    ax.set_title("Semantic-structure preservation: unresolved attribution branch", pad=15)
    fig.text(.5, .03, "Counterfactual and closed-loop preservation are evaluated separately; unavailable IG is not treated as zero.", ha="center", fontsize=8)
    save(fig, out / "semantic_structure_preservation")


def counterfactual_figure(data: dict[str, Any], out: Path) -> None:
    pairs = (("pedestrian_absent", "pedestrian_relevant", "Pedestrian"), ("stop_absent", "stop_required", "Stop"), ("lane_centered", "lane_curve", "Lane"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(pairs)); width=.34
    for action_index, (ax, action, label) in enumerate(zip(axes, ("v_cmd_mps", "omega_cmd_rad_s"), (r"$\Delta v$ (m/s)", r"$\Delta\omega$ (rad/s)"))):
        original=[]; a7=[]
        for op, phase, _ in pairs:
            m=data["summary"][op][phase][action]
            original.append(float(m["original_mean"])); a7.append(float(m["compressed_mean"]))
        ax.bar(x-width/2, original, width, color="#0072B2", label="Original")
        ax.bar(x+width/2, a7, width, color="#D55E00", label="A7 INT8")
        ax.axhline(0,color="#333",lw=.7); ax.set_xticks(x,[p[2] for p in pairs],rotation=20)
        ax.set_ylabel(label); ax.legend(); ax.set_title("Direct semantic intervention")
    fig.suptitle("Original vs compressed counterfactual response", fontweight="bold")
    fig.text(.5, -.02, "Stop release retained its direction but exceeded the frozen normalized mean-drift margin (0.1045 > 0.10).", ha="center", fontsize=8)
    save(fig, out / "counterfactual_original_vs_a7")


def representative_panels(config: dict[str, Any], rows: list[dict[str, Any]], out: Path, root: Path) -> None:
    source = ROOT / "artifacts/f11_ppo_explanation_v2/final_visualization/representative_frame_manifest.json"
    payload = json.loads(source.read_text())
    protocol = load_ppo_curriculum_protocol(resolve(config["frozen"]["contract"]["policy_config"]))
    original, _, _ = extract_original_actor(resolve(config["frozen"]["original"]["checkpoint"]), expected_sha256=config["frozen"]["original"]["sha256"])
    a7 = torch.jit.load(str(resolve(config["frozen"]["a7"]["checkpoint"])), map_location="cpu").eval()
    phases = ("lane_curve", "pedestrian_relevant", "stop_required", "stop_satisfied")
    fig, axes = plt.subplots(len(phases), 3, figsize=(10.3, 10.0))
    panel_data = {}
    for i, phase in enumerate(phases):
        item=payload["frames"][phase]; physical=np.asarray([item["public_29d"][n] for n in protocol.observation_order],dtype=np.float32)
        obs=normalize_physical(physical,protocol); oa=actor_physical(original,obs[None])[0]; aa=actor_physical(a7,obs[None])[0]
        image=plt.imread(ROOT/item["rgb_path"]); axes[i,0].imshow(image); axes[i,0].axis("off"); axes[i,0].set_title(f"{phase}: perception provenance")
        draw_public_bev(axes[i,1],item["public_29d"],phase)
        group={r["group"]:float(r["mean_absolute_group_share"]) for r in rows if r["target"]=="v_cmd_mps" and r["public_phase"]==phase}
        order=sorted(group,key=group.get,reverse=True)
        axes[i,2].axis("off")
        axes[i,2].text(0,1,f"Public belief + actor",weight="bold",va="top")
        axes[i,2].text(0,.84,f"Original: v={oa[0]:.3f} m/s, omega={oa[1]:.3f} rad/s")
        axes[i,2].text(0,.72,f"A7 INT8: v={aa[0]:.3f} m/s, omega={aa[1]:.3f} rad/s")
        axes[i,2].text(0,.56,"Original IG-v: "+", ".join(f"{g} {group[g]:.2f}" for g in order[:3]))
        axes[i,2].text(0,.43,"A7 IG: UNRESOLVED",color="#9A3412",weight="bold")
        if phase=="pedestrian_relevant":
            p=item["public_29d"]; axes[i,2].text(0,.25,f"P(ped)={p['pedestrian_existence_probability']:.3f}, r={p['pedestrian_range_mean_m']:.2f}±{p['pedestrian_range_std_m']:.2f} m")
        elif phase.startswith("stop"):
            p=item["public_29d"]; axes[i,2].text(0,.25,f"stop distance={p['stop_line_distance_m']:.2f} m; modes N/R/S={p['stop_mode_none']:.0f}/{p['stop_mode_required']:.0f}/{p['stop_mode_satisfied']:.0f}")
        else:
            p=item["public_29d"]; axes[i,2].text(0,.25,f"lane d={p['lane_lateral_error_mean_m']:.3f} m, phi={p['lane_heading_error_mean_rad']:.3f} rad")
        panel_data[phase]={"original_action":oa.tolist(),"a7_action":aa.tolist(),"source_rgb_sha256":item["rgb_sha256"]}
    fig.suptitle("Original vs A7 representative belief-action panels",fontweight="bold")
    fig.text(.5,.005,"Ego-centric BEV is reconstructed only from public belief variables; it is not the PPO input image and uses no world pose/GT.",ha="center",fontsize=8)
    save(fig,out/"bev_original_vs_a7_representative_panels")
    write_json(root/"final/representative_panel_data.json",panel_data)


def draw_public_bev(ax: Any, p: dict[str, float], phase: str) -> None:
    ax.set_aspect("equal"); ax.set_xlim(-1.35,1.35); ax.set_ylim(-.25,2.5); ax.grid(alpha=.15)
    y=np.linspace(-.2,2.5,100); k=float(p["lane_curvature_mean_inv_m"]); x=np.clip(.5*k*(y**2)*.08,-.7,.7)
    ax.plot(x,y,"--",color="#777",lw=1.2,label="lane belief")
    ax.arrow(0,0,0,.3,width=.035,head_width=.14,color="#0072B2",length_includes_head=True); ax.text(.08,.05,"ego")
    if float(p["pedestrian_existence_probability"])>.4:
        r=float(p["pedestrian_range_mean_m"]); b=float(p["pedestrian_bearing_mean_rad"]); px=r*np.sin(b); py=r*np.cos(b)
        sr=float(p["pedestrian_range_std_m"]); sb=float(p["pedestrian_bearing_std_rad"])
        ax.scatter([px],[py],c="#D55E00",s=40,zorder=4); ax.add_patch(Ellipse((px,py),max(.04,2*r*sb),max(.04,2*sr),angle=-np.degrees(b),fill=False,edgecolor="#D55E00",lw=1.2))
    distance=float(p["stop_line_distance_m"])
    if abs(distance)<=2.5 and (p["stop_mode_required"]>.5 or p["stop_mode_satisfied"]>.5): ax.plot([-.75,.75],[distance,distance],color="#E69F00",lw=2)
    ax.set_title("Ego-centric public-belief BEV"); ax.set_xlabel("x left (m)"); ax.set_ylabel("y forward (m)")


def failure_figure(data: dict[str, Any], out: Path) -> None:
    o=data["results"]["Original"]; a=data["results"]["A7"]
    labels=("Completion","No collision","No unsafe","Stop complete","No violation","Restart","No lane fail")
    def vals(s:dict[str,Any])->list[float]: return [s["completion_rate"],1-s["collision_rate"],1-s["unsafe_episode_rate"],s["stop_completion_rate"],1-s["stop_violation_rate"],s["restart_rate"],1-s["lane_failure_rate"]]
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.2)); x=np.arange(len(labels)); w=.36
    axes[0].bar(x-w/2,vals(o),w,color="#0072B2",label="Original"); axes[0].bar(x+w/2,vals(a),w,color="#D55E00",label="A7")
    axes[0].set_xticks(x,labels,rotation=40,ha="right"); axes[0].set_ylim(0,1.08); axes[0].set_ylabel("Episode rate"); axes[0].legend(); axes[0].set_title("C4 paired outcomes")
    axes[1].bar([0,1],[o["minimum_pedestrian_clearance_m"],a["minimum_pedestrian_clearance_m"]],color=["#0072B2","#D55E00"]); axes[1].set_xticks([0,1],["Original","A7"]); axes[1].set_ylabel("Minimum clearance (m)"); axes[1].set_title("Pedestrian clearance")
    so=o["phase_action"]["pedestrian_relevant"]["saturation_rate"]; sa=a["phase_action"]["pedestrian_relevant"]["saturation_rate"]
    axes[2].bar([0,1],[so,sa],color=["#0072B2","#D55E00"]); axes[2].axhline(so+.05,color="#555",ls="--",label="trigger"); axes[2].set_xticks([0,1],["Original","A7"]); axes[2].set_ylabel("Saturation rate"); axes[2].set_title(f"Pedestrian phase (delta={sa-so:+.3f})"); axes[2].legend()
    fig.suptitle("Explanation-guided C4 stress summary (4 paired seeds)",fontweight="bold")
    save(fig,out/"failure_mode_summary")


def annotate(ax: Any, values: np.ndarray) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if values[i,j]>.55 else "black")


def heat_axes(ax: Any) -> None:
    ax.set_xticks(np.arange(len(GROUPS)),[short(g) for g in GROUPS],rotation=30,ha="right")
    ax.set_yticks(np.arange(len(PHASES)),[p.replace("_"," ") for p in PHASES]); ax.set_xlabel("Semantic group"); ax.set_ylabel("Public phase")


def short(group: str) -> str:
    return {"PreviousAction":"Prev. action","StopLine":"Stop line"}.get(group,group)


def save(fig: Any, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path.with_suffix(".pdf")); fig.savefig(path.with_suffix(".png"),dpi=300); plt.close(fig)


def resolve(value: str) -> Path:
    return (CONFIG.parent / value).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
