#!/usr/bin/env python3
"""Figure 11: state and action at the first objective failure, for every pathway x curriculum.

For each FAIL cell the representative episode follows the frozen rule (lowest failing
seed, then first objective failure event). The state and action columns are read from the
primary telemetry at exactly that step — nothing is recomputed or replayed.

Outputs: figures/11_failure_state_action_table.png/.pdf and
results/failure_state_action_table.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import first_objective_failure_event  # noqa: E402

A = ROOT / "artifacts/f17_optimization_method_order_v1"
CUR = ["c0", "c1", "c2", "c3", "c4"]
ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
SHORT = {
    "A0": "Original", "A1": "prune", "A2": "prune+KD(C4)", "A3": "prune+KD(bal)",
    "A4": "PTQ only", "A5": "prune+PTQ", "A6": "KD(bal)+PTQ",
    "A7": "KD(C4)+PTQ+QAT", "A8": "KD(bal)+QAT",
}
FLAGS = ("collision", "unsafe", "stop_violation", "lane_failure", "invalid_pose", "timeout")


def episode_event(trace: Path):
    with np.load(trace, allow_pickle=False) as archive:
        flags = {n: np.asarray(archive[n], dtype=bool) for n in
                 FLAGS + ("terminated", "truncated", "completed")}
        physical = np.asarray(archive["public_physical_29d"], dtype=np.float32)
        action = np.asarray(archive["physical_action"], dtype=np.float32)
        names = [str(v) for v in archive["feature_names"]]
    steps = [{"step": i, **{n: bool(flags[n][i]) for n in flags}} for i in range(len(action))]
    event = first_objective_failure_event(steps)
    return event, physical, action, names, len(action)


def main() -> None:
    with open(A / "results/pathway_results.csv", newline="") as fh:
        verdicts = {(r["pathway_id"], r["curriculum"].lower()): r["status"]
                    for r in csv.DictReader(fh)}

    rows_out = []
    for pid in ORDER:
        for cur in CUR:
            status = verdicts.get((pid, cur))
            if status != "FAIL":
                continue
            # Lowest failing seed: an episode is failing if it has any objective event.
            chosen = None
            for seed_dir in sorted((A / "telemetry" / pid / cur).glob("seed_*"),
                                   key=lambda p: int(p.name.split("_")[1])):
                event, physical, action, names, length = episode_event(seed_dir / "trace.npz")
                if event is not None:
                    chosen = (int(seed_dir.name.split("_")[1]), event, physical, action, names, length)
                    break
            if chosen is None:
                continue
            seed, event, physical, action, names, length = chosen
            step = min(int(event["step"]), len(physical) - 1)
            ix = {n: names.index(n) for n in (
                "actual_linear_velocity_mps", "stop_line_distance_m",
                "lane_lateral_error_mean_m", "lane_heading_error_mean_rad",
                "stop_mode_none", "stop_mode_required", "stop_mode_satisfied",
                "pedestrian_existence_probability")}
            state = physical[step]
            mode = ("required" if state[ix["stop_mode_required"]] > 0.5 else
                    "satisfied" if state[ix["stop_mode_satisfied"]] > 0.5 else "none")
            rows_out.append({
                "pathway": pid, "construction": SHORT[pid], "curriculum": cur.upper(),
                "seed": seed, "failure_step": step, "episode_length": length,
                "failure_labels": "|".join(event["event_labels"]),
                "v_actual_mps": round(float(state[ix["actual_linear_velocity_mps"]]), 3),
                "stop_line_dist_m": round(float(state[ix["stop_line_distance_m"]]), 3),
                "lane_lat_err_m": round(float(state[ix["lane_lateral_error_mean_m"]]), 3),
                "heading_err_rad": round(float(state[ix["lane_heading_error_mean_rad"]]), 3),
                "stop_mode": mode,
                "ped_prob": round(float(state[ix["pedestrian_existence_probability"]]), 2),
                "v_cmd_mps": round(float(action[step, 0]), 3),
                "omega_cmd_rad_s": round(float(action[step, 1]), 3),
            })

    out_csv = A / "results/failure_state_action_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)
    print(f"wrote {out_csv.name}: {len(rows_out)} FAIL cells")

    # ---- figure: status grid on top, failure detail table below ----
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    n = len(rows_out)
    fig = plt.figure(figsize=(13.6, 3.1 + 0.30 * n))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, n], hspace=0.34)

    ax0 = fig.add_subplot(gs[0])
    colors = {"PASS": "#009E73", "FAIL": "#D55E00", "REFERENCE": "#0072B2"}
    for i, pid in enumerate(ORDER):
        for j, cur in enumerate(CUR):
            st = verdicts.get((pid, cur), "-")
            ax0.add_patch(plt.Rectangle((j, len(ORDER) - 1 - i), 1, 1,
                                        color=colors.get(st, "#B0BEC5"), ec="white", lw=1.5))
            ax0.text(j + 0.5, len(ORDER) - 0.5 - i, st[:4], ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white")
    ax0.set_xlim(0, 5); ax0.set_ylim(0, len(ORDER))
    ax0.set_xticks([j + 0.5 for j in range(5)], [c.upper() for c in CUR])
    ax0.set_yticks([len(ORDER) - 0.5 - i for i in range(len(ORDER))],
                   [f"{p}  {SHORT[p]}" for p in ORDER], fontsize=7)
    ax0.tick_params(length=0)
    for s in ax0.spines.values():
        s.set_visible(False)
    ax0.set_title("Where Each Optimization Pathway Fails: State and Action at the First Objective Failure",
                  fontsize=12, fontweight="bold", pad=12)

    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    headers = ["Pathway", "Cur", "Seed", "Step/Len", "Failure label(s)", "v_act\n(m/s)",
               "stop_dist\n(m)", "lat_err\n(m)", "head_err\n(rad)", "stop\nmode",
               "v_cmd\n(m/s)", "omega_cmd\n(rad/s)"]
    cells = [[f"{r['pathway']} {r['construction']}", r["curriculum"], r["seed"],
              f"{r['failure_step']}/{r['episode_length']}",
              r["failure_labels"].replace("|termination_without_completion", "|no-compl"),
              r["v_actual_mps"], r["stop_line_dist_m"], r["lane_lat_err_m"],
              r["heading_err_rad"], r["stop_mode"], r["v_cmd_mps"], r["omega_cmd_rad_s"]]
             for r in rows_out]
    table = ax1.table(cellText=cells, colLabels=headers, loc="center",
                      colWidths=[0.14, 0.035, 0.05, 0.06, 0.20, 0.05, 0.06, 0.05, 0.06, 0.055, 0.05, 0.06])
    table.auto_set_font_size(False); table.set_fontsize(6.8); table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#37474F"); cell.set_text_props(color="white", fontweight="bold")
        elif "collision" in str(cells[r - 1][4]) or "stop_violation" in str(cells[r - 1][4]):
            cell.set_facecolor("#FDEBD0")
    fig.text(0.02, 0.005,
             "Representative episode per FAIL cell: lowest failing seed, first objective failure event (frozen rule). "
             "State and action are read from the primary telemetry at exactly that step. Deterministic block 180201-208.",
             fontsize=6.5, color="#333333")
    (A / "figures").mkdir(exist_ok=True)
    fig.savefig(A / "figures/11_failure_state_action_table.png", dpi=300, bbox_inches="tight")
    fig.savefig(A / "figures/11_failure_state_action_table.pdf", bbox_inches="tight")
    print("wrote 11_failure_state_action_table.png/.pdf")


if __name__ == "__main__":
    main()
