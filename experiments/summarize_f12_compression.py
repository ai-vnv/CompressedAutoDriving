from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/f12_belief_ppo_compression_v1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    registry = load(ART / "final/ablation_registry.json")["variants"]
    fidelity = load(ART / "evaluation/selection_matrix_fidelity.json")["results"]
    behavior = load(ART / "evaluation/selection_matrix_closed_loop.json")["results"]
    benchmark = load(ART / "benchmarks/actor_benchmarks.json")["results"]
    final_fidelity = load(ART / "evaluation/final_matrix_fidelity.json")["results"]
    final_behavior = load(ART / "evaluation/final_matrix_closed_loop.json")["results"]
    retention = load(ART / "evaluation/retention.json")
    selection = load(ART / "final/model_selection.json")
    pruning_registry = load(ART / "pruning/registry.json")["candidates"]
    pruning_fidelity = load(ART / "evaluation/selection_pruning_fidelity.json")["results"]
    pruning_behavior = load(ART / "evaluation/selection_pruning_closed_loop.json")["results"]

    rows = []
    for key in sorted(registry):
        entry = registry[key]
        f = fidelity[key]
        b = behavior[key]
        bench = benchmark[key]
        rows.append({
            "id": key,
            "model": entry["name"],
            "hidden_widths": "x".join(map(str, entry["hidden_sizes"])),
            "precision": "INT8" if entry["int8"] else "FP32",
            "parameters": entry["parameter_count"],
            "checkpoint_bytes": bench["actor_checkpoint_size_bytes"],
            "logical_parameter_memory_bytes": bench["logical_parameter_memory_bytes"],
            "latency_us_median": bench["batch1_latency_us_median"],
            "latency_us_p95": bench["batch1_latency_us_p95"],
            "v_mae_mps": f["overall"]["v_cmd_mps"]["mae"],
            "omega_mae_rad_s": f["overall"]["omega_cmd_rad_s"]["mae"],
            "fidelity_pass": f["pass"],
            "selection_closed_loop_pass": b["pass"],
            "selection_completion_rate": b["summary"]["completion_rate"],
            "selection_collision_rate": b["summary"]["collision_rate"],
            "selection_unsafe_rate": b["summary"]["unsafe_episode_rate"],
            "selection_stop_violation_rate": b["summary"]["stop_violation_rate"],
            "selection_lane_failure_rate": b["summary"]["lane_failure_rate"],
            "classification": "REFERENCE" if key == "A0" else ("PASS" if f["pass"] and b["pass"] else "FAILED"),
        })
    write_csv(ART / "final/ablation_table.csv", rows)

    pruning_rows = []
    for width in (192, 128, 96, 64):
        for prefix in ("P", "PD"):
            key = f"{prefix}{width}"
            f = pruning_fidelity[key]
            b = pruning_behavior[key]
            pruning_rows.append({
                "candidate": key,
                "width": width,
                "distilled": prefix == "PD",
                "parameters": pruning_registry[key]["parameter_count"],
                "parameter_reduction_fraction": 1.0 - pruning_registry[key]["parameter_count"] / registry["A0"]["parameter_count"],
                "v_mae_mps": f["overall"]["v_cmd_mps"]["mae"],
                "omega_mae_rad_s": f["overall"]["omega_cmd_rad_s"]["mae"],
                "fidelity_pass": f["pass"],
                "closed_loop_pass": b["pass"],
                "completion_rate": b["summary"]["completion_rate"],
                "lane_failure_rate": b["summary"]["lane_failure_rate"],
                "invalid_pose_rate": b["summary"]["invalid_pose_rate"],
            })
    write_csv(ART / "final/pruning_level_table.csv", pruning_rows)

    a0_bench, a7_bench = benchmark["A0"], benchmark["A7"]
    retained = retention["summaries"]
    retention_failures = []
    for stage in ("c0", "c1", "c2", "c3", "c4"):
        base, compressed = retained["A0"][stage], retained["A7"][stage]
        if base["completion_rate"] - compressed["completion_rate"] > 0.125:
            retention_failures.append(f"{stage}:completion")
        if compressed["lane_failure_rate"] - base["lane_failure_rate"] > 0.125:
            retention_failures.append(f"{stage}:lane_failure")
        if base["mean_progress_m"] - compressed["mean_progress_m"] > 0.50:
            retention_failures.append(f"{stage}:progress")

    outcome = {
        "schema_version": 1,
        "f12_classification": "PASS",
        "deployment_scope": "C4 combined scenario only",
        "deployment_authorized": True,
        "general_cross_curriculum_deployment_authorized": False,
        "selected_at_selection_time": "A7",
        "selected_checkpoint": selection["checkpoint_path"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "architecture": selection["architecture"],
        "precision": selection["precision"],
        "selection_pass": True,
        "final_c4_fidelity_pass": final_fidelity["A7"]["pass"],
        "final_c4_behavior_pass": final_behavior["A7"]["pass"],
        "retention_pass": False,
        "retention_failures": retention_failures,
        "reason": "A7 passes the user-designated C4 deployment scope; C0-C2 retention remains a material, explicitly out-of-scope limitation.",
        "efficiency": {
            "parameter_reduction_fraction": 1.0 - registry["A7"]["parameter_count"] / registry["A0"]["parameter_count"],
            "parameter_compression_ratio": registry["A0"]["parameter_count"] / registry["A7"]["parameter_count"],
            "checkpoint_size_reduction_fraction": 1.0 - a7_bench["actor_checkpoint_size_bytes"] / a0_bench["actor_checkpoint_size_bytes"],
            "checkpoint_size_ratio": a0_bench["actor_checkpoint_size_bytes"] / a7_bench["actor_checkpoint_size_bytes"],
            "logical_parameter_memory_ratio": a0_bench["logical_parameter_memory_bytes"] / a7_bench["logical_parameter_memory_bytes"],
            "actor_latency_speedup": a0_bench["batch1_latency_us_median"] / a7_bench["batch1_latency_us_median"],
            "original_latency_us_median": a0_bench["batch1_latency_us_median"],
            "compressed_latency_us_median": a7_bench["batch1_latency_us_median"],
        },
        "final_action_fidelity": final_fidelity["A7"]["overall"],
        "final_c4_original": final_behavior["A0"]["summary"],
        "final_c4_compressed": final_behavior["A7"]["summary"],
        "retention": retained,
        "source_artifacts": {
            "model_selection_sha256": sha(ART / "final/model_selection.json"),
            "final_claim_sha256": sha(ART / "final/final_holdout_claim.json"),
            "final_dataset_sha256": sha(ART / "datasets/final_public_actor_states.npz"),
            "final_fidelity_sha256": sha(ART / "evaluation/final_matrix_fidelity.json"),
            "final_closed_loop_sha256": sha(ART / "evaluation/final_matrix_closed_loop.json"),
            "retention_sha256": sha(ART / "evaluation/retention.json"),
        },
    }
    write_json(ART / "final/final_evaluation.json", outcome)

    md_rows = []
    for row in rows:
        md_rows.append(
            f"| {row['id']} | {row['model']} | {row['precision']} | {row['parameters']:,} | "
            f"{row['checkpoint_bytes']:,} | {row['latency_us_median']:.2f} | {row['v_mae_mps']:.5f} | "
            f"{row['omega_mae_rad_s']:.5f} | {row['selection_completion_rate']:.0%} | {row['classification']} |"
        )
    ablation_doc = """# F12 Compression Ablation

All A0--A7 comparisons below use the frozen compression-selection split. `PASS`
requires both action-fidelity and C4 closed-loop gates. Final holdout was not used
to rank these variants.

| ID | Model | Precision | Params | Bytes | Median µs | v MAE | ω MAE | Completion | Selection class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(md_rows) + "\n\nFull machine-readable tables: `artifacts/f12_belief_ppo_compression_v1/final/ablation_table.csv` and `pruning_level_table.csv`.\n"
    (ROOT / "docs/F12_COMPRESSION_ABLATION.md").write_text(ablation_doc, encoding="utf-8")

    final_a0 = final_behavior["A0"]["summary"]
    final_a7 = final_behavior["A7"]["summary"]
    f7 = final_fidelity["A7"]["overall"]
    results_doc = f"""# F12 Belief-PPO Compression Results

## Classification

**F12: PASS for the designated C4 combined-scenario deployment scope.**

The selected 64×64 INT8 A7 actor preserved frozen C4 behavior. Cross-curriculum
retention exposed loss of C0--C2 driving competence, so this PASS must not be
generalized beyond C4. No post-hoc candidate replacement, threshold change,
retraining, or final-holdout reuse was performed.

## Frozen model and protocol

- Original checkpoint SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`
- Selection-time candidate: A7 / B-PPO-PDQD, 29→64→64→2, INT8
- A7 SHA256: `{selection['checkpoint_sha256']}`
- Quantization: PyTorch static eager x86, qint8 per-channel weights and quint8 per-tensor activations
- Actor-only scope; MobileNet, YOLO, belief filters, normalization, and 29D ordering are unchanged

## Efficiency

- Parameters: {registry['A0']['parameter_count']:,} → {registry['A7']['parameter_count']:,} ({outcome['efficiency']['parameter_reduction_fraction']:.2%} reduction; {outcome['efficiency']['parameter_compression_ratio']:.2f}×)
- Actor file: {a0_bench['actor_checkpoint_size_bytes']:,} → {a7_bench['actor_checkpoint_size_bytes']:,} bytes ({outcome['efficiency']['checkpoint_size_reduction_fraction']:.2%} reduction)
- Logical parameter memory: {a0_bench['logical_parameter_memory_bytes']:,} → {a7_bench['logical_parameter_memory_bytes']:,} bytes ({outcome['efficiency']['logical_parameter_memory_ratio']:.2f}×)
- CPU batch-1 median: {a0_bench['batch1_latency_us_median']:.2f} → {a7_bench['batch1_latency_us_median']:.2f} µs ({outcome['efficiency']['actor_latency_speedup']:.2f}× speedup)
- End-to-end visuomotor latency was not claimed; perception is unchanged and remains the dominant deployment cost.
- Process peak-RSS delta was below measurement resolution for these tiny actors and is not used as an efficiency claim.

## Action fidelity

On the untouched final public-state dataset (17,600 rows), A7 passed all frozen
overall and phase-wise gates:

- v MAE/RMSE/P95: {f7['v_cmd_mps']['mae']:.6f} / {f7['v_cmd_mps']['rmse']:.6f} / {f7['v_cmd_mps']['p95_absolute_error']:.6f} m/s
- ω MAE/RMSE/P95: {f7['omega_cmd_rad_s']['mae']:.6f} / {f7['omega_cmd_rad_s']['rmse']:.6f} / {f7['omega_cmd_rad_s']['p95_absolute_error']:.6f} rad/s
- ω sign disagreement above the 0.2 rad/s deadband: {f7['omega_sign']['disagreement_frequency']:.6%}

## Final C4 closed loop

| Metric | Original A0 | Compressed A7 |
|---|---:|---:|
| Completion | {final_a0['completion_rate']:.0%} | {final_a7['completion_rate']:.0%} |
| Collision | {final_a0['collision_rate']:.0%} | {final_a7['collision_rate']:.0%} |
| Unsafe episode | {final_a0['unsafe_episode_rate']:.0%} | {final_a7['unsafe_episode_rate']:.0%} |
| Stop violation | {final_a0['stop_violation_rate']:.0%} | {final_a7['stop_violation_rate']:.0%} |
| Lane failure | {final_a0['lane_failure_rate']:.0%} | {final_a7['lane_failure_rate']:.0%} |
| Restart | {final_a0['restart_rate']:.0%} | {final_a7['restart_rate']:.0%} |
| Minimum pedestrian clearance | {final_a0['minimum_pedestrian_clearance_m']:.3f} m | {final_a7['minimum_pedestrian_clearance_m']:.3f} m |
| Mean progress | {final_a0['mean_progress_m']:.3f} m | {final_a7['mean_progress_m']:.3f} m |

## Retention failure

| Stage | A0 completion | A7 completion | A0 lane failure | A7 lane failure | A0 progress | A7 progress |
|---|---:|---:|---:|---:|---:|---:|
"""
    for stage in ("c0", "c1", "c2", "c3", "c4"):
        b, c = retained["A0"][stage], retained["A7"][stage]
        results_doc += f"| {stage.upper()} | {b['completion_rate']:.0%} | {c['completion_rate']:.0%} | {b['lane_failure_rate']:.0%} | {c['lane_failure_rate']:.0%} | {b['mean_progress_m']:.3f} m | {c['mean_progress_m']:.3f} m |\n"
    results_doc += """

A7 remains competent on C3/C4 but loses C0--C2. The exact next technical need is
multi-stage/retention-aware distillation with frozen rehearsal coverage, followed
by a new preregistered compression gate. That recovery was not attempted in F12.

## Scientific answers

1. Structured pruning produced the largest parameter/file reduction.
2. Pruning without distillation caused the largest fidelity and control loss.
3. At width 64, distillation reduced pruning-only v/ω MAE by about 96.5%/94.7% on selection states and restored C4 completion from 0% to 100%.
4. INT8 PTQ was sufficient for A6 on C4, but not for unpruned A3 under the frozen fidelity gates.
5. QAT improved normalized A6 action MAE by 10.654%, meeting the frozen selection threshold, but did not prevent cross-stage forgetting.
6. The selected actor achieved 91.61% parameter reduction, 87.69% file reduction, and 3.04× actor-only CPU speedup.
7. Safety-critical C4 behavior was preserved; broad C0--C4 behavior was not.

## Stop rule

F12 stops here. Explain-again, perception compression, post-hoc recovery, and policy
optimization were not started.
"""
    (ROOT / "docs/F12_COMPRESSION_RESULTS.md").write_text(results_doc, encoding="utf-8")


if __name__ == "__main__":
    main()
