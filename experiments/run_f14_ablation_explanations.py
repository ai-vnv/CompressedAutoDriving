"""Run the frozen F14 A0--A7 same-state diagnostic exactly once."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from duckie_pomdp.explain.compression_diagnostics import (
    ACTION_NAMES,
    compare_group_summaries,
    counterfactual_comparison,
    counterfactual_preservation_classification,
    evaluate_semantic_counterfactuals,
    file_sha256,
    load_f14_config,
    load_frozen_actors,
    load_policy_contract,
    resolve_config_path,
    semantic_structure_classification,
    summarize_group_attribution,
    verify_frozen_file,
)
from duckie_pomdp.explain.group_shapley import GROUP_ORDER, exact_group_shapley


def _refuse(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"immutable F14 output already exists: {path}")


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _historical(config: dict) -> dict:
    out = {}
    for key in ("selection_fidelity", "selection_behavior", "benchmarks", "retention"):
        section = config["historical"]
        path = resolve_config_path(config, section[key])
        if file_sha256(path) != section[f"{key}_sha256"]:
            raise RuntimeError(f"historical provenance mismatch: {key}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[key] = {"path": str(path), "sha256": file_sha256(path), "results": payload.get("results", payload)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/f14_explainability_aware_compression_v1.toml")
    args = parser.parse_args()
    config = load_f14_config(args.config)
    root = resolve_config_path(config, config["outputs"]["directory"])
    metrics_path = root / "ablation_comparison_metrics.json"
    _refuse(metrics_path)
    thresholds = json.loads((root / "calibration/frozen_thresholds.json").read_text())
    if thresholds["reference_calibration_classification"] != "PASS" or thresholds["a1_a7_results_inspected"]:
        raise RuntimeError("A0 calibration was not frozen independently")
    protocol, feature_names, groups = load_policy_contract(config)
    actors = load_frozen_actors(config)
    states_path = root / "diagnostic/diagnostic_states.npz"
    refs_path = root / "diagnostic/reference_assignments.npz"
    states = np.load(states_path, allow_pickle=False)
    refs = np.load(refs_path, allow_pickle=False)
    x = states["observation"].astype(np.float32)
    physical = states["physical_observation"].astype(np.float32)
    phases = states["public_phase"].astype(str)
    references = refs["observation"].astype(np.float32)
    interventions = tuple(str(v) for v in config["counterfactual"]["interventions"])
    cf_section = config["counterfactual"]
    dev_cfg_path = resolve_config_path(config, cf_section["source_config"])
    if file_sha256(dev_cfg_path) != cf_section["source_config_sha256"]:
        raise RuntimeError("counterfactual protocol provenance mismatch")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 deployment environment
        import tomli as tomllib
    with dev_cfg_path.open("rb") as stream:
        dev_cfg = tomllib.load(stream)
    intervention_cfg = dev_cfg["r003"]

    summaries: dict[str, list[dict[str, object]]] = {}
    means: dict[str, np.ndarray] = {}
    draws: dict[str, np.ndarray] = {}
    effects: dict[str, np.ndarray] = {}
    factual_actions: dict[str, np.ndarray] = {}
    intended_fields = None
    max_residual: dict[str, float] = {}
    for variant, actor in actors.items():
        result = exact_group_shapley(
            actor.physical, x, references, groups, feature_names,
            observation_clip=float(config["frozen"]["observation_clip"]),
            state_batch_size=int(config["shapley"]["state_batch_size"]),
        )
        residual = float(np.max(np.abs(result.efficiency_residual)))
        if residual > float(config["shapley"]["efficiency_absolute_tolerance"]):
            raise RuntimeError(f"{variant} Shapley local accuracy failed: {residual}")
        max_residual[variant] = residual
        means[variant] = result.mean_attribution
        draws[variant] = result.draw_attribution
        summaries[variant] = summarize_group_attribution(result.mean_attribution, phases)
        factual, effect, fields = evaluate_semantic_counterfactuals(
            actor, x, physical, protocol, interventions,
            lane_low_confidence_validity=float(intervention_cfg["lane_low_confidence_validity"]),
            lane_low_confidence_min_lateral_std_m=float(intervention_cfg["lane_low_confidence_min_lateral_std_m"]),
            lane_low_confidence_min_heading_std_rad=float(intervention_cfg["lane_low_confidence_min_heading_std_rad"]),
            lane_low_confidence_min_curvature_std_inv_m=float(intervention_cfg["lane_low_confidence_min_curvature_std_inv_m"]),
        )
        sham = interventions.index("sham")
        if float(np.max(np.abs(effect[sham]))) > float(config["counterfactual"]["sham_absolute_tolerance"]):
            raise RuntimeError(f"{variant} sham intervention failed")
        factual_actions[variant] = factual; effects[variant] = effect
        intended_fields = fields if intended_fields is None else intended_fields

    np.savez_compressed(root / "diagnostic/ablation_attribution.npz",
        variants=np.asarray(tuple(actors), dtype="U2"),
        mean_attribution=np.stack([means[v] for v in actors]),
        draw_attribution=np.stack([draws[v] for v in actors]),
        factual_action=np.stack([factual_actions[v] for v in actors]),
        phase=phases, seed=states["seed"], step=states["step"],
        group_names=np.asarray(GROUP_ORDER), action_names=np.asarray(ACTION_NAMES))
    np.savez_compressed(root / "diagnostic/ablation_counterfactuals.npz",
        variants=np.asarray(tuple(actors), dtype="U2"), interventions=np.asarray(interventions),
        effects=np.stack([effects[v] for v in actors]), factual_action=np.stack([factual_actions[v] for v in actors]),
        phase=phases, seed=states["seed"], step=states["step"])

    shapley_rows: list[dict[str, object]] = []
    for variant in actors:
        for row_i in range(len(x)):
            absolute = np.abs(means[variant][row_i])
            shares = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-12)
            for action_i, action in enumerate(ACTION_NAMES):
                for group_i, group in enumerate(GROUP_ORDER):
                    shapley_rows.append({
                        "variant": variant, "actor_sha256": actors[variant].sha256,
                        "state_id": row_i, "seed": int(states["seed"][row_i]), "step": int(states["step"][row_i]),
                        "phase": phases[row_i], "action": action, "group": group,
                        "signed_shapley": float(means[variant][row_i, action_i, group_i]),
                        "absolute_shapley": float(absolute[action_i, group_i]),
                        "absolute_share": float(shares[action_i, group_i]),
                    })
    _csv(root / "ablation_shapley.csv", shapley_rows)
    summary_rows = [{"variant": variant, **row} for variant in actors for row in summaries[variant]]
    _csv(root / "ablation_group_summary.csv", summary_rows)

    cf_rows: list[dict[str, object]] = []
    for variant in actors:
        for intervention_i, intervention in enumerate(interventions):
            for row_i in range(len(x)):
                for action_i, action in enumerate(ACTION_NAMES):
                    cf_rows.append({
                        "variant": variant, "actor_sha256": actors[variant].sha256,
                        "state_id": row_i, "seed": int(states["seed"][row_i]), "step": int(states["step"][row_i]),
                        "phase": phases[row_i], "intervention": intervention, "action": action,
                        "factual_action": float(factual_actions[variant][row_i, action_i]),
                        "counterfactual_action": float(factual_actions[variant][row_i, action_i] + effects[variant][intervention_i, row_i, action_i]),
                        "effect": float(effects[variant][intervention_i, row_i, action_i]),
                    })
    _csv(root / "ablation_counterfactuals.csv", cf_rows)

    threshold_values = thresholds["thresholds"]
    comparisons = {}
    for variant in actors:
        cells = compare_group_summaries(summaries["A0"], summaries[variant], signed_deadband=float(config["shapley"]["signed_deadband"]))
        structure = semantic_structure_classification(cells, threshold_values)
        cf = counterfactual_comparison(effects["A0"], effects[variant], phases, interventions,
                                        direction_deadband=float(config["counterfactual"]["direction_deadband"]))
        cf_classification = counterfactual_preservation_classification(cf, config["counterfactual"])
        comparisons[variant] = {
            "actor_sha256": actors[variant].sha256, "precision": actors[variant].precision,
            "architecture": actors[variant].architecture, "maximum_local_accuracy_residual": max_residual[variant],
            "semantic_attribution": structure,
            "counterfactual_functional_sensitivity": cf_classification,
        }
    historical = _historical(config)
    metrics = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["_sha256"], "states_sha256": file_sha256(states_path),
        "references_sha256": file_sha256(refs_path), "thresholds_sha256": file_sha256(root / "calibration/frozen_thresholds.json"),
        "classification_axes": comparisons, "historical": historical,
        "intervention_field_masks": {k:list(v) for k,v in (intended_fields or {}).items()},
        "historical_results_reused_without_rerun": True,
    }
    _json(root / "integrated_historical_metrics.json", historical)
    _json(metrics_path, metrics)
    print(json.dumps({"classification":"PASS", "variants": list(actors), "axes": {v:{"semantic":comparisons[v]["semantic_attribution"]["classification"],"counterfactual":comparisons[v]["counterfactual_functional_sensitivity"]["classification"]} for v in actors}}, indent=2))


if __name__ == "__main__":
    main()
