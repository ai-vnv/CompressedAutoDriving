#!/usr/bin/env python3
"""F14-A: provenance audit and immutable public-state/reference preparation."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duckie_pomdp.explain.compression_diagnostics import (
    assign_complete_references,
    file_sha256,
    load_f14_config,
    load_frozen_actors,
    load_policy_contract,
    normalized_to_physical,
    resolve_config_path,
    select_stratified_public_rows,
    verify_frozen_file,
)
from duckie_pomdp.explain.group_shapley import GROUP_ORDER, coalition_schema
from duckie_pomdp.optimization.actor_compression import extract_original_actor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f14_explainability_aware_compression_v1.toml"
HISTORICAL_FILES = (
    "refine-logs/EXPERIMENT_PLAN.md",
    "docs/F11_FINAL_EXPLANATION_SUMMARY.md",
    "docs/F11_R004_REPORT_FOR_REVIEW.md",
    "docs/F12_COMPRESSION_PROTOCOL.md",
    "docs/F12_COMPRESSION_RESULTS.md",
    "docs/F12_COMPRESSION_ABLATION.md",
    "docs/F13_EXPLAIN_AGAIN_PROTOCOL.md",
    "docs/F13_EXPLANATION_COMPARISON.md",
    "docs/F13_FAILURE_MODE_REPORT.md",
    "docs/F13_FINAL_REPORT.md",
)


def artifact_root(config: dict[str, Any]) -> Path:
    return resolve_config_path(config, str(config["outputs"]["directory"]))


def audit(config_path: Path) -> dict[str, Any]:
    config = load_f14_config(config_path)
    protocol, feature_names, group_indexes = load_policy_contract(config)
    for path_key, hash_key in (
        ("original_checkpoint", "original_checkpoint_sha256"),
        ("f12_config", "f12_config_sha256"),
        ("f13_config", "f13_config_sha256"),
        ("model_selection", "model_selection_sha256"),
    ):
        verify_frozen_file(config, path_key, hash_key)
    actors = load_frozen_actors(config)
    dataset_path = resolve_config_path(config, str(config["development"]["source_dataset"]))
    dataset_hash = file_sha256(dataset_path)
    if dataset_hash != str(config["development"]["source_dataset_sha256"]):
        raise RuntimeError("development public-state dataset hash mismatch")
    manifest_path = resolve_config_path(config, str(config["development"]["source_manifest"]))
    if file_sha256(manifest_path) != str(config["development"]["source_manifest_sha256"]):
        raise RuntimeError("development public-state manifest hash mismatch")
    data = dict(np.load(dataset_path, allow_pickle=False))
    if tuple(str(value) for value in data["feature_names"]) != feature_names:
        raise RuntimeError("development feature ordering differs from frozen 29D contract")
    if data["observation"].shape != (17600, 29) or not np.isfinite(data["observation"]).all():
        raise RuntimeError("development public-state matrix is invalid")
    prohibited = {"evaluation_gt", "privileged", "world_pose", "bbox", "iou"}
    if any(str(key).lower() in prohibited for key in data):
        raise RuntimeError("development dataset contains privileged schema")

    checkpoint = verify_frozen_file(config, "original_checkpoint", "original_checkpoint_sha256")
    original, _, _ = extract_original_actor(
        checkpoint, expected_sha256=str(config["frozen"]["original_checkpoint_sha256"])
    )
    probe = np.asarray(data["observation"][:1024], dtype=np.float32)
    with torch.inference_mode():
        original_physical = normalized_to_physical(original(torch.from_numpy(probe)).numpy())
    a0_physical = actors["A0"].physical(probe)
    a0_replay_error = float(np.max(np.abs(original_physical - a0_physical)))
    if a0_replay_error > 2.0e-6:
        raise RuntimeError(f"registered A0 does not replay Original actor: {a0_replay_error}")
    repeatability = {}
    for variant, actor in actors.items():
        first = actor.physical(probe[:64])
        second = actor.physical(probe[:64])
        error = float(np.max(np.abs(first - second)))
        if error > 1.0e-7:
            raise RuntimeError(f"actor repeatability failed: {variant} {error}")
        repeatability[variant] = error

    result = {
        "schema_version": 1,
        "classification": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": str(config["_sha256"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "actor_registry_sha256": str(config["frozen"]["actor_registry_sha256"]),
        "actors": {
            key: {
                "name": actor.name,
                "path": str(actor.path),
                "sha256": actor.sha256,
                "architecture": list(actor.architecture),
                "precision": actor.precision,
            }
            for key, actor in actors.items()
        },
        "original_checkpoint_sha256": str(config["frozen"]["original_checkpoint_sha256"]),
        "a0_original_physical_replay_max_abs": a0_replay_error,
        "actor_repeatability_max_abs": repeatability,
        "feature_names": list(feature_names),
        "group_indices_zero_based": {key: list(value) for key, value in group_indexes.items()},
        "group_partition_exact": True,
        "development_dataset": str(dataset_path),
        "development_dataset_sha256": dataset_hash,
        "development_rows": int(len(data["observation"])),
        "development_contains_privileged_truth": False,
        "source_pipeline": "RGB->MobileNet/YOLO->belief->public29D->actor",
        "historical_status": {
            "F11_R002": "LIMITED", "F11_R002b": "PASS", "F11_R003": "PASS",
            "F11_R004": "PASS", "F11_R006": "FAILED", "F11_R007": "BLOCKED",
            "F12": "PASS_C4_ONLY", "F13": "LIMITED",
            "F13_A7_gradient_attribution": "UNRESOLVED_BLOCKED",
        },
    }
    return result


def write_audit(config_path: Path) -> None:
    config = load_f14_config(config_path)
    root = artifact_root(config)
    target = root / "integrity/actor_registry_verified.json"
    protocol_manifest = root / "protocol_manifest.json"
    coalition_target = root / "coalition_schema.json"
    historical_target = root / "integrity/historical_integrity_manifest.json"
    _refuse_existing(target, protocol_manifest, coalition_target, historical_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = audit(config_path)
    _write_json(target, result)
    protocol_path = ROOT / "docs/F14_PROTOCOL_20260815_140652.md"
    _write_json(
        protocol_manifest,
        {
            "schema_version": 1,
            "classification": "FROZEN_BEFORE_A1_A7",
            "config_path": str(config_path.resolve()),
            "config_sha256": str(config["_sha256"]),
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "actor_registry_verified_sha256": file_sha256(target),
            "primary_method": "phase-conditioned exact group Shapley",
            "secondary_method": "semantic counterfactual policy-input interventions",
            "a1_a7_results_inspected_at_freeze": False,
        },
    )
    coalitions = coalition_schema()
    _write_json(
        coalition_target,
        {
            "schema_version": 1,
            "groups": list(GROUP_ORDER),
            "coalition_count": 64,
            "coalitions": [
                {"mask": mask, "factual_groups": [GROUP_ORDER[i] for i in range(6) if row[i]]}
                for mask, row in enumerate(coalitions)
            ],
            "complement_rule": "all absent groups come from one complete reference row",
        },
    )
    _write_json(
        historical_target,
        {
            "schema_version": 1,
            "role": "pre-F14 immutable historical file witness",
            "files": {name: file_sha256(ROOT / name) for name in HISTORICAL_FILES},
        },
    )
    print(json.dumps(result, indent=2))


def prepare(config_path: Path) -> None:
    config = load_f14_config(config_path)
    root = artifact_root(config)
    audit_path = root / "integrity/actor_registry_verified.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["classification"] != "PASS":
        raise RuntimeError("F14 actor/contract audit must PASS before state preparation")
    diagnostic_target = root / "diagnostic/diagnostic_states.npz"
    reference_target = root / "diagnostic/reference_assignments.npz"
    state_manifest_target = root / "diagnostic_state_manifest.json"
    reference_manifest_target = root / "reference_assignment_manifest.json"
    _refuse_existing(
        diagnostic_target, reference_target, state_manifest_target, reference_manifest_target
    )
    source = resolve_config_path(config, str(config["development"]["source_dataset"]))
    data = dict(np.load(source, allow_pickle=False))
    phases = tuple(str(value) for value in config["development"]["phases"])
    indexes = select_stratified_public_rows(
        data["public_phase"], data["seed"], data["step"],
        phase_order=phases,
        states_per_phase=int(config["development"]["states_per_phase"]),
    )
    references, reference_indexes = assign_complete_references(
        data["observation"], data["public_phase"], data["seed"], indexes,
        draw_seeds=tuple(int(value) for value in config["development"]["draw_seeds"]),
        references_per_draw=int(config["development"]["references_per_draw"]),
    )
    diagnostic_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostic_target,
        source_index=indexes,
        seed=np.asarray(data["seed"][indexes], dtype=np.int64),
        episode=np.asarray(data["episode"][indexes], dtype=np.int64),
        step=np.asarray(data["step"][indexes], dtype=np.int64),
        public_phase=np.asarray(data["public_phase"][indexes]),
        observation=np.asarray(data["observation"][indexes], dtype=np.float32),
        physical_observation=np.asarray(data["physical_observation"][indexes], dtype=np.float32),
        feature_names=np.asarray(data["feature_names"]),
    )
    np.savez_compressed(
        reference_target,
        source_index=reference_indexes,
        source_seed=np.asarray(data["seed"][reference_indexes], dtype=np.int64),
        source_step=np.asarray(data["step"][reference_indexes], dtype=np.int64),
        public_phase=np.asarray(data["public_phase"][reference_indexes]),
        observation=references,
        draw_seeds=np.asarray(config["development"]["draw_seeds"], dtype=np.int64),
        feature_names=np.asarray(data["feature_names"]),
    )
    counts = {
        phase: int(np.sum(data["public_phase"][indexes] == phase)) for phase in phases
    }
    _write_json(
        state_manifest_target,
        {
            "schema_version": 1,
            "selection_rule": "seed-balanced evenly-spaced public rows within each phase",
            "uses_privileged_data": False,
            "source_dataset": str(source),
            "source_dataset_sha256": file_sha256(source),
            "diagnostic_states": str(diagnostic_target),
            "diagnostic_states_sha256": file_sha256(diagnostic_target),
            "rows": int(len(indexes)),
            "phase_counts": counts,
            "row_ids": [
                {
                    "source_index": int(index), "seed": int(data["seed"][index]),
                    "episode": int(data["episode"][index]), "step": int(data["step"][index]),
                    "phase": str(data["public_phase"][index]),
                }
                for index in indexes
            ],
        },
    )
    assignments = []
    for draw in range(reference_indexes.shape[0]):
        for reference in range(reference_indexes.shape[1]):
            for state in range(reference_indexes.shape[2]):
                source_index = int(reference_indexes[draw, reference, state])
                assignments.append(
                    {
                        "draw": draw, "reference": reference, "factual_row": state,
                        "reference_source_index": source_index,
                        "reference_seed": int(data["seed"][source_index]),
                        "reference_step": int(data["step"][source_index]),
                        "phase": str(data["public_phase"][source_index]),
                    }
                )
    _write_json(
        reference_manifest_target,
        {
            "schema_version": 1,
            "rule": "same-phase, cross-seed, four distinct reference trajectories per draw",
            "draws": int(reference_indexes.shape[0]),
            "references_per_draw": int(reference_indexes.shape[1]),
            "factual_rows": int(reference_indexes.shape[2]),
            "complete_reference_row_only": True,
            "reference_assignments": str(reference_target),
            "reference_assignments_sha256": file_sha256(reference_target),
            "assignments": assignments,
        },
    )
    print(json.dumps({"classification": "PASS", "rows": len(indexes), "phase_counts": counts}, indent=2))


def _refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite immutable F14 outputs: {existing}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "audit":
        write_audit(args.config.resolve())
    else:
        prepare(args.config.resolve())


if __name__ == "__main__":
    main()

