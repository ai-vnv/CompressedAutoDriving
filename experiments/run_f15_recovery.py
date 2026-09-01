#!/usr/bin/env python3
"""Controlled F15 recovery tree, run only after localization is frozen."""

from __future__ import annotations

import argparse
import json
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.development_protocol import public_phase
from duckie_pomdp.optimization.actor_compression import (
    ActorSpec,
    convert_qat,
    load_dense_actor,
    physical_actions,
    prepare_ptq,
    prepare_qat,
    save_dense_actor,
    save_quantized_actor,
)
from duckie_pomdp.optimization.compression_metrics import action_fidelity, actor_physical_predictions, benchmark_actor
from duckie_pomdp.optimization.cross_curriculum_recovery import (
    CURRICULA,
    distill_multicurriculum_actor,
    fidelity_pass,
    file_sha256,
    verify_registry,
)

from run_f15_cross_curriculum_recovery import (
    ActorPolicy,
    append_csv,
    artifact_root,
    evaluate_registry,
    frozen_paths,
    load_actor,
    load_config,
    phase_thresholds,
    provenance,
    read_json,
    read_csv,
    trace_path,
    verify_protocol,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"


def require_localization(config: Mapping[str, Any], config_path: Path) -> Path:
    path = artifact_root(config, config_path) / "localization/failure_localization_decision.json"
    if not path.exists() or read_json(path).get("classification") != "FROZEN":
        raise RuntimeError("F15 recovery requires the frozen localization decision")
    return path


def collect_recovery_dataset(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    localization = require_localization(config, config_path)
    paths = frozen_paths(config, config_path)
    root = artifact_root(config, config_path)
    target = root / "recovery/datasets/multicurriculum_public_states.npz"
    manifest_path = root / "recovery/datasets/dataset_manifest.json"
    if target.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite F15 recovery dataset")
    matrix_registry = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    teacher = load_actor(matrix_registry["A0"])
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    thresholds = phase_thresholds(paths["f12_config"])
    rows: dict[str, list[Any]] = {
        "curriculum": [], "seed": [], "step": [], "public_phase": [],
        "observation": [], "physical_observation": [], "teacher_physical_action": [],
    }
    seeds = tuple(int(value) for value in config["seeds"]["recovery_dataset"])
    for curriculum in CURRICULA:
        environment = PPOCurriculumEnvironment(
            paths["policy_config"], stage=curriculum,
            split=f"f15_recovery_dataset_{curriculum}", seeds=seeds,
        )
        policy = ActorPolicy("Original Policy", teacher)
        try:
            for seed in seeds:
                observation, info = environment.reset(seed=seed)
                policy.reset(seed)
                for step in range(protocol.stage(curriculum).episode_horizon_steps):
                    physical = np.asarray([info["policy"][name] for name in protocol.observation_order], dtype=np.float32)
                    phase = public_phase(physical, protocol.observation_order, thresholds)
                    if phase == "combined_pedestrian_stop":
                        phase = "stop_required"
                    with torch.inference_mode():
                        mean = teacher(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                        teacher_action = physical_actions(mean).cpu().numpy().astype(np.float32)
                    environment_action = np.clip(mean.cpu().numpy(), -1.0, 1.0).astype(np.float32)
                    next_observation, _, terminated, truncated, next_info = environment.step(environment_action)
                    rows["curriculum"].append(curriculum)
                    rows["seed"].append(seed)
                    rows["step"].append(step)
                    rows["public_phase"].append(phase)
                    rows["observation"].append(np.asarray(observation, dtype=np.float32))
                    rows["physical_observation"].append(physical)
                    rows["teacher_physical_action"].append(teacher_action)
                    observation, info = next_observation, next_info
                    if terminated or truncated:
                        break
        finally:
            environment.close()
    arrays = {
        "curriculum": np.asarray(rows["curriculum"], dtype="U4"),
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "step": np.asarray(rows["step"], dtype=np.int64),
        "public_phase": np.asarray(rows["public_phase"], dtype="U32"),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "physical_observation": np.asarray(rows["physical_observation"], dtype=np.float32),
        "teacher_physical_action": np.asarray(rows["teacher_physical_action"], dtype=np.float32),
        "feature_names": np.asarray(protocol.observation_order, dtype="U64"),
    }
    if arrays["observation"].shape[1] != 29 or not np.isfinite(arrays["observation"]).all():
        raise RuntimeError("F15 recovery dataset violates public 29D contract")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target)
    curriculum_counts = {curriculum: int(np.sum(arrays["curriculum"] == curriculum)) for curriculum in CURRICULA}
    phase_counts = {
        curriculum: {
            phase: int(np.sum((arrays["curriculum"] == curriculum) & (arrays["public_phase"] == phase)))
            for phase in sorted(set(arrays["public_phase"][arrays["curriculum"] == curriculum]))
        }
        for curriculum in CURRICULA
    }
    manifest = {
        **provenance(config, config_path),
        "dataset": str(target), "dataset_sha256": file_sha256(target), "rows": len(arrays["observation"]),
        "seeds": list(seeds), "curriculum_counts": curriculum_counts, "phase_counts": phase_counts,
        "teacher_sha256": matrix_registry["A0"]["sha256"], "contains_privileged_truth": False,
        "sampling_rule": "equal curriculum mass then equal supported phase mass during training",
        "localization_decision_sha256": file_sha256(localization),
    }
    write_json(manifest_path, manifest)
    root_manifest = root / "dataset_manifest.json"
    if root_manifest.exists():
        raise RuntimeError("refusing to overwrite root F15 dataset manifest")
    write_json(root_manifest, {
        **manifest,
        "canonical_manifest": str(manifest_path),
        "canonical_manifest_sha256": file_sha256(manifest_path),
    })
    return manifest


def load_recovery_dataset(config: Mapping[str, Any], config_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = artifact_root(config, config_path)
    target = root / "recovery/datasets/multicurriculum_public_states.npz"
    manifest = read_json(root / "recovery/datasets/dataset_manifest.json")
    if file_sha256(target) != manifest["dataset_sha256"] or manifest["contains_privileged_truth"]:
        raise RuntimeError("F15 recovery dataset provenance/privilege mismatch")
    with np.load(target, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}, manifest


def train_fp32(config_path: Path, width: int) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    require_localization(config, config_path)
    data, manifest = load_recovery_dataset(config, config_path)
    paths = frozen_paths(config, config_path)
    pruning = verify_registry(
        paths["pruning_registry"], expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"], collection_key="candidates"
    )
    if width not in tuple(int(value) for value in config["pruning"]["recovery_width_order"]):
        raise ValueError("width is outside frozen F15 recovery order")
    source, payload = load_dense_actor(pruning[f"P{width}"]["model_path"])
    root = artifact_root(config, config_path)
    target = root / f"recovery/fp32/w{width}/actor_multicurriculum_kd_fp32.pt"
    history_path = target.with_name("training_history.json")
    if target.exists() or history_path.exists():
        raise RuntimeError(f"refusing to overwrite F15 recovered width {width}")
    distill = config["distillation"]
    history = distill_multicurriculum_actor(
        source, data["observation"], data["teacher_physical_action"], data["curriculum"], data["public_phase"],
        epochs=int(distill["epochs"]), batch_size=int(distill["batch_size"]),
        learning_rate=float(distill["learning_rate"]), weight_decay=float(distill["weight_decay"]),
        seed=int(distill["seed"]) + width, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    save_dense_actor(
        target, source, log_std=payload["log_std"],
        metadata={
            "experiment": "F15", "width": width, "source_pruned_actor": pruning[f"P{width}"]["model_path"],
            "source_pruned_sha256": pruning[f"P{width}"]["sha256"], "dataset_sha256": manifest["dataset_sha256"],
            "teacher_sha256": manifest["teacher_sha256"], "config_sha256": config["_sha256"],
            "uses_privileged_truth": False, "uses_reward": False, "uses_critic": False,
        },
    )
    output = {
        **provenance(config, config_path), "width": width, "model_path": str(target), "model_sha256": file_sha256(target),
        "source_pruned_sha256": pruning[f"P{width}"]["sha256"], "dataset_sha256": manifest["dataset_sha256"],
        "history": history,
    }
    write_json(history_path, output)
    return output


def _entry(model_id: str, path: Path, width: int, int8: bool, name: str) -> dict[str, Any]:
    spec = ActorSpec(hidden_sizes=(width, width))
    parameter_count = 29 * width + width + width * width + width + width * 2 + 2
    return {"variant": model_id, "name": name, "model_path": str(path.resolve()), "sha256": file_sha256(path), "hidden_sizes": [width, width], "parameter_count": parameter_count, "int8": int8}


def candidate_fidelity(config_path: Path, split: str, entry: Mapping[str, Any], seeds: list[int]) -> dict[str, Any]:
    config = load_config(config_path)
    paths = frozen_paths(config, config_path)
    registry = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    root = artifact_root(config, config_path)
    episode_rows = read_csv(root / "recovery" / f"{split}_episodes.csv")
    results = {}
    passes = []
    for curriculum in CURRICULA:
        observations = []
        for seed in seeds:
            row = next(
                row for row in episode_rows
                if row["model_id"] == "A0"
                and row["curriculum"] == curriculum
                and int(row["seed"]) == int(seed)
            )
            with np.load(Path(row["trace_path"]), allow_pickle=False) as archive:
                observations.append(np.asarray(archive["public_normalized_29d"], dtype=np.float32))
        matrix = np.concatenate(observations)
        original = actor_physical_predictions(load_actor(registry["A0"]), matrix)
        candidate = actor_physical_predictions(load_actor(entry), matrix)
        metrics = action_fidelity(original, candidate, omega_deadband=float(config["evaluation"]["omega_sign_deadband_rad_s"]))
        passed, checks = fidelity_pass(metrics, config["fidelity"])
        results[curriculum] = {"metrics": metrics, "checks": checks, "pass": passed}
        passes.append(passed)
    return {"all_curricula_pass": all(passes), "results": results}


def evaluate_selection_candidate(
    config_path: Path,
    *,
    split: str,
    entry: Mapping[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluate one candidate against one shared, hash-identical A0 baseline."""
    config = load_config(config_path)
    paths = frozen_paths(config, config_path)
    registry = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    root = artifact_root(config, config_path)
    baseline_csv = root / "recovery/selection_baseline_episodes.csv"
    if not baseline_csv.exists():
        evaluate_registry(
            config_path,
            split="selection_baseline",
            registry={"A0": registry["A0"]},
            seeds=seeds,
            episode_csv_override=baseline_csv,
            build_results=False,
        )
    target_csv = root / "recovery" / f"{split}_episodes.csv"
    if not target_csv.exists():
        for row in read_csv(baseline_csv):
            copied = dict(row)
            copied["evidence_reuse"] = "byte_identical_shared_selection_baseline"
            append_csv(target_csv, copied)
    behavior = evaluate_registry(
        config_path,
        split=split,
        registry={"A0": registry["A0"], entry["variant"]: entry},
        seeds=seeds,
        episode_csv_override=target_csv,
    )
    reuse_manifest = root / "recovery" / f"{split}_baseline_reuse.json"
    if not reuse_manifest.exists():
        write_json(reuse_manifest, {
            **provenance(config, config_path),
            "baseline_csv": str(baseline_csv),
            "baseline_sha256": file_sha256(baseline_csv),
            "target_csv": str(target_csv),
            "target_sha256": file_sha256(target_csv),
            "actor_sha256_equal": True,
            "not_independent_replicates": True,
        })
    return behavior


def evaluate_fp32(config_path: Path, width: int) -> dict[str, Any]:
    config = load_config(config_path)
    require_localization(config, config_path)
    paths = frozen_paths(config, config_path)
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    target = artifact_root(config, config_path) / f"recovery/fp32/w{width}/actor_multicurriculum_kd_fp32.pt"
    entry = _entry(f"R{width}", target, width, False, f"Recovered {width}x{width} + Multi-Curriculum KD")
    seeds = [int(value) for value in config["seeds"]["recovery_selection"]]
    split = f"selection_fp32_w{width}"
    behavior = evaluate_selection_candidate(config_path, split=split, entry=entry, seeds=seeds)
    fidelity = candidate_fidelity(config_path, split, entry, seeds)
    all_behavior = all(behavior["decisions"][entry["variant"]][curriculum]["status"] == "PASS" for curriculum in CURRICULA)
    output = {
        **provenance(config, config_path), "width": width, "entry": entry, "behavior": behavior,
        "fidelity": fidelity, "all_curricula_behavior_pass": all_behavior,
        "eligible": all_behavior and fidelity["all_curricula_pass"],
    }
    write_json(artifact_root(config, config_path) / f"recovery/fp32/w{width}/selection_result.json", output)
    return output


def balanced_calibration(data: Mapping[str, np.ndarray], maximum: int) -> np.ndarray:
    selected: list[int] = []
    for curriculum in CURRICULA:
        curriculum_mask = data["curriculum"] == curriculum
        for phase in sorted(set(data["public_phase"][curriculum_mask])):
            indexes = np.flatnonzero(curriculum_mask & (data["public_phase"] == phase))[:maximum]
            selected.extend(indexes.tolist())
    return np.asarray(data["observation"][np.asarray(selected, dtype=np.int64)], dtype=np.float32)


def quantize_recovered(config_path: Path, width: int, method: str) -> dict[str, Any]:
    config = load_config(config_path)
    require_localization(config, config_path)
    root = artifact_root(config, config_path)
    fp32_result = read_json(root / f"recovery/fp32/w{width}/selection_result.json")
    if not fp32_result["eligible"]:
        raise RuntimeError("quantization requires an eligible FP32 recovered actor")
    fp32_path = root / f"recovery/fp32/w{width}/actor_multicurriculum_kd_fp32.pt"
    actor, _ = load_dense_actor(fp32_path)
    data, manifest = load_recovery_dataset(config, config_path)
    quant = config["quantization"]
    calibration = balanced_calibration(data, int(quant["calibration_max_rows_per_curriculum_phase"]))
    directory = "ptq" if method == "ptq" else "qat"
    target = root / f"recovery/{directory}/w{width}/actor_int8.pt"
    metadata_path = target.with_name("conversion.json")
    if target.exists() or metadata_path.exists():
        raise RuntimeError(f"refusing to overwrite F15 recovered {method.upper()} actor")
    if method == "ptq":
        converted = prepare_ptq(actor, calibration, backend=quant["backend"])
        history = None
    elif method == "qat":
        qat = prepare_qat(actor, backend=quant["backend"])
        history = distill_multicurriculum_actor(
            qat, data["observation"], data["teacher_physical_action"], data["curriculum"], data["public_phase"],
            epochs=int(quant["qat_epochs"]), batch_size=int(config["distillation"]["batch_size"]),
            learning_rate=float(quant["qat_learning_rate"]), weight_decay=float(config["distillation"]["weight_decay"]),
            seed=int(quant["qat_seed"]) + width, device="cuda" if torch.cuda.is_available() else "cpu",
        )
        converted = convert_qat(qat, backend=quant["backend"])
    else:
        raise ValueError("method must be ptq or qat")
    save_quantized_actor(target, converted)
    output = {
        **provenance(config, config_path), "method": method, "width": width, "model_path": str(target),
        "model_sha256": file_sha256(target), "source_fp32_sha256": file_sha256(fp32_path),
        "dataset_sha256": manifest["dataset_sha256"], "calibration_rows": len(calibration), "training_history": history,
    }
    write_json(metadata_path, output)
    return output


def evaluate_int8(config_path: Path, width: int, method: str) -> dict[str, Any]:
    config = load_config(config_path)
    paths = frozen_paths(config, config_path)
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    root = artifact_root(config, config_path)
    target = root / f"recovery/{method}/w{width}/actor_int8.pt"
    model_id = f"R{width}_{method.upper()}"
    entry = _entry(model_id, target, width, True, f"Recovered INT8 {width}x{width} ({method.upper()})")
    seeds = [int(value) for value in config["seeds"]["recovery_selection"]]
    split = f"selection_{method}_w{width}"
    behavior = evaluate_selection_candidate(config_path, split=split, entry=entry, seeds=seeds)
    fidelity = candidate_fidelity(config_path, split, entry, seeds)
    all_behavior = all(behavior["decisions"][model_id][curriculum]["status"] == "PASS" for curriculum in CURRICULA)
    output = {**provenance(config, config_path), "entry": entry, "behavior": behavior, "fidelity": fidelity, "all_curricula_behavior_pass": all_behavior, "eligible": all_behavior and fidelity["all_curricula_pass"]}
    write_json(root / f"recovery/{method}/w{width}/selection_result.json", output)
    return output


def freeze_candidate(config_path: Path, width: int, method: str) -> dict[str, Any]:
    config = load_config(config_path)
    if method not in {"ptq", "qat"}:
        raise RuntimeError("F15 final candidate must be a deployable INT8 PTQ or QAT actor")
    root = artifact_root(config, config_path)
    target = root / "final/final_candidate.json"
    if target.exists():
        raise RuntimeError("refusing to replace frozen F15 final candidate")
    result_path = root / f"recovery/{method}/w{width}/selection_result.json"
    selection = read_json(result_path)
    if not selection["eligible"]:
        raise RuntimeError("final candidate must pass all frozen selection gates")
    entry = selection["entry"]
    eligible_int8 = []
    for eligible_path in sorted((root / "recovery").glob("**/selection_result.json")):
        eligible_payload = read_json(eligible_path)
        eligible_entry = eligible_payload["entry"]
        if bool(eligible_entry["int8"]) and bool(eligible_payload["eligible"]):
            eligible_int8.append((int(eligible_entry["hidden_sizes"][0]), eligible_path, eligible_entry))
    minimum_eligible_width = min(item[0] for item in eligible_int8)
    if width != minimum_eligible_width:
        raise RuntimeError(
            f"candidate width {width} is not the smallest eligible INT8 width {minimum_eligible_width}"
        )
    ptq_result_path = root / f"recovery/ptq/w{width}/selection_result.json"
    if method == "qat" and ptq_result_path.exists() and read_json(ptq_result_path)["eligible"]:
        raise RuntimeError("QAT cannot replace an already eligible PTQ actor at the same minimum width")

    dataset_manifest_path = root / "recovery/datasets/dataset_manifest.json"
    fp32_actor_path = root / f"recovery/fp32/w{width}/actor_multicurriculum_kd_fp32.pt"
    fp32_history_path = root / f"recovery/fp32/w{width}/training_history.json"
    conversion_path = root / f"recovery/{method}/w{width}/conversion.json"
    provenance_paths = (dataset_manifest_path, fp32_actor_path, fp32_history_path, conversion_path)
    missing_provenance = [str(path) for path in provenance_paths if not path.exists()]
    if missing_provenance:
        raise RuntimeError(f"candidate recovery provenance is incomplete: {missing_provenance}")
    recovery_decision_path = root / "recovery/recovery_decision.json"
    if recovery_decision_path.exists():
        raise RuntimeError("refusing to overwrite frozen F15 recovery decision")
    experiments = []
    for path in sorted((root / "recovery").glob("**/selection_result.json")):
        payload = read_json(path)
        candidate_entry = payload["entry"]
        experiments.append({
            "result_path": str(path), "result_sha256": file_sha256(path),
            "model_id": candidate_entry["variant"], "model_name": candidate_entry["name"],
            "width": int(candidate_entry["hidden_sizes"][0]), "int8": bool(candidate_entry["int8"]),
            "behavior_all_curricula_pass": bool(payload["all_curricula_behavior_pass"]),
            "fidelity_all_curricula_pass": bool(payload["fidelity"]["all_curricula_pass"]),
            "eligible": bool(payload["eligible"]),
        })
    recovery_decision = {
        **provenance(config, config_path), "classification": "FROZEN", "selected_result": str(result_path),
        "selected_result_sha256": file_sha256(result_path), "selected_width": width, "selected_method": method,
        "multicurriculum_kd_recovered_64": any(row["width"] == 64 and not row["int8"] and row["eligible"] for row in experiments),
        "larger_width_required": width > 64,
        "ptq_preserved_recovery": method == "ptq",
        "qat_required": method == "qat",
        "progressive_pruning_required": False,
        "experiments": experiments,
    }
    experiments_csv = root / "recovery/recovery_experiments.csv"
    if experiments_csv.exists():
        raise RuntimeError("refusing to overwrite F15 recovery experiment table")
    for row in experiments:
        append_csv(experiments_csv, row)
    recovery_decision["experiments_csv"] = str(experiments_csv)
    recovery_decision["experiments_csv_sha256"] = file_sha256(experiments_csv)
    write_json(recovery_decision_path, recovery_decision)
    output = {
        **provenance(config, config_path), "classification": "FROZEN", "selected_model": entry,
        "architecture": [29, width, width, 2], "precision": "INT8" if method in {"ptq", "qat"} else "FP32",
        "recovery_method": method, "selection_result": str(result_path), "selection_result_sha256": file_sha256(result_path),
        "selection_rationale": "smallest INT8 width passing all frozen C0-C4 behavior, fidelity, and safety gates",
        "pruning_width": width,
        "distillation_dataset_manifest": str(dataset_manifest_path),
        "distillation_dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "recovered_fp32_actor": str(fp32_actor_path),
        "recovered_fp32_actor_sha256": file_sha256(fp32_actor_path),
        "recovery_training_history": str(fp32_history_path),
        "recovery_training_history_sha256": file_sha256(fp32_history_path),
        "quantization_conversion_manifest": str(conversion_path),
        "quantization_conversion_manifest_sha256": file_sha256(conversion_path),
        "failure_localization_decision_sha256": file_sha256(root / "localization/failure_localization_decision.json"),
        "recovery_decision_sha256": file_sha256(recovery_decision_path),
        "final_holdout_unopened_at_freeze": not (root / "final/final_holdout.json").exists(),
    }
    write_json(target, output)
    return output


def final_holdout(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    candidate_path = root / "final/final_candidate.json"
    target = root / "final/final_holdout.json"
    claim_path = root / "final/final_holdout_claim.json"
    if target.exists() or claim_path.exists():
        raise RuntimeError("F15 final holdout is once-only and already accessed/claimed")
    candidate = read_json(candidate_path)
    if candidate["classification"] != "FROZEN" or not candidate["final_holdout_unopened_at_freeze"]:
        raise RuntimeError("invalid F15 final candidate freeze")
    entry = candidate["selected_model"]
    if file_sha256(entry["model_path"]) != entry["sha256"]:
        raise RuntimeError("final candidate checkpoint changed after freeze")
    seeds = [int(value) for value in config["seeds"]["final_holdout"]]
    claim = {
        **provenance(config, config_path), "claim": "once-only final access begins after this claim",
        "candidate_sha256": entry["sha256"], "candidate_manifest_sha256": file_sha256(candidate_path), "seeds": seeds,
    }
    write_json(claim_path, claim)
    paths = frozen_paths(config, config_path)
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    behavior = evaluate_registry(config_path, split="final_holdout", registry={"A0": matrix["A0"], entry["variant"]: entry}, seeds=seeds)
    fidelity = candidate_fidelity(config_path, "final_holdout", entry, seeds)
    all_behavior = all(behavior["decisions"][entry["variant"]][curriculum]["status"] == "PASS" for curriculum in CURRICULA)
    classification = "PASS" if all_behavior and fidelity["all_curricula_pass"] else "FAIL"
    output = {**provenance(config, config_path), "classification": classification, "claim_sha256": file_sha256(claim_path), "candidate": entry, "behavior": behavior, "fidelity": fidelity}
    write_json(target, output)
    return output


def efficiency(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = artifact_root(config, config_path)
    final = read_json(root / "final/final_holdout.json")
    candidate = final["candidate"]
    paths = frozen_paths(config, config_path)
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    benchmark = config["benchmark"]
    entries = (matrix["A0"], candidate)
    models = []
    for entry in entries:
        width = int(entry["hidden_sizes"][0])
        actor = load_actor(entry)
        metrics = benchmark_actor(
            actor, ActorSpec(hidden_sizes=(width, width)), entry["model_path"],
            warmup=int(benchmark["warmup_iterations"]), iterations=int(benchmark["timed_iterations"]),
            repeats=int(benchmark["repeats"]), threads=int(benchmark["threads"]), int8=bool(entry["int8"]),
        )
        decisions = final["behavior"]["decisions"][entry["variant"]]
        curricula_passed = 5 if entry["variant"] == "A0" else sum(decisions[c]["status"] == "PASS" for c in CURRICULA)
        models.append({
            "model_id": entry["variant"], "name": "Original Policy" if entry["variant"] == "A0" else entry["name"],
            "sha256": entry["sha256"], "selected": entry["variant"] != "A0",
            "classification": "REFERENCE" if entry["variant"] == "A0" else final["classification"],
            "curricula_passed": curricula_passed, "parameter_count": metrics["dense_parameter_count"], **metrics,
        })
    original, compressed = models
    output = {
        **provenance(config, config_path), "models": models,
        "parameter_reduction_fraction": 1.0 - compressed["parameter_count"] / original["parameter_count"],
        "file_size_reduction_fraction": 1.0 - compressed["actor_checkpoint_size_bytes"] / original["actor_checkpoint_size_bytes"],
        "actor_only_cpu_speedup": original["batch1_latency_us_median"] / compressed["batch1_latency_us_median"],
        "claim_boundary": "actor-only CPU benchmark; perception is unchanged",
    }
    write_json(root / "final/efficiency_summary.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect-dataset", "train-fp32", "evaluate-fp32", "quantize", "evaluate-int8", "freeze-candidate", "final-holdout", "efficiency"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--width", type=int)
    parser.add_argument("--method", choices=("fp32", "ptq", "qat"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.command == "collect-dataset":
        result = collect_recovery_dataset(config_path)
    elif args.command == "train-fp32":
        result = train_fp32(config_path, int(args.width))
    elif args.command == "evaluate-fp32":
        result = evaluate_fp32(config_path, int(args.width))
    elif args.command == "quantize":
        result = quantize_recovered(config_path, int(args.width), str(args.method))
    elif args.command == "evaluate-int8":
        result = evaluate_int8(config_path, int(args.width), str(args.method))
    elif args.command == "freeze-candidate":
        result = freeze_candidate(config_path, int(args.width), str(args.method))
    elif args.command == "final-holdout":
        result = final_holdout(config_path)
    else:
        result = efficiency(config_path)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
