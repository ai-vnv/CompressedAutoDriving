#!/usr/bin/env python3
"""Reproducible F12 Belief-PPO actor-compression pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 project environment
    import tomli as tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import run_episode, summarize_episodes
from duckie_pomdp.explain.development_protocol import PhaseThresholds, public_phase
from duckie_pomdp.optimization.actor_compression import (
    ActorSpec,
    DenseBeliefActor,
    actor_parameter_count,
    build_pruned_actor,
    convert_qat,
    distill_dense_actor,
    extract_original_actor,
    file_sha256,
    load_dense_actor,
    physical_actions,
    prepare_ptq,
    prepare_qat,
    save_dense_actor,
    save_quantized_actor,
)
from duckie_pomdp.optimization.compression_metrics import (
    action_fidelity,
    actor_physical_predictions,
    benchmark_actor,
    phase_fidelity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f12_belief_ppo_compression_v1.toml"


class ActorPolicy:
    def __init__(self, name: str, actor: torch.nn.Module) -> None:
        self.name = name
        self.actor = actor.cpu().eval()

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            value = self.actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0)
        return np.clip(value.cpu().numpy(), -1.0, 1.0).astype(np.float32)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    raw["_path"] = str(path.resolve())
    raw["_sha256"] = file_sha256(path)
    return raw


def artifact_root(config: dict[str, Any], config_path: Path) -> Path:
    return (config_path.parent / config["artifacts"]["directory"]).resolve()


def frozen_paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    checkpoint = (config_path.parent / config["frozen"]["checkpoint"]).resolve()
    policy_config = (config_path.parent / config["frozen"]["policy_config"]).resolve()
    return checkpoint, policy_config


def verify_protocol(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    checkpoint, policy_config = frozen_paths(config, config_path)
    if file_sha256(checkpoint) != config["frozen"]["checkpoint_sha256"]:
        raise RuntimeError("frozen Original Belief-PPO SHA256 mismatch")
    protocol = load_ppo_curriculum_protocol(policy_config)
    if len(protocol.observation_order) != 29:
        raise RuntimeError("F12 requires the unchanged public 29D contract")
    if tuple(protocol.ppo.hidden_sizes) != (256, 256):
        raise RuntimeError("frozen PPO architecture mismatch")
    sets = {
        name: set(int(value) for value in values)
        for name, values in config["seeds"].items()
    }
    primary = (
        "compression_development", "compression_selection",
        "compression_final_holdout", "retention_c0", "retention_c1",
        "retention_c2", "retention_c3",
    )
    for index, name in enumerate(primary):
        for other in primary[index + 1 :]:
            if sets[name] & sets[other]:
                raise RuntimeError(f"F12 seed leakage: {name} intersects {other}")
        if sets[name] & sets["forbidden_f11_locked"]:
            raise RuntimeError(f"F12 seed leakage into locked F11 seeds: {name}")
        if sets[name] & set(protocol.historical_seeds):
            raise RuntimeError(f"F12 seed leakage into historical seeds: {name}")
    return checkpoint, policy_config


def collect_dataset(config_path: Path, split: str) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint, policy_config = verify_protocol(config, config_path)
    if split == "final":
        require_final_claim(config, config_path)
    key = {
        "development": "compression_development",
        "selection": "compression_selection",
        "final": "compression_final_holdout",
    }[split]
    root = artifact_root(config, config_path) / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{split}_public_actor_states.npz"
    manifest_path = root / f"{split}_manifest.json"
    if target.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite F12 {split} dataset")
    actor, _, _ = extract_original_actor(
        checkpoint, expected_sha256=config["frozen"]["checkpoint_sha256"]
    )
    protocol = load_ppo_curriculum_protocol(policy_config)
    thresholds = PhaseThresholds(
        pedestrian_existence=float(config["data"]["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(config["data"]["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(config["data"]["lane_curve_min_abs_curvature_inv_m"]),
        stop_satisfied_vicinity_m=float(config["data"]["stop_satisfied_vicinity_m"]),
    )
    seeds = tuple(int(value) for value in config["seeds"][key])
    env = PPOCurriculumEnvironment(
        policy_config, stage="c4", split=f"f12_{split}", seeds=seeds
    )
    rows: dict[str, list[Any]] = {
        "seed": [], "episode": [], "step": [], "observation": [],
        "physical_observation": [], "teacher_raw_mean": [],
        "teacher_environment_action": [], "teacher_physical_action": [],
        "public_phase": [], "terminated": [], "truncated": [],
    }
    try:
        for episode, seed in enumerate(seeds):
            observation, info = env.reset(seed=seed)
            for step in range(int(config["data"]["maximum_episode_steps"])):
                physical = np.asarray(
                    [info["policy"][name] for name in protocol.observation_order],
                    dtype=np.float32,
                )
                phase = public_phase(physical, protocol.observation_order, thresholds)
                if phase == "combined_pedestrian_stop":
                    # The frozen five-phase F11 taxonomy assigns overlapping hazard
                    # states to stop-required for compression stratification.
                    phase = "stop_required"
                if phase not in config["data"]["phases"]:
                    raise RuntimeError(f"unsupported public F12 phase: {phase}")
                with torch.inference_mode():
                    mean = actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0)
                    physical_action = physical_actions(mean).cpu().numpy().astype(np.float32)
                environment_action = np.clip(mean.cpu().numpy(), -1.0, 1.0).astype(np.float32)
                next_observation, _, terminated, truncated, next_info = env.step(environment_action)
                rows["seed"].append(seed)
                rows["episode"].append(episode)
                rows["step"].append(step)
                rows["observation"].append(np.asarray(observation, dtype=np.float32))
                rows["physical_observation"].append(physical)
                rows["teacher_raw_mean"].append(mean.cpu().numpy().astype(np.float32))
                rows["teacher_environment_action"].append(environment_action)
                rows["teacher_physical_action"].append(physical_action)
                rows["public_phase"].append(phase)
                rows["terminated"].append(bool(terminated))
                rows["truncated"].append(bool(truncated))
                observation, info = next_observation, next_info
                if terminated or truncated:
                    break
    finally:
        env.close()
    arrays = {
        name: np.asarray(values, dtype=(
            np.float32 if name in {
                "observation", "physical_observation", "teacher_raw_mean",
                "teacher_environment_action", "teacher_physical_action"
            } else np.int64 if name in {"seed", "episode", "step"}
            else np.bool_ if name in {"terminated", "truncated"}
            else "U32"
        ))
        for name, values in rows.items()
    }
    if arrays["observation"].shape[1] != 29 or not np.isfinite(arrays["observation"]).all():
        raise RuntimeError("collected dataset violates public 29D contract")
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays, feature_names=np.asarray(protocol.observation_order, dtype="U64"))
    temporary.replace(target)
    phases, counts = np.unique(arrays["public_phase"], return_counts=True)
    manifest = {
        "schema_version": 1,
        "split": split,
        "config_sha256": config["_sha256"],
        "original_checkpoint_sha256": config["frozen"]["checkpoint_sha256"],
        "seeds": list(seeds),
        "rows": int(len(arrays["observation"])),
        "phase_counts": {str(k): int(v) for k, v in zip(phases, counts, strict=True)},
        "keys": sorted([*arrays, "feature_names"]),
        "contains_privileged_truth": False,
        "source_pipeline": "RGB->MobileNet/YOLO->belief->public29D->Original Belief-PPO",
        "dataset_sha256": file_sha256(target),
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_dataset(config: dict[str, Any], config_path: Path, split: str) -> dict[str, np.ndarray]:
    path = artifact_root(config, config_path) / "datasets" / f"{split}_public_actor_states.npz"
    manifest = load_json(path.with_name(f"{split}_manifest.json"))
    if file_sha256(path) != manifest["dataset_sha256"] or manifest["contains_privileged_truth"]:
        raise RuntimeError("F12 dataset provenance/privilege contract failed")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def build_pruning_frontier(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint, _ = verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    registry_path = root / "pruning" / "registry.json"
    if registry_path.exists():
        raise RuntimeError("refusing to overwrite F12 pruning frontier")
    data = _load_dataset(config, config_path, "development")
    original, log_std, _ = extract_original_actor(
        checkpoint, expected_sha256=config["frozen"]["checkpoint_sha256"]
    )
    registry: dict[str, Any] = {}
    baseline_path = root / "baseline" / "a0_original_actor.pt"
    save_dense_actor(
        baseline_path, original, log_std=log_std,
        metadata={"variant": "A0", "source_checkpoint_sha256": config["frozen"]["checkpoint_sha256"]},
    )
    registry["A0"] = candidate_entry("A0", "B-PPO", baseline_path, original.spec, False)
    distill = config["distillation"]
    for width in config["pruning"]["widths"]:
        width = int(width)
        tag = width_tag(width)
        result = build_pruned_actor(original, width)
        directory = root / "pruning" / tag
        p_path = directory / "actor_pruned_fp32.pt"
        pd_path = root / "prune_distill" / f"pd{tag[1:]}" / "actor_pruned_distilled_fp32.pt"
        metadata = {
            "source": "direct_from_original",
            "criterion": config["pruning"]["criterion"],
            "first_layer_survivors": list(result.first_layer_survivors),
            "second_layer_survivors": list(result.second_layer_survivors),
        }
        save_dense_actor(p_path, result.actor, log_std=log_std, metadata=metadata | {"distilled": False})
        student, _ = load_dense_actor(p_path)
        history = distill_dense_actor(
            student, data["observation"], data["teacher_physical_action"], data["public_phase"],
            epochs=int(distill["epochs"]), batch_size=int(distill["batch_size"]),
            learning_rate=float(distill["learning_rate"]), weight_decay=float(distill["weight_decay"]),
            seed=int(distill["seed"]) + width, device="cuda" if torch.cuda.is_available() else "cpu",
        )
        save_dense_actor(pd_path, student, log_std=log_std, metadata=metadata | {"distilled": True})
        write_json(pd_path.with_name("distillation_history.json"), {"history": history})
        registry[f"P{width}"] = candidate_entry(f"P{width}", f"P-{width}", p_path, result.actor.spec, False)
        registry[f"PD{width}"] = candidate_entry(f"PD{width}", f"PD-{width}", pd_path, student.spec, False)
    output = provenance(config, config_path) | {"candidates": registry}
    write_json(registry_path, output)
    return output


def evaluate_fidelity(config_path: Path, family: str, split: str) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    if split == "final":
        require_final_claim(config, config_path)
        if family != "matrix":
            raise RuntimeError("final fidelity is restricted to the frozen A0-A7 matrix")
    registry = load_registry(config, config_path, family)
    if split == "final":
        selected_key = load_json(artifact_root(config, config_path) / "final/model_selection.json")["selected_variant"]
        registry = {key: registry[key] for key in ("A0", selected_key)}
    data = _load_dataset(config, config_path, split)
    output_path = artifact_root(config, config_path) / "evaluation" / f"{split}_{family}_fidelity.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite {output_path}")
    results = {}
    for key, entry in registry.items():
        actor = load_candidate(entry)
        prediction = actor_physical_predictions(actor, data["observation"])
        overall = action_fidelity(
            data["teacher_physical_action"], prediction,
            omega_deadband=float(config["fidelity"]["omega_sign_deadband_rad_s"]),
        )
        phases = phase_fidelity(
            data["teacher_physical_action"], prediction, data["public_phase"],
            omega_deadband=float(config["fidelity"]["omega_sign_deadband_rad_s"]),
        )
        checks = fidelity_checks(overall, phases, config)
        results[key] = {"entry": entry, "overall": overall, "by_phase": phases, "checks": checks, "pass": all(checks.values())}
    output = provenance(config, config_path) | {"family": family, "split": split, "results": results}
    write_json(output_path, output)
    return output


def evaluate_closed_loop(config_path: Path, family: str, split: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, policy_config = verify_protocol(config, config_path)
    if split == "final":
        require_final_claim(config, config_path)
        if family != "matrix":
            raise RuntimeError("final closed-loop is restricted to the frozen A0-A7 matrix")
    registry = load_registry(config, config_path, family)
    if split == "final":
        selected_key = load_json(artifact_root(config, config_path) / "final/model_selection.json")["selected_variant"]
        registry = {key: registry[key] for key in ("A0", selected_key)}
    seeds = tuple(int(value) for value in config["seeds"][
        "compression_selection" if split == "selection" else "compression_final_holdout"
    ])
    root = artifact_root(config, config_path) / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{split}_{family}_closed_loop.csv"
    json_path = root / f"{split}_{family}_closed_loop.json"
    reused_evidence: dict[str, Any] | None = None
    if split == "selection" and family == "matrix" and not csv_path.exists():
        # A0/A1/A2 are byte-identical to the already evaluated A0/P/PD
        # pruning-frontier actors.  Reuse those same-seed episodes instead of
        # pretending a duplicate simulator rollout is independent evidence.
        selected_width = int(load_json(artifact_root(config, config_path) / "pruning/selected_width.json")["selected_width"])
        source_csv = root / "selection_pruning_closed_loop.csv"
        source_summary = root / "selection_pruning_closed_loop.json"
        source_registry = load_registry(config, config_path, "pruning")
        mapping = {"A0": "A0", "A1": f"P{selected_width}", "A2": f"PD{selected_width}"}
        for matrix_key, pruning_key in mapping.items():
            if registry[matrix_key]["sha256"] != source_registry[pruning_key]["sha256"]:
                raise RuntimeError(f"cannot reuse {pruning_key} as {matrix_key}: checkpoint hashes differ")
        source_rows = read_csv_rows(source_csv)
        reused_rows = []
        inverse = {value: key for key, value in mapping.items()}
        for row in source_rows:
            if row["variant"] not in inverse:
                continue
            copied = dict(row)
            copied["variant"] = inverse[row["variant"]]
            copied["policy"] = copied["variant"]
            reused_rows.append(copied)
        expected = len(mapping) * len(seeds)
        if len(reused_rows) != expected:
            raise RuntimeError(f"expected {expected} reusable selection rows, found {len(reused_rows)}")
        write_csv(csv_path, reused_rows)
        reused_evidence = {
            "source_family": "pruning",
            "source_csv": str(source_csv),
            "source_csv_sha256": file_sha256(source_csv),
            "source_summary": str(source_summary),
            "source_summary_sha256": file_sha256(source_summary),
            "variant_mapping": mapping,
            "checkpoint_sha256_verified": True,
            "same_seed_rows_reused": expected,
        }
    existing = read_csv_rows(csv_path)
    completed = {(row["variant"], int(row["seed"])) for row in existing}
    protocol = load_ppo_curriculum_protocol(policy_config)
    rows = list(existing)
    for key, entry in registry.items():
        actor = load_candidate(entry)
        policy = ActorPolicy(key, actor)
        env = PPOCurriculumEnvironment(policy_config, stage="c4", split=f"f12_{split}_{key}", seeds=seeds)
        try:
            for seed in seeds:
                if (key, seed) in completed:
                    continue
                result = run_episode(env, seed=seed, policy=policy, protocol=protocol)
                row = {"variant": key, **asdict(result)}
                rows.append(row)
                append_csv_row(csv_path, row)
        finally:
            env.close()
    summaries = {
        key: summarize_episodes_from_dicts([row for row in rows if row["variant"] == key])
        for key in registry
    }
    baseline = summaries["A0"]
    results = {}
    for key, summary in summaries.items():
        checks = behavior_checks(summary, baseline, config)
        results[key] = {"summary": summary, "checks": checks, "pass": all(checks.values())}
    output = provenance(config, config_path) | {
        "family": family, "split": split, "seeds": list(seeds), "results": results,
        "episodes_sha256": file_sha256(csv_path),
        "reused_identical_selection_evidence": reused_evidence,
    }
    write_json(json_path, output)
    return output


def select_pruning_width(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = artifact_root(config, config_path)
    target = root / "pruning" / "selected_width.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite selected pruning width")
    fidelity = load_json(root / "evaluation/selection_pruning_fidelity.json")["results"]
    behavior = load_json(root / "evaluation/selection_pruning_closed_loop.json")["results"]
    eligible = []
    for width in sorted((int(value) for value in config["pruning"]["widths"])):
        key = f"PD{width}"
        if fidelity[key]["pass"] and behavior[key]["pass"]:
            eligible.append(width)
    result = provenance(config, config_path) | {
        "eligible_widths": eligible,
        "selected_width": min(eligible) if eligible else None,
        "rule": config["selection"]["selected_pruning_width_rule"],
        "classification": "PASS" if eligible else "FAILED",
    }
    write_json(target, result)
    if not eligible:
        raise RuntimeError("no pruning+distillation width passed frozen gates")
    return result


def build_matrix(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    registry_path = root / "final" / "ablation_registry.json"
    if registry_path.exists():
        raise RuntimeError("refusing to overwrite F12 A0-A7 registry")
    pruning_registry = load_registry(config, config_path, "pruning")
    selected = load_json(root / "pruning/selected_width.json")
    width = int(selected["selected_width"])
    data = _load_dataset(config, config_path, "development")
    calibration = balanced_calibration(data, int(config["quantization"]["calibration_max_rows_per_phase"]))
    registry = {
        "A0": pruning_registry["A0"],
        "A1": pruning_registry[f"P{width}"] | {"variant": "A1", "name": "B-PPO-P"},
        "A2": pruning_registry[f"PD{width}"] | {"variant": "A2", "name": "B-PPO-PD"},
    }
    a0, a0_payload = load_dense_actor(registry["A0"]["model_path"])
    a1, _ = load_dense_actor(registry["A1"]["model_path"])
    a2, _ = load_dense_actor(registry["A2"]["model_path"])
    quant = config["quantization"]
    definitions = (
        ("A3", "B-PPO-Q", prepare_ptq(a0, calibration, backend=quant["backend"]), a0.spec, "quant_only"),
        ("A5", "B-PPO-PQ", prepare_ptq(a1, calibration, backend=quant["backend"]), a1.spec, "prune_quant"),
        ("A6", "B-PPO-PDQ", prepare_ptq(a2, calibration, backend=quant["backend"]), a2.spec, "prune_distill_quant"),
    )
    for key, name, actor, spec, directory in definitions:
        path = root / directory / "actor_int8.pt"
        save_quantized_actor(path, actor)
        registry[key] = candidate_entry(key, name, path, spec, True)
    for key, name, base, directory, seed_offset in (
        ("A4", "B-PPO-QD", a0, "quant_distill", 0),
        ("A7", "B-PPO-PDQD", a2, "prune_distill_quant_distill", 1),
    ):
        qat = prepare_qat(base, backend=quant["backend"])
        history = distill_dense_actor(
            qat, data["observation"], data["teacher_physical_action"], data["public_phase"],
            epochs=int(quant["qat_epochs"]), batch_size=int(config["distillation"]["batch_size"]),
            learning_rate=float(quant["qat_learning_rate"]), weight_decay=float(config["distillation"]["weight_decay"]),
            seed=int(quant["qat_seed"]) + seed_offset, device="cuda" if torch.cuda.is_available() else "cpu",
        )
        converted = convert_qat(qat, backend=quant["backend"])
        path = root / directory / "actor_int8.pt"
        save_quantized_actor(path, converted)
        write_json(path.with_name("qat_distillation_history.json"), {"history": history})
        registry[key] = candidate_entry(key, name, path, base.spec, True)
    output = provenance(config, config_path) | {
        "selected_pruning_width": width,
        "variants": registry,
        "quantization": config["quantization"],
        "state_independent_log_std_sha256": tensor_sha256(a0_payload["log_std"]),
    }
    write_json(registry_path, output)
    return output


def benchmark_matrix(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    registry = load_registry(config, config_path, "matrix")
    path = artifact_root(config, config_path) / "benchmarks/actor_benchmarks.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite F12 actor benchmark")
    bench = config["benchmark"]
    results = {}
    baseline_params = int(registry["A0"]["parameter_count"])
    for key, entry in registry.items():
        actor = load_candidate(entry)
        spec = ActorSpec(hidden_sizes=tuple(entry["hidden_sizes"]))
        result = benchmark_actor(
            actor, spec, entry["model_path"], warmup=int(bench["warmup_iterations"]),
            iterations=int(bench["timed_iterations"]), repeats=int(bench["repeats"]),
            threads=int(bench["threads"]), int8=bool(entry["int8"]),
        )
        result["parameter_compression_ratio"] = baseline_params / int(entry["parameter_count"])
        results[key] = result
    output = provenance(config, config_path) | {
        "target_device": "CPU x86; INT8 and FP32 compared on identical one-thread path",
        "end_to_end_latency_measured": False,
        "end_to_end_limitation": "perception/simulator unchanged; actor-only benchmark isolates F12 scope",
        "results": results,
    }
    write_json(path, output)
    return output


def select_deployment(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = artifact_root(config, config_path)
    target = root / "final/model_selection.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite F12 model selection")
    registry = load_registry(config, config_path, "matrix")
    fidelity = load_json(root / "evaluation/selection_matrix_fidelity.json")["results"]
    behavior = load_json(root / "evaluation/selection_matrix_closed_loop.json")["results"]
    benchmark = load_json(root / "benchmarks/actor_benchmarks.json")["results"]
    eligible = [key for key in registry if key != "A0" and fidelity[key]["pass"] and behavior[key]["pass"]]
    pareto = pareto_frontier(eligible, fidelity, benchmark)
    if not eligible:
        selected = None
    elif "A6" in pareto:
        selected = "A6"
        if "A7" in pareto:
            mae6 = normalized_mae(fidelity["A6"]["overall"])
            mae7 = normalized_mae(fidelity["A7"]["overall"])
            improvement = (mae6 - mae7) / max(mae6, 1.0e-12)
            if improvement >= float(config["selection"]["minimum_qat_relative_normalized_mae_improvement"]):
                selected = "A7"
    else:
        selected = min(
            pareto,
            key=lambda key: (
                normalized_mae(fidelity[key]["overall"]),
                benchmark[key]["logical_parameter_memory_bytes"],
                benchmark[key]["actor_checkpoint_size_bytes"],
                benchmark[key]["batch1_latency_us_median"],
            ),
        )
    result = provenance(config, config_path) | {
        "selected_variant": selected,
        "eligible_variants": eligible,
        "pareto_frontier": pareto,
        "selection_rule": config["selection"],
        "architecture": None if selected is None else registry[selected]["hidden_sizes"],
        "precision": None if selected is None else ("INT8" if registry[selected]["int8"] else "FP32"),
        "checkpoint_path": None if selected is None else registry[selected]["model_path"],
        "checkpoint_sha256": None if selected is None else registry[selected]["sha256"],
        "parameter_count": None if selected is None else registry[selected]["parameter_count"],
        "size": None if selected is None else benchmark[selected]["actor_checkpoint_size_bytes"],
        "latency": None if selected is None else benchmark[selected],
        "action_fidelity_metrics": None if selected is None else fidelity[selected],
        "closed_loop_metrics": None if selected is None else behavior[selected],
        "selection_rationale": "PDQ is sufficient when eligible; QAT selected only for >=10% normalized-MAE recovery without behavior regression",
        "classification": "PASS" if selected else "FAILED",
    }
    write_json(target, result)
    if selected is None:
        raise RuntimeError("no compressed candidate passed frozen F12 gates")
    return result


def claim_final(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    selection = load_json(artifact_root(config, config_path) / "final/model_selection.json")
    path = artifact_root(config, config_path) / "final/final_holdout_claim.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = provenance(config, config_path) | {
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": selection["selected_variant"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "seeds": config["seeds"]["compression_final_holdout"],
        "once_only": True,
        "status": "CLAIMED_BEFORE_FINAL_ACCESS",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def require_final_claim(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = artifact_root(config, config_path)
    claim_path = root / "final/final_holdout_claim.json"
    if not claim_path.exists():
        raise RuntimeError("final access requires the immutable pre-access claim")
    claim = load_json(claim_path)
    selection = load_json(root / "final/model_selection.json")
    if claim.get("config_sha256") != config["_sha256"]:
        raise RuntimeError("final claim/config provenance mismatch")
    if claim.get("selected_variant") != selection.get("selected_variant"):
        raise RuntimeError("final claim/selection mismatch")
    if claim.get("selected_checkpoint_sha256") != selection.get("checkpoint_sha256"):
        raise RuntimeError("final claim/checkpoint mismatch")
    if tuple(claim.get("seeds", ())) != tuple(config["seeds"]["compression_final_holdout"]):
        raise RuntimeError("final claim seed mismatch")
    return claim


def retention(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, policy_config = verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    target = root / "evaluation/retention.json"
    csv_path = root / "evaluation/retention.csv"
    if target.exists() or csv_path.exists():
        raise RuntimeError("refusing to overwrite F12 retention")
    registry = load_registry(config, config_path, "matrix")
    selected_key = load_json(root / "final/model_selection.json")["selected_variant"]
    protocol = load_ppo_curriculum_protocol(policy_config)
    rows = []
    for key in ("A0", selected_key):
        policy = ActorPolicy(key, load_candidate(registry[key]))
        for stage in ("c0", "c1", "c2", "c3"):
            seeds = tuple(int(value) for value in config["seeds"][f"retention_{stage}"])
            env = PPOCurriculumEnvironment(policy_config, stage=stage, split=f"f12_retention_{stage}", seeds=seeds)
            try:
                for seed in seeds:
                    rows.append({"variant": key, **asdict(run_episode(env, seed=seed, policy=policy, protocol=protocol))})
            finally:
                env.close()
    write_csv(csv_path, rows)
    summaries = {
        key: {
            stage: summarize_episodes_from_dicts([row for row in rows if row["variant"] == key and row["stage"] == stage])
            for stage in ("c0", "c1", "c2", "c3")
        }
        for key in ("A0", selected_key)
    }
    final_c4 = load_json(root / "evaluation/final_matrix_closed_loop.json")["results"]
    summaries["A0"]["c4"] = final_c4["A0"]["summary"]
    summaries[selected_key]["c4"] = final_c4[selected_key]["summary"]
    output = provenance(config, config_path) | {"selected_variant": selected_key, "summaries": summaries, "episodes_sha256": file_sha256(csv_path)}
    write_json(target, output)
    return output


def candidate_entry(key: str, name: str, path: Path, spec: ActorSpec, int8: bool) -> dict[str, Any]:
    return {
        "variant": key, "name": name, "model_path": str(path.resolve()),
        "sha256": file_sha256(path), "hidden_sizes": list(spec.hidden_sizes),
        "parameter_count": actor_parameter_count(spec), "int8": bool(int8),
    }


def load_registry(config: dict[str, Any], config_path: Path, family: str) -> dict[str, dict[str, Any]]:
    root = artifact_root(config, config_path)
    if family == "pruning":
        raw = load_json(root / "pruning/registry.json")["candidates"]
    elif family == "matrix":
        raw = load_json(root / "final/ablation_registry.json")["variants"]
    else:
        raise ValueError("family must be pruning or matrix")
    for entry in raw.values():
        if file_sha256(entry["model_path"]) != entry["sha256"]:
            raise RuntimeError(f"candidate hash mismatch: {entry['variant']}")
    return raw


def load_candidate(entry: dict[str, Any]) -> torch.nn.Module:
    if entry["int8"]:
        return torch.jit.load(entry["model_path"], map_location="cpu").eval()
    return load_dense_actor(entry["model_path"])[0]


def fidelity_checks(overall: dict[str, Any], phases: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gate = config["fidelity"]
    checks = {
        "v_mae": overall["v_cmd_mps"]["mae"] <= gate["maximum_v_mae_mps"],
        "v_p95": overall["v_cmd_mps"]["p95_absolute_error"] <= gate["maximum_v_p95_abs_error_mps"],
        "omega_mae": overall["omega_cmd_rad_s"]["mae"] <= gate["maximum_omega_mae_rad_s"],
        "omega_p95": overall["omega_cmd_rad_s"]["p95_absolute_error"] <= gate["maximum_omega_p95_abs_error_rad_s"],
        "omega_sign": overall["omega_sign"]["disagreement_frequency"] <= gate["maximum_omega_sign_disagreement"],
        "pearson": min(overall[name]["pearson"] for name in ("v_cmd_mps", "omega_cmd_rad_s")) >= gate["minimum_pearson"],
        "spearman": min(overall[name]["spearman"] for name in ("v_cmd_mps", "omega_cmd_rad_s")) >= gate["minimum_spearman"],
    }
    for phase in ("pedestrian_relevant", "stop_required"):
        if phase not in phases:
            checks[f"{phase}_supported"] = False
        else:
            checks[f"{phase}_v_mae"] = phases[phase]["v_cmd_mps"]["mae"] <= gate["critical_phase_maximum_v_mae_mps"]
            checks[f"{phase}_omega_mae"] = phases[phase]["omega_cmd_rad_s"]["mae"] <= gate["critical_phase_maximum_omega_mae_rad_s"]
    return checks


def behavior_checks(candidate: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gate = config["behavior"]
    checks = {
        "no_new_collision_count": candidate["collision_episodes"] <= baseline["collision_episodes"] + gate["maximum_new_collisions"],
        "collision_rate": candidate["collision_rate"] - baseline["collision_rate"] <= gate["maximum_collision_rate_increase"],
        "unsafe": candidate["unsafe_episode_rate"] - baseline["unsafe_episode_rate"] <= gate["maximum_unsafe_episode_rate_increase"],
        "stop_violation": candidate["stop_violation_rate"] - baseline["stop_violation_rate"] <= gate["maximum_stop_violation_rate_increase"],
        "lane_failure": candidate["lane_failure_rate"] - baseline["lane_failure_rate"] <= gate["maximum_lane_failure_rate_increase"],
        "completion": baseline["completion_rate"] - candidate["completion_rate"] <= gate["maximum_completion_rate_drop"],
        "restart": baseline["restart_rate"] - candidate["restart_rate"] <= gate["maximum_restart_rate_drop"],
        "progress": baseline["mean_progress_m"] - candidate["mean_progress_m"] <= gate["maximum_mean_progress_drop_m"],
    }
    b_clear = baseline["minimum_pedestrian_clearance_m"]
    c_clear = candidate["minimum_pedestrian_clearance_m"]
    checks["clearance"] = b_clear is None or (c_clear is not None and b_clear - c_clear <= gate["maximum_minimum_clearance_drop_m"])
    return checks


def summarize_episodes_from_dicts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no closed-loop rows")
    rate = lambda name: float(np.mean([bool(row[name]) for row in rows]))
    mean = lambda name: float(np.mean([float(row[name]) for row in rows]))
    clearances = [float(row["minimum_pedestrian_clearance_m"]) for row in rows if row["minimum_pedestrian_clearance_m"] not in (None, "")]
    return {
        "episodes": len(rows), "completion_rate": rate("completed"),
        "mean_progress_m": mean("progress_m"), "collision_rate": rate("collision"),
        "collision_episodes": int(sum(bool(row["collision"]) for row in rows)),
        "unsafe_episode_rate": float(np.mean([int(row["unsafe_proximity_events"]) > 0 for row in rows])),
        "minimum_pedestrian_clearance_m": min(clearances) if clearances else None,
        "stop_completion_rate": rate("stop_completed"), "stop_violation_rate": rate("stop_violation"),
        "restart_rate": rate("restarted_after_stop"), "lane_failure_rate": rate("lane_failure"),
        "invalid_pose_rate": rate("invalid_pose"), "timeout_rate": rate("timeout"),
        "mean_return": mean("total_return"), "mean_v_cmd_mps": mean("mean_v_cmd_mps"),
        "mean_abs_omega_cmd_rad_s": mean("mean_abs_omega_cmd_rad_s"),
        "stationary_fraction": mean("stationary_fraction"),
    }


def balanced_calibration(data: dict[str, np.ndarray], maximum_per_phase: int) -> np.ndarray:
    selected = []
    for phase in np.unique(data["public_phase"]):
        indexes = np.flatnonzero(data["public_phase"] == phase)[:maximum_per_phase]
        selected.extend(indexes.tolist())
    return np.asarray(data["observation"][np.asarray(selected, dtype=np.int64)], dtype=np.float32)


def normalized_mae(overall: dict[str, Any]) -> float:
    return 0.5 * (overall["v_cmd_mps"]["mae"] / 0.4 + overall["omega_cmd_rad_s"]["mae"] / 8.0)


def pareto_frontier(
    eligible: Sequence[str], fidelity: dict[str, Any], benchmark: dict[str, Any]
) -> list[str]:
    objectives = {
        key: (
            normalized_mae(fidelity[key]["overall"]),
            float(benchmark[key]["logical_parameter_memory_bytes"]),
            float(benchmark[key]["actor_checkpoint_size_bytes"]),
            float(benchmark[key]["batch1_latency_us_median"]),
        )
        for key in eligible
    }
    frontier = []
    for key, values in objectives.items():
        dominated = any(
            other != key
            and all(left <= right for left, right in zip(other_values, values, strict=True))
            and any(left < right for left, right in zip(other_values, values, strict=True))
            for other, other_values in objectives.items()
        )
        if not dominated:
            frontier.append(key)
    return sorted(frontier)


def width_tag(width: int) -> str:
    return {192: "p25", 128: "p50", 96: "p62_5", 64: "p75"}[width]


def provenance(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    checkpoint, policy_config = frozen_paths(config, config_path)
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()), "config_sha256": config["_sha256"],
        "original_checkpoint": str(checkpoint), "original_checkpoint_sha256": file_sha256(checkpoint),
        "policy_config": str(policy_config), "policy_config_sha256": file_sha256(policy_config),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
        "quantized_engine": torch.backends.quantized.engine,
        "actor_contract": {
            "observation_dimension": int(config["frozen"]["observation_dimension"]),
            "hidden_sizes": list(config["frozen"]["hidden_sizes"]),
            "activation": config["frozen"]["activation"],
            "action_dimension": int(config["frozen"]["action_dimension"]),
        },
    }


def tensor_sha256(value: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in ("seed", "steps", "unsafe_proximity_events"):
            row[name] = int(row[name])
        for name in ("completed", "collision", "stop_completed", "stop_violation", "restarted_after_stop", "lane_failure", "invalid_pose", "timeout"):
            row[name] = row[name].lower() == "true"
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "verify", "collect", "build-pruning", "fidelity", "closed-loop",
        "select-width", "build-matrix", "benchmark", "select-deployment",
        "claim-final", "retention",
    ))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("development", "selection", "final"))
    parser.add_argument("--family", choices=("pruning", "matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config_path = args.config.resolve()
    if args.command == "verify":
        config = load_config(config_path); verify_protocol(config, config_path); result = provenance(config, config_path)
    elif args.command == "collect":
        if args.split is None: raise SystemExit("--split is required")
        result = collect_dataset(config_path, args.split)
    elif args.command == "build-pruning": result = build_pruning_frontier(config_path)
    elif args.command == "fidelity":
        if args.split is None or args.family is None: raise SystemExit("--split and --family are required")
        result = evaluate_fidelity(config_path, args.family, args.split)
    elif args.command == "closed-loop":
        if args.split not in {"selection", "final"} or args.family is None: raise SystemExit("selection/final --split and --family are required")
        result = evaluate_closed_loop(config_path, args.family, args.split)
    elif args.command == "select-width": result = select_pruning_width(config_path)
    elif args.command == "build-matrix": result = build_matrix(config_path)
    elif args.command == "benchmark": result = benchmark_matrix(config_path)
    elif args.command == "select-deployment": result = select_deployment(config_path)
    elif args.command == "claim-final": result = claim_final(config_path)
    elif args.command == "retention": result = retention(config_path)
    else: raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
