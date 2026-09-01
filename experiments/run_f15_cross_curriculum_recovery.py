#!/usr/bin/env python3
"""F15 cross-curriculum failure localization and controlled recovery runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import PPOEpisodeEvaluation
from duckie_pomdp.explain.development_protocol import PhaseThresholds, public_phase
from duckie_pomdp.optimization.actor_compression import (
    ActorSpec,
    build_pruned_actor,
    convert_qat,
    extract_original_actor,
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
)
from duckie_pomdp.optimization.cross_curriculum_recovery import (
    CURRICULA,
    HUMAN_NAMES,
    canonical_json_sha256,
    distill_multicurriculum_actor,
    fidelity_pass,
    file_sha256,
    first_objective_failure_event,
    retention_decision,
    validate_seed_partition,
    verify_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"


class ActorPolicy:
    def __init__(self, name: str, actor: torch.nn.Module) -> None:
        self.name = name
        self.actor = actor.cpu().eval()

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            value = self.actor(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        return np.clip(value.cpu().numpy(), -1.0, 1.0).astype(np.float32)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_path"] = str(path.resolve())
    config["_sha256"] = file_sha256(path)
    return config


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def artifact_root(config: Mapping[str, Any], config_path: Path) -> Path:
    return resolve_config_path(config_path, str(config["artifacts"]["directory"]))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    temporary.replace(path)


def validate_episode_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_ids: Sequence[str],
    seeds: Sequence[int],
) -> None:
    """Fail closed on missing, duplicate, or out-of-protocol episode keys."""
    expected = {
        (str(model_id), curriculum, int(seed))
        for model_id in model_ids
        for curriculum in CURRICULA
        for seed in seeds
    }
    counts: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (str(row["model_id"]), str(row["curriculum"]), int(row["seed"]))
        counts[key] = counts.get(key, 0) + 1
    observed = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if duplicates or missing or unexpected:
        raise RuntimeError(
            "invalid F15 paired episode coverage: "
            f"duplicates={duplicates[:8]}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


def provenance(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": config["_sha256"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "quantized_engine": torch.backends.quantized.engine,
    }


def frozen_paths(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    frozen = config["frozen"]
    return {
        "policy_config": resolve_config_path(config_path, frozen["policy_config"]),
        "original": resolve_config_path(config_path, frozen["original_ppo_checkpoint"]),
        "ablation_registry": resolve_config_path(config_path, frozen["f12_ablation_registry"]),
        "pruning_registry": resolve_config_path(config_path, frozen["f12_pruning_registry"]),
        "f12_config": resolve_config_path(config_path, frozen["f12_config"]),
    }


def verify_protocol(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    paths = frozen_paths(config, config_path)
    frozen = config["frozen"]
    expected = {
        "original": frozen["original_ppo_sha256"],
        "ablation_registry": frozen["f12_ablation_registry_sha256"],
        "pruning_registry": frozen["f12_pruning_registry_sha256"],
        "f12_config": frozen["f12_config_sha256"],
    }
    for key, sha in expected.items():
        if file_sha256(paths[key]) != sha:
            raise RuntimeError(f"frozen F15 provenance mismatch: {key}")
    matrix = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=frozen["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    pruning = verify_registry(
        paths["pruning_registry"],
        expected_registry_sha256=frozen["f12_pruning_registry_sha256"],
        collection_key="candidates",
    )
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    if len(protocol.observation_order) != 29 or tuple(protocol.ppo.hidden_sizes) != (256, 256):
        raise RuntimeError("F15 public actor contract mismatch")
    expected_order = tuple(protocol.observation_order)
    if len(expected_order) != len(set(expected_order)):
        raise RuntimeError("29D feature names are not unique")
    primary_seed_mapping = {
        key: config["seeds"][key]
        for key in ("localization", "recovery_dataset", "recovery_selection", "final_holdout")
    }
    all_forbidden: list[int] = list(protocol.historical_seeds)
    for key, values in config["seeds"].items():
        if key.startswith("known_forbidden"):
            all_forbidden.extend(int(value) for value in values)
    validate_seed_partition(primary_seed_mapping, all_forbidden)
    curricula = {}
    for stage in CURRICULA:
        spec = protocol.stage(stage)
        curricula[stage] = {
            "name": spec.name,
            "map_name": spec.map_name,
            "scenario_config": None if spec.scenario_config_path is None else str(spec.scenario_config_path),
            "pedestrian_active": spec.pedestrian_active,
            "stop_active": spec.stop_active,
            "episode_horizon_steps": spec.episode_horizon_steps,
        }
    return {
        **provenance(config, config_path),
        "classification": "PASS",
        "original_checkpoint_sha256": file_sha256(paths["original"]),
        "policy_config_sha256": file_sha256(paths["policy_config"]),
        "observation_order": list(expected_order),
        "observation_scales": list(protocol.observation_scales),
        "action_mapping": {"v_cmd_mps": "(clip(raw_v,-1,1)+1)*0.2", "omega_cmd_rad_s": "clip(raw_omega,-1,1)*4"},
        "curricula": curricula,
        "seeds": primary_seed_mapping,
        "matrix_actor_hashes": {key: value["sha256"] for key, value in matrix.items()},
        "pruning_actor_hashes": {key: value["sha256"] for key, value in pruning.items()},
        "historical_artifacts_read_only": True,
        "f15_artifact_root_exists_before_preflight": artifact_root(config, config_path).exists(),
    }


def load_actor(entry: Mapping[str, Any]) -> torch.nn.Module:
    if bool(entry["int8"]):
        return torch.jit.load(str(entry["model_path"]), map_location="cpu").eval()
    return load_dense_actor(entry["model_path"])[0]


def phase_thresholds(f12_config_path: Path) -> PhaseThresholds:
    with f12_config_path.open("rb") as stream:
        f12 = tomllib.load(stream)
    data = f12["data"]
    return PhaseThresholds(
        pedestrian_existence=float(data["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(data["pedestrian_relevant_max_range_m"]),
        lane_curve_min_abs_curvature_inv_m=float(data["lane_curve_min_abs_curvature_inv_m"]),
        stop_satisfied_vicinity_m=float(data["stop_satisfied_vicinity_m"]),
    )


def trace_path(root: Path, split: str, model: str, curriculum: str, seed: int) -> Path:
    return root / "telemetry" / split / model / curriculum / f"seed_{seed}" / "trace.npz"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_episode_with_telemetry(
    environment: PPOCurriculumEnvironment,
    *,
    seed: int,
    policy: ActorPolicy,
    protocol,
    thresholds: PhaseThresholds,
    target: Path,
) -> dict[str, Any]:
    observation, current_info = environment.reset(seed=seed)
    policy.reset(seed)
    stage = protocol.stage(str(current_info["stage"]))
    public_physical: list[np.ndarray] = []
    public_normalized: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    physical_actions_rows: list[np.ndarray] = []
    phases: list[str] = []
    progress: list[float] = []
    v_actual: list[float] = []
    omega_actual: list[float] = []
    flags = {name: [] for name in (
        "completed", "collision", "unsafe", "stop_completed", "stop_violation",
        "lane_failure", "invalid_pose", "terminated", "truncated", "timeout",
    )}
    evaluation_only = {name: [] for name in (
        "lane_lateral_error_m", "lane_heading_error_rad", "road_curvature_inv_m",
        "pedestrian_range_m", "pedestrian_bearing_rad", "stop_line_distance_m",
    )}
    termination_reason: list[str] = []
    truncation_reason: list[str] = []
    rewards = {name: 0.0 for name in (
        "reward_progress", "reward_lane", "reward_pedestrian", "reward_stop",
        "reward_smoothness", "reward_terminal",
    )}
    clearances: list[float] = []
    action_changes: list[float] = []
    prior_physical_action = np.zeros(2, dtype=np.float32)
    total_return = 0.0
    stop_ever_completed = False
    restarted = False
    yellow_contact_steps = 0
    yellow_recovery_events = 0
    yellow_recovery_successes = 0
    yellow_recovery_failures = 0
    last_info: dict[str, Any] | None = None
    for step in range(stage.episode_horizon_steps):
        physical = np.asarray([current_info["policy"][name] for name in protocol.observation_order], dtype=np.float32)
        phase = public_phase(physical, protocol.observation_order, thresholds)
        if phase == "combined_pedestrian_stop":
            phase = "stop_required"
        action = policy.act(observation)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        public_physical.append(physical)
        public_normalized.append(np.asarray(observation, dtype=np.float32))
        normalized_actions.append(np.asarray(action, dtype=np.float32))
        action_physical = np.asarray((info["v_cmd"], info["omega_cmd"]), dtype=np.float32)
        physical_actions_rows.append(action_physical)
        phases.append(phase)
        progress.append(float(info["progress_m"]))
        v_actual.append(float(info["v_actual"]))
        omega_actual.append(float(info["omega_actual"]))
        flags["completed"].append(bool(info["completed"]))
        flags["collision"].append(bool(info["collision"]))
        flags["unsafe"].append(bool(info["unsafe_proximity"]))
        flags["stop_completed"].append(bool(info["stop_completed"]))
        flags["stop_violation"].append(bool(info["stop_violation"]))
        flags["lane_failure"].append(bool(info["lane_failure"]))
        flags["invalid_pose"].append(bool(info["invalid_pose"]))
        flags["terminated"].append(bool(terminated))
        flags["truncated"].append(bool(truncated))
        flags["timeout"].append(bool(info["truncation_reason"]))
        termination_reason.append("" if info["termination_reason"] is None else str(info["termination_reason"]))
        truncation_reason.append("" if info["truncation_reason"] is None else str(info["truncation_reason"]))
        for name, value in info["evaluation_gt"].items():
            evaluation_only[name].append(np.nan if value is None else float(value))
        total_return += float(reward)
        for name in rewards:
            rewards[name] += float(info[name])
        clearance = info["pedestrian_clearance_m"]
        if clearance is not None and math.isfinite(float(clearance)):
            clearances.append(float(clearance))
        delta = np.asarray(((action_physical[0] - prior_physical_action[0]) / 0.4, (action_physical[1] - prior_physical_action[1]) / 4.0))
        action_changes.append(float(np.linalg.norm(delta)))
        prior_physical_action = action_physical
        stop_ever_completed = stop_ever_completed or bool(info["stop_completed"])
        restarted = restarted or (stop_ever_completed and float(action_physical[0]) > 0.08)
        yellow_contact_steps += int(bool(info["yellow_contact"]))
        yellow_recovery_events += int(bool(info["yellow_recovery_started"]))
        yellow_recovery_successes += int(bool(info["yellow_recovered"]))
        yellow_recovery_failures += int(info["termination_reason"] == "yellow_recovery_failed")
        last_info = info
        observation, current_info = next_observation, info
        if terminated or truncated:
            break
    if last_info is None:
        raise RuntimeError("F15 episode produced no steps")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        public_physical_29d=np.asarray(public_physical, dtype=np.float32),
        public_normalized_29d=np.asarray(public_normalized, dtype=np.float32),
        normalized_action=np.asarray(normalized_actions, dtype=np.float32),
        physical_action=np.asarray(physical_actions_rows, dtype=np.float32),
        public_phase=np.asarray(phases, dtype="U32"),
        progress_m=np.asarray(progress, dtype=np.float32),
        v_actual_mps=np.asarray(v_actual, dtype=np.float32),
        omega_actual_rad_s=np.asarray(omega_actual, dtype=np.float32),
        termination_reason=np.asarray(termination_reason, dtype="U80"),
        truncation_reason=np.asarray(truncation_reason, dtype="U80"),
        feature_names=np.asarray(protocol.observation_order, dtype="U64"),
        **{name: np.asarray(values, dtype=np.bool_) for name, values in flags.items()},
    )
    temporary.replace(target)
    eval_target = target.with_name("evaluation_only.npz")
    eval_temporary = eval_target.with_suffix(".tmp.npz")
    np.savez_compressed(eval_temporary, **{name: np.asarray(values, dtype=np.float32) for name, values in evaluation_only.items()})
    eval_temporary.replace(eval_target)
    physical_matrix = np.asarray(physical_actions_rows, dtype=np.float32)
    evaluation = PPOEpisodeEvaluation(
        stage=stage.key,
        policy=policy.name,
        checkpoint_step=None,
        seed=seed,
        scenario=str(current_info["scenario"]),
        steps=len(physical_matrix),
        total_return=total_return,
        completed=bool(last_info["completed"]),
        progress_m=float(last_info["progress_m"]),
        collision=bool(last_info["collision"]),
        unsafe_proximity_events=int(sum(flags["unsafe"])),
        minimum_pedestrian_clearance_m=min(clearances) if clearances else None,
        stop_completed=stop_ever_completed,
        stop_violation=bool(last_info["stop_violation"]),
        restarted_after_stop=restarted,
        lane_failure=bool(last_info["lane_failure"]),
        yellow_contact_steps=yellow_contact_steps,
        yellow_recovery_events=yellow_recovery_events,
        yellow_recovery_successes=yellow_recovery_successes,
        yellow_recovery_failures=yellow_recovery_failures,
        invalid_pose=bool(last_info["invalid_pose"]),
        timeout=bool(last_info["truncation_reason"]),
        mean_abs_lateral_error_m=_mean(np.abs(np.asarray(evaluation_only["lane_lateral_error_m"]))),
        mean_abs_heading_error_rad=_mean(np.abs(np.asarray(evaluation_only["lane_heading_error_rad"]))),
        mean_v_cmd_mps=_mean(physical_matrix[:, 0]),
        mean_abs_omega_cmd_rad_s=_mean(np.abs(physical_matrix[:, 1])),
        mean_action_change=_mean(action_changes),
        stationary_fraction=float(np.mean(physical_matrix[:, 0] <= 0.025)),
        **rewards,
        termination_reason=None if last_info["termination_reason"] is None else str(last_info["termination_reason"]),
        truncation_reason=None if last_info["truncation_reason"] is None else str(last_info["truncation_reason"]),
    )
    row = asdict(evaluation)
    row.update({
        "model_sha256": file_sha256(_CURRENT_MODEL_PATH),
        "trace_path": str(target),
        "trace_sha256": file_sha256(target),
        "evaluation_only_path": str(eval_target),
        "evaluation_only_sha256": file_sha256(eval_target),
    })
    return row


_CURRENT_MODEL_PATH = Path(".")


def summarize_episode_dicts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty F15 rows")
    def truth(value: Any) -> bool:
        return value if isinstance(value, bool) else str(value).lower() == "true"
    def rate(name: str) -> float:
        return float(np.mean([truth(row[name]) for row in rows]))
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))
    clearances = [float(row["minimum_pedestrian_clearance_m"]) for row in rows if row.get("minimum_pedestrian_clearance_m") not in (None, "")]
    return {
        "episodes": len(rows),
        "completion_rate": rate("completed"),
        "mean_progress_m": mean("progress_m"),
        "collision_rate": rate("collision"),
        "collision_episodes": int(sum(truth(row["collision"]) for row in rows)),
        "unsafe_episode_rate": float(np.mean([int(row["unsafe_proximity_events"]) > 0 for row in rows])),
        "minimum_pedestrian_clearance_m": min(clearances) if clearances else None,
        "stop_completion_rate": rate("stop_completed"),
        "stop_violation_rate": rate("stop_violation"),
        "restart_rate": rate("restarted_after_stop"),
        "lane_failure_rate": rate("lane_failure"),
        "invalid_pose_rate": rate("invalid_pose"),
        "timeout_rate": rate("timeout"),
        "mean_abs_lateral_error_m": mean("mean_abs_lateral_error_m"),
        "mean_abs_heading_error_rad": mean("mean_abs_heading_error_rad"),
        "mean_v_cmd_mps": mean("mean_v_cmd_mps"),
        "mean_abs_omega_cmd_rad_s": mean("mean_abs_omega_cmd_rad_s"),
        "stationary_fraction": mean("stationary_fraction"),
    }


def evaluate_registry(
    config_path: Path,
    *,
    split: str,
    registry: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    episode_csv_override: Path | None = None,
    build_results: bool = True,
) -> dict[str, Any]:
    global _CURRENT_MODEL_PATH
    config = load_config(config_path)
    paths = frozen_paths(config, config_path)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    thresholds = phase_thresholds(paths["f12_config"])
    root = artifact_root(config, config_path)
    section = "localization" if split in {"matrix", "pruning"} else "recovery"
    episode_csv = episode_csv_override or (root / section / f"{split}_episodes.csv")
    existing = read_csv(episode_csv)
    completed = {(row["model_id"], row["curriculum"], int(row["seed"])) for row in existing}
    rows = list(existing)
    for model_id, entry in registry.items():
        _CURRENT_MODEL_PATH = Path(entry["model_path"])
        actor = load_actor(entry)
        policy = ActorPolicy(HUMAN_NAMES.get(model_id, entry.get("name", model_id)), actor)
        for curriculum in CURRICULA:
            environment = PPOCurriculumEnvironment(
                paths["policy_config"], stage=curriculum, split=f"f15_{split}_{model_id}_{curriculum}", seeds=tuple(int(value) for value in seeds)
            )
            try:
                for seed in seeds:
                    key = (model_id, curriculum, int(seed))
                    if key in completed:
                        continue
                    target = trace_path(root, split, model_id, curriculum, int(seed))
                    row = run_episode_with_telemetry(
                        environment, seed=int(seed), policy=policy, protocol=protocol,
                        thresholds=thresholds, target=target,
                    )
                    row = {"model_id": model_id, "model_name": policy.name, "curriculum": curriculum, **row}
                    append_csv(episode_csv, row)
                    rows.append({key: str(value) if value is None else value for key, value in row.items()})
            finally:
                environment.close()
    if not build_results:
        return {
            **provenance(config, config_path), "split": split, "episode_csv": str(episode_csv),
            "episode_rows": len(rows), "episodes_sha256": file_sha256(episode_csv),
        }
    return build_retention_results(config_path, split=split, registry=registry, episode_csv=episode_csv, seeds=seeds)


def build_retention_results(
    config_path: Path,
    *,
    split: str,
    registry: Mapping[str, Mapping[str, Any]],
    episode_csv: Path,
    seeds: Sequence[int],
) -> dict[str, Any]:
    config = load_config(config_path)
    rows = read_csv(episode_csv)
    validate_episode_coverage(rows, model_ids=tuple(registry), seeds=seeds)
    summaries: dict[str, dict[str, Any]] = {}
    for model_id in registry:
        summaries[model_id] = {}
        for curriculum in CURRICULA:
            selected = [row for row in rows if row["model_id"] == model_id and row["curriculum"] == curriculum]
            summaries[model_id][curriculum] = summarize_episode_dicts(selected)
    baseline = summaries["A0"]
    decisions: dict[str, dict[str, Any]] = {"A0": {}}
    for curriculum in CURRICULA:
        original_checks = retention_decision(
            curriculum, baseline[curriculum], baseline[curriculum], config["retention"]["absolute"],
            config["retention"]["relative_to_original"], candidate_prior=baseline, original_prior=baseline,
        )
        decisions["A0"][curriculum] = asdict(original_checks) | {"status": "REFERENCE" if original_checks.original_absolute_pass else "UNRESOLVED"}
    for model_id in registry:
        if model_id == "A0":
            continue
        decisions[model_id] = {}
        for curriculum in CURRICULA:
            decisions[model_id][curriculum] = asdict(retention_decision(
                curriculum, summaries[model_id][curriculum], baseline[curriculum],
                config["retention"]["absolute"], config["retention"]["relative_to_original"],
                candidate_prior=summaries[model_id], original_prior=baseline,
            ))
    output = {
        **provenance(config, config_path),
        "split": split,
        "seeds": [int(value) for value in seeds],
        "summaries": summaries,
        "decisions": decisions,
        "episodes_csv": str(episode_csv),
        "episodes_sha256": file_sha256(episode_csv),
    }
    section = "localization" if split in {"matrix", "pruning"} else "recovery"
    path = artifact_root(config, config_path) / section / f"{split}_results.json"
    write_json(path, output)
    return output


def run_same_state_fidelity(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = frozen_paths(config, config_path)
    registry = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    root = artifact_root(config, config_path)
    results: dict[str, dict[str, Any]] = {}
    csv_target = root / "localization/open_loop_fidelity_by_curriculum.csv"
    if csv_target.exists():
        raise RuntimeError("refusing to overwrite F15 same-state fidelity")
    matrix_episode_rows = read_csv(root / "localization/matrix_episodes.csv")
    validate_episode_coverage(
        matrix_episode_rows,
        model_ids=tuple(registry),
        seeds=config["seeds"]["localization"],
    )
    for curriculum in CURRICULA:
        observations = []
        for seed in config["seeds"]["localization"]:
            row = next(
                row for row in matrix_episode_rows
                if row["model_id"] == "A0"
                and row["curriculum"] == curriculum
                and int(row["seed"]) == int(seed)
            )
            path = Path(row["trace_path"])
            with np.load(path, allow_pickle=False) as archive:
                observations.append(np.asarray(archive["public_normalized_29d"], dtype=np.float32))
        matrix = np.concatenate(observations)
        original = actor_physical_predictions(load_actor(registry["A0"]), matrix)
        results[curriculum] = {}
        for model_id, entry in registry.items():
            candidate = actor_physical_predictions(load_actor(entry), matrix)
            metrics = action_fidelity(original, candidate, omega_deadband=float(config["evaluation"]["omega_sign_deadband_rad_s"]))
            passed, checks = fidelity_pass(metrics, config["fidelity"])
            results[curriculum][model_id] = {"metrics": metrics, "checks": checks, "pass": passed}
            row = {
                "curriculum": curriculum, "model_id": model_id, "model_name": HUMAN_NAMES[model_id], "rows": metrics["rows"],
                "v_mae_mps": metrics["v_cmd_mps"]["mae"], "v_rmse_mps": metrics["v_cmd_mps"]["rmse"],
                "v_median_absolute_error_mps": metrics["v_cmd_mps"]["median_absolute_error"],
                "v_p95_mps": metrics["v_cmd_mps"]["p95_absolute_error"], "v_p99_mps": metrics["v_cmd_mps"]["p99_absolute_error"],
                "v_max_mps": metrics["v_cmd_mps"]["maximum_absolute_error"], "v_bias_mps": metrics["v_cmd_mps"]["bias"],
                "v_pearson": metrics["v_cmd_mps"]["pearson"], "v_spearman": metrics["v_cmd_mps"]["spearman"],
                "omega_mae_rad_s": metrics["omega_cmd_rad_s"]["mae"], "omega_rmse_rad_s": metrics["omega_cmd_rad_s"]["rmse"],
                "omega_median_absolute_error_rad_s": metrics["omega_cmd_rad_s"]["median_absolute_error"],
                "omega_p95_rad_s": metrics["omega_cmd_rad_s"]["p95_absolute_error"], "omega_p99_rad_s": metrics["omega_cmd_rad_s"]["p99_absolute_error"],
                "omega_max_rad_s": metrics["omega_cmd_rad_s"]["maximum_absolute_error"], "omega_bias_rad_s": metrics["omega_cmd_rad_s"]["bias"],
                "omega_pearson": metrics["omega_cmd_rad_s"]["pearson"], "omega_spearman": metrics["omega_cmd_rad_s"]["spearman"],
                "omega_sign_disagreement": metrics["omega_sign"]["disagreement_frequency"],
                "original_saturation_frequency": metrics["action_bound_saturation_frequency"]["original"],
                "candidate_saturation_frequency": metrics["action_bound_saturation_frequency"]["candidate"],
                "saturation_disagreement": metrics["action_bound_saturation_frequency"]["disagreement"],
                "pass": passed,
            }
            append_csv(csv_target, row)
    output = {**provenance(config, config_path), "results": results, "csv": str(csv_target), "csv_sha256": file_sha256(csv_target)}
    write_json(root / "localization/open_loop_fidelity_by_curriculum.json", output)
    return output


def freeze_localization(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = artifact_root(config, config_path)
    target = root / "localization/failure_localization_decision.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite F15 localization decision")
    matrix = read_json(root / "localization/matrix_results.json")
    pruning = read_json(root / "localization/pruning_results.json")
    fidelity = read_json(root / "localization/open_loop_fidelity_by_curriculum.json")
    competence_csv = root / "localization/cross_curriculum_results.csv"
    pruning_csv = root / "localization/pruning_width_retention.csv"
    event_csv = root / "localization/failure_event_registry.csv"
    if any(path.exists() for path in (competence_csv, pruning_csv, event_csv)):
        raise RuntimeError("refusing to overwrite frozen F15 localization tables")
    for model_id, curricula in matrix["decisions"].items():
        for curriculum, decision in curricula.items():
            summary = matrix["summaries"][model_id][curriculum]
            append_csv(competence_csv, {
                "model_id": model_id, "model_name": HUMAN_NAMES.get(model_id, model_id),
                "curriculum": curriculum.upper(), "status": decision["status"],
                "completion_rate": summary["completion_rate"], "mean_progress_m": summary["mean_progress_m"],
                "collision_rate": summary["collision_rate"], "lane_failure_rate": summary["lane_failure_rate"],
                "invalid_pose_rate": summary["invalid_pose_rate"], "stop_violation_rate": summary["stop_violation_rate"],
            })
    for model_id, curricula in pruning["decisions"].items():
        for curriculum, decision in curricula.items():
            summary = pruning["summaries"][model_id][curriculum]
            append_csv(pruning_csv, {
                "model_id": model_id, "model_name": model_id, "curriculum": curriculum.upper(),
                "status": decision["status"], "completion_rate": summary["completion_rate"],
                "mean_progress_m": summary["mean_progress_m"], "lane_failure_rate": summary["lane_failure_rate"],
                "invalid_pose_rate": summary["invalid_pose_rate"],
            })
    event_rows = []
    for family, result in (("matrix", matrix), ("pruning", pruning)):
        episode_rows = read_csv(Path(result["episodes_csv"]))
        for model_id, curricula in result["decisions"].items():
            if model_id == "A0":
                continue
            for curriculum, decision in curricula.items():
                if decision["status"] != "FAIL":
                    continue
                failing = sorted(
                    (
                        row for row in episode_rows
                        if row["model_id"] == model_id
                        and row["curriculum"] == curriculum
                        and (
                            str(row["completed"]).lower() != "true"
                            or str(row["collision"]).lower() == "true"
                            or int(row["unsafe_proximity_events"]) > 0
                            or str(row["stop_violation"]).lower() == "true"
                            or str(row["lane_failure"]).lower() == "true"
                            or str(row["invalid_pose"]).lower() == "true"
                            or str(row["timeout"]).lower() == "true"
                        )
                    ),
                    key=lambda row: int(row["seed"]),
                )
                if not failing:
                    continue
                episode = failing[0]
                trace = Path(episode["trace_path"])
                with np.load(trace, allow_pickle=False) as archive:
                    step_rows = [
                        {
                            "step": step,
                            "collision": bool(archive["collision"][step]),
                            "unsafe": bool(archive["unsafe"][step]),
                            "stop_violation": bool(archive["stop_violation"][step]),
                            "lane_failure": bool(archive["lane_failure"][step]),
                            "invalid_pose": bool(archive["invalid_pose"][step]),
                            "timeout": bool(archive["timeout"][step]),
                            "terminated": bool(archive["terminated"][step]),
                            "truncated": bool(archive["truncated"][step]),
                            "completed": bool(archive["completed"][step]),
                        }
                        for step in range(len(archive["completed"]))
                    ]
                event = first_objective_failure_event(step_rows)
                event_row = {
                    "family": family, "model_id": model_id,
                    "model_name": HUMAN_NAMES.get(model_id, result.get("summaries", {}).get(model_id, model_id) if isinstance(result.get("summaries", {}).get(model_id), str) else model_id),
                    "curriculum": curriculum, "seed": int(episode["seed"]),
                    "event_step": None if event is None else event["step"],
                    "event_labels": "UNRESOLVED" if event is None else "|".join(event["event_labels"]),
                    "trace_path": str(trace), "trace_sha256": file_sha256(trace),
                    "selection_rule": config["evaluation"]["representative_failure_rule"],
                }
                append_csv(event_csv, event_row)
                event_rows.append(event_row)
    final_path = ("A0", "A1", "A2", "A7")
    first: dict[str, Any] = {}
    for curriculum in CURRICULA:
        statuses = [matrix["decisions"][model][curriculum]["status"] for model in final_path]
        collapse = None
        for predecessor, successor, left, right in zip(final_path, final_path[1:], statuses, statuses[1:], strict=True):
            if left in {"REFERENCE", "PASS"} and right == "FAIL":
                collapse = {"transition": f"{predecessor}->{successor}", "after": HUMAN_NAMES[successor]}
                break
        first[curriculum] = {"path": list(final_path), "statuses": statuses, "first_collapse": collapse}

    def transition(left: str, right: str, curriculum: str) -> dict[str, Any]:
        left_status = matrix["decisions"][left][curriculum]["status"]
        right_status = matrix["decisions"][right][curriculum]["status"]
        if left_status in {"REFERENCE", "PASS"} and right_status == "FAIL":
            interpretation = "new_failure"
        elif left_status == "FAIL" and right_status == "PASS":
            interpretation = "recovered"
        elif left_status == right_status:
            interpretation = "unchanged"
        else:
            interpretation = "unresolved_transition"
        return {
            "from": left,
            "to": right,
            "from_status": left_status,
            "to_status": right_status,
            "interpretation": interpretation,
        }

    branch_diagnostics = {
        curriculum: {
            "direct_pruning": transition("A0", "A1", curriculum),
            "distillation_after_pruning": transition("A1", "A2", curriculum),
            "ptq_after_pruning_distillation": transition("A2", "A6", curriculum),
            "qat_kd_after_ptq_branch": transition("A6", "A7", curriculum),
            "ptq_only": transition("A0", "A3", curriculum),
            "qat_distillation_unpruned": transition("A3", "A4", curriculum),
            "ptq_after_direct_pruning": transition("A1", "A5", curriculum),
        }
        for curriculum in CURRICULA
    }

    pruning_width_diagnostics: dict[str, Any] = {}
    for curriculum in CURRICULA:
        pruning_only = {
            width: pruning["decisions"][f"P{width}"][curriculum]["status"]
            for width in (192, 128, 96, 64)
        }
        pruning_distilled = {
            width: pruning["decisions"][f"PD{width}"][curriculum]["status"]
            for width in (192, 128, 96, 64)
        }
        pruning_width_diagnostics[curriculum] = {
            "pruning_only": pruning_only,
            "pruning_plus_historical_kd": pruning_distilled,
            "smallest_passing_pruning_only_width": next(
                (width for width in (64, 96, 128, 192) if pruning_only[width] == "PASS"), None
            ),
            "smallest_passing_distilled_width": next(
                (width for width in (64, 96, 128, 192) if pruning_distilled[width] == "PASS"), None
            ),
            "capacity_claim_limit": (
                "Width comparisons localize an association under the historical pruning/KD protocol; "
                "they do not prove a neuron-level causal capacity threshold."
            ),
        }
    output = {
        **provenance(config, config_path),
        "classification": "FROZEN",
        "first_collapse_by_curriculum": first,
        "matrix_results_sha256": file_sha256(root / "localization/matrix_results.json"),
        "pruning_results_sha256": file_sha256(root / "localization/pruning_results.json"),
        "open_loop_fidelity_sha256": file_sha256(root / "localization/open_loop_fidelity_by_curriculum.json"),
        "cross_curriculum_results_csv": str(competence_csv),
        "cross_curriculum_results_sha256": file_sha256(competence_csv),
        "pruning_width_retention_csv": str(pruning_csv),
        "pruning_width_retention_sha256": file_sha256(pruning_csv),
        "failure_event_registry_csv": str(event_csv),
        "failure_event_registry_sha256": file_sha256(event_csv),
        "failure_events": event_rows,
        "branch_diagnostics": branch_diagnostics,
        "pruning_width_diagnostics": pruning_width_diagnostics,
        "rehearsal_coverage_hypothesis": {
            "status": "TO_BE_TESTED_AFTER_FREEZE",
            "basis": "historical F12 distillation used C4-focused public states; localization alone cannot establish causality",
            "test": "hold architecture, pruning survivors, loss, optimizer family, and budget fixed while balancing C0-C4 public rehearsal",
        },
        "optimization_order": {
            "status": "UNRESOLVED_UNLESS_DIRECT_RECOVERY_FAILS",
            "reason": "historical branches are partly parallel; progressive prune-distill is deferred by protocol",
        },
        "mechanism_claim_limit": "comparative localization supports associations, not neuron-level causality",
        "recovery_may_begin_after_this_file": True,
    }
    write_json(root / "localization/cross_curriculum_results.json", {
        **provenance(config, config_path),
        "seeds": matrix["seeds"],
        "summaries": matrix["summaries"],
        "decisions": matrix["decisions"],
        "csv": str(competence_csv),
        "csv_sha256": file_sha256(competence_csv),
    })
    write_json(target, output)
    return output


def verify_command(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    result = verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    if root.exists() and any((root / name).exists() for name in ("localization", "recovery", "final")):
        result["f15_scientific_outputs_absent"] = False
    else:
        result["f15_scientific_outputs_absent"] = True
    return result


def initialize_manifests(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verified = verify_protocol(config, config_path)
    paths = frozen_paths(config, config_path)
    root = artifact_root(config, config_path)
    targets = (
        root / "model_registry.json", root / "protocol_manifest.json", root / "seed_manifest.json",
    )
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite F15 frozen manifests")
    matrix = read_json(paths["ablation_registry"])["variants"]
    pruning = read_json(paths["pruning_registry"])["candidates"]
    write_json(targets[0], {
        **provenance(config, config_path), "historical_matrix": matrix, "historical_pruning_frontier": pruning,
        "all_registered_hashes_verified": True, "original_ppo_sha256": config["frozen"]["original_ppo_sha256"],
    })
    write_json(targets[1], {
        **provenance(config, config_path), "protocol_document": str((ROOT / "docs/F15_PROTOCOL.md").resolve()),
        "protocol_document_sha256": file_sha256(ROOT / "docs/F15_PROTOCOL.md"),
        "config_sha256": config["_sha256"], "preflight_classification": verified["classification"],
        "freeze_note": "docs/config were frozen and preflight passed before localization; this machine-readable copy was materialized without reading results",
        "no_explainability_methods": True,
    })
    write_json(targets[2], {
        **provenance(config, config_path), "seed_blocks": {
            key: values for key, values in config["seeds"].items() if not key.startswith("known_forbidden")
        }, "known_forbidden_blocks": {
            key: values for key, values in config["seeds"].items() if key.startswith("known_forbidden")
        }, "pairing": "same seed for every compared model within each curriculum", "final_holdout_unopened": True,
    })
    return {"created": [str(path) for path in targets], "config_sha256": config["_sha256"]}


def write_repository_audit(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verified = verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    target = root / "integrity/repository_audit.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite F15 repository audit")
    requested = (
        ROOT / "FORMULATION.md",
        ROOT / "GATES.md",
        ROOT / "IMPLEMENTATION_NOTES.md",
        ROOT / "docs/F10_PPO_CURRICULUM.md",
        ROOT / "docs/F10_PPO_OBJECTS_V30_FORMULATION.md",
        ROOT / "docs/F12_COMPRESSION_PROTOCOL.md",
        ROOT / "docs/F12_COMPRESSION_RESULTS.md",
        ROOT / "docs/F12_COMPRESSION_ABLATION.md",
        ROOT / "configs/f10_ppo_visual_objects_v30.toml",
        ROOT / "configs/f12_belief_ppo_compression_v1.toml",
        ROOT / "src/duckie_pomdp/control/ppo_environment.py",
        ROOT / "src/duckie_pomdp/control/ppo_protocol.py",
        ROOT / "src/duckie_pomdp/optimization/actor_compression.py",
        ROOT / "experiments/run_f12_compression.py",
    )
    present = {
        str(path.relative_to(ROOT)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in requested if path.exists()
    }
    output = {
        **provenance(config, config_path),
        "classification": verified["classification"],
        "files": present,
        "missing_requested_files": [str(path.relative_to(ROOT)) for path in requested if not path.exists()],
        "root_experiment_plan_present": (ROOT / "EXPERIMENT_PLAN.md").exists(),
        "archived_experiment_plan": str(ROOT / "refine-logs/EXPERIMENT_PLAN.md")
        if (ROOT / "refine-logs/EXPERIMENT_PLAN.md").exists() else None,
        "registered_actor_hashes_verified": True,
        "f15_results_read": False,
    }
    write_json(target, output)
    return output


def prepare_localization_shards(config_path: Path) -> dict[str, Any]:
    """Recover completed episode rows from interrupted pre-shard execution.

    The operation is provenance-preserving: raw CSVs are moved to an integrity
    quarantine, duplicates are recorded, and canonical shard files contain one
    row per completed model/curriculum/seed key.
    """
    config = load_config(config_path)
    verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    localization_dir = root / "localization"
    quarantine = root / "integrity/pre_explicit_shard_raw_csv"
    sources = {
        "matrix": [localization_dir / "matrix_episodes.csv", localization_dir / "matrix_tail_episodes.csv"],
        "pruning": [
            localization_dir / "pruning_only_unique_episodes.csv",
            localization_dir / "prune_distill_unique_episodes.csv",
            localization_dir / "pruning_unique_episodes.csv",
            localization_dir / "pruning_episodes.csv",
        ],
    }
    targets = {
        "matrix_shard_a0_a2.csv": {"A0", "A1", "A2"},
        "matrix_shard_a3_a5.csv": {"A3", "A4", "A5"},
        "matrix_shard_a6_a7.csv": {"A6", "A7"},
        "pruning_shard_p192_p128.csv": {"P192", "P128"},
        "pruning_shard_p96_pd192.csv": {"P96", "PD192"},
        "pruning_shard_pd128_pd96.csv": {"PD128", "PD96"},
    }
    if any((localization_dir / name).exists() for name in targets):
        raise RuntimeError("explicit F15 shard targets already exist")
    all_rows: list[dict[str, Any]] = []
    source_manifest = []
    for family_sources in sources.values():
        for path in family_sources:
            if not path.exists():
                continue
            rows = read_csv(path)
            all_rows.extend(rows)
            source_manifest.append({"path": str(path), "sha256": file_sha256(path), "rows": len(rows)})
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in all_rows:
        model_id = str(row.get("model_id", ""))
        if model_id not in {model for values in targets.values() for model in values}:
            continue
        key = (model_id, str(row["curriculum"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    duplicate_keys = {"|".join(map(str, key)): len(rows) for key, rows in grouped.items() if len(rows) > 1}
    canonical = {key: rows[-1] for key, rows in grouped.items()}
    created = []
    for name, model_ids in targets.items():
        rows = [row for key, row in sorted(canonical.items()) if key[0] in model_ids]
        if rows:
            target = localization_dir / name
            write_csv(target, rows)
            created.append({"path": str(target), "sha256": file_sha256(target), "rows": len(rows)})
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = []
    for family_sources in sources.values():
        for path in family_sources:
            if not path.exists():
                continue
            destination = quarantine / path.name
            if destination.exists():
                raise RuntimeError(f"quarantine destination exists: {destination}")
            path.replace(destination)
            moved.append({"from": str(path), "to": str(destination), "sha256": file_sha256(destination)})
    audit = {
        **provenance(config, config_path),
        "reason": "interrupted preliminary workers entered overlapping matrix evaluation; completed rows were deduplicated before scientific analysis",
        "selection": "last completed row per identical model/curriculum/seed key; trace target is deterministic and shared by repeated attempts",
        "source_files": source_manifest,
        "duplicate_keys": duplicate_keys,
        "created_shards": created,
        "moved_raw_files": moved,
        "scientific_results_read": False,
    }
    audit_path = root / "integrity/pre_explicit_shard_consolidation.json"
    if audit_path.exists():
        raise RuntimeError("F15 pre-shard audit already exists")
    write_json(audit_path, audit)
    return audit


def localization(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protocol(config, config_path)
    paths = frozen_paths(config, config_path)
    root = artifact_root(config, config_path)
    if (root / "localization/failure_localization_decision.json").exists():
        raise RuntimeError("localization is already frozen")
    matrix = verify_registry(paths["ablation_registry"], expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"], collection_key="variants")
    pruning_all = verify_registry(paths["pruning_registry"], expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"], collection_key="candidates")
    pruning = {key: entry for key, entry in pruning_all.items() if key not in {"A0", "P64", "PD64"}}
    # Reuse identical hash actors from matrix evaluation without calling them replicates.
    matrix_result = evaluate_registry(config_path, split="matrix", registry=matrix, seeds=config["seeds"]["localization"])
    pruning_csv = root / "localization/pruning_episodes.csv"
    if not pruning_csv.exists():
        mapping = {"A0": "A0", "A1": "P64", "A2": "PD64"}
        matrix_rows = read_csv(root / "localization/matrix_episodes.csv")
        reused = 0
        for row in matrix_rows:
            if row["model_id"] not in mapping:
                continue
            copied = dict(row)
            copied["model_id"] = mapping[row["model_id"]]
            copied["model_name"] = pruning_all[copied["model_id"]]["name"]
            copied["evidence_reuse"] = "byte_identical_matrix_actor"
            append_csv(pruning_csv, copied)
            reused += 1
        expected = 3 * len(CURRICULA) * len(config["seeds"]["localization"])
        if reused != expected:
            raise RuntimeError(f"expected {expected} hash-identical pruning reuse rows, found {reused}")
        write_json(root / "localization/pruning_reuse_manifest.json", {
            **provenance(config, config_path), "mapping": mapping, "rows_reused": reused,
            "source_matrix_episodes_sha256": file_sha256(root / "localization/matrix_episodes.csv"),
            "checkpoint_hashes_match": all(matrix[source]["sha256"] == pruning_all[target]["sha256"] for source, target in mapping.items()),
            "not_independent_replicates": True,
        })
        unique_csv = root / "localization/pruning_unique_episodes.csv"
        if unique_csv.exists():
            unique_rows = read_csv(unique_csv)
            expected_unique = 6 * len(CURRICULA) * len(config["seeds"]["localization"])
            if len(unique_rows) != expected_unique:
                raise RuntimeError(f"unique pruning worker is incomplete: {len(unique_rows)}/{expected_unique}")
            for row in unique_rows:
                copied = dict(row)
                copied["evidence_reuse"] = ""
                append_csv(pruning_csv, copied)
            write_json(root / "localization/pruning_unique_merge_manifest.json", {
                **provenance(config, config_path), "unique_rows": len(unique_rows),
                "source_csv": str(unique_csv), "source_csv_sha256": file_sha256(unique_csv),
            })
    pruning_result = evaluate_registry(
        config_path, split="pruning",
        registry={"A0": matrix["A0"], **pruning, "P64": pruning_all["P64"], "PD64": pruning_all["PD64"]},
        seeds=config["seeds"]["localization"],
    )
    fidelity = run_same_state_fidelity(config_path)
    return {"matrix": matrix_result, "pruning": pruning_result, "fidelity": fidelity}


def evaluate_localization_shard(
    config_path: Path,
    *,
    family: str,
    model_ids: Sequence[str],
    output_name: str,
) -> dict[str, Any]:
    """Evaluate an explicit localization subset without entering another stage.

    Shards are an execution-only acceleration.  Every shard uses the same frozen
    localization seeds and is consolidated before any scientific result is read.
    """
    config = load_config(config_path)
    verify_protocol(config, config_path)
    root = artifact_root(config, config_path)
    if (root / "localization/failure_localization_decision.json").exists():
        raise RuntimeError("localization is already frozen")
    paths = frozen_paths(config, config_path)
    if family == "matrix":
        registry = verify_registry(
            paths["ablation_registry"],
            expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
            collection_key="variants",
        )
    elif family == "pruning":
        registry = verify_registry(
            paths["pruning_registry"],
            expected_registry_sha256=config["frozen"]["f12_pruning_registry_sha256"],
            collection_key="candidates",
        )
    else:
        raise ValueError("family must be matrix or pruning")
    requested = tuple(str(value) for value in model_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("model IDs must be a non-empty unique list")
    missing = sorted(set(requested) - set(registry))
    if missing:
        raise KeyError(f"unregistered {family} models: {missing}")
    if not output_name.startswith(f"{family}_shard_") or not output_name.endswith(".csv"):
        raise ValueError(f"output name must match {family}_shard_*.csv")
    target = root / "localization" / output_name
    result = evaluate_registry(
        config_path,
        split=output_name.removesuffix(".csv"),
        registry={model_id: registry[model_id] for model_id in requested},
        seeds=config["seeds"]["localization"],
        episode_csv_override=target,
        build_results=False,
    )
    rows = read_csv(target)
    validate_episode_coverage(
        rows,
        model_ids=requested,
        seeds=config["seeds"]["localization"],
    )
    return {**result, "family": family, "models": list(requested), "coverage_valid": True}


def consolidate_localization_shards(config_path: Path, *, family: str) -> dict[str, Any]:
    config = load_config(config_path)
    root = artifact_root(config, config_path)
    directory = root / "localization"
    if family == "matrix":
        model_ids = tuple(f"A{index}" for index in range(8))
        sources = sorted(directory.glob("matrix_shard_*.csv"))
        target = directory / "matrix_episodes.csv"
    elif family == "pruning":
        model_ids = ("P192", "P128", "P96", "PD192", "PD128", "PD96")
        sources = sorted(directory.glob("pruning_shard_*.csv"))
        target = directory / "pruning_unique_episodes.csv"
    else:
        raise ValueError("family must be matrix or pruning")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite consolidated F15 episodes: {target}")
    if not sources:
        raise RuntimeError(f"no {family} shards found")
    rows: list[dict[str, Any]] = []
    manifest_sources = []
    for source in sources:
        source_rows = read_csv(source)
        rows.extend(source_rows)
        manifest_sources.append({"path": str(source), "sha256": file_sha256(source), "rows": len(source_rows)})
    validate_episode_coverage(rows, model_ids=model_ids, seeds=config["seeds"]["localization"])
    for row in rows:
        trace = Path(row["trace_path"])
        evaluation_only = Path(row["evaluation_only_path"])
        if not trace.exists() or file_sha256(trace) != row["trace_sha256"]:
            raise RuntimeError(f"F15 trace provenance mismatch: {trace}")
        if not evaluation_only.exists() or file_sha256(evaluation_only) != row["evaluation_only_sha256"]:
            raise RuntimeError(f"F15 evaluation-only provenance mismatch: {evaluation_only}")
    rows.sort(key=lambda row: (model_ids.index(row["model_id"]), CURRICULA.index(row["curriculum"]), int(row["seed"])))
    write_csv(target, rows)
    manifest = {
        **provenance(config, config_path), "family": family, "sources": manifest_sources,
        "target": str(target), "target_sha256": file_sha256(target), "rows": len(rows),
        "model_ids": list(model_ids), "coverage_valid": True,
    }
    manifest_path = directory / f"{family}_shard_consolidation_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite shard manifest: {manifest_path}")
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "initialize", "write-audit", "prepare-shards", "localization", "evaluate-shard", "consolidate-shards", "freeze-localization"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--family", choices=("matrix", "pruning"))
    parser.add_argument("--models")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.command == "verify":
        result = verify_command(config_path)
    elif args.command == "initialize":
        result = initialize_manifests(config_path)
    elif args.command == "write-audit":
        result = write_repository_audit(config_path)
    elif args.command == "prepare-shards":
        result = prepare_localization_shards(config_path)
    elif args.command == "localization":
        result = localization(config_path)
    elif args.command == "evaluate-shard":
        if args.family is None or args.models is None or args.output_name is None:
            parser.error("evaluate-shard requires --family, --models, and --output-name")
        result = evaluate_localization_shard(
            config_path,
            family=args.family,
            model_ids=tuple(value.strip() for value in args.models.split(",") if value.strip()),
            output_name=args.output_name,
        )
    elif args.command == "consolidate-shards":
        if args.family is None:
            parser.error("consolidate-shards requires --family")
        result = consolidate_localization_shards(config_path, family=args.family)
    else:
        result = freeze_localization(config_path)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
