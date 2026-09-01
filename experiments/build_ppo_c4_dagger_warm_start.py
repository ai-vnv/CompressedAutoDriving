"""Build a cumulative C4 DAgger dataset from the public 29D belief only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


@dataclass(frozen=True)
class Samples:
    observations: np.ndarray
    targets: np.ndarray
    executed: np.ndarray
    episodes: tuple[dict[str, object], ...]
    duckie_detected_steps: int
    stop_detected_steps: int


def _collect(
    config: Path,
    seeds: tuple[int, ...],
    teacher: BeliefAwareSimpleController,
    behavior: Callable[[np.ndarray], np.ndarray],
    dimension: int,
) -> Samples:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    episodes: list[dict[str, object]] = []
    duckie_detected_steps = 0
    stop_detected_steps = 0
    environment = PPOCurriculumEnvironment(config, stage="c4", split="training")
    try:
        horizon = environment.protocol.stage("c4").episode_horizon_steps
        for seed in seeds:
            observation, _ = environment.reset(seed=seed)
            teacher.reset(seed)
            episode_return = 0.0
            stop_completed = False
            restarted = False
            for step in range(1, horizon + 1):
                target = np.asarray(teacher.act(observation), dtype=np.float32)
                action = np.asarray(behavior(observation), dtype=np.float32)
                if observation.shape != (dimension,) or target.shape != (2,) or action.shape != (2,):
                    raise RuntimeError("C4 DAgger runtime contract changed")
                if not all(np.all(np.isfinite(value)) for value in (observation, target, action)):
                    raise RuntimeError("non-finite C4 DAgger sample")
                if np.any(action < -1.0) or np.any(action > 1.0):
                    raise RuntimeError("C4 learner action escaped normalized bounds")
                observations.append(observation.astype(np.float32, copy=True))
                targets.append(target.copy())
                executed.append(action.copy())
                observation, reward, terminated, truncated, info = environment.step(action)
                episode_return += float(reward)
                perception = dict(info.get("perception", {}))
                duckie_detected_steps += int(int(perception.get("duckie_detection_count", 0)) > 0)
                stop_detected_steps += int(int(perception.get("stop_sign_detection_count", 0)) > 0)
                stop_completed = stop_completed or bool(info.get("stop_completed", False))
                restarted = restarted or (stop_completed and float(info.get("v_cmd", 0.0)) > 0.05)
                if terminated or truncated:
                    episodes.append(
                        {
                            "seed": seed,
                            "steps": step,
                            "return": episode_return,
                            "completed": bool(info.get("completed", False)),
                            "stop_completed": stop_completed,
                            "stop_violation": bool(info.get("stop_violation", False)),
                            "restarted": restarted,
                            "collision": bool(info.get("collision", False)),
                            "invalid_pose": bool(info.get("invalid_pose", False)),
                            "termination_reason": info.get("termination_reason"),
                            "truncation_reason": info.get("truncation_reason"),
                        }
                    )
                    break
            else:  # pragma: no cover
                raise RuntimeError(f"C4 DAgger episode {seed} did not terminate")
    finally:
        environment.close()
    return Samples(
        observations=np.asarray(observations, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        executed=np.asarray(executed, dtype=np.float32),
        episodes=tuple(episodes),
        duckie_detected_steps=duckie_detected_steps,
        stop_detected_steps=stop_detected_steps,
    )


def _weights(
    anchor_rows: int,
    targets: np.ndarray,
    dagger_rows: int,
    anchor_weight_multiplier: float,
) -> tuple[np.ndarray, ...]:
    if not np.isfinite(anchor_weight_multiplier) or anchor_weight_multiplier <= 0.0:
        raise ValueError("anchor weight multiplier must be positive")
    mass = float(max(anchor_rows, len(targets), dagger_rows))
    anchor = np.full(
        anchor_rows,
        anchor_weight_multiplier * mass / anchor_rows,
        dtype=np.float32,
    )
    teacher = np.ones(len(targets), dtype=np.float32)
    teacher[targets[:, 0] <= -0.99] = 16.0
    teacher[(targets[:, 0] > -0.99) & (targets[:, 0] <= -0.50)] = 4.0
    teacher *= np.float32(mass / float(np.sum(teacher)))
    dagger = np.full(dagger_rows, mass / dagger_rows, dtype=np.float32)
    return anchor, teacher, dagger


def build(
    config: Path,
    learner_checkpoint: Path,
    learner_sha256: str,
    anchor_dataset: Path,
    anchor_sha256: str,
    output: Path,
    source_csv: Path,
    manifest_path: Path,
    seed_count: int,
    device: str,
    anchor_weight_multiplier: float,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    seeds = protocol.stage("c4").training_seeds[:seed_count]
    if seed_count <= 0 or len(seeds) != seed_count:
        raise ValueError("seed_count exceeds the frozen C4 training split")
    if file_sha256(learner_checkpoint) != learner_sha256:
        raise RuntimeError("frozen C4 learner checkpoint hash mismatch")
    if file_sha256(anchor_dataset) != anchor_sha256:
        raise RuntimeError("frozen cumulative anchor dataset hash mismatch")
    for destination in (output, source_csv, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    learner, learner_payload = PPOAgent.load(learner_checkpoint, device=device)
    if learner_payload.get("stage") != "c4" or int(learner_payload.get("global_step", -1)) < 1024:
        raise RuntimeError("DAgger source must be an updated failed C4 learner")
    teacher = BeliefAwareSimpleController(protocol)
    teacher_samples = _collect(config, seeds, teacher, lambda obs: teacher.act(obs), len(protocol.observation_order))
    if not teacher_samples.episodes or not all(
        row["completed"] and row["stop_completed"] and row["restarted"]
        and not row["stop_violation"] and not row["collision"] and not row["invalid_pose"]
        for row in teacher_samples.episodes
    ):
        raise RuntimeError("public-belief C4 teacher did not pass every source episode")
    dagger_samples = _collect(
        config,
        seeds,
        teacher,
        lambda obs: learner.act(obs, deterministic=True).environment_action,
        len(protocol.observation_order),
    )

    with np.load(anchor_dataset) as data:
        anchor_all_x = np.asarray(data["observations"], dtype=np.float32)
        anchor_all_y = np.asarray(data["actions"], dtype=np.float32)
    anchor_rows = max(len(teacher_samples.observations), len(dagger_samples.observations))
    if len(anchor_all_x) < anchor_rows:
        raise RuntimeError("cumulative anchor dataset is too small")
    indices = np.linspace(0, len(anchor_all_x) - 1, anchor_rows, dtype=np.int64)
    if len(np.unique(indices)) != anchor_rows:
        raise RuntimeError("cumulative anchor sampling duplicated rows")
    anchor_x, anchor_y = anchor_all_x[indices], anchor_all_y[indices]
    anchor_w, teacher_w, dagger_w = _weights(
        anchor_rows,
        teacher_samples.targets,
        len(dagger_samples.observations),
        anchor_weight_multiplier,
    )
    x = np.concatenate((anchor_x, teacher_samples.observations, dagger_samples.observations))
    y = np.concatenate((anchor_y, teacher_samples.targets, dagger_samples.targets))
    weights = np.concatenate((anchor_w, teacher_w, dagger_w))
    np.savez_compressed(output, observations=x, actions=y, weights=weights)

    fields = (
        "source_role", "source_seed", "source_step",
        *(f"policy_normalized.{name}" for name in protocol.observation_order),
        "target_action_normalized.linear", "target_action_normalized.angular",
        "executed_action_normalized.linear", "executed_action_normalized.angular",
        "sample_weight",
    )
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def emit(role: str, obs: np.ndarray, target: np.ndarray, action: np.ndarray, group_w: np.ndarray, episode_rows):
            provenance = [("", int(index)) for index in indices] if episode_rows is None else [
                (episode["seed"], step)
                for episode in episode_rows
                for step in range(1, int(episode["steps"]) + 1)
            ]
            if len(provenance) != len(obs):
                raise RuntimeError("C4 source provenance length mismatch")
            for index, vector in enumerate(obs):
                seed, step = provenance[index]
                row = {
                    "source_role": role, "source_seed": seed, "source_step": step,
                    "target_action_normalized.linear": float(target[index, 0]),
                    "target_action_normalized.angular": float(target[index, 1]),
                    "executed_action_normalized.linear": float(action[index, 0]),
                    "executed_action_normalized.angular": float(action[index, 1]),
                    "sample_weight": float(group_w[index]),
                }
                row.update({f"policy_normalized.{name}": float(vector[column]) for column, name in enumerate(protocol.observation_order)})
                writer.writerow(row)

        emit("cumulative_c3_anchor", anchor_x, anchor_y, anchor_y, anchor_w, None)
        emit("c4_teacher_trajectory", teacher_samples.observations, teacher_samples.targets, teacher_samples.executed, teacher_w, teacher_samples.episodes)
        emit("c4_dagger_learner_state", dagger_samples.observations, dagger_samples.targets, dagger_samples.executed, dagger_w, dagger_samples.episodes)

    imported = dict(protocol.raw["curriculum_import"]["c3"])
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "balanced_cumulative_c3_c4_teacher_and_dagger_v1",
        "source_config": str(config.resolve()),
        "source_config_sha256": file_sha256(config),
        "builder_sha256": file_sha256(Path(__file__)),
        "training_seeds_only": list(seeds),
        "uses_evaluation_gt": False,
        "runtime_source": "RGB -> lane belief + YOLO/F9c pedestrian and stop belief -> 29D policy observation; teacher labels use only that vector",
        "rows": int(len(x)), "anchor_rows": int(len(anchor_x)),
        "teacher_rows": int(len(teacher_samples.observations)), "dagger_rows": int(len(dagger_samples.observations)),
        "observation_dimension": int(x.shape[1]), "action_dimension": 2,
        "stop_action_rows": int(np.sum(teacher_samples.targets[:, 0] <= -0.99)),
        "dagger_teacher_drive_rows": int(np.sum(dagger_samples.targets[:, 0] > -0.50)),
        "duckie_detected_steps": int(teacher_samples.duckie_detected_steps + dagger_samples.duckie_detected_steps),
        "stop_detected_steps": int(teacher_samples.stop_detected_steps + dagger_samples.stop_detected_steps),
        "anchor_weight_mass": float(np.sum(anchor_w)), "teacher_weight_mass": float(np.sum(teacher_w)), "dagger_weight_mass": float(np.sum(dagger_w)),
        "anchor_weight_multiplier": float(anchor_weight_multiplier),
        "source_episodes": list(teacher_samples.episodes), "dagger_source_episodes": list(dagger_samples.episodes),
        "source_checkpoint_sha256": str(imported["selected_checkpoint_sha256"]),
        "learner_checkpoint": str(learner_checkpoint.resolve()), "learner_checkpoint_sha256": learner_sha256,
        "learner_checkpoint_stage": learner_payload["stage"], "learner_checkpoint_step": int(learner_payload["global_step"]),
        "anchor_dataset": str(anchor_dataset.resolve()), "anchor_dataset_sha256": anchor_sha256,
        "source_csv": str(source_csv.resolve()), "dataset": str(output.resolve()),
    }
    payload["source_csv_sha256"] = file_sha256(source_csv)
    payload["dataset_sha256"] = file_sha256(output)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--learner-checkpoint", type=Path, required=True)
    parser.add_argument("--learner-sha256", required=True)
    parser.add_argument("--anchor-dataset", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-weight-multiplier", type=float, default=3.0)
    args = parser.parse_args()
    result = build(
        args.config.resolve(), args.learner_checkpoint.resolve(), args.learner_sha256,
        args.anchor_dataset.resolve(), args.anchor_sha256, args.output.resolve(),
        args.source_csv.resolve(), args.manifest.resolve(), args.seed_count, args.device,
        args.anchor_weight_multiplier,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
