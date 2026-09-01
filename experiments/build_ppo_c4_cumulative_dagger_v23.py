"""Build C2/C3 policy-rehearsal plus C4 teacher/DAgger data from public belief."""

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
    stage: str,
    seeds: tuple[int, ...],
    target_policy: Callable[[np.ndarray], np.ndarray],
    behavior_policy: Callable[[np.ndarray], np.ndarray],
    dimension: int,
) -> Samples:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    episodes: list[dict[str, object]] = []
    duckie_detected_steps = 0
    stop_detected_steps = 0
    environment = PPOCurriculumEnvironment(config, stage=stage, split="training")
    try:
        horizon = environment.protocol.stage(stage).episode_horizon_steps
        for seed in seeds:
            observation, _ = environment.reset(seed=seed)
            episode_return = 0.0
            stop_completed = False
            restarted = False
            for step in range(1, horizon + 1):
                target = np.asarray(target_policy(observation), dtype=np.float32)
                action = np.asarray(behavior_policy(observation), dtype=np.float32)
                if observation.shape != (dimension,) or target.shape != (2,) or action.shape != (2,):
                    raise RuntimeError("cumulative DAgger runtime contract changed")
                if not all(np.all(np.isfinite(value)) for value in (observation, target, action)):
                    raise RuntimeError("non-finite cumulative DAgger sample")
                if np.any(action < -1.0) or np.any(action > 1.0):
                    raise RuntimeError("behavior action escaped normalized bounds")
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
                            "stage": stage,
                            "seed": seed,
                            "steps": step,
                            "return": episode_return,
                            "completed": bool(info.get("completed", False)),
                            "stop_completed": stop_completed,
                            "stop_violation": bool(info.get("stop_violation", False)),
                            "restarted": restarted,
                            "collision": bool(info.get("collision", False)),
                            "unsafe": bool(info.get("pedestrian_safety_event", False)),
                            "invalid_pose": bool(info.get("invalid_pose", False)),
                            "termination_reason": info.get("termination_reason"),
                            "truncation_reason": info.get("truncation_reason"),
                        }
                    )
                    break
            else:  # pragma: no cover
                raise RuntimeError(f"{stage} episode {seed} did not terminate")
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


def _role_weights(rows: int, mass: float) -> np.ndarray:
    if rows <= 0 or not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("role rows and mass must be positive")
    return np.full(rows, mass / rows, dtype=np.float32)


def build(
    config: Path,
    source_checkpoint: Path,
    source_sha256: str,
    learner_checkpoint: Path,
    learner_sha256: str,
    output: Path,
    source_csv: Path,
    manifest_path: Path,
    rehearsal_seed_count: int,
    c4_seed_count: int,
    device: str,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    c2_seeds = protocol.stage("c2").training_seeds[:rehearsal_seed_count]
    c3_seeds = protocol.stage("c3").training_seeds[:rehearsal_seed_count]
    c4_training_seeds = protocol.stage("c4").training_seeds
    c4_seeds = c4_training_seeds[:c4_seed_count]
    if (
        len(c2_seeds) != rehearsal_seed_count
        or len(c3_seeds) != rehearsal_seed_count
        or len(c4_seeds) != c4_seed_count
    ):
        raise ValueError("requested seed count exceeds frozen C4 training split")
    if file_sha256(source_checkpoint) != source_sha256:
        raise RuntimeError("source C3 checkpoint hash mismatch")
    if file_sha256(learner_checkpoint) != learner_sha256:
        raise RuntimeError("C4 learner checkpoint hash mismatch")
    for destination in (output, source_csv, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    source, source_payload = PPOAgent.load(source_checkpoint, device=device)
    learner, learner_payload = PPOAgent.load(learner_checkpoint, device=device)
    if source_payload.get("stage") != "c3" or learner_payload.get("stage") != "c4":
        raise RuntimeError("unexpected curriculum checkpoint stage")
    dimension = len(protocol.observation_order)
    source_action = lambda obs: source.act(obs, deterministic=True).environment_action
    learner_action = lambda obs: learner.act(obs, deterministic=True).environment_action
    teacher = BeliefAwareSimpleController(protocol)
    teacher_action = lambda obs: teacher.act(obs)

    c2 = _collect(config, "c2", c2_seeds, source_action, source_action, dimension)
    c3 = _collect(config, "c3", c3_seeds, source_action, source_action, dimension)
    c4_teacher = _collect(config, "c4", c4_seeds, teacher_action, teacher_action, dimension)
    c4_dagger = _collect(config, "c4", c4_seeds, teacher_action, learner_action, dimension)
    if not all(not row["collision"] and not row["unsafe"] for row in c2.episodes):
        raise RuntimeError("source policy did not retain safe C2 behavior on rehearsal seeds")
    if not all(
        row["completed"] and row["stop_completed"] and row["restarted"]
        and not row["stop_violation"] and not row["collision"] and not row["invalid_pose"]
        for row in c3.episodes
    ):
        raise RuntimeError("source policy did not retain C3 behavior on rehearsal seeds")
    if not all(
        row["completed"] and row["stop_completed"] and row["restarted"]
        and not row["stop_violation"] and not row["collision"] and not row["invalid_pose"]
        for row in c4_teacher.episodes
    ):
        raise RuntimeError("public-belief C4 teacher failed a source episode")

    roles = (
        ("c2_source_policy_rehearsal", c2, 2.0),
        ("c3_source_policy_rehearsal", c3, 2.0),
        ("c4_teacher_trajectory", c4_teacher, 1.0),
        ("c4_dagger_learner_state", c4_dagger, 1.0),
    )
    base_mass = float(max(len(samples.observations) for _, samples, _ in roles))
    weights = tuple(_role_weights(len(samples.observations), base_mass * multiplier) for _, samples, multiplier in roles)
    x = np.concatenate(tuple(samples.observations for _, samples, _ in roles))
    y = np.concatenate(tuple(samples.targets for _, samples, _ in roles))
    w = np.concatenate(weights)
    np.savez_compressed(output, observations=x, actions=y, weights=w)

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
        for (role, samples, _), role_weights in zip(roles, weights, strict=True):
            provenance = [
                (episode["seed"], step)
                for episode in samples.episodes
                for step in range(1, int(episode["steps"]) + 1)
            ]
            if len(provenance) != len(samples.observations):
                raise RuntimeError("source provenance length mismatch")
            for index, observation in enumerate(samples.observations):
                seed, step = provenance[index]
                row = {
                    "source_role": role,
                    "source_seed": seed,
                    "source_step": step,
                    "target_action_normalized.linear": float(samples.targets[index, 0]),
                    "target_action_normalized.angular": float(samples.targets[index, 1]),
                    "executed_action_normalized.linear": float(samples.executed[index, 0]),
                    "executed_action_normalized.angular": float(samples.executed[index, 1]),
                    "sample_weight": float(role_weights[index]),
                }
                row.update({
                    f"policy_normalized.{name}": float(observation[column])
                    for column, name in enumerate(protocol.observation_order)
                })
                writer.writerow(row)

    role_rows = {role: len(samples.observations) for role, samples, _ in roles}
    role_masses = {role: float(np.sum(role_weights)) for (role, _, _), role_weights in zip(roles, weights, strict=True)}
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "c2_c3_policy_rehearsal_c4_teacher_dagger_v2",
        "source_config": str(config.resolve()),
        "source_config_sha256": file_sha256(config),
        "builder_sha256": file_sha256(Path(__file__)),
        "training_seeds_only": sorted(set(c2_seeds + c3_seeds + c4_seeds)),
        "c2_rehearsal_seeds": list(c2_seeds),
        "c3_rehearsal_seeds": list(c3_seeds),
        "c4_seeds": list(c4_seeds),
        "uses_evaluation_gt": False,
        "runtime_source": "RGB -> lane belief + YOLO/F9c pedestrian/stop belief -> 29D policy observation",
        "rows": int(len(x)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "role_rows": role_rows,
        "role_weight_masses": role_masses,
        "role_mass_multipliers": {role: multiplier for role, _, multiplier in roles},
        "c2_source_episodes": list(c2.episodes),
        "c3_source_episodes": list(c3.episodes),
        "c4_teacher_episodes": list(c4_teacher.episodes),
        "c4_dagger_episodes": list(c4_dagger.episodes),
        "duckie_detected_steps": int(c2.duckie_detected_steps + c4_teacher.duckie_detected_steps + c4_dagger.duckie_detected_steps),
        "stop_detected_steps": int(c3.stop_detected_steps + c4_teacher.stop_detected_steps + c4_dagger.stop_detected_steps),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_stage": source_payload["stage"],
        "learner_checkpoint": str(learner_checkpoint.resolve()),
        "learner_checkpoint_sha256": learner_sha256,
        "learner_checkpoint_stage": learner_payload["stage"],
        "learner_checkpoint_step": int(learner_payload["global_step"]),
        "source_csv": str(source_csv.resolve()),
        "dataset": str(output.resolve()),
    }
    payload["source_csv_sha256"] = file_sha256(source_csv)
    payload["dataset_sha256"] = file_sha256(output)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--learner-checkpoint", type=Path, required=True)
    parser.add_argument("--learner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rehearsal-seed-count", type=int, default=4)
    parser.add_argument("--c4-seed-count", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build(
        args.config.resolve(), args.source_checkpoint.resolve(), args.source_sha256,
        args.learner_checkpoint.resolve(), args.learner_sha256, args.output.resolve(),
        args.source_csv.resolve(), args.manifest.resolve(), args.rehearsal_seed_count,
        args.c4_seed_count, args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
