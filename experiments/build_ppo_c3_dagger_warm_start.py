"""Build a C3 DAgger warm-start dataset from public runtime observations.

The dataset combines three equally weighted roles:

* frozen C2 retention anchors;
* successful C3 teacher trajectories;
* states visited by the failed C3 learner, relabelled by the same public-belief
  teacher.

The learner is executed to expose its own state distribution.  The teacher
label is computed from the 29D policy observation only.  Simulator truth is
used neither as an input nor as a target.
"""

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
class RolloutSamples:
    observations: np.ndarray
    targets: np.ndarray
    executed_actions: np.ndarray
    episodes: tuple[dict[str, object], ...]
    detected_steps: int


def _collect(
    *,
    config: Path,
    seeds: tuple[int, ...],
    target_policy: BeliefAwareSimpleController,
    behavior_policy: Callable[[np.ndarray], np.ndarray],
    observation_dimension: int,
) -> RolloutSamples:
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    episodes: list[dict[str, object]] = []
    detected_steps = 0
    environment = PPOCurriculumEnvironment(config, stage="c3", split="training")
    try:
        horizon = environment.protocol.stage("c3").episode_horizon_steps
        for seed in seeds:
            observation, _ = environment.reset(seed=seed)
            target_policy.reset(seed)
            episode_return = 0.0
            stop_completed = False
            restarted = False
            last_info: dict[str, object] | None = None
            for step in range(1, horizon + 1):
                target = np.asarray(target_policy.act(observation), dtype=np.float32)
                executed = np.asarray(behavior_policy(observation), dtype=np.float32)
                if observation.shape != (observation_dimension,):
                    raise RuntimeError("C3 runtime observation dimension changed")
                if target.shape != (2,) or executed.shape != (2,):
                    raise RuntimeError("C3 behavior action dimension changed")
                if not all(
                    np.all(np.isfinite(value))
                    for value in (observation, target, executed)
                ):
                    raise RuntimeError("non-finite public DAgger sample")
                if np.any(executed < -1.0) or np.any(executed > 1.0):
                    raise RuntimeError("learner action escaped normalized bounds")
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                targets.append(target.copy())
                executed_actions.append(executed.copy())
                observation, reward, terminated, truncated, info = environment.step(
                    executed
                )
                last_info = info
                episode_return += float(reward)
                detected_steps += int(
                    int(info.get("perception", {}).get("stop_sign_detection_count", 0))
                    > 0
                )
                stop_completed = stop_completed or bool(info.get("stop_completed", False))
                restarted = restarted or (
                    stop_completed and float(info.get("v_cmd", 0.0)) > 0.05
                )
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
            else:  # pragma: no cover - the environment owns horizon truncation
                raise RuntimeError(f"C3 DAgger episode {seed} did not terminate")
            if last_info is None:
                raise RuntimeError("C3 DAgger rollout produced no transition")
    finally:
        environment.close()
    return RolloutSamples(
        observations=np.asarray(observations, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        episodes=tuple(episodes),
        detected_steps=detected_steps,
    )


def _equal_mass_weights(
    anchor_rows: int,
    teacher_targets: np.ndarray,
    dagger_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_mass = float(max(anchor_rows, len(teacher_targets), dagger_rows))
    anchor = np.full(anchor_rows, common_mass / anchor_rows, dtype=np.float32)
    teacher = np.ones(len(teacher_targets), dtype=np.float32)
    teacher[teacher_targets[:, 0] <= -0.99] = 32.0
    teacher[(teacher_targets[:, 0] > -0.99) & (teacher_targets[:, 0] <= -0.50)] = 4.0
    teacher *= np.float32(common_mass / float(np.sum(teacher)))
    dagger = np.full(dagger_rows, common_mass / dagger_rows, dtype=np.float32)
    return anchor, teacher, dagger


def build(
    *,
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
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    stage = protocol.stage("c3")
    seeds = stage.training_seeds[:seed_count]
    if seed_count <= 0 or len(seeds) != seed_count:
        raise ValueError("seed_count exceeds the frozen C3 training split")
    if file_sha256(learner_checkpoint) != learner_sha256:
        raise RuntimeError("frozen C3 learner checkpoint hash mismatch")
    if file_sha256(anchor_dataset) != anchor_sha256:
        raise RuntimeError("frozen C2 anchor dataset hash mismatch")
    for destination in (output, source_csv, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    learner, learner_payload = PPOAgent.load(learner_checkpoint, device=device)
    if learner_payload.get("stage") != "c3" or int(learner_payload.get("global_step", -1)) != 0:
        raise RuntimeError("DAgger source must be the failed C3 step-zero actor")
    if learner.config.observation_dimension != len(protocol.observation_order):
        raise RuntimeError("learner observation contract changed")

    teacher = BeliefAwareSimpleController(protocol)
    teacher_samples = _collect(
        config=config,
        seeds=seeds,
        target_policy=teacher,
        behavior_policy=lambda observation: teacher.act(observation),
        observation_dimension=len(protocol.observation_order),
    )
    if not teacher_samples.episodes or not all(
        row["completed"]
        and row["stop_completed"]
        and row["restarted"]
        and not row["stop_violation"]
        and not row["collision"]
        and not row["invalid_pose"]
        for row in teacher_samples.episodes
    ):
        raise RuntimeError("public-belief C3 teacher did not pass every source episode")

    dagger_samples = _collect(
        config=config,
        seeds=seeds,
        target_policy=teacher,
        behavior_policy=lambda observation: learner.act(
            observation, deterministic=True
        ).environment_action,
        observation_dimension=len(protocol.observation_order),
    )

    with np.load(anchor_dataset) as data:
        anchor_x_all = np.asarray(data["observations"], dtype=np.float32)
        anchor_y_all = np.asarray(data["actions"], dtype=np.float32)
    anchor_rows = max(len(teacher_samples.observations), len(dagger_samples.observations))
    if len(anchor_x_all) < anchor_rows:
        raise RuntimeError("C2 anchor dataset is too small for DAgger balancing")
    anchor_indices = np.linspace(0, len(anchor_x_all) - 1, anchor_rows, dtype=np.int64)
    if len(np.unique(anchor_indices)) != anchor_rows:
        raise RuntimeError("C2 anchor sampling must not duplicate rows")
    anchor_x = anchor_x_all[anchor_indices]
    anchor_y = anchor_y_all[anchor_indices]
    anchor_w, teacher_w, dagger_w = _equal_mass_weights(
        anchor_rows, teacher_samples.targets, len(dagger_samples.observations)
    )

    x = np.concatenate(
        (anchor_x, teacher_samples.observations, dagger_samples.observations), axis=0
    )
    y = np.concatenate(
        (anchor_y, teacher_samples.targets, dagger_samples.targets), axis=0
    )
    weights = np.concatenate((anchor_w, teacher_w, dagger_w), axis=0)
    np.savez_compressed(output, observations=x, actions=y, weights=weights)

    fields = (
        "source_role",
        "source_seed",
        "source_step",
        *(f"policy_normalized.{name}" for name in protocol.observation_order),
        "target_action_normalized.linear",
        "target_action_normalized.angular",
        "executed_action_normalized.linear",
        "executed_action_normalized.angular",
        "sample_weight",
    )
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def write_group(
            role: str,
            observations: np.ndarray,
            targets: np.ndarray,
            executed: np.ndarray,
            weights_for_group: np.ndarray,
            episode_rows: tuple[dict[str, object], ...] | None,
        ) -> None:
            seed_steps: list[tuple[object, int]] = []
            if episode_rows is None:
                seed_steps = [("", int(index)) for index in anchor_indices]
            else:
                for episode in episode_rows:
                    seed_steps.extend(
                        (episode["seed"], step)
                        for step in range(1, int(episode["steps"]) + 1)
                    )
            if len(seed_steps) != len(observations):
                raise RuntimeError("source row provenance length mismatch")
            for index, observation in enumerate(observations):
                seed, step = seed_steps[index]
                row = {
                    "source_role": role,
                    "source_seed": seed,
                    "source_step": step,
                    "target_action_normalized.linear": float(targets[index, 0]),
                    "target_action_normalized.angular": float(targets[index, 1]),
                    "executed_action_normalized.linear": float(executed[index, 0]),
                    "executed_action_normalized.angular": float(executed[index, 1]),
                    "sample_weight": float(weights_for_group[index]),
                }
                row.update(
                    {
                        f"policy_normalized.{name}": float(observation[column])
                        for column, name in enumerate(protocol.observation_order)
                    }
                )
                writer.writerow(row)

        write_group(
            "c2_retention_anchor",
            anchor_x,
            anchor_y,
            anchor_y,
            anchor_w,
            None,
        )
        write_group(
            "c3_teacher_trajectory",
            teacher_samples.observations,
            teacher_samples.targets,
            teacher_samples.executed_actions,
            teacher_w,
            teacher_samples.episodes,
        )
        write_group(
            "c3_dagger_learner_state",
            dagger_samples.observations,
            dagger_samples.targets,
            dagger_samples.executed_actions,
            dagger_w,
            dagger_samples.episodes,
        )

    previous_linear = protocol.observation_order.index(
        "previous_linear_velocity_cmd_mps"
    )
    stop_satisfied = protocol.observation_order.index("stop_mode_satisfied")
    imported_c2 = dict(protocol.raw["curriculum_import"]["c2"])
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "balanced_c2_retention_c3_teacher_and_dagger_v1",
        "source_config": str(config.resolve()),
        "source_config_sha256": file_sha256(config),
        "source_scenario_config_sha256": file_sha256(stage.scenario_config_path),
        "builder_sha256": file_sha256(Path(__file__)),
        "training_seeds_only": list(seeds),
        "uses_evaluation_gt": False,
        "runtime_source": (
            "RGB -> YOLO stop_sign -> metric projection -> stop belief -> "
            "29D policy observation; teacher labels use the 29D observation only"
        ),
        "rows": int(len(x)),
        "anchor_rows": int(len(anchor_x)),
        "teacher_rows": int(len(teacher_samples.observations)),
        "dagger_rows": int(len(dagger_samples.observations)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "stop_action_rows": int(np.sum(teacher_samples.targets[:, 0] <= -0.99)),
        "satisfied_observation_rows": int(
            np.sum(teacher_samples.observations[:, stop_satisfied] > 0.5)
        ),
        "dagger_stalled_observation_rows": int(
            np.sum(dagger_samples.observations[:, previous_linear] <= 0.10)
        ),
        "dagger_teacher_drive_rows": int(
            np.sum(dagger_samples.targets[:, 0] > -0.50)
        ),
        "detected_steps": int(
            teacher_samples.detected_steps + dagger_samples.detected_steps
        ),
        "anchor_weight_mass": float(np.sum(anchor_w)),
        "teacher_weight_mass": float(np.sum(teacher_w)),
        "dagger_weight_mass": float(np.sum(dagger_w)),
        "source_episodes": list(teacher_samples.episodes),
        "dagger_source_episodes": list(dagger_samples.episodes),
        "source_checkpoint_sha256": str(
            imported_c2["selected_checkpoint_sha256"]
        ),
        "learner_checkpoint": str(learner_checkpoint.resolve()),
        "learner_checkpoint_sha256": learner_sha256,
        "learner_checkpoint_stage": learner_payload["stage"],
        "learner_checkpoint_step": int(learner_payload["global_step"]),
        "anchor_dataset": str(anchor_dataset.resolve()),
        "anchor_dataset_sha256": anchor_sha256,
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": file_sha256(source_csv),
        "dataset": str(output.resolve()),
        "dataset_sha256": file_sha256(output),
    }
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
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                config=args.config.resolve(),
                learner_checkpoint=args.learner_checkpoint.resolve(),
                learner_sha256=args.learner_sha256,
                anchor_dataset=args.anchor_dataset.resolve(),
                anchor_sha256=args.anchor_sha256,
                output=args.output.resolve(),
                source_csv=args.source_csv.resolve(),
                manifest_path=args.manifest.resolve(),
                seed_count=args.seed_count,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
