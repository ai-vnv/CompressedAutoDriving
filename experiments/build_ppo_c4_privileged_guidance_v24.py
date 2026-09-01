"""Add one privileged-teacher C4 episode to the public-belief warm start."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.privileged_c4_teacher import PrivilegedC4Teacher


ROLE_MULTIPLIERS = {
    "c2_source_policy_rehearsal": 4.0,
    "c3_source_policy_rehearsal": 4.0,
    "c4_teacher_trajectory": 1.0,
    "c4_dagger_learner_state": 1.0,
    "c4_privileged_guided_episode": 1.0,
}


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def _guided_episode(config: Path, seed: int) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    teacher = PrivilegedC4Teacher()
    environment = PPOCurriculumEnvironment(
        config, stage="c4", split="training", seeds=(seed,)
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    truths: list[dict[str, object]] = []
    last_info: dict[str, object] | None = None
    stop_completed = False
    restarted = False
    try:
        observation, info = environment.reset(seed=seed)
        for step in range(1, protocol.stage("c4").episode_horizon_steps + 1):
            action = teacher.act(info)
            truth = dict(info["evaluation_gt"])
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            actions.append(action.copy())
            truths.append(truth)
            observation, reward, terminated, truncated, info = environment.step(action)
            rewards.append(float(reward))
            last_info = info
            stop_completed = stop_completed or bool(info["stop_completed"])
            restarted = restarted or (
                stop_completed and float(info["v_cmd"]) > 0.08
            )
            if terminated or truncated:
                break
    finally:
        environment.close()
    if last_info is None:
        raise RuntimeError("privileged guided episode produced no transition")
    passed = bool(
        last_info["completed"]
        and stop_completed
        and restarted
        and not last_info["stop_violation"]
        and not last_info["collision"]
        and not last_info["lane_failure"]
        and not last_info["invalid_pose"]
    )
    if not passed:
        raise RuntimeError(f"privileged C4 teacher failed seed {seed}: {last_info}")
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "returns": _discounted_returns(rewards, protocol.ppo.gamma),
        "truths": truths,
        "episode": {
            "seed": seed,
            "steps": len(rewards),
            "return": float(sum(rewards)),
            "completed": bool(last_info["completed"]),
            "stop_completed": stop_completed,
            "stop_violation": bool(last_info["stop_violation"]),
            "restarted": restarted,
            "collision": bool(last_info["collision"]),
            "unsafe": bool(last_info["unsafe_proximity"]),
            "lane_failure": bool(last_info["lane_failure"]),
            "invalid_pose": bool(last_info["invalid_pose"]),
            "termination_reason": last_info["termination_reason"],
            "truncation_reason": last_info["truncation_reason"],
        },
    }


def build(
    config: Path,
    base_dataset: Path,
    base_manifest: Path,
    base_csv: Path,
    output: Path,
    source_csv: Path,
    manifest_path: Path,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    guided_seed = protocol.stage("c4").training_seeds[0]
    for target in (output, source_csv, manifest_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    base_meta = json.loads(base_manifest.read_text(encoding="utf-8"))
    if file_sha256(base_dataset) != base_meta["dataset_sha256"]:
        raise RuntimeError("base V23 dataset hash mismatch")
    if file_sha256(base_csv) != base_meta["source_csv_sha256"]:
        raise RuntimeError("base V23 source CSV hash mismatch")
    with np.load(base_dataset) as data:
        base_x = np.asarray(data["observations"], dtype=np.float32)
        base_y = np.asarray(data["actions"], dtype=np.float32)
    with base_csv.open(newline="", encoding="utf-8") as stream:
        base_rows = list(csv.DictReader(stream))
    if len(base_rows) != len(base_x):
        raise RuntimeError("base V23 row provenance mismatch")

    guided = _guided_episode(config, guided_seed)
    guided_x = np.asarray(guided["observations"], dtype=np.float32)
    guided_y = np.asarray(guided["actions"], dtype=np.float32)
    guided_returns = np.asarray(guided["returns"], dtype=np.float32)
    if guided_x.shape[1:] != (len(protocol.observation_order),):
        raise RuntimeError("guided public observation dimension changed")

    roles = np.asarray(
        [row["source_role"] for row in base_rows]
        + ["c4_privileged_guided_episode"] * len(guided_x)
    )
    x = np.concatenate((base_x, guided_x))
    y = np.concatenate((base_y, guided_y))
    base_mass = float(max(int(count) for count in base_meta["role_rows"].values()))
    weights = np.empty(len(x), dtype=np.float32)
    role_rows: dict[str, int] = {}
    role_masses: dict[str, float] = {}
    for role, multiplier in ROLE_MULTIPLIERS.items():
        selected = roles == role
        count = int(np.sum(selected))
        if count <= 0:
            raise RuntimeError(f"missing V24 behavior role {role}")
        mass = base_mass * multiplier
        weights[selected] = mass / count
        role_rows[role] = count
        role_masses[role] = float(np.sum(weights[selected]))

    value_targets = np.zeros(len(x), dtype=np.float32)
    value_weights = np.zeros(len(x), dtype=np.float32)
    value_targets[-len(guided_x) :] = guided_returns
    value_weights[-len(guided_x) :] = base_mass / len(guided_x)
    np.savez_compressed(
        output,
        observations=x,
        actions=y,
        weights=weights,
        value_targets=value_targets,
        value_weights=value_weights,
    )

    policy_fields = [f"policy_normalized.{name}" for name in protocol.observation_order]
    truth_fields = (
        "teacher_gt.lane_lateral_error_m",
        "teacher_gt.lane_heading_error_rad",
        "teacher_gt.road_curvature_inv_m",
        "teacher_gt.pedestrian_exists",
        "teacher_gt.pedestrian_range_m",
        "teacher_gt.pedestrian_bearing_rad",
        "teacher_gt.stop_line_distance_m",
    )
    fields = tuple(base_rows[0]) + (
        "value_target",
        "value_weight",
        "teacher_uses_privileged_truth",
        *truth_fields,
    )
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(base_rows):
            row = dict(row)
            row["sample_weight"] = float(weights[index])
            row.update(
                value_target=0.0,
                value_weight=0.0,
                teacher_uses_privileged_truth=False,
            )
            writer.writerow(row)
        offset = len(base_rows)
        for index, (observation, action, truth) in enumerate(
            zip(guided_x, guided_y, guided["truths"], strict=True)
        ):
            row = {
                "source_role": "c4_privileged_guided_episode",
                "source_seed": guided_seed,
                "source_step": index + 1,
                "target_action_normalized.linear": float(action[0]),
                "target_action_normalized.angular": float(action[1]),
                "executed_action_normalized.linear": float(action[0]),
                "executed_action_normalized.angular": float(action[1]),
                "sample_weight": float(weights[offset + index]),
                "value_target": float(guided_returns[index]),
                "value_weight": float(value_weights[offset + index]),
                "teacher_uses_privileged_truth": True,
            }
            row.update(
                {
                    field: float(observation[column])
                    for column, field in enumerate(policy_fields)
                }
            )
            for field in truth_fields:
                key = field.removeprefix("teacher_gt.")
                row[field] = truth.get(key)
            writer.writerow(row)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "v23_cumulative_plus_one_privileged_c4_guided_episode_v1",
        "source_config": str(config.resolve()),
        "source_config_sha256": file_sha256(config),
        "builder_sha256": file_sha256(Path(__file__)),
        "base_dataset": str(base_dataset.resolve()),
        "base_dataset_sha256": file_sha256(base_dataset),
        "base_manifest": str(base_manifest.resolve()),
        "base_manifest_sha256": file_sha256(base_manifest),
        "base_source_csv": str(base_csv.resolve()),
        "base_source_csv_sha256": file_sha256(base_csv),
        "training_seeds_only": sorted(
            set(base_meta["training_seeds_only"] + [guided_seed])
        ),
        "guided_seed": guided_seed,
        "guided_episode_count": 1,
        "teacher_uses_evaluation_gt": True,
        "student_observations_use_evaluation_gt": False,
        "student_observation_source": "RGB -> lane belief + YOLO/F9c object beliefs -> 29D",
        "privileged_truth_stored_in_npz": False,
        "rows": int(len(x)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "role_rows": role_rows,
        "role_weight_masses": role_masses,
        "role_mass_multipliers": ROLE_MULTIPLIERS,
        "critic_supervised_rows": int(len(guided_x)),
        "critic_target": "discounted_environment_return",
        "gamma": float(protocol.ppo.gamma),
        "guided_episode": guided["episode"],
        "source_checkpoint_sha256": base_meta["source_checkpoint_sha256"],
        "learner_checkpoint_sha256": base_meta["learner_checkpoint_sha256"],
        "learner_checkpoint_stage": base_meta["learner_checkpoint_stage"],
        "learner_checkpoint_step": base_meta["learner_checkpoint_step"],
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
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.config.resolve(),
        args.base_dataset.resolve(),
        args.base_manifest.resolve(),
        args.base_csv.resolve(),
        args.output.resolve(),
        args.source_csv.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
