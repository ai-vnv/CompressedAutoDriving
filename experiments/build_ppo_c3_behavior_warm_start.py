"""Build a balanced C2-retention/C3-stop actor warm-start dataset.

The C3 samples are collected from the real runtime boundary and contain only
the normalized 29D policy observation plus the public-belief reference
controller action.  Privileged evaluation fields are never read.  A frozen
C2 behavior dataset supplies the retention anchors.
"""

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
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController


def build(
    *,
    config: Path,
    anchor_dataset: Path,
    anchor_sha256: str,
    output: Path,
    source_csv: Path,
    manifest_path: Path,
    seed_count: int,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    stage = protocol.stage("c3")
    seeds = stage.training_seeds[:seed_count]
    if seed_count <= 0 or len(seeds) != seed_count:
        raise ValueError("seed_count exceeds the frozen C3 training split")
    if file_sha256(anchor_dataset) != anchor_sha256:
        raise RuntimeError("frozen C2 anchor dataset hash mismatch")
    for destination in (output, source_csv, manifest_path):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    controller = BeliefAwareSimpleController(protocol)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_rows: list[dict[str, object]] = []
    detected_steps = 0
    environment = PPOCurriculumEnvironment(config, stage="c3", split="training")
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=seed)
            episode_return = 0.0
            stop_completed = False
            restarted = False
            termination_reason = None
            truncation_reason = None
            steps = 0
            for steps in range(1, stage.episode_horizon_steps + 1):
                action = np.asarray(controller.act(observation), dtype=np.float32)
                if observation.shape != (len(protocol.observation_order),):
                    raise RuntimeError("C3 runtime observation dimension changed")
                if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
                    raise RuntimeError("non-finite public behavior sample")
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(action.copy())
                observation, reward, terminated, truncated, info = environment.step(action)
                episode_return += float(reward)
                detected_steps += int(
                    int(info.get("perception", {}).get("stop_sign_detection_count", 0)) > 0
                )
                stop_completed = stop_completed or bool(info.get("stop_completed", False))
                if stop_completed and float(info.get("v_cmd", 0.0)) > 0.05:
                    restarted = True
                termination_reason = info.get("termination_reason")
                truncation_reason = info.get("truncation_reason")
                if terminated or truncated:
                    episode_rows.append(
                        {
                            "seed": seed,
                            "steps": steps,
                            "return": episode_return,
                            "completed": bool(info.get("completed", False)),
                            "stop_completed": stop_completed,
                            "stop_violation": bool(info.get("stop_violation", False)),
                            "restarted": restarted,
                            "collision": bool(info.get("collision", False)),
                            "invalid_pose": bool(info.get("invalid_pose", False)),
                            "termination_reason": termination_reason,
                            "truncation_reason": truncation_reason,
                        }
                    )
                    break
            else:  # pragma: no cover - horizon is expected to truncate in env
                raise RuntimeError(f"C3 behavior episode {seed} did not terminate")
    finally:
        environment.close()

    stop_x = np.asarray(observations, dtype=np.float32)
    stop_y = np.asarray(actions, dtype=np.float32)
    if not episode_rows or not all(
        row["stop_completed"]
        and row["restarted"]
        and not row["stop_violation"]
        and not row["collision"]
        and not row["invalid_pose"]
        for row in episode_rows
    ):
        raise RuntimeError("public-belief C3 teacher did not pass every source episode")

    with np.load(anchor_dataset) as data:
        anchor_x_all = np.asarray(data["observations"], dtype=np.float32)
        anchor_y_all = np.asarray(data["actions"], dtype=np.float32)
    if anchor_x_all.shape[1:] != stop_x.shape[1:] or anchor_y_all.shape[1:] != (2,):
        raise RuntimeError("C2 anchor and C3 observation/action contracts differ")
    anchor_indices = np.linspace(
        0, len(anchor_x_all) - 1, len(stop_x), dtype=np.int64
    )
    if len(np.unique(anchor_indices)) != len(stop_x):
        raise RuntimeError("C2 anchor dataset is too small for balanced sampling")
    anchor_x = anchor_x_all[anchor_indices]
    anchor_y = anchor_y_all[anchor_indices]

    # Holding at the stop line is a brief but safety-critical part of a lap.
    # Weight it explicitly while keeping total C2 and C3 loss mass equal.
    stop_weights = np.ones(len(stop_y), dtype=np.float32)
    stop_weights[stop_y[:, 0] <= -0.99] = 32.0
    stop_weights[(stop_y[:, 0] > -0.99) & (stop_y[:, 0] <= -0.50)] = 4.0
    anchor_weight = float(np.sum(stop_weights) / len(anchor_x))
    anchor_weights = np.full(len(anchor_x), anchor_weight, dtype=np.float32)
    x = np.concatenate((anchor_x, stop_x), axis=0)
    y = np.concatenate((anchor_y, stop_y), axis=0)
    weights = np.concatenate((anchor_weights, stop_weights), axis=0)

    np.savez_compressed(output, observations=x, actions=y, weights=weights)
    fields = (
        "source_role",
        "source_seed",
        "source_step",
        *(f"policy_normalized.{name}" for name in protocol.observation_order),
        "action_normalized.linear",
        "action_normalized.angular",
        "sample_weight",
    )
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(anchor_x)):
            row = {
                "source_role": "c2_retention_anchor",
                "source_seed": "",
                "source_step": int(anchor_indices[index]),
                "action_normalized.linear": float(anchor_y[index, 0]),
                "action_normalized.angular": float(anchor_y[index, 1]),
                "sample_weight": float(anchor_weights[index]),
            }
            row.update(
                {
                    f"policy_normalized.{name}": float(anchor_x[index, column])
                    for column, name in enumerate(protocol.observation_order)
                }
            )
            writer.writerow(row)
        offset = 0
        for episode in episode_rows:
            for step in range(int(episode["steps"])):
                index = offset + step
                row = {
                    "source_role": "c3_stop_teacher",
                    "source_seed": episode["seed"],
                    "source_step": step + 1,
                    "action_normalized.linear": float(stop_y[index, 0]),
                    "action_normalized.angular": float(stop_y[index, 1]),
                    "sample_weight": float(stop_weights[index]),
                }
                row.update(
                    {
                        f"policy_normalized.{name}": float(stop_x[index, column])
                        for column, name in enumerate(protocol.observation_order)
                    }
                )
                writer.writerow(row)
            offset += int(episode["steps"])

    stop_mode_satisfied = protocol.observation_order.index("stop_mode_satisfied")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_role": "balanced_c2_retention_and_c3_public_belief_teacher",
        "source_config": str(config.resolve()),
        "source_config_sha256": file_sha256(config),
        "source_scenario_config_sha256": file_sha256(stage.scenario_config_path),
        "builder_sha256": file_sha256(Path(__file__)),
        "training_seeds_only": list(seeds),
        "uses_evaluation_gt": False,
        "runtime_source": "RGB -> YOLO stop_sign -> metric projection -> stop belief -> 29D policy observation",
        "rows": int(len(x)),
        "anchor_rows": int(len(anchor_x)),
        "stop_rows": int(len(stop_x)),
        "observation_dimension": int(x.shape[1]),
        "action_dimension": 2,
        "stop_action_rows": int(np.sum(stop_y[:, 0] <= -0.99)),
        "satisfied_observation_rows": int(
            np.sum(stop_x[:, stop_mode_satisfied] > 0.5)
        ),
        "detected_steps": detected_steps,
        "anchor_weight_mass": float(np.sum(anchor_weights)),
        "stop_weight_mass": float(np.sum(stop_weights)),
        "source_episodes": episode_rows,
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
    parser.add_argument("--anchor-dataset", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                config=args.config.resolve(),
                anchor_dataset=args.anchor_dataset.resolve(),
                anchor_sha256=args.anchor_sha256,
                output=args.output.resolve(),
                source_csv=args.source_csv.resolve(),
                manifest_path=args.manifest.resolve(),
                seed_count=args.seed_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
