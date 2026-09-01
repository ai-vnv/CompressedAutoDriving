"""Diagnose C4 policy divergence on training-only trajectories.

The rollout policy consumes only the public 29-D vector.  Privileged values are
written under an explicit ``evaluation_gt`` prefix after the action is chosen;
they are diagnostics only and never influence selection or control.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("at least one training-only seed is required")
    return result


def diagnose(
    *,
    config: Path,
    checkpoint: Path,
    reference_checkpoint: Path,
    seeds: tuple[int, ...],
    device: str,
    output: Path,
) -> None:
    protocol = load_ppo_curriculum_protocol(config)
    stage = protocol.stage("c4")
    forbidden = set(stage.development_seeds) | set(stage.stage_final_seeds)
    overlap = sorted(set(seeds) & forbidden)
    if overlap:
        raise RuntimeError(f"diagnostic seeds overlap C4 dev/final seeds: {overlap}")
    if output.exists():
        raise FileExistsError(output)

    agent, _ = PPOAgent.load(checkpoint, device=device)
    reference, _ = PPOAgent.load(reference_checkpoint, device=device)
    env = PPOCurriculumEnvironment(
        config,
        stage="c4",
        split="training",
        seeds=seeds,
    )
    rows: list[dict[str, object]] = []
    try:
        for seed in seeds:
            observation, reset_info = env.reset(seed=seed)
            state_info = reset_info
            for step in range(stage.episode_horizon_steps):
                chosen = agent.act(observation, deterministic=True).environment_action
                baseline = reference.act(
                    observation, deterministic=True
                ).environment_action
                next_observation, reward, terminated, truncated, next_info = env.step(
                    chosen
                )
                policy = state_info["policy"]
                evaluation_gt = next_info["evaluation_gt"]
                row: dict[str, object] = {
                    "seed": seed,
                    "step": step,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": next_info["termination_reason"],
                    "truncation_reason": next_info["truncation_reason"],
                    "progress_m": next_info["progress_m"],
                    "completed": next_info["completed"],
                    "lane_failure": next_info["lane_failure"],
                    "collision": next_info["collision"],
                    "unsafe_proximity": next_info["unsafe_proximity"],
                    "stop_completed": next_info["stop_completed"],
                    "stop_violation": next_info["stop_violation"],
                    "duckie_detection_count": next_info["perception"].get(
                        "duckie_detection_count", 0
                    ),
                    "pedestrian_measurement_accepted": next_info["perception"].get(
                        "pedestrian_measurement_accepted", False
                    ),
                    "actor_linear": float(chosen[0]),
                    "actor_angular": float(chosen[1]),
                    "reference_linear": float(baseline[0]),
                    "reference_angular": float(baseline[1]),
                    "action_delta_l2": float(np.linalg.norm(chosen - baseline)),
                }
                row.update({f"policy.{key}": value for key, value in policy.items()})
                row.update(
                    {
                        f"evaluation_gt.{key}": value
                        for key, value in evaluation_gt.items()
                    }
                )
                rows.append(row)
                observation = next_observation
                state_info = next_info
                if terminated or truncated:
                    break
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(
        config=args.config.resolve(),
        checkpoint=args.checkpoint.resolve(),
        reference_checkpoint=args.reference_checkpoint.resolve(),
        seeds=_seeds(args.seeds),
        device=args.device,
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
