"""Validate C1 retention and C2 skill before substantive multitask PPO training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo import PPOAgent
from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import (
    PPODeterministicPolicy,
    run_episode,
    summarize_episodes,
)


def _evaluate_checkpoint(
    *,
    config: Path,
    stage_key: str,
    checkpoint: Path,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one checkpoint on the stage's fixed development trajectories."""

    protocol = load_ppo_curriculum_protocol(config)
    agent, payload = PPOAgent.load(checkpoint, device=device)
    policy = PPODeterministicPolicy(agent)
    stage = protocol.stage(stage_key)
    env = PPOCurriculumEnvironment(config, stage=stage_key, split="development")
    try:
        episodes = [
            run_episode(
                env,
                seed=seed,
                policy=policy,
                protocol=protocol,
                checkpoint_step=int(payload["global_step"]),
            )
            for seed in stage.development_seeds
        ]
    finally:
        env.close()
    return summarize_episodes(episodes), payload


def validate(config: Path, checkpoint: Path, device: str) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    agent, payload = PPOAgent.load(checkpoint, device=device)
    if int(payload["global_step"]) != 0 or payload["stage"] != "c2":
        raise RuntimeError("distillation gate requires the C2 step-zero checkpoint")
    policy = PPODeterministicPolicy(agent)
    summaries: dict[str, dict[str, object]] = {}
    for stage_key in ("c1", "c2"):
        stage = protocol.stage(stage_key)
        env = PPOCurriculumEnvironment(config, stage=stage_key, split="development")
        try:
            episodes = [
                run_episode(
                    env,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=0,
                )
                for seed in stage.development_seeds
            ]
        finally:
            env.close()
        summaries[stage_key] = summarize_episodes(episodes)

    imported_c1 = dict(protocol.raw["curriculum_import"]["c1"])
    source_checkpoint = (
        config.parent / str(imported_c1["selected_checkpoint"])
    ).resolve()
    if file_sha256(source_checkpoint) != str(
        imported_c1["selected_checkpoint_sha256"]
    ):
        raise RuntimeError("imported C1 checkpoint hash mismatch")
    source_c1, source_payload = _evaluate_checkpoint(
        config=config,
        stage_key="c1",
        checkpoint=source_checkpoint,
        device=device,
    )
    if source_payload["stage"] != "c1":
        raise RuntimeError("imported retention baseline must be a C1 checkpoint")

    c1 = summaries["c1"]
    c2 = summaries["c2"]
    acceptance = protocol.raw["acceptance"]
    selection = protocol.raw["checkpoint_selection"]
    completion_drop = float(source_c1["completion_rate"]) - float(
        c1["completion_rate"]
    )
    c1_pass = (
        completion_drop
        <= float(acceptance["c2"]["maximum_c1_completion_drop"])
    )
    c2_checks = {
        "collision": float(c2["collision_rate"])
        <= float(acceptance["c2"]["maximum_collision_rate"]),
        "unsafe": float(c2["unsafe_episode_rate"])
        <= float(acceptance["c2"]["maximum_unsafe_episode_rate"]),
        "progress": float(c2["mean_progress_m"])
        >= float(acceptance["c2"]["minimum_mean_progress_m"]),
        "not_stationary": float(c2["stationary_fraction"])
        <= float(acceptance["c2"]["maximum_stationary_fraction"]),
        "lane": float(c2["lane_failure_rate"])
        <= float(selection["maximum_lane_failure_rate"]),
        "valid_pose": float(c2["invalid_pose_rate"])
        <= float(selection["maximum_invalid_pose_rate"]),
    }
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": 0,
        "development_seeds_only": True,
        "summaries": summaries,
        "c1_matched_trajectory_baseline": {
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": str(
                imported_c1["selected_checkpoint_sha256"]
            ),
            "summary": source_c1,
            "metric": "completion_rate",
            "baseline": float(source_c1["completion_rate"]),
            "current": float(c1["completion_rate"]),
            "drop": completion_drop,
            "maximum_allowed": float(
                acceptance["c2"]["maximum_c1_completion_drop"]
            ),
        },
        "c1_retention_pass": c1_pass,
        "c2_checks": c2_checks,
        "passed": c1_pass and all(c2_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = validate(
        args.config.resolve(), args.checkpoint.resolve(), args.device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
