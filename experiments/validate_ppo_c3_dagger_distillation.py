"""Gate a C3 DAgger-distilled step-zero actor on training-only trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import (
    PPODeterministicPolicy,
    run_episode,
    summarize_episodes,
)


def _summary(config: Path, stage_key: str, checkpoint: Path, device: str) -> dict:
    protocol = load_ppo_curriculum_protocol(config)
    agent, payload = PPOAgent.load(checkpoint, device=device)
    policy = PPODeterministicPolicy(agent)
    stage = protocol.stage(stage_key)
    seeds = stage.training_seeds[:2]
    environment = PPOCurriculumEnvironment(
        config, stage=stage_key, split="training", seeds=seeds
    )
    try:
        episodes = [
            run_episode(
                environment,
                seed=seed,
                policy=policy,
                protocol=protocol,
                checkpoint_step=int(payload["global_step"]),
            )
            for seed in seeds
        ]
    finally:
        environment.close()
    return {
        "seeds": list(seeds),
        "summary": summarize_episodes(episodes),
        "checkpoint_stage": payload["stage"],
        "checkpoint_step": int(payload["global_step"]),
    }


def validate(config: Path, checkpoint: Path, device: str) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    agent, payload = PPOAgent.load(checkpoint, device=device)
    del agent
    if payload.get("stage") != "c3" or int(payload.get("global_step", -1)) != 0:
        raise RuntimeError("DAgger distillation gate requires the C3 step-zero actor")

    imported = dict(protocol.raw["curriculum_import"]["c2"])
    source = (config.parent / str(imported["selected_checkpoint"])).resolve()
    if file_sha256(source) != str(imported["selected_checkpoint_sha256"]):
        raise RuntimeError("imported C2 checkpoint hash mismatch")

    source_c2 = _summary(config, "c2", source, device)
    current_c2 = _summary(config, "c2", checkpoint, device)
    current_c3 = _summary(config, "c3", checkpoint, device)
    gate = dict(protocol.raw["behavior_warm_start"]["c3"]["gate"])
    completion_drop = float(source_c2["summary"]["completion_rate"]) - float(
        current_c2["summary"]["completion_rate"]
    )
    c2_retention_pass = completion_drop <= float(
        gate["maximum_c2_completion_drop"]
    )
    c3 = current_c3["summary"]
    c3_checks = {
        "completion": float(c3["completion_rate"])
        >= float(gate["minimum_completion_rate"]),
        "progress": float(c3["mean_progress_m"])
        >= float(gate["minimum_mean_progress_m"]),
        "stop_completion": float(c3["stop_completion_rate"])
        >= float(gate["minimum_stop_completion_rate"]),
        "restart": float(c3["restart_rate"])
        >= float(gate["minimum_restart_rate"]),
        "no_stop_violation": float(c3["stop_violation_rate"]) == 0.0,
        "no_collision": float(c3["collision_rate"]) == 0.0,
        "lane": float(c3["lane_failure_rate"])
        <= float(gate["maximum_lane_failure_rate"]),
        "valid_pose": float(c3["invalid_pose_rate"]) == 0.0,
        "not_stationary": float(c3["stationary_fraction"])
        <= float(gate["maximum_stationary_fraction"]),
    }
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": 0,
        "training_seeds_only": True,
        "source_c2": source_c2,
        "current_c2": current_c2,
        "current_c3": current_c3,
        "c2_completion_drop": completion_drop,
        "c2_retention_pass": c2_retention_pass,
        "c3_checks": c3_checks,
        "passed": c2_retention_pass and all(c3_checks.values()),
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
    result = validate(args.config.resolve(), args.checkpoint.resolve(), args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
