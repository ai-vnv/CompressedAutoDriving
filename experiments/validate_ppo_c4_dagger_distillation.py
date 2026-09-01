"""Gate a C4 DAgger step-zero actor on training-only trajectories."""

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
    seeds = protocol.stage(stage_key).training_seeds[:2]
    environment = PPOCurriculumEnvironment(
        config, stage=stage_key, split="training", seeds=seeds
    )
    try:
        episodes = [
            run_episode(
                environment,
                seed=seed,
                policy=PPODeterministicPolicy(agent),
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
    _, payload = PPOAgent.load(checkpoint, device=device)
    if payload.get("stage") != "c4" or int(payload.get("global_step", -1)) != 0:
        raise RuntimeError("C4 DAgger gate requires the C4 step-zero actor")
    imported = dict(protocol.raw["curriculum_import"]["c3"])
    source = (config.parent / str(imported["selected_checkpoint"])).resolve()
    if file_sha256(source) != str(imported["selected_checkpoint_sha256"]):
        raise RuntimeError("imported C3 checkpoint hash mismatch")

    source_c3 = _summary(config, "c3", source, device)
    current_c3 = _summary(config, "c3", checkpoint, device)
    current_c4 = _summary(config, "c4", checkpoint, device)
    gate = dict(protocol.raw["behavior_warm_start"]["c4"]["gate"])
    c3 = current_c3["summary"]
    c3_retention_checks = {
        "stop_completion": float(c3["stop_completion_rate"]) >= float(gate["minimum_c3_stop_completion_rate"]),
        "stop_violation": float(c3["stop_violation_rate"]) <= float(gate["maximum_c3_stop_violation_rate"]),
        "restart": float(c3["restart_rate"]) >= float(gate["minimum_c3_restart_rate"]),
        "collision": float(c3["collision_rate"]) <= float(gate["maximum_c3_collision_rate"]),
    }
    c4 = current_c4["summary"]
    c4_checks = {
        "completion": float(c4["completion_rate"]) >= float(gate["minimum_completion_rate"]),
        "progress": float(c4["mean_progress_m"]) >= float(gate["minimum_mean_progress_m"]),
        "stop_completion": float(c4["stop_completion_rate"]) >= float(gate["minimum_stop_completion_rate"]),
        "stop_violation": float(c4["stop_violation_rate"]) <= float(gate["maximum_stop_violation_rate"]),
        "restart": float(c4["restart_rate"]) >= float(gate["minimum_restart_rate"]),
        "collision": float(c4["collision_rate"]) <= float(gate["maximum_collision_rate"]),
        "unsafe": float(c4["unsafe_episode_rate"]) <= float(gate["maximum_unsafe_episode_rate"]),
        "lane": float(c4["lane_failure_rate"]) <= float(gate["maximum_lane_failure_rate"]),
        "not_stationary": float(c4["stationary_fraction"]) <= float(gate["maximum_stationary_fraction"]),
    }
    c3_retention_pass = all(c3_retention_checks.values())
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": 0,
        "training_seeds_only": True,
        "source_c3": source_c3,
        "current_c3": current_c3,
        "current_c4": current_c4,
        "c3_retention_checks": c3_retention_checks,
        "c3_retention_pass": c3_retention_pass,
        "c4_checks": c4_checks,
        "passed": c3_retention_pass and all(c4_checks.values()),
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
