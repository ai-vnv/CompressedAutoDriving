"""Gate the C4 step-zero actor on C2, C3, and C4 training trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import PPODeterministicPolicy, run_episode, summarize_episodes


def _summary(config: Path, stage: str, checkpoint: Path, device: str, count: int) -> dict:
    protocol = load_ppo_curriculum_protocol(config)
    agent, payload = PPOAgent.load(checkpoint, device=device)
    seeds = protocol.stage(stage).training_seeds[:count]
    environment = PPOCurriculumEnvironment(config, stage=stage, split="training", seeds=seeds)
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
    return {"seeds": list(seeds), "summary": summarize_episodes(episodes)}


def validate(config: Path, checkpoint: Path, device: str) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    _, payload = PPOAgent.load(checkpoint, device=device)
    if payload.get("stage") != "c4" or int(payload.get("global_step", -1)) != 0:
        raise RuntimeError("cumulative DAgger gate requires the C4 step-zero actor")
    gate = dict(protocol.raw["behavior_warm_start"]["c4"]["gate"])
    current = {
        "c2": _summary(config, "c2", checkpoint, device, 4),
        "c3": _summary(config, "c3", checkpoint, device, 4),
        "c4": _summary(config, "c4", checkpoint, device, 2),
    }
    c2 = current["c2"]["summary"]
    c2_checks = {
        "collision": c2["collision_rate"] <= float(gate["maximum_c2_collision_rate"]),
        "unsafe": c2["unsafe_episode_rate"] <= float(gate["maximum_c2_unsafe_episode_rate"]),
        "progress": c2["mean_progress_m"] >= float(gate["minimum_c2_mean_progress_m"]),
        "valid_pose": c2["invalid_pose_rate"] <= float(gate["maximum_c2_invalid_pose_rate"]),
    }
    c3 = current["c3"]["summary"]
    c3_checks = {
        "stop_completion": c3["stop_completion_rate"] >= float(gate["minimum_c3_stop_completion_rate"]),
        "stop_violation": c3["stop_violation_rate"] <= float(gate["maximum_c3_stop_violation_rate"]),
        "restart": c3["restart_rate"] >= float(gate["minimum_c3_restart_rate"]),
        "collision": c3["collision_rate"] <= float(gate["maximum_c3_collision_rate"]),
    }
    c4 = current["c4"]["summary"]
    c4_checks = {
        "completion": c4["completion_rate"] >= float(gate["minimum_completion_rate"]),
        "progress": c4["mean_progress_m"] >= float(gate["minimum_mean_progress_m"]),
        "stop_completion": c4["stop_completion_rate"] >= float(gate["minimum_stop_completion_rate"]),
        "stop_violation": c4["stop_violation_rate"] <= float(gate["maximum_stop_violation_rate"]),
        "restart": c4["restart_rate"] >= float(gate["minimum_restart_rate"]),
        "collision": c4["collision_rate"] <= float(gate["maximum_collision_rate"]),
        "unsafe": c4["unsafe_episode_rate"] <= float(gate["maximum_unsafe_episode_rate"]),
        "lane": c4["lane_failure_rate"] <= float(gate["maximum_lane_failure_rate"]),
        "not_stationary": c4["stationary_fraction"] <= float(gate["maximum_stationary_fraction"]),
    }
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": 0,
        "training_seeds_only": True,
        "current": current,
        "c2_checks": c2_checks,
        "c3_checks": c3_checks,
        "c4_checks": c4_checks,
        "passed": all(c2_checks.values()) and all(c3_checks.values()) and all(c4_checks.values()),
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
