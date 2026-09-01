"""Pre-training reward audit using only frozen training seeds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import PPOCurriculumEnvironment, load_ppo_curriculum_protocol
from duckie_pomdp.control.ppo_protocol import protocol_artifact_root
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.f10_ppo_policy import (
    AlwaysStopPolicy,
    BeliefAwareSimpleController,
    RandomPolicy,
    run_episode,
    summarize_episodes,
)
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def audit(config_path: Path, stage_key: str, output_path: Path) -> dict:
    protocol = load_ppo_curriculum_protocol(config_path)
    stage = protocol.stage(stage_key)
    seeds = stage.training_seeds[:2]
    policies = (RandomPolicy(), AlwaysStopPolicy(), BeliefAwareSimpleController(protocol))
    environment = PPOCurriculumEnvironment(
        config_path, stage=stage_key, split="training"
    )
    try:
        rows = {
            policy.name: [
                run_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                )
                for seed in seeds
            ]
            for policy in policies
        }
    finally:
        environment.close()
    summaries = {name: summarize_episodes(values) for name, values in rows.items()}
    simple = summaries["simple_controller"]
    random = summaries["random"]
    stopped = summaries["always_stop"]
    checks = {
        "always_stop_not_complete": stopped["completion_rate"] == 0.0,
        "simple_progresses_more_than_stop": simple["mean_progress_m"] > stopped["mean_progress_m"] + 0.25,
        "simple_return_exceeds_random": simple["mean_return"] > random["mean_return"],
        "inactive_pedestrian_reward_zero": (
            stage.pedestrian_active or simple["mean_reward_pedestrian"] == 0.0
        ),
        "inactive_stop_reward_zero": stage.stop_active or simple["mean_reward_stop"] == 0.0,
    }
    if stage_key in {"c0", "c1"}:
        checks.update(
            {
                "simple_completes": simple["completion_rate"] >= 0.5,
                "simple_no_lane_failure": simple["lane_failure_rate"] == 0.0,
                "simple_no_invalid_pose": simple["invalid_pose_rate"] == 0.0,
            }
        )
    if stage.pedestrian_active:
        checks["simple_no_collision"] = simple["collision_rate"] == 0.0
    if stage.stop_active:
        checks["simple_stops"] = simple["stop_completion_rate"] >= 0.5
        checks["simple_does_not_violate_stop"] = simple["stop_violation_rate"] <= 0.5
        checks["simple_restarts_after_stop"] = simple["restart_rate"] >= 0.5
        checks["simple_no_collision"] = simple["collision_rate"] == 0.0
        checks["simple_no_invalid_pose"] = simple["invalid_pose_rate"] == 0.0

    scenario_provenance = None
    if stage.scenario_config_path is not None:
        scenario = load_scenario(stage.scenario_config_path)
        scenario_provenance = {
            "config": str(stage.scenario_config_path.resolve()),
            "config_sha256": file_sha256(stage.scenario_config_path),
            "map": str(scenario.map_path.resolve()),
            "map_sha256": file_sha256(scenario.map_path),
        }
    result = {
        "schema_version": 1,
        "stage": f"F10_PPO_{stage_key.upper()}_REWARD_AUDIT",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_sha256": file_sha256(Path(__file__)),
        "scenario_provenance": scenario_provenance,
        "reference_policy_sha256": file_sha256(
            ROOT / "src" / "duckie_pomdp" / "evaluation" / "f10_ppo_policy.py"
        ),
        "seeds": list(seeds),
        "summaries": summaries,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("c0", "c1", "c2", "c3", "c4"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_v1.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_ppo_curriculum_protocol(args.config.resolve())
    output = args.output or protocol_artifact_root(protocol) / args.stage / "reward_audit.json"
    print(json.dumps(audit(args.config, args.stage, output), indent=2))


if __name__ == "__main__":
    main()
