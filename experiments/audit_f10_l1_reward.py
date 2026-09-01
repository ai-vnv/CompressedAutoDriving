"""Pre-training qualitative reward audit for the F10-L1 lane stage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import LaneCurriculumEnvironment, load_lane_protocol
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.lane_policy import (
    LaneAlwaysStopPolicy,
    LaneRandomPolicy,
    LaneSimpleControllerPolicy,
    run_lane_episode,
    summarize_lane_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def audit(config_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite reward audit: {output_path}")
    protocol = load_lane_protocol(config_path)
    policies = (
        LaneRandomPolicy(),
        LaneAlwaysStopPolicy(),
        LaneSimpleControllerPolicy(protocol),
    )
    seeds = protocol.seeds.development[:2]
    environment = LaneCurriculumEnvironment(config_path, split="development")
    rows = {}
    try:
        for policy in policies:
            episodes = [
                run_lane_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                )
                for seed in seeds
            ]
            rows[policy.name] = summarize_lane_episodes(episodes)
    finally:
        environment.close()
    simple = rows["simple_controller"]
    random = rows["random"]
    stopped = rows["always_stop"]
    checks = {
        "simple_completes_lap": float(simple["lap_success_rate"]) >= 0.5,
        "simple_beats_random_success": float(simple["lap_success_rate"])
        > float(random["lap_success_rate"]),
        "simple_beats_stop_success": float(simple["lap_success_rate"])
        > float(stopped["lap_success_rate"]),
        "simple_beats_random_return": float(simple["mean_return"])
        > float(random["mean_return"]),
        "simple_beats_stop_return": float(simple["mean_return"])
        > float(stopped["mean_return"]),
        "always_stop_not_successful": float(stopped["lap_success_rate"]) == 0.0,
        "simple_has_no_invalid_pose": float(simple["invalid_pose_rate"]) == 0.0,
        "simple_has_no_yellow_crossing": float(simple["yellow_crossing_rate"])
        == 0.0,
    }
    result = {
        "schema_version": 1,
        "stage": "F10_L1_REWARD_AUDIT",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "seeds": list(seeds),
        "summaries": rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l1_lane_v1.toml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10_l1" / "reward_audit.json",
    )
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

