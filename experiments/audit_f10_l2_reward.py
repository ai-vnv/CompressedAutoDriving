"""Pre-training reward and zero-shot transfer audit for F10-L2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.control import (
    LaneTransferEnvironment,
    SACAgent,
    load_lane_transfer_protocol,
)
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.lane_policy import (
    LaneAlwaysStopPolicy,
    LaneRandomPolicy,
    LaneSACPolicy,
    LaneSimpleControllerPolicy,
    run_lane_episode,
    summarize_lane_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


class SourceSACPolicy(LaneSACPolicy):
    name = "source_f10_l1_sac"


def audit(config_path: Path, output: Path, *, device: str) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite F10-L2 reward audit: {output}")
    protocol = load_lane_transfer_protocol(config_path)
    agent, payload = SACAgent.load(protocol.transfer_checkpoint_path, device=device)
    policies = (
        LaneRandomPolicy(),
        LaneAlwaysStopPolicy(),
        LaneSimpleControllerPolicy(protocol),
        SourceSACPolicy(agent),
    )
    seeds = protocol.seeds.development[:2]
    environment = LaneTransferEnvironment(config_path, split="development")
    summaries: dict[str, object] = {}
    try:
        for policy in policies:
            rows = [
                run_lane_episode(
                    environment,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    checkpoint_step=(
                        int(payload["global_step"])
                        if policy.name == "source_f10_l1_sac"
                        else None
                    ),
                )
                for seed in seeds
            ]
            summaries[policy.name] = summarize_lane_episodes(rows)
    finally:
        environment.close()
    simple = summaries["simple_controller"]
    random = summaries["random"]
    stopped = summaries["always_stop"]
    checks = {
        "simple_completes_lap": simple["lap_success_rate"] == 1.0,
        "simple_beats_random_success": (
            simple["lap_success_rate"] > random["lap_success_rate"]
        ),
        "simple_beats_stop_success": (
            simple["lap_success_rate"] > stopped["lap_success_rate"]
        ),
        "simple_beats_random_return": simple["mean_return"] > random["mean_return"],
        "simple_beats_stop_return": simple["mean_return"] > stopped["mean_return"],
        "always_stop_not_successful": stopped["lap_success_rate"] == 0.0,
        "simple_has_no_invalid_pose": simple["invalid_pose_rate"] == 0.0,
        "simple_has_no_yellow_crossing": simple["yellow_crossing_rate"] == 0.0,
    }
    result = {
        "schema_version": 1,
        "stage": "F10_L2_REWARD_AND_ZERO_SHOT_AUDIT",
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_checkpoint": str(protocol.transfer_checkpoint_path),
        "source_checkpoint_sha256": protocol.transfer_checkpoint_sha256,
        "source_checkpoint_step": int(payload["global_step"]),
        "seeds": list(seeds),
        "summaries": summaries,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_l2_transfer_v1.toml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10_l2" / "reward_audit.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(audit(args.config.resolve(), args.output.resolve(), device=args.device), indent=2))


if __name__ == "__main__":
    main()

