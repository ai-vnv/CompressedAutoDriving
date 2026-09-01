"""Disjoint closed-loop gate for camera lane belief on C0 and C1 maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from duckie_pomdp.control import PPOCurriculumEnvironment, load_ppo_curriculum_protocol
from duckie_pomdp.evaluation.f10_ppo_policy import (
    BeliefAwareSimpleController,
    run_episode,
    summarize_episodes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_ppo_visual_v7.toml"
OUTPUT = ROOT / "artifacts" / "f10_ppo_visual_v7" / "lane_closed_loop_gate"
SEEDS = {
    "development": {"c0": (85_001, 85_002, 85_003, 85_004), "c1": (85_011, 85_012, 85_013, 85_014)},
    "final": {"c0": (85_101, 85_102, 85_103, 85_104), "c1": (85_111, 85_112, 85_113, 85_114)},
}
CRITERIA = {
    "c0": {"minimum_completion_rate": 0.75, "maximum_mean_abs_lateral_error_m": 0.08},
    "c1": {"minimum_completion_rate": 0.50, "maximum_mean_abs_lateral_error_m": 0.08},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SEEDS), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed-base", type=int, default=85_000)
    args = parser.parse_args()
    split = str(args.split)
    config_path = args.config.resolve()
    output = args.output.resolve()
    seeds = _seeds(args.seed_base)[split]
    csv_path = output / f"{split}.csv"
    metrics_path = output / f"{split}_metrics.json"
    if csv_path.exists() or metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite closed-loop {split} gate")
    protocol = load_ppo_curriculum_protocol(config_path)
    rows = []
    summaries = {}
    checks = {}
    for stage in ("c0", "c1"):
        environment = PPOCurriculumEnvironment(
            config_path,
            # Config is explicit so a remediation cannot accidentally run
            # against the previous checkpoint's observation source.
            stage=stage,
            split=f"lane_gate_{split}",
            seeds=seeds[stage],
        )
        try:
            policy = BeliefAwareSimpleController(protocol)
            episodes = [
                run_episode(environment, seed=seed, policy=policy, protocol=protocol)
                for seed in seeds[stage]
            ]
        finally:
            environment.close()
        rows.extend(episode.to_row() for episode in episodes)
        summary = summarize_episodes(episodes)
        summaries[stage] = summary
        criteria = CRITERIA[stage]
        checks[stage] = {
            "completion": summary["completion_rate"] >= criteria["minimum_completion_rate"],
            "no_lane_failure": summary["lane_failure_rate"] == 0.0,
            "no_invalid_pose": summary["invalid_pose_rate"] == 0.0,
            "lateral_error": summary["mean_abs_lateral_error_m"]
            <= criteria["maximum_mean_abs_lateral_error_m"],
        }
    result = {
        "schema_version": 1,
        "gate": f"camera lane-belief closed-loop {split}",
        "split": split,
        "seed_role": "development" if split == "development" else "once-only held-out final",
        "seeds": {name: list(values) for name, values in seeds.items()},
        "direction": "counter-clockwise",
        "runtime_chain": "front_rgb -> MobileNet lane measurement -> lane EKF belief -> simple controller",
        "privileged_use": "reward/evaluation only after controller action",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pre_registered_criteria": CRITERIA,
        "summaries": summaries,
        "checks": checks,
        "gate_pass": all(all(stage_checks.values()) for stage_checks in checks.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result["episodes_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _seeds(base: int):
    return {
        "development": {
            "c0": tuple(range(base + 1, base + 5)),
            "c1": tuple(range(base + 11, base + 15)),
        },
        "final": {
            "c0": tuple(range(base + 101, base + 105)),
            "c1": tuple(range(base + 111, base + 115)),
        },
    }


if __name__ == "__main__":
    main()
