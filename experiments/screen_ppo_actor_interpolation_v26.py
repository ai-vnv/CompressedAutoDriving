"""Screen actor interpolants on explicitly supplied, training-only trajectories.

This is a diagnostic/model-development utility.  It deliberately refuses the
development and final seed ranges used by the curriculum protocol, so its
result cannot become an accidental final evaluation.
"""

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


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one screening seed is required")
    return seeds


def screen(
    *,
    config: Path,
    stage: str,
    seeds: tuple[int, ...],
    checkpoints: tuple[Path, ...],
    device: str,
) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    stage_protocol = protocol.stage(stage)
    forbidden = set(stage_protocol.development_seeds) | set(
        stage_protocol.stage_final_seeds
    )
    overlap = sorted(set(seeds) & forbidden)
    if overlap:
        raise RuntimeError(
            f"screening seeds overlap {stage} development/final seeds: {overlap}"
        )

    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        agent, payload = PPOAgent.load(checkpoint, device=device)
        environment = PPOCurriculumEnvironment(
            config,
            stage=stage,
            split="training",
            seeds=seeds,
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
        results.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "actor_interpolation": payload.get("metadata", {}).get(
                    "actor_interpolation"
                ),
                "summary": summarize_episodes(episodes),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "training-only actor interpolation screening",
        "config": str(config.resolve()),
        "config_sha256": file_sha256(config),
        "stage": stage,
        "seeds": list(seeds),
        "development_or_final_seeds_used": False,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("c2", "c3", "c4"), required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = screen(
        config=args.config.resolve(),
        stage=args.stage,
        seeds=_parse_seeds(args.seeds),
        checkpoints=tuple(item.resolve() for item in args.checkpoint),
        device=args.device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
