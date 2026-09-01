"""Exercise the real development evaluator before substantive F10 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.control import F10GymEnvironment, SACAgent, load_f10_protocol
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.evaluation.f10_policy import (
    SACDeterministicPolicy,
    run_episode,
    summarize_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    device: str,
) -> dict[str, object]:
    protocol = load_f10_protocol(config_path)
    smoke_manifest_path = checkpoint_path.parents[1] / "training_run_manifest.json"
    smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    config_sha = file_sha256(config_path)
    if smoke_manifest.get("f10_config_sha256") != config_sha:
        raise RuntimeError("smoke checkpoint was not produced by the frozen config")
    agent, payload = SACAgent.load(checkpoint_path, device=device)
    environment = F10GymEnvironment(config_path, split="development")
    seed = protocol.seeds.development[0]
    try:
        episode = run_episode(
            environment,
            seed=seed,
            policy=SACDeterministicPolicy(agent),
            protocol=protocol,
            checkpoint_step=int(payload["global_step"]),
        )
    finally:
        environment.close()
    summary = summarize_episodes([episode])
    result: dict[str, object] = {
        "schema_version": 1,
        "stage": "F10_EVALUATOR_SMOKE",
        "passed": True,
        "f10_config_sha256": config_sha,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_global_step": int(payload["global_step"]),
        "split": "development",
        "seed": seed,
        "final_seed_used": False,
        "episode": episode.to_row(),
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "f10_sac_v1.toml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "f10"
            / "smoke"
            / "checkpoints"
            / "sac_step_0000096.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10" / "evaluator_smoke.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
