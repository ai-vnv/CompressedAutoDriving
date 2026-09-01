"""Create auditable actor interpolants between retained and rehearsed PPO skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from duckie_pomdp.control import PPOAgent
from duckie_pomdp.control.f10_protocol import file_sha256


def interpolate(
    retained: Path,
    rehearsed: Path,
    alpha: float,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    retained_agent, retained_payload = PPOAgent.load(retained, device="cpu")
    rehearsed_agent, rehearsed_payload = PPOAgent.load(rehearsed, device="cpu")
    retained_state = retained_agent.model.actor.state_dict()
    rehearsed_state = rehearsed_agent.model.actor.state_dict()
    if retained_state.keys() != rehearsed_state.keys():
        raise RuntimeError("actor architectures differ")
    interpolated = {
        name: (1.0 - alpha) * retained_state[name] + alpha * rehearsed_state[name]
        for name in retained_state
    }
    # Keep the guided critic and policy optimizer payload from the rehearsed
    # checkpoint; only the deterministic actor mean is interpolated.
    rehearsed_agent.model.actor.load_state_dict(interpolated)
    metadata = dict(rehearsed_payload.get("metadata", {}))
    metadata["actor_interpolation"] = {
        "alpha": alpha,
        "retained_checkpoint": str(retained.resolve()),
        "retained_sha256": file_sha256(retained),
        "rehearsed_checkpoint": str(rehearsed.resolve()),
        "rehearsed_sha256": file_sha256(rehearsed),
        "selection_domain": "training-only C2/C3/C4 closed-loop gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rehearsed_agent.save(
        output,
        global_step=0,
        stage="c4",
        metadata=metadata,
    )
    result = {
        "alpha": alpha,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "retained_stage": retained_payload["stage"],
        "rehearsed_stage": rehearsed_payload["stage"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--rehearsed", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            interpolate(
                args.retained.resolve(),
                args.rehearsed.resolve(),
                args.alpha,
                args.output.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
