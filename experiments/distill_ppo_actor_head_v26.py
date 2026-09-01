"""Distil multitask teacher actions into only the PPO actor output head.

The retained actor backbone is frozen.  This keeps the visual/belief features
that already solve C3/C4 while allowing a small, auditable correction of the
two physical action outputs.  The critic is copied from the separately guided
checkpoint and is never used to generate actor labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duckie_pomdp.control import PPOAgent
from duckie_pomdp.control.f10_protocol import file_sha256


def distil(
    *,
    retained: Path,
    guided_critic: Path,
    dataset: Path,
    output: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    anchor_coefficient: float,
    device: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if epochs <= 0 or learning_rate <= 0.0 or batch_size <= 0:
        raise ValueError("epochs, learning rate, and batch size must be positive")
    if anchor_coefficient < 0.0:
        raise ValueError("anchor coefficient must be nonnegative")

    agent, retained_payload = PPOAgent.load(retained, device=device)
    critic_agent, critic_payload = PPOAgent.load(guided_critic, device=device)
    if (
        agent.config.observation_dimension
        != critic_agent.config.observation_dimension
        or agent.config.action_dimension != critic_agent.config.action_dimension
        or agent.config.hidden_sizes != critic_agent.config.hidden_sizes
    ):
        raise RuntimeError("retained actor and guided critic architectures differ")
    agent.model.critic.load_state_dict(critic_agent.model.critic.state_dict())

    with np.load(dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
    if observations.shape != (len(observations), agent.config.observation_dimension):
        raise ValueError("observation shape mismatch")
    if actions.shape != (len(observations), agent.config.action_dimension):
        raise ValueError("action shape mismatch")
    if weights.shape != (len(observations),) or np.any(weights <= 0.0):
        raise ValueError("sample weights are invalid")

    x = torch.as_tensor(observations, device=agent.device)
    y = torch.as_tensor(actions, device=agent.device)
    w = torch.as_tensor(weights, device=agent.device)
    w = w / w.mean()
    with torch.no_grad():
        anchor = agent.model.actor(x).detach()
        mse_before = float(torch.mean((anchor - y) ** 2).item())

    for parameter in agent.model.actor.parameters():
        parameter.requires_grad_(False)
    output_head = agent.model.actor[-1]
    for parameter in output_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        output_head.parameters(), lr=learning_rate, eps=1.0e-5
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size].to(agent.device)
            prediction = agent.model.actor(x[indices])
            imitation = torch.mean(
                w[indices, None] * (prediction - y[indices]) ** 2
            )
            anchor_loss = torch.mean((prediction - anchor[indices]) ** 2)
            loss = imitation + anchor_coefficient * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(output_head.parameters(), 1.0)
            optimizer.step()
    for parameter in agent.model.actor.parameters():
        parameter.requires_grad_(True)

    with torch.no_grad():
        prediction = agent.model.actor(x)
        mse_after = float(torch.mean((prediction - y) ** 2).item())
        drift_mse = float(torch.mean((prediction - anchor) ** 2).item())
    if not np.isfinite(mse_after) or mse_after >= mse_before:
        raise RuntimeError("output-head distillation did not improve imitation")

    metadata = dict(retained_payload.get("metadata", {}))
    metadata["actor_head_distillation"] = {
        "retained_checkpoint": str(retained.resolve()),
        "retained_sha256": file_sha256(retained),
        "guided_critic_checkpoint": str(guided_critic.resolve()),
        "guided_critic_sha256": file_sha256(guided_critic),
        "guided_critic_source_stage": critic_payload["stage"],
        "dataset": str(dataset.resolve()),
        "dataset_sha256": file_sha256(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "anchor_coefficient": anchor_coefficient,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "retained_actor_drift_mse": drift_mse,
        "trainable_actor_module": "actor[-1]",
        "student_observation_uses_privileged_truth": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output, global_step=0, stage="c4", metadata=metadata)
    return {
        **metadata["actor_head_distillation"],
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--guided-critic", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=75000)
    parser.add_argument("--anchor-coefficient", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            distil(
                retained=args.retained.resolve(),
                guided_critic=args.guided_critic.resolve(),
                dataset=args.dataset.resolve(),
                output=args.output.resolve(),
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                seed=args.seed,
                anchor_coefficient=args.anchor_coefficient,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
