"""Fit only a neutral-preserving pedestrian input adapter in the PPO actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duckie_pomdp.control import PPOAgent
from duckie_pomdp.control.f10_protocol import file_sha256


PED_START = 10
PED_STOP = 19
PED_NEUTRAL = torch.tensor(
    (0.0, 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    dtype=torch.float32,
)


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
    device: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    agent, retained_payload = PPOAgent.load(retained, device=device)
    critic_agent, critic_payload = PPOAgent.load(guided_critic, device=device)
    agent.model.critic.load_state_dict(critic_agent.model.critic.state_dict())
    with np.load(dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
    x = torch.as_tensor(observations, device=agent.device)
    y = torch.as_tensor(actions, device=agent.device)
    w = torch.as_tensor(weights, device=agent.device)
    first = agent.model.actor[0]
    if not isinstance(first, torch.nn.Linear):
        raise RuntimeError("PPO actor first module must be Linear")
    base_weight = first.weight.detach().clone()
    base_bias = first.bias.detach().clone()
    neutral = PED_NEUTRAL.to(agent.device)

    for parameter in agent.model.actor.parameters():
        parameter.requires_grad_(False)
    first.weight.requires_grad_(True)
    first.bias.requires_grad_(True)
    optimizer = torch.optim.Adam((first.weight,), lr=learning_rate, eps=1.0e-5)
    with torch.no_grad():
        before = float(
            torch.mean(w[:, None] * (agent.model.actor(x) - y) ** 2).item()
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size].to(agent.device)
            prediction = agent.model.actor(x[indices])
            loss = torch.mean(w[indices, None] * (prediction - y[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            first.bias.grad = None
            loss.backward()
            if first.weight.grad is None or first.bias.grad is None:
                raise RuntimeError("pedestrian adapter produced no gradient")
            # Constrained derivative for b = b0 - (Wped-Wped0) @ neutral:
            # dL/dWped = direct_gradient - dL/db outer neutral.
            first.weight.grad[:, PED_START:PED_STOP] -= (
                first.bias.grad[:, None] * neutral[None, :]
            )
            first.weight.grad[:, :PED_START] = 0.0
            first.weight.grad[:, PED_STOP:] = 0.0
            torch.nn.utils.clip_grad_norm_((first.weight,), 1.0)
            optimizer.step()
            with torch.no_grad():
                first.weight[:, :PED_START].copy_(base_weight[:, :PED_START])
                first.weight[:, PED_STOP:].copy_(base_weight[:, PED_STOP:])
                delta = first.weight[:, PED_START:PED_STOP] - base_weight[:, PED_START:PED_STOP]
                first.bias.copy_(base_bias - delta @ neutral)
    for parameter in agent.model.actor.parameters():
        parameter.requires_grad_(True)

    with torch.no_grad():
        after = float(
            torch.mean(w[:, None] * (agent.model.actor(x) - y) ** 2).item()
        )
        neutral_x = x[: min(1024, len(x))].clone()
        neutral_x[:, PED_START:PED_STOP] = neutral
        retained_agent, _ = PPOAgent.load(retained, device=device)
        neutral_error = float(
            torch.max(
                torch.abs(
                    agent.model.actor(neutral_x)
                    - retained_agent.model.actor(neutral_x)
                )
            ).item()
        )
    if not np.isfinite(after) or after >= before:
        raise RuntimeError("pedestrian adapter did not improve behavior fit")
    if neutral_error > 1.0e-6:
        raise RuntimeError("pedestrian adapter changed neutral behavior")

    metadata = dict(retained_payload.get("metadata", {}))
    metadata["pedestrian_input_adapter"] = {
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
        "trainable_parameters": "actor[0].weight[:,10:19]",
        "neutral_bias_compensation": True,
        "neutral_max_abs_actor_error": neutral_error,
        "mse_before": before,
        "mse_after": after,
        "student_observation_uses_privileged_truth": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output, global_step=0, stage="c4", metadata=metadata)
    return {
        **metadata["pedestrian_input_adapter"],
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
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=77000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(distil(
        retained=args.retained.resolve(),
        guided_critic=args.guided_critic.resolve(),
        dataset=args.dataset.resolve(),
        output=args.output.resolve(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
