"""Distil a conditional multitask actor while importing a guided critic."""

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
    device: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    agent, retained_payload = PPOAgent.load(retained, device=device)
    critic_agent, critic_payload = PPOAgent.load(guided_critic, device=device)
    architecture = lambda item: (
        item.config.observation_dimension,
        item.config.action_dimension,
        item.config.hidden_sizes,
    )
    if architecture(agent) != architecture(critic_agent):
        raise RuntimeError("actor/critic checkpoint architectures differ")
    agent.model.critic.load_state_dict(critic_agent.model.critic.state_dict())

    with np.load(dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
    x = torch.as_tensor(observations, device=agent.device)
    y = torch.as_tensor(actions, device=agent.device)
    w = torch.as_tensor(weights, device=agent.device)
    with torch.no_grad():
        before = float(torch.mean((agent.model.actor(x) - y) ** 2).item())
    optimizer = torch.optim.Adam(
        agent.model.actor.parameters(), lr=learning_rate, eps=1.0e-5
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.actor.parameters(), 1.0)
            optimizer.step()
    with torch.no_grad():
        after = float(torch.mean((agent.model.actor(x) - y) ** 2).item())
    if not np.isfinite(after) or after >= before:
        raise RuntimeError("conditional actor distillation did not improve fit")

    metadata = dict(retained_payload.get("metadata", {}))
    metadata["conditional_actor_distillation"] = {
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
        "mse_before": before,
        "mse_after": after,
        "student_observation_uses_privileged_truth": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output, global_step=0, stage="c4", metadata=metadata)
    return {
        **metadata["conditional_actor_distillation"],
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--guided-critic", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=76000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = distil(
        retained=args.retained.resolve(),
        guided_critic=args.guided_critic.resolve(),
        dataset=args.dataset.resolve(),
        output=args.output.resolve(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    if args.manifest is not None:
        manifest = args.manifest.resolve()
        if manifest.exists():
            raise FileExistsError(manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
