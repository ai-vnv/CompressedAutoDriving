"""Canonical feed-forward PPO for continuous normalized actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class PPOConfig:
    observation_dimension: int
    action_dimension: int
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    value_function_coefficient: float
    max_gradient_norm: float
    initial_log_std: float
    seed: int
    device: str
    target_kl: float | None = None


def _mlp(input_size: int, output_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.Tanh()))
        previous = size
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.actor = _mlp(
            config.observation_dimension,
            config.action_dimension,
            config.hidden_sizes,
        )
        self.critic = _mlp(config.observation_dimension, 1, config.hidden_sizes)
        self.log_std = nn.Parameter(
            torch.full((config.action_dimension,), config.initial_log_std)
        )
        self._orthogonal_initialize()

    def _orthogonal_initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.actor(observation)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


@dataclass(frozen=True)
class PPOAction:
    raw_action: NDArray[np.float32]
    environment_action: NDArray[np.float32]
    log_probability: float
    value: float


class PPORolloutBuffer:
    def __init__(self, capacity: int, observation_dimension: int, action_dimension: int) -> None:
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_dimension), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dimension), dtype=np.float32)
        self.log_probabilities = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.episode_done = np.zeros(capacity, dtype=np.bool_)
        self.size = 0

    def add(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        log_probability: float,
        value: float,
        reward: float,
        next_value: float,
        *,
        terminated: bool,
        episode_done: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("PPO rollout buffer is full")
        index = self.size
        self.observations[index] = observation
        self.actions[index] = action
        self.log_probabilities[index] = log_probability
        self.values[index] = value
        self.rewards[index] = reward
        self.next_values[index] = next_value
        self.terminated[index] = terminated
        self.episode_done[index] = episode_done
        self.size += 1

    def advantages_and_returns(
        self, gamma: float, gae_lambda: float
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.size != self.capacity:
            raise RuntimeError("PPO update requires a full rollout")
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(self.size)):
            bootstrap = 0.0 if self.terminated[index] else self.next_values[index]
            delta = self.rewards[index] + gamma * bootstrap - self.values[index]
            continuation = 0.0 if self.episode_done[index] else 1.0
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
        return advantages, advantages + self.values[: self.size]


class PPOAgent:
    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.model = PPOActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1.0e-5
        )
        self.update_count = 0

    def act(self, observation: NDArray[np.float32], *, deterministic: bool) -> PPOAction:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.distribution(tensor)
            raw = distribution.mean if deterministic else distribution.sample()
            log_probability = distribution.log_prob(raw).sum(dim=-1)
            value = self.model.value(tensor)
        raw_array = raw.squeeze(0).cpu().numpy().astype(np.float32)
        return PPOAction(
            raw_action=raw_array,
            environment_action=np.clip(raw_array, -1.0, 1.0).astype(np.float32),
            log_probability=float(log_probability.item()),
            value=float(value.item()),
        )

    def value(self, observation: NDArray[np.float32]) -> float:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.model.value(tensor).item())

    def update(self, rollout: PPORolloutBuffer) -> dict[str, float]:
        advantages, returns = rollout.advantages_and_returns(
            self.config.gamma, self.config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        observation = torch.as_tensor(rollout.observations, device=self.device)
        action = torch.as_tensor(rollout.actions, device=self.device)
        old_log_probability = torch.as_tensor(rollout.log_probabilities, device=self.device)
        advantage = torch.as_tensor(advantages, device=self.device)
        return_tensor = torch.as_tensor(returns, device=self.device)
        indices = np.arange(rollout.size)
        metrics: list[dict[str, float]] = []
        early_stopped = False
        optimization_steps = 0
        for _ in range(self.config.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout.size, self.config.batch_size):
                batch = indices[start : start + self.config.batch_size]
                distribution = self.model.distribution(observation[batch])
                new_log_probability = distribution.log_prob(action[batch]).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()
                ratio = (new_log_probability - old_log_probability[batch]).exp()
                approximate_kl = ((ratio - 1.0) - ratio.log()).mean()
                if (
                    self.config.target_kl is not None
                    and float(approximate_kl.item())
                    > 1.5 * self.config.target_kl
                ):
                    early_stopped = True
                    break
                unclipped = ratio * advantage[batch]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantage[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_prediction = self.model.value(observation[batch])
                value_loss = 0.5 * (return_tensor[batch] - value_prediction).pow(2).mean()
                loss = (
                    policy_loss
                    + self.config.value_function_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()
                optimization_steps += 1
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_range)
                    .float()
                    .mean()
                )
                metrics.append(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "approximate_kl": float(approximate_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "gradient_norm": float(gradient_norm.item()),
                    }
                )
            if early_stopped:
                break
        if not metrics:
            raise RuntimeError("PPO target-KL guard stopped before any optimizer step")
        self.update_count += 1
        return {
            name: float(np.mean([row[name] for row in metrics]))
            for name in metrics[0]
        } | {
            "explained_variance": _explained_variance(
                rollout.values[: rollout.size], returns
            ),
            "mean_log_std": float(self.model.log_std.detach().mean().cpu().item()),
            "update_count": float(self.update_count),
            "optimization_steps": float(optimization_steps),
            "early_stopped": float(early_stopped),
        }

    def save(
        self,
        path: str | Path,
        *,
        global_step: int,
        stage: str,
        metadata: dict[str, Any],
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "algorithm": "canonical_feedforward_ppo",
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": int(global_step),
                "stage": stage,
                "update_count": self.update_count,
                "metadata": metadata,
            },
            target,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str,
        learning_rate: float | None = None,
    ) -> tuple[PPOAgent, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        raw = dict(payload["config"])
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
        raw["device"] = device
        if learning_rate is not None:
            raw["learning_rate"] = learning_rate
        agent = cls(PPOConfig(**raw))
        agent.model.load_state_dict(payload["model_state"])
        agent.optimizer.load_state_dict(payload["optimizer_state"])
        if learning_rate is not None:
            for group in agent.optimizer.param_groups:
                group["lr"] = learning_rate
        agent.update_count = int(payload["update_count"])
        return agent, payload


def _explained_variance(prediction: NDArray[np.float32], target: NDArray[np.float32]) -> float:
    variance = float(np.var(target))
    if variance <= 1.0e-12:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / variance)
