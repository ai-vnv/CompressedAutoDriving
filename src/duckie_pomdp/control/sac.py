"""Canonical Soft Actor-Critic implementation for the F10 continuous policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class SACConfig:
    observation_dimension: int
    action_dimension: int
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    gamma: float
    tau: float
    batch_size: int
    replay_buffer_size: int
    learning_starts: int
    train_frequency: int
    gradient_steps: int
    initial_entropy_coefficient: float
    target_entropy: float
    seed: int
    device: str


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_dimension: int, action_dimension: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.backbone, output_dimension = _mlp_backbone(observation_dimension, hidden_sizes)
        self.mean = nn.Linear(output_dimension, action_dimension)
        self.log_std = nn.Linear(output_dimension, action_dimension)

    def distribution_parameters(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(observation)
        return self.mean(features), self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_probability = distribution.log_prob(pre_tanh) - torch.log(
            1.0 - action.pow(2) + 1.0e-6
        )
        return action, log_probability.sum(dim=-1, keepdim=True)

    def deterministic(self, observation: Tensor) -> Tensor:
        mean, _ = self.distribution_parameters(observation)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, observation_dimension: int, action_dimension: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.network = _mlp(observation_dimension + action_dimension, hidden_sizes, 1)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dimension: int,
        action_dimension: int,
        *,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("replay-buffer capacity must be positive")
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dimension), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, action_dimension), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_observation: NDArray[np.float32],
        terminated: bool,
    ) -> None:
        self.observations[self._position] = observation
        self.actions[self._position] = action
        self.rewards[self._position, 0] = reward
        self.next_observations[self._position] = next_observation
        self.terminated[self._position, 0] = float(terminated)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than the replay buffer contains")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "terminated": torch.as_tensor(self.terminated[indices], device=device),
        }


class SACAgent:
    def __init__(self, config: SACConfig) -> None:
        self.config = config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAC config requests CUDA but torch.cuda is unavailable")
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.actor = SquashedGaussianActor(
            config.observation_dimension,
            config.action_dimension,
            config.hidden_sizes,
        ).to(self.device)
        self.q1 = QNetwork(config.observation_dimension, config.action_dimension, config.hidden_sizes).to(self.device)
        self.q2 = QNetwork(config.observation_dimension, config.action_dimension, config.hidden_sizes).to(self.device)
        self.target_q1 = QNetwork(config.observation_dimension, config.action_dimension, config.hidden_sizes).to(self.device)
        self.target_q2 = QNetwork(config.observation_dimension, config.action_dimension, config.hidden_sizes).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            tuple(self.q1.parameters()) + tuple(self.q2.parameters()),
            lr=config.learning_rate,
        )
        self.log_alpha = torch.tensor(
            np.log(config.initial_entropy_coefficient),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.update_count = 0

    @property
    def entropy_coefficient(self) -> float:
        return float(self.log_alpha.exp().detach().cpu())

    def act(self, observation: NDArray[np.float32], *, deterministic: bool) -> NDArray[np.float32]:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.config.observation_dimension,):
            raise ValueError("SAC observation has the wrong shape")
        if not np.all(np.isfinite(array)):
            raise ValueError("SAC observation must be finite")
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        batch = replay_buffer.sample(self.config.batch_size, self.device)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        terminated = batch["terminated"]

        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_observations)
            target_q = torch.minimum(
                self.target_q1(next_observations, next_actions),
                self.target_q2(next_observations, next_actions),
            ) - self.log_alpha.exp() * next_log_probability
            target = rewards + self.config.gamma * (1.0 - terminated) * target_q

        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        q1_loss = nn.functional.mse_loss(q1, target)
        q2_loss = nn.functional.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        sampled_actions, log_probability = self.actor.sample(observations)
        actor_q = torch.minimum(
            self.q1(observations, sampled_actions),
            self.q2(observations, sampled_actions),
        )
        actor_loss = (self.log_alpha.exp().detach() * log_probability - actor_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability + self.config.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        _soft_update(self.target_q1, self.q1, self.config.tau)
        _soft_update(self.target_q2, self.q2, self.config.tau)
        self.update_count += 1
        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "entropy_coefficient": self.entropy_coefficient,
            "mean_log_probability": float(log_probability.mean().detach().cpu()),
            "mean_q": float(actor_q.mean().detach().cpu()),
        }

    def save(self, path: str | Path, *, global_step: int, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "config": asdict(self.config),
                "global_step": int(global_step),
                "update_count": self.update_count,
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> tuple[SACAgent, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        config_data = dict(payload["config"])
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        if device is not None:
            config_data["device"] = device
        agent = cls(SACConfig(**config_data))
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.target_q1.load_state_dict(payload["target_q1"])
        agent.target_q2.load_state_dict(payload["target_q2"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.q_optimizer.load_state_dict(payload["q_optimizer"])
        agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        agent.update_count = int(payload["update_count"])
        return agent, payload


def _mlp_backbone(input_dimension: int, hidden_sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    if not hidden_sizes:
        return nn.Sequential(nn.Identity()), input_dimension
    layers: list[nn.Module] = []
    previous = input_dimension
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    return nn.Sequential(*layers), previous


def _mlp(input_dimension: int, hidden_sizes: tuple[int, ...], output_dimension: int) -> nn.Sequential:
    backbone, previous = _mlp_backbone(input_dimension, hidden_sizes)
    return nn.Sequential(backbone, nn.Linear(previous, output_dimension))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
