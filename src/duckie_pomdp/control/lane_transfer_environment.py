"""Real-simulator Gym environment for the F10-L2 map-transfer curriculum."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gym
import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.control.action_mapping import SACActionMapper
from duckie_pomdp.control.lane_environment import LaneCurriculumEnvironment
from duckie_pomdp.control.lane_policy_observation import LaneObservationNormalizer
from duckie_pomdp.control.lane_transfer_protocol import load_lane_transfer_protocol
from duckie_pomdp.domain.action import PolicyAction


class LaneTransferEnvironment(LaneCurriculumEnvironment):
    """F10-L2 specialization reusing the tested lane reset/step behavior."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        split: str = "training",
        integration_factory: Callable[[GymDuckietownConfig], Any] = create_gym_duckietown,
    ) -> None:
        # Only the protocol seam differs from F10-L1. Parent reset, step,
        # action, reward, termination, and observation code remains shared.
        gym.Env.__init__(self)
        self.protocol = load_lane_transfer_protocol(config_path)
        if split not in {"training", "development", "final_evaluation"}:
            raise ValueError("invalid F10-L2 split")
        self.split = split
        self._seeds = tuple(getattr(self.protocol.seeds, split))
        self._integration_factory = integration_factory
        self._normalizer = LaneObservationNormalizer.from_protocol(self.protocol)
        _, max_linear, _, max_angular = self.protocol.action_bounds
        self._action_mapper = SACActionMapper(max_linear, max_angular)
        clip = self.protocol.observation_clip
        self.observation_space = gym.spaces.Box(
            low=-clip,
            high=clip,
            shape=(len(self.protocol.observation_order),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._integration = None
        self._reward = None
        self._sensor = None
        self._previous_action = PolicyAction(0.0, 0.0)
        self._episode_index = 0
        self._step_count = 0
        self._done = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        info["scenario"] = "experiment_loop_mixed_turns"
        return observation, info

