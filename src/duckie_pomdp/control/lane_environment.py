"""Gym interface for the real-simulator F10-L1 lane curriculum."""

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
from duckie_pomdp.control.lane_policy_observation import (
    LaneObservationNormalizer,
    LanePolicyObservation,
)
from duckie_pomdp.control.lane_protocol import LaneProtocol, load_lane_protocol
from duckie_pomdp.control.lane_reward import LaneRewardConfig, LaneRewardEvaluator
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation


class LaneCurriculumEnvironment(gym.Env):
    """Lane-only SAC environment with a strict truth/reward boundary."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path,
        *,
        split: str = "training",
        integration_factory: Callable[[GymDuckietownConfig], Any] = create_gym_duckietown,
    ) -> None:
        super().__init__()
        self.protocol: LaneProtocol = load_lane_protocol(config_path)
        if split not in {"training", "development", "final_evaluation"}:
            raise ValueError("invalid F10-L1 split")
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
        self._integration: Any | None = None
        self._reward: LaneRewardEvaluator | None = None
        self._sensor: SensorObservation | None = None
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
        del options
        if seed is None:
            seed = self._seeds[self._episode_index % len(self._seeds)]
        elif seed not in self._seeds:
            raise ValueError(f"seed {seed} is not in the frozen {self.split} split")
        self._episode_index += 1
        self._close_integration()
        simulator = self.protocol.raw["simulator"]
        start_pose = _sample_start_pose(self.protocol, seed)
        self._integration = self._integration_factory(
            GymDuckietownConfig(
                map_name=str(simulator["map"]),
                seed=seed,
                domain_randomization=bool(simulator["domain_randomization"]),
                dynamics_randomization=bool(simulator["dynamics_randomization"]),
                frame_rate_hz=int(simulator["frame_rate_hz"]),
                frame_skip=int(simulator["frame_skip"]),
                maximum_steps=int(simulator["episode_horizon_steps"]) + 2,
                camera_width=int(simulator["camera_width_px"]),
                camera_height=int(simulator["camera_height_px"]),
                headless=True,
                start_tile=tuple(int(value) for value in simulator["start_tile"]),
                start_pose=start_pose,
            )
        )
        self._sensor = self._integration.agent.reset(seed=seed)
        self._reward = LaneRewardEvaluator(
            LaneRewardConfig.from_protocol(self.protocol),
            dt_s=1.0 / float(simulator["frame_rate_hz"]),
        )
        # Privileged pose is consumed only by the reward/lap boundary.
        self._reward.reset(self._integration.privileged.read().ego_world_pose)
        self._previous_action = PolicyAction(0.0, 0.0)
        self._step_count = 0
        self._done = False
        return self._policy_vector(), {
            "seed": seed,
            "scenario": "small_loop_counterclockwise",
            "start_pose": {
                "local_x_m": start_pose[0][0],
                "local_z_m": start_pose[0][2],
                "heading_rad": start_pose[1],
            },
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done or self._integration is None or self._reward is None:
            raise RuntimeError("reset is required before step and after completion")
        mapping = self._action_mapper.map(action)
        transition = self._integration.agent.step(mapping.policy_action)
        self._sensor = transition.observation
        diagnostics = self._integration.diagnostics.read()
        self._step_count += 1
        horizon = self._step_count >= int(
            self.protocol.raw["simulator"]["episode_horizon_steps"]
        )
        outcome = self._reward.evaluate(
            action=mapping.policy_action,
            observation=self._sensor,
            world_pose=diagnostics.world_pose,
            simulator_terminated=transition.terminated,
            simulator_truncated=transition.truncated,
            simulator_done_code=diagnostics.done_code,
            horizon_reached=horizon,
        )
        self._previous_action = mapping.policy_action
        self._done = outcome.terminated or outcome.truncated
        terms = outcome.reward_terms
        info = {
            "reward_progress": terms.progress,
            "reward_lane": terms.lane,
            "reward_yellow": terms.yellow,
            "reward_comfort": terms.comfort,
            "reward_living": terms.living,
            "reward_terminal": terms.terminal,
            "path_length_m": outcome.path_length_m,
            "yellow_clearance_m": outcome.yellow_clearance_m,
            "lap_completed": outcome.lap_completed,
            "yellow_crossing": outcome.yellow_crossing,
            "lane_departure": outcome.lane_departure,
            "invalid_pose": outcome.invalid_pose,
            "termination_reason": outcome.termination_reason,
            "truncation_reason": outcome.truncation_reason,
            "v_cmd": mapping.policy_action.linear_velocity_mps,
            "omega_cmd": mapping.policy_action.angular_velocity_rad_s,
            "normalized_action_clipped": mapping.clipped,
            "v_actual": diagnostics.actual_motion.linear_velocity_mps,
            "omega_actual": diagnostics.actual_motion.yaw_rate_rad_s,
            "lateral_error_m": self._sensor.ego.lateral_error_m,
            "heading_error_rad": self._sensor.ego.heading_error_rad,
            "done_code": diagnostics.done_code,
        }
        return (
            self._policy_vector(),
            outcome.reward,
            outcome.terminated,
            outcome.truncated,
            info,
        )

    def latest_rgb(self) -> np.ndarray:
        if self._sensor is None:
            raise RuntimeError("RGB is unavailable before reset")
        return np.asarray(self._sensor.front_rgb).copy()

    def close(self) -> None:
        self._close_integration()
        self._done = True

    def _policy_vector(self) -> np.ndarray:
        if self._sensor is None:
            raise RuntimeError("lane observation is unavailable before reset")
        semantic = LanePolicyObservation.from_sensor(
            self._sensor, self._previous_action
        )
        return self._normalizer.normalize(semantic)

    def _close_integration(self) -> None:
        if self._integration is not None:
            self._integration.close()
        self._integration = None
        self._reward = None
        self._sensor = None


def _sample_start_pose(
    protocol: LaneProtocol, seed: int
) -> tuple[tuple[float, float, float], float]:
    simulator = protocol.raw["simulator"]
    rng = np.random.default_rng(seed)
    local_x = float(simulator["base_local_x_m"]) + float(
        rng.uniform(*simulator["start_longitudinal_jitter_m"])
    )
    local_z = float(simulator["base_local_z_m"]) + float(
        rng.uniform(*simulator["start_lateral_jitter_m"])
    )
    heading = float(simulator["base_heading_rad"]) + float(
        rng.uniform(*simulator["start_heading_jitter_rad"])
    )
    return ((local_x, 0.0, local_z), heading)

