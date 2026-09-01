"""Gym 0.26 environment for the leak-free F10 SAC runtime."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import gym
import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.scenario import PedestrianMode, load_scenario

from .action_mapping import SACActionMapper
from .belief_runtime import F10BeliefRuntimeFactory
from .f10_protocol import F10Protocol, load_f10_protocol
from .policy_observation import FixedObservationNormalizer, PolicyObservation
from .reward import F10RewardConfig, F10RewardEvaluator
from .road_observer import F10RoadObserver


class F10GymEnvironment(gym.Env):
    """One-environment SAC interface; privileged truth stays in reward/info."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path,
        *,
        split: str = "training",
        integration_factory: Callable[[GymDuckietownConfig], Any] = create_gym_duckietown,
        belief_runtime_factory: Any | None = None,
    ) -> None:
        super().__init__()
        self.protocol = load_f10_protocol(config_path)
        if split not in {"training", "development", "final_evaluation"}:
            raise ValueError("split must be training, development, or final_evaluation")
        self.split = split
        self._seeds = tuple(getattr(self.protocol.seeds, split))
        self._integration_factory = integration_factory
        self._belief_factory = belief_runtime_factory or F10BeliefRuntimeFactory(self.protocol)
        self._normalizer = FixedObservationNormalizer.from_protocol(self.protocol)
        _, maximum_linear, _, maximum_angular = self.protocol.action_bounds
        self._action_mapper = SACActionMapper(maximum_linear, maximum_angular)
        clip = self.protocol.observation_clip
        self.observation_space = gym.spaces.Box(
            low=-clip,
            high=clip,
            shape=(len(self.protocol.observation_order),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self._integration: Any | None = None
        self._belief_runtime: Any | None = None
        self._reward: F10RewardEvaluator | None = None
        self._belief = None
        self._previous_action = PolicyAction(0.0, 0.0)
        self._last_timestamp_s = 0.0
        self._episode_index = 0
        self._step_count = 0
        self._done = True
        self._road_observer: F10RoadObserver | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None:
            seed = self._seeds[self._episode_index % len(self._seeds)]
            scenario_index = self._episode_index
        else:
            if seed not in self._seeds:
                raise ValueError(f"seed {seed} is not part of the frozen {self.split} split")
            scenario_index = self._seeds.index(seed)
        self._episode_index += 1
        self._close_integration()

        scenario = scenario_for(self.protocol, seed, scenario_index)
        simulator = self.protocol.raw["simulator"]
        self._integration = self._integration_factory(
            GymDuckietownConfig(
                scenario=scenario,
                domain_randomization=bool(simulator["domain_randomization"]),
                dynamics_randomization=bool(simulator["dynamics_randomization"]),
                frame_rate_hz=int(simulator["frame_rate_hz"]),
                frame_skip=int(simulator["frame_skip"]),
                maximum_steps=int(simulator["episode_horizon_steps"]) + 2,
                camera_width=int(simulator["camera_width_px"]),
                camera_height=int(simulator["camera_height_px"]),
            )
        )
        raw_observation = self._integration.agent.reset(seed=seed)
        self._road_observer = F10RoadObserver(
            scenario,
            map_tile_size_m=float(
                self.protocol.raw["road_observer"]["map_tile_size_m"]
            ),
        )
        observation = _with_road(raw_observation, self._road_observer.reset())
        self._belief_runtime = self._belief_factory.create(self._integration)
        belief_step = self._belief_runtime.reset(
            observation,
            dt_s=1.0 / float(simulator["frame_rate_hz"]),
        )
        self._belief = belief_step.belief
        self._reward = F10RewardEvaluator(
            F10RewardConfig.from_protocol(self.protocol),
            route_heading_rad=scenario.stop_line.route_heading_rad,
        )
        # Privileged truth is first touched only after RGB->YOLO->belief above.
        self._reward.reset(observation, self._integration.privileged.read())
        self._previous_action = PolicyAction(0.0, 0.0)
        self._last_timestamp_s = 0.0
        self._step_count = 0
        self._done = False
        vector = self._policy_vector()
        return vector, {
            "seed": seed,
            "scenario": scenario.pedestrian.mode.value,
            "perception": _perception_info(belief_step.diagnostics),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done or self._integration is None or self._belief_runtime is None or self._reward is None:
            raise RuntimeError("reset must be called before step, and after episode completion")
        mapping = self._action_mapper.map(action)
        transition = self._integration.agent.step(mapping.policy_action)
        diagnostics = self._integration.diagnostics.read()
        dt_s = diagnostics.timestamp_s - self._last_timestamp_s
        if dt_s <= 0.0:
            dt_s = 1.0 / float(self.protocol.raw["simulator"]["frame_rate_hz"])
        if self._road_observer is None:  # pragma: no cover - reset invariant
            raise RuntimeError("road observer is unavailable")
        observation = _with_road(
            transition.observation,
            self._road_observer.update(transition.observation.ego, dt_s=dt_s),
        )
        belief_step = self._belief_runtime.update(
            observation,
            previous_action=mapping.policy_action,
            dt_s=dt_s,
        )
        self._belief = belief_step.belief
        self._step_count += 1
        horizon = self._step_count >= int(
            self.protocol.raw["simulator"]["episode_horizon_steps"]
        )
        # Evaluation/reward boundary: truth is read only after perception.
        outcome = self._reward.evaluate(
            action=mapping.policy_action,
            observation=observation,
            privileged=self._integration.privileged.read(),
            simulator_terminated=transition.terminated,
            simulator_truncated=transition.truncated,
            simulator_done_code=diagnostics.done_code,
            horizon_reached=horizon,
        )
        self._previous_action = mapping.policy_action
        self._last_timestamp_s = diagnostics.timestamp_s
        self._done = outcome.terminated or outcome.truncated
        return (
            self._policy_vector(),
            float(outcome.reward),
            outcome.terminated,
            outcome.truncated,
            _step_info(mapping, diagnostics, belief_step.diagnostics, outcome),
        )

    def close(self) -> None:
        self._close_integration()
        self._done = True

    def _policy_vector(self) -> np.ndarray:
        if self._belief is None:
            raise RuntimeError("policy belief is unavailable before reset")
        semantic = PolicyObservation.from_belief(
            self._belief,
            self._previous_action,
        )
        return self._normalizer.normalize(semantic)

    def _close_integration(self) -> None:
        if self._integration is not None:
            self._integration.close()
        self._integration = None
        self._belief_runtime = None
        self._reward = None
        self._road_observer = None


def scenario_for(protocol: F10Protocol, seed: int, scenario_index: int):
    distribution = protocol.raw["scenario_distribution"]
    modes = tuple(PedestrianMode(value) for value in distribution["pedestrian_modes"])
    base = load_scenario(protocol.scenario_path).with_pedestrian_mode(
        modes[scenario_index % len(modes)]
    )
    rng = np.random.default_rng(seed + 100_003 * scenario_index)
    pose = base.ego_start_pose_m
    return replace(
        base,
        seed=seed,
        ego_start_pose_m=(
            float(rng.uniform(*distribution["start_longitudinal_range_m"])),
            pose[1],
            pose[2] + float(rng.uniform(*distribution["start_lateral_offset_range_m"])),
        ),
        ego_heading_rad=base.ego_heading_rad
        + float(rng.uniform(*distribution["start_heading_offset_range_rad"])),
    )


def _with_road(observation: SensorObservation, road: Any) -> SensorObservation:
    return SensorObservation(
        front_rgb=observation.front_rgb,
        ego=observation.ego,
        road=road,
    )


def _perception_info(diagnostics: Any) -> dict[str, Any]:
    return {
        "duckie_detection_count": diagnostics.duckie_detection_count,
        "duplicate_selection": diagnostics.duplicate_selection,
        "stop_sign_detection_count": diagnostics.stop_sign_detection_count,
        "projection_error": diagnostics.projection_error,
        "frame_mode": diagnostics.frame_mode,
        "measurement_accepted": diagnostics.measurement_accepted,
        "nis": diagnostics.nis,
    }


def _step_info(mapping: Any, diagnostics: Any, perception: Any, outcome: Any) -> dict[str, Any]:
    reward_terms = outcome.reward_terms
    return {
        "reward_progress": reward_terms.progress,
        "reward_lane": reward_terms.lane,
        "reward_stop": reward_terms.stop,
        "reward_pedestrian": reward_terms.pedestrian,
        "reward_comfort": reward_terms.comfort,
        "reward_terminal": reward_terms.terminal,
        "progress_m": outcome.progress_m,
        "pedestrian_clearance_m": outcome.pedestrian_clearance_m,
        "collision": outcome.pedestrian_collision,
        "unsafe_proximity": outcome.unsafe_proximity,
        "lane_departure": outcome.lane_departure,
        "stop_completed": outcome.stop_completed,
        "stop_violation": outcome.stop_violation,
        "invalid_pose": outcome.invalid_pose,
        "termination_reason": outcome.termination_reason,
        "truncation_reason": outcome.truncation_reason,
        "v_cmd": mapping.policy_action.linear_velocity_mps,
        "omega_cmd": mapping.policy_action.angular_velocity_rad_s,
        "normalized_action_clipped": mapping.clipped,
        "v_actual": diagnostics.actual_motion.linear_velocity_mps,
        "omega_actual": diagnostics.actual_motion.yaw_rate_rad_s,
        "done_code": diagnostics.done_code,
        "perception": _perception_info(perception),
    }
