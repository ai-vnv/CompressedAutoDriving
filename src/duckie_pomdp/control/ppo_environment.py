"""Real-simulator fixed-observation environment for staged F10-PPO."""

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
from duckie_pomdp.domain.belief import BeliefState, LaneBelief, RoadBelief
from duckie_pomdp.domain.observation import RoadMeasurement, SensorObservation
from duckie_pomdp.domain.state import StopMode
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
)
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.scenario import (
    PedestrianMode,
    forward_route_distance_m,
    load_scenario,
)

from .action_mapping import NormalizedActionMapper
from .belief_runtime import F10BeliefRuntimeFactory
from .lane_belief_runtime import VisualLaneBeliefRuntime
from .lane_belief_uncertainty import resolve_runtime_calibration
from .ppo_observation import (
    PPOFixedObservationNormalizer,
    PPOPolicyObservation,
    PPOVisualPolicyObservation,
    neutral_pedestrian,
    neutral_stop_sign,
    policy_observation_from_belief,
)
from .ppo_protocol import (
    CurriculumStage,
    PPOCurriculumProtocol,
    load_ppo_curriculum_protocol,
    require_stage_in_protocol_scope,
)
from .ppo_reward import PPORewardEvaluator
from .road_observer import F10RoadObserver
from .start_sampler import (
    LoopStartSampler,
    load_small_loop_tiles,
    resolve_start_randomisation_enabled,
)
from .stop_belief import RuntimeStopBeliefUpdater


class PPOCurriculumEnvironment(gym.Env):
    """Fixed actor/critic boundary for every stage of one frozen protocol."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path,
        *,
        stage: str,
        split: str = "training",
        seeds: tuple[int, ...] | None = None,
        integration_factory: Callable[[GymDuckietownConfig], Any] = create_gym_duckietown,
        belief_runtime_factory: Any | None = None,
    ) -> None:
        super().__init__()
        self.protocol = load_ppo_curriculum_protocol(config_path)
        require_stage_in_protocol_scope(self.protocol, stage)
        self.stage: CurriculumStage = self.protocol.stage(stage)
        if seeds is None:
            if split not in {"training", "development", "stage_final"}:
                raise ValueError("explicit seeds are required outside frozen stage splits")
            seeds = self.stage.seeds_for(split)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("environment seeds must be non-empty and unique")
        self.split = split
        self._seeds = tuple(int(value) for value in seeds)
        self._integration_factory = integration_factory
        self._visual_lane_enabled = (
            str(
                self.protocol.raw["observation"].get(
                    "representation", "legacy_state_v1"
                )
            )
            == "visual_lane_belief_v2"
        )
        self._belief_factory = belief_runtime_factory
        if (
            self._belief_factory is None
            and (self.stage.map_name == "pomdp_v1" or self._visual_lane_enabled)
        ):
            self._belief_factory = F10BeliefRuntimeFactory(self.protocol)
        # Off (False) unless [v4_changes].start_randomisation is explicitly
        # true in this protocol's raw TOML -- v3 configs carry no
        # [v4_changes] table and therefore always resolve to False,
        # reproducing v3's single-tile start distribution exactly. Scoped to
        # the small_loop stage only (Task 2's brief); any other native map
        # keeps using [native_start] unconditionally.
        self._start_randomisation_enabled = resolve_start_randomisation_enabled(
            self.protocol.raw
        )
        self._loop_start_sampler: LoopStartSampler | None = None
        self._normalizer = PPOFixedObservationNormalizer(self.protocol)
        self._action_mapper = NormalizedActionMapper(
            self.protocol.action_bounds[1], self.protocol.action_bounds[3]
        )
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
        self._integration_has_scenario: bool | None = None
        self._belief_runtime: Any | None = None
        self._lane_runtime: VisualLaneBeliefRuntime | None = None
        self._stop_updater: RuntimeStopBeliefUpdater | None = None
        self._road_observer: F10RoadObserver | None = None
        self._reward: PPORewardEvaluator | None = None
        self._sensor: SensorObservation | None = None
        self._belief: BeliefState | None = None
        self._semantic: PPOPolicyObservation | PPOVisualPolicyObservation | None = None
        self._previous_action = PolicyAction(0.0, 0.0)
        self._episode_index = 0
        self._step_count = 0
        self._last_timestamp_s = 0.0
        self._done = True
        self._pedestrian_mode: str | None = None
        self._pedestrian_speed_mps: float | None = None
        self._pedestrian_training_phase: int | None = None
        self._pedestrian_start_delay_s: float | None = None
        self._c2_rehearsal_no_pedestrian = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None:
            scenario_index = self._episode_index % len(self._seeds)
            seed = self._seeds[scenario_index]
            # A monotonically increasing counter, not the cycling
            # scenario_index -- this is what lets the loop-wide start
            # sampler (Task 2) draw hundreds of distinct start poses across
            # a training run instead of repeating with the seed list's
            # period.
            pose_episode_index = self._episode_index
            self._episode_index += 1
        elif seed not in self._seeds:
            raise ValueError(f"seed {seed} is not allowed for this evaluation boundary")
        else:
            # Explicit evaluation seeds define the complete scenario.  The
            # result must not depend on which checkpoint or baseline ran first.
            scenario_index = self._seeds.index(seed)
            pose_episode_index = scenario_index
        config, scenario = self._integration_config(seed, scenario_index, pose_episode_index)
        current_is_scenario = bool(self._integration_has_scenario)
        reuse_native = (
            self._integration is not None
            and scenario is None
            and not current_is_scenario
        )
        reuse_scenario = (
            self._integration is not None
            and scenario is not None
            and current_is_scenario
        )
        if reuse_native:
            self._clear_episode_state()
            self._integration.reconfigure_native_episode(config)
        elif reuse_scenario:
            self._clear_episode_state()
            self._integration.reconfigure_scenario_episode(config)
        else:
            self._close_integration()
            self._integration = self._integration_factory(config)
            self._integration_has_scenario = scenario is not None
        assert self._integration is not None
        raw = self._integration.agent.reset(seed=seed)
        simulator = self.protocol.raw["simulator"]
        dt_s = 1.0 / float(simulator["frame_rate_hz"])
        lane_belief: LaneBelief | None = None
        if self._visual_lane_enabled:
            if self.protocol.lane_belief_config_path is None:
                raise RuntimeError("visual lane protocol has no lane belief config")
            self._lane_runtime = VisualLaneBeliefRuntime(
                CalibratedGroundProjector(self._integration.camera_calibration.read()),
                config_path=self.protocol.lane_belief_config_path,
                # Off (None) unless [v4_changes].belief_uncertainty_refit is
                # explicitly true in this protocol's raw TOML -- v3 configs
                # carry no [v4_changes] table and therefore always resolve
                # to None, reproducing v3's reported sigma bit-for-bit.
                uncertainty_calibration=resolve_runtime_calibration(
                    self.protocol.raw, self.protocol.lane_belief_config_path
                ),
            )
            lane_belief = self._lane_runtime.reset(raw.front_rgb).belief
        if scenario is None:
            road = RoadMeasurement(
                curvature_inv_m=0.0,
                stop_line_distance_m=float(
                    self.protocol.raw["neutral"]["stop_line_distance_m"]
                ),
            )
            self._sensor = _with_road(raw, road)
        else:
            self._road_observer = F10RoadObserver(
                scenario,
                map_tile_size_m=float(self.protocol.raw["pomdp_start"]["map_tile_size_m"]),
                initial_stop_line_distance_m=self._configured_route_stop_distance(
                    scenario
                ),
            )
            self._sensor = _with_road(raw, self._road_observer.reset())
        if self._belief_factory is None:
            self._belief = self._neutral_belief(self._sensor, lane_belief)
        else:
            assert self._belief_factory is not None
            self._belief_runtime = self._belief_factory.create(self._integration)
            belief_step = self._belief_runtime.reset(self._sensor, dt_s=dt_s)
            self._stop_updater = self._make_stop_updater()
            stop_step = self._stop_updater.reset()
            stop_step = self._stop_updater.update(
                belief_step.diagnostics.stop_sign_detections,
                stop_line_distance_m=self._sensor.road.stop_line_distance_m,
                ego=self._sensor.ego,
            )
            self._belief = self._compose_belief(
                belief_step.belief,
                stop_step.belief,
                stop_step.mode,
                lane_belief,
            )

        self._previous_action = PolicyAction(0.0, 0.0)
        self._step_count = 0
        self._last_timestamp_s = 0.0
        self._done = False
        vector = self._policy_vector()
        self._reward = PPORewardEvaluator(
            self.protocol,
            self.stage,
            dt_s=dt_s,
            route_heading_rad=(None if scenario is None else scenario.stop_line.route_heading_rad),
        )
        # Reward/evaluation truth is first read after the complete policy
        # observation has been constructed.
        privileged = self._integration.privileged.read()
        self._reward.reset(self._sensor, privileged)
        return vector, {
            "stage": self.stage.key,
            "scenario": self.stage.name,
            "seed": seed,
            "scenario_index": scenario_index,
            "pedestrian_mode": self._pedestrian_mode,
            "pedestrian_speed_mps": self._pedestrian_speed_mps,
            "pedestrian_training_phase": self._pedestrian_training_phase,
            "pedestrian_start_delay_s": self._pedestrian_start_delay_s,
            "c2_rehearsal_no_pedestrian": self._c2_rehearsal_no_pedestrian,
            "policy": self._policy_info(),
            # Explicitly evaluation/teacher-only.  This mapping is returned
            # beside, never inside, the fixed policy vector.  Normal PPO
            # inference receives only ``vector``; an offline guided rollout
            # may use this truth to label an action for that public belief.
            "evaluation_gt": _evaluation_gt(privileged),
            "stop_completed": False,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done or self._integration is None or self._sensor is None or self._reward is None:
            raise RuntimeError("reset is required before step and after completion")
        mapping = self._action_mapper.map(action)
        transition = self._integration.agent.step(mapping.policy_action)
        diagnostics = self._integration.diagnostics.read()
        dt_s = diagnostics.timestamp_s - self._last_timestamp_s
        if dt_s <= 0.0:
            dt_s = 1.0 / float(self.protocol.raw["simulator"]["frame_rate_hz"])

        lane_belief = None
        lane_info: dict[str, Any] = {}
        if self._lane_runtime is not None:
            lane_step = self._lane_runtime.update(
                transition.observation.front_rgb,
                actual_ego_motion=diagnostics.actual_motion,
                dt_s=dt_s,
            )
            lane_belief = lane_step.belief
            lane_info = {
                "lane_detected": lane_step.measurement.detected,
                "lane_visible_point_count": lane_step.measurement.visible_point_count,
                "lane_validity_probability": lane_step.belief.validity_probability,
            }
            boundary = lane_step.boundary_diagnostics
            if boundary is not None:
                lane_info.update(
                    {
                        "lane_boundary_source": boundary.source,
                        "lane_boundary_disagreement_m": boundary.boundary_disagreement_m,
                        "lane_strict_yellow_pixel_count": boundary.strict_yellow_pixel_count,
                        "lane_strict_white_pixel_count": boundary.strict_white_pixel_count,
                        "lane_adaptive_unknown_pixel_count": boundary.adaptive_unknown_pixel_count,
                    }
                )

        if self._road_observer is None:
            road = RoadMeasurement(
                0.0,
                float(self.protocol.raw["neutral"]["stop_line_distance_m"]),
            )
            self._sensor = _with_road(transition.observation, road)
        else:
            self._sensor = _with_road(
                transition.observation,
                self._road_observer.update(transition.observation.ego, dt_s=dt_s),
            )

        perception_info: dict[str, Any] = dict(lane_info)
        if self._belief_runtime is None:
            self._belief = self._neutral_belief(self._sensor, lane_belief)
        else:
            assert self._stop_updater is not None
            belief_step = self._belief_runtime.update(
                self._sensor,
                previous_action=mapping.policy_action,
                dt_s=dt_s,
            )
            stop_step = self._stop_updater.update(
                belief_step.diagnostics.stop_sign_detections,
                stop_line_distance_m=self._sensor.road.stop_line_distance_m,
                ego=self._sensor.ego,
            )
            self._belief = self._compose_belief(
                belief_step.belief,
                stop_step.belief,
                stop_step.mode,
                lane_belief,
            )
            perception_info.update(
                {
                    "duckie_detection_count": belief_step.diagnostics.duckie_detection_count,
                    "duplicate_selection": belief_step.diagnostics.duplicate_selection,
                    "pedestrian_measurement_accepted": belief_step.diagnostics.measurement_accepted,
                    "pedestrian_nis": belief_step.diagnostics.nis,
                    "stop_sign_detection_count": belief_step.diagnostics.stop_sign_detection_count,
                    "stop_sign_selected_confidence": stop_step.selected_confidence,
                    "stop_sign_projection_error": stop_step.projection_error,
                }
            )

        self._step_count += 1
        horizon = self._step_count >= self.stage.episode_horizon_steps
        self._previous_action = mapping.policy_action
        vector = self._policy_vector()
        privileged = self._integration.privileged.read()
        outcome = self._reward.evaluate(
            action=mapping.policy_action,
            observation=self._sensor,
            privileged=privileged,
            world_pose=diagnostics.world_pose,
            simulator_terminated=transition.terminated,
            simulator_truncated=transition.truncated,
            simulator_done_code=diagnostics.done_code,
            horizon_reached=horizon,
        )
        self._last_timestamp_s = diagnostics.timestamp_s
        self._done = outcome.terminated or outcome.truncated
        terms = outcome.terms
        info = {
            "stage": self.stage.key,
            "scenario": self.stage.name,
            "pedestrian_mode": self._pedestrian_mode,
            "pedestrian_speed_mps": self._pedestrian_speed_mps,
            "pedestrian_training_phase": self._pedestrian_training_phase,
            "pedestrian_start_delay_s": self._pedestrian_start_delay_s,
            "reward_progress": terms.progress,
            "reward_lane": terms.lane,
            "reward_pedestrian": terms.pedestrian,
            "reward_stop": terms.stop,
            "reward_smoothness": terms.smoothness,
            "reward_terminal": terms.terminal,
            "progress_m": outcome.progress_m,
            "completed": outcome.completed,
            "lane_failure": outcome.lane_failure,
            "yellow_contact": outcome.yellow_contact,
            "yellow_recovery_started": outcome.yellow_recovery_started,
            "yellow_recovery_active": outcome.yellow_recovery_active,
            "yellow_recovered": outcome.yellow_recovered,
            "invalid_pose": outcome.invalid_pose,
            "collision": outcome.collision,
            "unsafe_proximity": outcome.unsafe_proximity,
            "pedestrian_clearance_m": outcome.pedestrian_clearance_m,
            "stop_completed": outcome.stop_completed,
            "stop_violation": outcome.stop_violation,
            "v_cmd": mapping.policy_action.linear_velocity_mps,
            "omega_cmd": mapping.policy_action.angular_velocity_rad_s,
            "v_actual": diagnostics.actual_motion.linear_velocity_mps,
            "omega_actual": diagnostics.actual_motion.yaw_rate_rad_s,
            "termination_reason": outcome.termination_reason,
            "truncation_reason": outcome.truncation_reason,
            "normalized_action_clipped": mapping.clipped,
            "policy": self._policy_info(),
            "perception": perception_info,
            "evaluation_gt": _evaluation_gt(privileged),
        }
        return vector, outcome.reward, outcome.terminated, outcome.truncated, info

    @property
    def current_belief(self) -> BeliefState:
        if self._belief is None:
            raise RuntimeError("belief is unavailable before reset")
        return self._belief

    def latest_rgb(self) -> np.ndarray:
        if self._sensor is None:
            raise RuntimeError("RGB is unavailable before reset")
        return np.asarray(self._sensor.front_rgb).copy()

    def close(self) -> None:
        self._close_integration()
        self._done = True

    def _integration_config(
        self,
        seed: int,
        scenario_index: int,
        episode_index: int | None = None,
    ):
        simulator = self.protocol.raw["simulator"]
        common = dict(
            seed=seed,
            domain_randomization=self.stage.domain_randomization,
            dynamics_randomization=bool(simulator["dynamics_randomization"]),
            frame_rate_hz=int(simulator["frame_rate_hz"]),
            frame_skip=int(simulator["frame_skip"]),
            maximum_steps=self.stage.episode_horizon_steps + 2,
            camera_width=int(simulator["camera_width_px"]),
            camera_height=int(simulator["camera_height_px"]),
            headless=True,
        )
        scenario_path = self.stage.scenario_config_path
        self._c2_rehearsal_no_pedestrian = False
        if scenario_path is None and self.stage.map_name == "pomdp_v1":
            scenario_path = self.protocol.scenario_path
        if scenario_path is None:
            self._pedestrian_mode = None
            self._pedestrian_speed_mps = None
            self._pedestrian_training_phase = None
            self._pedestrian_start_delay_s = None
            start = self.protocol.raw["native_start"]
            # Native closed-loop stages use the same deterministic loop-wide
            # sampler. Restricting this to small_loop left C1 anchored to one
            # tile, so its two right-hand curves were rarely observed before
            # an episode-ending failure.
            if self._start_randomisation_enabled:
                pose_index = scenario_index if episode_index is None else episode_index
                sampled = self._get_loop_start_sampler().sample(pose_index)
                start_tile = sampled.tile
                pose = (
                    (sampled.local_x_m, 0.0, sampled.local_z_m),
                    sampled.heading_rad,
                )
            else:
                rng = np.random.default_rng(seed)
                start_tile = tuple(int(value) for value in start["start_tile"])
                pose = (
                    (
                        float(start["base_local_x_m"]) + float(rng.uniform(*start["start_longitudinal_jitter_m"])),
                        0.0,
                        float(start["base_local_z_m"]) + float(rng.uniform(*start["start_lateral_jitter_m"])),
                    ),
                    float(start["base_heading_rad"]) + float(rng.uniform(*start["start_heading_jitter_rad"])),
                )
            return GymDuckietownConfig(
                map_name=self.stage.map_name,
                start_tile=start_tile,
                start_pose=pose,
                **common,
            ), None

        modes = tuple(PedestrianMode(value) for value in self.stage.pedestrian_modes)
        mode_index = scenario_index % len(modes)
        rehearsal = self.protocol.raw.get("curriculum_rehearsal", {}).get("c2", {})
        if self.stage.key == "c2" and self.split == "training" and rehearsal:
            if episode_index is None:
                raise RuntimeError("C2 rehearsal requires an episode index")
            period = int(rehearsal["period_episodes"])
            offsets = tuple(int(value) for value in rehearsal["no_pedestrian_offsets"])
            if period <= 0 or not offsets or any(value < 0 or value >= period for value in offsets):
                raise ValueError("invalid C2 rehearsal schedule")
            episode_offset = episode_index % period
            self._c2_rehearsal_no_pedestrian = episode_offset in offsets
            active_offsets = tuple(
                value for value in range(period) if value not in offsets
            )
            if len(active_offsets) != len(modes):
                raise ValueError(
                    "C2 rehearsal active offsets must map one-to-one to pedestrian modes"
                )
            if not self._c2_rehearsal_no_pedestrian:
                mode_index = active_offsets.index(episode_offset)
        base = load_scenario(scenario_path).with_pedestrian_mode(modes[mode_index])
        self._pedestrian_mode = base.pedestrian.mode.value
        self._pedestrian_speed_mps = float(base.pedestrian.speed_mps)
        self._pedestrian_training_phase = None
        self._pedestrian_start_delay_s = float(base.pedestrian.start_delay_for_mode())
        curriculum = self.protocol.raw.get("pedestrian_training_curriculum")
        if (
            self.stage.key == "c2"
            and self.split == "training"
            and curriculum
            and not self._c2_rehearsal_no_pedestrian
        ):
            if episode_index is None:
                raise RuntimeError("C2 training curriculum requires an episode index")
            speeds = tuple(float(value) for value in curriculum["speeds_mps"])
            phase_ends = tuple(int(value) for value in curriculum["phase_end_episodes"])
            if len(speeds) != len(phase_ends) + 1:
                raise ValueError(
                    "pedestrian curriculum needs one more speed than phase boundary"
                )
            if not speeds or any(value <= 0.0 for value in speeds):
                raise ValueError("pedestrian curriculum speeds must be positive")
            if any(right <= left for left, right in zip(phase_ends, phase_ends[1:])):
                raise ValueError("pedestrian curriculum phase boundaries must increase")
            phase = sum(episode_index >= boundary for boundary in phase_ends)
            speed = speeds[phase]
            reverse_delays = tuple(
                float(value)
                for value in curriculum.get("reverse_start_delays_s", ())
            )
            if reverse_delays and len(reverse_delays) != len(speeds):
                raise ValueError(
                    "reverse start-delay curriculum must match speed phases"
                )
            reverse_delay = (
                base.pedestrian.reverse_start_delay_s
                if not reverse_delays
                else reverse_delays[phase]
            )
            base = replace(
                base,
                pedestrian=replace(
                    base.pedestrian,
                    speed_mps=speed,
                    reverse_start_delay_s=reverse_delay,
                ),
            )
            self._pedestrian_training_phase = phase
            self._pedestrian_speed_mps = speed
            self._pedestrian_start_delay_s = float(
                base.pedestrian.start_delay_for_mode()
            )
        rng = np.random.default_rng(seed + 100_003 * scenario_index)
        pose = base.ego_start_pose_m
        start_tile = base.ego_start_tile
        if self._c2_rehearsal_no_pedestrian:
            assert episode_index is not None
            sampler = self._get_loop_start_sampler()
            sampled = sampler.sample(episode_index)
            pose_m = (
                sampled.local_x_m,
                pose[1],
                sampled.local_z_m,
            )
            heading_rad = sampled.heading_rad
            start_tile = sampled.tile
            self._pedestrian_mode = None
            self._pedestrian_speed_mps = None
            self._pedestrian_training_phase = None
            self._pedestrian_start_delay_s = None
        elif self.stage.scenario_config_path is None:
            start = self.protocol.raw["pomdp_start"]
            pose_m = (
                float(rng.uniform(*start["start_longitudinal_range_m"])),
                pose[1],
                pose[2]
                + float(rng.uniform(*start["start_lateral_offset_range_m"])),
            )
            heading_rad = base.ego_heading_rad + float(
                rng.uniform(*start["start_heading_offset_range_rad"])
            )
        else:
            start = self.protocol.raw["object_start"]
            longitudinal_m = float(
                rng.uniform(*start["longitudinal_jitter_m"])
            )
            lateral_m = float(rng.uniform(*start["lateral_jitter_m"]))
            heading_rad = base.ego_heading_rad + float(
                rng.uniform(*start["heading_jitter_rad"])
            )
            direction_x = float(np.cos(base.ego_heading_rad))
            direction_z = float(-np.sin(base.ego_heading_rad))
            right_x, right_z = -direction_z, direction_x
            pose_m = (
                pose[0] + longitudinal_m * direction_x + lateral_m * right_x,
                pose[1],
                pose[2] + longitudinal_m * direction_z + lateral_m * right_z,
            )
        scenario = replace(
            base,
            seed=seed,
            ego_start_tile=start_tile,
            ego_start_pose_m=pose_m,
            ego_heading_rad=heading_rad,
        )
        return GymDuckietownConfig(
            scenario=scenario,
            scenario_pedestrian_enabled=(
                self.stage.pedestrian_active
                and not self._c2_rehearsal_no_pedestrian
            ),
            scenario_stop_sign_enabled=self.stage.stop_active,
            **common,
        ), scenario

    def _get_loop_start_sampler(self) -> LoopStartSampler:
        """Lazily build the loop-wide sampler -- never touched when the flag is off."""

        if self._loop_start_sampler is None:
            scenario_path = self.stage.scenario_config_path
            if self._c2_rehearsal_no_pedestrian and scenario_path is not None:
                scenario = load_scenario(scenario_path)
                map_name = str(scenario.map_path)
                anchor_tile = scenario.ego_start_tile
                anchor_heading = scenario.ego_heading_rad
            else:
                start = self.protocol.raw["native_start"]
                map_name = self.stage.map_name
                anchor_tile = tuple(int(value) for value in start["start_tile"])
                anchor_heading = float(start["base_heading_rad"])
            tiles = load_small_loop_tiles(
                map_name=map_name,
                anchor_tile=anchor_tile,
                anchor_heading_rad=anchor_heading,
            )
            self._loop_start_sampler = LoopStartSampler(
                tiles, rng_seed=int(self._seeds[0])
            )
        return self._loop_start_sampler

    def _configured_route_stop_distance(self, scenario) -> float:
        """Resolve spawn-to-stop arc length from the configured route only."""

        start = self.protocol.raw["native_start"]
        tiles = load_small_loop_tiles(
            map_name=str(scenario.map_path),
            anchor_tile=tuple(int(value) for value in start["start_tile"]),
            anchor_heading_rad=float(start["base_heading_rad"]),
        )
        tile_size_m = float(self.protocol.raw["pomdp_start"]["map_tile_size_m"])
        local_x, _, local_z = scenario.ego_start_pose_m
        tile_x, tile_z = scenario.ego_start_tile
        return forward_route_distance_m(
            tiles,
            start_world=(
                tile_x * tile_size_m + local_x,
                tile_z * tile_size_m + local_z,
            ),
            destination_world=(
                scenario.stop_line.world_x_m,
                scenario.stop_line.world_z_m,
            ),
        )

    def _make_stop_updater(self) -> RuntimeStopBeliefUpdater:
        assert self._integration is not None
        projector = YoloMeasurementProjector(
            CalibratedGroundProjector(self._integration.camera_calibration.read()),
            MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
        )
        return RuntimeStopBeliefUpdater(
            self.protocol, projector, active=self.stage.stop_active
        )

    def _neutral_belief(
        self, sensor: SensorObservation, lane: LaneBelief | None
    ) -> BeliefState:
        assert sensor.road is not None
        return BeliefState(
            ego=sensor.ego,
            road=RoadBelief(
                sensor.road.curvature_inv_m,
                sensor.road.stop_line_distance_m,
                StopMode.NONE,
            ),
            stop_sign=neutral_stop_sign(self.protocol),
            pedestrian=neutral_pedestrian(self.protocol),
            lane=lane,
        )

    def _compose_belief(
        self,
        base: BeliefState,
        stop_sign,
        stop_mode: StopMode,
        lane: LaneBelief | None,
    ) -> BeliefState:
        assert self._sensor is not None and self._sensor.road is not None
        return BeliefState(
            ego=self._sensor.ego,
            road=RoadBelief(
                self._sensor.road.curvature_inv_m,
                (
                    self._sensor.road.stop_line_distance_m
                    if self.stage.stop_active
                    else float(
                        self.protocol.raw["neutral"]["stop_line_distance_m"]
                    )
                ),
                stop_mode if self.stage.stop_active else StopMode.NONE,
            ),
            stop_sign=(stop_sign if self.stage.stop_active else neutral_stop_sign(self.protocol)),
            pedestrian=(
                base.pedestrian
                if self.stage.pedestrian_active
                and not self._c2_rehearsal_no_pedestrian
                else neutral_pedestrian(self.protocol)
            ),
            lane=lane,
        )

    def _policy_vector(self) -> np.ndarray:
        if self._belief is None:
            raise RuntimeError("belief is unavailable")
        self._semantic = policy_observation_from_belief(
            self.protocol, self._belief, self._previous_action
        )
        return self._normalizer.normalize(self._semantic)

    def _policy_info(self) -> dict[str, float]:
        if self._semantic is None:
            raise RuntimeError("semantic observation is unavailable")
        return {
            name: float(getattr(self._semantic, name))
            for name in self.protocol.observation_order
        }

    def _close_integration(self) -> None:
        if self._integration is not None:
            self._integration.close()
        self._integration = None
        self._integration_has_scenario = None
        self._clear_episode_state()

    def _clear_episode_state(self) -> None:
        """Drop episode-local beliefs while optionally retaining the simulator."""

        self._belief_runtime = None
        self._lane_runtime = None
        self._stop_updater = None
        self._road_observer = None
        self._reward = None
        self._sensor = None
        self._belief = None
        self._semantic = None


def _with_road(observation: SensorObservation, road: RoadMeasurement) -> SensorObservation:
    return SensorObservation(
        front_rgb=observation.front_rgb,
        ego=observation.ego,
        road=road,
    )


def _evaluation_gt(privileged) -> dict[str, float | bool | None]:
    """Serialize simulator truth for reward/evaluation or an offline teacher.

    The returned mapping is intentionally separate from the policy vector.
    """

    state = privileged.true_pomdp_state
    return {
        "lane_lateral_error_m": state.ego.lateral_error_m,
        "lane_heading_error_rad": state.ego.heading_error_rad,
        "road_curvature_inv_m": state.road.curvature_inv_m,
        "pedestrian_range_m": state.pedestrian.range_m,
        "pedestrian_bearing_rad": state.pedestrian.bearing_rad,
        "stop_line_distance_m": state.road.stop_line_distance_m,
    }
