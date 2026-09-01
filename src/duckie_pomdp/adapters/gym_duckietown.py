"""Gym-Duckietown integration for the agent and privileged-state paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin
from pathlib import Path
from typing import Any

import numpy as np

from duckie_pomdp.adapters.differential_drive import (
    ActionConversion,
    DifferentialDriveActionAdapter,
    DifferentialDriveCalibration,
)
from duckie_pomdp.adapters.ego_motion import (
    SimulatorPoseSample,
    estimate_actual_motion,
)
from duckie_pomdp.adapters.lane_geometry import lane_curvature_inv_m
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, ImagePoint
from duckie_pomdp.domain.observation import EgoObservation, SensorObservation
from duckie_pomdp.domain.privileged import (
    PrivilegedSimulatorState,
    WorldFootprint,
    WorldPoint,
    WorldPose,
    WorldVelocity,
)
from duckie_pomdp.domain.reward import RewardTerms
from duckie_pomdp.domain.state import (
    EgoMotion,
    EgoState,
    POMDPState,
    PedestrianState,
    RoadState,
    StopMode,
    StopSignState,
)
from duckie_pomdp.domain.transition import Transition
from duckie_pomdp.perception.camera_geometry import (
    CameraCalibration,
    world_to_ego,
)
from duckie_pomdp.scenario import MinimalPOMDPScenario, PedestrianMode


@dataclass(frozen=True)
class GymDuckietownConfig:
    map_name: str = "small_loop"
    seed: int = 73
    domain_randomization: bool = False
    dynamics_randomization: bool = False
    frame_rate_hz: int = 30
    frame_skip: int = 1
    maximum_steps: int = 1500
    camera_width: int = 640
    camera_height: int = 480
    headless: bool = True
    start_tile: tuple[int, int] | None = None
    start_pose: tuple[tuple[float, float, float], float] | None = None
    scenario: MinimalPOMDPScenario | None = None
    scenario_pedestrian_enabled: bool = True
    scenario_stop_sign_enabled: bool = True

    def __post_init__(self) -> None:
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if self.maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera dimensions must be positive")
        if (self.start_tile is None) != (self.start_pose is None):
            raise ValueError("start_tile and start_pose must be configured together")


@dataclass(frozen=True)
class SimulatorStepDiagnostics:
    requested_action: PolicyAction
    action_conversion: ActionConversion
    actual_motion: EgoMotion
    simulator_reward: float
    done_code: str
    offroad: bool
    terminated: bool
    truncated: bool
    timestamp_s: float
    world_pose: WorldPose


@dataclass(frozen=True)
class _PedestrianTruthSample:
    timestamp_s: float
    position: WorldPoint
    range_m: float
    bearing_rad: float


@dataclass(frozen=True)
class ObjectGroundContactPixel:
    """Validation-only silhouette bottom-center from a rendered object."""

    object_kind: str
    pixel: ImagePoint
    silhouette_pixel_count: int


@dataclass(frozen=True)
class ObjectSilhouetteAnnotation:
    """Offline-only bounding box derived from an object-specific render mask."""

    object_kind: str
    bounding_box: BoundingBox
    visible_pixel_count: int


class GymDuckietownAgentEnvironment:
    """Agent-only view of one shared Gym-Duckietown simulator session."""

    def __init__(self, session: _GymDuckietownSession) -> None:
        self._session = session

    def reset(self, seed: int | None = None) -> SensorObservation:
        return self._session.reset(seed)

    def step(self, action: PolicyAction) -> Transition:
        return self._session.step(action)

    def close(self) -> None:
        self._session.close()


class GymDuckietownPrivilegedStateSource:
    """Evaluation-only view of the shared simulator session."""

    def __init__(self, session: _GymDuckietownSession) -> None:
        self._session = session

    def read(self) -> PrivilegedSimulatorState:
        return self._session.read_privileged_state()


class GymDuckietownDiagnosticsSource:
    """Experiment-only access to actuator and simulator step diagnostics."""

    def __init__(self, session: _GymDuckietownSession) -> None:
        self._session = session

    def read(self) -> SimulatorStepDiagnostics:
        diagnostics = self._session.last_step_diagnostics
        if diagnostics is None:
            raise RuntimeError("step diagnostics are unavailable before env.step")
        return diagnostics


class GymDuckietownCameraCalibrationSource:
    """Camera parameters available to runtime geometry without object truth."""

    def __init__(self, session: _GymDuckietownSession) -> None:
        self._session = session

    def read(self) -> CameraCalibration:
        return self._session.read_camera_calibration()


class GymDuckietownProjectionValidationSource:
    """Validation-only rendered object pixels; never expose this to a policy."""

    def __init__(self, session: _GymDuckietownSession) -> None:
        self._session = session

    def sample_visible_objects(
        self,
        object_kinds: tuple[str, ...],
    ) -> tuple[ObjectGroundContactPixel, ...]:
        return self._session.sample_visible_object_pixels(object_kinds)

    def sample_object_silhouettes(
        self,
        object_kinds: tuple[str, ...],
    ) -> tuple[ObjectSilhouetteAnnotation, ...]:
        """Return object masks reduced to boxes for offline annotation only."""

        return self._session.sample_object_silhouettes(object_kinds)

    def render_without_objects(self, object_kinds: tuple[str, ...]) -> np.ndarray:
        """Render an offline class-negative frame for detector calibration.

        This method deliberately lives on the privileged validation source. It
        is not exposed through ``GymDuckietownAgentEnvironment`` and therefore
        cannot be called by the runtime detector/EKF chain.
        """

        return self._session.render_without_objects(object_kinds)

    def remove_scenario_pedestrian(self) -> None:
        """F9d Task 4's B3 controlled-removal intervention: permanently mark
        the scenario pedestrian not-visible.

        Unlike ``render_without_objects`` (a temporary toggle restored before
        it returns, used for one offline render), this persists -- the
        pedestrian stays gone for the rest of the episode, in both the
        rendered frame (``render_obs`` skips non-visible objects, the same
        mechanism ``render_without_objects`` already relies on) and every
        privileged-truth query that already filters on the object's
        ``visible`` attribute (``_first_visible_object``/``_visible_objects``,
        used by both ``read_privileged_state`` and the per-step pedestrian
        truth history). A caller decides WHEN to call this (from a schedule
        computed before the episode runs, exactly like B2's
        ``DetectorDropout``); this method only performs the removal itself.

        Deliberately lives on the privileged validation source, exactly like
        ``render_without_objects``: not exposed through
        ``GymDuckietownAgentEnvironment``, so the runtime detector/EKF chain
        can never call it.
        """

        self._session.remove_scenario_pedestrian()


@dataclass(frozen=True)
class GymDuckietownIntegration:
    agent: GymDuckietownAgentEnvironment
    privileged: GymDuckietownPrivilegedStateSource
    diagnostics: GymDuckietownDiagnosticsSource
    camera_calibration: GymDuckietownCameraCalibrationSource
    projection_validation: GymDuckietownProjectionValidationSource

    def reconfigure_native_episode(self, config: GymDuckietownConfig) -> None:
        """Reuse one native-map simulator while changing seed/start pose."""

        self.agent._session.reconfigure_native_episode(config)

    def reconfigure_scenario_episode(self, config: GymDuckietownConfig) -> None:
        """Reuse one fixed scenario map while changing episode-local settings."""

        self.agent._session.reconfigure_scenario_episode(config)

    def close(self) -> None:
        self.agent.close()


def create_gym_duckietown(
    config: GymDuckietownConfig | None = None,
) -> GymDuckietownIntegration:
    session = _GymDuckietownSession(config or GymDuckietownConfig())
    return GymDuckietownIntegration(
        agent=GymDuckietownAgentEnvironment(session),
        privileged=GymDuckietownPrivilegedStateSource(session),
        diagnostics=GymDuckietownDiagnosticsSource(session),
        camera_calibration=GymDuckietownCameraCalibrationSource(session),
        projection_validation=GymDuckietownProjectionValidationSource(session),
    )


class _GymDuckietownSession:
    def __init__(self, config: GymDuckietownConfig) -> None:
        self.config = config
        if config.headless:
            os.environ.setdefault("DUCKIETOWN_HEADLESS", "1")

        from gym_duckietown.envs import DuckietownEnv
        from gym_duckietown.simulator import CAMERA_FORWARD_DIST, Simulator

        self._simulator_type = Simulator
        self._camera_forward_distance_m = float(CAMERA_FORWARD_DIST)
        environment_type = _environment_type(DuckietownEnv, config.scenario)
        map_name = (
            str(config.scenario.map_path)
            if config.scenario is not None
            else config.map_name
        )
        seed = config.scenario.seed if config.scenario is not None else config.seed
        self._simulator = environment_type(
            map_name=map_name,
            max_steps=config.maximum_steps,
            domain_rand=config.domain_randomization,
            dynamics_rand=config.dynamics_randomization,
            frame_rate=config.frame_rate_hz,
            frame_skip=config.frame_skip,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            seed=seed,
        )
        start_tile = config.start_tile
        start_pose = config.start_pose
        if config.scenario is not None:
            start_tile = config.scenario.ego_start_tile
            start_pose = (
                config.scenario.ego_start_pose_m,
                config.scenario.ego_heading_rad,
            )
        if start_tile is not None and start_pose is not None:
            tile = self._simulator._get_tile(*start_tile)
            if tile is None or not bool(tile.get("drivable", False)):
                raise ValueError("configured start_tile is not drivable")
            self._simulator.start_tile = tile
            self._simulator.start_pose = start_pose

        self._scenario_stop_sign: Any | None = None
        self._scenario_pedestrian: Any | None = None
        self._pedestrian_initial_position: np.ndarray | None = None
        if config.scenario is not None:
            self._initialize_scenario_objects(config.scenario)

        self._action_adapter = self._make_action_adapter()
        self._last_pose = self._pose_sample()
        self._last_observation: SensorObservation | None = None
        self.last_step_diagnostics: SimulatorStepDiagnostics | None = None
        self._previous_pedestrian_sample: _PedestrianTruthSample | None = None
        self._pedestrian_world_velocity_estimate: WorldVelocity | None = None
        self._pedestrian_relative_rates: tuple[float, float] | None = None
        self.reset(seed)

    def reconfigure_native_episode(self, config: GymDuckietownConfig) -> None:
        """Update only episode-local native-map settings without rebuilding GL.

        Gym-Duckietown's Mesa/Pyglet close path retains roughly one rendered
        context allocation per reconstructed simulator. Long RL runs therefore
        reuse the simulator for a fixed native map and let ``reset(seed)``
        regenerate domain randomization after installing the new start pose.
        """

        if self.config.scenario is not None or config.scenario is not None:
            raise ValueError("native episode reuse does not support scenario maps")
        stable_fields = (
            "map_name",
            "domain_randomization",
            "dynamics_randomization",
            "frame_rate_hz",
            "frame_skip",
            "maximum_steps",
            "camera_width",
            "camera_height",
            "headless",
        )
        changed = tuple(
            name
            for name in stable_fields
            if getattr(self.config, name) != getattr(config, name)
        )
        if changed:
            raise ValueError(f"native episode reuse changed simulator fields: {changed}")
        if config.start_tile is None or config.start_pose is None:
            raise ValueError("native episode reuse requires an explicit start pose")
        tile = self._simulator._get_tile(*config.start_tile)
        if tile is None or not bool(tile.get("drivable", False)):
            raise ValueError("configured start_tile is not drivable")
        self.config = config
        self._simulator.start_tile = tile
        self._simulator.start_pose = config.start_pose

    def reconfigure_scenario_episode(self, config: GymDuckietownConfig) -> None:
        """Update one stage's scenario without rebuilding the Mesa/GL context.

        Stop-sign activation remains immutable within a curriculum environment.
        The dynamic scenario pedestrian may be toggled between episodes so C2
        can interleave hazard episodes with no-pedestrian driving rehearsal
        without recreating Mesa/Pyglet contexts.  C2 and C3 still use separate
        environment instances and never exchange object state.
        """

        previous = self.config.scenario
        scenario = config.scenario
        if previous is None or scenario is None:
            raise ValueError("scenario episode reuse requires scenario configs")
        stable_fields = (
            "domain_randomization",
            "dynamics_randomization",
            "frame_rate_hz",
            "frame_skip",
            "maximum_steps",
            "camera_width",
            "camera_height",
            "headless",
            "scenario_stop_sign_enabled",
        )
        changed = tuple(
            name
            for name in stable_fields
            if getattr(self.config, name) != getattr(config, name)
        )
        if changed:
            raise ValueError(f"scenario episode reuse changed stable fields: {changed}")
        if previous.map_path.resolve() != scenario.map_path.resolve():
            raise ValueError("scenario episode reuse cannot change maps")
        if (
            previous.stop_sign_kind != scenario.stop_sign_kind
            or previous.pedestrian.object_kind != scenario.pedestrian.object_kind
        ):
            raise ValueError("scenario episode reuse cannot change object kinds")
        tile = self._simulator._get_tile(*scenario.ego_start_tile)
        if tile is None or not bool(tile.get("drivable", False)):
            raise ValueError("configured scenario start tile is not drivable")
        self.config = config
        self._simulator.start_tile = tile
        self._simulator.start_pose = (
            scenario.ego_start_pose_m,
            scenario.ego_heading_rad,
        )

    def reset(self, seed: int | None = None) -> SensorObservation:
        if seed is not None:
            self._simulator.seed(seed)
        frame = self._simulator.reset()
        if isinstance(frame, tuple):
            frame = frame[0]
        if self.config.scenario is not None:
            self._configure_scenario_pedestrian(self.config.scenario)
            if self._scenario_pedestrian is None or self._scenario_stop_sign is None:
                raise RuntimeError("scenario objects are unavailable")
            self._set_scenario_pedestrian_enabled(
                self.config.scenario_pedestrian_enabled
            )
            if self.config.scenario_stop_sign_enabled:
                self._scenario_stop_sign.visible = True
            else:
                self._remove_scenario_object(self._scenario_stop_sign)
            frame = self._simulator.render_obs()
        self._action_adapter = self._make_action_adapter()
        self._last_pose = self._pose_sample()
        self.last_step_diagnostics = None
        observation = self._agent_observation(
            frame=np.asarray(frame),
            motion=EgoMotion(0.0, 0.0),
        )
        self._last_observation = observation
        self._reset_pedestrian_truth_history()
        return observation

    def step(self, action: PolicyAction) -> Transition:
        previous_pose = self._pose_sample()
        pedestrian_was_active = bool(
            self._scenario_pedestrian is not None
            and getattr(self._scenario_pedestrian, "pedestrian_active", False)
        )
        conversion = self._action_adapter.convert(action)
        wheel_action = np.array(
            [conversion.wheel_command.left, conversion.wheel_command.right],
            dtype=float,
        )

        frame, simulator_reward, done, _ = self._simulator_type.step(
            self._simulator,
            wheel_action,
        )
        self._finish_one_shot_pedestrian_if_needed(pedestrian_was_active)
        current_pose = self._pose_sample()
        motion = estimate_actual_motion(previous_pose, current_pose)
        done_info = self._simulator._compute_done_reward()
        done_code = str(done_info.done_code)
        truncated = bool(done and done_code == "max-steps-reached")
        terminated = bool(done and not truncated)
        offroad = bool(done_code == "invalid-pose")

        observation = self._agent_observation(np.asarray(frame), motion)
        self._last_pose = current_pose
        self._last_observation = observation
        self._update_pedestrian_truth_history(current_pose.timestamp_s)
        self.last_step_diagnostics = SimulatorStepDiagnostics(
            requested_action=action,
            action_conversion=conversion,
            actual_motion=motion,
            simulator_reward=float(simulator_reward),
            done_code=done_code,
            offroad=offroad,
            terminated=terminated,
            truncated=truncated,
            timestamp_s=current_pose.timestamp_s,
            world_pose=WorldPose(
                x_m=current_pose.x_m,
                z_m=current_pose.z_m,
                heading_rad=current_pose.heading_rad,
            ),
        )
        return Transition(
            observation=observation,
            reward_terms=_placeholder_reward(),
            terminated=terminated,
            truncated=truncated,
        )

    def _finish_one_shot_pedestrian_if_needed(
        self,
        pedestrian_was_active: bool,
    ) -> None:
        """Remove a completed one-shot crossing from every runtime path.

        Gym-Duckietown's native ``DuckieObj`` reverses direction after an
        eight-second pause. That behavior remains the default for historical
        scenarios. A scenario with ``repeat_crossing = false`` instead models
        a pedestrian that enters, crosses once, and exits: when the native
        object transitions from active to inactive at the far endpoint, it is
        removed from rendering, collision checks, privileged lookup, and the
        subsequent detector input. Reset restores and reconfigures the same
        object, avoiding a new Mesa/Pyglet context.
        """

        scenario = self.config.scenario
        pedestrian = self._scenario_pedestrian
        if (
            scenario is None
            or scenario.pedestrian.repeat_crossing
            or pedestrian is None
            or not pedestrian_was_active
            or bool(getattr(pedestrian, "pedestrian_active", False))
        ):
            return
        self._set_scenario_pedestrian_enabled(False)

    def read_privileged_state(self) -> PrivilegedSimulatorState:
        if self._last_observation is None:
            raise RuntimeError("privileged state is unavailable before reset")

        ego_observation = self._last_observation.ego
        stop_sign_kind = (
            self.config.scenario.stop_sign_kind
            if self.config.scenario is not None
            else "sign_stop"
        )
        pedestrian_kind = (
            self.config.scenario.pedestrian.object_kind
            if self.config.scenario is not None
            else "duckie"
        )
        stop_sign_object = _first_visible_object(
            self._simulator.objects,
            stop_sign_kind,
        )
        pedestrian_object = _first_visible_object(
            self._simulator.objects,
            pedestrian_kind,
        )
        stop_sign_position = _world_point(stop_sign_object)
        pedestrian_position = _world_point(pedestrian_object)
        pedestrian_velocity = self._pedestrian_world_velocity_estimate
        ego_pose = WorldPose(
            x_m=float(self._simulator.cur_pos[0]),
            z_m=float(self._simulator.cur_pos[2]),
            heading_rad=float(self._simulator.cur_angle),
        )

        stop_sign_state = _static_object_state(stop_sign_position, ego_pose)
        pedestrian_state = _pedestrian_state(
            position=pedestrian_position,
            ego_pose=ego_pose,
            relative_rates=self._pedestrian_relative_rates,
        )
        stop_line_position = _stop_line_world_position(self.config.scenario)
        stop_line_distance = _signed_stop_line_distance(
            stop_line_position,
            self.config.scenario,
            ego_pose,
        )
        true_state = POMDPState(
            ego=EgoState(
                lateral_error_m=ego_observation.lateral_error_m,
                heading_error_rad=ego_observation.heading_error_rad,
                linear_velocity_mps=ego_observation.linear_velocity_mps,
                yaw_rate_rad_s=ego_observation.yaw_rate_rad_s,
            ),
            road=RoadState(
                curvature_inv_m=lane_curvature_inv_m(
                    self._simulator,
                    np.asarray(self._simulator.cur_pos, dtype=float),
                    float(self._simulator.cur_angle),
                ),
                stop_line_distance_m=stop_line_distance,
                stop_mode=StopMode.NONE,
            ),
            stop_sign=stop_sign_state,
            pedestrian=pedestrian_state,
        )
        return PrivilegedSimulatorState(
            true_pomdp_state=true_state,
            ego_world_pose=ego_pose,
            stop_sign_world_position=stop_sign_position,
            stop_sign_world_footprint=_world_footprint(stop_sign_object),
            stop_line_world_position=stop_line_position,
            pedestrian_world_position=pedestrian_position,
            pedestrian_world_footprint=_world_footprint(pedestrian_object),
            pedestrian_world_velocity=pedestrian_velocity,
            collision=None,
        )

    def read_camera_calibration(self) -> CameraCalibration:
        return CameraCalibration(
            image_width_px=int(self._simulator.camera_width),
            image_height_px=int(self._simulator.camera_height),
            vertical_fov_deg=float(self._simulator.cam_fov_y),
            camera_height_m=float(self._simulator.cam_height),
            camera_pitch_deg=float(self._simulator.cam_angle[0]),
            camera_forward_offset_m=self._camera_forward_distance_m,
            distortion_enabled=bool(
                self._simulator.distortion and not self._simulator.undistort
            ),
        )

    def sample_visible_object_pixels(
        self,
        object_kinds: tuple[str, ...],
    ) -> tuple[ObjectGroundContactPixel, ...]:
        base_frame = np.asarray(self._simulator.render_obs()).astype(np.int16)
        samples: list[ObjectGroundContactPixel] = []
        for kind in object_kinds:
            target = _first_visible_object(self._simulator.objects, kind)
            if target is None:
                continue
            target.visible = False
            try:
                hidden_frame = np.asarray(self._simulator.render_obs()).astype(np.int16)
            finally:
                target.visible = True
            difference = np.max(np.abs(base_frame - hidden_frame), axis=2)
            rows, columns = np.nonzero(difference > 2)
            if len(rows) == 0:
                continue
            samples.append(
                ObjectGroundContactPixel(
                    object_kind=kind,
                    pixel=ImagePoint(
                        x_px=0.5 * (float(np.min(columns)) + float(np.max(columns))),
                        y_px=float(np.max(rows)),
                    ),
                    silhouette_pixel_count=int(len(rows)),
                )
            )
        return tuple(samples)

    def sample_object_silhouettes(
        self,
        object_kinds: tuple[str, ...],
    ) -> tuple[ObjectSilhouetteAnnotation, ...]:
        """Render object-specific RGB-difference masks for offline annotation.

        Gym-Duckietown 6.2.0's segmentation mesh path fails while hashing its
        list-valued segmentation color. The already validated F5/F5b strategy
        therefore remains the reliable source: hide one target at a time and
        difference two otherwise identical RGB renders. This retains rendered
        visibility and occlusion and never enters the agent path.
        """

        visible_frame = np.asarray(self._simulator.render_obs()).astype(np.int16)
        samples: list[ObjectSilhouetteAnnotation] = []
        for kind in object_kinds:
            target = _first_visible_object(self._simulator.objects, kind)
            if target is None:
                continue
            was_visible = bool(target.visible)
            target.visible = False
            try:
                hidden = np.asarray(self._simulator.render_obs()).astype(np.int16)
            finally:
                target.visible = was_visible
            mask = np.max(np.abs(visible_frame - hidden), axis=2) > 2
            rows, columns = np.nonzero(mask)
            if len(rows) == 0:
                continue
            samples.append(
                ObjectSilhouetteAnnotation(
                    object_kind=kind,
                    bounding_box=BoundingBox(
                        x_min_px=float(np.min(columns)),
                        y_min_px=float(np.min(rows)),
                        x_max_px=float(np.max(columns) + 1),
                        y_max_px=float(np.max(rows) + 1),
                    ),
                    visible_pixel_count=int(len(rows)),
                )
            )
        return tuple(samples)

    def render_without_objects(self, object_kinds: tuple[str, ...]) -> np.ndarray:
        targets = [
            item
            for item in self._simulator.objects
            if str(getattr(item, "kind", "")) in object_kinds
            and bool(getattr(item, "visible", True))
        ]
        visibility = [bool(target.visible) for target in targets]
        for target in targets:
            target.visible = False
        try:
            return np.asarray(self._simulator.render_obs()).copy()
        finally:
            for target, was_visible in zip(targets, visibility):
                target.visible = was_visible

    def remove_scenario_pedestrian(self) -> None:
        pedestrian = self._scenario_pedestrian
        if pedestrian is None:
            raise RuntimeError(
                "no scenario pedestrian to remove -- session was not built "
                "from a scenario"
            )
        pedestrian.visible = False

    def _initialize_scenario_objects(
        self,
        scenario: MinimalPOMDPScenario,
    ) -> None:
        stop_signs = _visible_objects(self._simulator.objects, scenario.stop_sign_kind)
        pedestrians = _visible_objects(
            self._simulator.objects,
            scenario.pedestrian.object_kind,
        )
        if len(stop_signs) != 1:
            raise ValueError(
                f"minimal scenario requires exactly one stop sign; found {len(stop_signs)}"
            )
        if len(pedestrians) != 1:
            raise ValueError(
                f"minimal scenario requires exactly one pedestrian; found {len(pedestrians)}"
            )
        self._scenario_stop_sign = stop_signs[0]
        self._scenario_pedestrian = pedestrians[0]
        self._pedestrian_initial_position = np.asarray(
            pedestrians[0].pos,
            dtype=float,
        ).copy()

    def _remove_scenario_object(self, target: Any) -> None:
        """Physically remove an inactive curriculum object for this session.

        ``visible=False`` is not enough: Gym-Duckietown collision checks can
        still consider invisible objects. Each curriculum stage owns a separate
        simulator session, so removal remains deterministic across that stage's
        resets and cannot leak into another stage.
        """

        target.visible = False
        if target in self._simulator.objects:
            self._simulator.objects.remove(target)
        corners = np.asarray(self._simulator.collidable_corners)
        target_corners = np.asarray(target.obj_corners).T
        if corners.ndim == 3 and corners.shape[1:] == target_corners.shape:
            keep = np.asarray(
                [not np.allclose(item, target_corners) for item in corners],
                dtype=bool,
            )
            if not np.all(keep):
                self._simulator.collidable_corners = corners[keep]
                self._simulator.collidable_norms = np.asarray(
                    self._simulator.collidable_norms
                )[keep]
                self._simulator.collidable_centers = np.asarray(
                    self._simulator.collidable_centers
                )[keep]
                self._simulator.collidable_safety_radii = np.asarray(
                    self._simulator.collidable_safety_radii
                )[keep]

    def _set_scenario_pedestrian_enabled(self, enabled: bool) -> None:
        """Toggle the dynamic Duckie without rebuilding the simulator context.

        Gym-Duckietown checks dynamic-object collision and proximity by walking
        ``simulator.objects``; ``visible=False`` alone would therefore leave an
        invisible collision target.  The scenario Duckie is declared
        ``static: false``, so removing it from that list disables rendering,
        stepping, proximity, collision, and privileged lookup together.  A
        later hazard episode reuses the same Python object after
        ``_configure_scenario_pedestrian`` has reset its pose and motion state.
        Static pedestrians are rejected because their batched collision arrays
        cannot be toggled by this dynamic-object operation.
        """

        pedestrian = self._scenario_pedestrian
        if pedestrian is None:
            raise RuntimeError("scenario pedestrian was not initialized")
        if bool(getattr(pedestrian, "static", False)):
            raise RuntimeError(
                "episode-level pedestrian toggling requires a dynamic scenario object"
            )

        if enabled:
            if pedestrian not in self._simulator.objects:
                self._simulator.objects.append(pedestrian)
            pedestrian.visible = True
            return

        pedestrian.visible = False
        if pedestrian in self._simulator.objects:
            self._simulator.objects.remove(pedestrian)

    def _configure_scenario_pedestrian(
        self,
        scenario: MinimalPOMDPScenario,
    ) -> None:
        pedestrian = self._scenario_pedestrian
        initial_position = self._pedestrian_initial_position
        if pedestrian is None or initial_position is None:
            raise RuntimeError("scenario pedestrian was not initialized")

        from gym_duckietown.collision import generate_corners, generate_norm

        configured_start = initial_position.copy()
        configured_heading: np.ndarray | None = None
        explicit_path = scenario.pedestrian.path_for_mode()
        if explicit_path is not None:
            (start_x, start_z), (end_x, end_z) = explicit_path
            configured_start[0] = start_x
            configured_start[2] = start_z
            delta_x = end_x - start_x
            delta_z = end_z - start_z
            distance = hypot(delta_x, delta_z)
            configured_heading = np.array(
                [delta_x / distance, 0.0, delta_z / distance], dtype=float
            )
        elif scenario.pedestrian.mode is PedestrianMode.CROSS_RIGHT_TO_LEFT:
            # Legacy pomdp_v1 semantics: the map origin is the left endpoint.
            configured_start[2] += scenario.pedestrian.crossing_distance_m
        pedestrian.pos = configured_start.copy()
        pedestrian.center = configured_start.copy()
        pedestrian.start = configured_start.copy()
        pedestrian.walk_distance = scenario.pedestrian.crossing_distance_m
        pedestrian.time = 0.0
        pedestrian.wiggle = float(np.asarray(pedestrian.wiggle).reshape(-1)[0])
        pedestrian.pedestrian_wait_time = float("inf")
        mode = scenario.pedestrian.mode
        if mode is PedestrianMode.STATIONARY:
            pedestrian.pedestrian_active = False
            pedestrian.vel = 0.0
            pedestrian.pedestrian_wait_time = float("inf")
        else:
            start_delay_s = scenario.pedestrian.start_delay_for_mode()
            pedestrian.pedestrian_active = start_delay_s <= 0.0
            pedestrian.pedestrian_wait_time = float(
                start_delay_s
            )
            pedestrian.vel = (
                scenario.pedestrian.speed_mps / self.config.frame_rate_hz
            )
            if configured_heading is None:
                pedestrian.angle = (
                    -0.5 * pi
                    if mode is PedestrianMode.CROSS_LEFT_TO_RIGHT
                    else 0.5 * pi
                )
                pedestrian.heading = np.array(
                    [
                        0.0,
                        0.0,
                        1.0
                        if mode is PedestrianMode.CROSS_LEFT_TO_RIGHT
                        else -1.0,
                    ],
                    dtype=float,
                )
            else:
                pedestrian.heading = configured_heading
                pedestrian.angle = atan2(
                    -float(configured_heading[2]),
                    float(configured_heading[0]),
                )
            pedestrian.y_rot = np.rad2deg(pedestrian.angle)
        pedestrian.obj_corners = generate_corners(
            pedestrian.pos,
            pedestrian.min_coords,
            pedestrian.max_coords,
            pedestrian.angle,
            pedestrian.scale,
        )
        pedestrian.obj_norm = generate_norm(pedestrian.obj_corners)

    def _reset_pedestrian_truth_history(self) -> None:
        sample = self._current_pedestrian_sample(float(self._simulator.timestamp))
        self._previous_pedestrian_sample = sample
        if sample is None:
            self._pedestrian_world_velocity_estimate = None
            self._pedestrian_relative_rates = None
            return
        self._pedestrian_world_velocity_estimate = WorldVelocity(0.0, 0.0)
        self._pedestrian_relative_rates = (0.0, 0.0)

    def _update_pedestrian_truth_history(self, timestamp_s: float) -> None:
        current = self._current_pedestrian_sample(timestamp_s)
        previous = self._previous_pedestrian_sample
        self._previous_pedestrian_sample = current
        if current is None:
            self._pedestrian_world_velocity_estimate = None
            self._pedestrian_relative_rates = None
            return
        if previous is None:
            self._pedestrian_world_velocity_estimate = WorldVelocity(0.0, 0.0)
            self._pedestrian_relative_rates = (0.0, 0.0)
            return
        dt_s = current.timestamp_s - previous.timestamp_s
        if dt_s <= 0.0:
            raise RuntimeError("pedestrian truth timestamps must increase")
        self._pedestrian_world_velocity_estimate = WorldVelocity(
            x_velocity_mps=(current.position.x_m - previous.position.x_m) / dt_s,
            z_velocity_mps=(current.position.z_m - previous.position.z_m) / dt_s,
        )
        self._pedestrian_relative_rates = (
            (current.range_m - previous.range_m) / dt_s,
            _wrap_angle(current.bearing_rad - previous.bearing_rad) / dt_s,
        )

    def _current_pedestrian_sample(
        self,
        timestamp_s: float,
    ) -> _PedestrianTruthSample | None:
        kind = (
            self.config.scenario.pedestrian.object_kind
            if self.config.scenario is not None
            else "duckie"
        )
        pedestrian = _first_visible_object(self._simulator.objects, kind)
        position = _world_point(pedestrian)
        if position is None:
            return None
        ego_pose = WorldPose(
            x_m=float(self._simulator.cur_pos[0]),
            z_m=float(self._simulator.cur_pos[2]),
            heading_rad=float(self._simulator.cur_angle),
        )
        relative = world_to_ego(position, ego_pose)
        return _PedestrianTruthSample(
            timestamp_s=timestamp_s,
            position=position,
            range_m=hypot(relative.x_left_m, relative.y_forward_m),
            bearing_rad=atan2(relative.x_left_m, relative.y_forward_m),
        )

    def close(self) -> None:
        self._simulator.close()

    def _make_action_adapter(self) -> DifferentialDriveActionAdapter:
        return DifferentialDriveActionAdapter(
            DifferentialDriveCalibration(
                wheel_radius_m=float(self._simulator.radius),
                wheel_separation_m=float(self._simulator.wheel_dist),
                motor_constant_rad_s_per_unit=float(self._simulator.k),
                gain=float(self._simulator.gain),
                trim=float(self._simulator.trim),
                command_limit=float(self._simulator.limit),
            )
        )

    def _pose_sample(self) -> SimulatorPoseSample:
        return SimulatorPoseSample(
            x_m=float(self._simulator.cur_pos[0]),
            z_m=float(self._simulator.cur_pos[2]),
            heading_rad=float(self._simulator.cur_angle),
            timestamp_s=float(self._simulator.timestamp),
        )

    def _agent_observation(
        self,
        frame: np.ndarray,
        motion: EgoMotion,
    ) -> SensorObservation:
        lane_position = self._simulator.get_lane_pos2(
            self._simulator.cur_pos,
            self._simulator.cur_angle,
        )
        ego = EgoObservation(
            lateral_error_m=float(lane_position.dist),
            heading_error_rad=float(lane_position.angle_rad),
            linear_velocity_mps=motion.linear_velocity_mps,
            yaw_rate_rad_s=motion.yaw_rate_rad_s,
        )
        if not all(
            isfinite(value)
            for value in (
                ego.lateral_error_m,
                ego.heading_error_rad,
                ego.linear_velocity_mps,
                ego.yaw_rate_rad_s,
            )
        ):
            raise RuntimeError("simulator produced non-finite ego observation")
        return SensorObservation(front_rgb=frame, ego=ego, road=None)


def _placeholder_reward() -> RewardTerms:
    return RewardTerms(
        progress=0.0,
        lane=0.0,
        stop=0.0,
        pedestrian=0.0,
        comfort=0.0,
        collision=0.0,
    )


def _first_visible_object(objects: list[Any], kind: str) -> Any | None:
    return next(iter(_visible_objects(objects, kind)), None)


def _visible_objects(objects: list[Any], kind: str) -> list[Any]:
    return [
        item
        for item in objects
        if str(getattr(item, "kind", "")) == kind
        and bool(getattr(item, "visible", True))
    ]


def _world_point(item: Any | None) -> WorldPoint | None:
    if item is None:
        return None
    position = np.asarray(item.pos, dtype=float)
    return WorldPoint(x_m=float(position[0]), z_m=float(position[2]))


def _world_footprint(item: Any | None) -> WorldFootprint | None:
    if item is None or not hasattr(item, "obj_corners"):
        return None
    corners = np.asarray(item.obj_corners, dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 2 or len(corners) < 3:
        raise RuntimeError("simulator object footprint has an invalid shape")
    return WorldFootprint(
        vertices=tuple(
            WorldPoint(x_m=float(corner[0]), z_m=float(corner[1]))
            for corner in corners
        )
    )


def _stop_line_world_position(
    scenario: MinimalPOMDPScenario | None,
) -> WorldPoint | None:
    if scenario is None:
        return None
    return WorldPoint(
        x_m=scenario.stop_line.world_x_m,
        z_m=scenario.stop_line.world_z_m,
    )


def _signed_stop_line_distance(
    position: WorldPoint | None,
    scenario: MinimalPOMDPScenario | None,
    ego_pose: WorldPose,
) -> float | None:
    if position is None or scenario is None:
        return None
    delta_x = position.x_m - ego_pose.x_m
    delta_z = position.z_m - ego_pose.z_m
    heading = scenario.stop_line.route_heading_rad
    return delta_x * cos(heading) - delta_z * sin(heading)


def _environment_type(base_type: type, scenario: MinimalPOMDPScenario | None) -> type:
    if scenario is None:
        return base_type

    return external_map_environment_type(base_type)


def external_map_environment_type(base_type: type) -> type:
    """Return a Gym-Duckietown environment class accepting an absolute map path."""

    class ExternalMapDuckietownEnv(base_type):
        def _load_map(self, map_name: str) -> None:
            map_path = Path(map_name)
            if not map_path.is_file():
                super()._load_map(map_name)
                return
            import yaml

            self.map_name = map_path.stem
            self.map_file_path = str(map_path.resolve())
            with map_path.open("r", encoding="utf-8") as stream:
                self.map_data = yaml.load(stream, Loader=yaml.Loader)
            self._interpret_map(self.map_data)

    return ExternalMapDuckietownEnv


def _static_object_state(
    position: WorldPoint | None,
    ego_pose: WorldPose,
) -> StopSignState:
    if position is None:
        return StopSignState(False, None, None)
    relative = world_to_ego(position, ego_pose)
    return StopSignState(
        exists=True,
        range_m=hypot(relative.x_left_m, relative.y_forward_m),
        bearing_rad=atan2(relative.x_left_m, relative.y_forward_m),
    )


def _pedestrian_state(
    *,
    position: WorldPoint | None,
    ego_pose: WorldPose,
    relative_rates: tuple[float, float] | None,
) -> PedestrianState:
    if position is None or relative_rates is None:
        return PedestrianState(False, None, None, None, None)

    relative = world_to_ego(position, ego_pose)
    range_m = hypot(relative.x_left_m, relative.y_forward_m)
    radial_velocity, bearing_rate = relative_rates

    return PedestrianState(
        exists=True,
        range_m=range_m,
        bearing_rad=atan2(relative.x_left_m, relative.y_forward_m),
        radial_velocity_mps=radial_velocity,
        bearing_rate_rad_s=bearing_rate,
    )


def _wrap_angle(angle_rad: float) -> float:
    return atan2(sin(angle_rad), cos(angle_rad))
