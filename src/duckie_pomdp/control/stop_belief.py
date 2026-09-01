"""Agent-side stop-sign belief and obligation state machine for F10-PPO."""

from __future__ import annotations

from dataclasses import dataclass

from duckie_pomdp.domain.belief import StopSignBelief
from duckie_pomdp.domain.detection import Detection, ObjectClass
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import StopMode
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector

from .ppo_observation import neutral_stop_sign
from .ppo_protocol import PPOCurriculumProtocol


@dataclass(frozen=True)
class StopBeliefStep:
    belief: StopSignBelief
    mode: StopMode
    detected: bool
    selected_confidence: float | None
    projection_error: str | None


class RuntimeStopBeliefUpdater:
    """Uses only YOLO detections, camera geometry, ego motion, and route prior."""

    def __init__(
        self,
        protocol: PPOCurriculumProtocol,
        projector: YoloMeasurementProjector,
        *,
        active: bool,
    ) -> None:
        self._protocol = protocol
        self._projector = projector
        self._active = active
        self._config = protocol.raw["stop_belief"]
        self._belief = neutral_stop_sign(protocol)
        self._mode = StopMode.NONE
        self._hold_steps = 0

    def reset(self) -> StopBeliefStep:
        self._belief = neutral_stop_sign(self._protocol)
        self._mode = StopMode.NONE
        self._hold_steps = 0
        return StopBeliefStep(self._belief, self._mode, False, None, None)

    def update(
        self,
        detections: tuple[Detection, ...],
        *,
        stop_line_distance_m: float,
        ego: EgoObservation,
    ) -> StopBeliefStep:
        if not self._active:
            return self.reset()
        candidates = tuple(
            detection
            for detection in detections
            if detection.object_class is ObjectClass.STOP_SIGN
        )
        selected = (
            None
            if not candidates
            else min(
                candidates,
                key=lambda item: (
                    -item.confidence,
                    item.bounding_box.x_min_px,
                    item.bounding_box.y_min_px,
                    item.bounding_box.x_max_px,
                    item.bounding_box.y_max_px,
                ),
            )
        )
        projection_error: str | None = None
        if selected is None:
            probability = (
                self._belief.existence_probability
                * float(self._config["miss_survival_probability"])
            )
            self._belief = StopSignBelief(
                probability,
                self._belief.range_mean_m,
                self._belief.range_std_m,
                self._belief.bearing_mean_rad,
                self._belief.bearing_std_rad,
            )
        else:
            try:
                projected = self._projector.project_raw(selected)
            except ValueError as error:
                projection_error = str(error)
            else:
                probability = 1.0 - (
                    1.0 - self._belief.existence_probability
                ) * (1.0 - float(self._config["detection_update_probability"]))
                polar = projected.raw_polar
                self._belief = StopSignBelief(
                    existence_probability=probability,
                    range_mean_m=polar.range_m,
                    range_std_m=float(self._config["range_std_m"]),
                    bearing_mean_rad=polar.bearing_rad,
                    bearing_std_rad=float(self._config["bearing_std_rad"]),
                )

        if (
            self._mode is StopMode.NONE
            and self._belief.existence_probability
            >= float(self._config["activation_probability"])
            and stop_line_distance_m <= float(self._config["activation_distance_m"])
        ):
            self._mode = StopMode.REQUIRED

        if self._mode is StopMode.REQUIRED:
            in_zone = (
                float(self._config["completion_zone_after_m"])
                <= stop_line_distance_m
                <= float(self._config["completion_zone_before_m"])
            )
            if in_zone and ego.linear_velocity_mps <= float(
                self._config["stop_speed_threshold_mps"]
            ):
                self._hold_steps += 1
                if self._hold_steps >= int(self._config["hold_steps"]):
                    self._mode = StopMode.SATISFIED
            else:
                self._hold_steps = 0

        return StopBeliefStep(
            self._belief,
            self._mode,
            selected is not None,
            None if selected is None else selected.confidence,
            projection_error,
        )

