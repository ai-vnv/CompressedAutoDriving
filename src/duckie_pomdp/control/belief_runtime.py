"""F10 adapter around the frozen RGB -> YOLO -> robust F9c belief path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.belief import (
    CandidateMeasurement,
    PredictedObservabilityModel,
    RobustPedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState, RoadBelief
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.evaluation.f9c_calibration import (
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.confidence_filter import (
    ClassConfidenceFilter,
    DuckieImageDomainFilter,
)
from duckie_pomdp.perception.f9_pipeline import (
    F9ImageObservation,
    YoloPedestrianMeasurementPipeline,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.ports.detector import ObjectDetector

from .f10_protocol import F10Protocol


@dataclass(frozen=True)
class F10PerceptionDiagnostics:
    duckie_detection_count: int
    duplicate_selection: bool
    stop_sign_detection_count: int
    projection_error: str | None
    frame_mode: str
    measurement_accepted: bool
    nis: float | None
    stop_sign_detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class F10BeliefStep:
    belief: BeliefState
    diagnostics: F10PerceptionDiagnostics


class F10BeliefRuntime:
    """Consumes only agent sensors, physical action, and measured ego motion."""

    def __init__(
        self,
        pipeline: YoloPedestrianMeasurementPipeline,
        updater: RobustPedestrianBeliefUpdater,
        *,
        existence_prior: float,
    ) -> None:
        self._pipeline = pipeline
        self._updater = updater
        self._existence_prior = existence_prior
        self._belief: BeliefState | None = None

    def reset(self, observation: SensorObservation, *, dt_s: float) -> F10BeliefStep:
        self._belief = initial_belief(
            observation.ego,
            observation.road,
            existence_prior=self._existence_prior,
        )
        return self.update(
            observation,
            previous_action=PolicyAction(0.0, 0.0),
            dt_s=dt_s,
        )

    def update(
        self,
        observation: SensorObservation,
        *,
        previous_action: PolicyAction,
        dt_s: float,
    ) -> F10BeliefStep:
        if self._belief is None:
            raise RuntimeError("belief runtime must be reset before update")
        result = self._pipeline.observe(observation.front_rgb)
        posterior, record = self._updater.update(
            previous_belief=self._belief,
            previous_action=previous_action,
            ego_motion=observation.ego.motion,
            candidates=_candidates(result),
            dt_s=dt_s,
        )
        road = posterior.road
        if observation.road is not None:
            road = RoadBelief(
                curvature_inv_m=observation.road.curvature_inv_m,
                stop_line_distance_m=observation.road.stop_line_distance_m,
                stop_mode=posterior.road.stop_mode,
            )
        self._belief = BeliefState(
            ego=observation.ego,
            road=road,
            stop_sign=posterior.stop_sign,
            pedestrian=posterior.pedestrian,
        )
        return F10BeliefStep(
            belief=self._belief,
            diagnostics=F10PerceptionDiagnostics(
                duckie_detection_count=result.duckie_detection_count,
                duplicate_selection=result.duplicate_selection,
                stop_sign_detection_count=len(result.stop_sign_detections),
                projection_error=result.projection_error,
                frame_mode=record.frame_mode,
                measurement_accepted=record.kinematic_measurement_accepted,
                nis=record.nis,
                stop_sign_detections=result.stop_sign_detections,
            ),
        )


class F10BeliefRuntimeFactory:
    """Loads the frozen detector once, then builds episode-local EKF state."""

    def __init__(
        self,
        protocol: F10Protocol,
        *,
        detector: ObjectDetector | None = None,
    ) -> None:
        self._protocol = protocol
        with protocol.belief_config_path.open("rb") as stream:
            self._raw: dict[str, Any] = tomllib.load(stream)
        detector_config = self._raw["detector"]
        base_detector = detector or YoloObjectDetector(
            protocol.detector_checkpoint_path,
            confidence_threshold=float(detector_config["confidence_threshold"]),
            iou_threshold=float(detector_config["nms_iou_threshold"]),
            image_size=int(detector_config["image_size"]),
            device=detector_config["device"],
            max_detections=int(detector_config["max_detections"]),
        )
        runtime_detection = protocol.raw.get("runtime_detection", {})
        duckie_minimum = runtime_detection.get("duckie_minimum_confidence")
        confidence_filtered = (
            base_detector
            if duckie_minimum is None
            else ClassConfidenceFilter(
                base_detector,
                {ObjectClass.DUCKIE: float(duckie_minimum)},
            )
        )
        maximum_bottom = runtime_detection.get("duckie_maximum_bottom_y_px")
        self._detector = (
            confidence_filtered
            if maximum_bottom is None
            else DuckieImageDomainFilter(
                confidence_filtered,
                maximum_bottom_y_px=float(maximum_bottom),
            )
        )

    def create(self, integration: Any) -> F10BeliefRuntime:
        config_path = self._protocol.belief_config_path
        camera_projector = CalibratedGroundProjector(
            integration.camera_calibration.read()
        )
        pipeline = YoloPedestrianMeasurementPipeline(
            self._detector,
            YoloMeasurementProjector(
                camera_projector,
                MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
            ),
        )
        existence_config = replace(
            load_existence_filter_config(config_path),
            miss_likelihood_floor=load_miss_likelihood_floor(config_path),
        )
        updater = RobustPedestrianBeliefUpdater(
            ekf_config=load_pedestrian_ekf_config(config_path),
            measurement_noise=load_polar_measurement_noise(config_path),
            existence_config=existence_config,
            bias_frozen=load_frozen_bias_correction(
                config_path,
                section="baseline_measurement_model",
            ),
            bias_fitted=load_frozen_bias_correction(
                config_path,
                section="measurement_model",
            ),
            observability_model=PredictedObservabilityModel(
                camera_projector,
                image_width_px=int(self._raw["simulator"]["image_width_px"]),
            ),
            config=load_robust_observation_config(config_path),
        )
        return F10BeliefRuntime(
            pipeline,
            updater,
            existence_prior=existence_config.prior_probability,
        )


def _bbox_key(box: BoundingBox) -> tuple[int, int, int, int]:
    return (
        int(round(box.x_min_px)),
        int(round(box.y_min_px)),
        int(round(box.x_max_px)),
        int(round(box.y_max_px)),
    )


def _candidates(result: F9ImageObservation) -> tuple[CandidateMeasurement, ...]:
    return tuple(
        CandidateMeasurement(
            measurement=candidate.measurement,
            confidence=candidate.detection.confidence,
            bbox_key=_bbox_key(candidate.detection.bounding_box),
        )
        for candidate in result.duckie_candidates
        if candidate.projection_error is None
    )
