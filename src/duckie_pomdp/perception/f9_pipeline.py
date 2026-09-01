"""Image-only F9 pedestrian measurement path for the frozen YOLO detector."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Sequence

from numpy import uint8
from numpy.typing import NDArray

from duckie_pomdp.domain.detection import Detection, ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.ports.detector import ObjectDetector


@dataclass(frozen=True)
class DuckieSelection:
    detection_count: int
    selected: Detection | None

    @property
    def multiplicity(self) -> bool:
        return self.detection_count > 1


@dataclass(frozen=True)
class AdditiveMeasurementBias:
    range_bias_m: float
    bearing_bias_rad: float

    def __post_init__(self) -> None:
        if not isfinite(self.range_bias_m) or not isfinite(self.bearing_bias_rad):
            raise ValueError("measurement biases must be finite")

    @classmethod
    def zero(cls) -> AdditiveMeasurementBias:
        return cls(0.0, 0.0)

    def correct(self, measurement: ObjectMeasurement) -> ObjectMeasurement:
        """Subtract fixed offline biases without accepting any ground truth."""

        if not measurement.detected:
            return measurement
        if measurement.range_m is None or measurement.bearing_rad is None:
            raise RuntimeError("detected measurement has incomplete polar geometry")
        corrected_range = max(0.0, measurement.range_m - self.range_bias_m)
        corrected_bearing = wrap_angle(
            measurement.bearing_rad - self.bearing_bias_rad
        )
        return ObjectMeasurement(
            object_class=measurement.object_class,
            detected=True,
            confidence=measurement.confidence,
            x_left_m=corrected_range * sin(corrected_bearing),
            y_forward_m=corrected_range * cos(corrected_bearing),
            range_m=corrected_range,
            bearing_rad=corrected_bearing,
        )


@dataclass(frozen=True)
class CandidateProjection:
    detection: Detection
    measurement: ObjectMeasurement
    projection_error: str | None


@dataclass(frozen=True)
class F9ImageObservation:
    pedestrian: ObjectMeasurement
    duckie_detection_count: int
    selected_duckie: Detection | None
    duplicate_selection: bool
    projection_error: str | None
    stop_sign_detections: tuple[Detection, ...]
    duckie_candidates: tuple[CandidateProjection, ...] = ()


def select_single_duckie(detections: Sequence[Detection]) -> DuckieSelection:
    """Select highest confidence; bbox coordinates break exact score ties."""

    candidates = tuple(
        detection
        for detection in detections
        if detection.object_class is ObjectClass.DUCKIE
    )
    selected = (
        None
        if not candidates
        else min(
            candidates,
            key=lambda detection: (
                -detection.confidence,
                detection.bounding_box.x_min_px,
                detection.bounding_box.y_min_px,
                detection.bounding_box.x_max_px,
                detection.bounding_box.y_max_px,
            ),
        )
    )
    return DuckieSelection(len(candidates), selected)


class YoloPedestrianMeasurementPipeline:
    """Runtime chain: front RGB -> YOLO -> one Duckie -> raw metric polar."""

    def __init__(
        self,
        detector: ObjectDetector,
        projector: YoloMeasurementProjector,
    ) -> None:
        self._detector = detector
        self._projector = projector

    def observe(self, front_rgb: NDArray[uint8]) -> F9ImageObservation:
        detections = tuple(self._detector.detect(front_rgb))
        selection = select_single_duckie(detections)
        stop_signs = tuple(
            detection
            for detection in detections
            if detection.object_class is ObjectClass.STOP_SIGN
        )
        duckie_detections = tuple(
            detection
            for detection in detections
            if detection.object_class is ObjectClass.DUCKIE
        )
        candidates = tuple(
            self._project_candidate(detection) for detection in duckie_detections
        )
        if selection.selected is None:
            return F9ImageObservation(
                pedestrian=ObjectMeasurement.missing(ObjectClass.DUCKIE),
                duckie_detection_count=selection.detection_count,
                selected_duckie=None,
                duplicate_selection=selection.multiplicity,
                projection_error=None,
                stop_sign_detections=stop_signs,
                duckie_candidates=candidates,
            )
        try:
            projected = self._projector.project_raw(selection.selected)
        except ValueError as error:
            return F9ImageObservation(
                pedestrian=ObjectMeasurement.missing(ObjectClass.DUCKIE),
                duckie_detection_count=selection.detection_count,
                selected_duckie=selection.selected,
                duplicate_selection=selection.multiplicity,
                projection_error=str(error),
                stop_sign_detections=stop_signs,
                duckie_candidates=candidates,
            )
        polar = projected.raw_polar
        return F9ImageObservation(
            pedestrian=ObjectMeasurement(
                object_class=ObjectClass.DUCKIE,
                detected=True,
                confidence=selection.selected.confidence,
                x_left_m=projected.ground_point.x_left_m,
                y_forward_m=projected.ground_point.y_forward_m,
                range_m=polar.range_m,
                bearing_rad=polar.bearing_rad,
            ),
            duckie_detection_count=selection.detection_count,
            selected_duckie=selection.selected,
            duplicate_selection=selection.multiplicity,
            projection_error=None,
            stop_sign_detections=stop_signs,
            duckie_candidates=candidates,
        )

    def _project_candidate(self, detection: Detection) -> CandidateProjection:
        try:
            projected = self._projector.project_raw(detection)
        except ValueError as error:
            return CandidateProjection(
                detection=detection,
                measurement=ObjectMeasurement.missing(ObjectClass.DUCKIE),
                projection_error=str(error),
            )
        polar = projected.raw_polar
        return CandidateProjection(
            detection=detection,
            measurement=ObjectMeasurement(
                object_class=ObjectClass.DUCKIE,
                detected=True,
                confidence=detection.confidence,
                x_left_m=projected.ground_point.x_left_m,
                y_forward_m=projected.ground_point.y_forward_m,
                range_m=polar.range_m,
                bearing_rad=polar.bearing_rad,
            ),
            projection_error=None,
        )
