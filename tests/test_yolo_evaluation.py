from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.evaluation.yolo_detection import (
    DetectionOpportunity,
    evaluate_opportunity,
    intersection_over_union,
    summarize_detection,
)
from duckie_pomdp.evaluation.yolo_measurement import (
    MeasurementResidual,
    gaussian_assessment,
    summarize_measurements,
)
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
)
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector


def _opportunity(box: BoundingBox | None) -> DetectionOpportunity:
    return DetectionOpportunity(
        image_id="image",
        episode_id="episode",
        seed=1,
        frame_index=2,
        object_class=ObjectClass.DUCKIE,
        eligible_visible=box is not None,
        ground_truth_box=box,
        ground_truth_range_m=0.7,
        ground_truth_bearing_rad=0.1,
        distance_bin="medium",
        fov_region="center",
    )


def _detection(box: BoundingBox, confidence: float = 0.8) -> Detection:
    return Detection(ObjectClass.DUCKIE, confidence, box)


def test_iou_and_single_gt_matching_are_deterministic() -> None:
    ground_truth = BoundingBox(10.0, 10.0, 30.0, 30.0)
    perfect = _detection(ground_truth, 0.7)
    partial = _detection(BoundingBox(20.0, 10.0, 40.0, 30.0), 0.9)
    assert intersection_over_union(ground_truth, perfect.bounding_box) == 1.0
    assert intersection_over_union(ground_truth, partial.bounding_box) == pytest.approx(
        1.0 / 3.0
    )
    result = evaluate_opportunity(
        _opportunity(ground_truth), (partial, perfect), iou_threshold=0.5
    )
    assert result.matched_detection is perfect
    assert result.match_iou == 1.0
    assert result.unmatched_detections == (partial,)


def test_bad_localization_counts_as_false_negative_and_false_positive() -> None:
    ground_truth = BoundingBox(10.0, 10.0, 30.0, 30.0)
    prediction = _detection(BoundingBox(25.0, 10.0, 45.0, 30.0))
    result = evaluate_opportunity(
        _opportunity(ground_truth), (prediction,), iou_threshold=0.5
    )
    assert result.false_negative
    assert result.unmatched_detections == (prediction,)
    metrics = summarize_detection((result,))
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_class_negative_frame_is_false_alarm_denominator() -> None:
    prediction = _detection(BoundingBox(1.0, 2.0, 5.0, 8.0))
    result = evaluate_opportunity(
        _opportunity(None), (prediction,), iou_threshold=0.5
    )
    metrics = summarize_detection((result,))
    assert metrics["class_negative_frames"] == 1
    assert metrics["false_positives_on_negative_frames"] == 1
    assert metrics["false_alarm_event_probability"] == 1.0


def test_yolo_measurement_projector_has_no_privileged_input_and_clips_boundary() -> None:
    assert list(signature(YoloMeasurementProjector.project).parameters) == [
        "self",
        "detection",
    ]
    projector = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )
    measurement_projector = YoloMeasurementProjector(
        projector,
        MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
    )
    detection = _detection(BoundingBox(300.0, 300.0, 340.0, 480.0))
    measurement = measurement_projector.project(detection)
    assert measurement.detector_bottom_center.y_px == 480.0
    assert measurement.projection_pixel.y_px == 479.0
    assert measurement.pixel_clipped_to_image
    assert np.isfinite(measurement.raw_polar.range_m)


def test_measurement_summary_keeps_raw_and_calibrated_residuals_separate() -> None:
    rows = tuple(
        MeasurementResidual(
            image_id=str(index),
            object_class="duckie",
            distance_bin="near",
            fov_region="center",
            confidence=0.5 + index * 0.01,
            raw_range_error_m=0.10,
            calibrated_range_error_m=0.01 * (-1) ** index,
            bearing_error_rad=0.005 * (-1) ** index,
        )
        for index in range(20)
    )
    summary = summarize_measurements(rows)
    assert summary["global"]["raw_range"]["bias"] == pytest.approx(0.10)
    assert summary["global"]["f5b_calibrated_range"]["bias"] == pytest.approx(0.0)
    assert gaussian_assessment(summary["global"]["bearing"]) == (
        "reasonable_gaussian_approximation"
    )
