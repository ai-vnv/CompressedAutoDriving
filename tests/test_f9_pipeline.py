from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    PedestrianEKF,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.evaluation.f9_protocol import load_f9_protocol
from duckie_pomdp.evaluation.f9_belief import (
    current_measurement_metrics,
    nis_metrics,
)
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)
from duckie_pomdp.perception.f9_pipeline import (
    AdditiveMeasurementBias,
    CandidateProjection,
    YoloPedestrianMeasurementPipeline,
    select_single_duckie,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CHECKPOINT_SHA256 = (
    "3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c"
)


class FakeDetector:
    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self.detections = detections
        self.received: np.ndarray | None = None

    def detect(self, rgb: np.ndarray) -> tuple[Detection, ...]:
        self.received = np.asarray(rgb)
        return self.detections


def detection(
    object_class: ObjectClass,
    confidence: float,
    box: tuple[float, float, float, float],
) -> Detection:
    return Detection(object_class, confidence, BoundingBox(*box))


def projector() -> YoloMeasurementProjector:
    ground = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )
    return YoloMeasurementProjector(
        ground,
        MeasurementCalibrator(LinearRangeCalibration(2.0, 1.0)),
    )


def polar_measurement(range_m: float, bearing_rad: float) -> ObjectMeasurement:
    return ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=1.0,
        x_left_m=range_m * np.sin(bearing_rad),
        y_forward_m=range_m * np.cos(bearing_rad),
        range_m=range_m,
        bearing_rad=bearing_rad,
    )


def clean_ekf() -> PedestrianEKF:
    return PedestrianEKF(
        config=load_pedestrian_ekf_config(ROOT / "configs" / "oracle_ekf_v1.toml"),
        measurement_noise=load_polar_measurement_noise(
            ROOT / "configs" / "measurement_model_v1.toml"
        ),
        measurement_profile=MeasurementProfile.CLEAN,
    )


def test_frozen_detector_checkpoint_hash() -> None:
    checkpoint = ROOT / "artifacts" / "yolo_v1" / "best.pt"
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        FROZEN_CHECKPOINT_SHA256
    )


def test_f9_split_guard_is_disjoint_and_parameters_are_frozen() -> None:
    protocol = load_f9_protocol(ROOT / "configs" / "f9_yolo_ekf_v1.toml")
    assert set(protocol.calibration_seeds).isdisjoint(
        protocol.final_evaluation_seeds
    )
    assert set(protocol.calibration_seeds) == {4101, 4102, 4103, 4104}
    assert set(protocol.final_evaluation_seeds) == {5101, 5102, 5103, 5104}
    frozen = load_f9_protocol(
        ROOT / "configs" / "f9_yolo_ekf_v1.toml",
        require_frozen=True,
    )
    assert frozen.parameters_frozen
    assert frozen.calibration_artifact_sha256


def test_f9_bias_is_preprocessing_only_and_covariance_is_calibration_fixed() -> None:
    config = ROOT / "configs" / "f9_yolo_ekf_v1.toml"
    protocol = load_f9_protocol(config, require_frozen=True)
    noise = load_polar_measurement_noise(config)

    assert protocol.range_bias_m == pytest.approx(-0.045904804710162034)
    assert protocol.bearing_bias_rad == pytest.approx(0.00414567890700929)
    assert np.allclose(noise.bias(0.4), 0.0)
    assert noise.covariance(0.4)[0, 0] ** 0.5 == pytest.approx(
        0.003047553825257025
    )
    assert noise.covariance(0.7)[0, 0] ** 0.5 == pytest.approx(
        0.012491135642519849
    )
    assert noise.covariance(1.0)[0, 0] ** 0.5 == pytest.approx(
        0.01606193180500766
    )
    assert noise.covariance(1.0)[1, 1] ** 0.5 == pytest.approx(
        0.012647917882523984
    )


def test_f9_calibration_only_scenarios_cannot_enter_final_evaluation() -> None:
    protocol = load_f9_protocol(ROOT / "configs" / "f9_yolo_ekf_v1.toml")
    calibration_only = {
        spec.name
        for spec in protocol.scenarios
        if spec.use_for_calibration and not spec.use_for_final_evaluation
    }
    final = {
        spec.name for spec in protocol.scenarios if spec.use_for_final_evaluation
    }

    assert calibration_only == {
        "calibration_near_stationary",
        "calibration_medium_stationary",
    }
    assert calibration_only.isdisjoint(final)


def test_single_duckie_selection_is_highest_confidence_without_gt() -> None:
    low = detection(ObjectClass.DUCKIE, 0.4, (10.0, 10.0, 20.0, 30.0))
    high = detection(ObjectClass.DUCKIE, 0.9, (30.0, 10.0, 40.0, 30.0))
    sign = detection(ObjectClass.STOP_SIGN, 0.99, (50.0, 10.0, 60.0, 30.0))

    selection = select_single_duckie((low, sign, high))

    assert selection.detection_count == 2
    assert selection.selected is high
    assert selection.multiplicity
    assert list(inspect.signature(select_single_duckie).parameters) == ["detections"]


def test_yolo_miss_is_structural_none_measurement() -> None:
    detector = FakeDetector(())
    pipeline = YoloPedestrianMeasurementPipeline(detector, projector())

    result = pipeline.observe(np.zeros((480, 640, 3), dtype=np.uint8))

    assert not result.pedestrian.detected
    assert result.pedestrian.range_m is None
    assert result.pedestrian.bearing_rad is None
    assert result.duckie_detection_count == 0


def test_runtime_pipeline_uses_raw_projection_and_preserves_stop_detections() -> None:
    duckie = detection(ObjectClass.DUCKIE, 0.8, (300.0, 300.0, 340.0, 420.0))
    sign = detection(ObjectClass.STOP_SIGN, 0.7, (100.0, 100.0, 140.0, 250.0))
    detector = FakeDetector((sign, duckie))
    measurement_projector = projector()
    pipeline = YoloPedestrianMeasurementPipeline(detector, measurement_projector)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.observe(rgb)
    raw = measurement_projector.project_raw(duckie)
    historical_f5b = measurement_projector.project(duckie)

    assert detector.received is rgb
    assert result.pedestrian.range_m == pytest.approx(raw.raw_polar.range_m)
    assert result.pedestrian.range_m != pytest.approx(
        historical_f5b.calibrated_polar.range_m
    )
    assert result.stop_sign_detections == (sign,)


def test_pipeline_projects_every_duckie_candidate_not_only_the_selected_one() -> None:
    low = detection(ObjectClass.DUCKIE, 0.4, (10.0, 300.0, 60.0, 420.0))
    high = detection(ObjectClass.DUCKIE, 0.8, (300.0, 300.0, 340.0, 420.0))
    sign = detection(ObjectClass.STOP_SIGN, 0.7, (100.0, 100.0, 140.0, 250.0))
    detector = FakeDetector((sign, low, high))
    pipeline = YoloPedestrianMeasurementPipeline(detector, projector())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.observe(rgb)

    assert len(result.duckie_candidates) == 2
    assert all(candidate.measurement.detected for candidate in result.duckie_candidates)
    assert result.duckie_detection_count == 2


def test_selected_duckie_still_matches_highest_confidence_for_baseline_parity() -> None:
    low = detection(ObjectClass.DUCKIE, 0.4, (10.0, 300.0, 60.0, 420.0))
    high = detection(ObjectClass.DUCKIE, 0.8, (300.0, 300.0, 340.0, 420.0))
    detector = FakeDetector((low, high))
    pipeline = YoloPedestrianMeasurementPipeline(detector, projector())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.observe(rgb)
    best = max(result.duckie_candidates, key=lambda item: item.detection.confidence)

    assert result.selected_duckie is best.detection
    assert result.pedestrian.range_m == pytest.approx(best.measurement.range_m)


def test_a_candidate_whose_projection_fails_is_kept_with_its_error_and_no_measurement() -> None:
    ok = detection(ObjectClass.DUCKIE, 0.4, (10.0, 300.0, 60.0, 420.0))
    on_horizon = detection(ObjectClass.DUCKIE, 0.8, (300.0, 0.0, 340.0, 80.0))
    detector = FakeDetector((ok, on_horizon))
    pipeline = YoloPedestrianMeasurementPipeline(detector, projector())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.observe(rgb)
    failed = [c for c in result.duckie_candidates if c.projection_error is not None]

    assert failed
    assert all(not c.measurement.detected for c in failed)
    assert all(isinstance(c, CandidateProjection) for c in result.duckie_candidates)


def test_no_duckie_detections_yields_no_candidates_and_a_missing_measurement() -> None:
    detector = FakeDetector(())
    pipeline = YoloPedestrianMeasurementPipeline(detector, projector())
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.observe(rgb)

    assert result.duckie_candidates == ()
    assert not result.pedestrian.detected


def test_bias_correction_is_fixed_image_measurement_only_transform() -> None:
    raw = ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=0.8,
        x_left_m=0.0,
        y_forward_m=1.2,
        range_m=1.2,
        bearing_rad=0.1,
    )
    correction = AdditiveMeasurementBias(0.2, 0.03)

    corrected = correction.correct(raw)

    assert corrected.range_m == pytest.approx(1.0)
    assert corrected.bearing_rad == pytest.approx(0.07)
    assert corrected.x_left_m == pytest.approx(np.sin(0.07))
    assert corrected.y_forward_m == pytest.approx(np.cos(0.07))
    assert list(inspect.signature(AdditiveMeasurementBias.correct).parameters) == [
        "self",
        "measurement",
    ]


def test_nis_diagnostic_matches_innovation_quadratic_form() -> None:
    ekf: PedestrianEKF = clean_ekf()
    ekf.initialize(polar_measurement(1.0, 0.0))
    ekf.predict(EgoMotion(0.0, 0.0), 1.0 / 30.0)

    diagnostics = ekf.correct(polar_measurement(1.05, 0.02))

    innovation = np.array(
        [diagnostics.innovation_range_m, diagnostics.innovation_bearing_rad]
    )
    covariance = np.asarray(diagnostics.innovation_covariance)
    expected = float(innovation @ np.linalg.solve(covariance, innovation))
    assert diagnostics.nis == pytest.approx(expected)
    assert diagnostics.nis > 0.0


def test_f9_runtime_module_has_no_privileged_dependency() -> None:
    import duckie_pomdp.perception.f9_pipeline as module

    source = inspect.getsource(module)
    assert "PrivilegedSimulatorState" not in source
    assert "domain.privileged" not in source


def test_counterfactual_negative_renderer_is_privileged_only() -> None:
    from duckie_pomdp.adapters.gym_duckietown import (
        GymDuckietownAgentEnvironment,
        GymDuckietownProjectionValidationSource,
    )

    assert not hasattr(GymDuckietownAgentEnvironment, "render_without_objects")
    assert hasattr(
        GymDuckietownProjectionValidationSource,
        "render_without_objects",
    )


def test_f9_metrics_keep_real_misses_structural() -> None:
    rows = [
        {
            "eligible_visible": True,
            "measurement_detected": True,
            "selected_correct_iou50": True,
            "raw_measurement_range_m": 1.1,
            "raw_measurement_bearing_rad": 0.2,
            "gt_range_m": 1.0,
            "gt_bearing_rad": 0.1,
            "raw_nis": 2.0,
        },
        {
            "eligible_visible": True,
            "measurement_detected": False,
            "selected_correct_iou50": False,
            "raw_measurement_range_m": None,
            "raw_measurement_bearing_rad": None,
            "gt_range_m": 1.0,
            "gt_bearing_rad": 0.1,
            "raw_nis": None,
        },
    ]

    measurement = current_measurement_metrics(rows, corrected=False)
    nis = nis_metrics(rows, variant="raw", threshold=5.991464547107979)

    assert measurement["eligible_opportunities"] == 2
    assert measurement["detected_visible"] == 1
    assert measurement["range"]["rmse"] == pytest.approx(0.1)
    assert nis["count"] == 1


def test_calibration_statistics_use_only_correct_selected_measurements() -> None:
    from duckie_pomdp.evaluation.f9_calibration import summarize_calibration

    protocol = load_f9_protocol(ROOT / "configs" / "f9_yolo_ekf_v1.toml")
    rows: list[dict[str, object]] = []
    for bin_index, distance_bin in enumerate(("near", "medium", "far")):
        scale = (0.01, 0.03, 0.10)[bin_index]
        for index in range(31):
            rows.append(
                {
                    "selected_correct_iou50": True,
                    "range_error_m": 0.12 + scale * (index - 15) / 15.0,
                    "bearing_error_rad": -0.02 + 0.005 * (index - 15) / 15.0,
                    "distance_bin": distance_bin,
                    "fov_region": ("center", "mid_fov", "edge_fov")[bin_index],
                    "eligible_visible": True,
                    "metric_measurement_present": True,
                    "duplicate_selection": False,
                    "projection_error": "",
                }
            )
    for index in range(10):
        rows.append(
            {
                "selected_correct_iou50": False,
                "range_error_m": "",
                "bearing_error_rad": "",
                "distance_bin": "far",
                "fov_region": "outside",
                "eligible_visible": False,
                "metric_measurement_present": index == 0,
                "duplicate_selection": index == 0,
                "projection_error": "",
            }
        )

    report = summarize_calibration(rows, protocol)

    assert report["raw_measurement"]["global"]["sample_count"] == 93
    assert report["bias_correction"]["range_bias_m"] == pytest.approx(0.12)
    assert report["bias_correction"]["bearing_bias_rad"] == pytest.approx(-0.02)
    assert report["covariance"]["range_model"] == "distance_bins"
    assert report["existence_observation"]["detection_trials"] == 93
    assert report["existence_observation"]["false_alarm_trials"] == 10
    assert report["existence_observation"]["false_alarm_events"] == 1


def test_f9_final_artifacts_are_disjoint_and_misses_have_no_geometry() -> None:
    protocol = load_f9_protocol(
        ROOT / "configs" / "f9_yolo_ekf_v1.toml",
        require_frozen=True,
    )
    with (ROOT / "artifacts" / "f9_yolo_ekf_validation.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    metrics = json.loads(
        (ROOT / "artifacts" / "f9_belief_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(rows) == 2172
    assert {int(row["seed"]) for row in rows} == set(
        protocol.final_evaluation_seeds
    )
    assert not {int(row["seed"]) for row in rows} & set(
        protocol.calibration_seeds
    )
    assert metrics["metrics"]["row_count"] == len(rows)
    for row in rows:
        if row["measurement_detected"] == "False":
            assert row["raw_measurement_range_m"] == ""
            assert row["raw_measurement_bearing_rad"] == ""
            assert row["corrected_measurement_range_m"] == ""
            assert row["corrected_measurement_bearing_rad"] == ""
