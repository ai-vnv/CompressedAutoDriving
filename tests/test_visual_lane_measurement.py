from __future__ import annotations

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control.lane_belief_runtime import VisualLaneBeliefRuntime
from duckie_pomdp.domain.measurement import GroundPoint, LaneMeasurement
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    LaneMeasurementCalibration,
    LanePerceptionConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _projector() -> CalibratedGroundProjector:
    return CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )


def _synthetic_lane_image(
    *, lateral_error_m: float, heading_error_rad: float, curvature_inv_m: float
) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    projector = _projector()
    c1 = np.tan(heading_error_rad)
    c2 = 0.5 * curvature_inv_m * (1.0 + c1 * c1) ** 1.5
    for forward in np.linspace(0.10, 1.10, 250):
        centre = lateral_error_m + c1 * forward + c2 * forward**2
        for boundary, colour in (
            (centre + 0.117, np.array([220, 180, 10], dtype=np.uint8)),
            (centre - 0.117, np.array([210, 210, 210], dtype=np.uint8)),
        ):
            try:
                pixel = projector.ground_to_pixel(GroundPoint(boundary, forward))
            except ValueError:
                continue
            column, row = int(round(pixel.x_px)), int(round(pixel.y_px))
            if 2 <= column < 638 and 2 <= row < 478:
                image[row - 1 : row + 2, column - 2 : column + 3] = colour
    return image


def test_lane_measurement_missing_is_structural() -> None:
    missing = LaneMeasurement.missing(visible_point_count=4)
    assert not missing.detected
    assert missing.lateral_error_m is None
    assert missing.heading_error_rad is None
    assert missing.curvature_inv_m is None
    with pytest.raises(ValueError):
        LaneMeasurement(
            detected=False,
            lateral_error_m=0.0,
            lateral_error_std_m=None,
            heading_error_rad=None,
            heading_error_std_rad=None,
            curvature_inv_m=None,
            curvature_std_inv_m=None,
            visible_point_count=0,
            fit_residual_m=None,
        )


def test_camera_lane_estimator_recovers_locked_sign_conventions() -> None:
    truth = (0.035, -0.08, 0.0)
    measurement = CameraLaneMeasurementEstimator(
        _projector(),
        LanePerceptionConfig(minimum_point_count=10),
    ).estimate(
        _synthetic_lane_image(
            lateral_error_m=truth[0],
            heading_error_rad=truth[1],
            curvature_inv_m=truth[2],
        )
    )
    assert measurement.detected
    assert measurement.lateral_error_m == pytest.approx(truth[0], abs=0.015)
    assert measurement.heading_error_rad == pytest.approx(truth[1], abs=0.04)
    assert measurement.curvature_inv_m == pytest.approx(truth[2], abs=0.25)
    assert measurement.visible_point_count >= 10

    left_curve = CameraLaneMeasurementEstimator(_projector()).estimate(
        _synthetic_lane_image(
            lateral_error_m=0.0,
            heading_error_rad=0.0,
            curvature_inv_m=0.4,
        )
    )
    assert left_curve.detected
    assert left_curve.curvature_inv_m is not None
    assert left_curve.curvature_inv_m > 0.0


def test_camera_lane_estimator_uses_local_tangent_on_curve() -> None:
    """Heading is the tangent at the ego, not an average preview tangent."""

    truth = (0.025, -0.24, 1.6)
    measurement = CameraLaneMeasurementEstimator(
        _projector(),
        LanePerceptionConfig(minimum_point_count=10),
    ).estimate(
        _synthetic_lane_image(
            lateral_error_m=truth[0],
            heading_error_rad=truth[1],
            curvature_inv_m=truth[2],
        )
    )
    assert measurement.detected
    assert measurement.lateral_error_m == pytest.approx(truth[0], abs=0.025)
    assert measurement.heading_error_rad == pytest.approx(truth[1], abs=0.07)
    assert measurement.curvature_inv_m == pytest.approx(truth[2], abs=0.40)


def test_dual_boundary_mode_keeps_adaptive_pixels_semantically_unknown() -> None:
    image = _synthetic_lane_image(
        lateral_error_m=0.025,
        heading_error_rad=-0.06,
        curvature_inv_m=0.0,
    )
    marking = np.any(image != 0, axis=2)
    image[marking] = np.asarray([30, 140, 35], dtype=np.uint8)
    estimator = CameraLaneMeasurementEstimator(
        _projector(),
        LanePerceptionConfig(
            minimum_point_count=10,
            dual_boundary_fusion_enabled=True,
        ),
    )
    measurement, diagnostics = estimator.estimate_with_diagnostics(image)
    assert diagnostics.strict_yellow_pixel_count == 0
    assert diagnostics.strict_white_pixel_count == 0
    assert diagnostics.adaptive_unknown_pixel_count > 0
    assert diagnostics.source == "dual_with_adaptive"
    assert measurement.detected
    assert measurement.lateral_error_m == pytest.approx(0.025, abs=0.02)
    assert measurement.heading_error_rad == pytest.approx(-0.06, abs=0.05)


def test_dual_boundary_mode_fuses_strict_yellow_and_white() -> None:
    estimator = CameraLaneMeasurementEstimator(
        _projector(),
        LanePerceptionConfig(
            minimum_point_count=10,
            dual_boundary_fusion_enabled=True,
        ),
    )
    measurement, diagnostics = estimator.estimate_with_diagnostics(
        _synthetic_lane_image(
            lateral_error_m=-0.03,
            heading_error_rad=0.05,
            curvature_inv_m=0.25,
        )
    )
    assert diagnostics.source == "dual_strict"
    assert diagnostics.yellow_center_point_count >= 10
    assert diagnostics.white_center_point_count >= 10
    assert diagnostics.boundary_disagreement_m is not None
    assert diagnostics.boundary_disagreement_m < 0.060
    assert measurement.detected


def test_fusion_flag_off_preserves_legacy_measurement() -> None:
    image = _synthetic_lane_image(
        lateral_error_m=0.01,
        heading_error_rad=-0.04,
        curvature_inv_m=0.10,
    )
    default = CameraLaneMeasurementEstimator(_projector()).estimate(image)
    explicit = CameraLaneMeasurementEstimator(
        _projector(), LanePerceptionConfig(dual_boundary_fusion_enabled=False)
    ).estimate(image)
    assert explicit == default


def test_camera_lane_estimator_rejects_non_rgb_contracts() -> None:
    estimator = CameraLaneMeasurementEstimator(_projector())
    with pytest.raises(ValueError, match="uint8"):
        estimator.estimate(np.zeros((480, 640, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        estimator.estimate(np.zeros((240, 320, 3), dtype=np.uint8))


def test_lane_calibration_is_fixed_and_deterministic() -> None:
    raw = LaneMeasurement(
        True, 0.03, 0.005, -0.12, 0.01, 0.4, 0.1, 25, 0.004
    )
    calibration = LaneMeasurementCalibration(0.01, -0.02, 0.1, 0.02, 0.08, 0.5)
    first = calibration.apply(raw)
    second = calibration.apply(raw)
    assert first == second
    assert first.lateral_error_m == pytest.approx(0.02)
    assert first.heading_error_rad == pytest.approx(-0.10)
    assert first.curvature_inv_m == pytest.approx(0.3)
    assert first.lateral_error_std_m == pytest.approx(0.02)
    assert first.heading_error_std_rad == pytest.approx(0.08)
    assert first.curvature_std_inv_m == pytest.approx(0.5)


def test_lane_affine_calibration_maps_preview_without_runtime_truth() -> None:
    raw = LaneMeasurement(
        True, 0.10, 0.01, 0.20, 0.02, 0.30, 0.10, 30, 0.003
    )
    calibration = LaneMeasurementCalibration(
        0.0,
        0.0,
        0.0,
        0.01,
        0.02,
        0.10,
        affine_matrix=((0.5, 0.1, 0.0), (0.0, 0.5, 0.1), (0.0, 0.0, 1.0)),
        affine_offset=(0.01, -0.02, 0.0),
    )
    corrected = calibration.apply(raw)
    assert corrected.lateral_error_m == pytest.approx(0.08)
    assert corrected.heading_error_rad == pytest.approx(0.11)
    assert corrected.curvature_inv_m == pytest.approx(0.30)
    assert corrected.lateral_error_std_m is not None
    assert corrected.heading_error_std_rad is not None


def test_visual_lane_runtime_has_no_privileged_lane_or_command_input() -> None:
    assert set(signature(VisualLaneBeliefRuntime.reset).parameters) == {
        "self",
        "front_rgb",
    }
    assert set(signature(VisualLaneBeliefRuntime.update).parameters) == {
        "self",
        "front_rgb",
        "actual_ego_motion",
        "dt_s",
    }
    source = (ROOT / "src/duckie_pomdp/control/lane_belief_runtime.py").read_text()
    lowered = source.lower()
    assert "privilegedsimulatorstate" not in lowered
    assert "policyaction" not in lowered
