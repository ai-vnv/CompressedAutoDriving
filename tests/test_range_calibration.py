from __future__ import annotations

import inspect
from math import inf, nan, pi

import pytest

from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.domain.privileged import WorldFootprint, WorldPoint, WorldPose
from duckie_pomdp.evaluation.range_calibration import (
    RangeSemanticsSample,
    fit_and_evaluate_range_calibration,
    nearest_surface_point,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
    load_measurement_calibrator,
    wrap_angle,
)


def test_origin_and_surface_range_are_distinct_semantics() -> None:
    footprint = WorldFootprint(
        vertices=(
            WorldPoint(1.5, -0.5),
            WorldPoint(2.5, -0.5),
            WorldPoint(2.5, 0.5),
            WorldPoint(1.5, 0.5),
        )
    )
    ego = WorldPose(0.0, 0.0, 0.0)

    surface = nearest_surface_point(footprint, ego)

    assert hypot2(WorldPoint(2.0, 0.0), ego) == pytest.approx(2.0)
    assert hypot2(surface, ego) == pytest.approx(1.5)


def test_runtime_calibrator_has_no_privileged_state_input() -> None:
    parameters = inspect.signature(MeasurementCalibrator.apply).parameters
    source = inspect.getsource(MeasurementCalibrator.apply)

    assert tuple(parameters) == ("self", "raw_range_m", "raw_bearing_rad")
    assert "PrivilegedSimulatorState" not in source


def test_fixed_calibration_is_deterministic() -> None:
    calibrator = MeasurementCalibrator(LinearRangeCalibration(0.95, 0.05))

    first = calibrator.apply(0.8, 0.2)
    second = calibrator.apply(0.8, 0.2)

    assert first == second
    assert first.range_m == pytest.approx(0.81)
    assert first.bearing_rad == pytest.approx(0.2)


def test_version_one_measurement_config_loads_fixed_runtime_parameters() -> None:
    calibrator = load_measurement_calibrator("configs/measurement_model_v1.toml")

    result = calibrator.apply(0.8, 0.2)

    assert result.range_m == pytest.approx(
        0.9507847585432267 * 0.8 + 0.05181745469768865
    )
    assert result.bearing_rad == pytest.approx(0.2)


@pytest.mark.parametrize("invalid", [nan, inf, -inf, -0.01])
def test_range_calibration_rejects_invalid_range(invalid: float) -> None:
    calibration = LinearRangeCalibration(1.0, 0.0)

    with pytest.raises(ValueError):
        calibration.apply(invalid)


def test_range_calibration_rejects_negative_calibrated_output() -> None:
    calibration = LinearRangeCalibration(1.0, -0.5)

    with pytest.raises(ValueError):
        calibration.apply(0.1)


def test_bearing_residual_wraps_across_pi() -> None:
    assert wrap_angle(-pi + 0.01 - (pi - 0.01)) == pytest.approx(0.02)


def test_calibration_fit_never_uses_held_out_episode_targets() -> None:
    calibration_samples = (
        sample("calibration_approach", 0.4, 0.5),
        sample("turn_left", 0.8, 0.9),
        sample("pedestrian_crossing", 1.0, 1.1),
    )
    validation = (
        sample("straight_approach", 0.6, 0.7),
        sample("turn_right", 0.7, 0.8),
    )
    shifted_validation = (
        sample("straight_approach", 0.6, 3.7),
        sample("turn_right", 0.7, 4.8),
    )

    first = fit_and_evaluate_range_calibration(calibration_samples + validation)
    second = fit_and_evaluate_range_calibration(
        calibration_samples + shifted_validation
    )

    assert first.calibration == second.calibration
    assert {row.split for row in first.rows} == {"calibration", "validation"}
    assert all(
        row.episode not in {"straight_approach", "turn_right"}
        for row in first.rows
        if row.split == "calibration"
    )


def sample(episode: str, raw_range: float, origin_range: float) -> RangeSemanticsSample:
    return RangeSemanticsSample(
        episode=episode,
        frame=0,
        step=0,
        object_type="duckie",
        pixel_u=320.0,
        pixel_v=200.0,
        projected=GroundPoint(0.0, raw_range),
        origin=GroundPoint(0.0, origin_range),
        surface=GroundPoint(0.0, max(0.0, origin_range - 0.04)),
        fov_region="center",
        silhouette_pixel_count=100,
    )


def hypot2(point: WorldPoint, ego: WorldPose) -> float:
    return ((point.x_m - ego.x_m) ** 2 + (point.z_m - ego.z_m) ** 2) ** 0.5
