from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState, WorldPose
from duckie_pomdp.domain.state import (
    EgoState,
    POMDPState,
    PedestrianState,
    RoadState,
    StopMode,
    StopSignState,
)
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.oracle_measurement import (
    OracleDetectionConfig,
    OracleMode,
    OracleObservationModel,
    load_oracle_detection_config,
)


ROOT = Path(__file__).resolve().parents[1]
NOISE_PATH = ROOT / "configs" / "measurement_model_v1.toml"
ORACLE_PATH = ROOT / "configs" / "oracle_ekf_v1.toml"


def privileged_state(
    *,
    pedestrian_range_m: float = 0.7,
    pedestrian_bearing_rad: float = 0.2,
    pedestrian_exists: bool = True,
) -> PrivilegedSimulatorState:
    pedestrian = (
        PedestrianState(
            True,
            pedestrian_range_m,
            pedestrian_bearing_rad,
            0.0,
            0.0,
        )
        if pedestrian_exists
        else PedestrianState(False, None, None, None, None)
    )
    return PrivilegedSimulatorState(
        true_pomdp_state=POMDPState(
            ego=EgoState(0.0, 0.0, 0.0, 0.0),
            road=RoadState(0.0, 1.0, StopMode.NONE),
            stop_sign=StopSignState(True, 0.9, -0.1),
            pedestrian=pedestrian,
        ),
        ego_world_pose=WorldPose(0.0, 0.0, 0.0),
        stop_sign_world_position=None,
        stop_sign_world_footprint=None,
        stop_line_world_position=None,
        pedestrian_world_position=None,
        pedestrian_world_footprint=None,
        pedestrian_world_velocity=None,
        collision=None,
    )


def oracle(mode: OracleMode, seed: int = 9) -> OracleObservationModel:
    return OracleObservationModel(
        mode=mode,
        measurement_noise=load_polar_measurement_noise(NOISE_PATH),
        detection=load_oracle_detection_config(ORACLE_PATH),
        seed=seed,
    )


def test_oracle_clean_returns_exact_canonical_geometry_without_recalibration() -> None:
    measurement = oracle(OracleMode.CLEAN).observe(
        privileged_state(pedestrian_range_m=0.7, pedestrian_bearing_rad=0.2),
        ObjectClass.DUCKIE,
    )

    assert measurement.detected
    assert measurement.confidence == 1.0
    assert measurement.range_m == pytest.approx(0.7)
    assert measurement.bearing_rad == pytest.approx(0.2)
    assert measurement.x_left_m == pytest.approx(0.7 * np.sin(0.2))
    assert measurement.y_forward_m == pytest.approx(0.7 * np.cos(0.2))


def test_oracle_sequence_is_reproducible_without_global_random_state() -> None:
    first = oracle(OracleMode.DROPOUT, seed=101)
    second = oracle(OracleMode.DROPOUT, seed=101)
    truth = privileged_state()

    sequence_a = [first.observe(truth, ObjectClass.DUCKIE) for _ in range(30)]
    np.random.seed(777)
    sequence_b = [second.observe(truth, ObjectClass.DUCKIE) for _ in range(30)]

    assert sequence_a == sequence_b


def test_oracle_dropout_and_absence_use_structural_missing_values() -> None:
    always_miss = OracleObservationModel(
        mode=OracleMode.DROPOUT,
        measurement_noise=load_polar_measurement_noise(NOISE_PATH),
        detection=OracleDetectionConfig(0.05, 2.0, 1.2, 1.0, 0.0),
        seed=1,
    )
    missed = always_miss.observe(privileged_state(), ObjectClass.DUCKIE)
    absent = oracle(OracleMode.NOISY).observe(
        privileged_state(pedestrian_exists=False),
        ObjectClass.DUCKIE,
    )

    for measurement in (missed, absent):
        assert not measurement.detected
        assert measurement.range_m is None
        assert measurement.bearing_rad is None
        assert measurement.confidence is None


def test_oracle_noise_matches_configured_bin_statistics() -> None:
    model = oracle(OracleMode.NOISY, seed=1234)
    truth = privileged_state(pedestrian_range_m=0.7, pedestrian_bearing_rad=-0.2)
    measurements = [model.observe(truth, ObjectClass.DUCKIE) for _ in range(50_000)]
    range_errors = np.array([item.range_m - 0.7 for item in measurements])
    bearing_errors = np.array(
        [wrap_angle(item.bearing_rad + 0.2) for item in measurements]
    )
    expected_bin = model.measurement_noise.range_bin(0.7)

    assert np.mean(range_errors) == pytest.approx(
        expected_bin.residual_bias_m,
        abs=1.0e-4,
    )
    assert np.std(range_errors, ddof=1) == pytest.approx(
        expected_bin.sigma_m,
        rel=0.02,
    )
    assert np.mean(bearing_errors) == pytest.approx(
        model.measurement_noise.bearing_bias_rad,
        abs=2.0e-4,
    )
    assert np.std(bearing_errors, ddof=1) == pytest.approx(
        model.measurement_noise.bearing_sigma_rad,
        rel=0.02,
    )


def test_oracle_boundary_does_not_leak_privileged_fields() -> None:
    measurement = oracle(OracleMode.CLEAN).observe(
        privileged_state(),
        ObjectClass.DUCKIE,
    )
    names = {field.name for field in fields(ObjectMeasurement)}

    assert "privileged" not in names
    assert "ground_truth" not in names
    assert not hasattr(measurement, "true_pomdp_state")


def test_oracle_rejects_unsupported_false_positive_generation() -> None:
    with pytest.raises(ValueError, match="false positives are disabled"):
        OracleDetectionConfig(0.05, 2.0, 1.2, 0.2, 0.01)

