from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.belief.ego_motion import compensated_transition
from duckie_pomdp.belief.existence_filter import (
    ExistenceFilter,
    ExistenceFilterConfig,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    PedestrianEKF,
    load_pedestrian_ekf_config,
    measurement_function,
    measurement_jacobian,
)
from duckie_pomdp.belief.polar_transform import (
    cartesian_to_polar_state,
    polar_state_jacobian,
)
from duckie_pomdp.belief.updater import (
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise


ROOT = Path(__file__).resolve().parents[1]


def measurement(range_m: float, bearing_rad: float) -> ObjectMeasurement:
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


def finite_difference(function, state: np.ndarray, angular_rows: set[int]) -> np.ndarray:
    epsilon = 1.0e-6
    output = np.asarray(function(state), dtype=float)
    result = np.zeros((len(output), len(state)), dtype=float)
    for column in range(len(state)):
        delta = np.zeros_like(state)
        delta[column] = epsilon
        difference = np.asarray(function(state + delta)) - np.asarray(
            function(state - delta)
        )
        for row in angular_rows:
            difference[row] = wrap_angle(float(difference[row]))
        result[:, column] = difference / (2.0 * epsilon)
    return result


def test_polar_measurement_jacobian_matches_finite_difference() -> None:
    state = np.array([0.35, 0.8, -0.1, 0.2], dtype=float)
    analytical = measurement_jacobian(state)
    numerical = finite_difference(measurement_function, state, {1})

    assert analytical == pytest.approx(numerical, abs=1.0e-7)
    assert analytical[1, 0] > 0.0  # atan2(x_left, y_forward)
    assert analytical[1, 1] < 0.0


def test_full_polar_state_jacobian_matches_finite_difference() -> None:
    state = np.array([0.31, 0.82, 0.18, -0.07], dtype=float)
    ego = EgoMotion(0.12, 0.35)
    analytical = polar_state_jacobian(state, ego)
    numerical = finite_difference(
        lambda value: cartesian_to_polar_state(value, ego),
        state,
        {1},
    )

    assert analytical == pytest.approx(numerical, abs=2.0e-7)


def test_polar_to_cartesian_initialization_has_velocity_uncertainty() -> None:
    ekf = clean_ekf()
    ekf.initialize(measurement(0.8, 0.3))

    assert ekf.state[:2] == pytest.approx(
        [0.8 * np.sin(0.3), 0.8 * np.cos(0.3)]
    )
    assert ekf.state[2:] == pytest.approx([0.0, 0.0])
    assert ekf.covariance[2, 2] > 0.0
    assert ekf.covariance[3, 3] > 0.0


def test_actual_ego_translation_does_not_create_pedestrian_physical_velocity() -> None:
    ekf = clean_ekf()
    ekf.initialize(measurement(1.0, 0.0))
    ekf.predict(EgoMotion(0.2, 0.0), 1.0)

    assert ekf.state[:2] == pytest.approx([0.0, 0.8], abs=1.0e-12)
    assert ekf.state[2:] == pytest.approx([0.0, 0.0], abs=1.0e-12)
    polar = ekf.polar_moments(EgoMotion(0.2, 0.0))
    assert polar.mean[2] == pytest.approx(-0.2)


def test_actual_ego_rotation_changes_bearing_not_physical_velocity() -> None:
    ekf = clean_ekf()
    ekf.initialize(measurement(1.0, 0.0))
    ekf.predict(EgoMotion(0.0, np.pi / 2.0), 1.0)

    assert ekf.state[:2] == pytest.approx([-1.0, 0.0], abs=1.0e-12)
    assert ekf.state[2:] == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert measurement_function(ekf.state)[1] == pytest.approx(-np.pi / 2.0)


def test_ego_motion_affine_transition_uses_actual_motion() -> None:
    transition, offset = compensated_transition(EgoMotion(0.3, 0.0), 0.5)
    stationary_world_object = np.array([0.1, 1.0, 0.0, 0.0])
    predicted = transition @ stationary_world_object + offset

    assert predicted == pytest.approx([0.1, 0.85, 0.0, 0.0])


def test_prediction_only_grows_covariance_and_reobservation_contracts_it() -> None:
    ekf = clean_ekf()
    observed = measurement(0.9, 0.2)
    ekf.initialize(observed)
    traces = [float(np.trace(ekf.covariance))]
    missing = ObjectMeasurement.missing(ObjectClass.DUCKIE)
    for _ in range(5):
        ekf.step(missing, EgoMotion(0.0, 0.0), 1.0 / 30.0)
        traces.append(float(np.trace(ekf.covariance)))

    assert traces[1] > traces[0]
    assert traces[3] > traces[1]
    assert traces[5] > traces[3]
    before = traces[-1]
    ekf.step(observed, EgoMotion(0.0, 0.0), 1.0 / 30.0)
    assert float(np.trace(ekf.covariance)) < before


def test_ekf_bearing_innovation_wraps_across_pi() -> None:
    ekf = clean_ekf()
    ekf.initialize(measurement(1.0, np.pi - 0.01))
    diagnostics = ekf.correct(measurement(1.0, -np.pi + 0.01))

    assert diagnostics.innovation_bearing_rad == pytest.approx(0.02, abs=1.0e-9)


def test_existence_probability_is_bounded_and_responds_to_hits_and_misses() -> None:
    filter_ = ExistenceFilter(
        ExistenceFilterConfig(
            prior_probability=0.5,
            detection_probability=0.8,
            false_positive_probability=0.02,
            survival_probability=0.995,
            birth_probability=0.005,
        )
    )
    after_detection = filter_.update(True)
    after_misses = [filter_.update(False) for _ in range(5)]

    assert 0.0 <= after_detection <= 1.0
    assert all(0.0 <= value <= 1.0 for value in after_misses)
    assert after_detection > 0.5
    assert after_misses[-1] < after_detection


def updater() -> PedestrianBeliefUpdater:
    return PedestrianBeliefUpdater(
        ekf_config=load_pedestrian_ekf_config(
            ROOT / "configs" / "oracle_ekf_v1.toml"
        ),
        existence_config=load_existence_filter_config(
            ROOT / "configs" / "oracle_ekf_v1.toml"
        ),
        measurement_noise=load_polar_measurement_noise(
            ROOT / "configs" / "measurement_model_v1.toml"
        ),
        measurement_profile=MeasurementProfile.CLEAN,
    )


def perception(
    pedestrian: ObjectMeasurement,
    ego_motion: EgoMotion,
) -> PerceptionObservation:
    return PerceptionObservation(
        ego=EgoObservation(
            lateral_error_m=0.0,
            heading_error_rad=0.0,
            linear_velocity_mps=ego_motion.linear_velocity_mps,
            yaw_rate_rad_s=ego_motion.yaw_rate_rad_s,
        ),
        road=None,
        stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
        pedestrian=pedestrian,
    )


def test_belief_updater_uses_actual_motion_not_previous_command() -> None:
    actual = EgoMotion(0.0, 0.0)
    initial = initial_belief(perception(measurement(1.0, 0.0), actual).ego)
    first = updater()
    second = updater()
    first_belief = first.update(
        initial,
        PolicyAction(0.0, 0.0),
        actual,
        perception(measurement(1.0, 0.0), actual),
        1.0 / 30.0,
    )
    second_belief = second.update(
        initial,
        PolicyAction(0.4, 4.0),
        actual,
        perception(measurement(1.0, 0.0), actual),
        1.0 / 30.0,
    )

    assert first_belief == second_belief
    assert first.ekf.state == pytest.approx(second.ekf.state)
    assert first.last_diagnostics.previous_action != second.last_diagnostics.previous_action
    assert first.last_diagnostics.actual_ego_motion == actual


def test_belief_update_path_has_no_privileged_dependency() -> None:
    import inspect

    import duckie_pomdp.belief.updater as updater_module

    source = inspect.getsource(updater_module)
    assert "PrivilegedSimulatorState" not in source
    assert "domain.privileged" not in source


def test_public_belief_is_probabilistic_and_finite_after_detection() -> None:
    actual = EgoMotion(0.0, 0.0)
    observation = perception(measurement(0.9, -0.2), actual)
    belief: BeliefState = updater().update(
        initial_belief(observation.ego),
        PolicyAction(0.0, 0.0),
        actual,
        observation,
        1.0 / 30.0,
    )

    assert belief.pedestrian.existence_probability > 0.5
    assert belief.pedestrian.range_mean_m == pytest.approx(0.9)
    assert belief.pedestrian.bearing_mean_rad == pytest.approx(-0.2)
    assert belief.pedestrian.range_std_m > 0.0
    assert belief.pedestrian.radial_velocity_std_mps > 0.0
