from __future__ import annotations

import numpy as np
import pytest

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import (
    BeliefState,
    PedestrianBelief,
    RoadBelief,
    StopSignBelief,
)
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.domain.measurement import (
    ObjectMeasurement,
    PerceptionObservation,
)
from duckie_pomdp.domain.observation import (
    EgoObservation,
    RoadMeasurement,
    SensorObservation,
)
from duckie_pomdp.domain.reward import RewardTerms
from duckie_pomdp.domain.state import (
    EgoState,
    PedestrianState,
    RoadState,
    StopMode,
    StopSignState,
)
from duckie_pomdp.domain.transition import Transition


def ego_observation() -> EgoObservation:
    return EgoObservation(
        lateral_error_m=0.01,
        heading_error_rad=-0.02,
        linear_velocity_mps=0.25,
        yaw_rate_rad_s=0.1,
    )


def road_measurement() -> RoadMeasurement:
    return RoadMeasurement(
        curvature_inv_m=0.0,
        stop_line_distance_m=0.8,
    )


def test_actual_state_and_command_are_distinct_contracts() -> None:
    ego = EgoState(0.01, -0.02, 0.25, 0.1)
    action = PolicyAction(0.1, -0.4)

    assert ego.linear_velocity_mps == 0.25
    assert ego.yaw_rate_rad_s == 0.1
    assert action.linear_velocity_mps == 0.1
    assert action.angular_velocity_rad_s == -0.4


def test_stop_sign_range_and_stop_line_distance_are_separate() -> None:
    road = RoadState(0.0, 0.35, StopMode.REQUIRED)
    sign = StopSignState(True, 0.9, 0.15)

    assert road.stop_line_distance_m == 0.35
    assert sign.range_m == 0.9


def test_absent_true_object_has_no_kinematic_values() -> None:
    pedestrian = PedestrianState(False, None, None, None, None)

    assert pedestrian.range_m is None

    with pytest.raises(ValueError, match="absent pedestrian"):
        PedestrianState(False, 0.0, 0.0, 0.0, 0.0)


def test_bbox_bottom_center_is_ground_contact_pixel() -> None:
    detection = Detection(
        object_class=ObjectClass.DUCKIE,
        confidence=0.8,
        bounding_box=BoundingBox(10.0, 20.0, 30.0, 60.0),
    )

    assert detection.bottom_center.x_px == 20.0
    assert detection.bottom_center.y_px == 60.0


def test_missed_detection_uses_none_not_zero() -> None:
    measurement = ObjectMeasurement.missing(ObjectClass.DUCKIE)

    assert not measurement.detected
    assert measurement.confidence is None
    assert measurement.range_m is None
    assert measurement.bearing_rad is None

    with pytest.raises(ValueError, match="must use None"):
        ObjectMeasurement(
            object_class=ObjectClass.DUCKIE,
            detected=False,
            confidence=0.0,
            x_left_m=0.0,
            y_forward_m=0.0,
            range_m=0.0,
            bearing_rad=0.0,
        )


def test_perception_observation_cannot_swap_object_classes() -> None:
    stop_sign = ObjectMeasurement.missing(ObjectClass.STOP_SIGN)
    pedestrian = ObjectMeasurement.missing(ObjectClass.DUCKIE)

    perception = PerceptionObservation(
        ego=ego_observation(),
        road=road_measurement(),
        stop_sign=stop_sign,
        pedestrian=pedestrian,
    )

    assert perception.stop_sign.object_class is ObjectClass.STOP_SIGN

    with pytest.raises(ValueError, match="stop-sign measurement"):
        PerceptionObservation(
            ego=ego_observation(),
            road=road_measurement(),
            stop_sign=pedestrian,
            pedestrian=pedestrian,
        )


def test_policy_state_is_probabilistic_belief() -> None:
    belief = BeliefState(
        ego=ego_observation(),
        road=RoadBelief(0.0, 0.8, StopMode.NONE),
        stop_sign=StopSignBelief(0.1, 1.2, 0.3, 0.1, 0.05),
        pedestrian=PedestrianBelief(
            0.7,
            0.9,
            0.2,
            -0.1,
            0.04,
            -0.2,
            0.1,
            0.05,
            0.02,
        ),
    )

    assert belief.pedestrian.existence_probability == 0.7
    assert belief.pedestrian.radial_velocity_std_mps == 0.1


def test_sensor_observation_enforces_rgb_contract() -> None:
    observation = SensorObservation(
        front_rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        ego=ego_observation(),
        road=road_measurement(),
    )

    assert observation.front_rgb.shape == (48, 64, 3)

    with pytest.raises(ValueError, match="uint8"):
        SensorObservation(
            front_rgb=np.zeros((48, 64, 3), dtype=np.float32),
            ego=ego_observation(),
            road=road_measurement(),
        )


def test_transition_reward_is_sum_of_logged_terms() -> None:
    terms = RewardTerms(
        progress=1.0,
        lane=-0.1,
        stop=0.2,
        pedestrian=-0.3,
        comfort=-0.05,
        collision=0.0,
    )
    transition = Transition(
        observation=SensorObservation(
            front_rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            ego=ego_observation(),
            road=road_measurement(),
        ),
        reward_terms=terms,
        terminated=False,
        truncated=False,
    )

    assert transition.reward == pytest.approx(0.75)
