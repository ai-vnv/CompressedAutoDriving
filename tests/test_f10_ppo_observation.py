from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control.ppo_observation import (
    PPOFixedObservationNormalizer,
    PPOPolicyObservation,
    PPOVisualPolicyObservation,
    neutral_pedestrian,
    neutral_stop_sign,
    policy_observation_from_belief,
)
from duckie_pomdp.control.ppo_protocol import STAGE_NAMES, load_ppo_curriculum_protocol
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import (
    BeliefState,
    LaneBelief,
    PedestrianBelief,
    RoadBelief,
)
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import StopMode


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_v1.toml"
VISUAL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_visual_v2.toml"
)
EXISTENCE_GATED_VISUAL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "f10_ppo_visual_objects_v30_source.toml"
)


def _neutral_belief(protocol):
    return BeliefState(
        ego=EgoObservation(0.0, 0.0, 0.0, 0.0),
        road=RoadBelief(0.0, protocol.raw["neutral"]["stop_line_distance_m"], StopMode.NONE),
        stop_sign=neutral_stop_sign(protocol),
        pedestrian=neutral_pedestrian(protocol),
    )


def test_policy_observation_order_and_dimension_are_stable():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    assert PPOPolicyObservation.ordering() == protocol.observation_order
    assert len(PPOPolicyObservation.ordering()) == 25
    for stage in STAGE_NAMES:
        assert len(protocol.stage(stage).training_seeds) > 0


def test_neutral_hazard_is_absent_not_zero_range():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    observation = PPOPolicyObservation.from_belief(
        _neutral_belief(protocol), PolicyAction(0.0, 0.0)
    )
    assert observation.pedestrian_existence_probability == 0.0
    assert observation.pedestrian_range_mean_m > 0.0
    assert observation.stop_sign_existence_probability == 0.0
    assert observation.stop_sign_range_mean_m > 0.0
    assert (observation.stop_mode_none, observation.stop_mode_required, observation.stop_mode_satisfied) == (1.0, 0.0, 0.0)


def test_fixed_normalization_is_finite_and_immutable():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    normalizer = PPOFixedObservationNormalizer(protocol)
    semantic = PPOPolicyObservation.from_belief(
        _neutral_belief(protocol), PolicyAction(0.1, -0.2)
    )
    before = normalizer.scales
    first = normalizer.normalize(semantic)
    second = normalizer.normalize(semantic)
    assert first.shape == (25,)
    assert np.array_equal(first, second)
    assert np.array_equal(before, normalizer.scales)
    assert not normalizer.scales.flags.writeable


def test_policy_observation_contains_no_privileged_names():
    forbidden = ("true_", "gt_", "world_", "bbox", "_iou", "collision", "future")
    assert not any(
        token in name
        for name in PPOPolicyObservation.ordering()
        for token in forbidden
    )


def _visual_neutral_belief(protocol, *, with_lane: bool = True):
    return BeliefState(
        ego=EgoObservation(0.0, 0.0, 0.12, -0.3),
        road=RoadBelief(
            99.0,
            protocol.raw["neutral"]["stop_line_distance_m"],
            StopMode.NONE,
        ),
        stop_sign=neutral_stop_sign(protocol),
        pedestrian=neutral_pedestrian(protocol),
        lane=(
            LaneBelief(0.8, 0.02, 0.03, -0.04, 0.07, 1.2, 0.8)
            if with_lane
            else None
        ),
    )


def test_visual_observation_has_frozen_29d_order_and_lane_curvature_source():
    protocol = load_ppo_curriculum_protocol(VISUAL_CONFIG)
    belief = _visual_neutral_belief(protocol)
    observation = PPOVisualPolicyObservation.from_belief(
        belief, PolicyAction(0.21, 0.4)
    )
    assert PPOVisualPolicyObservation.ordering() == protocol.observation_order
    assert observation.to_vector().shape == (29,)
    assert observation.lane_curvature_mean_inv_m == 1.2
    assert observation.lane_curvature_mean_inv_m != belief.road.curvature_inv_m
    assert observation.actual_linear_velocity_mps == 0.12
    assert observation.previous_linear_velocity_cmd_mps == 0.21
    assert np.all(np.isfinite(PPOFixedObservationNormalizer(protocol).normalize(observation)))


def test_visual_observation_requires_explicit_lane_belief():
    protocol = load_ppo_curriculum_protocol(VISUAL_CONFIG)
    with pytest.raises(ValueError, match="explicit LaneBelief"):
        PPOVisualPolicyObservation.from_belief(
            _visual_neutral_belief(protocol, with_lane=False),
            PolicyAction(0.0, 0.0),
        )


def test_visual_neutral_hazards_cannot_mean_zero_range():
    protocol = load_ppo_curriculum_protocol(VISUAL_CONFIG)
    observation = PPOVisualPolicyObservation.from_belief(
        _visual_neutral_belief(protocol), PolicyAction(0.0, 0.0)
    )
    assert observation.pedestrian_existence_probability == 0.0
    assert observation.pedestrian_range_mean_m > 0.0
    assert observation.stop_sign_existence_probability == 0.0
    assert observation.stop_sign_range_mean_m > 0.0
    assert observation.stop_line_distance_m == 2.0


def test_low_existence_masks_stale_pedestrian_kinematics_when_configured():
    protocol = load_ppo_curriculum_protocol(EXISTENCE_GATED_VISUAL_CONFIG)
    neutral = _visual_neutral_belief(protocol)
    stale = BeliefState(
        ego=neutral.ego,
        road=neutral.road,
        stop_sign=neutral.stop_sign,
        pedestrian=PedestrianBelief(
            0.003,
            0.69,
            0.02,
            -0.26,
            0.03,
            -0.4,
            0.1,
            0.5,
            0.2,
        ),
        lane=neutral.lane,
    )
    observation = policy_observation_from_belief(
        protocol, stale, PolicyAction(0.0, 0.0)
    )
    expected = neutral_pedestrian(protocol)
    assert observation.pedestrian_existence_probability == 0.0
    assert observation.pedestrian_range_mean_m == expected.range_mean_m
    assert observation.pedestrian_bearing_mean_rad == expected.bearing_mean_rad


def test_high_existence_preserves_public_pedestrian_belief_when_configured():
    protocol = load_ppo_curriculum_protocol(EXISTENCE_GATED_VISUAL_CONFIG)
    neutral = _visual_neutral_belief(protocol)
    active = PedestrianBelief(0.9, 0.7, 0.03, 0.2, 0.04, -0.2, 0.1, 0.3, 0.2)
    belief = BeliefState(
        ego=neutral.ego,
        road=neutral.road,
        stop_sign=neutral.stop_sign,
        pedestrian=active,
        lane=neutral.lane,
    )
    observation = policy_observation_from_belief(
        protocol, belief, PolicyAction(0.0, 0.0)
    )
    assert observation.pedestrian_existence_probability == 0.9
    assert observation.pedestrian_range_mean_m == 0.7
    assert observation.pedestrian_bearing_mean_rad == 0.2
