from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import (
    FixedObservationNormalizer,
    PolicyObservation,
    SACActionMapper,
    load_f10_protocol,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import (
    BeliefState,
    PedestrianBelief,
    RoadBelief,
    StopSignBelief,
)
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import StopMode


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_f10_protocol(ROOT / "configs" / "f10_sac_v1.toml")


def belief() -> BeliefState:
    return BeliefState(
        ego=EgoObservation(0.10, -0.20, 0.30, -0.40),
        road=RoadBelief(0.50, 0.75, StopMode.NONE),
        stop_sign=StopSignBelief(0.5, 0.0, 2.0, 0.0, np.pi),
        pedestrian=PedestrianBelief(
            0.8, 0.9, 0.1, 0.2, 0.03, -0.15, 0.12, 0.4, 0.2
        ),
    )


def test_policy_vector_order_is_config_frozen_and_finite() -> None:
    observation = PolicyObservation.from_belief(belief(), PolicyAction(0.1, -0.5))
    vector = observation.to_vector()
    assert PolicyObservation.ordering() == PROTOCOL.observation_order
    assert vector.shape == (17,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert vector.tolist() == pytest.approx([
        0.10, -0.20, 0.30, -0.40, 0.50, 0.75,
        0.8, 0.9, 0.1, 0.2, 0.03, -0.15, 0.12, 0.4, 0.2,
        0.1, -0.5,
    ])


def test_fixed_normalization_is_deterministic_and_cannot_update() -> None:
    normalizer = FixedObservationNormalizer.from_protocol(PROTOCOL)
    observation = PolicyObservation.from_belief(belief(), PolicyAction(0.1, -0.5))
    before = normalizer.normalize(observation)
    after = normalizer.normalize(observation)
    assert np.array_equal(before, after)
    assert not hasattr(normalizer, "update")
    scales = normalizer.scales
    with pytest.raises(ValueError):
        scales[0] = 999.0


def test_policy_interface_contains_no_privileged_state() -> None:
    names = set(PolicyObservation.ordering())
    forbidden_tokens = {"world", "truth", "privileged", "gt", "iou", "silhouette"}
    assert not any(forbidden_tokens & set(name.split("_")) for name in names)


@pytest.mark.parametrize(
    ("normalized", "expected"),
    [
        ((-1.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), (0.4, 4.0)),
        ((1.0, -1.0), (0.4, -4.0)),
        ((0.0, 0.0), (0.2, 0.0)),
    ],
)
def test_sac_action_mapping_reuses_physical_chassis_bounds(normalized, expected) -> None:
    mapping = SACActionMapper(0.4, 4.0).map(normalized)
    assert mapping.policy_action.linear_velocity_mps == pytest.approx(expected[0])
    assert mapping.policy_action.angular_velocity_rad_s == pytest.approx(expected[1])
    assert not mapping.clipped


def test_sac_action_mapping_clips_and_reports_out_of_bounds() -> None:
    mapping = SACActionMapper(0.4, 4.0).map((2.0, -3.0))
    assert mapping.final_normalized == (1.0, -1.0)
    assert mapping.policy_action == PolicyAction(0.4, -4.0)
    assert mapping.clipped


@pytest.mark.parametrize("bad", [(np.nan, 0.0), (0.0, np.inf), (0.0,), (0.0, 0.0, 0.0)])
def test_sac_action_mapping_rejects_invalid_actions(bad) -> None:
    with pytest.raises(ValueError):
        SACActionMapper(0.4, 4.0).map(bad)
