from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.control import (
    LaneObservationNormalizer,
    LanePolicyObservation,
    load_lane_protocol,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import EgoObservation, SensorObservation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_lane_protocol(ROOT / "configs" / "f10_l1_lane_v1.toml")


def test_lane_observation_order_and_normalization_are_frozen() -> None:
    sensor = SensorObservation(
        front_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        ego=EgoObservation(0.075, -0.25, 0.20, -2.0),
    )
    semantic = LanePolicyObservation.from_sensor(sensor, PolicyAction(0.20, 2.0))
    assert semantic.ordering() == PROTOCOL.observation_order
    vector = LaneObservationNormalizer.from_protocol(PROTOCOL).normalize(semantic)
    assert vector.dtype == np.float32
    assert vector.shape == (6,)
    assert vector == pytest.approx((0.5, -0.5, 0.5, -0.5, 0.5, 0.5))


def test_lane_observation_contains_no_privileged_semantics() -> None:
    names = set(LanePolicyObservation.ordering())
    forbidden_fragments = ("world", "truth", "privileged", "lap", "yellow")
    assert not any(fragment in name for fragment in forbidden_fragments for name in names)


def test_lane_observation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        LanePolicyObservation(np.nan, 0.0, 0.0, 0.0, 0.0, 0.0)

