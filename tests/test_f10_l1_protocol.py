from pathlib import Path

import pytest

from duckie_pomdp.control import load_lane_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l1_lane_v1.toml"


def test_lane_protocol_is_counterclockwise_and_seed_disjoint() -> None:
    protocol = load_lane_protocol(CONFIG)
    assert protocol.raw["simulator"]["map"] == "small_loop"
    assert protocol.raw["simulator"]["direction"] == "counterclockwise"
    train = set(protocol.seeds.training)
    development = set(protocol.seeds.development)
    final = set(protocol.seeds.final_evaluation)
    assert train.isdisjoint(development)
    assert train.isdisjoint(final)
    assert development.isdisjoint(final)
    assert (train | development | final).isdisjoint(
        protocol.seeds.historical_evaluation
    )


def test_lane_protocol_reuses_validated_action_envelope() -> None:
    protocol = load_lane_protocol(CONFIG)
    assert protocol.action_bounds == pytest.approx((0.0, 0.4, -4.0, 4.0))
    assert protocol.sac.training_steps == 60_000

