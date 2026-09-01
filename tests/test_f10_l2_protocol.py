from pathlib import Path

import pytest

from duckie_pomdp.control import load_lane_transfer_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l2_transfer_v1.toml"


def test_transfer_protocol_is_frozen_to_new_mixed_turn_map() -> None:
    protocol = load_lane_transfer_protocol(CONFIG)
    assert protocol.raw["stage"] == "F10-L2"
    assert protocol.raw["simulator"]["map"] == "experiment_loop"
    assert protocol.raw["simulator"]["direction"] == "closed_loop_mixed_turns"
    assert protocol.map_path.name == "experiment_loop.yaml"
    assert protocol.raw["lap"]["minimum_path_length_m"] == pytest.approx(6.8)
    assert protocol.sac.training_steps == 40_000


def test_transfer_protocol_preserves_source_checkpoint_and_seed_isolation() -> None:
    protocol = load_lane_transfer_protocol(CONFIG)
    assert protocol.source_global_step == 50_000
    assert protocol.transfer_checkpoint_path.name == "sac_lane_baseline.pt"
    assert protocol.transfer_checkpoint_sha256.startswith("7d492fbf")
    active = (
        set(protocol.seeds.training),
        set(protocol.seeds.development),
        set(protocol.seeds.final_evaluation),
    )
    assert active[0].isdisjoint(active[1])
    assert active[0].isdisjoint(active[2])
    assert active[1].isdisjoint(active[2])
    assert set.union(*active).isdisjoint(protocol.seeds.historical_evaluation)


def test_transfer_protocol_keeps_f10_l1_observation_and_action_contract() -> None:
    protocol = load_lane_transfer_protocol(CONFIG)
    assert protocol.observation_order == (
        "lateral_error_m",
        "heading_error_rad",
        "actual_linear_velocity_mps",
        "actual_yaw_rate_rad_s",
        "previous_linear_velocity_cmd_mps",
        "previous_angular_velocity_cmd_rad_s",
    )
    assert protocol.action_bounds == pytest.approx((0.0, 0.4, -4.0, 4.0))

