import inspect
from pathlib import Path

import numpy as np

from duckie_pomdp.control import SACAgent, load_lane_transfer_protocol
from experiments.train_f10_l2_sac import load_transfer_agent, train


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_l2_transfer_v1.toml"


def test_warm_start_preserves_source_policy_exactly() -> None:
    protocol = load_lane_transfer_protocol(CONFIG)
    source, payload = SACAgent.load(protocol.transfer_checkpoint_path, device="cpu")
    transferred, transferred_payload = load_transfer_agent(protocol, smoke=False)
    observation = np.asarray((0.1, -0.2, 0.3, -0.1, 0.4, -0.3), dtype=np.float32)
    assert np.allclose(
        source.act(observation, deterministic=True),
        transferred.act(observation, deterministic=True),
        atol=1.0e-7,
        rtol=0.0,
    )
    for name, source_tensor in source.actor.state_dict().items():
        assert np.array_equal(
            source_tensor.cpu().numpy(),
            transferred.actor.state_dict()[name].cpu().numpy(),
        )
    assert payload["global_step"] == transferred_payload["global_step"] == 50_000
    assert transferred.update_count == source.update_count
    assert transferred.config.seed == 16_000


def test_transfer_buffer_fill_uses_policy_not_uniform_random_action() -> None:
    source = inspect.getsource(train)
    assert "agent.act(observation, deterministic=False)" in source
    assert "action_rng" not in source


def test_transfer_runtime_api_has_no_privileged_input() -> None:
    from duckie_pomdp.control.lane_transfer_environment import LaneTransferEnvironment

    reset_parameters = inspect.signature(LaneTransferEnvironment.reset).parameters
    step_parameters = inspect.signature(LaneTransferEnvironment.step).parameters
    assert set(reset_parameters) == {"self", "seed", "options"}
    assert set(step_parameters) == {"self", "action"}
    assert "privileged" not in inspect.getsource(LaneTransferEnvironment)
