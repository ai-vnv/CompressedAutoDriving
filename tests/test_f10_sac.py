from pathlib import Path

import numpy as np
import pytest
import torch

from duckie_pomdp.control import ReplayBuffer, SACAgent, SACConfig


def config() -> SACConfig:
    return SACConfig(
        observation_dimension=5,
        action_dimension=2,
        hidden_sizes=(32, 32),
        learning_rate=3.0e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=16,
        replay_buffer_size=100,
        learning_starts=16,
        train_frequency=1,
        gradient_steps=1,
        initial_entropy_coefficient=0.2,
        target_entropy=-2.0,
        seed=7,
        device="cpu",
    )


def populated_buffer() -> ReplayBuffer:
    replay = ReplayBuffer(100, 5, 2, seed=8)
    rng = np.random.default_rng(9)
    for index in range(40):
        observation = rng.normal(size=5).astype(np.float32)
        replay.add(
            observation,
            np.tanh(rng.normal(size=2)).astype(np.float32),
            float(rng.normal()),
            (observation + 0.01).astype(np.float32),
            terminated=index % 17 == 0,
        )
    return replay


def test_sac_action_is_finite_and_bounded() -> None:
    agent = SACAgent(config())
    for deterministic in (False, True):
        action = agent.act(np.zeros(5, dtype=np.float32), deterministic=deterministic)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_sac_update_changes_actor_and_critic_parameters() -> None:
    agent = SACAgent(config())
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    q_before = [parameter.detach().clone() for parameter in agent.q1.parameters()]
    metrics = agent.update(populated_buffer())
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(q_before, agent.q1.parameters()))
    assert agent.update_count == 1


def test_sac_checkpoint_round_trip_preserves_deterministic_action(tmp_path: Path) -> None:
    agent = SACAgent(config())
    agent.update(populated_buffer())
    observation = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    expected = agent.act(observation, deterministic=True)
    path = tmp_path / "sac.pt"
    agent.save(path, global_step=123, metadata={"purpose": "test"})
    loaded, payload = SACAgent.load(path, device="cpu")
    assert np.array_equal(expected, loaded.act(observation, deterministic=True))
    assert payload["global_step"] == 123
    assert payload["metadata"] == {"purpose": "test"}


def test_replay_buffer_bootstrap_flag_uses_termination_not_truncation() -> None:
    replay = ReplayBuffer(2, 5, 2, seed=1)
    replay.add(np.zeros(5, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0, np.ones(5, dtype=np.float32), terminated=False)
    sample = replay.sample(1, torch.device("cpu"))
    assert sample["terminated"].item() == pytest.approx(0.0)
