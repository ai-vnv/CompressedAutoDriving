from pathlib import Path

import numpy as np
import pytest
import torch

from duckie_pomdp.control.ppo import PPOAgent, PPOConfig, PPORolloutBuffer


def _config() -> PPOConfig:
    return PPOConfig(
        observation_dimension=25,
        action_dimension=2,
        hidden_sizes=(32, 32),
        learning_rate=3e-4,
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_function_coefficient=0.5,
        max_gradient_norm=0.5,
        initial_log_std=-0.5,
        seed=123,
        device="cpu",
    )


def test_ppo_actor_and_critic_share_only_policy_visible_shape():
    agent = PPOAgent(_config())
    observation = np.zeros(25, dtype=np.float32)
    action = agent.act(observation, deterministic=True)
    assert action.raw_action.shape == (2,)
    assert np.all(np.isfinite(action.raw_action))
    assert np.all(action.environment_action <= 1.0)
    assert np.all(action.environment_action >= -1.0)


def test_ppo_update_changes_parameters_and_is_finite():
    config = _config()
    agent = PPOAgent(config)
    buffer = PPORolloutBuffer(config.n_steps, 25, 2)
    rng = np.random.default_rng(7)
    for index in range(config.n_steps):
        observation = rng.normal(size=25).astype(np.float32)
        action = agent.act(observation, deterministic=False)
        next_observation = rng.normal(size=25).astype(np.float32)
        buffer.add(
            observation,
            action.raw_action,
            action.log_probability,
            action.value,
            float(rng.normal()),
            agent.value(next_observation),
            terminated=index == config.n_steps - 1,
            episode_done=index == config.n_steps - 1,
        )
    before = [parameter.detach().clone() for parameter in agent.model.parameters()]
    metrics = agent.update(buffer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(left, right) for left, right in zip(before, agent.model.parameters()))


def test_checkpoint_retains_actor_critic_optimizer_and_exact_action(tmp_path: Path):
    agent = PPOAgent(_config())
    observation = np.linspace(-1, 1, 25, dtype=np.float32)
    expected = agent.act(observation, deterministic=True).environment_action
    path = tmp_path / "ppo.pt"
    agent.save(path, global_step=32, stage="c0", metadata={"test": True})
    loaded, payload = PPOAgent.load(path, device="cpu")
    actual = loaded.act(observation, deterministic=True).environment_action
    assert np.array_equal(expected, actual)
    assert payload["algorithm"] == "canonical_feedforward_ppo"
    assert payload["stage"] == "c0"
    assert "optimizer_state" in payload


def test_time_limit_bootstraps_value_but_stops_gae_across_episode():
    buffer = PPORolloutBuffer(1, 25, 2)
    buffer.add(
        np.zeros(25, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        0.0,
        0.5,
        1.0,
        2.0,
        terminated=False,
        episode_done=True,
    )
    advantages, returns = buffer.advantages_and_returns(0.99, 0.95)
    assert advantages[0] == pytest.approx(1.0 + 0.99 * 2.0 - 0.5)
    assert returns[0] == pytest.approx(1.0 + 0.99 * 2.0)


def test_true_termination_does_not_bootstrap_value():
    buffer = PPORolloutBuffer(1, 25, 2)
    buffer.add(
        np.zeros(25, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        0.0,
        0.5,
        1.0,
        999.0,
        terminated=True,
        episode_done=True,
    )
    advantages, returns = buffer.advantages_and_returns(0.99, 0.95)
    assert advantages[0] == pytest.approx(0.5)
    assert returns[0] == pytest.approx(1.0)
