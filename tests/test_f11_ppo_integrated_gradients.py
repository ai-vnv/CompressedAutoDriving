from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    integrated_gradients,
    target_values,
)


class _LinearActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(3, 2, bias=True)
        self.critic = nn.Linear(3, 1, bias=True)
        with torch.no_grad():
            self.actor.weight.copy_(
                torch.tensor([[0.20, -0.10, 0.05], [0.10, 0.20, -0.15]])
            )
            self.actor.bias.copy_(torch.tensor([-0.20, 0.10]))
            self.critic.weight.copy_(torch.tensor([[0.50, -0.25, 0.75]]))
            self.critic.bias.copy_(torch.tensor([0.30]))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


LIMITS = PPOActionLimits(0.4, 4.0)
ROOT = Path(__file__).resolve().parents[1]


def test_physical_targets_follow_existing_action_mapping() -> None:
    model = _LinearActorCritic()
    observation = torch.zeros((1, 3))
    velocity = target_values(model, observation, target="v_cmd_mps", action_limits=LIMITS)
    omega = target_values(model, observation, target="omega_cmd_rad_s", action_limits=LIMITS)
    assert velocity.item() == pytest.approx(0.16)
    assert omega.item() == pytest.approx(0.4)


@pytest.mark.parametrize("target", ["v_cmd_mps", "omega_cmd_rad_s", "critic_value"])
def test_integrated_gradients_is_complete_for_linear_model(target: str) -> None:
    model = _LinearActorCritic()
    observations = torch.tensor([[0.5, -0.25, 0.75], [-0.2, 0.4, 0.1]])
    baselines = torch.zeros_like(observations)
    result = integrated_gradients(
        model,
        observations,
        baselines,
        target=target,
        action_limits=LIMITS,
        path_steps=32,
        sample_batch_size=1,
    )
    assert result.attributions.shape == observations.shape
    assert torch.max(torch.abs(result.completeness_delta)).item() < 1.0e-6


def test_integrated_gradients_does_not_modify_model_or_parameter_gradients() -> None:
    model = _LinearActorCritic()
    before = copy.deepcopy(model.state_dict())
    integrated_gradients(
        model,
        torch.ones((2, 3)),
        torch.zeros((2, 3)),
        target="critic_value",
        action_limits=LIMITS,
        path_steps=8,
    )
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_integrated_gradients_rejects_invalid_inputs() -> None:
    model = _LinearActorCritic()
    with pytest.raises(ValueError, match="same shape"):
        integrated_gradients(
            model,
            torch.zeros((2, 3)),
            torch.zeros((1, 3)),
            target="critic_value",
            action_limits=LIMITS,
        )
    invalid = torch.zeros((1, 3))
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        integrated_gradients(
            model,
            invalid,
            torch.zeros((1, 3)),
            target="critic_value",
            action_limits=LIMITS,
        )


def test_explanation_trajectory_schema_rejects_privileged_fields() -> None:
    script_path = ROOT / "experiments" / "explain_f11_ppo_integrated_gradients.py"
    spec = importlib.util.spec_from_file_location("f11_experiment", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="forbidden"):
        module.validate_trajectory_schema({"evaluation_gt": torch.zeros(1).numpy()})
