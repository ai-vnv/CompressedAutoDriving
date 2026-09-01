from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from duckie_pomdp.explain.development_protocol import (
    draw_phase_conditioned_references,
)
from duckie_pomdp.explain.ppo_integrated_gradients import (
    PPOActionLimits,
    distributional_integrated_gradients,
)


ROOT = Path(__file__).resolve().parents[1]


class LinearPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_layer = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.actor_layer.weight.copy_(
                torch.tensor([[0.2, -0.1, 0.3], [-0.1, 0.2, 0.1]])
            )

    def actor(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_layer(observation)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.sum(dim=1)


def test_phase_conditioned_references_are_deterministic_and_cross_seed() -> None:
    observations = np.arange(36, dtype=np.float32).reshape(12, 3)
    phases = np.asarray(["nominal"] * 6 + ["lane_curve"] * 6)
    seeds = np.asarray([1, 1, 2, 2, 3, 3] * 2, dtype=np.int64)
    references, indexes = draw_phase_conditioned_references(
        observations,
        phases,
        seeds,
        draw_seed=2026081401,
        references_per_input=2,
    )
    repeated, repeated_indexes = draw_phase_conditioned_references(
        observations,
        phases,
        seeds,
        draw_seed=2026081401,
        references_per_input=2,
    )
    np.testing.assert_array_equal(references, repeated)
    np.testing.assert_array_equal(indexes, repeated_indexes)
    assert references.shape == (2, 12, 3)
    for row in range(len(observations)):
        assert np.all(phases[indexes[:, row]] == phases[row])
        assert np.all(seeds[indexes[:, row]] != seeds[row])


def test_distributional_ig_matches_linear_expected_reference_identity() -> None:
    model = LinearPolicy()
    observations = torch.tensor([[0.2, 0.3, 0.1], [0.1, -0.2, 0.2]])
    references = torch.tensor(
        [
            [[0.0, 0.1, 0.0], [-0.1, -0.1, 0.0]],
            [[0.1, 0.0, -0.1], [0.0, 0.0, 0.1]],
        ]
    )
    result = distributional_integrated_gradients(
        model,
        observations,
        references,
        target="omega_cmd_rad_s",
        action_limits=PPOActionLimits(0.4, 4.0),
        path_steps=8,
        sample_batch_size=2,
    )
    expected = (observations - references.mean(dim=0)) * torch.tensor(
        [-0.4, 0.8, 0.4]
    )
    torch.testing.assert_close(result.attributions, expected, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        result.completeness_delta,
        torch.zeros(2),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert result.reference_count == 2


def test_distributional_ig_rejects_invalid_reference_shape() -> None:
    model = LinearPolicy()
    with np.testing.assert_raises(ValueError):
        distributional_integrated_gradients(
            model,
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            target="v_cmd_mps",
            action_limits=PPOActionLimits(0.4, 4.0),
        )


def test_r002b_protocol_does_not_contain_locked_seed_execution_mode() -> None:
    source = (ROOT / "experiments" / "run_f11_r002b_expected_gradients.py").read_text()
    config = (ROOT / "configs" / "f11_ppo_explanation_r002b_v1.toml").read_text()
    assert "PPOCurriculumEnvironment" not in source
    assert "locked_evaluation_seeds" in source
    assert "development_trace.npz" in config
