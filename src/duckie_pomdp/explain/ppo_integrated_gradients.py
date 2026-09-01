"""Integrated Gradients for the frozen feed-forward PPO actor and critic.

The module deliberately accepts only normalized policy vectors. Simulator
truth, rendered images, and belief-updater internals are outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ExplanationTarget = Literal["v_cmd_mps", "omega_cmd_rad_s", "critic_value"]


@dataclass(frozen=True)
class PPOActionLimits:
    maximum_linear_velocity_mps: float
    maximum_angular_velocity_rad_s: float

    def __post_init__(self) -> None:
        if self.maximum_linear_velocity_mps <= 0.0:
            raise ValueError("maximum linear velocity must be positive")
        if self.maximum_angular_velocity_rad_s <= 0.0:
            raise ValueError("maximum angular velocity must be positive")


@dataclass(frozen=True)
class IntegratedGradientsResult:
    attributions: torch.Tensor
    input_values: torch.Tensor
    baseline_values: torch.Tensor
    completeness_delta: torch.Tensor


@dataclass(frozen=True)
class DistributionalIntegratedGradientsResult:
    """Mean IG over an empirical distribution of valid reference states."""

    attributions: torch.Tensor
    input_values: torch.Tensor
    mean_reference_values: torch.Tensor
    completeness_delta: torch.Tensor
    reference_count: int


def target_values(
    model: nn.Module,
    observation: torch.Tensor,
    *,
    target: ExplanationTarget,
    action_limits: PPOActionLimits,
) -> torch.Tensor:
    """Return one differentiable scalar policy target per observation row."""

    if observation.ndim != 2:
        raise ValueError("observation must be a rank-2 tensor")
    if target == "critic_value":
        value = model.value(observation)
        if value.ndim != 1:
            raise ValueError("critic value target must have shape (batch,)")
        return value

    mean = model.actor(observation)
    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError("PPO actor must return shape (batch, 2)")
    bounded = torch.clamp(mean, -1.0, 1.0)
    if target == "v_cmd_mps":
        return 0.5 * (bounded[:, 0] + 1.0) * action_limits.maximum_linear_velocity_mps
    if target == "omega_cmd_rad_s":
        return bounded[:, 1] * action_limits.maximum_angular_velocity_rad_s
    raise ValueError(f"unsupported explanation target: {target}")


def integrated_gradients(
    model: nn.Module,
    observations: torch.Tensor,
    baselines: torch.Tensor,
    *,
    target: ExplanationTarget,
    action_limits: PPOActionLimits,
    path_steps: int = 128,
    sample_batch_size: int = 128,
) -> IntegratedGradientsResult:
    """Compute trapezoidal Integrated Gradients for a frozen PPO target.

    The function differentiates only with respect to the interpolated input.
    It neither calls an optimizer nor populates parameter ``.grad`` fields.
    """

    _validate_inputs(observations, baselines, path_steps, sample_batch_size)
    original_training = model.training
    model.eval()
    attribution_batches: list[torch.Tensor] = []
    input_value_batches: list[torch.Tensor] = []
    baseline_value_batches: list[torch.Tensor] = []
    delta_batches: list[torch.Tensor] = []

    try:
        for start in range(0, observations.shape[0], sample_batch_size):
            stop = min(start + sample_batch_size, observations.shape[0])
            current = observations[start:stop]
            baseline = baselines[start:stop]
            difference = current - baseline
            gradient_integral = torch.zeros_like(current)

            for path_index in range(path_steps + 1):
                alpha = path_index / path_steps
                point = (baseline + alpha * difference).detach().requires_grad_(True)
                values = target_values(
                    model,
                    point,
                    target=target,
                    action_limits=action_limits,
                )
                gradient = torch.autograd.grad(
                    values.sum(), point, create_graph=False, retain_graph=False
                )[0]
                weight = 0.5 if path_index in (0, path_steps) else 1.0
                gradient_integral.add_(gradient, alpha=weight / path_steps)

            attribution = difference * gradient_integral
            with torch.no_grad():
                input_value = target_values(
                    model,
                    current,
                    target=target,
                    action_limits=action_limits,
                )
                baseline_value = target_values(
                    model,
                    baseline,
                    target=target,
                    action_limits=action_limits,
                )
                completeness = attribution.sum(dim=1) - (
                    input_value - baseline_value
                )
            attribution_batches.append(attribution.detach())
            input_value_batches.append(input_value.detach())
            baseline_value_batches.append(baseline_value.detach())
            delta_batches.append(completeness.detach())
    finally:
        model.train(original_training)

    return IntegratedGradientsResult(
        attributions=torch.cat(attribution_batches, dim=0),
        input_values=torch.cat(input_value_batches, dim=0),
        baseline_values=torch.cat(baseline_value_batches, dim=0),
        completeness_delta=torch.cat(delta_batches, dim=0),
    )


def distributional_integrated_gradients(
    model: nn.Module,
    observations: torch.Tensor,
    references: torch.Tensor,
    *,
    target: ExplanationTarget,
    action_limits: PPOActionLimits,
    path_steps: int = 128,
    sample_batch_size: int = 128,
) -> DistributionalIntegratedGradientsResult:
    """Average exact path IG across sampled public reference states.

    ``references`` has shape ``(reference_count, sample_count, feature_count)``.
    The completeness target is ``F(x) - mean(F(x'))`` for those references.
    """

    if references.ndim != 3 or references.shape[0] == 0:
        raise ValueError("references must be a non-empty rank-3 tensor")
    if tuple(references.shape[1:]) != tuple(observations.shape):
        raise ValueError("each reference set must match observations")
    if references.device != observations.device:
        raise ValueError("observations and references must use the same device")
    if not torch.is_floating_point(references) or not torch.isfinite(references).all():
        raise ValueError("references must contain only finite floating-point values")

    attribution_sum = torch.zeros_like(observations)
    reference_value_sum: torch.Tensor | None = None
    input_values: torch.Tensor | None = None
    for reference_index in range(references.shape[0]):
        result = integrated_gradients(
            model,
            observations,
            references[reference_index],
            target=target,
            action_limits=action_limits,
            path_steps=path_steps,
            sample_batch_size=sample_batch_size,
        )
        attribution_sum.add_(result.attributions)
        if reference_value_sum is None:
            reference_value_sum = torch.zeros_like(result.baseline_values)
            input_values = result.input_values
        reference_value_sum.add_(result.baseline_values)

    count = int(references.shape[0])
    attributions = attribution_sum / count
    if reference_value_sum is None or input_values is None:  # pragma: no cover
        raise RuntimeError("reference aggregation unexpectedly produced no result")
    mean_reference_values = reference_value_sum / count
    completeness_delta = attributions.sum(dim=1) - (
        input_values - mean_reference_values
    )
    return DistributionalIntegratedGradientsResult(
        attributions=attributions,
        input_values=input_values,
        mean_reference_values=mean_reference_values,
        completeness_delta=completeness_delta,
        reference_count=count,
    )


def _validate_inputs(
    observations: torch.Tensor,
    baselines: torch.Tensor,
    path_steps: int,
    sample_batch_size: int,
) -> None:
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("observations must be a non-empty rank-2 tensor")
    if observations.shape != baselines.shape:
        raise ValueError("observations and baselines must have the same shape")
    if observations.device != baselines.device:
        raise ValueError("observations and baselines must use the same device")
    if not torch.is_floating_point(observations) or not torch.is_floating_point(baselines):
        raise TypeError("observations and baselines must be floating-point tensors")
    if not torch.isfinite(observations).all() or not torch.isfinite(baselines).all():
        raise ValueError("observations and baselines must contain only finite values")
    if path_steps < 2:
        raise ValueError("path_steps must be at least two")
    if sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive")
