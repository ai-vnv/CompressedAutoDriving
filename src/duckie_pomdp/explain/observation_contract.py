"""Audit helpers for the frozen RGB-to-belief-to-PPO observation boundary."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


_FORBIDDEN_POLICY_TOKENS = (
    "evaluation_gt",
    "privileged",
    "world_pose",
    "ground_truth",
    "gt_",
    "true_",
    "bbox",
    "silhouette",
)


def _is_forbidden_policy_name(name: str) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in _FORBIDDEN_POLICY_TOKENS):
        return True
    # IoU is an evaluation label, but the raw substring also occurs in the
    # legitimate word ``previous``.  Match it as a field token only.
    parts = tuple(part for part in lowered.split("_") if part)
    return "iou" in parts


def validate_feature_group_partition(
    observation_order: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> None:
    """Require every public feature exactly once in the primary group view."""

    order = tuple(str(name) for name in observation_order)
    flattened = tuple(str(name) for values in groups.values() for name in values)
    if len(flattened) != len(set(flattened)):
        raise ValueError("feature groups contain duplicate fields")
    if set(flattened) != set(order):
        missing = sorted(set(order) - set(flattened))
        unexpected = sorted(set(flattened) - set(order))
        raise ValueError(
            f"feature groups do not partition observation: "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_public_policy_mapping(
    mapping: Mapping[str, float], observation_order: Sequence[str]
) -> None:
    """Validate the semantic mapping before numerical normalization."""

    order = tuple(str(name) for name in observation_order)
    if tuple(mapping.keys()) != order:
        raise ValueError("public policy mapping ordering differs from frozen order")
    forbidden = sorted(
        name
        for name in mapping
        if _is_forbidden_policy_name(name)
    )
    if forbidden:
        raise ValueError(f"privileged/evaluation fields entered policy mapping: {forbidden}")
    values = np.asarray([mapping[name] for name in order], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("public policy mapping contains non-finite values")


def reconstruct_normalized_observation(
    mapping: Mapping[str, float],
    observation_order: Sequence[str],
    scales: Sequence[float],
    clip: float,
) -> NDArray[np.float32]:
    """Reconstruct the exact fixed physical normalization used by PPO."""

    validate_public_policy_mapping(mapping, observation_order)
    scale_array = np.asarray(scales, dtype=np.float64)
    if scale_array.shape != (len(observation_order),):
        raise ValueError("normalization scale shape differs from observation")
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0.0):
        raise ValueError("normalization scales must be finite and positive")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("normalization clip must be finite and positive")
    physical = np.asarray(
        [mapping[name] for name in observation_order], dtype=np.float64
    )
    return np.asarray(np.clip(physical / scale_array, -clip, clip), dtype=np.float32)


def deterministic_actor_statistics(
    model: nn.Module, observations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PPO actor distribution means and critic values without sampling."""

    if observations.ndim != 2:
        raise ValueError("observations must be rank two")
    if not torch.is_floating_point(observations) or not torch.isfinite(observations).all():
        raise ValueError("observations must be finite floating-point tensors")
    with torch.no_grad():
        actor_mean = model.actor(observations)
        critic_value = model.value(observations)
    if actor_mean.shape != (observations.shape[0], 2):
        raise ValueError("PPO actor mean must have shape (batch, 2)")
    if critic_value.shape != (observations.shape[0],):
        raise ValueError("PPO critic value must have shape (batch,)")
    return actor_mean, critic_value


def assert_policy_vector_precedes_privileged_read(method: object) -> None:
    """AST-check that one environment method constructs policy input first."""

    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    policy_lines: list[int] = []
    privileged_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "_policy_vector":
            policy_lines.append(node.lineno)
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "read"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "privileged"
        ):
            privileged_lines.append(node.lineno)
    if not policy_lines or not privileged_lines:
        raise ValueError("method lacks an auditable policy-vector/privileged boundary")
    if min(privileged_lines) <= min(policy_lines):
        raise ValueError("privileged state is read before the policy vector is built")
