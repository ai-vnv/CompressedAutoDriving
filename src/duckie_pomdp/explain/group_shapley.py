"""Exact semantic-group Shapley for the frozen 29D Belief-PPO boundary."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


GROUP_ORDER = (
    "Lane",
    "Ego",
    "StopLine",
    "Pedestrian",
    "Stop",
    "PreviousAction",
)


@dataclass(frozen=True)
class ExactGroupShapleyResult:
    """Reference-level, draw-level, and equally aggregated exact attributions."""

    reference_attribution: NDArray[np.float32]  # [draw, ref, state, action, group]
    draw_attribution: NDArray[np.float32]  # [draw, state, action, group]
    mean_attribution: NDArray[np.float32]  # [state, action, group]
    reference_delta: NDArray[np.float32]  # [draw, ref, state, action]
    efficiency_residual: NDArray[np.float32]  # [draw, ref, state, action]


def validate_group_partition(
    feature_names: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> dict[str, tuple[int, ...]]:
    """Return ordered feature indexes after proving an exact 29D partition."""

    names = tuple(str(value) for value in feature_names)
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("F14 requires 29 unique public feature names")
    if tuple(groups) != GROUP_ORDER:
        raise ValueError(f"group order must be {GROUP_ORDER}")
    flattened = tuple(field for group in GROUP_ORDER for field in groups[group])
    if len(flattened) != 29 or len(set(flattened)) != 29 or set(flattened) != set(names):
        raise ValueError("six semantic groups must partition all 29 features exactly once")
    forbidden = {
        "evaluation_gt", "ground_truth", "privileged", "world_pose", "gt_iou",
        "future_reward", "true_pedestrian_position", "true_pedestrian_velocity",
    }
    if any(name.lower() in forbidden for name in names):
        raise ValueError("privileged feature found in public actor contract")
    return {
        group: tuple(names.index(field) for field in groups[group])
        for group in GROUP_ORDER
    }


def coalition_schema(players: int = 6) -> NDArray[np.bool_]:
    """All coalitions in stable integer-mask order, exactly once."""

    if players != 6:
        raise ValueError("F14 is frozen to six semantic players")
    masks = np.arange(1 << players, dtype=np.uint8)[:, None]
    bits = np.arange(players, dtype=np.uint8)[None, :]
    result = (masks & (1 << bits)) != 0
    if result.shape != (64, 6) or len({tuple(row) for row in result.tolist()}) != 64:
        raise RuntimeError("coalition enumeration failed")
    return result


def coalition_feature_mask(
    group_indices: Mapping[str, Sequence[int]], dimension: int = 29
) -> NDArray[np.bool_]:
    """Expand the 64 group coalitions to the frozen feature dimension."""

    group_mask = coalition_schema()
    feature_mask = np.zeros((64, dimension), dtype=np.bool_)
    for group_index, group in enumerate(GROUP_ORDER):
        feature_mask[:, tuple(group_indices[group])] = group_mask[:, group_index, None]
    return feature_mask


def build_coalition_vectors(
    factual: NDArray[np.floating],
    reference: NDArray[np.floating],
    group_indices: Mapping[str, Sequence[int]],
) -> NDArray[np.float32]:
    """Mix each factual/reference pair using all 64 complete-group coalitions."""

    left = np.asarray(factual, dtype=np.float32)
    right = np.asarray(reference, dtype=np.float32)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 29:
        raise ValueError("factual/reference rows must be aligned [N,29]")
    mask = coalition_feature_mask(group_indices)
    return np.where(mask[None, :, :], left[:, None, :], right[:, None, :]).astype(
        np.float32, copy=False
    )


def validate_public_matrix(
    values: NDArray[np.floating], feature_names: Sequence[str], *, clip: float
) -> None:
    """Vectorized schema checks applied to factual, reference, and coalition rows."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 29:
        raise ValueError("public actor matrix must have shape [N,29]")
    if not np.isfinite(matrix).all():
        raise ValueError("public actor matrix contains NaN/Inf")
    if np.max(np.abs(matrix), initial=0.0) > float(clip) + 1.0e-6:
        raise ValueError("public actor matrix exceeds frozen normalized bounds")
    order = tuple(str(value) for value in feature_names)
    for name in (
        "lane_validity_probability",
        "pedestrian_existence_probability",
        "stop_sign_existence_probability",
    ):
        column = matrix[:, order.index(name)]
        if np.any(column < -1.0e-6) or np.any(column > 1.0 + 1.0e-6):
            raise ValueError(f"invalid normalized probability: {name}")
    for name in (
        "lane_lateral_error_std_m", "lane_heading_error_std_rad",
        "lane_curvature_std_inv_m", "pedestrian_range_std_m",
        "pedestrian_bearing_std_rad", "pedestrian_radial_velocity_std_mps",
        "pedestrian_bearing_rate_std_rad_s", "stop_sign_range_std_m",
        "stop_sign_bearing_std_rad",
    ):
        if np.any(matrix[:, order.index(name)] < -1.0e-6):
            raise ValueError(f"negative normalized uncertainty: {name}")
    mode_indexes = [
        order.index("stop_mode_none"), order.index("stop_mode_required"),
        order.index("stop_mode_satisfied"),
    ]
    modes = matrix[:, mode_indexes]
    if (
        np.any(modes < -1.0e-6)
        or np.any(modes > 1.0 + 1.0e-6)
        or not np.allclose(modes.sum(axis=1), 1.0, atol=1.0e-6)
    ):
        raise ValueError("invalid stop-mode one-hot tuple")


def exact_group_shapley(
    actor_physical: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    factual: NDArray[np.float32],
    references: NDArray[np.float32],
    group_indices: Mapping[str, Sequence[int]],
    feature_names: Sequence[str],
    *,
    observation_clip: float,
    state_batch_size: int = 16,
) -> ExactGroupShapleyResult:
    """Evaluate exact six-player Shapley for complete-row distributional references.

    ``references`` is ``[draw, reference, state, 29]``. Each draw is averaged
    over its complete reference rows, then all draws are equally averaged.
    """

    x = np.asarray(factual, dtype=np.float32)
    refs = np.asarray(references, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 29:
        raise ValueError("factual matrix must have shape [N,29]")
    if refs.ndim != 4 or refs.shape[2:] != x.shape:
        raise ValueError("references must have shape [draw,reference,N,29]")
    if state_batch_size <= 0:
        raise ValueError("state_batch_size must be positive")
    validate_public_matrix(x, feature_names, clip=observation_clip)
    validate_public_matrix(refs.reshape(-1, 29), feature_names, clip=observation_clip)

    draws, references_per_draw, states, _ = refs.shape
    ref_phi = np.empty((draws, references_per_draw, states, 2, 6), dtype=np.float32)
    deltas = np.empty((draws, references_per_draw, states, 2), dtype=np.float32)
    residuals = np.empty_like(deltas)
    weights = _shapley_weight_matrix()

    for start in range(0, states, state_batch_size):
        stop = min(states, start + state_batch_size)
        batch = stop - start
        reference_batch = refs[:, :, start:stop, :]
        factual_batch = np.broadcast_to(x[None, None, start:stop, :], reference_batch.shape)
        pairs = draws * references_per_draw * batch
        coalitions = build_coalition_vectors(
            factual_batch.reshape(pairs, 29),
            reference_batch.reshape(pairs, 29),
            group_indices,
        )
        flat = coalitions.reshape(pairs * 64, 29)
        validate_public_matrix(flat, feature_names, clip=observation_clip)
        outputs = np.asarray(actor_physical(flat), dtype=np.float32)
        if outputs.shape != (pairs * 64, 2) or not np.isfinite(outputs).all():
            raise ValueError("actor must return finite physical [N,2] actions")
        outputs = outputs.reshape(pairs, 64, 2)
        phi = np.einsum("gm,pma->pag", weights, outputs, optimize=True)
        delta = outputs[:, -1, :] - outputs[:, 0, :]
        residual = phi.sum(axis=-1) - delta
        ref_phi[:, :, start:stop, :, :] = phi.reshape(
            draws, references_per_draw, batch, 2, 6
        )
        deltas[:, :, start:stop, :] = delta.reshape(
            draws, references_per_draw, batch, 2
        )
        residuals[:, :, start:stop, :] = residual.reshape(
            draws, references_per_draw, batch, 2
        )

    draw = ref_phi.mean(axis=1, dtype=np.float64).astype(np.float32)
    mean = draw.mean(axis=0, dtype=np.float64).astype(np.float32)
    return ExactGroupShapleyResult(ref_phi, draw, mean, deltas, residuals)


def _shapley_weight_matrix() -> NDArray[np.float64]:
    """Matrix W[g, coalition] whose product with f(coalition) is phi_g."""

    players = 6
    result = np.zeros((players, 1 << players), dtype=np.float64)
    for group in range(players):
        bit = 1 << group
        for coalition in range(1 << players):
            if coalition & bit:
                continue
            size = int(coalition.bit_count())
            weight = factorial(size) * factorial(players - size - 1) / factorial(players)
            result[group, coalition | bit] += weight
            result[group, coalition] -= weight
    return result
