"""Pure alignment and aggregation helpers for evaluation-only IG BEV maps.

The policy and Integrated-Gradients paths deliberately remain unaware of world
pose. This module joins an already-computed attribution trace with a separate
evaluation-only pose trace *after* both have been produced. It never invokes
the policy, simulator, detector, belief updater, or optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


POSE_TRACE_KEYS = (
    "seed",
    "step",
    "world_x_m",
    "world_z_m",
    "heading_rad",
)


@dataclass(frozen=True)
class GroupAttribution:
    names: tuple[str, ...]
    absolute: np.ndarray
    share: np.ndarray
    dominant_index: np.ndarray


def validate_pose_trace(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the deliberately small evaluation-only world-pose schema."""

    if tuple(arrays) != POSE_TRACE_KEYS:
        raise ValueError(
            "evaluation pose trace schema mismatch: "
            f"expected {POSE_TRACE_KEYS}, got {tuple(arrays)}"
        )
    rows = np.asarray(arrays["seed"]).shape[0]
    if rows == 0:
        raise ValueError("evaluation pose trace must not be empty")
    for key in POSE_TRACE_KEYS:
        value = np.asarray(arrays[key])
        if value.ndim != 1 or value.shape[0] != rows:
            raise ValueError(f"evaluation pose trace field {key!r} has wrong shape")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"evaluation pose trace field {key!r} is not finite")
    pairs = np.column_stack((arrays["seed"], arrays["step"]))
    if np.unique(pairs, axis=0).shape[0] != rows:
        raise ValueError("evaluation pose trace contains duplicate seed/step pairs")


def align_pose_to_samples(
    pose_trace: Mapping[str, np.ndarray],
    *,
    sample_seed: np.ndarray,
    sample_step: np.ndarray,
) -> dict[str, np.ndarray]:
    """Align pose rows to IG sample order using exact ``(seed, step)`` keys."""

    validate_pose_trace(pose_trace)
    seeds = np.asarray(sample_seed)
    steps = np.asarray(sample_step)
    if seeds.ndim != 1 or steps.shape != seeds.shape:
        raise ValueError("sample seed and step arrays must be matching vectors")
    lookup = {
        (int(seed), int(step)): index
        for index, (seed, step) in enumerate(
            zip(pose_trace["seed"], pose_trace["step"])
        )
    }
    try:
        indices = np.asarray(
            [lookup[(int(seed), int(step))] for seed, step in zip(seeds, steps)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"IG sample has no evaluation pose: {error.args[0]}") from error
    return {
        key: np.asarray(pose_trace[key])[indices].copy() for key in POSE_TRACE_KEYS
    }


def resolve_feature_groups(
    feature_names: Sequence[str],
    group_features: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, ...]]:
    """Resolve named semantic groups and require exact one-time coverage."""

    names = tuple(str(name) for name in feature_names)
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    index = {name: position for position, name in enumerate(names)}
    groups: dict[str, tuple[int, ...]] = {}
    for group, members in group_features.items():
        try:
            groups[str(group)] = tuple(index[str(member)] for member in members)
        except KeyError as error:
            raise ValueError(f"unknown feature in group {group!r}: {error.args[0]}") from error
    flattened = [position for positions in groups.values() for position in positions]
    if sorted(flattened) != list(range(len(names))):
        raise ValueError("feature groups must cover every feature exactly once")
    return groups


def aggregate_groups(
    attributions: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> GroupAttribution:
    """Aggregate per-feature IG magnitudes into semantic groups per frame."""

    values = np.asarray(attributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("attributions must be a non-empty (frames, features) array")
    names = tuple(str(name) for name in groups)
    if not names:
        raise ValueError("at least one attribution group is required")
    absolute = np.column_stack(
        [np.abs(values[:, tuple(indices)]).sum(axis=1) for indices in groups.values()]
    )
    totals = absolute.sum(axis=1, keepdims=True)
    share = np.divide(
        absolute,
        totals,
        out=np.zeros_like(absolute),
        where=totals > np.finfo(np.float64).eps,
    )
    return GroupAttribution(
        names=names,
        absolute=absolute,
        share=share,
        dominant_index=np.argmax(absolute, axis=1),
    )


def signed_total(attributions: np.ndarray) -> np.ndarray:
    """Return the signed IG sum, i.e. target change from the baseline."""

    values = np.asarray(attributions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("attributions must have shape (frames, features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("attributions must be finite")
    return values.sum(axis=1)

