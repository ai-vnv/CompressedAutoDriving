"""Frozen helpers for once-only F11 R004 actor attribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class InsufficientReferenceSupport(RuntimeError):
    """Raised when the locked same-phase reference rule cannot be satisfied."""


def locked_phase_seed_support(
    phases: Sequence[str], seeds: NDArray[np.int64]
) -> dict[str, tuple[int, ...]]:
    phase_array = np.asarray(phases, dtype="U40")
    seed_array = np.asarray(seeds, dtype=np.int64)
    if phase_array.shape != seed_array.shape or phase_array.ndim != 1:
        raise ValueError("phase and seed arrays must be aligned rank-1 vectors")
    return {
        str(phase): tuple(
            int(value) for value in np.unique(seed_array[phase_array == phase])
        )
        for phase in np.unique(phase_array)
    }


def draw_locked_same_phase_distinct_seed_references(
    observations: NDArray[np.float32],
    phases: Sequence[str],
    seeds: NDArray[np.int64],
    *,
    draw_seed: int,
    references_per_input: int,
    minimum_other_seed_support: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Draw one same-phase state from each of distinct non-factual seeds."""

    matrix = np.asarray(observations, dtype=np.float32)
    phase_array = np.asarray(phases, dtype="U40")
    seed_array = np.asarray(seeds, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("observations must be a non-empty rank-2 matrix")
    if phase_array.shape != (len(matrix),) or seed_array.shape != (len(matrix),):
        raise ValueError("phase and seed rows must align with observations")
    if not np.isfinite(matrix).all():
        raise ValueError("locked reference pool contains non-finite values")
    if references_per_input <= 0:
        raise ValueError("references_per_input must be positive")
    if minimum_other_seed_support < references_per_input:
        raise ValueError("minimum support cannot be below references per input")

    support = locked_phase_seed_support(phase_array, seed_array)
    rng = np.random.default_rng(int(draw_seed))
    indexes = np.empty((references_per_input, len(matrix)), dtype=np.int64)
    for row in range(len(matrix)):
        eligible_seeds = np.asarray(
            [value for value in support[str(phase_array[row])] if value != seed_array[row]],
            dtype=np.int64,
        )
        if len(eligible_seeds) < minimum_other_seed_support:
            raise InsufficientReferenceSupport(
                f"phase={phase_array[row]!r}, factual_seed={seed_array[row]}, "
                f"other_seed_support={len(eligible_seeds)}, "
                f"required={minimum_other_seed_support}"
            )
        selected_seeds = rng.choice(
            eligible_seeds, size=references_per_input, replace=False
        )
        for reference_index, selected_seed in enumerate(selected_seeds):
            pool = np.flatnonzero(
                (phase_array == phase_array[row]) & (seed_array == selected_seed)
            )
            indexes[reference_index, row] = int(rng.choice(pool))
    return np.asarray(matrix[indexes], dtype=np.float32), indexes


def mean_all_reference_attributions(
    draw_attributions: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply the frozen final estimator: equal mean over all draw estimates."""

    values = np.asarray(draw_attributions, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("draw attributions must have shape (draw,target,row,feature)")
    if not np.isfinite(values).all():
        raise ValueError("draw attributions contain non-finite values")
    return np.asarray(np.mean(values, axis=0), dtype=np.float32)

