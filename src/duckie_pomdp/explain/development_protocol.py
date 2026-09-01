"""Public-only development protocol for F11 R002 and R003."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.control.ppo_observation import PPOVisualPolicyObservation
from duckie_pomdp.control.ppo_protocol import PPOCurriculumProtocol


PHASE_PRIORITY = (
    "combined_pedestrian_stop",
    "stop_required",
    "stop_satisfied",
    "pedestrian_relevant",
    "lane_curve",
    "nominal",
)


@dataclass(frozen=True)
class PhaseThresholds:
    pedestrian_existence: float
    pedestrian_max_range_m: float
    lane_curve_min_abs_curvature_inv_m: float
    stop_satisfied_vicinity_m: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.pedestrian_existence <= 1.0:
            raise ValueError("pedestrian existence threshold must be within [0,1]")
        if self.pedestrian_max_range_m <= 0.0:
            raise ValueError("pedestrian range threshold must be positive")
        if self.lane_curve_min_abs_curvature_inv_m < 0.0:
            raise ValueError("curve threshold cannot be negative")
        if self.stop_satisfied_vicinity_m <= 0.0:
            raise ValueError("stop-satisfied vicinity must be positive")


def public_phase(
    physical: Sequence[float],
    observation_order: Sequence[str],
    thresholds: PhaseThresholds,
) -> str:
    """Assign a mutually exclusive phase using only public policy fields."""

    values = _mapping(physical, observation_order)
    pedestrian = (
        values["pedestrian_existence_probability"]
        >= thresholds.pedestrian_existence
        and values["pedestrian_range_mean_m"]
        <= thresholds.pedestrian_max_range_m
    )
    required = values["stop_mode_required"] > 0.5
    satisfied = (
        values["stop_mode_satisfied"] > 0.5
        and abs(values["stop_line_distance_m"])
        <= thresholds.stop_satisfied_vicinity_m
    )
    curve = (
        abs(values["lane_curvature_mean_inv_m"])
        >= thresholds.lane_curve_min_abs_curvature_inv_m
    )
    if pedestrian and required:
        return "combined_pedestrian_stop"
    if required:
        return "stop_required"
    if satisfied:
        return "stop_satisfied"
    if pedestrian:
        return "pedestrian_relevant"
    if curve:
        return "lane_curve"
    return "nominal"


def schema_valid_public_vector(
    physical: Sequence[float], observation_order: Sequence[str]
) -> None:
    values = _mapping(physical, observation_order)
    PPOVisualPolicyObservation(**values)


def normalize_physical(
    physical: Sequence[float], protocol: PPOCurriculumProtocol
) -> NDArray[np.float32]:
    schema_valid_public_vector(physical, protocol.observation_order)
    values = np.asarray(physical, dtype=np.float64)
    scales = np.asarray(protocol.observation_scales, dtype=np.float64)
    return np.asarray(
        np.clip(values / scales, -protocol.observation_clip, protocol.observation_clip),
        dtype=np.float32,
    )


def build_r002_baselines(
    observations: NDArray[np.float32],
    physical: NDArray[np.float32],
    seeds: NDArray[np.int64],
    protocol: PPOCurriculumProtocol,
) -> dict[str, NDArray[np.float32]]:
    """Build the three frozen development baselines without GT."""

    _validate_matrix(observations, physical, seeds, protocol)
    reset = np.empty_like(observations)
    for seed in np.unique(seeds):
        indexes = np.flatnonzero(seeds == seed)
        reset[indexes] = observations[indexes[0]]

    median_physical = np.median(physical.astype(np.float64), axis=0)
    median_mapping = _mapping(median_physical, protocol.observation_order)
    modes = np.asarray(
        [
            np.mean(physical[:, protocol.observation_order.index(name)])
            for name in ("stop_mode_none", "stop_mode_required", "stop_mode_satisfied")
        ]
    )
    median_mapping["stop_mode_none"] = 0.0
    median_mapping["stop_mode_required"] = 0.0
    median_mapping["stop_mode_satisfied"] = 0.0
    median_mapping[
        ("stop_mode_none", "stop_mode_required", "stop_mode_satisfied")[
            int(np.argmax(modes))
        ]
    ] = 1.0
    public_median_row = normalize_physical(
        [median_mapping[name] for name in protocol.observation_order], protocol
    )
    public_median = np.repeat(public_median_row[None, :], len(observations), axis=0)

    neutral_mapping = dict(median_mapping)
    neutral = protocol.raw["neutral"]
    for name in _pedestrian_fields(protocol.observation_order):
        neutral_mapping[name] = float(neutral[name])
    for name in _stop_belief_fields(protocol.observation_order):
        neutral_mapping[name] = float(neutral[name])
    neutral_mapping["stop_mode_none"] = 1.0
    neutral_mapping["stop_mode_required"] = 0.0
    neutral_mapping["stop_mode_satisfied"] = 0.0
    neutral_row = normalize_physical(
        [neutral_mapping[name] for name in protocol.observation_order], protocol
    )
    neutral_hazard = np.repeat(neutral_row[None, :], len(observations), axis=0)
    return {
        "episode_reset": reset,
        "public_median": public_median,
        "semantic_neutral_hazard": neutral_hazard,
    }


def draw_phase_conditioned_references(
    observations: NDArray[np.float32],
    phases: Sequence[str],
    seeds: NDArray[np.int64],
    *,
    draw_seed: int,
    references_per_input: int,
    exclude_same_seed: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Draw public references from the same phase and another trajectory."""

    matrix = np.asarray(observations, dtype=np.float32)
    phase_array = np.asarray(phases, dtype="U40")
    seed_array = np.asarray(seeds, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("observations must be a non-empty rank-2 matrix")
    if phase_array.shape != (len(matrix),) or seed_array.shape != (len(matrix),):
        raise ValueError("phase/seed rows must align with observations")
    if not np.isfinite(matrix).all():
        raise ValueError("reference pool contains non-finite public observations")
    if references_per_input <= 0:
        raise ValueError("references_per_input must be positive")

    rng = np.random.default_rng(int(draw_seed))
    indexes = np.empty((references_per_input, len(matrix)), dtype=np.int64)
    all_indexes = np.arange(len(matrix), dtype=np.int64)
    for row in range(len(matrix)):
        eligible = phase_array == phase_array[row]
        if exclude_same_seed:
            eligible &= seed_array != seed_array[row]
        else:
            eligible[row] = False
        pool = all_indexes[eligible]
        if len(pool) < references_per_input:
            raise ValueError(
                f"insufficient reference support for phase={phase_array[row]!r}"
            )
        indexes[:, row] = rng.choice(
            pool, size=references_per_input, replace=False
        )
    return np.asarray(matrix[indexes], dtype=np.float32), indexes


def apply_semantic_intervention(
    physical: Sequence[float],
    intervention: str,
    protocol: PPOCurriculumProtocol,
    *,
    lane_low_confidence_validity: float,
    lane_low_confidence_min_lateral_std_m: float,
    lane_low_confidence_min_heading_std_rad: float,
    lane_low_confidence_min_curvature_std_inv_m: float,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    """Apply one complete semantic intervention and return normalized input."""

    factual = _mapping(physical, protocol.observation_order)
    changed = dict(factual)
    neutral = protocol.raw["neutral"]
    intended: tuple[str, ...]
    if intervention == "pedestrian_absent":
        intended = _pedestrian_fields(protocol.observation_order)
        for name in intended:
            changed[name] = float(neutral[name])
    elif intervention == "stop_absent":
        intended = (
            "stop_line_distance_m",
            *_stop_belief_fields(protocol.observation_order),
            "stop_mode_none",
            "stop_mode_required",
            "stop_mode_satisfied",
        )
        changed["stop_line_distance_m"] = float(neutral["stop_line_distance_m"])
        for name in _stop_belief_fields(protocol.observation_order):
            changed[name] = float(neutral[name])
        changed["stop_mode_none"] = 1.0
        changed["stop_mode_required"] = 0.0
        changed["stop_mode_satisfied"] = 0.0
    elif intervention == "lane_centered":
        intended = ("lane_lateral_error_mean_m", "lane_heading_error_mean_rad")
        changed["lane_lateral_error_mean_m"] = 0.0
        changed["lane_heading_error_mean_rad"] = 0.0
    elif intervention == "lane_low_confidence":
        intended = (
            "lane_validity_probability",
            "lane_lateral_error_std_m",
            "lane_heading_error_std_rad",
            "lane_curvature_std_inv_m",
        )
        changed["lane_validity_probability"] = lane_low_confidence_validity
        changed["lane_lateral_error_std_m"] = max(
            changed["lane_lateral_error_std_m"],
            lane_low_confidence_min_lateral_std_m,
        )
        changed["lane_heading_error_std_rad"] = max(
            changed["lane_heading_error_std_rad"],
            lane_low_confidence_min_heading_std_rad,
        )
        changed["lane_curvature_std_inv_m"] = max(
            changed["lane_curvature_std_inv_m"],
            lane_low_confidence_min_curvature_std_inv_m,
        )
    elif intervention == "previous_action_neutral":
        intended = (
            "previous_linear_velocity_cmd_mps",
            "previous_angular_velocity_cmd_rad_s",
        )
        changed[intended[0]] = 0.0
        changed[intended[1]] = 0.0
    elif intervention == "sham":
        intended = ()
    else:
        raise ValueError(f"unsupported semantic intervention: {intervention}")

    factual_values = np.asarray(
        [factual[name] for name in protocol.observation_order], dtype=np.float64
    )
    changed_values = np.asarray(
        [changed[name] for name in protocol.observation_order], dtype=np.float64
    )
    actual_changed = {
        protocol.observation_order[index]
        for index in np.flatnonzero(~np.isclose(factual_values, changed_values))
    }
    if not actual_changed.issubset(set(intended)):
        raise ValueError("intervention changed fields outside its registered group")
    return normalize_physical(changed_values, protocol), intended


def group_absolute_shares(
    attributions: NDArray[np.float32],
    observation_order: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> NDArray[np.float64]:
    totals = []
    for fields in groups.values():
        indexes = [observation_order.index(name) for name in fields]
        totals.append(np.abs(attributions[:, indexes]).sum(axis=1))
    values = np.stack(totals, axis=1).astype(np.float64)
    denominator = np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)
    return values / denominator


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = _rank(np.asarray(values_a, dtype=np.float64))
    b = _rank(np.asarray(values_b, dtype=np.float64))
    if np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _mapping(
    values: Sequence[float], observation_order: Sequence[str]
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(observation_order),) or not np.isfinite(array).all():
        raise ValueError("public vector shape/finite contract failed")
    return {name: float(array[index]) for index, name in enumerate(observation_order)}


def _validate_matrix(
    observations: NDArray[np.float32],
    physical: NDArray[np.float32],
    seeds: NDArray[np.int64],
    protocol: PPOCurriculumProtocol,
) -> None:
    expected = (len(observations), len(protocol.observation_order))
    if observations.shape != expected or physical.shape != expected:
        raise ValueError("development trace has the wrong observation shape")
    if seeds.shape != (len(observations),):
        raise ValueError("development trace seed shape mismatch")
    if not np.isfinite(observations).all() or not np.isfinite(physical).all():
        raise ValueError("development trace contains non-finite values")


def _pedestrian_fields(observation_order: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in observation_order if name.startswith("pedestrian_"))


def _stop_belief_fields(observation_order: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in observation_order if name.startswith("stop_sign_"))
