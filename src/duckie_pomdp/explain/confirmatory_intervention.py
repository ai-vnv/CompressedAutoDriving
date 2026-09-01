"""Statistics for preregistered, paired semantic policy interventions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def seed_cluster_bootstrap_ci(
    values: Sequence[float],
    seeds: Sequence[int],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Return an equal-seed cluster-bootstrap CI for the mean effect."""

    array = np.asarray(values, dtype=np.float64)
    seed_array = np.asarray(seeds, dtype=np.int64)
    if array.ndim != 1 or seed_array.shape != array.shape or len(array) == 0:
        raise ValueError("paired values/seeds must be non-empty one-dimensional arrays")
    if not np.isfinite(array).all():
        raise ValueError("paired effects must be finite")
    unique = np.unique(seed_array)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two seeds")
    if replicates <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    seed_means = np.asarray(
        [np.mean(array[seed_array == value]) for value in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        seed_means, size=(replicates, len(seed_means)), replace=True
    ).mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def paired_effect_summary(
    values: Sequence[float],
    seeds: Sequence[int],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
    direction_tolerance: float = 0.0,
) -> dict[str, float | int]:
    """Summarize a paired action delta without treating frames as IID."""

    array = np.asarray(values, dtype=np.float64)
    seed_array = np.asarray(seeds, dtype=np.int64)
    low, high = seed_cluster_bootstrap_ci(
        array,
        seed_array,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    seed_means = np.asarray(
        [np.mean(array[seed_array == value]) for value in np.unique(seed_array)],
        dtype=np.float64,
    )
    seed_sd = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    standardized = float(np.mean(seed_means) / seed_sd) if seed_sd > 1.0e-12 else None
    return {
        "n": int(len(array)),
        "seed_count": int(len(np.unique(seed_array))),
        "mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(np.abs(array))),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "positive_fraction": float(np.mean(array > direction_tolerance)),
        "negative_fraction": float(np.mean(array < -direction_tolerance)),
        "seed_mean_sd": seed_sd,
        "standardized_seed_mean_effect": standardized,
    }


def intervention_gate_results(
    *,
    pedestrian_delta_v: NDArray[np.floating],
    pedestrian_seeds: NDArray[np.integer],
    pedestrian_irrelevant_delta_v: NDArray[np.floating],
    stop_delta_v: NDArray[np.floating],
    stop_seeds: NDArray[np.integer],
    stop_control_delta_v: NDArray[np.floating],
    lane_delta_omega: NDArray[np.floating],
    lane_seeds: NDArray[np.integer],
    lane_control_delta_omega: NDArray[np.floating],
    sham_maximum_absolute_effect: float,
    gate: dict[str, float | int],
    bootstrap: dict[str, float | int],
) -> tuple[dict[str, bool], dict[str, object]]:
    """Evaluate only the hypotheses frozen before confirmatory execution."""

    common = {
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "confidence_level": float(bootstrap["confidence_level"]),
        "direction_tolerance": float(gate["direction_tolerance"]),
    }
    pedestrian = paired_effect_summary(
        pedestrian_delta_v,
        pedestrian_seeds,
        bootstrap_seed=int(bootstrap["seed"]),
        **common,
    )
    stop = paired_effect_summary(
        stop_delta_v,
        stop_seeds,
        bootstrap_seed=int(bootstrap["seed"]),
        **common,
    )
    lane_absolute = paired_effect_summary(
        np.abs(lane_delta_omega),
        lane_seeds,
        bootstrap_seed=int(bootstrap["seed"]),
        **common,
    )
    lane_control_mean = float(np.mean(np.abs(lane_control_delta_omega)))
    lane_ratio = float(lane_absolute["mean_absolute"]) / max(lane_control_mean, 1.0e-12)
    diagnostics: dict[str, object] = {
        "pedestrian_relevant_delta_v": pedestrian,
        "stop_required_delta_v": stop,
        "lane_curve_absolute_delta_omega": lane_absolute,
        "pedestrian_nonrelevant_mean_absolute_delta_v": float(
            np.mean(np.abs(pedestrian_irrelevant_delta_v))
        ),
        "stop_pedestrian_phase_control_mean_absolute_delta_v": float(
            np.mean(np.abs(stop_control_delta_v))
        ),
        "lane_pedestrian_phase_control_mean_absolute_delta_omega": lane_control_mean,
        "lane_curve_to_control_effect_ratio": lane_ratio,
        "sham_maximum_absolute_effect": float(sham_maximum_absolute_effect),
    }
    criteria = {
        "pedestrian_relevant_velocity_effect": (
            float(pedestrian["mean"])
            >= float(gate["pedestrian_minimum_mean_delta_v_mps"])
            and float(pedestrian["bootstrap_ci_low"]) > 0.0
            and float(pedestrian["positive_fraction"])
            >= float(gate["minimum_directional_consistency"])
        ),
        "pedestrian_nonrelevant_effect_small": float(
            diagnostics["pedestrian_nonrelevant_mean_absolute_delta_v"]
        )
        <= float(gate["pedestrian_nonrelevant_maximum_mean_absolute_delta_v_mps"]),
        "stop_required_velocity_effect": (
            float(stop["mean"]) >= float(gate["stop_minimum_mean_delta_v_mps"])
            and float(stop["bootstrap_ci_low"]) > 0.0
            and float(stop["positive_fraction"])
            >= float(gate["minimum_directional_consistency"])
        ),
        "stop_negative_control_small": float(
            diagnostics["stop_pedestrian_phase_control_mean_absolute_delta_v"]
        )
        <= float(gate["stop_control_maximum_mean_absolute_delta_v_mps"]),
        "lane_curve_steering_effect": (
            float(lane_absolute["mean_absolute"])
            >= float(gate["lane_minimum_mean_absolute_delta_omega_rad_s"])
            and float(lane_absolute["bootstrap_ci_low"])
            >= float(gate["lane_minimum_mean_absolute_delta_omega_rad_s"])
        ),
        "lane_effect_phase_specific": lane_ratio
        >= float(gate["lane_minimum_curve_to_control_effect_ratio"]),
        "sham_exact": float(sham_maximum_absolute_effect)
        <= float(gate["sham_action_absolute_tolerance"]),
    }
    return criteria, diagnostics
