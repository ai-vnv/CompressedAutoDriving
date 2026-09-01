"""Pure calibration-only statistics for the F9 Duckie measurement model."""

from __future__ import annotations

import numpy as np

from duckie_pomdp.evaluation.f9_protocol import F9Protocol, sha256
from duckie_pomdp.evaluation.range_calibration import residual_metrics


def _beta_mean(successes: int, trials: int, alpha: float, beta: float) -> float:
    return (successes + alpha) / (trials + alpha + beta)


def measurement_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    range_errors = np.asarray([float(row["range_error_m"]) for row in rows])
    bearing_errors = np.asarray([float(row["bearing_error_rad"]) for row in rows])
    return {
        "sample_count": len(rows),
        "range": residual_metrics(range_errors),
        "bearing": residual_metrics(bearing_errors),
        "range_bearing_correlation": (
            None
            if len(rows) < 2
            else float(np.corrcoef(range_errors, bearing_errors)[0, 1])
        ),
        "range_bearing_covariance_m_rad": (
            None
            if len(rows) < 2
            else float(np.cov(range_errors, bearing_errors, ddof=1)[0, 1])
        ),
    }


def summarize_calibration(
    rows: list[dict[str, object]],
    protocol: F9Protocol,
) -> dict[str, object]:
    matched = [row for row in rows if row["selected_correct_iou50"]]
    if len(matched) < 2:
        raise RuntimeError("F9 calibration produced insufficient correct measurements")
    global_metrics = measurement_summary(matched)
    range_bias = float(global_metrics["range"]["bias"])
    bearing_bias = float(global_metrics["bearing"]["bias"])
    corrected_range_errors = np.asarray(
        [float(row["range_error_m"]) - range_bias for row in matched]
    )
    corrected_bearing_errors = np.asarray(
        [float(row["bearing_error_rad"]) - bearing_bias for row in matched]
    )
    corrected = {
        "range": residual_metrics(corrected_range_errors),
        "bearing": residual_metrics(corrected_bearing_errors),
    }

    by_distance: dict[str, object] = {}
    bin_sigmas: dict[str, float] = {}
    sufficiently_supported = True
    for name in ("near", "medium", "far"):
        selected = [row for row in matched if row["distance_bin"] == name]
        metrics = measurement_summary(selected) if selected else {"sample_count": 0}
        by_distance[name] = metrics
        if len(selected) < protocol.minimum_samples_per_range_bin:
            sufficiently_supported = False
        else:
            bin_sigmas[name] = float(metrics["range"]["residual_sd"])
    global_sigma = float(global_metrics["range"]["residual_sd"])
    sigma_ratio = (
        max(bin_sigmas.values()) / min(bin_sigmas.values())
        if len(bin_sigmas) == 3 and min(bin_sigmas.values()) > 0.0
        else None
    )
    use_distance_bins = bool(
        sufficiently_supported
        and sigma_ratio is not None
        and sigma_ratio >= protocol.heteroscedastic_sigma_ratio_threshold
    )
    selected_sigmas = {
        name: (
            float(by_distance[name]["range"]["residual_sd"])
            if use_distance_bins
            else global_sigma
        )
        for name in ("near", "medium", "far")
    }

    eligible = [row for row in rows if row["eligible_visible"]]
    correct = [row for row in eligible if row["selected_correct_iou50"]]
    counterfactual_negative = [
        row
        for row in rows
        if row.get("opportunity_kind") == "counterfactual_duckie_hidden"
    ]
    negative = (
        counterfactual_negative
        if counterfactual_negative
        else [row for row in rows if not row["eligible_visible"]]
    )
    false_alarm_events = [
        row for row in negative if row["metric_measurement_present"]
    ]
    alpha = protocol.detection_beta_prior_alpha
    beta = protocol.detection_beta_prior_beta
    detection_probability = _beta_mean(len(correct), len(eligible), alpha, beta)
    false_alarm_probability = _beta_mean(
        len(false_alarm_events), len(negative), alpha, beta
    )

    return {
        "schema_version": 1,
        "gate": "F9a",
        "status": "calibration_only_candidate_not_yet_frozen",
        "source": {
            "config_path": str(protocol.config_path),
            "prefreeze_config_sha256": protocol.config_sha256,
            "checkpoint_path": str(protocol.checkpoint_path),
            "checkpoint_sha256": protocol.checkpoint_sha256,
            "frozen_f7_config_sha256": sha256(protocol.frozen_f7_config_path),
            "calibration_seeds": list(protocol.calibration_seeds),
            "final_evaluation_seeds_reserved_not_read": list(
                protocol.final_evaluation_seeds
            ),
            "runtime_input": "front_rgb_only",
            "ground_truth_use": "offline_visibility_matching_and_statistics_only",
        },
        "opportunities": {
            "total_frames": len(rows),
            "eligible_visible": len(eligible),
            "correct_selected_measurements": len(correct),
            "negative_frames": len(negative),
            "natural_ineligible_frames": sum(
                not bool(row["eligible_visible"])
                and row.get("opportunity_kind") != "counterfactual_duckie_hidden"
                for row in rows
            ),
            "false_alarm_events": len(false_alarm_events),
            "duplicate_selection_events": sum(
                bool(row["duplicate_selection"]) for row in rows
            ),
            "projection_failures": sum(
                bool(row["projection_error"]) for row in rows
            ),
        },
        "raw_measurement": {
            "global": global_metrics,
            "by_distance": by_distance,
            "by_fov": {
                name: measurement_summary(
                    [row for row in matched if row["fov_region"] == name]
                )
                for name in ("center", "mid_fov", "edge_fov")
                if any(row["fov_region"] == name for row in matched)
            },
        },
        "bias_correction": {
            "type": "global_additive",
            "range_bias_m": range_bias,
            "bearing_bias_rad": bearing_bias,
            "corrected_calibration_residual": corrected,
        },
        "covariance": {
            "structure": "diagonal",
            "range_model": "distance_bins" if use_distance_bins else "global",
            "minimum_samples_per_bin": protocol.minimum_samples_per_range_bin,
            "heteroscedastic_sigma_ratio_threshold": (
                protocol.heteroscedastic_sigma_ratio_threshold
            ),
            "observed_sigma_ratio": sigma_ratio,
            "range_sigma_m": selected_sigmas,
            "bearing_sigma_rad": float(global_metrics["bearing"]["residual_sd"]),
            "range_bearing_correlation": global_metrics[
                "range_bearing_correlation"
            ],
            "off_diagonal_enabled": False,
        },
        "existence_observation": {
            "estimator": "Beta posterior mean",
            "false_alarm_opportunity_definition": (
                "same simulator RGB scene rendered with Duckie hidden; "
                "detector input remains RGB only"
                if counterfactual_negative
                else "naturally class-negative frame"
            ),
            "prior_alpha": alpha,
            "prior_beta": beta,
            "detection_successes": len(correct),
            "detection_trials": len(eligible),
            "detection_probability": detection_probability,
            "false_alarm_events": len(false_alarm_events),
            "false_alarm_trials": len(negative),
            "false_alarm_probability": false_alarm_probability,
        },
        "recommended_config": {
            "range_bias_m": range_bias,
            "bearing_bias_rad": bearing_bias,
            "range_sigma_m": selected_sigmas,
            "bearing_sigma_rad": float(global_metrics["bearing"]["residual_sd"]),
            "detection_probability": detection_probability,
            "false_positive_probability": false_alarm_probability,
        },
    }
