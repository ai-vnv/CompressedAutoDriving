"""Pure quantitative summaries for the frozen F9 YOLO-to-EKF experiment."""

from __future__ import annotations

from math import pi
from typing import Iterable

import numpy as np

from duckie_pomdp.perception.measurement_calibration import wrap_angle


def scalar_error_metrics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array):
        return {"count": 0, "bias": None, "mae": None, "rmse": None, "sd": None}
    return {
        "count": int(len(array)),
        "bias": float(np.mean(array)),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def current_measurement_metrics(
    rows: list[dict[str, object]],
    *,
    corrected: bool,
) -> dict[str, object]:
    prefix = "corrected" if corrected else "raw"
    selected = [
        row
        for row in rows
        if bool(row["eligible_visible"])
        and bool(row["measurement_detected"])
        and row[f"{prefix}_measurement_range_m"] is not None
    ]
    range_errors = [
        float(row[f"{prefix}_measurement_range_m"]) - float(row["gt_range_m"])
        for row in selected
    ]
    bearing_errors = [
        wrap_angle(
            float(row[f"{prefix}_measurement_bearing_rad"])
            - float(row["gt_bearing_rad"])
        )
        for row in selected
    ]
    return {
        "eligible_opportunities": sum(bool(row["eligible_visible"]) for row in rows),
        "detected_visible": len(selected),
        "correct_iou50": sum(bool(row["selected_correct_iou50"]) for row in selected),
        "range": scalar_error_metrics(range_errors),
        "bearing": scalar_error_metrics(bearing_errors),
        "rates_available": False,
    }


def belief_metrics(
    rows: list[dict[str, object]],
    *,
    variant: str,
) -> dict[str, object]:
    selected = [row for row in rows if bool(row[f"{variant}_belief_initialized"])]
    fields = {
        "range": ("range_m", "gt_range_m", "range_std_m", True),
        "bearing": ("bearing_rad", "gt_bearing_rad", "bearing_std_rad", True),
        "range_rate": (
            "range_rate_mps",
            "gt_range_rate_mps",
            "range_rate_std_mps",
            False,
        ),
        "bearing_rate": (
            "bearing_rate_rad_s",
            "gt_bearing_rate_rad_s",
            "bearing_rate_std_rad_s",
            False,
        ),
    }
    output: dict[str, object] = {"count": len(selected)}
    for name, (mean_suffix, truth_field, std_suffix, wrap) in fields.items():
        errors = np.asarray(
            [
                (
                    wrap_angle(
                        float(row[f"{variant}_belief_{mean_suffix}"])
                        - float(row[truth_field])
                    )
                    if wrap and name == "bearing"
                    else float(row[f"{variant}_belief_{mean_suffix}"])
                    - float(row[truth_field])
                )
                for row in selected
            ],
            dtype=float,
        )
        sigma = np.asarray(
            [float(row[f"{variant}_belief_{std_suffix}"]) for row in selected],
            dtype=float,
        )
        base = scalar_error_metrics(errors)
        if len(errors):
            safe = np.maximum(sigma, 1.0e-12)
            base.update(
                {
                    "mean_predicted_std": float(np.mean(sigma)),
                    "coverage_68": float(np.mean(np.abs(errors) <= safe)),
                    "coverage_95": float(np.mean(np.abs(errors) <= 1.96 * safe)),
                    "mean_marginal_nll": float(
                        np.mean(
                            0.5
                            * (
                                np.log(2.0 * pi * safe**2)
                                + errors**2 / safe**2
                            )
                        )
                    ),
                }
            )
        output[name] = base
    output["existence"] = _existence_metrics(selected, variant)
    return output


def nis_metrics(
    rows: list[dict[str, object]],
    *,
    variant: str,
    threshold: float,
) -> dict[str, object]:
    values = np.asarray(
        [float(row[f"{variant}_nis"]) for row in rows if row[f"{variant}_nis"] is not None],
        dtype=float,
    )
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "fraction_above_chi_square_2d_95": float(np.mean(values > threshold)),
        "chi_square_2d_95_threshold": threshold,
    }


def track_continuity(
    rows: list[dict[str, object]],
    *,
    variant: str,
    active_probability_threshold: float,
) -> dict[str, object]:
    visible = [row for row in rows if bool(row["eligible_visible"])]
    missed = [row for row in visible if not bool(row["measurement_detected"])]

    def active(row: dict[str, object]) -> bool:
        return bool(row[f"{variant}_belief_initialized"]) and float(
            row[f"{variant}_existence_probability"]
        ) >= active_probability_threshold

    run_lengths: list[int] = []
    recovery_lengths: list[int] = []
    for episode in sorted({str(row["episode"]) for row in rows}):
        episode_rows = sorted(
            (row for row in rows if row["episode"] == episode),
            key=lambda row: int(row["frame"]),
        )
        run = 0
        for row_index, row in enumerate(episode_rows):
            if bool(row["eligible_visible"]) and not bool(row["measurement_detected"]):
                run += 1
                continue
            if run:
                run_lengths.append(run)
                if bool(row["measurement_detected"]):
                    recovery = next(
                        (
                            offset + 1
                            for offset, later in enumerate(
                                episode_rows[row_index:]
                            )
                            if active(later)
                        ),
                        None,
                    )
                    if recovery is not None:
                        recovery_lengths.append(recovery)
                run = 0
        if run:
            run_lengths.append(run)
    return {
        "visible_frames": len(visible),
        "visible_frames_with_active_belief": sum(active(row) for row in visible),
        "visible_active_fraction": (
            None if not visible else sum(active(row) for row in visible) / len(visible)
        ),
        "genuine_missed_frames": len(missed),
        "missed_frames_with_maintained_belief": sum(active(row) for row in missed),
        "missed_maintained_fraction": (
            None if not missed else sum(active(row) for row in missed) / len(missed)
        ),
        "mean_consecutive_miss_length": (
            None if not run_lengths else float(np.mean(run_lengths))
        ),
        "maximum_consecutive_miss_length": max(run_lengths, default=0),
        "mean_frames_to_recover_after_redetection": (
            None if not recovery_lengths else float(np.mean(recovery_lengths))
        ),
    }


def summarize_f9(
    rows: list[dict[str, object]],
    *,
    active_probability_threshold: float,
    nis_threshold: float,
) -> tuple[dict[str, object], dict[str, object]]:
    variants = ("raw", "corrected")
    metrics: dict[str, object] = {
        "row_count": len(rows),
        "current_frame_yolo": {
            variant: current_measurement_metrics(
                rows,
                corrected=variant == "corrected",
            )
            for variant in variants
        },
        "ekf": {variant: belief_metrics(rows, variant=variant) for variant in variants},
        "track_continuity": {
            variant: track_continuity(
                rows,
                variant=variant,
                active_probability_threshold=active_probability_threshold,
            )
            for variant in variants
        },
        "events": {
            "eligible_visible": sum(bool(row["eligible_visible"]) for row in rows),
            "detected_visible": sum(
                bool(row["eligible_visible"]) and bool(row["measurement_detected"])
                for row in rows
            ),
            "natural_misses": sum(
                bool(row["eligible_visible"]) and not bool(row["measurement_detected"])
                for row in rows
            ),
            "outside_domain_frames": sum(not bool(row["eligible_visible"]) for row in rows),
            "false_measurement_events": sum(bool(row["false_measurement_event"]) for row in rows),
            "class_negative_false_alarm_events": sum(
                not bool(row["eligible_visible"])
                and bool(row["measurement_detected"])
                for row in rows
            ),
            "visible_localization_mismatch_updates": sum(
                bool(row["eligible_visible"])
                and bool(row["measurement_detected"])
                and not bool(row["selected_correct_iou50"])
                for row in rows
            ),
            "false_track_initializations": sum(bool(row["false_track_initialization"]) for row in rows),
            "duplicate_selection_events": sum(bool(row["duplicate_selection"]) for row in rows),
        },
        "selection_diagnostics": {
            "duplicate_frames": _selection_quality(
                [row for row in rows if bool(row["duplicate_selection"])]
            ),
            "single_or_missed_frames": _selection_quality(
                [row for row in rows if not bool(row["duplicate_selection"])]
            ),
            "localization_mismatch_updates": _selection_quality(
                [row for row in rows if bool(row["false_measurement_event"])]
            ),
            "iou50_correct_updates": _selection_quality(
                [row for row in rows if bool(row["selected_correct_iou50"])]
            ),
            "confidence_correlation": _confidence_diagnostics(rows),
        },
    }
    for grouping, values in (
        ("scenario", sorted({str(row["scenario"]) for row in rows})),
        ("distance_bin", ("near", "medium", "far")),
        ("fov_region", ("center", "mid_fov", "edge_fov", "outside")),
    ):
        metrics[f"by_{grouping}"] = {
            value: {
                "row_count": len(subset := [row for row in rows if row[grouping] == value]),
                "detection_recall": _detection_recall(subset),
                "current_frame_raw": current_measurement_metrics(subset, corrected=False),
                "current_frame_corrected": current_measurement_metrics(subset, corrected=True),
                "raw_ekf": belief_metrics(subset, variant="raw"),
                "corrected_ekf": belief_metrics(subset, variant="corrected"),
            }
            for value in values
        }
    nis = {
        "global": {
            variant: nis_metrics(rows, variant=variant, threshold=nis_threshold)
            for variant in variants
        },
        "by_distance": {
            distance: {
                variant: nis_metrics(
                    [row for row in rows if row["distance_bin"] == distance],
                    variant=variant,
                    threshold=nis_threshold,
                )
                for variant in variants
            }
            for distance in ("near", "medium", "far")
        },
    }
    return metrics, nis


def _detection_recall(rows: list[dict[str, object]]) -> float | None:
    eligible = [row for row in rows if bool(row["eligible_visible"])]
    if not eligible:
        return None
    return sum(bool(row["measurement_detected"]) for row in eligible) / len(eligible)


def _existence_metrics(
    rows: list[dict[str, object]],
    variant: str,
) -> dict[str, float | int | None]:
    if not rows:
        return {"count": 0}
    visible = [row for row in rows if bool(row["eligible_visible"])]
    misses = [row for row in visible if not bool(row["measurement_detected"])]
    outside = [row for row in rows if not bool(row["eligible_visible"])]
    values = [float(row[f"{variant}_existence_probability"]) for row in rows]
    return {
        "count": len(rows),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "mean_when_visible": (
            None
            if not visible
            else float(np.mean([float(row[f"{variant}_existence_probability"]) for row in visible]))
        ),
        "minimum_during_genuine_miss": (
            None
            if not misses
            else float(np.min([float(row[f"{variant}_existence_probability"]) for row in misses]))
        ),
        "mean_outside_observation_domain": (
            None
            if not outside
            else float(np.mean([float(row[f"{variant}_existence_probability"]) for row in outside]))
        ),
    }


def _selection_quality(rows: list[dict[str, object]]) -> dict[str, object]:
    measurement_errors = [
        float(row["corrected_measurement_range_m"]) - float(row["gt_range_m"])
        for row in rows
        if row["corrected_measurement_range_m"] is not None
    ]
    belief_errors = [
        float(row["corrected_belief_range_m"]) - float(row["gt_range_m"])
        for row in rows
        if bool(row["corrected_belief_initialized"])
    ]
    nis = np.asarray(
        [float(row["corrected_nis"]) for row in rows if row["corrected_nis"] is not None],
        dtype=float,
    )
    return {
        "frame_count": len(rows),
        "measurement_range": scalar_error_metrics(measurement_errors),
        "belief_range": scalar_error_metrics(belief_errors),
        "nis_p95": None if not len(nis) else float(np.quantile(nis, 0.95)),
    }


def _confidence_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if bool(row["eligible_visible"])
        and bool(row["measurement_detected"])
        and row["selected_confidence"] is not None
    ]
    confidence = np.asarray(
        [float(row["selected_confidence"]) for row in selected], dtype=float
    )
    range_error = np.asarray(
        [abs(float(row["corrected_measurement_range_error_m"])) for row in selected],
        dtype=float,
    )
    bearing_error = np.asarray(
        [abs(float(row["corrected_measurement_bearing_error_rad"])) for row in selected],
        dtype=float,
    )
    with_nis = [row for row in selected if row["corrected_nis"] is not None]
    return {
        "sample_count": len(selected),
        "pearson_vs_absolute_range_error": _correlation(confidence, range_error),
        "pearson_vs_absolute_bearing_error": _correlation(
            confidence, bearing_error
        ),
        "pearson_vs_nis": _correlation(
            np.asarray([float(row["selected_confidence"]) for row in with_nis]),
            np.asarray([float(row["corrected_nis"]) for row in with_nis]),
        ),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
