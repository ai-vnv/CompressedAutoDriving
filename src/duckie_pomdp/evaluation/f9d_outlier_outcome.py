"""Pure outcome metrics for F9d-A natural localization-outlier stress.

The support predicate comes from :mod:`f9d_stress`; this module only asks
what Baseline A and frozen Robust B did on those exact frames.  It performs
no rendering, inference, filtering, or parameter selection.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

from duckie_pomdp.evaluation.f9d_stress import (
    contiguous_outlier_events,
    is_localization_outlier_frame,
)


def _error_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "bias": None, "mae": None, "rmse": None, "max_abs": None}
    count = len(values)
    return {
        "count": count,
        "bias": sum(values) / count,
        "mae": sum(abs(value) for value in values) / count,
        "rmse": sqrt(sum(value * value for value in values) / count),
        "max_abs": max(abs(value) for value in values),
    }


def _belief_errors(rows: Sequence[Mapping[str, Any]], variant: str) -> list[float]:
    return [
        float(row[f"{variant}_belief_range_m"]) - float(row["gt_range_m"])
        for row in rows
        if bool(row.get(f"{variant}_belief_initialized"))
        and row.get(f"{variant}_belief_range_m") not in (None, "")
        and row.get("gt_range_m") not in (None, "")
    ]


def _recovery_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    recovery_error_m: float,
) -> dict[str, Any]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode"]), []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["frame"]))

    recoveries: list[dict[str, Any]] = []
    for event in contiguous_outlier_events(rows):
        recovery = None
        for row in by_episode[event.episode]:
            frame = int(row["frame"])
            if frame <= event.end_frame or is_localization_outlier_frame(row):
                continue
            if not bool(row.get(f"{variant}_belief_initialized")):
                continue
            belief_range = row.get(f"{variant}_belief_range_m")
            gt_range = row.get("gt_range_m")
            if belief_range in (None, "") or gt_range in (None, ""):
                continue
            if abs(float(belief_range) - float(gt_range)) <= recovery_error_m:
                recovery = frame - event.end_frame
                break
        recoveries.append(
            {
                "episode": event.episode,
                "seed": event.seed,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "recovery_frames": recovery,
            }
        )

    observed = [item["recovery_frames"] for item in recoveries if item["recovery_frames"] is not None]
    return {
        "definition": (
            "first subsequent non-outlier frame with initialized belief and "
            f"absolute range error <= {recovery_error_m} m"
        ),
        "event_count": len(recoveries),
        "recovered_event_count": len(observed),
        "unrecovered_event_count": len(recoveries) - len(observed),
        "mean_recovery_frames": None if not observed else sum(observed) / len(observed),
        "max_recovery_frames": None if not observed else max(observed),
        "events": recoveries,
    }


def gate_confusion(rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.5) -> dict[str, Any]:
    """F9c-compatible IoU-vs-innovation-gate confusion table."""

    counts = {
        "good_accept": 0,
        "good_reject": 0,
        "outlier_accept": 0,
        "outlier_reject": 0,
    }
    for row in rows:
        iou = row.get("robust_b_associated_iou")
        decision = row.get("robust_b_gate_decision")
        if iou in (None, "") or decision not in {"accepted", "rejected"}:
            continue
        quality = "good" if float(iou) >= threshold else "outlier"
        counts[f"{quality}_{'accept' if decision == 'accepted' else 'reject'}"] += 1

    outlier_total = counts["outlier_accept"] + counts["outlier_reject"]
    good_total = counts["good_accept"] + counts["good_reject"]
    rejected_total = counts["outlier_reject"] + counts["good_reject"]
    return {
        **counts,
        "outlier_rejection_sensitivity": (
            None if not outlier_total else counts["outlier_reject"] / outlier_total
        ),
        "good_measurement_false_rejection_rate": (
            None if not good_total else counts["good_reject"] / good_total
        ),
        "rejection_precision": (
            None if not rejected_total else counts["outlier_reject"] / rejected_total
        ),
    }


def outlier_outcome_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    recovery_error_m: float,
) -> dict[str, Any]:
    outliers = [row for row in rows if is_localization_outlier_frame(row)]
    baseline = _error_summary(_belief_errors(outliers, "baseline_a"))
    robust = _error_summary(_belief_errors(outliers, "robust_b"))
    ratio = None
    if baseline["rmse"] not in (None, 0.0) and robust["rmse"] is not None:
        ratio = float(robust["rmse"]) / float(baseline["rmse"])
    return {
        "outlier_frame_count": len(outliers),
        "baseline_a_range_error": baseline,
        "robust_b_range_error": robust,
        "robust_to_baseline_rmse_ratio": ratio,
        "baseline_a_recovery": _recovery_metrics(
            rows, variant="baseline_a", recovery_error_m=recovery_error_m
        ),
        "robust_b_recovery": _recovery_metrics(
            rows, variant="robust_b", recovery_error_m=recovery_error_m
        ),
        "gate_confusion": gate_confusion(rows),
    }
