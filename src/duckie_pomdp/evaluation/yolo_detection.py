"""Deterministic one-GT-per-class detector matching and F8a metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

import numpy as np

from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass


@dataclass(frozen=True)
class DetectionOpportunity:
    image_id: str
    episode_id: str
    seed: int
    frame_index: int
    object_class: ObjectClass
    eligible_visible: bool
    ground_truth_box: BoundingBox | None
    ground_truth_range_m: float
    ground_truth_bearing_rad: float
    distance_bin: str
    fov_region: str

    def __post_init__(self) -> None:
        if self.eligible_visible != (self.ground_truth_box is not None):
            raise ValueError("eligible opportunities require exactly one GT box")


@dataclass(frozen=True)
class EvaluatedOpportunity:
    opportunity: DetectionOpportunity
    matched_detection: Detection | None
    match_iou: float | None
    unmatched_detections: tuple[Detection, ...]
    unmatched_ious: tuple[float | None, ...]

    @property
    def true_positive(self) -> bool:
        return self.matched_detection is not None

    @property
    def false_negative(self) -> bool:
        return self.opportunity.eligible_visible and not self.true_positive


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(
        0.0, min(first.x_max_px, second.x_max_px) - max(first.x_min_px, second.x_min_px)
    )
    intersection_height = max(
        0.0, min(first.y_max_px, second.y_max_px) - max(first.y_min_px, second.y_min_px)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x_max_px - first.x_min_px) * max(
        0.0, first.y_max_px - first.y_min_px
    )
    second_area = max(0.0, second.x_max_px - second.x_min_px) * max(
        0.0, second.y_max_px - second.y_min_px
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def evaluate_opportunity(
    opportunity: DetectionOpportunity,
    predictions: Sequence[Detection],
    *,
    iou_threshold: float,
) -> EvaluatedOpportunity:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be within [0, 1]")
    class_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.object_class is opportunity.object_class
    )
    if opportunity.ground_truth_box is None:
        return EvaluatedOpportunity(
            opportunity,
            matched_detection=None,
            match_iou=None,
            unmatched_detections=class_predictions,
            unmatched_ious=tuple(None for _ in class_predictions),
        )

    ious = tuple(
        intersection_over_union(opportunity.ground_truth_box, prediction.bounding_box)
        for prediction in class_predictions
    )
    if not class_predictions:
        return EvaluatedOpportunity(opportunity, None, None, (), ())
    best_index = max(
        range(len(class_predictions)),
        key=lambda index: (ious[index], class_predictions[index].confidence, -index),
    )
    matched_index = best_index if ious[best_index] >= iou_threshold else None
    return EvaluatedOpportunity(
        opportunity,
        matched_detection=(
            class_predictions[matched_index] if matched_index is not None else None
        ),
        match_iou=ious[matched_index] if matched_index is not None else None,
        unmatched_detections=tuple(
            prediction
            for index, prediction in enumerate(class_predictions)
            if index != matched_index
        ),
        unmatched_ious=tuple(
            value for index, value in enumerate(ious) if index != matched_index
        ),
    )


def summarize_detection(
    evaluations: Iterable[EvaluatedOpportunity],
) -> dict[str, object]:
    rows = tuple(evaluations)
    if not rows:
        raise ValueError("detector summary requires opportunities")
    classes = {row.opportunity.object_class for row in rows}
    if len(classes) != 1:
        raise ValueError("summarize_detection expects one object class")
    eligible = tuple(row for row in rows if row.opportunity.eligible_visible)
    matched = tuple(row for row in eligible if row.true_positive)
    true_positives = len(matched)
    false_negatives = len(eligible) - true_positives
    false_positives = sum(len(row.unmatched_detections) for row in rows)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    negative = tuple(row for row in rows if not row.opportunity.eligible_visible)
    negative_fp_count = sum(len(row.unmatched_detections) for row in negative)
    negative_frames_with_fp = sum(bool(row.unmatched_detections) for row in negative)

    tp_confidences = [row.matched_detection.confidence for row in matched]
    fp_confidences = [
        detection.confidence
        for row in rows
        for detection in row.unmatched_detections
    ]
    ious = [float(row.match_iou) for row in matched]
    bottom_errors = [_bottom_error_px(row) for row in matched]
    localization = _localization_metrics(matched)
    return {
        "opportunities": len(eligible),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_negative_frames": len(negative),
        "false_positives_on_negative_frames": negative_fp_count,
        "negative_frames_with_false_detection": negative_frames_with_fp,
        "false_positives_per_negative_frame": _ratio(negative_fp_count, len(negative)),
        "false_alarm_event_probability": _ratio(negative_frames_with_fp, len(negative)),
        "false_positives_per_total_frame": _ratio(false_positives, len(rows)),
        "localization": localization,
        "confidence": {
            "true_positive": descriptive_statistics(tp_confidences),
            "false_positive": descriptive_statistics(fp_confidences),
            "pearson_vs_iou": pearson_correlation(tp_confidences, ious),
            "pearson_vs_absolute_bottom_error_px": pearson_correlation(
                tp_confidences, bottom_errors
            ),
        },
        "by_distance": _stratified(eligible, "distance_bin"),
        "by_fov": _stratified(eligible, "fov_region"),
    }


def descriptive_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def pearson_correlation(
    first: Iterable[float], second: Iterable[float]
) -> float | None:
    x = np.asarray(tuple(first), dtype=float)
    y = np.asarray(tuple(second), dtype=float)
    if len(x) != len(y):
        raise ValueError("correlation inputs must have equal length")
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _stratified(
    rows: Sequence[EvaluatedOpportunity], attribute: str
) -> dict[str, dict[str, object]]:
    values = sorted({getattr(row.opportunity, attribute) for row in rows})
    output: dict[str, dict[str, object]] = {}
    for value in values:
        selected = tuple(
            row for row in rows if getattr(row.opportunity, attribute) == value
        )
        matched = tuple(row for row in selected if row.true_positive)
        confidences = [row.matched_detection.confidence for row in matched]
        ious = [float(row.match_iou) for row in matched]
        output[value] = {
            "opportunities": len(selected),
            "true_positives": len(matched),
            "false_negatives": len(selected) - len(matched),
            "recall": _ratio(len(matched), len(selected)),
            "confidence_given_true_positive": descriptive_statistics(confidences),
            "iou_given_true_positive": descriptive_statistics(ious),
        }
    return output


def _localization_metrics(
    matched: Sequence[EvaluatedOpportunity],
) -> dict[str, object]:
    values: dict[str, list[float]] = {
        "iou": [],
        "center_x_error_px": [],
        "center_y_error_px": [],
        "bottom_u_error_px": [],
        "bottom_v_error_px": [],
        "bottom_center_error_px": [],
        "width_error_px": [],
        "height_error_px": [],
    }
    for row in matched:
        gt = row.opportunity.ground_truth_box
        prediction = row.matched_detection.bounding_box
        if gt is None:  # pragma: no cover - matched implies eligible
            raise RuntimeError("matched row lacks GT")
        gt_center_x = 0.5 * (gt.x_min_px + gt.x_max_px)
        gt_center_y = 0.5 * (gt.y_min_px + gt.y_max_px)
        pred_center_x = 0.5 * (prediction.x_min_px + prediction.x_max_px)
        pred_center_y = 0.5 * (prediction.y_min_px + prediction.y_max_px)
        bottom_u_error = prediction.bottom_center.x_px - gt.bottom_center.x_px
        bottom_v_error = prediction.bottom_center.y_px - gt.bottom_center.y_px
        values["iou"].append(float(row.match_iou))
        values["center_x_error_px"].append(pred_center_x - gt_center_x)
        values["center_y_error_px"].append(pred_center_y - gt_center_y)
        values["bottom_u_error_px"].append(bottom_u_error)
        values["bottom_v_error_px"].append(bottom_v_error)
        values["bottom_center_error_px"].append(hypot(bottom_u_error, bottom_v_error))
        values["width_error_px"].append(
            (prediction.x_max_px - prediction.x_min_px) - (gt.x_max_px - gt.x_min_px)
        )
        values["height_error_px"].append(
            (prediction.y_max_px - prediction.y_min_px) - (gt.y_max_px - gt.y_min_px)
        )
    return {name: descriptive_statistics(series) for name, series in values.items()}


def _bottom_error_px(row: EvaluatedOpportunity) -> float:
    gt = row.opportunity.ground_truth_box
    prediction = row.matched_detection
    if gt is None or prediction is None:  # pragma: no cover - caller uses matches
        raise RuntimeError("bottom error requires a matched row")
    return hypot(
        prediction.bottom_center.x_px - gt.bottom_center.x_px,
        prediction.bottom_center.y_px - gt.bottom_center.y_px,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator

