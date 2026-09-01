"""F8b residual statistics and candidate YOLO observation-noise analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from duckie_pomdp.evaluation.range_calibration import residual_metrics
from duckie_pomdp.evaluation.yolo_detection import pearson_correlation


@dataclass(frozen=True)
class MeasurementResidual:
    image_id: str
    object_class: str
    distance_bin: str
    fov_region: str
    confidence: float
    raw_range_error_m: float
    calibrated_range_error_m: float
    bearing_error_rad: float


def summarize_measurements(
    residuals: Iterable[MeasurementResidual],
) -> dict[str, object]:
    rows = tuple(residuals)
    if not rows:
        raise ValueError("measurement summary requires matched detections")
    global_summary = _metrics(rows)
    return {
        "global": global_summary,
        "by_class": _grouped(rows, "object_class"),
        "by_distance": _grouped(rows, "distance_bin"),
        "by_fov": _grouped(rows, "fov_region"),
        "by_class_and_distance": _nested_group(rows, "object_class", "distance_bin"),
        "by_class_and_fov": _nested_group(rows, "object_class", "fov_region"),
        "confidence_analysis": {
            "pearson_vs_absolute_raw_range_error": pearson_correlation(
                (row.confidence for row in rows),
                (abs(row.raw_range_error_m) for row in rows),
            ),
            "pearson_vs_absolute_calibrated_range_error": pearson_correlation(
                (row.confidence for row in rows),
                (abs(row.calibrated_range_error_m) for row in rows),
            ),
            "pearson_vs_absolute_bearing_error": pearson_correlation(
                (row.confidence for row in rows),
                (abs(row.bearing_error_rad) for row in rows),
            ),
            "bins": _confidence_bins(rows),
        },
    }


def gaussian_assessment(metrics: dict[str, float | int | None]) -> str:
    count = int(metrics["count"] or 0)
    skewness = metrics["skewness"]
    kurtosis = metrics["excess_kurtosis"]
    if count < 20 or skewness is None or kurtosis is None:
        return "insufficient_samples"
    if abs(float(skewness)) <= 1.0 and abs(float(kurtosis)) <= 2.0:
        return "reasonable_gaussian_approximation"
    if abs(float(skewness)) <= 2.0 and abs(float(kurtosis)) <= 7.0:
        return "provisional_gaussian_approximation"
    return "poor_gaussian_approximation"


def residual_correlation(
    residuals: Sequence[MeasurementResidual],
) -> dict[str, float | int | None]:
    if len(residuals) < 2:
        return {"count": len(residuals), "correlation": None, "covariance_m_rad": None}
    range_errors = np.asarray(
        [row.calibrated_range_error_m for row in residuals], dtype=float
    )
    bearing_errors = np.asarray([row.bearing_error_rad for row in residuals], dtype=float)
    correlation = pearson_correlation(range_errors, bearing_errors)
    covariance = float(np.cov(range_errors, bearing_errors, ddof=1)[0, 1])
    return {
        "count": len(residuals),
        "correlation": correlation,
        "covariance_m_rad": covariance,
    }


def _metrics(rows: Sequence[MeasurementResidual]) -> dict[str, object]:
    raw = residual_metrics(row.raw_range_error_m for row in rows)
    calibrated = residual_metrics(row.calibrated_range_error_m for row in rows)
    bearing = residual_metrics(row.bearing_error_rad for row in rows)
    return {
        "count": len(rows),
        "raw_range": raw,
        "f5b_calibrated_range": calibrated,
        "bearing": bearing,
        "gaussian_assessment": {
            "raw_range": gaussian_assessment(raw),
            "f5b_calibrated_range": gaussian_assessment(calibrated),
            "bearing": gaussian_assessment(bearing),
        },
        "range_bearing_residual": residual_correlation(rows),
    }


def _grouped(
    rows: Sequence[MeasurementResidual], attribute: str
) -> dict[str, object]:
    return {
        value: _metrics(tuple(row for row in rows if getattr(row, attribute) == value))
        for value in sorted({getattr(row, attribute) for row in rows})
    }


def _nested_group(
    rows: Sequence[MeasurementResidual], first: str, second: str
) -> dict[str, object]:
    return {
        first_value: _grouped(
            tuple(row for row in rows if getattr(row, first) == first_value), second
        )
        for first_value in sorted({getattr(row, first) for row in rows})
    }


def _confidence_bins(rows: Sequence[MeasurementResidual]) -> dict[str, object]:
    bins = ((0.10, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.0000001))
    output: dict[str, object] = {}
    for lower, upper in bins:
        selected = tuple(row for row in rows if lower <= row.confidence < upper)
        label = f"{lower:.2f}_{min(upper, 1.0):.2f}"
        output[label] = _metrics(selected) if selected else {"count": 0}
    return output

