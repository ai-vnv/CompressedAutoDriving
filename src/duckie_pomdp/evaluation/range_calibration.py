"""F5b range-semantics comparison and held-out calibration analysis."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import atan2, hypot, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np

from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.domain.privileged import WorldFootprint, WorldPoint, WorldPose
from duckie_pomdp.perception.camera_geometry import world_to_ego
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    wrap_angle,
)


CALIBRATION_EPISODES = (
    "calibration_approach",
    "turn_left",
    "pedestrian_crossing",
)
VALIDATION_EPISODES = ("straight_approach", "turn_right")


@dataclass(frozen=True)
class RangeSemanticsSample:
    episode: str
    frame: int
    step: int
    object_type: str
    pixel_u: float
    pixel_v: float
    projected: GroundPoint
    origin: GroundPoint
    surface: GroundPoint
    fov_region: str
    silhouette_pixel_count: int


@dataclass(frozen=True)
class RangeCalibrationRow:
    split: str
    episode: str
    frame: int
    step: int
    object_type: str
    pixel_u: float
    pixel_v: float
    projected_x: float
    projected_y: float
    gt_origin_x: float
    gt_origin_y: float
    gt_surface_x: float
    gt_surface_y: float
    raw_range: float
    calibrated_range: float
    gt_origin_range: float
    gt_surface_range: float
    raw_error_origin: float
    raw_error_surface: float
    calibrated_error: float
    bearing: float
    gt_bearing: float
    gt_surface_bearing: float
    bearing_error: float
    distance_bin: str
    fov_region: str
    silhouette_pixel_count: int


@dataclass(frozen=True)
class RangeCalibrationResult:
    calibration: LinearRangeCalibration
    rows: tuple[RangeCalibrationRow, ...]
    summary: dict[str, object]


def nearest_surface_point(
    footprint: WorldFootprint,
    ego_pose: WorldPose,
) -> WorldPoint:
    """Return the nearest point on a collision-footprint boundary to ego."""

    query = np.array([ego_pose.x_m, ego_pose.z_m], dtype=float)
    vertices = np.array(
        [[vertex.x_m, vertex.z_m] for vertex in footprint.vertices],
        dtype=float,
    )
    if _point_in_polygon(query, vertices):
        return WorldPoint(x_m=ego_pose.x_m, z_m=ego_pose.z_m)

    closest: np.ndarray | None = None
    closest_distance_sq = float("inf")
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        segment = end - start
        denominator = float(segment @ segment)
        fraction = 0.0 if denominator == 0.0 else float((query - start) @ segment) / denominator
        candidate = start + np.clip(fraction, 0.0, 1.0) * segment
        distance_sq = float((candidate - query) @ (candidate - query))
        if distance_sq < closest_distance_sq:
            closest = candidate
            closest_distance_sq = distance_sq
    if closest is None:  # pragma: no cover - guarded by WorldFootprint validation
        raise RuntimeError("unable to find a footprint boundary point")
    return WorldPoint(x_m=float(closest[0]), z_m=float(closest[1]))


def surface_point_in_ego(
    footprint: WorldFootprint,
    ego_pose: WorldPose,
) -> GroundPoint:
    return world_to_ego(nearest_surface_point(footprint, ego_pose), ego_pose)


def fit_and_evaluate_range_calibration(
    samples: Iterable[RangeSemanticsSample],
    *,
    calibration_episodes: tuple[str, ...] = CALIBRATION_EPISODES,
    validation_episodes: tuple[str, ...] = VALIDATION_EPISODES,
) -> RangeCalibrationResult:
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("range calibration requires samples")
    overlap = set(calibration_episodes) & set(validation_episodes)
    if overlap:
        raise ValueError(f"calibration and validation episodes overlap: {sorted(overlap)}")
    known = set(calibration_episodes) | set(validation_episodes)
    unknown = {sample.episode for sample in materialized} - known
    if unknown:
        raise ValueError(f"samples have unassigned episodes: {sorted(unknown)}")

    fit_samples = [
        sample for sample in materialized if sample.episode in calibration_episodes
    ]
    held_out_samples = [
        sample for sample in materialized if sample.episode in validation_episodes
    ]
    if len(fit_samples) < 2 or not held_out_samples:
        raise ValueError("calibration requires at least two fit samples and held-out data")

    design = np.array(
        [[_range(sample.projected), 1.0] for sample in fit_samples],
        dtype=float,
    )
    targets = np.array([_range(sample.origin) for sample in fit_samples], dtype=float)
    scale, offset = np.linalg.lstsq(design, targets, rcond=None)[0]
    calibration = LinearRangeCalibration(float(scale), float(offset))

    rows = tuple(
        _make_row(
            sample,
            split=(
                "calibration"
                if sample.episode in calibration_episodes
                else "validation"
            ),
            calibration=calibration,
        )
        for sample in materialized
    )
    summary = _build_summary(
        rows,
        calibration,
        calibration_episodes,
        validation_episodes,
    )
    return RangeCalibrationResult(calibration, rows, summary)


def write_rows_csv(rows: Iterable[RangeCalibrationRow], path: str | Path) -> None:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("cannot write empty range-calibration rows")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in materialized]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def residual_metrics(values: Iterable[float]) -> dict[str, float | int | None]:
    errors = np.asarray(tuple(values), dtype=float)
    if len(errors) == 0:
        return {
            "count": 0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "residual_sd": None,
            "median": None,
            "p05": None,
            "p95": None,
            "skewness": None,
            "excess_kurtosis": None,
        }
    mean = float(np.mean(errors))
    population_sd = float(np.std(errors, ddof=0))
    sample_sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    if population_sd > 0.0:
        standardized = (errors - mean) / population_sd
        skewness = float(np.mean(standardized**3))
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0
    return {
        "count": len(errors),
        "bias": mean,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(sqrt(float(np.mean(errors**2)))),
        "residual_sd": sample_sd,
        "median": float(np.median(errors)),
        "p05": float(np.quantile(errors, 0.05)),
        "p95": float(np.quantile(errors, 0.95)),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }


def _make_row(
    sample: RangeSemanticsSample,
    *,
    split: str,
    calibration: LinearRangeCalibration,
) -> RangeCalibrationRow:
    raw_range = _range(sample.projected)
    origin_range = _range(sample.origin)
    surface_range = _range(sample.surface)
    calibrated_range = calibration.apply(raw_range)
    bearing = _bearing(sample.projected)
    origin_bearing = _bearing(sample.origin)
    return RangeCalibrationRow(
        split=split,
        episode=sample.episode,
        frame=sample.frame,
        step=sample.step,
        object_type=sample.object_type,
        pixel_u=sample.pixel_u,
        pixel_v=sample.pixel_v,
        projected_x=sample.projected.x_left_m,
        projected_y=sample.projected.y_forward_m,
        gt_origin_x=sample.origin.x_left_m,
        gt_origin_y=sample.origin.y_forward_m,
        gt_surface_x=sample.surface.x_left_m,
        gt_surface_y=sample.surface.y_forward_m,
        raw_range=raw_range,
        calibrated_range=calibrated_range,
        gt_origin_range=origin_range,
        gt_surface_range=surface_range,
        raw_error_origin=raw_range - origin_range,
        raw_error_surface=raw_range - surface_range,
        calibrated_error=calibrated_range - origin_range,
        bearing=bearing,
        gt_bearing=origin_bearing,
        gt_surface_bearing=_bearing(sample.surface),
        bearing_error=wrap_angle(bearing - origin_bearing),
        distance_bin=_distance_bin(origin_range),
        fov_region=sample.fov_region,
        silhouette_pixel_count=sample.silhouette_pixel_count,
    )


def _build_summary(
    rows: tuple[RangeCalibrationRow, ...],
    calibration: LinearRangeCalibration,
    calibration_episodes: tuple[str, ...],
    validation_episodes: tuple[str, ...],
) -> dict[str, object]:
    fit_rows = tuple(row for row in rows if row.split == "calibration")
    held_out = tuple(row for row in rows if row.split == "validation")
    return {
        "canonical_range_semantics": "object_origin",
        "surface_definition": "nearest simulator collision-footprint boundary",
        "split": {
            "calibration_episodes": list(calibration_episodes),
            "validation_episodes": list(validation_episodes),
            "calibration_count": len(fit_rows),
            "validation_count": len(held_out),
        },
        "calibration": {
            "type": "linear",
            "scale": calibration.scale,
            "offset_m": calibration.offset_m,
        },
        "all_samples": {
            "raw_vs_origin": _group_metrics(rows, "raw_error_origin"),
            "raw_vs_surface": _group_metrics(rows, "raw_error_surface"),
        },
        "calibration_split": {
            "raw_vs_origin": residual_metrics(
                row.raw_error_origin for row in fit_rows
            ),
            "calibrated_vs_origin": residual_metrics(
                row.calibrated_error for row in fit_rows
            ),
        },
        "held_out_validation": {
            "raw_vs_origin": _group_metrics(held_out, "raw_error_origin"),
            "raw_vs_surface": _group_metrics(held_out, "raw_error_surface"),
            "calibrated_vs_origin": _group_metrics(held_out, "calibrated_error"),
            "bearing_vs_origin": _group_metrics(held_out, "bearing_error"),
            "range_bearing_residual_correlation": _correlation(
                [row.calibrated_error for row in held_out],
                [row.bearing_error for row in held_out],
            ),
            "range_bearing_residual_covariance_m_rad": _covariance(
                [row.calibrated_error for row in held_out],
                [row.bearing_error for row in held_out],
            ),
        },
        "measurement_noise": {
            "range_global": residual_metrics(
                row.calibrated_error for row in held_out
            ),
            "range_by_distance": {
                name: {
                    "min_m": lower,
                    "max_m": upper,
                    **residual_metrics(
                        row.calibrated_error
                        for row in held_out
                        if row.distance_bin == name
                    ),
                }
                for name, lower, upper in (
                    ("near", 0.0, 0.55),
                    ("medium", 0.55, 0.80),
                    ("far", 0.80, None),
                )
            },
            "bearing_global": residual_metrics(
                row.bearing_error for row in held_out
            ),
            "bearing_by_fov": {
                name: residual_metrics(
                    row.bearing_error for row in held_out if row.fov_region == name
                )
                for name in ("center", "mid_fov", "edge_fov")
            },
        },
    }


def _group_metrics(
    rows: Iterable[RangeCalibrationRow],
    attribute: str,
) -> dict[str, object]:
    materialized = tuple(rows)
    return {
        "global": residual_metrics(getattr(row, attribute) for row in materialized),
        "by_distance": {
            name: residual_metrics(
                getattr(row, attribute)
                for row in materialized
                if row.distance_bin == name
            )
            for name in ("near", "medium", "far")
        },
        "by_fov": {
            name: residual_metrics(
                getattr(row, attribute)
                for row in materialized
                if row.fov_region == name
            )
            for name in ("center", "mid_fov", "edge_fov")
        },
        "by_object": {
            name: residual_metrics(
                getattr(row, attribute)
                for row in materialized
                if row.object_type == name
            )
            for name in ("sign_stop", "duckie")
        },
    }


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _covariance(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2:
        return None
    return float(np.cov(first, second, ddof=1)[0, 1])


def _range(point: GroundPoint) -> float:
    return hypot(point.x_left_m, point.y_forward_m)


def _bearing(point: GroundPoint) -> float:
    return atan2(point.x_left_m, point.y_forward_m)


def _distance_bin(range_m: float) -> str:
    if range_m < 0.55:
        return "near"
    if range_m < 0.80:
        return "medium"
    return "far"
