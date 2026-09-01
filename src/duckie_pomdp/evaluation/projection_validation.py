"""F5 projection error rows and aggregate metric calculation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import atan2, cos, sqrt, sin
from pathlib import Path

from duckie_pomdp.domain.detection import ImagePoint
from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    ground_to_polar,
)


@dataclass(frozen=True)
class ProjectionValidationRow:
    frame: int
    episode: str
    step: int
    object_type: str
    pixel_u: float
    pixel_v: float
    gt_x: float
    gt_y: float
    gt_r: float
    gt_beta: float
    pred_x: float
    pred_y: float
    pred_r: float
    pred_beta: float
    error_x: float
    error_y: float
    error_r: float
    error_beta: float
    distance_bin: str
    fov_region: str
    silhouette_pixel_count: int


class ProjectionValidationLogger:
    def __init__(self) -> None:
        self.rows: list[ProjectionValidationRow] = []

    def record(
        self,
        *,
        frame: int,
        episode: str,
        step: int,
        object_type: str,
        pixel: ImagePoint,
        silhouette_pixel_count: int,
        ground_truth: GroundPoint,
        projector: CalibratedGroundProjector,
    ) -> ProjectionValidationRow:
        prediction = projector.pixel_to_ground(pixel)
        truth_polar = ground_to_polar(ground_truth)
        predicted_polar = ground_to_polar(prediction)
        normalized_offset = abs(
            pixel.x_px - 0.5 * projector.calibration.image_width_px
        ) / (0.5 * projector.calibration.image_width_px)
        row = ProjectionValidationRow(
            frame=frame,
            episode=episode,
            step=step,
            object_type=object_type,
            pixel_u=pixel.x_px,
            pixel_v=pixel.y_px,
            gt_x=ground_truth.x_left_m,
            gt_y=ground_truth.y_forward_m,
            gt_r=truth_polar.range_m,
            gt_beta=truth_polar.bearing_rad,
            pred_x=prediction.x_left_m,
            pred_y=prediction.y_forward_m,
            pred_r=predicted_polar.range_m,
            pred_beta=predicted_polar.bearing_rad,
            error_x=prediction.x_left_m - ground_truth.x_left_m,
            error_y=prediction.y_forward_m - ground_truth.y_forward_m,
            error_r=predicted_polar.range_m - truth_polar.range_m,
            error_beta=_wrap_angle(
                predicted_polar.bearing_rad - truth_polar.bearing_rad
            ),
            distance_bin=_distance_bin(truth_polar.range_m),
            fov_region=_fov_region(normalized_offset),
            silhouette_pixel_count=silhouette_pixel_count,
        )
        self.rows.append(row)
        return row

    def write_csv(self, output_path: str | Path) -> None:
        if not self.rows:
            raise ValueError("cannot write an empty projection-validation artifact")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(row) for row in self.rows]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def metrics(self) -> dict[str, object]:
        return {
            "global": _metrics(self.rows),
            "by_distance": {
                name: _metrics([row for row in self.rows if row.distance_bin == name])
                for name in ("near", "medium", "far")
            },
            "by_fov": {
                name: _metrics([row for row in self.rows if row.fov_region == name])
                for name in ("center", "mid_fov", "edge_fov")
            },
            "by_object": {
                name: _metrics([row for row in self.rows if row.object_type == name])
                for name in ("sign_stop", "duckie")
            },
        }


def _distance_bin(range_m: float) -> str:
    if range_m < 0.55:
        return "near"
    if range_m < 0.80:
        return "medium"
    return "far"


def _fov_region(normalized_offset: float) -> str:
    if normalized_offset < 1.0 / 3.0:
        return "center"
    if normalized_offset < 2.0 / 3.0:
        return "mid_fov"
    return "edge_fov"


def _metrics(rows: list[ProjectionValidationRow]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "count": 0,
            "mae_x": None,
            "rmse_x": None,
            "mae_y": None,
            "rmse_y": None,
            "mae_r": None,
            "rmse_r": None,
            "mae_beta": None,
            "rmse_beta": None,
        }
    result: dict[str, float | int | None] = {"count": len(rows)}
    for name in ("x", "y", "r", "beta"):
        errors = [getattr(row, f"error_{name}") for row in rows]
        result[f"mae_{name}"] = sum(abs(value) for value in errors) / len(errors)
        result[f"rmse_{name}"] = sqrt(
            sum(value * value for value in errors) / len(errors)
        )
    return result


def _wrap_angle(angle_rad: float) -> float:
    return atan2(sin(angle_rad), cos(angle_rad))
