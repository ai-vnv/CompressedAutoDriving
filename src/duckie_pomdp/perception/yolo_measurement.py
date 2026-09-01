"""Convert detector boxes to calibrated metric observations without GT input."""

from __future__ import annotations

from dataclasses import dataclass

from duckie_pomdp.domain.detection import Detection, ImagePoint
from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    PolarGroundPoint,
    ground_to_polar,
)
from duckie_pomdp.perception.measurement_calibration import MeasurementCalibrator


@dataclass(frozen=True)
class RawProjectedYoloMeasurement:
    detector_bottom_center: ImagePoint
    projection_pixel: ImagePoint
    pixel_clipped_to_image: bool
    ground_point: GroundPoint
    raw_polar: PolarGroundPoint


@dataclass(frozen=True)
class ProjectedYoloMeasurement(RawProjectedYoloMeasurement):
    calibrated_polar: PolarGroundPoint


class YoloMeasurementProjector:
    """Runtime path: Detection -> bottom-center -> ground -> calibrated polar."""

    def __init__(
        self,
        ground_projector: CalibratedGroundProjector,
        calibrator: MeasurementCalibrator,
    ) -> None:
        self._ground_projector = ground_projector
        self._calibrator = calibrator

    def project(self, detection: Detection) -> ProjectedYoloMeasurement:
        raw = self.project_raw(detection)
        calibrated = self._calibrator.apply(
            raw.raw_polar.range_m, raw.raw_polar.bearing_rad
        )
        return ProjectedYoloMeasurement(
            detector_bottom_center=raw.detector_bottom_center,
            projection_pixel=raw.projection_pixel,
            pixel_clipped_to_image=raw.pixel_clipped_to_image,
            ground_point=raw.ground_point,
            raw_polar=raw.raw_polar,
            calibrated_polar=calibrated,
        )

    def project_raw(self, detection: Detection) -> RawProjectedYoloMeasurement:
        """Project a box without applying the F5b point-localization fit."""

        raw_pixel = detection.bottom_center
        calibration = self._ground_projector.calibration
        projection_pixel = ImagePoint(
            x_px=min(max(raw_pixel.x_px, 0.0), calibration.image_width_px - 1.0),
            y_px=min(max(raw_pixel.y_px, 0.0), calibration.image_height_px - 1.0),
        )
        ground = self._ground_projector.pixel_to_ground(projection_pixel)
        raw_polar = ground_to_polar(ground)
        return RawProjectedYoloMeasurement(
            detector_bottom_center=raw_pixel,
            projection_pixel=projection_pixel,
            pixel_clipped_to_image=projection_pixel != raw_pixel,
            ground_point=ground,
            raw_polar=raw_polar,
        )
