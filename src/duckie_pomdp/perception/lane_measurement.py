"""Front-RGB lane-marking extraction in calibrated ego-ground coordinates.

This module is deliberately independent of Gym-Duckietown internals.  Its
runtime inputs are an RGB image and the frozen camera projector only.  World
pose and simulator lane geometry are evaluation concerns and cannot enter the
estimator API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, isfinite
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.detection import ImagePoint
from duckie_pomdp.domain.measurement import LaneMeasurement

from .camera_geometry import CalibratedGroundProjector
from .measurement_calibration import wrap_angle


@dataclass(frozen=True)
class LanePerceptionConfig:
    """Conservative Version-1 colour/geometry rules for Duckietown markings."""

    lane_half_width_m: float = 0.117
    roi_top_fraction: float = 0.38
    row_stride_px: int = 2
    minimum_run_width_px: int = 2
    minimum_point_count: int = 8
    minimum_forward_m: float = 0.10
    maximum_forward_m: float = 0.75
    pose_minimum_forward_m: float = 0.12
    pose_maximum_forward_m: float = 0.40
    maximum_abs_lateral_m: float = 0.60
    minimum_forward_span_m: float = 0.08
    boundary_normal_fit_radius_m: float = 0.10
    boundary_start_tolerance_m: float = 0.20
    yellow_min_red: int = 115
    yellow_min_green: int = 65
    yellow_red_blue_margin: int = 90
    yellow_green_blue_margin: int = 55
    white_min_intensity: int = 150
    white_max_channel_spread: int = 80
    adaptive_marking_min_value: int = 80
    adaptive_marking_dark_percentile: float = 80.0
    adaptive_marking_bright_percentile: float = 95.0
    robust_sigma_threshold: float = 2.75
    lateral_std_floor_m: float = 0.008
    heading_std_floor_rad: float = 0.015
    curvature_std_floor_inv_m: float = 0.12
    dual_boundary_fusion_enabled: bool = False
    maximum_boundary_disagreement_m: float = 0.060
    single_boundary_std_multiplier: float = 1.50
    adaptive_boundary_std_multiplier: float = 2.00

    def __post_init__(self) -> None:
        positive = (
            self.lane_half_width_m,
            self.row_stride_px,
            self.minimum_run_width_px,
            self.minimum_point_count,
            self.minimum_forward_m,
            self.maximum_forward_m,
            self.pose_minimum_forward_m,
            self.pose_maximum_forward_m,
            self.maximum_abs_lateral_m,
            self.minimum_forward_span_m,
            self.boundary_normal_fit_radius_m,
            self.boundary_start_tolerance_m,
            self.robust_sigma_threshold,
            self.lateral_std_floor_m,
            self.heading_std_floor_rad,
            self.curvature_std_floor_inv_m,
            self.maximum_boundary_disagreement_m,
            self.single_boundary_std_multiplier,
            self.adaptive_boundary_std_multiplier,
        )
        if not all(isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise ValueError("lane perception positive parameters must be finite")
        if not 0.0 <= self.roi_top_fraction < 1.0:
            raise ValueError("ROI top fraction must be within [0, 1)")
        if self.minimum_forward_m >= self.maximum_forward_m:
            raise ValueError("minimum forward range must be below maximum")
        if not (
            self.minimum_forward_m
            <= self.pose_minimum_forward_m
            < self.pose_maximum_forward_m
            <= self.maximum_forward_m
        ):
            raise ValueError("pose fit range must lie inside the projection range")
        for threshold in (
            self.yellow_min_red,
            self.yellow_min_green,
            self.yellow_red_blue_margin,
            self.yellow_green_blue_margin,
            self.white_min_intensity,
            self.white_max_channel_spread,
            self.adaptive_marking_min_value,
        ):
            if not 0 <= threshold <= 255:
                raise ValueError("RGB thresholds must be within [0, 255]")
        if not (
            0.0
            <= self.adaptive_marking_dark_percentile
            < self.adaptive_marking_bright_percentile
            <= 100.0
        ):
            raise ValueError("adaptive marking percentiles must be ordered in [0, 100]")


@dataclass(frozen=True)
class LaneBoundaryDiagnostics:
    """Image-only evidence used to audit how a lane estimate was formed.

    The diagnostics deliberately contain pixel/fit evidence only.  Simulator
    lane pose and privileged state cannot enter this type or the estimator
    that creates it.
    """

    source: str
    strict_yellow_pixel_count: int
    strict_white_pixel_count: int
    adaptive_unknown_pixel_count: int
    yellow_center_point_count: int
    white_center_point_count: int
    boundary_disagreement_m: float | None


@dataclass(frozen=True)
class LaneMeasurementCalibration:
    """Fixed offline correction/noise floor for visual lane measurements.

    The optional affine map is fitted only on calibration trajectories.  It
    compensates the deterministic camera-preview effect without admitting
    simulator lane state into the runtime path.  Identity is the backwards-
    compatible default used by the original additive calibration.
    """

    lateral_bias_m: float
    heading_bias_rad: float
    curvature_bias_inv_m: float
    lateral_sigma_m: float
    heading_sigma_rad: float
    curvature_sigma_inv_m: float
    affine_matrix: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    affine_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quadratic_matrix: tuple[tuple[float, float, float, float, float, float], ...] = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        values = (
            self.lateral_bias_m,
            self.heading_bias_rad,
            self.curvature_bias_inv_m,
            self.lateral_sigma_m,
            self.heading_sigma_rad,
            self.curvature_sigma_inv_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("lane calibration values must be finite")
        if any(
            value <= 0.0
            for value in (
                self.lateral_sigma_m,
                self.heading_sigma_rad,
                self.curvature_sigma_inv_m,
            )
        ):
            raise ValueError("lane calibration sigmas must be positive")
        matrix = np.asarray(self.affine_matrix, dtype=float)
        offset = np.asarray(self.affine_offset, dtype=float)
        quadratic = np.asarray(self.quadratic_matrix, dtype=float)
        if matrix.shape != (3, 3) or offset.shape != (3,):
            raise ValueError("lane affine calibration must be 3x3 plus a 3-vector")
        if quadratic.shape != (3, 6):
            raise ValueError("lane quadratic calibration must be a 3x6 matrix")
        if (
            not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(offset))
            or not np.all(np.isfinite(quadratic))
        ):
            raise ValueError("lane affine calibration values must be finite")

    def apply(self, raw: LaneMeasurement) -> LaneMeasurement:
        if not raw.detected:
            return raw
        assert raw.lateral_error_m is not None
        assert raw.lateral_error_std_m is not None
        assert raw.heading_error_rad is not None
        assert raw.heading_error_std_rad is not None
        assert raw.curvature_inv_m is not None
        assert raw.curvature_std_inv_m is not None
        raw_vector = np.asarray(
            [
                raw.lateral_error_m - self.lateral_bias_m,
                wrap_angle(raw.heading_error_rad - self.heading_bias_rad),
                raw.curvature_inv_m - self.curvature_bias_inv_m,
            ],
            dtype=float,
        )
        matrix = np.asarray(self.affine_matrix, dtype=float)
        quadratic = np.asarray(self.quadratic_matrix, dtype=float)
        d_value, phi_value, kappa_value = raw_vector
        quadratic_features = np.asarray(
            (
                d_value * d_value,
                phi_value * phi_value,
                kappa_value * kappa_value,
                d_value * phi_value,
                d_value * kappa_value,
                phi_value * kappa_value,
            ),
            dtype=float,
        )
        corrected = (
            matrix @ raw_vector
            + quadratic @ quadratic_features
            + np.asarray(self.affine_offset, dtype=float)
        )
        raw_covariance = np.diag(
            np.square(
                [
                    raw.lateral_error_std_m,
                    raw.heading_error_std_rad,
                    raw.curvature_std_inv_m,
                ]
            )
        )
        feature_jacobian = np.asarray(
            (
                (2.0 * d_value, 0.0, 0.0),
                (0.0, 2.0 * phi_value, 0.0),
                (0.0, 0.0, 2.0 * kappa_value),
                (phi_value, d_value, 0.0),
                (kappa_value, 0.0, d_value),
                (0.0, kappa_value, phi_value),
            ),
            dtype=float,
        )
        calibration_jacobian = matrix + quadratic @ feature_jacobian
        propagated_std = np.sqrt(
            np.maximum(
                np.diag(
                    calibration_jacobian
                    @ raw_covariance
                    @ calibration_jacobian.T
                ),
                0.0,
            )
        )
        return LaneMeasurement(
            detected=True,
            lateral_error_m=float(corrected[0]),
            lateral_error_std_m=max(float(propagated_std[0]), self.lateral_sigma_m),
            heading_error_rad=wrap_angle(float(corrected[1])),
            heading_error_std_rad=max(
                float(propagated_std[1]), self.heading_sigma_rad
            ),
            curvature_inv_m=float(corrected[2]),
            curvature_std_inv_m=max(
                float(propagated_std[2]), self.curvature_sigma_inv_m
            ),
            visible_point_count=raw.visible_point_count,
            fit_residual_m=raw.fit_residual_m,
        )


class CameraLaneMeasurementEstimator:
    """Estimate ``d, phi, kappa`` from visible yellow/white lane markings.

    A quadratic lane centreline is fitted in ego ground coordinates:

    ``x_left(y_forward) = c0 + c1*y + c2*y^2``.

    Gym-Duckietown's locked lane-position semantics and the calibrated ego
    projection imply ``d=c0`` and ``phi=atan(c1)``.  This sign is verified with
    real lateral/heading pose sweeps; it must not be inferred from image-column
    direction alone.  Positive curvature is counter-clockwise/left.
    """

    def __init__(
        self,
        projector: CalibratedGroundProjector,
        config: LanePerceptionConfig | None = None,
    ) -> None:
        self.projector = projector
        self.config = config or LanePerceptionConfig()

    def estimate(self, front_rgb: NDArray[np.uint8]) -> LaneMeasurement:
        return self.estimate_with_diagnostics(front_rgb)[0]

    def estimate_with_diagnostics(
        self, front_rgb: NDArray[np.uint8]
    ) -> tuple[LaneMeasurement, LaneBoundaryDiagnostics]:
        calibration = self.projector.calibration
        expected_shape = (
            calibration.image_height_px,
            calibration.image_width_px,
            3,
        )
        image = np.asarray(front_rgb)
        if image.shape != expected_shape:
            raise ValueError(
                f"front RGB shape {image.shape} does not match calibration {expected_shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError("front RGB must use uint8 channels")

        if self.config.dual_boundary_fusion_enabled:
            return self._estimate_dual_boundary(image)

        yellow, white = self._colour_masks(image)
        # The yellow divider is the project control reference and has a known
        # centre offset.  Prefer it to avoid stop bars and pale tile seams that
        # satisfy the white threshold.  White is a structural fallback only.
        empty = np.zeros_like(yellow)
        yellow_points = self._centreline_points(yellow, empty)
        white_points = self._centreline_points(empty, white)
        if self._points_are_usable(yellow_points):
            centre_points = yellow_points
        elif self._points_are_usable(white_points):
            centre_points = white_points
        else:
            centre_points = yellow_points + white_points
        measurement = self._measurement_from_points(centre_points)
        return measurement, LaneBoundaryDiagnostics(
            source="legacy_yellow_preferred",
            strict_yellow_pixel_count=int(np.count_nonzero(yellow)),
            strict_white_pixel_count=int(np.count_nonzero(white)),
            adaptive_unknown_pixel_count=0,
            yellow_center_point_count=len(yellow_points),
            white_center_point_count=len(white_points),
            boundary_disagreement_m=None,
        )

    def _estimate_dual_boundary(
        self, image: NDArray[np.uint8]
    ) -> tuple[LaneMeasurement, LaneBoundaryDiagnostics]:
        yellow, white, unknown = self._separated_colour_masks(image)
        empty = np.zeros_like(yellow)
        strict_yellow_points = self._centreline_points(yellow, empty)
        strict_white_points = self._centreline_points(empty, white)

        yellow_is_strict = self._points_are_usable(strict_yellow_points)
        white_is_strict = self._points_are_usable(strict_white_points)
        yellow_points = strict_yellow_points
        white_points = strict_white_points
        if not yellow_is_strict:
            yellow_points = self._centreline_points(unknown, empty)
        if not white_is_strict:
            white_points = self._centreline_points(empty, unknown)

        yellow_usable = self._points_are_usable(yellow_points)
        white_usable = self._points_are_usable(white_points)
        disagreement: float | None = None
        source = "missing"
        uncertainty_scale = 1.0
        centre_points: list[tuple[float, float]] = []

        if yellow_usable and white_usable:
            disagreement = _centreline_disagreement(
                yellow_points,
                white_points,
                minimum_forward_span_m=self.config.minimum_forward_span_m,
                sigma_threshold=self.config.robust_sigma_threshold,
            )
            if (
                disagreement is not None
                and disagreement <= self.config.maximum_boundary_disagreement_m
            ):
                centre_points = yellow_points + white_points
                source = (
                    "dual_strict"
                    if yellow_is_strict and white_is_strict
                    else "dual_with_adaptive"
                )
                if not (yellow_is_strict and white_is_strict):
                    uncertainty_scale = self.config.adaptive_boundary_std_multiplier
            elif yellow_is_strict:
                # The yellow divider is the locked control reference.  A
                # conflicting pale/white candidate is often a stop bar, tile
                # seam, or adaptive-paint false positive.
                centre_points = yellow_points
                source = "strict_yellow_conflict"
                uncertainty_scale = self.config.single_boundary_std_multiplier
            elif white_is_strict:
                centre_points = white_points
                source = "strict_white_conflict"
                uncertainty_scale = self.config.single_boundary_std_multiplier
            else:
                # Two geometry-only hypotheses that disagree provide no safe
                # semantic evidence.  Missing is preferable to a confident,
                # wrong centreline; the EKF will perform prediction-only.
                source = "adaptive_conflict_missing"
        elif yellow_usable:
            centre_points = yellow_points
            source = "single_strict_yellow" if yellow_is_strict else "single_adaptive_yellow"
            uncertainty_scale = (
                self.config.single_boundary_std_multiplier
                if yellow_is_strict
                else self.config.adaptive_boundary_std_multiplier
            )
        elif white_usable:
            centre_points = white_points
            source = "single_strict_white" if white_is_strict else "single_adaptive_white"
            uncertainty_scale = (
                self.config.single_boundary_std_multiplier
                if white_is_strict
                else self.config.adaptive_boundary_std_multiplier
            )

        measurement = self._measurement_from_points(
            centre_points, uncertainty_scale=uncertainty_scale
        )
        return measurement, LaneBoundaryDiagnostics(
            source=source,
            strict_yellow_pixel_count=int(np.count_nonzero(yellow)),
            strict_white_pixel_count=int(np.count_nonzero(white)),
            adaptive_unknown_pixel_count=int(np.count_nonzero(unknown)),
            yellow_center_point_count=len(yellow_points),
            white_center_point_count=len(white_points),
            boundary_disagreement_m=disagreement,
        )

    def _measurement_from_points(
        self,
        centre_points: list[tuple[float, float]],
        *,
        uncertainty_scale: float = 1.0,
    ) -> LaneMeasurement:
        if len(centre_points) < self.config.minimum_point_count:
            return LaneMeasurement.missing(visible_point_count=len(centre_points))

        points = np.asarray(centre_points, dtype=float)
        pose_points = points[
            (points[:, 1] >= self.config.pose_minimum_forward_m)
            & (points[:, 1] <= self.config.pose_maximum_forward_m)
        ]
        if pose_points.shape[0] < self.config.minimum_point_count:
            return LaneMeasurement.missing(visible_point_count=len(centre_points))
        forward_span = float(np.ptp(pose_points[:, 1]))
        if forward_span < self.config.minimum_forward_span_m:
            return LaneMeasurement.missing(visible_point_count=len(centre_points))

        geometry = _fit_local_lane_geometry(
            pose_points,
            sigma_threshold=self.config.robust_sigma_threshold,
        )
        if geometry is None:
            return LaneMeasurement.missing(visible_point_count=len(centre_points))
        lateral_error, heading_error, curvature, residual_rms, kept_count = geometry

        leverage = max(forward_span, self.config.minimum_forward_span_m)
        sample_factor = max(1.0, np.sqrt(kept_count / self.config.minimum_point_count))
        lateral_std = uncertainty_scale * max(
            self.config.lateral_std_floor_m,
            residual_rms / sample_factor,
        )
        heading_std = uncertainty_scale * max(
            self.config.heading_std_floor_rad,
            residual_rms / (leverage * sample_factor),
        )
        curvature_std = uncertainty_scale * max(
            self.config.curvature_std_floor_inv_m,
            2.0
            * residual_rms
            / (
                leverage
                * leverage
                * sample_factor
            ),
        )
        values = (lateral_error, heading_error, curvature)
        if not all(isfinite(value) for value in values):
            return LaneMeasurement.missing(visible_point_count=len(centre_points))
        return LaneMeasurement(
            detected=True,
            lateral_error_m=lateral_error,
            lateral_error_std_m=lateral_std,
            heading_error_rad=heading_error,
            heading_error_std_rad=heading_std,
            curvature_inv_m=curvature,
            curvature_std_inv_m=curvature_std,
            visible_point_count=kept_count,
            fit_residual_m=residual_rms,
        )

    def _points_are_usable(self, points: list[tuple[float, float]]) -> bool:
        if len(points) < self.config.minimum_point_count:
            return False
        forward = np.asarray([point[1] for point in points], dtype=float)
        return float(np.ptp(forward)) >= self.config.minimum_forward_span_m

    def _colour_masks(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        config = self.config
        rgb = image.astype(np.int16, copy=False)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        yellow = (
            (red >= config.yellow_min_red)
            & (green >= config.yellow_min_green)
            & ((red - blue) >= config.yellow_red_blue_margin)
            & ((green - blue) >= config.yellow_green_blue_margin)
        )
        maximum = np.max(rgb, axis=2)
        minimum = np.min(rgb, axis=2)
        white = (
            (minimum >= config.white_min_intensity)
            & ((maximum - minimum) <= config.white_max_channel_spread)
        )
        roi_top = int(round(image.shape[0] * config.roi_top_fraction))
        roi_value = maximum[roi_top:]
        dark_level, bright_level = np.percentile(
            roi_value,
            (
                config.adaptive_marking_dark_percentile,
                config.adaptive_marking_bright_percentile,
            ),
        )
        adaptive_threshold = max(
            float(config.adaptive_marking_min_value),
            0.5 * (float(dark_level) + float(bright_level)),
        )
        # Domain randomization can turn nominal yellow/white paint green/cyan.
        # Geometry, not simulator truth, disambiguates the left and right lane
        # boundaries, so both colour masks admit the same bright-paint fallback.
        bright_marking = maximum >= adaptive_threshold
        yellow |= bright_marking
        white |= bright_marking
        yellow[:roi_top] = False
        white[:roi_top] = False
        return yellow, white

    def _separated_colour_masks(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Return strict yellow, strict white, and colour-unknown bright paint.

        Unlike the legacy path, adaptive bright pixels are not aliased to both
        semantic colours.  Geometry may use them as a fallback for either
        boundary, but the two resulting hypotheses must agree before fusion.
        """

        config = self.config
        rgb = image.astype(np.int16, copy=False)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        yellow = (
            (red >= config.yellow_min_red)
            & (green >= config.yellow_min_green)
            & ((red - blue) >= config.yellow_red_blue_margin)
            & ((green - blue) >= config.yellow_green_blue_margin)
        )
        maximum = np.max(rgb, axis=2)
        minimum = np.min(rgb, axis=2)
        white = (
            (minimum >= config.white_min_intensity)
            & ((maximum - minimum) <= config.white_max_channel_spread)
        )
        roi_top = int(round(image.shape[0] * config.roi_top_fraction))
        dark_level, bright_level = np.percentile(
            maximum[roi_top:],
            (
                config.adaptive_marking_dark_percentile,
                config.adaptive_marking_bright_percentile,
            ),
        )
        adaptive_threshold = max(
            float(config.adaptive_marking_min_value),
            0.5 * (float(dark_level) + float(bright_level)),
        )
        unknown = (maximum >= adaptive_threshold) & ~yellow & ~white
        yellow[:roi_top] = False
        white[:roi_top] = False
        unknown[:roi_top] = False
        return yellow, white, unknown

    def _centreline_points(
        self,
        yellow: NDArray[np.bool_],
        white: NDArray[np.bool_],
    ) -> list[tuple[float, float]]:
        config = self.config
        height = yellow.shape[0]
        roi_top = int(round(height * config.roi_top_fraction))
        points: list[tuple[float, float]] = []
        for mask, normal_sign in ((yellow, -1.0), (white, +1.0)):
            candidates: list[tuple[int, float, float]] = []
            for row in range(roi_top, height, config.row_stride_px):
                for start, stop in _contiguous_runs(mask[row]):
                    if stop - start < config.minimum_run_width_px:
                        continue
                    pixel = ImagePoint(
                        x_px=0.5 * (start + stop - 1),
                        y_px=float(row),
                    )
                    try:
                        ground = self.projector.pixel_to_ground(pixel)
                    except ValueError:
                        continue
                    if not (
                        config.minimum_forward_m
                        <= ground.y_forward_m
                        <= config.maximum_forward_m
                    ):
                        continue
                    if abs(ground.x_left_m) > config.maximum_abs_lateral_m:
                        continue
                    candidates.append((row, ground.x_left_m, ground.y_forward_m))
            expected_boundary_x = -normal_sign * config.lane_half_width_m
            boundary_points = _track_boundary_candidates(
                candidates,
                expected_x_left_m=expected_boundary_x,
                start_tolerance_m=config.boundary_start_tolerance_m,
            )
            points.extend(
                _offset_boundary_to_centerline(
                    boundary_points,
                    normal_sign=normal_sign,
                    lane_half_width_m=config.lane_half_width_m,
                    normal_fit_radius_m=config.boundary_normal_fit_radius_m,
                    sigma_threshold=config.robust_sigma_threshold,
                )
            )
        return points


def _track_boundary_candidates(
    candidates: list[tuple[int, float, float]],
    *,
    expected_x_left_m: float,
    start_tolerance_m: float,
) -> list[tuple[float, float]]:
    """Select one geometrically continuous marking run per image row."""

    by_row: dict[int, list[tuple[float, float]]] = {}
    for row, x_left, y_forward in candidates:
        by_row.setdefault(row, []).append((x_left, y_forward))
    groups = sorted(
        by_row.values(),
        key=lambda values: min(point[1] for point in values),
    )
    tracked: list[tuple[float, float]] = []
    previous_x = expected_x_left_m
    previous_y = 0.0
    for group in groups:
        chosen = min(group, key=lambda point: abs(point[0] - previous_x))
        if not tracked and abs(chosen[0] - expected_x_left_m) > start_tolerance_m:
            continue
        delta_y = max(0.0, chosen[1] - previous_y)
        maximum_jump = 0.08 + 2.5 * delta_y
        if tracked and abs(chosen[0] - previous_x) > maximum_jump:
            continue
        tracked.append(chosen)
        previous_x, previous_y = chosen
    return tracked


def _centreline_disagreement(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
    *,
    minimum_forward_span_m: float,
    sigma_threshold: float,
) -> float | None:
    """Median metric separation of two independently inferred centrelines."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 3 or b.shape[0] < 3:
        return None
    lower = max(float(np.min(a[:, 1])), float(np.min(b[:, 1])))
    upper = min(float(np.max(a[:, 1])), float(np.max(b[:, 1])))
    if upper - lower < minimum_forward_span_m:
        return None
    fit_a = _robust_polynomial_fit(
        a[:, 1], a[:, 0], degree=2, sigma_threshold=sigma_threshold
    )
    fit_b = _robust_polynomial_fit(
        b[:, 1], b[:, 0], degree=2, sigma_threshold=sigma_threshold
    )
    if fit_a is None or fit_b is None:
        return None
    forward = np.linspace(lower, upper, 21, dtype=float)
    separation = np.abs(
        np.polyval(fit_a[0], forward) - np.polyval(fit_b[0], forward)
    )
    return float(np.median(separation))


def _contiguous_runs(mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    padded = np.pad(mask.astype(np.int8, copy=False), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def _robust_quadratic_fit(
    forward_m: NDArray[np.float64],
    left_m: NDArray[np.float64],
    *,
    sigma_threshold: float,
) -> tuple[NDArray[np.float64], float, int] | None:
    return _robust_polynomial_fit(
        forward_m,
        left_m,
        degree=2,
        sigma_threshold=sigma_threshold,
    )


def _fit_local_lane_geometry(
    points: NDArray[np.float64],
    *,
    sigma_threshold: float,
) -> tuple[float, float, float, float, int] | None:
    """Fit a straight line or circular arc and return ego-local geometry.

    A preview-window line slope is not the ego-local tangent on a curve.  The
    line/arc model instead finds the lane point nearest the ego and evaluates
    both signed lateral distance and tangent there.  This matches the locked
    ``x_left, y_forward`` convention without simulator pose input.
    """

    line = _robust_polynomial_fit(
        points[:, 1],
        points[:, 0],
        degree=1,
        sigma_threshold=sigma_threshold,
    )
    if line is None:
        return None
    line_coefficients, line_residual_rms, line_kept = line
    slope, intercept = (float(value) for value in line_coefficients)

    circle = _robust_circle_fit(points, sigma_threshold=sigma_threshold)
    use_circle = False
    if circle is not None:
        center_x, center_y, radius, circle_residual_rms, circle_kept = circle
        # A circle always has another degree of freedom.  Select it only when
        # the visible bend is material and the radial residual improves the
        # straight-line explanation.  Duckietown curves are tight; the broad
        # upper bound rejects numerically fitted near-lines.
        use_circle = bool(
            0.04 <= radius <= 2.0
            and line_residual_rms >= 5.0e-4
            and circle_residual_rms <= 0.80 * line_residual_rms
        )

    if use_circle:
        center = np.asarray((center_x, center_y), dtype=float)
        intersection_squared = radius * radius - center_y * center_y
        if intersection_squared < 0.0:
            return None
        root = float(np.sqrt(max(intersection_squared, 0.0)))
        candidates = (center_x - root, center_x + root)
        lane_x = min(candidates, key=abs)
        local_point = np.asarray((lane_x, 0.0), dtype=float)
        radial = local_point - center
        tangent = np.asarray((-radial[1], radial[0]), dtype=float)
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        if tangent[1] < 0.0:
            tangent = -tangent
        left_normal = np.asarray((tangent[1], -tangent[0]), dtype=float)
        lateral = float(lane_x)
        heading = float(np.arctan2(tangent[0], tangent[1]))
        curvature_sign = float(np.sign((center - local_point) @ left_normal))
        curvature = curvature_sign / radius
        return (
            lateral,
            heading,
            curvature,
            float(circle_residual_rms),
            int(circle_kept),
        )

    normalizer = float(np.sqrt(1.0 + slope * slope))
    lateral = intercept / normalizer
    heading = atan(slope)
    return lateral, heading, 0.0, float(line_residual_rms), int(line_kept)


def _robust_circle_fit(
    points: NDArray[np.float64],
    *,
    sigma_threshold: float,
) -> tuple[float, float, float, float, int] | None:
    """Algebraic circle fit with robust radial-residual rejection."""

    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 5:
        return None
    keep = np.ones(points.shape[0], dtype=bool)
    solution: NDArray[np.float64] | None = None
    radius = 0.0
    radial_residual = np.zeros(points.shape[0], dtype=float)
    for _ in range(4):
        selected = points[keep]
        if selected.shape[0] < 5:
            return None
        x_left, y_forward = selected[:, 0], selected[:, 1]
        design = np.column_stack((2.0 * x_left, 2.0 * y_forward, np.ones_like(x_left)))
        target = np.square(x_left) + np.square(y_forward)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        center_x, center_y, constant = (float(value) for value in solution)
        radius_squared = constant + center_x * center_x + center_y * center_y
        if not isfinite(radius_squared) or radius_squared <= 0.0:
            return None
        radius = float(np.sqrt(radius_squared))
        radial_residual = (
            np.sqrt(
                np.square(points[:, 0] - center_x)
                + np.square(points[:, 1] - center_y)
            )
            - radius
        )
        median = float(np.median(radial_residual[keep]))
        mad = float(np.median(np.abs(radial_residual[keep] - median)))
        robust_sigma = max(1.4826 * mad, 1.0e-4)
        updated = np.abs(radial_residual - median) <= sigma_threshold * robust_sigma
        if np.array_equal(updated, keep):
            break
        keep = updated
    if solution is None or int(np.count_nonzero(keep)) < 5:
        return None
    center_x, center_y, _ = (float(value) for value in solution)
    residual_rms = float(np.sqrt(np.mean(np.square(radial_residual[keep]))))
    return center_x, center_y, radius, residual_rms, int(np.count_nonzero(keep))


def _offset_boundary_to_centerline(
    boundary_points: list[tuple[float, float]],
    *,
    normal_sign: float,
    lane_half_width_m: float,
    normal_fit_radius_m: float,
    sigma_threshold: float,
) -> list[tuple[float, float]]:
    """Offset a marking along its local normal, not image/ego horizontal.

    ``normal_sign=-1`` moves the yellow left boundary rightward to the lane
    centre; ``+1`` moves the white right boundary leftward.  For a straight
    line this reduces to the familiar constant x offset.
    """

    if len(boundary_points) < 3:
        return []
    points = np.asarray(boundary_points, dtype=float)
    global_fit = _robust_polynomial_fit(
        points[:, 1], points[:, 0], degree=1, sigma_threshold=sigma_threshold
    )
    if global_fit is None:
        return []
    global_slope = float(global_fit[0][0])
    slopes = np.empty(points.shape[0], dtype=float)
    for index, y_forward in enumerate(points[:, 1]):
        local = np.abs(points[:, 1] - y_forward) <= normal_fit_radius_m
        local_fit = _robust_polynomial_fit(
            points[local, 1],
            points[local, 0],
            degree=1,
            sigma_threshold=sigma_threshold,
        )
        slopes[index] = (
            global_slope if local_fit is None else float(local_fit[0][0])
        )
    normalizers = np.sqrt(1.0 + np.square(slopes))
    shifted_x = points[:, 0] + normal_sign * lane_half_width_m / normalizers
    shifted_y = (
        points[:, 1]
        - normal_sign * lane_half_width_m * slopes / normalizers
    )
    return [
        (float(x_left), float(y_forward))
        for x_left, y_forward in zip(shifted_x, shifted_y)
    ]


def _robust_polynomial_fit(
    forward_m: NDArray[np.float64],
    left_m: NDArray[np.float64],
    *,
    degree: int,
    sigma_threshold: float,
) -> tuple[NDArray[np.float64], float, int] | None:
    minimum_points = degree + 1
    if forward_m.size < minimum_points or left_m.size != forward_m.size:
        return None
    keep = np.ones(forward_m.size, dtype=bool)
    for _ in range(4):
        if int(np.count_nonzero(keep)) < minimum_points:
            return None
        coefficients = np.polyfit(forward_m[keep], left_m[keep], deg=degree)
        residuals = left_m - np.polyval(coefficients, forward_m)
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        robust_sigma = max(1.4826 * mad, 1.0e-4)
        updated = np.abs(residuals - median) <= sigma_threshold * robust_sigma
        if np.array_equal(updated, keep):
            break
        keep = updated
    if int(np.count_nonzero(keep)) < minimum_points:
        return None
    coefficients = np.polyfit(forward_m[keep], left_m[keep], deg=degree)
    kept_residuals = left_m[keep] - np.polyval(coefficients, forward_m[keep])
    residual_rms = float(np.sqrt(np.mean(np.square(kept_residuals))))
    return coefficients.astype(float), residual_rms, int(np.count_nonzero(keep))


def load_lane_perception_config(path: str | Path) -> LanePerceptionConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)["lane_perception"]
    return LanePerceptionConfig(
        lane_half_width_m=float(raw["lane_half_width_m"]),
        roi_top_fraction=float(raw["roi_top_fraction"]),
        row_stride_px=int(raw["row_stride_px"]),
        minimum_run_width_px=int(raw["minimum_run_width_px"]),
        minimum_point_count=int(raw["minimum_point_count"]),
        minimum_forward_m=float(raw["minimum_forward_m"]),
        maximum_forward_m=float(raw["maximum_forward_m"]),
        pose_minimum_forward_m=float(raw["pose_minimum_forward_m"]),
        pose_maximum_forward_m=float(raw["pose_maximum_forward_m"]),
        maximum_abs_lateral_m=float(raw["maximum_abs_lateral_m"]),
        minimum_forward_span_m=float(raw["minimum_forward_span_m"]),
        boundary_normal_fit_radius_m=float(raw["boundary_normal_fit_radius_m"]),
        boundary_start_tolerance_m=float(raw["boundary_start_tolerance_m"]),
        yellow_min_red=int(raw["yellow_min_red"]),
        yellow_min_green=int(raw["yellow_min_green"]),
        yellow_red_blue_margin=int(raw["yellow_red_blue_margin"]),
        yellow_green_blue_margin=int(raw["yellow_green_blue_margin"]),
        white_min_intensity=int(raw["white_min_intensity"]),
        white_max_channel_spread=int(raw["white_max_channel_spread"]),
        adaptive_marking_min_value=int(raw["adaptive_marking_min_value"]),
        adaptive_marking_dark_percentile=float(
            raw["adaptive_marking_dark_percentile"]
        ),
        adaptive_marking_bright_percentile=float(
            raw["adaptive_marking_bright_percentile"]
        ),
        robust_sigma_threshold=float(raw["robust_sigma_threshold"]),
        lateral_std_floor_m=float(raw["lateral_std_floor_m"]),
        heading_std_floor_rad=float(raw["heading_std_floor_rad"]),
        curvature_std_floor_inv_m=float(raw["curvature_std_floor_inv_m"]),
        dual_boundary_fusion_enabled=bool(
            raw.get("dual_boundary_fusion_enabled", False)
        ),
        maximum_boundary_disagreement_m=float(
            raw.get("maximum_boundary_disagreement_m", 0.060)
        ),
        single_boundary_std_multiplier=float(
            raw.get("single_boundary_std_multiplier", 1.50)
        ),
        adaptive_boundary_std_multiplier=float(
            raw.get("adaptive_boundary_std_multiplier", 2.00)
        ),
    )


def load_lane_measurement_calibration(
    path: str | Path,
) -> LaneMeasurementCalibration:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)["lane_calibration"]
    affine = raw.get("affine", {})
    quadratic = raw.get("quadratic", {})
    return LaneMeasurementCalibration(
        lateral_bias_m=float(raw["lateral_bias_m"]),
        heading_bias_rad=float(raw["heading_bias_rad"]),
        curvature_bias_inv_m=float(raw["curvature_bias_inv_m"]),
        lateral_sigma_m=float(raw["lateral_sigma_m"]),
        heading_sigma_rad=float(raw["heading_sigma_rad"]),
        curvature_sigma_inv_m=float(raw["curvature_sigma_inv_m"]),
        affine_matrix=tuple(
            tuple(float(value) for value in row)
            for row in affine.get(
                "matrix",
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )
        ),
        affine_offset=tuple(
            float(value) for value in affine.get("offset", (0.0, 0.0, 0.0))
        ),
        quadratic_matrix=tuple(
            tuple(float(value) for value in row)
            for row in quadratic.get(
                "matrix",
                (
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            )
        ),
    )
