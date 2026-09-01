"""Calibrated pinhole projection between pixels and the ego ground plane."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, radians, sin, tan

import numpy as np

from duckie_pomdp.domain.detection import ImagePoint
from duckie_pomdp.domain.measurement import GroundPoint
from duckie_pomdp.domain.privileged import WorldPoint, WorldPose


@dataclass(frozen=True)
class CameraCalibration:
    image_width_px: int
    image_height_px: int
    vertical_fov_deg: float
    camera_height_m: float
    camera_pitch_deg: float
    camera_forward_offset_m: float
    distortion_enabled: bool = False

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("camera image dimensions must be positive")
        values = (
            self.vertical_fov_deg,
            self.camera_height_m,
            self.camera_pitch_deg,
            self.camera_forward_offset_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("camera calibration values must be finite")
        if not 0.0 < self.vertical_fov_deg < 180.0:
            raise ValueError("vertical FOV must be within (0, 180) degrees")
        if self.camera_height_m <= 0.0:
            raise ValueError("camera height must be positive")
        if self.camera_forward_offset_m < 0.0:
            raise ValueError("camera forward offset cannot be negative")


@dataclass(frozen=True)
class PolarGroundPoint:
    range_m: float
    bearing_rad: float


class CalibratedGroundProjector:
    """Project using only camera calibration, never simulator object truth."""

    def __init__(self, calibration: CameraCalibration) -> None:
        if calibration.distortion_enabled:
            raise ValueError(
                "distorted frames require undistortion before calibrated projection"
            )
        self.calibration = calibration
        self._view_projection = _view_projection_matrix(calibration)
        self._inverse_view_projection = np.linalg.inv(self._view_projection)

    def pixel_to_ground(self, point: ImagePoint) -> GroundPoint:
        calibration = self.calibration
        if not isfinite(point.x_px) or not isfinite(point.y_px):
            raise ValueError("pixel coordinates must be finite")
        if not 0.0 <= point.x_px <= calibration.image_width_px - 1:
            raise ValueError("pixel x lies outside the calibrated image")
        if not 0.0 <= point.y_px <= calibration.image_height_px - 1:
            raise ValueError("pixel y lies outside the calibrated image")

        ndc_x = 2.0 * point.x_px / calibration.image_width_px - 1.0
        ndc_y = 1.0 - 2.0 * point.y_px / calibration.image_height_px
        near = self._unproject(ndc_x, ndc_y, -1.0)
        far = self._unproject(ndc_x, ndc_y, 1.0)
        ray = far - near
        if abs(ray[1]) <= 1e-12:
            raise ValueError("pixel ray is parallel to the ground plane")
        scale = -near[1] / ray[1]
        if scale <= 0.0:
            raise ValueError("pixel ray intersects the ground behind the camera")
        ground = near + scale * ray
        return GroundPoint(
            x_left_m=float(-ground[2]),
            y_forward_m=float(ground[0]),
        )

    def ground_to_pixel(self, point: GroundPoint) -> ImagePoint:
        world = np.array(
            [point.y_forward_m, 0.0, -point.x_left_m, 1.0],
            dtype=float,
        )
        clip = self._view_projection @ world
        if clip[3] <= 0.0:
            raise ValueError("ground point is behind the camera")
        ndc = clip[:3] / clip[3]
        return ImagePoint(
            x_px=float((ndc[0] + 1.0) * 0.5 * self.calibration.image_width_px),
            y_px=float((1.0 - ndc[1]) * 0.5 * self.calibration.image_height_px),
        )

    def _unproject(self, ndc_x: float, ndc_y: float, ndc_z: float) -> np.ndarray:
        homogeneous = self._inverse_view_projection @ np.array(
            [ndc_x, ndc_y, ndc_z, 1.0],
            dtype=float,
        )
        return homogeneous[:3] / homogeneous[3]


def ground_to_polar(point: GroundPoint) -> PolarGroundPoint:
    return PolarGroundPoint(
        range_m=hypot(point.x_left_m, point.y_forward_m),
        bearing_rad=atan2(point.x_left_m, point.y_forward_m),
    )


def world_to_ego(position: WorldPoint, ego_pose: WorldPose) -> GroundPoint:
    delta_x = position.x_m - ego_pose.x_m
    delta_z = position.z_m - ego_pose.z_m
    heading = ego_pose.heading_rad
    return GroundPoint(
        x_left_m=-delta_x * sin(heading) - delta_z * cos(heading),
        y_forward_m=delta_x * cos(heading) - delta_z * sin(heading),
    )


def _view_projection_matrix(calibration: CameraCalibration) -> np.ndarray:
    near = 0.04
    far = 100.0
    aspect = calibration.image_width_px / calibration.image_height_px
    focal = 1.0 / tan(radians(calibration.vertical_fov_deg) * 0.5)
    projection = np.array(
        [
            [focal / aspect, 0.0, 0.0, 0.0],
            [0.0, focal, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    pitch = radians(calibration.camera_pitch_deg)
    rotate_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos(pitch), -sin(pitch), 0.0],
            [0.0, sin(pitch), cos(pitch), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    translate = np.eye(4, dtype=float)
    translate[2, 3] = calibration.camera_forward_offset_m
    eye = np.array([0.0, calibration.camera_height_m, 0.0], dtype=float)
    # Gym-Duckietown's gluLookAt target has the same y coordinate as the
    # camera. The downward pitch is applied separately by glRotatef above.
    # Pointing look-at at y=0 would introduce an additional, fictitious pitch.
    center = np.array(
        [1.0, calibration.camera_height_m, 0.0],
        dtype=float,
    )
    look_at = _look_at_matrix(eye, center, np.array([0.0, 1.0, 0.0]))
    return projection @ rotate_x @ translate @ look_at


def _look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    corrected_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=float)
    matrix[0, :3] = side
    matrix[1, :3] = corrected_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(side @ eye)
    matrix[1, 3] = -float(corrected_up @ eye)
    matrix[2, 3] = float(forward @ eye)
    return matrix
