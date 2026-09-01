"""Single source of truth for Version-1 spatial conventions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateConvention:
    distance_unit: str
    angle_unit: str
    linear_velocity_unit: str
    angular_velocity_unit: str
    world_ground_plane: str
    ego_forward_direction: str
    ego_lateral_direction: str
    positive_yaw_direction: str
    positive_heading_error_direction: str
    positive_bearing_direction: str


DUCKIETOWN_COORDINATES = CoordinateConvention(
    distance_unit="meter",
    angle_unit="radian",
    linear_velocity_unit="meter_per_second",
    angular_velocity_unit="radian_per_second",
    world_ground_plane="x_z",
    ego_forward_direction="positive_y_along_heading",
    ego_lateral_direction="positive_x_to_left",
    positive_yaw_direction="counter_clockwise",
    positive_heading_error_direction="left_of_lane_tangent",
    positive_bearing_direction="left_of_ego_heading",
)
