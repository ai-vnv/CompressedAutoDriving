"""Simulator-only truth used for labels, calibration, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .state import POMDPState


@dataclass(frozen=True)
class WorldPoint:
    x_m: float
    z_m: float


@dataclass(frozen=True)
class WorldVelocity:
    x_velocity_mps: float
    z_velocity_mps: float


@dataclass(frozen=True)
class WorldPose:
    x_m: float
    z_m: float
    heading_rad: float


@dataclass(frozen=True)
class WorldFootprint:
    """Simulator collision footprint on the world ground plane."""

    vertices: tuple[WorldPoint, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("a world footprint requires at least three vertices")
        if not all(
            isfinite(vertex.x_m) and isfinite(vertex.z_m)
            for vertex in self.vertices
        ):
            raise ValueError("world footprint vertices must be finite")


@dataclass(frozen=True)
class PrivilegedSimulatorState:
    """Ground truth that must never be passed to the POMDP policy."""

    true_pomdp_state: POMDPState
    ego_world_pose: WorldPose
    stop_sign_world_position: WorldPoint | None
    stop_sign_world_footprint: WorldFootprint | None
    stop_line_world_position: WorldPoint | None
    pedestrian_world_position: WorldPoint | None
    pedestrian_world_footprint: WorldFootprint | None
    pedestrian_world_velocity: WorldVelocity | None
    collision: bool | None
