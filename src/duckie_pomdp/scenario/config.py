"""Typed configuration for the deterministic Version-1 POMDP scenario."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import hypot, isfinite
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class PedestrianMode(str, Enum):
    STATIONARY = "stationary"
    CROSS_LEFT_TO_RIGHT = "cross_left_to_right"
    CROSS_RIGHT_TO_LEFT = "cross_right_to_left"


@dataclass(frozen=True)
class StopLineScenario:
    world_x_m: float
    world_z_m: float
    route_heading_rad: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.world_x_m, self.world_z_m, self.route_heading_rad)
        ):
            raise ValueError("stop-line geometry must be finite")


@dataclass(frozen=True)
class PedestrianScenario:
    mode: PedestrianMode
    speed_mps: float
    crossing_distance_m: float
    object_kind: str = "duckie"
    path_start_world_x_m: float | None = None
    path_start_world_z_m: float | None = None
    path_end_world_x_m: float | None = None
    path_end_world_z_m: float | None = None
    start_delay_s: float = 0.0
    reverse_start_delay_s: float | None = None
    repeat_crossing: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.speed_mps) or self.speed_mps < 0.0:
            raise ValueError("pedestrian speed_mps must be finite and nonnegative")
        if not isfinite(self.crossing_distance_m) or self.crossing_distance_m <= 0.0:
            raise ValueError("pedestrian crossing_distance_m must be positive")
        if self.mode is not PedestrianMode.STATIONARY and self.speed_mps == 0.0:
            raise ValueError("crossing pedestrian must have positive speed")
        path_values = (
            self.path_start_world_x_m,
            self.path_start_world_z_m,
            self.path_end_world_x_m,
            self.path_end_world_z_m,
        )
        if any(value is not None for value in path_values):
            if not all(value is not None and isfinite(value) for value in path_values):
                raise ValueError("explicit pedestrian path needs four finite coordinates")
            assert all(value is not None for value in path_values)
            distance = hypot(
                self.path_end_world_x_m - self.path_start_world_x_m,
                self.path_end_world_z_m - self.path_start_world_z_m,
            )
            if distance <= 0.0:
                raise ValueError("explicit pedestrian path endpoints must differ")
            if abs(distance - self.crossing_distance_m) > 1e-6:
                raise ValueError(
                    "crossing_distance_m must equal the explicit path length"
                )
        if not isfinite(self.start_delay_s) or self.start_delay_s < 0.0:
            raise ValueError("pedestrian start_delay_s must be finite and nonnegative")
        if self.reverse_start_delay_s is not None and (
            not isfinite(self.reverse_start_delay_s)
            or self.reverse_start_delay_s < 0.0
        ):
            raise ValueError(
                "pedestrian reverse_start_delay_s must be finite and nonnegative"
            )

    @property
    def has_explicit_path(self) -> bool:
        return self.path_start_world_x_m is not None

    def path_for_mode(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return start/end world points for the configured crossing direction."""

        if not self.has_explicit_path:
            return None
        assert self.path_start_world_x_m is not None
        assert self.path_start_world_z_m is not None
        assert self.path_end_world_x_m is not None
        assert self.path_end_world_z_m is not None
        forward = (
            (self.path_start_world_x_m, self.path_start_world_z_m),
            (self.path_end_world_x_m, self.path_end_world_z_m),
        )
        if self.mode is PedestrianMode.CROSS_RIGHT_TO_LEFT:
            return forward[1], forward[0]
        return forward

    def start_delay_for_mode(self) -> float:
        if (
            self.mode is PedestrianMode.CROSS_RIGHT_TO_LEFT
            and self.reverse_start_delay_s is not None
        ):
            return self.reverse_start_delay_s
        return self.start_delay_s


@dataclass(frozen=True)
class MinimalPOMDPScenario:
    name: str
    map_path: Path
    seed: int
    ego_start_tile: tuple[int, int]
    ego_start_pose_m: tuple[float, float, float]
    ego_heading_rad: float
    stop_sign_kind: str
    stop_line: StopLineScenario
    pedestrian: PedestrianScenario

    def __post_init__(self) -> None:
        if not self.map_path.is_file():
            raise FileNotFoundError(f"scenario map does not exist: {self.map_path}")
        if len(self.ego_start_tile) != 2 or len(self.ego_start_pose_m) != 3:
            raise ValueError("ego start tile/pose dimensions are invalid")
        if not all(isfinite(value) for value in self.ego_start_pose_m):
            raise ValueError("ego start pose must be finite")
        if not isfinite(self.ego_heading_rad):
            raise ValueError("ego heading must be finite")

    def with_pedestrian_mode(self, mode: PedestrianMode) -> MinimalPOMDPScenario:
        speed = 0.0 if mode is PedestrianMode.STATIONARY else self.pedestrian.speed_mps
        return replace(
            self,
            pedestrian=replace(self.pedestrian, mode=mode, speed_mps=speed),
        )


def load_scenario(path: str | Path) -> MinimalPOMDPScenario:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    scenario = data["scenario"]
    ego = data["ego"]
    stop_sign = data["stop_sign"]
    stop_line = data["stop_line"]
    pedestrian = data["pedestrian"]
    _require_enabled("stop_sign", stop_sign)
    _require_enabled("stop_line", stop_line)
    _require_enabled("pedestrian", pedestrian)

    map_path = (config_path.parent / str(scenario["map"])).resolve()
    return MinimalPOMDPScenario(
        name=str(scenario["name"]),
        map_path=map_path,
        seed=int(scenario["seed"]),
        ego_start_tile=tuple(int(value) for value in ego["start_tile"]),
        ego_start_pose_m=tuple(float(value) for value in ego["start_pose_m"]),
        ego_heading_rad=float(ego["heading_rad"]),
        stop_sign_kind=str(stop_sign["object_kind"]),
        stop_line=StopLineScenario(
            world_x_m=float(stop_line["world_x_m"]),
            world_z_m=float(stop_line["world_z_m"]),
            route_heading_rad=float(stop_line["route_heading_rad"]),
        ),
        pedestrian=PedestrianScenario(
            mode=PedestrianMode(str(pedestrian["mode"])),
            speed_mps=float(pedestrian["speed_mps"]),
            crossing_distance_m=float(pedestrian["crossing_distance_m"]),
            object_kind=str(pedestrian["object_kind"]),
            path_start_world_x_m=_optional_float(
                pedestrian, "path_start_world_x_m"
            ),
            path_start_world_z_m=_optional_float(
                pedestrian, "path_start_world_z_m"
            ),
            path_end_world_x_m=_optional_float(
                pedestrian, "path_end_world_x_m"
            ),
            path_end_world_z_m=_optional_float(
                pedestrian, "path_end_world_z_m"
            ),
            start_delay_s=float(pedestrian.get("start_delay_s", 0.0)),
            reverse_start_delay_s=_optional_float(
                pedestrian, "reverse_start_delay_s"
            ),
            repeat_crossing=bool(pedestrian.get("repeat_crossing", True)),
        ),
    )


def _require_enabled(name: str, section: dict[str, Any]) -> None:
    if section.get("enabled") is not True:
        raise ValueError(f"minimal POMDP scenario requires enabled {name}")


def _optional_float(section: dict[str, Any], name: str) -> float | None:
    value = section.get(name)
    return None if value is None else float(value)
