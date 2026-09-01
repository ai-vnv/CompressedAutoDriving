"""Agent-side route dead reckoning without simulator world-pose access."""

from __future__ import annotations

from math import cos, isfinite, sin

from duckie_pomdp.domain.observation import EgoObservation, RoadMeasurement
from duckie_pomdp.scenario import MinimalPOMDPScenario


class F10RoadObserver:
    """Estimate signed stop-line distance from route prior and ego motion.

    The reset prior comes from the configured navigation route/spawn, not a
    simulator-state read. Closed routes may provide a centerline arc-distance
    prior; legacy straight scenarios retain the geometric projection fallback.
    Updates integrate measured actual chassis motion in both cases.
    """

    def __init__(
        self,
        scenario: MinimalPOMDPScenario,
        *,
        map_tile_size_m: float,
        initial_stop_line_distance_m: float | None = None,
    ) -> None:
        if not isfinite(map_tile_size_m) or map_tile_size_m <= 0.0:
            raise ValueError("map tile size must be positive and finite")
        self._scenario = scenario
        self._map_tile_size_m = map_tile_size_m
        if initial_stop_line_distance_m is not None and (
            not isfinite(initial_stop_line_distance_m)
            or initial_stop_line_distance_m < 0.0
        ):
            raise ValueError(
                "initial stop-line route distance must be finite and nonnegative"
            )
        self._initial_stop_line_distance_m = initial_stop_line_distance_m
        self._distance_m: float | None = None

    def reset(self) -> RoadMeasurement:
        if self._initial_stop_line_distance_m is not None:
            self._distance_m = self._initial_stop_line_distance_m
            return self.measurement
        local_x, _, local_z = self._scenario.ego_start_pose_m
        tile_x, tile_z = self._scenario.ego_start_tile
        start_x = tile_x * self._map_tile_size_m + local_x
        start_z = tile_z * self._map_tile_size_m + local_z
        stop = self._scenario.stop_line
        delta_x = stop.world_x_m - start_x
        delta_z = stop.world_z_m - start_z
        self._distance_m = (
            delta_x * cos(stop.route_heading_rad)
            - delta_z * sin(stop.route_heading_rad)
        )
        return self.measurement

    def update(self, ego: EgoObservation, *, dt_s: float) -> RoadMeasurement:
        if self._distance_m is None:
            raise RuntimeError("road observer must be reset before update")
        if not isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("road-observer dt_s must be positive and finite")
        forward_route_velocity = ego.linear_velocity_mps * cos(ego.heading_error_rad)
        self._distance_m -= forward_route_velocity * dt_s
        return self.measurement

    @property
    def measurement(self) -> RoadMeasurement:
        if self._distance_m is None:
            raise RuntimeError("road observer is unavailable before reset")
        return RoadMeasurement(
            curvature_inv_m=0.0,
            stop_line_distance_m=self._distance_m,
        )
