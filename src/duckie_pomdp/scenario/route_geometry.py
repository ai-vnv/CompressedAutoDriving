"""Offline geometry gates for the separated experiment-loop object curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Sequence

import numpy as np

@dataclass(frozen=True)
class RouteObjectGeometry:
    pedestrian_intersection_world: tuple[float, float]
    pedestrian_route_error_m: float
    pedestrian_endpoint_side_a_m: float
    pedestrian_endpoint_side_b_m: float
    stop_route_error_m: float
    forward_route_separation_m: float
    euclidean_object_separation_m: float

    @property
    def pedestrian_crosses_route(self) -> bool:
        return (
            self.pedestrian_route_error_m <= 0.01
            and self.pedestrian_endpoint_side_a_m
            * self.pedestrian_endpoint_side_b_m
            < 0.0
        )


def validate_route_object_geometry(
    tiles: Sequence[Any],
    *,
    pedestrian_start_world: tuple[float, float],
    pedestrian_end_world: tuple[float, float],
    stop_line_world: tuple[float, float],
    stop_sign_world: tuple[float, float],
    samples_per_curve: int = 301,
) -> RouteObjectGeometry:
    """Measure path/centreline intersection and along-route object separation."""

    if samples_per_curve < 3:
        raise ValueError("samples_per_curve must be at least three")
    route = _ordered_route_polyline(tiles, samples_per_curve=samples_per_curve)
    start = np.asarray(pedestrian_start_world, dtype=float)
    end = np.asarray(pedestrian_end_world, dtype=float)
    stop = np.asarray(stop_line_world, dtype=float)
    if np.linalg.norm(end - start) <= 0.0:
        raise ValueError("pedestrian path endpoints must differ")

    pedestrian_projection = _closest_polyline_pair(route, start, end)
    stop_projection = _closest_point_on_polyline(route, stop)
    route_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    separation = (
        stop_projection[1] - pedestrian_projection[2]
    ) % route_length

    route_tangent = pedestrian_projection[3]
    right = np.array([-route_tangent[1], route_tangent[0]], dtype=float)
    centre = pedestrian_projection[0]
    side_a = float(np.dot(start - centre, right))
    side_b = float(np.dot(end - centre, right))
    return RouteObjectGeometry(
        pedestrian_intersection_world=(float(centre[0]), float(centre[1])),
        pedestrian_route_error_m=float(pedestrian_projection[1]),
        pedestrian_endpoint_side_a_m=side_a,
        pedestrian_endpoint_side_b_m=side_b,
        stop_route_error_m=float(stop_projection[0]),
        forward_route_separation_m=float(separation),
        euclidean_object_separation_m=hypot(
            stop_sign_world[0] - centre[0],
            stop_sign_world[1] - centre[1],
        ),
    )


def forward_route_distance_m(
    tiles: Sequence[Any],
    *,
    start_world: tuple[float, float],
    destination_world: tuple[float, float],
    samples_per_curve: int = 301,
    maximum_projection_error_m: float = 0.15,
) -> float:
    """Return forward arc distance between two points on a closed route.

    This is a navigation-route prior, not a simulator-state query. The route
    comes from configured map centerlines and the start point comes from the
    configured spawn. Runtime updates can then dead-reckon the remaining
    distance using measured ego motion without leaking world pose.
    """

    if samples_per_curve < 3:
        raise ValueError("samples_per_curve must be at least three")
    if maximum_projection_error_m <= 0.0:
        raise ValueError("maximum projection error must be positive")
    route = _ordered_route_polyline(tiles, samples_per_curve=samples_per_curve)
    start_error, start_arc = _closest_point_on_polyline(
        route, np.asarray(start_world, dtype=float)
    )
    destination_error, destination_arc = _closest_point_on_polyline(
        route, np.asarray(destination_world, dtype=float)
    )
    if start_error > maximum_projection_error_m:
        raise ValueError(
            f"configured route start is {start_error:.3f} m from the centerline"
        )
    if destination_error > maximum_projection_error_m:
        raise ValueError(
            "configured route destination is "
            f"{destination_error:.3f} m from the centerline"
        )
    route_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    return float((destination_arc - start_arc) % route_length)


def _ordered_route_polyline(
    tiles: Sequence[Any], *, samples_per_curve: int
) -> np.ndarray:
    from gym_duckietown.graphics import bezier_point

    remaining = list(tiles)
    if not remaining:
        raise ValueError("route needs at least one drivable tile")
    ordered = [remaining.pop(0)]
    while remaining:
        endpoint = np.asarray(ordered[-1].curve_world[-1], dtype=float)
        distances = [
            float(np.linalg.norm(np.asarray(tile.curve_world[0]) - endpoint))
            for tile in remaining
        ]
        index = int(np.argmin(distances))
        if distances[index] > 1e-3:
            raise RuntimeError("drivable tile curves do not form one closed route")
        ordered.append(remaining.pop(index))

    points: list[np.ndarray] = []
    for tile_index, tile in enumerate(ordered):
        curve = np.asarray(tile.curve_world, dtype=float)
        for index, parameter in enumerate(np.linspace(0.0, 1.0, samples_per_curve)):
            if tile_index and index == 0:
                continue
            point = bezier_point(curve, float(parameter))
            points.append(np.array([float(point[0]), float(point[2])]))
    points.append(points[0].copy())
    return np.asarray(points, dtype=float)


def _closest_polyline_pair(
    route: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray]:
    best: tuple[np.ndarray, float, float, np.ndarray] | None = None
    cumulative = 0.0
    for first, second in zip(route[:-1], route[1:]):
        point_route, point_segment, t_route = _closest_segment_pair(
            first, second, segment_start, segment_end
        )
        distance = float(np.linalg.norm(point_route - point_segment))
        tangent = second - first
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            continue
        candidate = (
            point_route,
            distance,
            cumulative + t_route * length,
            tangent / length,
        )
        if best is None or candidate[1] < best[1]:
            best = candidate
        cumulative += length
    if best is None:
        raise RuntimeError("route polyline has no nonzero segment")
    return best


def _closest_point_on_polyline(
    route: np.ndarray, point: np.ndarray
) -> tuple[float, float]:
    best_distance = float("inf")
    best_arc = 0.0
    cumulative = 0.0
    for first, second in zip(route[:-1], route[1:]):
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            continue
        fraction = float(
            np.clip(np.dot(point - first, delta) / (length * length), 0.0, 1.0)
        )
        projected = first + fraction * delta
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_arc = cumulative + fraction * length
        cumulative += length
    return best_distance, best_arc


def _closest_segment_pair(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest points on two 2-D segments, including exact intersections."""

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    s = 0.0 if denominator == 0.0 else (b * e - c * d) / denominator
    t = 0.0 if denominator == 0.0 else (a * e - b * d) / denominator
    s = float(np.clip(s, 0.0, 1.0))
    t = float(np.clip((b * s + e) / c if c > 0.0 else 0.0, 0.0, 1.0))
    s = float(np.clip((b * t - d) / a if a > 0.0 else 0.0, 0.0, 1.0))
    return p1 + s * u, q1 + t * v, s
