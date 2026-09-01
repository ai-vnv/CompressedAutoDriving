"""Numerical lane geometry derived from Gym-Duckietown Bezier centerlines."""

from __future__ import annotations

from math import cos, isfinite, sin
from typing import Any

import numpy as np


def lane_curvature_inv_m(
    simulator: Any,
    position: np.ndarray,
    heading_rad: float,
) -> float | None:
    """Return signed lane curvature at the closest route centerline point.

    Positive curvature turns left under the project's CCW-positive yaw
    convention. ``None`` means the pose is outside a drivable tile.
    """

    i, j = simulator.get_grid_coords(position)
    tile = simulator._get_tile(i, j)
    if tile is None or not bool(tile.get("drivable", False)):
        return None

    curves = np.asarray(tile["curves"], dtype=float)
    curve_headings = curves[:, -1, :] - curves[:, 0, :]
    norms = np.linalg.norm(curve_headings, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("simulator lane curve has zero-length heading")
    curve_headings = curve_headings / norms[:, np.newaxis]
    forward = np.array([cos(heading_rad), 0.0, -sin(heading_rad)])
    control_points = curves[int(np.argmax(curve_headings @ forward))]
    parameter = _closest_parameter(control_points, np.asarray(position, dtype=float))
    first = _first_derivative(control_points, parameter)
    second = _second_derivative(control_points, parameter)

    dx = float(first[0])
    dy = float(-first[2])
    ddx = float(second[0])
    ddy = float(-second[2])
    denominator = (dx * dx + dy * dy) ** 1.5
    if denominator <= 1e-12:
        raise RuntimeError("simulator lane curve derivative is degenerate")
    curvature = (dx * ddy - dy * ddx) / denominator
    if not isfinite(curvature):
        raise RuntimeError("simulator lane curvature is non-finite")
    return curvature


def _closest_parameter(control_points: np.ndarray, position: np.ndarray) -> float:
    samples = np.linspace(0.0, 1.0, 65)
    distances = np.array(
        [np.sum((_point(control_points, value) - position) ** 2) for value in samples]
    )
    best = int(np.argmin(distances))
    lower = float(samples[max(0, best - 1)])
    upper = float(samples[min(len(samples) - 1, best + 1)])
    for _ in range(32):
        one_third = (upper - lower) / 3.0
        left = lower + one_third
        right = upper - one_third
        left_error = np.sum((_point(control_points, left) - position) ** 2)
        right_error = np.sum((_point(control_points, right) - position) ** 2)
        if left_error <= right_error:
            upper = right
        else:
            lower = left
    return 0.5 * (lower + upper)


def _point(control_points: np.ndarray, t: float) -> np.ndarray:
    return (
        (1.0 - t) ** 3 * control_points[0]
        + 3.0 * t * (1.0 - t) ** 2 * control_points[1]
        + 3.0 * t**2 * (1.0 - t) * control_points[2]
        + t**3 * control_points[3]
    )


def _first_derivative(control_points: np.ndarray, t: float) -> np.ndarray:
    return (
        3.0 * (1.0 - t) ** 2 * (control_points[1] - control_points[0])
        + 6.0 * (1.0 - t) * t * (control_points[2] - control_points[1])
        + 3.0 * t**2 * (control_points[3] - control_points[2])
    )


def _second_derivative(control_points: np.ndarray, t: float) -> np.ndarray:
    return (
        6.0
        * (1.0 - t)
        * (control_points[2] - 2.0 * control_points[1] + control_points[0])
        + 6.0
        * t
        * (control_points[3] - 2.0 * control_points[2] + control_points[1])
    )
