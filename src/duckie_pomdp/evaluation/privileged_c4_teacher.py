"""Privileged C4 teacher used only to label offline guided rollouts."""

from __future__ import annotations

from math import isfinite
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


class PrivilegedC4Teacher:
    """Deterministic safety teacher over simulator truth, never a runtime policy.

    The student observation is not an argument by design.  The dataset builder
    pairs this teacher's action with the independently produced public 29D
    belief vector.
    """

    def act(self, info: Mapping[str, object]) -> NDArray[np.float32]:
        truth = info.get("evaluation_gt")
        if not isinstance(truth, Mapping):
            raise ValueError("privileged C4 teacher requires evaluation_gt")

        lateral = _finite(truth, "lane_lateral_error_m")
        heading = _finite(truth, "lane_heading_error_rad")
        curvature = _optional_finite(truth, "road_curvature_inv_m", 0.0)
        velocity = 0.18

        if truth.get("pedestrian_range_m") is not None:
            pedestrian_range = _finite(truth, "pedestrian_range_m")
            pedestrian_bearing = abs(_finite(truth, "pedestrian_bearing_rad"))
            closing_rate = _optional_finite(
                truth, "pedestrian_radial_velocity_mps", 0.0
            )
            if pedestrian_bearing <= 0.65:
                if pedestrian_range <= 0.78 or (
                    pedestrian_range <= 0.95 and closing_rate < -0.05
                ):
                    velocity = 0.0
                elif pedestrian_range <= 1.15:
                    velocity = min(velocity, 0.08)

        stop_completed = bool(info.get("stop_completed", False))
        stop_distance = _optional_finite(truth, "stop_line_distance_m", 10.0)
        if not stop_completed and stop_distance <= 0.15:
            velocity = 0.0
        elif not stop_completed and stop_distance <= 0.50:
            velocity = min(velocity, 0.08)
        elif stop_completed:
            velocity = min(velocity, 0.20)

        omega = float(
            np.clip(
                2.0 * lateral
                + 3.0 * heading
                + velocity * float(np.clip(curvature, -4.0, 4.0)),
                -4.0,
                4.0,
            )
        )
        turn_fraction = min(abs(omega) / 4.0, 1.0)
        velocity = max(
            velocity * (1.0 - 0.55 * turn_fraction), min(velocity, 0.08)
        )
        return np.asarray(
            (2.0 * velocity / 0.4 - 1.0, omega / 4.0), dtype=np.float32
        )


def _finite(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if value is None or not isfinite(float(value)):
        raise ValueError(f"privileged teacher field {key} must be finite")
    return float(value)


def _optional_finite(
    mapping: Mapping[str, object], key: str, default: float
) -> float:
    value = mapping.get(key)
    if value is None:
        return float(default)
    if not isfinite(float(value)):
        raise ValueError(f"privileged teacher field {key} must be finite")
    return float(value)
