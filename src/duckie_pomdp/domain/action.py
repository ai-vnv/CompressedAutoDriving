"""Policy and actuator commands are intentionally different contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PolicyAction:
    """Driving decision produced by the policy.

    ``linear_velocity_mps`` is the commanded forward velocity ``v_cmd``.
    ``angular_velocity_rad_s`` is the commanded yaw rate ``omega_cmd``; it is
    not an Ackermann steering-wheel angle.
    """

    linear_velocity_mps: float
    angular_velocity_rad_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.linear_velocity_mps):
            raise ValueError("linear_velocity_mps must be finite")
        if not isfinite(self.angular_velocity_rad_s):
            raise ValueError("angular_velocity_rad_s must be finite")
        if self.linear_velocity_mps < 0.0:
            raise ValueError("Version 1 does not allow reverse commands")


@dataclass(frozen=True)
class WheelCommand:
    """Dimensionless left/right duty command accepted by the simulator.

    These values are not wheel angular velocities. Gym-Duckietown clips this
    command at its actuator boundary, normally ``[-1, 1]``.
    """

    left: float
    right: float

    def __post_init__(self) -> None:
        if not isfinite(self.left) or not isfinite(self.right):
            raise ValueError("wheel commands must be finite")
        if not -1.0 <= self.left <= 1.0 or not -1.0 <= self.right <= 1.0:
            raise ValueError("wheel commands must be within [-1, 1]")


@dataclass(frozen=True)
class NormalizedPolicyAction:
    """Network output before conversion into physical command units."""

    linear: float
    angular: float

    def __post_init__(self) -> None:
        if not isfinite(self.linear) or not isfinite(self.angular):
            raise ValueError("normalized policy actions must be finite")
        if not -1.0 <= self.linear <= 1.0:
            raise ValueError("normalized linear action must be within [-1, 1]")
        if not -1.0 <= self.angular <= 1.0:
            raise ValueError("normalized angular action must be within [-1, 1]")
