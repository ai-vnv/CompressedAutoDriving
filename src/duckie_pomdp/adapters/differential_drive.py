"""Convert chassis-level policy commands into Duckietown wheel commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from duckie_pomdp.domain.action import (
    NormalizedPolicyAction,
    PolicyAction,
    WheelCommand,
)


@dataclass(frozen=True)
class DifferentialDriveCalibration:
    """Geometry and motor calibration used by Gym-Duckietown.

    ``motor_constant_rad_s_per_unit`` maps a unit duty command to motor angular
    velocity before gain and trim compensation. Defaults mirror
    ``gym_duckietown.envs.DuckietownEnv``.
    """

    wheel_radius_m: float = 0.0318
    wheel_separation_m: float = 0.102
    motor_constant_rad_s_per_unit: float = 27.0
    gain: float = 1.0
    trim: float = 0.0
    command_limit: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_separation_m": self.wheel_separation_m,
            "motor_constant_rad_s_per_unit": self.motor_constant_rad_s_per_unit,
            "command_limit": self.command_limit,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.command_limit > 1.0:
            raise ValueError("command_limit cannot exceed simulator bounds")
        if not isfinite(self.gain) or not isfinite(self.trim):
            raise ValueError("gain and trim must be finite")
        if self.gain + self.trim <= 0.0 or self.gain - self.trim <= 0.0:
            raise ValueError("gain +/- trim must remain positive")


@dataclass(frozen=True)
class WheelAngularVelocity:
    """Intermediate physical wheel rates; never passed to the simulator."""

    left_rad_s: float
    right_rad_s: float


@dataclass(frozen=True)
class UnclippedWheelCommand:
    """Raw duty demand retained only for saturation diagnostics."""

    left: float
    right: float


@dataclass(frozen=True)
class ActionConversion:
    """Trace of one conversion, including any actuator saturation."""

    requested_action: PolicyAction
    wheel_angular_velocity: WheelAngularVelocity
    unclipped_wheel_command: UnclippedWheelCommand
    wheel_command: WheelCommand
    left_saturated: bool
    right_saturated: bool

    @property
    def saturated(self) -> bool:
        return self.left_saturated or self.right_saturated


class DifferentialDriveActionAdapter:
    """Duckietown-compatible ``(v, omega)`` to wheel-duty conversion.

    Positive angular velocity is counter-clockwise: the right wheel receives
    the larger command and the left wheel receives the smaller command.
    """

    def __init__(
        self,
        calibration: DifferentialDriveCalibration | None = None,
    ) -> None:
        self.calibration = calibration or DifferentialDriveCalibration()

    def to_wheels(self, action: PolicyAction) -> WheelCommand:
        return self.convert(action).wheel_command

    def convert(self, action: PolicyAction) -> ActionConversion:
        calibration = self.calibration
        half_baseline_omega = (
            0.5 * calibration.wheel_separation_m * action.angular_velocity_rad_s
        )
        left_rate = (
            action.linear_velocity_mps - half_baseline_omega
        ) / calibration.wheel_radius_m
        right_rate = (
            action.linear_velocity_mps + half_baseline_omega
        ) / calibration.wheel_radius_m

        left_unclipped = left_rate * (
            calibration.gain - calibration.trim
        ) / calibration.motor_constant_rad_s_per_unit
        right_unclipped = right_rate * (
            calibration.gain + calibration.trim
        ) / calibration.motor_constant_rad_s_per_unit

        limit = calibration.command_limit
        left = _clip(left_unclipped, -limit, limit)
        right = _clip(right_unclipped, -limit, limit)
        return ActionConversion(
            requested_action=action,
            wheel_angular_velocity=WheelAngularVelocity(
                left_rad_s=left_rate,
                right_rad_s=right_rate,
            ),
            unclipped_wheel_command=UnclippedWheelCommand(
                left=left_unclipped,
                right=right_unclipped,
            ),
            wheel_command=WheelCommand(left=left, right=right),
            left_saturated=left != left_unclipped,
            right_saturated=right != right_unclipped,
        )


@dataclass(frozen=True)
class PolicyActionBounds:
    """Configured candidate bounds evaluated by an actuator-envelope gate."""

    maximum_linear_velocity_mps: float
    maximum_angular_velocity_rad_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.maximum_linear_velocity_mps):
            raise ValueError("maximum_linear_velocity_mps must be finite")
        if self.maximum_linear_velocity_mps <= 0.0:
            raise ValueError("maximum_linear_velocity_mps must be positive")
        if not isfinite(self.maximum_angular_velocity_rad_s):
            raise ValueError("maximum_angular_velocity_rad_s must be finite")
        if self.maximum_angular_velocity_rad_s <= 0.0:
            raise ValueError("maximum_angular_velocity_rad_s must be positive")


class NormalizedActionScaler:
    """Scale network output to the non-reversing physical action space."""

    def __init__(self, bounds: PolicyActionBounds) -> None:
        self.bounds = bounds

    def to_policy_action(self, action: NormalizedPolicyAction) -> PolicyAction:
        return PolicyAction(
            linear_velocity_mps=(action.linear + 1.0)
            * 0.5
            * self.bounds.maximum_linear_velocity_mps,
            angular_velocity_rad_s=action.angular
            * self.bounds.maximum_angular_velocity_rad_s,
        )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
