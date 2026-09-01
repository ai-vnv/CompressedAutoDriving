"""Single normalized policy-action boundary above the chassis adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duckie_pomdp.adapters.differential_drive import (
    NormalizedActionScaler,
    PolicyActionBounds,
)
from duckie_pomdp.domain.action import NormalizedPolicyAction, PolicyAction


@dataclass(frozen=True)
class NormalizedActionMapping:
    requested_normalized: tuple[float, float]
    final_normalized: tuple[float, float]
    policy_action: PolicyAction
    clipped: bool


class NormalizedActionMapper:
    def __init__(self, maximum_linear_velocity_mps: float, maximum_angular_velocity_rad_s: float) -> None:
        self._scaler = NormalizedActionScaler(
            PolicyActionBounds(
                maximum_linear_velocity_mps=maximum_linear_velocity_mps,
                maximum_angular_velocity_rad_s=maximum_angular_velocity_rad_s,
            )
        )

    def map(self, action: ArrayLike) -> NormalizedActionMapping:
        requested: NDArray[np.float64] = np.asarray(action, dtype=np.float64)
        if requested.shape != (2,):
            raise ValueError("normalized policy action must have shape (2,)")
        if not np.all(np.isfinite(requested)):
            raise ValueError("normalized policy action must contain only finite values")
        final = np.clip(requested, -1.0, 1.0)
        normalized = NormalizedPolicyAction(
            linear=float(final[0]),
            angular=float(final[1]),
        )
        return NormalizedActionMapping(
            requested_normalized=(float(requested[0]), float(requested[1])),
            final_normalized=(normalized.linear, normalized.angular),
            policy_action=self._scaler.to_policy_action(normalized),
            clipped=bool(np.any(final != requested)),
        )


# Backwards-compatible names for the completed SAC artifacts. New solvers use
# the generic names above; no second physical action path is introduced.
SACActionMapping = NormalizedActionMapping
SACActionMapper = NormalizedActionMapper
