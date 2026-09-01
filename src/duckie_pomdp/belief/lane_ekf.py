"""Camera-lane EKF using actual ego motion for prediction.

The state is ``[d, phi, kappa]`` under the project conventions:

* ``d > 0``: ego is left of lane centre;
* ``phi > 0``: Gym-Duckietown's signed lane heading error;
* ``kappa > 0``: lane curves counter-clockwise/left.

No simulator pose or lane geometry is accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.belief import LaneBelief
from duckie_pomdp.domain.measurement import LaneMeasurement
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.perception.measurement_calibration import wrap_angle


@dataclass(frozen=True)
class LaneEKFConfig:
    lateral_process_std_m_per_sqrt_s: float
    heading_process_std_rad_per_sqrt_s: float
    curvature_process_std_inv_m_per_sqrt_s: float
    initial_lateral_std_m: float
    initial_heading_std_rad: float
    initial_curvature_std_inv_m: float
    covariance_floor: float
    detection_validity_gain: float
    miss_validity_decay: float

    def __post_init__(self) -> None:
        positive = (
            self.lateral_process_std_m_per_sqrt_s,
            self.heading_process_std_rad_per_sqrt_s,
            self.curvature_process_std_inv_m_per_sqrt_s,
            self.initial_lateral_std_m,
            self.initial_heading_std_rad,
            self.initial_curvature_std_inv_m,
            self.covariance_floor,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("lane EKF noise/initial values must be finite and positive")
        for name, value in (
            ("detection_validity_gain", self.detection_validity_gain),
            ("miss_validity_decay", self.miss_validity_decay),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True)
class LaneEKFDiagnostics:
    initialized: bool
    corrected: bool
    innovation: tuple[float, float, float] | None
    covariance_trace: float
    validity_probability: float


class LaneBeliefUpdater:
    """Three-state EKF for camera-derived lane pose and curvature."""

    def __init__(self, config: LaneEKFConfig) -> None:
        self.config = config
        self._state: NDArray[np.float64] | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._validity_probability = 0.0

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> NDArray[np.float64]:
        if self._state is None:
            raise RuntimeError("lane EKF is not initialized")
        return self._state.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        if self._covariance is None:
            raise RuntimeError("lane EKF is not initialized")
        return self._covariance.copy()

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._validity_probability = 0.0

    def initialize(self, measurement: LaneMeasurement) -> None:
        values, covariance = _measurement_vector_and_covariance(measurement)
        self._state = values
        initial_variances = np.square(
            np.asarray(
                [
                    self.config.initial_lateral_std_m,
                    self.config.initial_heading_std_rad,
                    self.config.initial_curvature_std_inv_m,
                ],
                dtype=float,
            )
        )
        self._covariance = np.diag(
            np.maximum(np.diag(covariance), initial_variances)
        )
        self._validity_probability = self.config.detection_validity_gain
        self._stabilize_covariance()

    def predict(self, ego_motion: EgoMotion, dt_s: float) -> None:
        if not self.initialized:
            return
        if not isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("lane EKF dt_s must be finite and positive")
        assert self._state is not None and self._covariance is not None
        lateral, heading, curvature = self._state
        velocity = ego_motion.linear_velocity_mps
        yaw_rate = ego_motion.yaw_rate_rad_s
        transition = np.array(
            [
                [1.0, velocity * cos(heading) * dt_s, 0.0],
                [0.0, 1.0, velocity * dt_s],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self._state = np.array(
            [
                lateral + velocity * sin(heading) * dt_s,
                wrap_angle(heading + (velocity * curvature - yaw_rate) * dt_s),
                curvature,
            ],
            dtype=float,
        )
        process_variance = dt_s * np.square(
            np.asarray(
                [
                    self.config.lateral_process_std_m_per_sqrt_s,
                    self.config.heading_process_std_rad_per_sqrt_s,
                    self.config.curvature_process_std_inv_m_per_sqrt_s,
                ],
                dtype=float,
            )
        )
        self._covariance = (
            transition @ self._covariance @ transition.T
            + np.diag(process_variance)
        )
        self._stabilize_covariance()

    def correct(self, measurement: LaneMeasurement) -> LaneEKFDiagnostics:
        if not measurement.detected:
            raise ValueError("cannot correct lane EKF with a missing measurement")
        if not self.initialized:
            self.initialize(measurement)
            return self._diagnostics(corrected=True, innovation=(0.0, 0.0, 0.0))
        assert self._state is not None and self._covariance is not None
        observed, measurement_covariance = _measurement_vector_and_covariance(measurement)
        innovation = observed - self._state
        innovation[1] = wrap_angle(float(innovation[1]))
        innovation_covariance = self._covariance + measurement_covariance
        gain = np.linalg.solve(innovation_covariance, self._covariance).T
        self._state = self._state + gain @ innovation
        self._state[1] = wrap_angle(float(self._state[1]))
        identity = np.eye(3, dtype=float)
        residual = identity - gain
        self._covariance = (
            residual @ self._covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        self._validity_probability += self.config.detection_validity_gain * (
            1.0 - self._validity_probability
        )
        self._stabilize_covariance()
        return self._diagnostics(
            corrected=True,
            innovation=tuple(float(value) for value in innovation),
        )

    def step(
        self,
        measurement: LaneMeasurement,
        ego_motion: EgoMotion,
        dt_s: float,
    ) -> LaneEKFDiagnostics:
        self.predict(ego_motion, dt_s)
        if measurement.detected:
            return self.correct(measurement)
        self._validity_probability *= self.config.miss_validity_decay
        return self._diagnostics(corrected=False, innovation=None)

    def belief(self) -> LaneBelief:
        if self._state is None or self._covariance is None:
            return LaneBelief(
                validity_probability=0.0,
                lateral_error_mean_m=0.0,
                lateral_error_std_m=self.config.initial_lateral_std_m,
                heading_error_mean_rad=0.0,
                heading_error_std_rad=self.config.initial_heading_std_rad,
                curvature_mean_inv_m=0.0,
                curvature_std_inv_m=self.config.initial_curvature_std_inv_m,
            )
        standard_deviations = np.sqrt(np.maximum(np.diag(self._covariance), 0.0))
        return LaneBelief(
            validity_probability=float(np.clip(self._validity_probability, 0.0, 1.0)),
            lateral_error_mean_m=float(self._state[0]),
            lateral_error_std_m=float(standard_deviations[0]),
            heading_error_mean_rad=float(self._state[1]),
            heading_error_std_rad=float(standard_deviations[1]),
            curvature_mean_inv_m=float(self._state[2]),
            curvature_std_inv_m=float(standard_deviations[2]),
        )

    def _diagnostics(
        self,
        *,
        corrected: bool,
        innovation: tuple[float, float, float] | None,
    ) -> LaneEKFDiagnostics:
        covariance_trace = (
            0.0 if self._covariance is None else float(np.trace(self._covariance))
        )
        return LaneEKFDiagnostics(
            initialized=self.initialized,
            corrected=corrected,
            innovation=innovation,
            covariance_trace=covariance_trace,
            validity_probability=float(self._validity_probability),
        )

    def _stabilize_covariance(self) -> None:
        if self._covariance is None:
            return
        covariance = 0.5 * (self._covariance + self._covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, self.config.covariance_floor)
        self._covariance = (eigenvectors * eigenvalues) @ eigenvectors.T


def lane_motion_function(
    state: NDArray[np.float64],
    ego_motion: EgoMotion,
    dt_s: float,
) -> NDArray[np.float64]:
    """Pure form of the lane prediction, exposed for numerical tests."""

    vector = np.asarray(state, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("lane state must be a finite three-vector")
    if not isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    lateral, heading, curvature = vector
    velocity = ego_motion.linear_velocity_mps
    return np.array(
        [
            lateral + velocity * sin(heading) * dt_s,
            wrap_angle(
                heading
                + (velocity * curvature - ego_motion.yaw_rate_rad_s) * dt_s
            ),
            curvature,
        ],
        dtype=float,
    )


def lane_motion_jacobian(
    state: NDArray[np.float64],
    ego_motion: EgoMotion,
    dt_s: float,
) -> NDArray[np.float64]:
    vector = np.asarray(state, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("lane state must be a finite three-vector")
    if not isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    heading = float(vector[1])
    velocity = ego_motion.linear_velocity_mps
    return np.array(
        [
            [1.0, velocity * cos(heading) * dt_s, 0.0],
            [0.0, 1.0, velocity * dt_s],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def load_lane_ekf_config(path: str | Path) -> LaneEKFConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)["lane_belief"]
    return LaneEKFConfig(
        lateral_process_std_m_per_sqrt_s=float(raw["lateral_process_std_m_per_sqrt_s"]),
        heading_process_std_rad_per_sqrt_s=float(raw["heading_process_std_rad_per_sqrt_s"]),
        curvature_process_std_inv_m_per_sqrt_s=float(
            raw["curvature_process_std_inv_m_per_sqrt_s"]
        ),
        initial_lateral_std_m=float(raw["initial_lateral_std_m"]),
        initial_heading_std_rad=float(raw["initial_heading_std_rad"]),
        initial_curvature_std_inv_m=float(raw["initial_curvature_std_inv_m"]),
        covariance_floor=float(raw["covariance_floor"]),
        detection_validity_gain=float(raw["detection_validity_gain"]),
        miss_validity_decay=float(raw["miss_validity_decay"]),
    )


def _measurement_vector_and_covariance(
    measurement: LaneMeasurement,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not measurement.detected:
        raise ValueError("lane measurement is missing")
    assert measurement.lateral_error_m is not None
    assert measurement.lateral_error_std_m is not None
    assert measurement.heading_error_rad is not None
    assert measurement.heading_error_std_rad is not None
    assert measurement.curvature_inv_m is not None
    assert measurement.curvature_std_inv_m is not None
    values = np.array(
        [
            measurement.lateral_error_m,
            measurement.heading_error_rad,
            measurement.curvature_inv_m,
        ],
        dtype=float,
    )
    covariance = np.diag(
        np.square(
            [
                measurement.lateral_error_std_m,
                measurement.heading_error_std_rad,
                measurement.curvature_std_inv_m,
            ]
        )
    ).astype(float)
    return values, covariance
