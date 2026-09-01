"""Ego-motion-compensated EKF for one pedestrian in Cartesian coordinates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import cos, isfinite, sin, sqrt
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.belief.ego_motion import compensated_transition
from duckie_pomdp.belief.polar_transform import (
    PolarBeliefMoments,
    cartesian_to_polar_moments,
)
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import PolarMeasurementNoiseModel


class MeasurementProfile(str, Enum):
    CLEAN = "clean"
    CALIBRATED_RESIDUAL = "calibrated_residual"


@dataclass(frozen=True)
class PedestrianEKFConfig:
    position_process_std_m_per_sqrt_s: float
    velocity_process_std_mps_per_sqrt_s: float
    initial_velocity_std_mps: float
    minimum_range_m: float
    covariance_floor: float
    clean_range_sigma_m: float
    clean_bearing_sigma_rad: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def with_process_noise(
        self,
        *,
        position_std: float,
        velocity_std: float,
    ) -> PedestrianEKFConfig:
        return replace(
            self,
            position_process_std_m_per_sqrt_s=position_std,
            velocity_process_std_mps_per_sqrt_s=velocity_std,
        )


@dataclass(frozen=True)
class EKFStepDiagnostics:
    initialized: bool
    corrected: bool
    innovation_range_m: float | None
    innovation_bearing_rad: float | None
    innovation_covariance: tuple[tuple[float, float], tuple[float, float]] | None
    nis: float | None
    covariance_trace: float


class PedestrianEKF:
    """State is ``[x_left, y_forward, v_left, v_forward]``.

    Velocity is pedestrian physical velocity expressed in current ego axes.
    It is not apparent relative velocity caused by ego translation/rotation.
    """

    def __init__(
        self,
        *,
        config: PedestrianEKFConfig,
        measurement_noise: PolarMeasurementNoiseModel,
        measurement_profile: MeasurementProfile,
    ) -> None:
        self.config = config
        self.measurement_noise = measurement_noise
        self.measurement_profile = MeasurementProfile(measurement_profile)
        self._state: NDArray[np.float64] | None = None
        self._covariance: NDArray[np.float64] | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> NDArray[np.float64]:
        if self._state is None:
            raise RuntimeError("EKF is not initialized")
        return self._state.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        if self._covariance is None:
            raise RuntimeError("EKF is not initialized")
        return self._covariance.copy()

    def reset(self) -> None:
        self._state = None
        self._covariance = None

    def initialize(self, measurement: ObjectMeasurement) -> None:
        measured_range, measured_bearing = _measurement_values(measurement)
        covariance_polar = self.measurement_covariance(measured_range)
        expected_bias = self.measurement_bias(measured_range)
        range_m = max(0.0, measured_range - float(expected_bias[0]))
        bearing_rad = wrap_angle(measured_bearing - float(expected_bias[1]))
        x_left = range_m * sin(bearing_rad)
        y_forward = range_m * cos(bearing_rad)
        transform = np.array(
            [
                [sin(bearing_rad), range_m * cos(bearing_rad)],
                [cos(bearing_rad), -range_m * sin(bearing_rad)],
            ],
            dtype=float,
        )
        position_covariance = transform @ covariance_polar @ transform.T
        self._state = np.array([x_left, y_forward, 0.0, 0.0], dtype=float)
        self._covariance = np.zeros((4, 4), dtype=float)
        self._covariance[:2, :2] = position_covariance
        velocity_variance = self.config.initial_velocity_std_mps**2
        self._covariance[2:, 2:] = np.eye(2, dtype=float) * velocity_variance
        self._stabilize_covariance()

    def predict(self, ego_motion: EgoMotion, dt_s: float) -> None:
        if not self.initialized:
            return
        transition, offset = compensated_transition(ego_motion, dt_s)
        position_variance = (
            self.config.position_process_std_m_per_sqrt_s**2 * dt_s
        )
        velocity_variance = (
            self.config.velocity_process_std_mps_per_sqrt_s**2 * dt_s
        )
        process_covariance = np.diag(
            [
                position_variance,
                position_variance,
                velocity_variance,
                velocity_variance,
            ]
        )
        self._state = transition @ self._state + offset
        self._covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )
        self._stabilize_covariance()

    def correct(self, measurement: ObjectMeasurement) -> EKFStepDiagnostics:
        if not self.initialized:
            self.initialize(measurement)
            return EKFStepDiagnostics(
                initialized=True,
                corrected=True,
                innovation_range_m=0.0,
                innovation_bearing_rad=0.0,
                innovation_covariance=None,
                nis=None,
                covariance_trace=float(np.trace(self._covariance)),
            )
        measured_range, measured_bearing = _measurement_values(measurement)
        predicted_measurement = measurement_function(
            self._state,
            minimum_range_m=self.config.minimum_range_m,
        )
        expected_bias = self.measurement_bias(float(predicted_measurement[0]))
        innovation = np.array(
            [
                measured_range - predicted_measurement[0] - expected_bias[0],
                wrap_angle(
                    measured_bearing - predicted_measurement[1] - expected_bias[1]
                ),
            ],
            dtype=float,
        )
        measurement_matrix = measurement_jacobian(
            self._state,
            minimum_range_m=self.config.minimum_range_m,
        )
        measurement_covariance = self.measurement_covariance(measured_range)
        innovation_covariance = (
            measurement_matrix
            @ self._covariance
            @ measurement_matrix.T
            + measurement_covariance
        )
        nis = float(
            innovation
            @ np.linalg.solve(innovation_covariance, innovation)
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ self._covariance,
        ).T
        self._state = self._state + gain @ innovation
        identity = np.eye(4, dtype=float)
        residual_operator = identity - gain @ measurement_matrix
        self._covariance = (
            residual_operator @ self._covariance @ residual_operator.T
            + gain @ measurement_covariance @ gain.T
        )
        self._stabilize_covariance()
        return EKFStepDiagnostics(
            initialized=True,
            corrected=True,
            innovation_range_m=float(innovation[0]),
            innovation_bearing_rad=float(innovation[1]),
            innovation_covariance=(
                (
                    float(innovation_covariance[0, 0]),
                    float(innovation_covariance[0, 1]),
                ),
                (
                    float(innovation_covariance[1, 0]),
                    float(innovation_covariance[1, 1]),
                ),
            ),
            nis=nis,
            covariance_trace=float(np.trace(self._covariance)),
        )

    def step(
        self,
        measurement: ObjectMeasurement,
        ego_motion: EgoMotion,
        dt_s: float,
    ) -> EKFStepDiagnostics:
        self.predict(ego_motion, dt_s)
        if measurement.detected:
            return self.correct(measurement)
        covariance_trace = (
            0.0 if self._covariance is None else float(np.trace(self._covariance))
        )
        return EKFStepDiagnostics(
            initialized=self.initialized,
            corrected=False,
            innovation_range_m=None,
            innovation_bearing_rad=None,
            innovation_covariance=None,
            nis=None,
            covariance_trace=covariance_trace,
        )

    def polar_moments(self, ego_motion: EgoMotion) -> PolarBeliefMoments:
        if self._state is None or self._covariance is None:
            raise RuntimeError("EKF is not initialized")
        return cartesian_to_polar_moments(
            self._state,
            self._covariance,
            ego_motion,
            minimum_range_m=self.config.minimum_range_m,
        )

    def measurement_covariance(self, range_m: float) -> NDArray[np.float64]:
        if self.measurement_profile is MeasurementProfile.CLEAN:
            return np.diag(
                [
                    self.config.clean_range_sigma_m**2,
                    self.config.clean_bearing_sigma_rad**2,
                ]
            ).astype(float)
        return self.measurement_noise.covariance(range_m)

    def measurement_bias(self, range_m: float) -> NDArray[np.float64]:
        if self.measurement_profile is MeasurementProfile.CLEAN:
            return np.zeros(2, dtype=float)
        return self.measurement_noise.bias(range_m)

    def _stabilize_covariance(self) -> None:
        if self._covariance is None:
            return
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
        eigenvalues = np.maximum(eigenvalues, self.config.covariance_floor)
        self._covariance = (eigenvectors * eigenvalues) @ eigenvectors.T


def measurement_function(
    state: NDArray[np.float64],
    *,
    minimum_range_m: float = 1.0e-3,
) -> NDArray[np.float64]:
    vector = _state_vector(state)
    x_left, y_forward = vector[:2]
    range_m = sqrt(x_left**2 + y_forward**2)
    return np.array([range_m, np.arctan2(x_left, y_forward)], dtype=float)


def measurement_jacobian(
    state: NDArray[np.float64],
    *,
    minimum_range_m: float = 1.0e-3,
) -> NDArray[np.float64]:
    vector = _state_vector(state)
    x_left, y_forward = vector[:2]
    range_squared = max(x_left**2 + y_forward**2, minimum_range_m**2)
    range_m = sqrt(range_squared)
    return np.array(
        [
            [x_left / range_m, y_forward / range_m, 0.0, 0.0],
            [y_forward / range_squared, -x_left / range_squared, 0.0, 0.0],
        ],
        dtype=float,
    )


def load_pedestrian_ekf_config(path: str | Path) -> PedestrianEKFConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)["ekf"]
    return PedestrianEKFConfig(
        position_process_std_m_per_sqrt_s=float(
            data["position_process_std_m_per_sqrt_s"]
        ),
        velocity_process_std_mps_per_sqrt_s=float(
            data["velocity_process_std_mps_per_sqrt_s"]
        ),
        initial_velocity_std_mps=float(data["initial_velocity_std_mps"]),
        minimum_range_m=float(data["minimum_range_m"]),
        covariance_floor=float(data["covariance_floor"]),
        clean_range_sigma_m=float(data["clean_range_sigma_m"]),
        clean_bearing_sigma_rad=float(data["clean_bearing_sigma_rad"]),
    )


def _measurement_values(measurement: ObjectMeasurement) -> tuple[float, float]:
    if not measurement.detected:
        raise ValueError("EKF correction requires a detected measurement")
    if measurement.range_m is None or measurement.bearing_rad is None:
        raise ValueError("detected measurement has incomplete polar geometry")
    return measurement.range_m, measurement.bearing_rad


def _state_vector(state: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = np.asarray(state, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError("EKF state must be a finite length-4 vector")
    return vector
