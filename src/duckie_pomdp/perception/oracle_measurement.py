"""Privileged-to-measurement boundary used before a real detector exists."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sin, cos
from pathlib import Path

import numpy as np

from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState
from duckie_pomdp.domain.state import PedestrianState, StopSignState
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import PolarMeasurementNoiseModel


class OracleMode(str, Enum):
    CLEAN = "oracle_clean"
    NOISY = "oracle_noisy"
    DROPOUT = "oracle_dropout"


@dataclass(frozen=True)
class OracleDetectionConfig:
    """Synthetic visibility/dropout assumptions, not YOLO performance."""

    minimum_range_m: float
    maximum_range_m: float
    maximum_abs_bearing_rad: float
    miss_probability: float
    false_positive_probability: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_range_m,
            self.maximum_range_m,
            self.maximum_abs_bearing_rad,
            self.miss_probability,
            self.false_positive_probability,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("oracle detection parameters must be finite")
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError("oracle observation range is invalid")
        if self.maximum_abs_bearing_rad <= 0.0:
            raise ValueError("oracle bearing domain must be positive")
        if not 0.0 <= self.miss_probability <= 1.0:
            raise ValueError("oracle miss probability must be within [0, 1]")
        if self.false_positive_probability != 0.0:
            raise ValueError(
                "Version-1 oracle false positives are disabled until the "
                "single-object contract can represent their geometry cleanly"
            )


class OracleObservationModel:
    """Generate detector-shaped measurements and terminate privileged access.

    GT is already the canonical object-origin quantity. The raw camera range
    calibration is deliberately not applied here; noisy modes model only the
    final calibrated residual measured in F5b.
    """

    def __init__(
        self,
        *,
        mode: OracleMode,
        measurement_noise: PolarMeasurementNoiseModel,
        detection: OracleDetectionConfig,
        seed: int,
    ) -> None:
        self.mode = OracleMode(mode)
        self.measurement_noise = measurement_noise
        self.detection = detection
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def observe(
        self,
        privileged: PrivilegedSimulatorState,
        object_class: ObjectClass,
    ) -> ObjectMeasurement:
        state = _object_state(privileged, object_class)
        if not state.exists:
            return ObjectMeasurement.missing(object_class)
        range_m = _required(state.range_m, "range")
        bearing_rad = _required(state.bearing_rad, "bearing")
        if not self._within_domain(range_m, bearing_rad):
            return ObjectMeasurement.missing(object_class)
        if (
            self.mode is OracleMode.DROPOUT
            and self._rng.random() < self.detection.miss_probability
        ):
            return ObjectMeasurement.missing(object_class)

        measured_range = range_m
        measured_bearing = bearing_rad
        if self.mode in (OracleMode.NOISY, OracleMode.DROPOUT):
            noise_bin = self.measurement_noise.range_bin(range_m)
            measured_range += noise_bin.residual_bias_m + self._rng.normal(
                0.0,
                noise_bin.sigma_m,
            )
            measured_bearing = wrap_angle(
                measured_bearing
                + self.measurement_noise.bearing_bias_rad
                + self._rng.normal(0.0, self.measurement_noise.bearing_sigma_rad)
            )
            # The configured observation domain is bounded away from zero, so
            # this guard is practically inactive but keeps the output physical.
            measured_range = max(0.0, measured_range)

        return ObjectMeasurement(
            object_class=object_class,
            detected=True,
            confidence=1.0,
            x_left_m=measured_range * sin(measured_bearing),
            y_forward_m=measured_range * cos(measured_bearing),
            range_m=measured_range,
            bearing_rad=wrap_angle(measured_bearing),
        )

    def _within_domain(self, range_m: float, bearing_rad: float) -> bool:
        return (
            self.detection.minimum_range_m <= range_m <= self.detection.maximum_range_m
            and abs(wrap_angle(bearing_rad)) <= self.detection.maximum_abs_bearing_rad
        )


def load_oracle_detection_config(path: str | Path) -> OracleDetectionConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)["oracle_detection"]
    return OracleDetectionConfig(
        minimum_range_m=float(data["minimum_range_m"]),
        maximum_range_m=float(data["maximum_range_m"]),
        maximum_abs_bearing_rad=float(data["maximum_abs_bearing_rad"]),
        miss_probability=float(data["miss_probability"]),
        false_positive_probability=float(data["false_positive_probability"]),
    )


def _object_state(
    privileged: PrivilegedSimulatorState,
    object_class: ObjectClass,
) -> StopSignState | PedestrianState:
    if object_class is ObjectClass.STOP_SIGN:
        return privileged.true_pomdp_state.stop_sign
    if object_class is ObjectClass.DUCKIE:
        return privileged.true_pomdp_state.pedestrian
    raise ValueError(f"unsupported oracle object class: {object_class}")


def _required(value: float | None, name: str) -> float:
    if value is None:
        raise RuntimeError(f"existing oracle object has no {name}")
    return value

