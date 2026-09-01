"""Fixed runtime calibration for metric range/bearing measurements.

This module intentionally has no dependency on simulator privileged state.
Parameters are fitted offline and loaded as immutable runtime values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, sin, cos
from pathlib import Path

from duckie_pomdp.perception.camera_geometry import PolarGroundPoint


@dataclass(frozen=True)
class LinearRangeCalibration:
    scale: float
    offset_m: float

    def __post_init__(self) -> None:
        if not isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("range calibration scale must be finite and positive")
        if not isfinite(self.offset_m):
            raise ValueError("range calibration offset must be finite")

    def apply(self, raw_range_m: float) -> float:
        if not isfinite(raw_range_m):
            raise ValueError("raw range must be finite")
        if raw_range_m < 0.0:
            raise ValueError("raw range cannot be negative")
        calibrated = self.scale * raw_range_m + self.offset_m
        if calibrated < 0.0:
            raise ValueError("range calibration produced an impossible negative range")
        return calibrated


@dataclass(frozen=True)
class MeasurementCalibrator:
    range_calibration: LinearRangeCalibration
    bearing_correction_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.bearing_correction_rad):
            raise ValueError("bearing correction must be finite")

    def apply(self, raw_range_m: float, raw_bearing_rad: float) -> PolarGroundPoint:
        if not isfinite(raw_bearing_rad):
            raise ValueError("raw bearing must be finite")
        return PolarGroundPoint(
            range_m=self.range_calibration.apply(raw_range_m),
            bearing_rad=wrap_angle(raw_bearing_rad + self.bearing_correction_rad),
        )


def load_measurement_calibrator(path: str | Path) -> MeasurementCalibrator:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    range_config = data["range"]
    bearing_config = data["bearing"]
    if range_config["semantics"] != "object_origin":
        raise ValueError("Version-1 runtime range semantics must be object_origin")
    if range_config["calibration_type"] != "linear":
        raise ValueError("unsupported range calibration type")
    if bearing_config["calibration_type"] != "identity":
        raise ValueError("unsupported bearing calibration type")
    return MeasurementCalibrator(
        range_calibration=LinearRangeCalibration(
            scale=float(range_config["scale"]),
            offset_m=float(range_config["offset_m"]),
        ),
        bearing_correction_rad=float(bearing_config["correction_rad"]),
    )


def wrap_angle(angle_rad: float) -> float:
    if not isfinite(angle_rad):
        raise ValueError("angle must be finite")
    return atan2(sin(angle_rad), cos(angle_rad))
