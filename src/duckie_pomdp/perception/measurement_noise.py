"""Validated polar measurement residual model shared by F6 and F7."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, isnan
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RangeNoiseBin:
    name: str
    minimum_m: float
    maximum_m: float
    residual_bias_m: float
    sigma_m: float

    def __post_init__(self) -> None:
        values = (self.minimum_m, self.residual_bias_m, self.sigma_m)
        if not all(isfinite(value) for value in values):
            raise ValueError("finite range-noise parameters are required")
        if isnan(self.maximum_m):
            raise ValueError("range-noise maximum cannot be NaN")
        if self.minimum_m < 0.0 or self.maximum_m <= self.minimum_m:
            raise ValueError("range-noise bin bounds are invalid")
        if self.sigma_m <= 0.0:
            raise ValueError("range-noise sigma must be positive")

    def contains(self, range_m: float) -> bool:
        return self.minimum_m <= range_m < self.maximum_m


@dataclass(frozen=True)
class PolarMeasurementNoiseModel:
    """Fixed F5b residual model for final calibrated measurements.

    Bearing Gaussianity is a provisional Version-1 approximation. The source
    data is skewed and heavy-tailed, so this class does not claim otherwise.
    """

    range_bins: tuple[RangeNoiseBin, ...]
    bearing_bias_rad: float
    bearing_sigma_rad: float
    covariance_structure: str

    def __post_init__(self) -> None:
        if not self.range_bins:
            raise ValueError("at least one range-noise bin is required")
        if not isfinite(self.bearing_bias_rad):
            raise ValueError("bearing bias must be finite")
        if not isfinite(self.bearing_sigma_rad) or self.bearing_sigma_rad <= 0.0:
            raise ValueError("bearing sigma must be finite and positive")
        if self.covariance_structure != "diagonal":
            raise ValueError("Version 1 requires diagonal measurement covariance")

    def range_bin(self, range_m: float) -> RangeNoiseBin:
        if not isfinite(range_m) or range_m < 0.0:
            raise ValueError("range lookup requires a finite nonnegative range")
        for noise_bin in self.range_bins:
            if noise_bin.contains(range_m):
                return noise_bin
        raise ValueError(f"range {range_m} lies outside configured noise bins")

    def bias(self, range_m: float) -> NDArray[np.float64]:
        noise_bin = self.range_bin(range_m)
        return np.array(
            [noise_bin.residual_bias_m, self.bearing_bias_rad],
            dtype=float,
        )

    def covariance(self, range_m: float) -> NDArray[np.float64]:
        noise_bin = self.range_bin(range_m)
        return np.diag(
            [noise_bin.sigma_m**2, self.bearing_sigma_rad**2]
        ).astype(float)


def load_polar_measurement_noise(
    path: str | Path,
) -> PolarMeasurementNoiseModel:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    if data["range"]["semantics"] != "object_origin":
        raise ValueError("Version-1 measurement range must target object_origin")
    range_noise = data["range_noise"]
    if range_noise["model"] != "distance_bins":
        raise ValueError("unsupported range-noise model")
    covariance = data["covariance"]
    if bool(covariance["use_off_diagonal"]):
        raise ValueError("Version 1 does not use off-diagonal measurement noise")

    bins: list[RangeNoiseBin] = []
    for name in ("near", "medium", "far"):
        item = range_noise[name]
        maximum = item["max_m"]
        bins.append(
            RangeNoiseBin(
                name=name,
                minimum_m=float(item["min_m"]),
                maximum_m=inf if maximum is None else float(maximum),
                residual_bias_m=float(item["residual_bias_m"]),
                sigma_m=float(item["sigma_m"]),
            )
        )
    return PolarMeasurementNoiseModel(
        range_bins=tuple(bins),
        bearing_bias_rad=float(data["bearing"]["residual_bias_rad"]),
        bearing_sigma_rad=float(data["bearing"]["sigma_rad"]),
        covariance_structure=str(covariance["structure"]),
    )
