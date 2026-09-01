"""Frozen F9c bias correction as an explicit runtime stage.

Runtime order: ``z_raw -> FROZEN F9c BIAS CORRECTION -> z_corr -> association
-> gate -> EKF``. Bias correction must run before association, and it must
never look at anything but the raw measurement it is correcting: association
compares a *corrected* measurement against the predicted innovation, so
feeding this stage a predicted or ground-truth range would inject the bias
into the wrong place and, if it consumed the filter's own prediction, would
close a feedback loop back into the estimator it is meant to protect.

This mirrors the shape of ``AdditiveMeasurementBias`` in
``perception/f9_pipeline.py``, which is left untouched because it is the
frozen class Baseline A runs. ``FrozenBiasCorrection`` is Robust B's
independent, separately fitted counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Mapping

from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.perception.measurement_calibration import wrap_angle

_RANGE_BINS = ("near", "medium", "far")


@dataclass(frozen=True)
class FrozenBiasCorrection:
    model: str
    range_bias_m: float
    bearing_bias_rad: float
    range_bin_bias_m: Mapping[str, float] | None
    near_max_m: float
    medium_max_m: float

    def __post_init__(self) -> None:
        if self.model not in {"identity", "global_additive", "per_range_bin"}:
            raise ValueError(
                "model must be one of 'identity', 'global_additive', "
                "'per_range_bin'"
            )
        if self.model == "per_range_bin":
            if self.range_bin_bias_m is None or set(
                self.range_bin_bias_m
            ) != set(_RANGE_BINS):
                raise ValueError(
                    "per_range_bin model requires range_bin_bias_m with "
                    "exactly the keys near, medium, far"
                )

    @classmethod
    def identity(cls) -> FrozenBiasCorrection:
        return cls(
            model="identity",
            range_bias_m=0.0,
            bearing_bias_rad=0.0,
            range_bin_bias_m=None,
            near_max_m=0.55,
            medium_max_m=0.80,
        )

    @classmethod
    def from_config(cls, data: Mapping) -> FrozenBiasCorrection:
        return cls(
            model=data["model"],
            range_bias_m=data["range_bias_m"],
            bearing_bias_rad=data["bearing_bias_rad"],
            range_bin_bias_m=data.get("range_bin_bias_m"),
            near_max_m=data["near_max_m"],
            medium_max_m=data["medium_max_m"],
        )

    def _select_range_bias(self, measured_range_m: float) -> float:
        """Pick the bin bias using the measured range.

        This must always be the raw, measured range of the incoming
        measurement -- never a predicted or ground-truth range -- or the
        correction would depend on the filter's own output and close a
        feedback loop.
        """

        assert self.range_bin_bias_m is not None
        if measured_range_m <= self.near_max_m:
            bin_name = "near"
        elif measured_range_m <= self.medium_max_m:
            bin_name = "medium"
        else:
            bin_name = "far"
        return self.range_bin_bias_m[bin_name]

    def correct(self, measurement: ObjectMeasurement) -> ObjectMeasurement:
        """Subtract the frozen F9c bias without accepting any ground truth.

        The `model == "identity"` no-op below is unconditional on purpose:
        `from_config` can construct `model="identity"` carrying stray nonzero
        bias fields without ever passing through the `identity()` classmethod,
        so identity must mean no-op regardless of what those fields contain.
        """

        if not measurement.detected:
            return measurement
        if self.model == "identity":
            return measurement
        if measurement.range_m is None or measurement.bearing_rad is None:
            raise RuntimeError("detected measurement has incomplete polar geometry")

        if self.model == "per_range_bin":
            range_bias_m = self._select_range_bias(measurement.range_m)
        else:
            range_bias_m = self.range_bias_m

        corrected_range = max(0.0, measurement.range_m - range_bias_m)
        corrected_bearing = wrap_angle(
            measurement.bearing_rad - self.bearing_bias_rad
        )
        return ObjectMeasurement(
            object_class=measurement.object_class,
            detected=True,
            confidence=measurement.confidence,
            x_left_m=corrected_range * sin(corrected_bearing),
            y_forward_m=corrected_range * cos(corrected_bearing),
            range_m=corrected_range,
            bearing_rad=corrected_bearing,
        )
