"""Runtime boundary for front RGB -> lane measurement -> lane belief."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 runtime used by Gym-Duckietown.
    import tomli as tomllib

from duckie_pomdp.belief.lane_ekf import (
    LaneBeliefUpdater,
    LaneEKFDiagnostics,
    load_lane_ekf_config,
)
from duckie_pomdp.domain.belief import LaneBelief
from duckie_pomdp.domain.measurement import LaneMeasurement
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.lane_measurement import (
    CameraLaneMeasurementEstimator,
    LaneBoundaryDiagnostics,
    load_lane_measurement_calibration,
    load_lane_perception_config,
)
from duckie_pomdp.perception.lane_rgb_model import (
    LaneRGBMeasurementEstimator,
    LaneRGBModelConfig,
)

from .f10_protocol import file_sha256
from .lane_belief_uncertainty import (
    LaneUncertaintyCalibration,
    apply_calibration_to_lane_belief,
)


def _resolve_filter_config_path(config_path: Path) -> Path:
    """Resolve the config that actually carries the frozen filter tables.

    ``configs/lane_belief_v1.toml`` is a complete config: it has
    ``[lane_perception]``/``[lane_calibration]``/``[lane_belief]`` directly
    and is returned unchanged (this is the only path v3 ever takes).

    ``configs/lane_belief_v2.toml`` is an *extension* config: it carries only
    ``[lane_belief_uncertainty]``/``[uncertainty_gate]`` plus a
    ``source_config``/``source_config_sha256`` pointer back at v1. When a v4
    protocol's ``lane_belief_config`` provenance points at v2, the filter
    tables (point extractor, bias calibration, EKF process noise -- none of
    which this task may change) are loaded from the file ``source_config``
    names, with its hash verified against ``source_config_sha256`` first.
    A hash mismatch fails loudly rather than silently loading a drifted
    filter config.
    """

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    source = raw.get("source_config")
    if source is None:
        return config_path
    source_path = (config_path.parent / str(source)).resolve()
    expected_sha256 = raw.get("source_config_sha256")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if expected_sha256 is not None and file_sha256(source_path) != str(expected_sha256):
        raise RuntimeError(
            f"{config_path} source_config_sha256 does not match {source_path} "
            "-- the frozen point-extractor/bias-calibration config has drifted"
        )
    return source_path


@dataclass(frozen=True)
class VisualLaneStep:
    measurement: LaneMeasurement
    belief: LaneBelief
    diagnostics: LaneEKFDiagnostics
    boundary_diagnostics: LaneBoundaryDiagnostics | None = None


class VisualLaneBeliefRuntime:
    """Stateful visual lane estimator with an intentionally narrow API.

    Simulator lane state, world pose, policy command, and privileged state are
    absent by construction.  Prediction consumes measured actual ego motion.
    """

    def __init__(
        self,
        projector: CalibratedGroundProjector,
        *,
        config_path: str | Path,
        uncertainty_calibration: LaneUncertaintyCalibration | None = None,
    ) -> None:
        filter_config_path = _resolve_filter_config_path(Path(config_path))
        with filter_config_path.open("rb") as handle:
            runtime_config = tomllib.load(handle)
        learned = runtime_config.get("lane_rgb_model")
        if learned is None or not bool(learned.get("enabled", False)):
            self.estimator = CameraLaneMeasurementEstimator(
                projector,
                load_lane_perception_config(filter_config_path),
            )
        else:
            checkpoint = (filter_config_path.parent / str(learned["checkpoint"])).resolve()
            self.estimator = LaneRGBMeasurementEstimator(
                LaneRGBModelConfig(
                    checkpoint=checkpoint,
                    checkpoint_sha256=str(learned["checkpoint_sha256"]),
                    device=str(learned.get("device", "auto")),
                    image_size_px=int(learned["image_size_px"]),
                    target_scales=tuple(
                        float(value) for value in learned["target_scales"]
                    ),
                    residual_sigmas=tuple(
                        float(value) for value in learned["residual_sigmas"]
                    ),
                    crop_top_fraction=float(learned.get("crop_top_fraction", 0.0)),
                )
            )
        self.calibrator = load_lane_measurement_calibration(filter_config_path)
        self.updater = LaneBeliefUpdater(load_lane_ekf_config(filter_config_path))
        # v4 fix round 1: gated behind [v4_changes].belief_uncertainty_refit
        # (see lane_belief_uncertainty.resolve_runtime_calibration). None
        # here reproduces the pre-v4 (v3) reported sigma bit-for-bit -- see
        # _report_belief below, which never mutates self.updater's state.
        self.uncertainty_calibration = uncertainty_calibration

    def reset(self, front_rgb: NDArray[np.uint8]) -> VisualLaneStep:
        self.updater.reset()
        raw_measurement, boundary_diagnostics = self.estimator.estimate_with_diagnostics(
            front_rgb
        )
        measurement = self.calibrator.apply(raw_measurement)
        if measurement.detected:
            diagnostics = self.updater.correct(measurement)
        else:
            diagnostics = LaneEKFDiagnostics(
                initialized=False,
                corrected=False,
                innovation=None,
                covariance_trace=0.0,
                validity_probability=0.0,
            )
        return VisualLaneStep(
            measurement,
            self._report_belief(),
            diagnostics,
            boundary_diagnostics,
        )

    def update(
        self,
        front_rgb: NDArray[np.uint8],
        *,
        actual_ego_motion: EgoMotion,
        dt_s: float,
    ) -> VisualLaneStep:
        raw_measurement, boundary_diagnostics = self.estimator.estimate_with_diagnostics(
            front_rgb
        )
        measurement = self.calibrator.apply(raw_measurement)
        diagnostics = self.updater.step(measurement, actual_ego_motion, dt_s)
        return VisualLaneStep(
            measurement,
            self._report_belief(),
            diagnostics,
            boundary_diagnostics,
        )

    def _report_belief(self) -> LaneBelief:
        """The reporting boundary: ``self.updater``'s internal covariance is
        read here and never written back to. ``apply_calibration_to_lane_belief``
        only ever replaces the two fields on the *returned* ``LaneBelief``
        copy -- widening the value reported downstream (to
        ``ppo_observation.PPOVisualPolicyObservation.from_belief``) can
        never feed back into ``self.updater.predict()``/``.correct()``,
        which is exactly the corruption the F9c posterior-floor comment in
        ``belief/robust_updater.py`` warns against for the identical
        pattern on the pedestrian belief.
        """

        return apply_calibration_to_lane_belief(
            self.updater.belief(), self.uncertainty_calibration
        )
