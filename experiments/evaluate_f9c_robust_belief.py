"""Gate F9c final evaluation: Baseline A vs Robust B on seeds 7101-7104.

Structured on ``experiments/validate_f9_yolo_ekf.py``: one CSV row per
rendered frame, error-case images scored on the headline (Robust B) system,
metrics assembled at the end via ``evaluation.f9c_belief.summarize_f9c``.

**Two systems, one inference pass per frame** (mirrors how F9b ran raw and
corrected in parallel):

    Baseline A: corrected YOLO (F9b frozen bias) -> highest-confidence
                selection -> frozen F7 EKF (``PedestrianBeliefUpdater``,
                UNMODIFIED) -> frozen existence
    Robust  B:  candidates -> F9c bias -> association -> gate ->
                lambda-inflated R -> frozen F7 EKF -> P_D^eff + I8 floor
                existence -> posterior floor
                (``RobustPedestrianBeliefUpdater``)

**Two ways to produce a row set, ONE shared row-building path.**

The 2026-08 final-evaluation render (``collect_final_rows``) completed all
40 episodes / 3328 frames exactly once -- seeds 7101-7104 may not be
rendered again. That run's own post-processing crashed on a bug in
``f9c_belief.py`` (see ``task-11-report.md``) before the CSV/metrics
artifacts were written, but the runtime cache and evaluation truth it wrote
first are intact and hash-verified. ``replay_from_cache`` reconstructs the
same row set from those two files alone -- **no simulator, no detector, no
GPU** -- so the missing artifacts can be produced without a second render.

Both entry points reduce every frame to the same three shared, pure
functions, so the two paths cannot silently diverge on the fragile parts
(highest-confidence tie-breaking, ``duplicate_selection``,
``selected_correct_iou50`` IoU matching, ``distance_bin``/``fov_region``):

    _selection_from_render_result(result)      -\\
    _selection_from_cache_frame(cache_frame)    -+-> _step_both_systems(...) -> build_row(...)

``tests/test_evaluate_f9c_robust_belief.py`` is the guard that makes this
trustworthy: it feeds one synthetic detector output through both
``_selection_from_render_result`` and (via a hand-built
``RuntimeCacheFrame``) ``_selection_from_cache_frame``, and asserts
``build_row`` produces identical rows either way.

Both updaters are stepped from the SAME per-frame candidate set -- one
detector inference per frame in the render path, one cached candidate set
per frame in the replay path -- and BOTH are stepped before ground truth is
read (``integration.privileged.read()`` in the render path; the
already-recorded ``TruthFrame`` in the replay path), preserving the
runtime/privileged boundary either way.

**Invariant I5.** The runtime cache (``f9c_runtime_cache.write_runtime_cache``)
is built from ``result.duckie_candidates`` -- the RAW, pre-bias candidates
straight off ``observe()`` -- captured before either bias stage runs. Do not
move this capture below the bias-correction lines below it.

**A disclosed limitation of the replay path**: ``PredictedObservabilityModel``
needs a camera calibration to classify the EKF's predicted state into
CENTER/MID_FOV/EDGE_FOV/OUTSIDE_DOMAIN for Robust B's P_D^eff and
existence-informativeness. Under ``domain_randomization = true``,
gym-duckietown perturbs the camera's height/angle/FOV per episode
(``camera_height`` +-8%, ``camera_angle``/``camera_fov_y`` +-20%) -- and that
per-episode value is not, and per invariant I5 must not be, part of the
runtime-cache schema (I5 stores only raw candidate/ego-motion/GT data, not
camera intrinsics). ``replay_from_cache`` therefore classifies observability
using gym-duckietown's NOMINAL (unrandomized) camera constants -- the same
values ``tests/test_f9c_robust_updater.py``'s ``observability_model()``
fixture already uses for simulator-free tests. This affects ONLY Robust B's
existence-filter trajectory (``existence_probability``/``track_active``/
``frame_mode``/``track_deleted``) -- never the EKF kinematic state
(range/bearing/std/NIS/innovation), never association/gate decisions (both
are computed purely from ``H P^- H^T + lambda R`` and the candidates, with
no camera dependency at all), and never Baseline A, none of which read
observability classification anywhere. See the module-level constant
``_NOMINAL_CAMERA_CALIBRATION`` and the report for the full disclosure.

The detector and the simulator are constructed only inside
``collect_final_rows`` -- never in ``replay_from_cache``, never at module
scope, never inside a metrics-only helper -- so the replay path is
structurally guaranteed to perform no inference and no render.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from math import cos, sin
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.existence_filter import ExistenceFilterConfig
from duckie_pomdp.belief.measurement_association import CandidateMeasurement
from duckie_pomdp.belief.observability import PredictedObservabilityModel
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    load_pedestrian_ekf_config,
    measurement_function,
)
from duckie_pomdp.belief.robust_updater import (
    RobustObservationConfig,
    RobustObservationSwitches,
    RobustPedestrianBeliefUpdater,
)
from duckie_pomdp.belief.updater import (
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.dataset.annotations import assess_silhouette
from duckie_pomdp.dataset.config import load_dataset_config
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass, yolo_class_id
from duckie_pomdp.domain.measurement import ObjectMeasurement, PerceptionObservation
from duckie_pomdp.domain.observation import EgoObservation, RoadMeasurement
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.evaluation.f9c_belief import summarize_f9c
from duckie_pomdp.evaluation.f9c_calibration import (
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.evaluation.f9c_protocol import F9cProtocol, load_f9c_protocol
from duckie_pomdp.evaluation.f9c_runtime_cache import (
    RuntimeCacheFrame,
    TruthFrame,
    read_evaluation_truth,
    read_runtime_cache,
    write_evaluation_truth,
    write_runtime_cache,
)
from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.yolo_detection import intersection_over_union
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector, CameraCalibration
from duckie_pomdp.perception.f9_pipeline import (
    AdditiveMeasurementBias,
    YoloPedestrianMeasurementPipeline,
    select_single_duckie,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
    wrap_angle,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]

# gym-duckietown's nominal (unrandomized) camera constants -- see the module
# docstring's "disclosed limitation" section. Source:
# gym_duckietown/simulator.py: CAMERA_FOV_Y=75, CAMERA_FLOOR_DIST=0.108,
# CAMERA_ANGLE=19.15, CAMERA_FORWARD_DIST=0.066 -- independently confirmed
# against the package installed at
# /home/pannntastic/aivnv/duckie/.venv, and matching the constants
# tests/test_f9c_robust_updater.py's observability_model() fixture already
# uses for simulator-free tests.
_NOMINAL_CAMERA_CALIBRATION = CameraCalibration(
    image_width_px=640,
    image_height_px=480,
    vertical_fov_deg=75.0,
    camera_height_m=0.108,
    camera_pitch_deg=19.15,
    camera_forward_offset_m=0.066,
)

# The hashes recorded in task-11-report.md from the ORIGINAL 2026-08
# final-evaluation render (seeds 7101-7104, 3328 frames, 40 episodes, zero
# early terminations). replay_from_cache refuses to run against any file
# that does not match these -- a silently regenerated or corrupted cache
# must never be replayed as if it were the run that actually happened.
EXPECTED_RUNTIME_CACHE_SHA256 = (
    "fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d"
)
EXPECTED_EVALUATION_TRUTH_SHA256 = (
    "26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9"
)


# ---------------------------------------------------------------------------
# Plumbing config not exposed by F9cProtocol (which exists for the
# freeze-boundary guards, not rendering plumbing -- see f9c_protocol.py's
# module docstring). Mirrors experiments/calibrate_f9c_robust_belief.py's
# _ExperimentSettings, duplicated rather than imported: an experiment script
# importing a private class from a sibling experiment script is more
# surprising than the small duplication, and this task must not modify the
# calibration script.
# ---------------------------------------------------------------------------


class _ExperimentSettings:
    def __init__(self, config_path: Path) -> None:
        with config_path.open("rb") as stream:
            data: dict[str, Any] = tomllib.load(stream)

        def relative(value: str) -> Path:
            return (config_path.parent / value).resolve()

        provenance = data["provenance"]
        detector = data["detector"]
        simulator = data["simulator"]
        perturbation = data["trajectory_perturbation"]
        calibration_protocol = data["calibration_protocol"]

        self.scenario_path = relative(str(provenance["scenario"]))
        self.annotation_config_path = relative(str(provenance["annotation_rules"]))
        self.detector_confidence_threshold = float(detector["confidence_threshold"])
        self.detector_nms_iou_threshold = float(detector["nms_iou_threshold"])
        self.detector_image_size = int(detector["image_size"])
        self.detector_device = detector["device"]
        self.detector_max_detections = int(detector["max_detections"])
        self.image_width_px = int(simulator["image_width_px"])
        self.image_height_px = int(simulator["image_height_px"])
        self.domain_randomization = bool(simulator["domain_randomization"])
        self.dynamics_randomization = bool(simulator["dynamics_randomization"])
        self.start_x_range_m = _bounds(perturbation["start_x_range_m"])
        self.lateral_offset_range_m = _bounds(perturbation["lateral_offset_range_m"])
        self.heading_range_rad = _bounds(perturbation["heading_range_rad"])
        self.matching_iou_threshold = float(calibration_protocol["matching_iou_threshold"])
        self.near_max_m = float(calibration_protocol["distance_bin_near_max_m"])
        self.medium_max_m = float(calibration_protocol["distance_bin_medium_max_m"])
        self.error_cases_per_category = 4


def _bounds(values: list[float]) -> tuple[float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("bounds must be an ordered pair")
    return bounds


def _scenario_for(settings: _ExperimentSettings, spec, seed: int, scenario_index: int):
    """Identical RNG scheme to calibrate_f9c_robust_belief.py's own
    _scenario_for (deliberately -- final-evaluation seeds must go through
    the same perturbation shape as calibration seeds, just with different
    seed values, or the two splits would not be comparable renders of the
    same scenario matrix)."""

    base = load_scenario(settings.scenario_path).with_pedestrian_mode(spec.pedestrian_mode)
    rng = np.random.default_rng(seed + 100_003 * scenario_index)
    base_pose = base.ego_start_pose_m
    return replace(
        base,
        seed=seed,
        ego_start_pose_m=(
            float(rng.uniform(*settings.start_x_range_m)) + spec.ego_start_x_offset_m,
            base_pose[1],
            base_pose[2] + float(rng.uniform(*settings.lateral_offset_range_m)),
        ),
        ego_heading_rad=(
            float(rng.uniform(*settings.heading_range_rad)) + spec.ego_heading_offset_rad
        ),
    )


def _bbox_key(box: BoundingBox) -> tuple[int, int, int, int]:
    return (
        int(round(box.x_min_px)),
        int(round(box.y_min_px)),
        int(round(box.x_max_px)),
        int(round(box.y_max_px)),
    )


def _fov_region(box: BoundingBox | None, image_width_px: int) -> str:
    if box is None:
        return "outside"
    center = 0.5 * (box.x_min_px + box.x_max_px)
    normalized = abs(center - 0.5 * image_width_px) / (0.5 * image_width_px)
    if normalized < 1.0 / 3.0:
        return "center"
    if normalized < 2.0 / 3.0:
        return "mid_fov"
    return "edge_fov"


def _distance_bin(range_m: float, near_max_m: float, medium_max_m: float) -> str:
    if range_m < near_max_m:
        return "near"
    if range_m < medium_max_m:
        return "medium"
    return "far"


def _optional(value: Any) -> Any:
    return "" if value is None else value


def _box_fields(prefix: str, box: BoundingBox | None) -> dict[str, object]:
    return {
        f"{prefix}_x1": None if box is None else box.x_min_px,
        f"{prefix}_y1": None if box is None else box.y_min_px,
        f"{prefix}_x2": None if box is None else box.x_max_px,
        f"{prefix}_y2": None if box is None else box.y_max_px,
    }


def _perception(
    ego: EgoObservation, road: RoadMeasurement | None, pedestrian: ObjectMeasurement
) -> PerceptionObservation:
    return PerceptionObservation(
        ego=ego,
        road=road,
        stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
        pedestrian=pedestrian,
    )


def _baseline_belief_fields(prefix: str, updater: PedestrianBeliefUpdater, belief) -> dict[str, object]:
    initialized = updater.ekf.initialized
    pedestrian = belief.pedestrian
    diagnostics = updater.last_diagnostics.ekf
    state = updater.ekf.state if initialized else None
    return {
        f"{prefix}_belief_initialized": initialized,
        f"{prefix}_belief_range_m": pedestrian.range_mean_m,
        f"{prefix}_belief_range_std_m": pedestrian.range_std_m,
        f"{prefix}_belief_bearing_rad": pedestrian.bearing_mean_rad,
        f"{prefix}_belief_bearing_std_rad": pedestrian.bearing_std_rad,
        f"{prefix}_belief_range_rate_mps": pedestrian.radial_velocity_mean_mps,
        f"{prefix}_belief_range_rate_std_mps": pedestrian.radial_velocity_std_mps,
        f"{prefix}_belief_bearing_rate_rad_s": pedestrian.bearing_rate_mean_rad_s,
        f"{prefix}_belief_bearing_rate_std_rad_s": pedestrian.bearing_rate_std_rad_s,
        f"{prefix}_existence_probability": pedestrian.existence_probability,
        f"{prefix}_internal_x_left_m": None if state is None else float(state[0]),
        f"{prefix}_internal_y_forward_m": None if state is None else float(state[1]),
        f"{prefix}_innovation_range_m": diagnostics.innovation_range_m,
        f"{prefix}_innovation_bearing_rad": diagnostics.innovation_bearing_rad,
        f"{prefix}_nis": diagnostics.nis,
        f"{prefix}_covariance_trace": diagnostics.covariance_trace,
    }


def _robust_belief_fields(
    prefix: str,
    updater: RobustPedestrianBeliefUpdater,
    belief,
    record,
) -> dict[str, object]:
    initialized = updater.ekf.initialized
    pedestrian = belief.pedestrian
    diagnostics = updater.last_ekf_diagnostics
    state = updater.ekf.state if initialized else None
    return {
        f"{prefix}_belief_initialized": initialized,
        f"{prefix}_belief_range_m": pedestrian.range_mean_m,
        f"{prefix}_belief_range_std_m": pedestrian.range_std_m,
        f"{prefix}_belief_bearing_rad": pedestrian.bearing_mean_rad,
        f"{prefix}_belief_bearing_std_rad": pedestrian.bearing_std_rad,
        f"{prefix}_belief_range_rate_mps": pedestrian.radial_velocity_mean_mps,
        f"{prefix}_belief_range_rate_std_mps": pedestrian.radial_velocity_std_mps,
        f"{prefix}_belief_bearing_rate_rad_s": pedestrian.bearing_rate_mean_rad_s,
        f"{prefix}_belief_bearing_rate_std_rad_s": pedestrian.bearing_rate_std_rad_s,
        f"{prefix}_existence_probability": pedestrian.existence_probability,
        f"{prefix}_internal_x_left_m": None if state is None else float(state[0]),
        f"{prefix}_internal_y_forward_m": None if state is None else float(state[1]),
        f"{prefix}_innovation_range_m": None if diagnostics is None else diagnostics.innovation_range_m,
        f"{prefix}_innovation_bearing_rad": None if diagnostics is None else diagnostics.innovation_bearing_rad,
        f"{prefix}_nis": None if diagnostics is None else diagnostics.nis,
        f"{prefix}_covariance_trace": None if diagnostics is None else diagnostics.covariance_trace,
        f"{prefix}_observability_class": record.observability_class.value,
        f"{prefix}_effective_detection_probability": record.effective_detection_probability,
        f"{prefix}_observation_informative": record.observation_informative,
        f"{prefix}_frame_mode": record.frame_mode,
        f"{prefix}_track_active": record.track_active,
        f"{prefix}_track_deleted": record.track_deleted,
        f"{prefix}_association_mode": record.association.mode,
        f"{prefix}_association_differed_from_highest_confidence": (
            record.association.differed_from_highest_confidence
        ),
    }


def _predicted_polar_after_update(
    updater: RobustPedestrianBeliefUpdater, minimum_range_m: float
) -> tuple[float | None, float | None]:
    """Best-effort recovery of this frame's predicted (pre-correction) polar
    measurement, for the gate log only, on frames the coordinator did NOT
    correct (rejected candidate, or no candidate). Valid only when the
    track survived the frame with x_hat_minus untouched by ekf.correct() --
    i.e. the coordinator did not reset() the EKF this same frame (a
    same-frame existence-driven deletion). Returns (None, None) rather than
    guessing when that state is unavailable. Uses only PUBLIC state
    (``updater.ekf.state``) and the pure, already-public
    ``measurement_function`` -- no frozen module is touched or
    reimplemented."""

    if not updater.ekf.initialized:
        return None, None
    try:
        predicted = measurement_function(updater.ekf.state, minimum_range_m=minimum_range_m)
    except Exception:
        return None, None
    return float(predicted[0]), float(predicted[1])


def _gate_log_fields(
    prefix: str,
    updater: RobustPedestrianBeliefUpdater,
    record,
    minimum_range_m: float,
) -> dict[str, object]:
    """Per gated measurement: frame/confidence/predicted/measurement/
    innovation/NIS/threshold/decision, as required by the brief. Present
    (non-None) only when association actually selected a candidate this
    frame in "temporal" mode -- there is nothing to gate on
    initialization/no_candidate/all_gated_out frames."""

    empty = {
        f"{prefix}_gate_confidence": None,
        f"{prefix}_gate_predicted_range_m": None,
        f"{prefix}_gate_predicted_bearing_rad": None,
        f"{prefix}_gate_measurement_range_m": None,
        f"{prefix}_gate_measurement_bearing_rad": None,
        f"{prefix}_gate_innovation_range_m": None,
        f"{prefix}_gate_innovation_bearing_rad": None,
        f"{prefix}_gate_nis": None,
        f"{prefix}_gate_threshold": None,
        f"{prefix}_gate_decision": None,
    }
    gate = record.gate
    if gate is None:
        return empty

    selected = record.association.selected
    measurement = None if selected is None else selected.measurement
    diagnostics = updater.last_ekf_diagnostics

    if diagnostics is not None and diagnostics.innovation_range_m is not None and measurement is not None:
        predicted_range = measurement.range_m - diagnostics.innovation_range_m
        predicted_bearing = wrap_angle(measurement.bearing_rad - diagnostics.innovation_bearing_rad)
        innovation_range = diagnostics.innovation_range_m
        innovation_bearing = diagnostics.innovation_bearing_rad
    else:
        predicted_range, predicted_bearing = _predicted_polar_after_update(
            updater, minimum_range_m
        )
        innovation_range = (
            None
            if predicted_range is None or measurement is None
            else measurement.range_m - predicted_range
        )
        innovation_bearing = (
            None
            if predicted_bearing is None or measurement is None
            else wrap_angle(measurement.bearing_rad - predicted_bearing)
        )

    return {
        f"{prefix}_gate_confidence": None if selected is None else selected.confidence,
        f"{prefix}_gate_predicted_range_m": predicted_range,
        f"{prefix}_gate_predicted_bearing_rad": predicted_bearing,
        f"{prefix}_gate_measurement_range_m": None if measurement is None else measurement.range_m,
        f"{prefix}_gate_measurement_bearing_rad": (
            None if measurement is None else measurement.bearing_rad
        ),
        f"{prefix}_gate_innovation_range_m": innovation_range,
        f"{prefix}_gate_innovation_bearing_rad": innovation_bearing,
        f"{prefix}_gate_nis": gate.nis,
        f"{prefix}_gate_threshold": gate.threshold,
        f"{prefix}_gate_decision": "accepted" if gate.accepted else "rejected",
    }


# ---------------------------------------------------------------------------
# The shared core: _FrameSelection + two constructors + _step_both_systems +
# build_row. Both collect_final_rows (render) and replay_from_cache (no
# simulator/detector) funnel every frame through these same functions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameSelection:
    """Everything derivable from ONE frame's raw candidate set, independent
    of whether that set came from a live detector call or a cached
    ``RuntimeCacheFrame``."""

    duckie_detection_count: int
    duplicate_selection: bool
    raw_pedestrian_measurement: ObjectMeasurement
    selected_confidence: float | None
    selected_bbox: BoundingBox | None
    robust_candidates: tuple[CandidateMeasurement, ...]
    bbox_by_key: dict[tuple[int, int, int, int], BoundingBox]


def _selection_from_render_result(result) -> _FrameSelection:
    """Render path: read the fields ``YoloPedestrianMeasurementPipeline
    .observe()`` already computed. Thin wrapper, no re-derivation -- this is
    the same logic collect_final_rows used before this refactor, only moved."""

    valid_candidates = [
        candidate for candidate in result.duckie_candidates if candidate.projection_error is None
    ]
    bbox_by_key = {
        _bbox_key(candidate.detection.bounding_box): candidate.detection.bounding_box
        for candidate in valid_candidates
    }
    robust_candidates = tuple(
        CandidateMeasurement(
            measurement=candidate.measurement,
            confidence=candidate.detection.confidence,
            bbox_key=_bbox_key(candidate.detection.bounding_box),
        )
        for candidate in valid_candidates
    )
    selected = result.selected_duckie
    return _FrameSelection(
        duckie_detection_count=result.duckie_detection_count,
        duplicate_selection=result.duplicate_selection,
        raw_pedestrian_measurement=result.pedestrian,
        selected_confidence=None if selected is None else selected.confidence,
        selected_bbox=None if selected is None else selected.bounding_box,
        robust_candidates=robust_candidates,
        bbox_by_key=bbox_by_key,
    )


def _selection_from_cache_frame(cache_frame: RuntimeCacheFrame) -> _FrameSelection:
    """Replay path: reconstruct the same selection using the REAL, frozen
    ``select_single_duckie`` (imported unchanged, never reimplemented) fed
    ``Detection`` objects rebuilt from the cache's raw candidate arrays.
    No simulator, no detector, no re-projection -- range/bearing come
    straight from the already-projected cached values."""

    detections: list[Detection] = [
        Detection(
            object_class=ObjectClass.DUCKIE,
            confidence=cache_frame.raw_candidate_confidence[index],
            bounding_box=BoundingBox(*cache_frame.raw_candidate_bbox[index]),
        )
        for index in range(cache_frame.raw_candidate_count)
    ]
    selection = select_single_duckie(detections)
    selected_index = (
        None
        if selection.selected is None
        else next(index for index, detection in enumerate(detections) if detection is selection.selected)
    )

    def _measurement_at(index: int) -> ObjectMeasurement:
        range_m = cache_frame.raw_candidate_range_m[index]
        bearing_rad = cache_frame.raw_candidate_bearing_rad[index]
        return ObjectMeasurement(
            object_class=ObjectClass.DUCKIE,
            detected=True,
            confidence=cache_frame.raw_candidate_confidence[index],
            x_left_m=range_m * sin(bearing_rad),
            y_forward_m=range_m * cos(bearing_rad),
            range_m=range_m,
            bearing_rad=bearing_rad,
        )

    if selected_index is None or cache_frame.raw_candidate_projection_failed[selected_index]:
        raw_pedestrian_measurement = ObjectMeasurement.missing(ObjectClass.DUCKIE)
    else:
        raw_pedestrian_measurement = _measurement_at(selected_index)

    robust_candidates: list[CandidateMeasurement] = []
    bbox_by_key: dict[tuple[int, int, int, int], BoundingBox] = {}
    for index in range(cache_frame.raw_candidate_count):
        if cache_frame.raw_candidate_projection_failed[index]:
            continue
        box = detections[index].bounding_box
        key = _bbox_key(box)
        robust_candidates.append(
            CandidateMeasurement(
                measurement=_measurement_at(index),
                confidence=cache_frame.raw_candidate_confidence[index],
                bbox_key=key,
            )
        )
        bbox_by_key[key] = box

    return _FrameSelection(
        duckie_detection_count=cache_frame.raw_candidate_count,
        duplicate_selection=cache_frame.raw_candidate_count > 1,
        raw_pedestrian_measurement=raw_pedestrian_measurement,
        selected_confidence=(
            None if selected_index is None else cache_frame.raw_candidate_confidence[selected_index]
        ),
        selected_bbox=None if selected_index is None else detections[selected_index].bounding_box,
        robust_candidates=tuple(robust_candidates),
        bbox_by_key=bbox_by_key,
    )


def _step_both_systems(
    *,
    selection: _FrameSelection,
    baseline_updater: PedestrianBeliefUpdater,
    baseline_belief,
    baseline_bias: AdditiveMeasurementBias,
    robust_updater: RobustPedestrianBeliefUpdater,
    robust_belief,
    previous_action: PolicyAction,
    ego: EgoObservation,
    road: RoadMeasurement | None,
    ego_motion: EgoMotion,
    dt_s: float,
):
    """Step Baseline A and Robust B from one frame's selection. Identical
    for both paths: the EKF/existence math has no dependency on how the
    candidates were obtained, only on their values."""

    corrected_measurement = baseline_bias.correct(selection.raw_pedestrian_measurement)
    baseline_belief = baseline_updater.update(
        previous_belief=baseline_belief,
        previous_action=previous_action,
        ego_motion=ego_motion,
        perception=_perception(ego, road, corrected_measurement),
        dt_s=dt_s,
    )
    robust_belief, record = robust_updater.update(
        robust_belief, previous_action, ego_motion, list(selection.robust_candidates), dt_s
    )
    return baseline_belief, corrected_measurement, robust_belief, record


def build_row(
    *,
    episode: str,
    seed: int,
    scenario: str,
    pedestrian_mode: str,
    frame: int,
    timestamp_s: float,
    dt_s: float,
    previous_action: PolicyAction,
    ego_motion: EgoMotion,
    truth: TruthFrame,
    selection: _FrameSelection,
    baseline_updater: PedestrianBeliefUpdater,
    baseline_belief,
    corrected_measurement: ObjectMeasurement,
    robust_updater: RobustPedestrianBeliefUpdater,
    robust_belief,
    record,
    matching_iou_threshold: float,
    minimum_range_m: float,
) -> dict[str, object]:
    """THE shared row-assembly function -- the one place both
    ``collect_final_rows`` and ``replay_from_cache`` build a CSV row. GT
    (``eligible_visible``/``distance_bin``/``fov_region``/``gt_*``) comes
    exclusively from ``truth``, never recomputed here, so it cannot drift
    between a live render and a cache replay of the same underlying data.
    ``visibility_reason`` (the free-text silhouette-assessment reason) is
    NOT part of this function's output: it is not stored in ``TruthFrame``
    (frozen, already written to disk before this refactor), so it cannot be
    recovered during replay. Each caller sets it afterward: the render path
    still has the live ``AnnotationDecision`` and sets the real reason;
    the replay path sets an explicit empty string, documented as not
    recoverable, rather than fabricating one.
    """

    gt_box = None if truth.gt_bbox is None else BoundingBox(*truth.gt_bbox)
    iou = (
        None
        if not truth.eligible_visible or gt_box is None or selection.selected_bbox is None
        else intersection_over_union(gt_box, selection.selected_bbox)
    )
    selected_correct = bool(
        selection.raw_pedestrian_measurement.detected
        and truth.eligible_visible
        and iou is not None
        and iou >= matching_iou_threshold
    )
    associated_iou = None
    if record.association.selected is not None and truth.eligible_visible and gt_box is not None:
        associated_box = selection.bbox_by_key.get(record.association.selected.bbox_key)
        if associated_box is not None:
            associated_iou = intersection_over_union(gt_box, associated_box)

    row: dict[str, object] = {
        "episode": episode,
        "seed": seed,
        "scenario": scenario,
        "pedestrian_mode": pedestrian_mode,
        "frame": frame,
        "timestamp_s": timestamp_s,
        "dt_s": dt_s,
        "eligible_visible": truth.eligible_visible,
        "visible_pixel_count": truth.visible_pixel_count,
        "distance_bin": truth.distance_bin,
        "fov_region": truth.fov_region,
        "gt_range_m": truth.gt_range_m,
        "gt_bearing_rad": truth.gt_bearing_rad,
        "gt_range_rate_mps": truth.gt_range_rate_mps,
        "gt_bearing_rate_rad_s": truth.gt_bearing_rate_rad_s,
        "actual_ego_velocity_mps": ego_motion.linear_velocity_mps,
        "actual_ego_yaw_rate_rad_s": ego_motion.yaw_rate_rad_s,
        "commanded_velocity_mps": previous_action.linear_velocity_mps,
        "commanded_yaw_rate_rad_s": previous_action.angular_velocity_rad_s,
        # Shared detector/candidate columns -- one inference (or its cached
        # replay) feeds both systems.
        "detector_detected": record.detector_detected,
        "measurement_detected": record.detector_detected,
        "kinematic_measurement_accepted": record.kinematic_measurement_accepted,
        "duckie_detection_count": selection.duckie_detection_count,
        "candidate_count": len(selection.robust_candidates),
        "duplicate_selection": selection.duplicate_selection,
        "selected_confidence": _optional(selection.selected_confidence),
        "selected_iou": _optional(iou),
        "selected_correct_iou50": selected_correct,
        "raw_range_m": _optional(selection.raw_pedestrian_measurement.range_m),
        "raw_bearing_rad": _optional(selection.raw_pedestrian_measurement.bearing_rad),
        # Baseline A.
        "baseline_a_measurement_range_m": _optional(corrected_measurement.range_m),
        "baseline_a_measurement_bearing_rad": _optional(corrected_measurement.bearing_rad),
        "baseline_a_measurement_range_error_m": _optional(
            None
            if corrected_measurement.range_m is None or truth.gt_range_m is None
            else corrected_measurement.range_m - truth.gt_range_m
        ),
        "baseline_a_measurement_bearing_error_rad": _optional(
            None
            if corrected_measurement.bearing_rad is None or truth.gt_bearing_rad is None
            else wrap_angle(corrected_measurement.bearing_rad - truth.gt_bearing_rad)
        ),
        # Robust B.
        "robust_b_associated_iou": _optional(associated_iou),
    }
    row.update(_box_fields("gt_bbox", gt_box))
    row.update(_box_fields("selected_bbox", selection.selected_bbox))
    row.update(_baseline_belief_fields("baseline_a", baseline_updater, baseline_belief))
    row.update(_robust_belief_fields("robust_b", robust_updater, robust_belief, record))
    row.update(_gate_log_fields("robust_b", robust_updater, record, minimum_range_m))
    return row


# ---------------------------------------------------------------------------
# Error-case image bookkeeping -- render-path-only (needs the live RGB
# frame, which the replay path never has).
# ---------------------------------------------------------------------------


def _remember_case(
    cases: dict[str, list[dict[str, object]]],
    category: str,
    score: float | None,
    row: dict[str, object],
    rgb: np.ndarray,
    gt_box: BoundingBox | None,
    selected_box: BoundingBox | None,
    maximum: int,
) -> None:
    if score is None or not np.isfinite(score):
        return
    case = {
        "score": float(score),
        "row": dict(row),
        "rgb": np.asarray(rgb).copy(),
        "gt_box": gt_box,
        "selected_box": selected_box,
    }
    bucket = cases.setdefault(category, [])
    bucket.append(case)
    bucket.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["row"]["episode"]),
            int(item["row"]["frame"]),
        )
    )
    del bucket[maximum:]


def _save_error_cases(cases: dict[str, list[dict[str, object]]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for category in sorted(cases):
        for index, case in enumerate(cases[category]):
            row = case["row"]
            image = Image.fromarray(case["rgb"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box, color in ((case["gt_box"], (0, 255, 0)), (case["selected_box"], (255, 0, 0))):
                if box is None:
                    continue
                draw.rectangle(
                    (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                    outline=color,
                    width=3,
                )
            label = (
                f"{category} score={case['score']:.3f} "
                f"baseline_a=({row['baseline_a_belief_range_m']}) "
                f"robust_b=({row['robust_b_belief_range_m']}) "
                f"gt=({row['gt_range_m']:.3f},{row['gt_bearing_rad']:.3f})"
            )
            draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
            draw.text((4, 4), label, fill=(255, 255, 255))
            destination = output_dir / (
                f"{category}_{index:02d}_{row['episode']}_f{int(row['frame']):04d}.png"
            )
            image.save(destination)
            outputs.append(str(destination))
    return outputs


# ---------------------------------------------------------------------------
# Render path.
# ---------------------------------------------------------------------------


def collect_final_rows(
    protocol: F9cProtocol,
    settings: _ExperimentSettings,
) -> tuple[list[dict[str, object]], list[RuntimeCacheFrame], list[TruthFrame], list[str]]:
    """Render every final-evaluation (seed, scenario) episode ONCE, stepping
    Baseline A and Robust B off the same single YOLO inference per frame.

    Detector and simulator construction happen ONLY here -- never at module
    scope, never in a metrics-only helper, never in ``replay_from_cache`` --
    see the module docstring.

    An early simulator termination (terminated/truncated) raises
    immediately rather than logging a warning and continuing: this run may
    only be rendered once, so a partial render must stop the whole process
    rather than silently producing a validation CSV that under-represents
    one episode. (Task 9's calibration run takes the opposite,
    continue-on-early-termination stance deliberately -- calibration seeds
    are reusable; final-evaluation seeds are not.)
    """

    annotation = load_dataset_config(settings.annotation_config_path)
    detector = YoloObjectDetector(
        protocol.checkpoint_path,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_nms_iou_threshold,
        image_size=settings.detector_image_size,
        device=settings.detector_device,
        max_detections=settings.detector_max_detections,
    )
    ekf_config = load_pedestrian_ekf_config(protocol.config_path)
    measurement_noise = load_polar_measurement_noise(protocol.config_path)

    baseline_existence_config = load_existence_filter_config(protocol.config_path)
    robust_existence_config = replace(
        baseline_existence_config,
        miss_likelihood_floor=load_miss_likelihood_floor(protocol.config_path),
    )

    with protocol.config_path.open("rb") as stream:
        raw_config = tomllib.load(stream)
    baseline_bias = AdditiveMeasurementBias(
        float(raw_config["baseline_measurement_model"]["range_bias_m"]),
        float(raw_config["baseline_measurement_model"]["bearing_bias_rad"]),
    )

    robust_config = load_robust_observation_config(protocol.config_path)
    robust_bias_frozen = load_frozen_bias_correction(
        protocol.config_path, section="baseline_measurement_model"
    )
    robust_bias_fitted = load_frozen_bias_correction(
        protocol.config_path, section="measurement_model"
    )

    rows: list[dict[str, object]] = []
    cache_frames: list[RuntimeCacheFrame] = []
    truth_frames: list[TruthFrame] = []
    cases: dict[str, list[dict[str, object]]] = {}

    final_specs = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
    if not final_specs:
        raise RuntimeError("F9c protocol has no final-evaluation scenarios")

    for seed in protocol.final_evaluation_seeds:
        for scenario_index, spec in enumerate(protocol.scenarios):
            if not spec.use_for_final_evaluation:
                continue
            scenario = _scenario_for(settings, spec, seed, scenario_index)
            episode = f"evaluation_{seed}_{spec.name}"
            integration = create_gym_duckietown(
                GymDuckietownConfig(
                    scenario=scenario,
                    domain_randomization=settings.domain_randomization,
                    dynamics_randomization=settings.dynamics_randomization,
                    maximum_steps=spec.steps + 2,
                    camera_width=settings.image_width_px,
                    camera_height=settings.image_height_px,
                )
            )
            try:
                observation = integration.agent.reset(seed=seed)
                camera_calibration = integration.camera_calibration.read()
                projector = YoloMeasurementProjector(
                    CalibratedGroundProjector(camera_calibration),
                    MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
                )
                runtime_pipeline = YoloPedestrianMeasurementPipeline(detector, projector)
                observability_model = PredictedObservabilityModel(
                    CalibratedGroundProjector(camera_calibration),
                    image_width_px=settings.image_width_px,
                )

                baseline_updater = PedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    existence_config=baseline_existence_config,
                    measurement_noise=measurement_noise,
                    measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
                )
                robust_updater = RobustPedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    measurement_noise=measurement_noise,
                    existence_config=robust_existence_config,
                    bias_frozen=robust_bias_frozen,
                    bias_fitted=robust_bias_fitted,
                    observability_model=observability_model,
                    config=robust_config,
                )

                baseline_belief = initial_belief(
                    observation.ego, existence_prior=baseline_existence_config.prior_probability
                )
                robust_belief = initial_belief(
                    observation.ego, existence_prior=robust_existence_config.prior_probability
                )
                timestamp_s = 0.0
                dt_s = 1.0 / 30.0
                previous_action = PolicyAction(0.0, 0.0)

                for frame in range(spec.steps + 1):
                    # --- Runtime chain: one detector inference feeds both
                    # systems. -------------------------------------------
                    result = runtime_pipeline.observe(observation.front_rgb)

                    # Invariant I5: capture the runtime cache BEFORE any
                    # bias correction, straight off observe()'s raw
                    # candidates.
                    ego_motion = observation.ego.motion
                    cache_frame = RuntimeCacheFrame(
                        episode=episode,
                        seed=seed,
                        scenario=spec.name,
                        frame=frame,
                        dt_s=dt_s,
                        raw_candidate_range_m=tuple(
                            float("nan")
                            if candidate.projection_error is not None
                            else float(candidate.measurement.range_m)
                            for candidate in result.duckie_candidates
                        ),
                        raw_candidate_bearing_rad=tuple(
                            float("nan")
                            if candidate.projection_error is not None
                            else float(candidate.measurement.bearing_rad)
                            for candidate in result.duckie_candidates
                        ),
                        raw_candidate_confidence=tuple(
                            float(candidate.detection.confidence)
                            for candidate in result.duckie_candidates
                        ),
                        raw_candidate_bbox=tuple(
                            (
                                candidate.detection.bounding_box.x_min_px,
                                candidate.detection.bounding_box.y_min_px,
                                candidate.detection.bounding_box.x_max_px,
                                candidate.detection.bounding_box.y_max_px,
                            )
                            for candidate in result.duckie_candidates
                        ),
                        raw_candidate_projection_failed=tuple(
                            candidate.projection_error is not None
                            for candidate in result.duckie_candidates
                        ),
                        ego_linear_velocity_mps=ego_motion.linear_velocity_mps,
                        ego_yaw_rate_rad_s=ego_motion.yaw_rate_rad_s,
                    )
                    cache_frames.append(cache_frame)

                    selection = _selection_from_render_result(result)

                    baseline_belief, corrected_measurement, robust_belief, record = _step_both_systems(
                        selection=selection,
                        baseline_updater=baseline_updater,
                        baseline_belief=baseline_belief,
                        baseline_bias=baseline_bias,
                        robust_updater=robust_updater,
                        robust_belief=robust_belief,
                        previous_action=previous_action,
                        ego=observation.ego,
                        road=observation.road,
                        ego_motion=ego_motion,
                        dt_s=dt_s,
                    )

                    # --- Runtime/privileged boundary: both updaters have
                    # been stepped; only now is ground truth read. --------
                    privileged = integration.privileged.read()
                    silhouettes = {
                        item.object_kind: item
                        for item in integration.projection_validation.sample_object_silhouettes(
                            ("duckie",)
                        )
                    }
                    silhouette = silhouettes.get("duckie")
                    gt_box = None if silhouette is None else silhouette.bounding_box
                    visible_pixels = 0 if silhouette is None else silhouette.visible_pixel_count
                    decision = assess_silhouette(
                        class_id=yolo_class_id(ObjectClass.DUCKIE),
                        box=gt_box,
                        visible_pixel_count=visible_pixels,
                        image_width_px=settings.image_width_px,
                        image_height_px=settings.image_height_px,
                        rules=annotation.duckie_rules,
                    )
                    truth_state = privileged.true_pomdp_state.pedestrian
                    if (
                        not truth_state.exists
                        or truth_state.range_m is None
                        or truth_state.bearing_rad is None
                        or truth_state.radial_velocity_mps is None
                        or truth_state.bearing_rate_rad_s is None
                    ):
                        raise RuntimeError(f"{episode} lost pedestrian truth")

                    distance_bin = _distance_bin(
                        truth_state.range_m, settings.near_max_m, settings.medium_max_m
                    )
                    fov_region = _fov_region(
                        gt_box if decision.accepted else None, settings.image_width_px
                    )

                    truth = TruthFrame(
                        episode=episode,
                        frame=frame,
                        gt_exists=bool(truth_state.exists),
                        gt_range_m=truth_state.range_m,
                        gt_bearing_rad=truth_state.bearing_rad,
                        gt_range_rate_mps=truth_state.radial_velocity_mps,
                        gt_bearing_rate_rad_s=truth_state.bearing_rate_rad_s,
                        eligible_visible=decision.accepted,
                        visible_pixel_count=visible_pixels,
                        gt_bbox=(
                            None
                            if gt_box is None
                            else (gt_box.x_min_px, gt_box.y_min_px, gt_box.x_max_px, gt_box.y_max_px)
                        ),
                        distance_bin=distance_bin,
                        fov_region=fov_region,
                    )
                    truth_frames.append(truth)

                    row = build_row(
                        episode=episode,
                        seed=seed,
                        scenario=spec.name,
                        pedestrian_mode=spec.pedestrian_mode.value,
                        frame=frame,
                        timestamp_s=timestamp_s,
                        dt_s=dt_s,
                        previous_action=previous_action,
                        ego_motion=ego_motion,
                        truth=truth,
                        selection=selection,
                        baseline_updater=baseline_updater,
                        baseline_belief=baseline_belief,
                        corrected_measurement=corrected_measurement,
                        robust_updater=robust_updater,
                        robust_belief=robust_belief,
                        record=record,
                        matching_iou_threshold=settings.matching_iou_threshold,
                        minimum_range_m=ekf_config.minimum_range_m,
                    )
                    row["visibility_reason"] = decision.reason
                    rows.append(row)

                    robust_diag = robust_updater.last_ekf_diagnostics
                    _remember_case(
                        cases,
                        "range_innovation",
                        (
                            None
                            if robust_diag is None or robust_diag.innovation_range_m is None
                            else abs(robust_diag.innovation_range_m)
                        ),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selection.selected_bbox,
                        settings.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "bearing_innovation",
                        None
                        if robust_diag is None or robust_diag.innovation_bearing_rad is None
                        else abs(robust_diag.innovation_bearing_rad),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selection.selected_bbox,
                        settings.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "nis",
                        None if robust_diag is None else robust_diag.nis,
                        row,
                        observation.front_rgb,
                        gt_box,
                        selection.selected_bbox,
                        settings.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "belief_range_error",
                        (
                            abs(robust_belief.pedestrian.range_mean_m - truth_state.range_m)
                            if robust_updater.ekf.initialized
                            else None
                        ),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selection.selected_bbox,
                        settings.error_cases_per_category,
                    )

                    if frame == spec.steps:
                        break
                    transition = integration.agent.step(spec.action)
                    diagnostics = integration.diagnostics.read()
                    if transition.terminated or transition.truncated:
                        raise RuntimeError(
                            f"{episode} ended early at frame {frame}/{spec.steps} "
                            f"({diagnostics.done_code}) -- final-evaluation seeds "
                            "may be rendered exactly once; stopping rather than "
                            "producing a partial render"
                        )
                    observation = transition.observation
                    dt_s = diagnostics.timestamp_s - timestamp_s
                    timestamp_s = diagnostics.timestamp_s
                    previous_action = spec.action
            finally:
                integration.close()
            print(f"completed {episode}: total_rows={len(rows)}", flush=True)

    return rows, cache_frames, truth_frames, _save_error_cases(cases, protocol.artifacts["error_case_dir"])


# ---------------------------------------------------------------------------
# Replay path -- NO simulator, NO detector, NO GPU. See the module
# docstring's "disclosed limitation" section for the one thing this path
# cannot reconstruct exactly (per-episode randomized camera calibration).
# ---------------------------------------------------------------------------


def _load_cache_and_truth(
    protocol: F9cProtocol,
    *,
    expected_runtime_cache_sha256: str,
    expected_evaluation_truth_sha256: str,
) -> tuple[tuple[RuntimeCacheFrame, ...], dict[tuple[str, int], TruthFrame], str, str]:
    """Verify both hashes BEFORE any read, then load the cache/truth files.

    Shared by ``replay_from_cache`` (Task 11) and ``run_ablation`` (Task 12)
    -- the only two functions in this module permitted to read these files
    -- so both refuse a silently regenerated cache identically, and the
    ablation reads the file exactly once regardless of how many switch
    combinations it later replays. Touches no simulator, no detector.
    """

    runtime_cache_path = protocol.artifacts["runtime_cache"]
    evaluation_truth_path = protocol.artifacts["evaluation_truth"]

    runtime_cache_sha256 = sha256(runtime_cache_path)
    if runtime_cache_sha256 != expected_runtime_cache_sha256:
        raise ValueError(
            f"runtime cache {runtime_cache_path} (sha256={runtime_cache_sha256}) "
            f"does not match the expected hash {expected_runtime_cache_sha256} "
            "from the original final-evaluation render -- refusing to replay a "
            "cache that may not be the one that render produced"
        )
    evaluation_truth_sha256 = sha256(evaluation_truth_path)
    if evaluation_truth_sha256 != expected_evaluation_truth_sha256:
        raise ValueError(
            f"evaluation truth {evaluation_truth_path} (sha256={evaluation_truth_sha256}) "
            f"does not match the expected hash {expected_evaluation_truth_sha256}"
        )

    cache_frames = read_runtime_cache(runtime_cache_path, expected_sha256=runtime_cache_sha256)
    truth_by_key = read_evaluation_truth(evaluation_truth_path, expected_sha256=evaluation_truth_sha256)
    return cache_frames, truth_by_key, runtime_cache_sha256, evaluation_truth_sha256


def _replay_components(protocol: F9cProtocol, settings: _ExperimentSettings) -> dict[str, Any]:
    """Every config object the replay loop needs that does NOT vary across
    Task 12's seven ablation rows (or between Baseline A and Robust B) --
    loaded once and shared by every row, so ``run_ablation`` reads the
    frozen config file's invariant sections exactly once per process rather
    than seven times.
    """

    ekf_config = load_pedestrian_ekf_config(protocol.config_path)
    measurement_noise = load_polar_measurement_noise(protocol.config_path)
    baseline_existence_config = load_existence_filter_config(protocol.config_path)
    with protocol.config_path.open("rb") as stream:
        raw_config = tomllib.load(stream)
    baseline_bias = AdditiveMeasurementBias(
        float(raw_config["baseline_measurement_model"]["range_bias_m"]),
        float(raw_config["baseline_measurement_model"]["bearing_bias_rad"]),
    )
    robust_bias_frozen = load_frozen_bias_correction(
        protocol.config_path, section="baseline_measurement_model"
    )
    robust_bias_fitted = load_frozen_bias_correction(protocol.config_path, section="measurement_model")

    # Disclosed limitation (module docstring): the nominal, unrandomized
    # camera model, not the per-episode randomized one the render used.
    observability_model = PredictedObservabilityModel(
        CalibratedGroundProjector(_NOMINAL_CAMERA_CALIBRATION),
        image_width_px=settings.image_width_px,
    )
    return {
        "ekf_config": ekf_config,
        "measurement_noise": measurement_noise,
        "baseline_existence_config": baseline_existence_config,
        "baseline_bias": baseline_bias,
        "robust_bias_frozen": robust_bias_frozen,
        "robust_bias_fitted": robust_bias_fitted,
        "observability_model": observability_model,
    }


def _replay_rows(
    protocol: F9cProtocol,
    settings: _ExperimentSettings,
    cache_frames: tuple[RuntimeCacheFrame, ...],
    truth_by_key: dict[tuple[str, int], TruthFrame],
    *,
    ekf_config,
    measurement_noise,
    baseline_existence_config: ExistenceFilterConfig,
    baseline_bias: AdditiveMeasurementBias,
    robust_existence_config: ExistenceFilterConfig,
    robust_config: RobustObservationConfig,
    robust_bias_frozen: FrozenBiasCorrection,
    robust_bias_fitted: FrozenBiasCorrection,
    observability_model: PredictedObservabilityModel,
) -> list[dict[str, object]]:
    """The per-episode cache-replay loop -- NO simulator, NO detector, NO
    GPU -- shared by ``replay_from_cache`` (Task 11, one fixed config) and
    ``run_ablation`` (Task 12, called once per switch combination with a
    different ``robust_existence_config``/``robust_config``, everything
    else held fixed). Every frame is stepped from the already-cached raw
    candidates via ``_step_both_systems``/``build_row``, the same shared
    core the render path uses (see module docstring).
    """

    scenario_by_name = {spec.name: spec for spec in protocol.scenarios}

    frames_by_episode: dict[str, list[RuntimeCacheFrame]] = {}
    for cache_frame in cache_frames:
        frames_by_episode.setdefault(cache_frame.episode, []).append(cache_frame)

    rows: list[dict[str, object]] = []
    for episode in sorted(frames_by_episode):
        episode_frames = sorted(frames_by_episode[episode], key=lambda frame: frame.frame)
        scenario_name = episode_frames[0].scenario
        seed = episode_frames[0].seed
        spec = scenario_by_name[scenario_name]

        baseline_updater = PedestrianBeliefUpdater(
            ekf_config=ekf_config,
            existence_config=baseline_existence_config,
            measurement_noise=measurement_noise,
            measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
        )
        robust_updater = RobustPedestrianBeliefUpdater(
            ekf_config=ekf_config,
            measurement_noise=measurement_noise,
            existence_config=robust_existence_config,
            bias_frozen=robust_bias_frozen,
            bias_fitted=robust_bias_fitted,
            observability_model=observability_model,
            config=robust_config,
        )

        placeholder_ego = EgoObservation(
            0.0, 0.0, episode_frames[0].ego_linear_velocity_mps, episode_frames[0].ego_yaw_rate_rad_s
        )
        baseline_belief = initial_belief(
            placeholder_ego, existence_prior=baseline_existence_config.prior_probability
        )
        robust_belief = initial_belief(
            placeholder_ego, existence_prior=robust_existence_config.prior_probability
        )

        timestamp_s = 0.0
        previous_action = PolicyAction(0.0, 0.0)

        for index, cache_frame in enumerate(episode_frames):
            truth = truth_by_key.get((episode, cache_frame.frame))
            if truth is None:
                raise RuntimeError(f"{episode} frame {cache_frame.frame} missing from evaluation truth")
            if (
                not truth.gt_exists
                or truth.gt_range_m is None
                or truth.gt_bearing_rad is None
                or truth.gt_range_rate_mps is None
                or truth.gt_bearing_rate_rad_s is None
            ):
                raise RuntimeError(f"{episode} lost pedestrian truth at frame {cache_frame.frame}")

            ego_motion = EgoMotion(cache_frame.ego_linear_velocity_mps, cache_frame.ego_yaw_rate_rad_s)
            ego = EgoObservation(
                0.0, 0.0, cache_frame.ego_linear_velocity_mps, cache_frame.ego_yaw_rate_rad_s
            )
            selection = _selection_from_cache_frame(cache_frame)

            baseline_belief, corrected_measurement, robust_belief, record = _step_both_systems(
                selection=selection,
                baseline_updater=baseline_updater,
                baseline_belief=baseline_belief,
                baseline_bias=baseline_bias,
                robust_updater=robust_updater,
                robust_belief=robust_belief,
                previous_action=previous_action,
                ego=ego,
                road=None,
                ego_motion=ego_motion,
                dt_s=cache_frame.dt_s,
            )

            row = build_row(
                episode=episode,
                seed=seed,
                scenario=scenario_name,
                pedestrian_mode=spec.pedestrian_mode.value,
                frame=cache_frame.frame,
                timestamp_s=timestamp_s,
                dt_s=cache_frame.dt_s,
                previous_action=previous_action,
                ego_motion=ego_motion,
                truth=truth,
                selection=selection,
                baseline_updater=baseline_updater,
                baseline_belief=baseline_belief,
                corrected_measurement=corrected_measurement,
                robust_updater=robust_updater,
                robust_belief=robust_belief,
                record=record,
                matching_iou_threshold=settings.matching_iou_threshold,
                minimum_range_m=ekf_config.minimum_range_m,
            )
            # Not recoverable from TruthFrame's already-committed schema --
            # see build_row's docstring.
            row["visibility_reason"] = ""
            rows.append(row)

            if index + 1 < len(episode_frames):
                timestamp_s += episode_frames[index + 1].dt_s
                previous_action = spec.action
        print(f"replayed {episode}: total_rows={len(rows)}", flush=True)

    return rows


def replay_from_cache(
    protocol: F9cProtocol,
    *,
    expected_runtime_cache_sha256: str = EXPECTED_RUNTIME_CACHE_SHA256,
    expected_evaluation_truth_sha256: str = EXPECTED_EVALUATION_TRUTH_SHA256,
) -> tuple[list[dict[str, object]], str, str]:
    """Reconstruct every row of the final-evaluation run from the already-
    written runtime cache and evaluation truth. Touches no simulator, no
    detector, no GPU. Deterministic given identical inputs: the EKF/
    existence math has no randomness, so replaying the same (candidates,
    ego_motion, dt_s, previous_action) sequence through fresh updaters
    reproduces the same belief trajectory a live run would have produced,
    modulo the one disclosed limitation (module docstring): Robust B's
    observability classification -- and only that -- uses the nominal,
    non-randomized camera model rather than the per-episode randomized one
    the original render used, because that per-episode value is not part of
    the (now immutable, already-written) runtime-cache schema.

    Both hashes are checked against the caller-supplied (default: the
    values recorded from the original render, see the module constants)
    expected hash BEFORE any read is attempted, and ``read_runtime_cache``/
    ``read_evaluation_truth`` re-verify internally -- a cache that does not
    match is refused, never silently replayed.

    Thin wrapper around ``_load_cache_and_truth``/``_replay_components``/
    ``_replay_rows`` (Task 12 factored these out of what used to be this
    function's own body so ``run_ablation`` could reuse them unchanged) --
    this function's external signature and behavior are unchanged from
    Task 11.
    """

    cache_frames, truth_by_key, runtime_cache_sha256, evaluation_truth_sha256 = _load_cache_and_truth(
        protocol,
        expected_runtime_cache_sha256=expected_runtime_cache_sha256,
        expected_evaluation_truth_sha256=expected_evaluation_truth_sha256,
    )

    settings = _ExperimentSettings(protocol.config_path)
    components = _replay_components(protocol, settings)
    robust_existence_config = replace(
        components["baseline_existence_config"],
        miss_likelihood_floor=load_miss_likelihood_floor(protocol.config_path),
    )
    robust_config = load_robust_observation_config(protocol.config_path)

    rows = _replay_rows(
        protocol,
        settings,
        cache_frames,
        truth_by_key,
        ekf_config=components["ekf_config"],
        measurement_noise=components["measurement_noise"],
        baseline_existence_config=components["baseline_existence_config"],
        baseline_bias=components["baseline_bias"],
        robust_existence_config=robust_existence_config,
        robust_config=robust_config,
        robust_bias_frozen=components["robust_bias_frozen"],
        robust_bias_fitted=components["robust_bias_fitted"],
        observability_model=components["observability_model"],
    )
    return rows, runtime_cache_sha256, evaluation_truth_sha256


# ---------------------------------------------------------------------------
# Task 12: ablation from the runtime cache.
#
# Seven ``RobustObservationSwitches`` combinations, each replayed from the
# SAME hash-verified runtime cache via ``_replay_rows`` -- invariant I4:
# this section constructs no detector and no simulator, and never imports
# ``YoloObjectDetector``/``create_gym_duckietown`` calls into its own
# functions (both remain confined to ``collect_final_rows`` above).
#
# **The bias stage is always present; only which constants it carries
# varies.** Every row with ``bias_refit=False`` carries the F9b frozen
# constants (``robust_bias_frozen``, loaded from
# ``[baseline_measurement_model]`` -- identical to what Baseline A itself
# uses), never ``FrozenBiasCorrection.identity()``. This is what makes the
# ``baseline`` row equal Baseline A by construction (see
# ``RobustPedestrianBeliefUpdater.update``'s own ``bias = self._bias_fitted
# if switches.bias_refit else self._bias_frozen``) rather than by
# coincidence.
#
# **Two switches have no runtime knob inside
# ``RobustPedestrianBeliefUpdater.update()`` itself** -- ``covariance``
# (``CovarianceCalibration``, invariant I1's lambda-R provider and the
# posterior-variance floor) and the existence track's I8
# ``miss_likelihood_floor`` are both baked into their config objects at
# CONSTRUCTION time and consumed unconditionally by ``update()``/
# ``ExistenceFilter.update()``. ``_robust_config_for_switches``/
# ``_existence_config_for_switches`` below are what make
# ``covariance_calibration``/``conditional_detection`` have any observable
# effect at all: identity covariance (lambda=1, floor=0) and a 0.0
# (strict-no-op) miss-likelihood floor when off, the frozen fitted values
# when on. The task brief bundles the I8 floor with
# ``conditional_detection`` ("P_D^eff + I3 routing + I8 floor"), so the two
# are switched together.
# ---------------------------------------------------------------------------


_IDENTITY_COVARIANCE_CALIBRATION = CovarianceCalibration(1.0, 1.0, 0.0, 0.0)

# A THIRD piece of config with no runtime switch-check inside
# RobustPedestrianBeliefUpdater.update() -- discovered empirically while
# building this ablation, not named in the task-12 brief's five-switch
# table, but required for the "baseline" endpoint to equal Baseline A
# exactly (see run_ablation's docstring for the full argument):
#
# update()'s track-lifecycle policy (steps 10-11: discard a just-accepted
# candidate whose same-frame existence_probability falls below
# initialization_threshold; delete an active track whose existence_probability
# falls below delete_threshold) runs UNCONDITIONALLY -- it is not gated by
# ANY of bias_refit/innovation_gate/temporal_association/
# covariance_calibration/conditional_detection. PedestrianBeliefUpdater (the
# frozen F7 updater Baseline A runs) has NO such concept at all: once its
# EKF is first corrected, ekf.initialized never becomes False again. So
# "all five switches off" alone is necessary but NOT sufficient for the
# "baseline" row to equal Baseline A -- confirmed empirically: with the
# frozen config's real thresholds (0.50/0.05) still in force, the two
# diverge by ~114 out of 3328 frames on `ekf.initialized`, entirely inside
# episodes containing one of the run's eight track deletions.
#
# delete_threshold=0.0 makes deletion structurally UNREACHABLE, not merely
# unlikely: existence_probability is clamped to [0.0, 1.0]
# (belief/existence_filter.py's `min(1.0, max(0.0, ...))`), so
# `existence_probability < 0.0` can never be true, for any input. This is
# exact, not an approximation.
#
# initialization_threshold=1e-9 (as small as the strictly-greater-than-
# delete_threshold validation in RobustObservationConfig.__post_init__
# allows) matters only ONCE per episode: the very first frame a candidate
# is ever accepted, before any track has existed (every subsequent frame is
# "temporal" once track_was_active is sticky-True, which it becomes forever
# once delete_threshold=0.0 rules out deletion). On that one frame,
# existence_probability is a single Bayesian update from the config's own
# prior with `detected=True`. This argument is scoped to the FROZEN config
# actually in force here (prior_probability=0.50, survival_probability=0.995,
# birth_probability=0.005, detection_probability=0.9766775777414075,
# false_positive_probability=0.00078003120124805) -- it is NOT a claim about
# every valid ExistenceFilterConfig: `predicted = survival*prior +
# birth*(1-prior)` has no lower bound in general (ExistenceFilterConfig
# places none on survival_probability/birth_probability), so a config with
# both near zero could drive `predicted` -- and hence the posterior -- below
# 1e-9 even with detection_probability > false_positive_probability. For
# THIS frozen config, `predicted = 0.995*0.50 + 0.005*0.50 = 0.50` exactly,
# and the resulting posterior is `0.9992019795087668` -- five orders of
# magnitude above 1e-9, computed directly from ExistenceFilter.update's own
# formula, not merely argued to be "above the prior".
#
# Both together were verified, empirically, to make ekf.initialized track
# Baseline A's own bit-for-bit on all 3328 real frames (every field of
# summarize_f9c's ekf/track_continuity/nis/miss_sequence sections, not just
# range) -- see task-12-report.md.
_BASELINE_INITIALIZATION_THRESHOLD = 1e-9
_BASELINE_DELETE_THRESHOLD = 0.0

#: The seven ablation rows, in report order. See the task-12 brief's table:
#: the bias stage is ALWAYS present (only which constants it carries
#: varies); "baseline" (all switches off) must equal Baseline A exactly,
#: and "all_combined" (all switches on) must equal Robust B exactly -- both
#: by construction, verified by
#: ``test_ablation_endpoints_match_the_two_headline_systems``.
ABLATION_ROWS: tuple[tuple[str, RobustObservationSwitches], ...] = (
    (
        "baseline",
        RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=False,
            temporal_association=False,
            covariance_calibration=False,
            conditional_detection=False,
        ),
    ),
    (
        "bias_refit_only",
        RobustObservationSwitches(
            bias_refit=True,
            innovation_gate=False,
            temporal_association=False,
            covariance_calibration=False,
            conditional_detection=False,
        ),
    ),
    (
        "innovation_gate_only",
        RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=True,
            temporal_association=False,
            covariance_calibration=False,
            conditional_detection=False,
        ),
    ),
    (
        "temporal_association_only",
        RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=False,
            temporal_association=True,
            covariance_calibration=False,
            conditional_detection=False,
        ),
    ),
    (
        "covariance_calibration_only",
        RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=False,
            temporal_association=False,
            covariance_calibration=True,
            conditional_detection=False,
        ),
    ),
    (
        "conditional_detection_only",
        RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=False,
            temporal_association=False,
            covariance_calibration=False,
            conditional_detection=True,
        ),
    ),
    (
        "all_combined",
        RobustObservationSwitches(
            bias_refit=True,
            innovation_gate=True,
            temporal_association=True,
            covariance_calibration=True,
            conditional_detection=True,
        ),
    ),
)


def _robust_config_for_switches(
    fitted_config: RobustObservationConfig, switches: RobustObservationSwitches
) -> RobustObservationConfig:
    """Build one ablation row's ``RobustObservationConfig``.

    ``bias_refit``/``innovation_gate``/``temporal_association``/
    ``conditional_detection`` are switched INSIDE
    ``RobustPedestrianBeliefUpdater.update()`` itself -- passing the real
    fitted ``gate``/``association``/``effective_detection`` objects
    unconditionally is correct because the SWITCH, not the object, controls
    whether ``update()`` consults them.

    ``covariance_calibration`` has no such runtime check anywhere in
    ``update()`` (see the section docstring above): it is this function
    that gives the switch an effect, by substituting an identity
    calibration when off.
    """

    return replace(
        fitted_config,
        switches=switches,
        covariance=(
            fitted_config.covariance
            if switches.covariance_calibration
            else _IDENTITY_COVARIANCE_CALIBRATION
        ),
    )


def _existence_config_for_switches(
    baseline_existence_config: ExistenceFilterConfig,
    *,
    fitted_miss_likelihood_floor: float,
    switches: RobustObservationSwitches,
) -> ExistenceFilterConfig:
    """Build one ablation row's existence-track config.

    Invariant I8's miss-likelihood floor has no runtime switch check inside
    ``ExistenceFilter.update()`` either (see the section docstring above):
    it is this function that gives ``conditional_detection`` its I8-floor
    component, by substituting the strict-no-op ``0.0`` default (bit-for-
    bit identical to ``baseline_existence_config`` -- never the fitted
    value) when off.
    """

    return replace(
        baseline_existence_config,
        miss_likelihood_floor=(
            fitted_miss_likelihood_floor if switches.conditional_detection else 0.0
        ),
    )


def run_ablation(
    protocol: F9cProtocol,
    *,
    expected_runtime_cache_sha256: str = EXPECTED_RUNTIME_CACHE_SHA256,
    expected_evaluation_truth_sha256: str = EXPECTED_EVALUATION_TRUTH_SHA256,
) -> dict[str, Any]:
    """Task 12: replay the SAME hash-verified runtime cache once per
    ``ABLATION_ROWS`` switch combination, summarizing each with the
    unmodified ``f9c_belief.summarize_f9c``.

    Invariant I4: constructs no detector and no simulator. Every function
    this calls (``_load_cache_and_truth``, ``_replay_components``,
    ``_replay_rows``) reads only the frozen config and the already-cached
    candidates; ``YoloObjectDetector``/``create_gym_duckietown`` are
    constructed ONLY inside ``collect_final_rows``, never here.

    Returns a dict keyed by the seven row names in ``ABLATION_ROWS``, each
    value ``{"switches": {...}, "row_count": int, "metrics": <summarize_f9c
    metrics dict>}``, plus a ``"_provenance"`` key carrying the two cache
    hashes (recorded verbatim into ``artifacts/f9c_ablation_metrics.json``
    by ``main()`` so the ablation and the headline result are provably the
    same 3328 frames).

    The ``"baseline"`` row ALSO overrides ``initialization_threshold``/
    ``delete_threshold`` (see ``_BASELINE_INITIALIZATION_THRESHOLD``/
    ``_BASELINE_DELETE_THRESHOLD``'s own docstring): with the five named
    switches off but the frozen config's real thresholds (0.50/0.05) left
    in force, ``RobustPedestrianBeliefUpdater``'s own track-lifecycle
    policy -- which none of the five switches gate -- still deletes/
    discards tracks Baseline A's ``PedestrianBeliefUpdater`` never would,
    diverging on ~114/3328 frames. Every other row keeps the frozen
    thresholds unchanged, including ``"all_combined"``, which must match
    the actual Robust B (whose thresholds are exactly the frozen ones).
    """

    cache_frames, truth_by_key, runtime_cache_sha256, evaluation_truth_sha256 = _load_cache_and_truth(
        protocol,
        expected_runtime_cache_sha256=expected_runtime_cache_sha256,
        expected_evaluation_truth_sha256=expected_evaluation_truth_sha256,
    )

    settings = _ExperimentSettings(protocol.config_path)
    components = _replay_components(protocol, settings)
    fitted_robust_config = load_robust_observation_config(protocol.config_path)
    fitted_miss_likelihood_floor = load_miss_likelihood_floor(protocol.config_path)

    result: dict[str, Any] = {
        "_provenance": {
            "runtime_cache_sha256": runtime_cache_sha256,
            "evaluation_truth_sha256": evaluation_truth_sha256,
        }
    }
    for name, switches in ABLATION_ROWS:
        existence_config = _existence_config_for_switches(
            components["baseline_existence_config"],
            fitted_miss_likelihood_floor=fitted_miss_likelihood_floor,
            switches=switches,
        )
        robust_config = _robust_config_for_switches(fitted_robust_config, switches)
        if name == "baseline":
            # See the constants' own docstring above: neutralizes the
            # coordinator's OWN track-lifecycle policy, which is not gated
            # by any of the five named switches, so "baseline" reproduces
            # Baseline A's "never un-initialize" behavior exactly.
            robust_config = replace(
                robust_config,
                initialization_threshold=_BASELINE_INITIALIZATION_THRESHOLD,
                delete_threshold=_BASELINE_DELETE_THRESHOLD,
            )

        rows = _replay_rows(
            protocol,
            settings,
            cache_frames,
            truth_by_key,
            ekf_config=components["ekf_config"],
            measurement_noise=components["measurement_noise"],
            baseline_existence_config=components["baseline_existence_config"],
            baseline_bias=components["baseline_bias"],
            robust_existence_config=existence_config,
            robust_config=robust_config,
            robust_bias_frozen=components["robust_bias_frozen"],
            robust_bias_fitted=components["robust_bias_fitted"],
            observability_model=components["observability_model"],
        )
        metrics, _nis = summarize_f9c(rows, protocol=protocol)
        result[name] = {
            "switches": {
                "bias_refit": switches.bias_refit,
                "innovation_gate": switches.innovation_gate,
                "temporal_association": switches.temporal_association,
                "covariance_calibration": switches.covariance_calibration,
                "conditional_detection": switches.conditional_detection,
            },
            "row_count": len(rows),
            "track_lifecycle_thresholds": {
                "initialization_threshold": robust_config.initialization_threshold,
                "delete_threshold": robust_config.delete_threshold,
                "overridden_from_frozen_config": name == "baseline",
            },
            "metrics": metrics,
        }
        print(f"ablation row {name}: total_rows={len(rows)}", flush=True)

    # Diagnostic-only replay, NOT one of the seven ABLATION_ROWS: all five
    # switches off, but the FROZEN (non-overridden) track-lifecycle
    # thresholds every non-"baseline" row uses. Distinct from "baseline"
    # above, whose thresholds are overridden so it can match Baseline A
    # exactly -- which means "baseline" is no longer an apples-to-apples
    # comparison against "innovation_gate_only" (they now differ in BOTH
    # the gate switch AND the thresholds). This diagnostic isolates JUST
    # the gate switch under the same frozen thresholds "innovation_gate_only"
    # itself uses, which is what actually supports
    # ablation_known_limitations's inertness finding.
    diagnostic_switches = RobustObservationSwitches(
        bias_refit=False,
        innovation_gate=False,
        temporal_association=False,
        covariance_calibration=False,
        conditional_detection=False,
    )
    diagnostic_existence_config = _existence_config_for_switches(
        components["baseline_existence_config"],
        fitted_miss_likelihood_floor=fitted_miss_likelihood_floor,
        switches=diagnostic_switches,
    )
    diagnostic_robust_config = _robust_config_for_switches(fitted_robust_config, diagnostic_switches)
    diagnostic_rows = _replay_rows(
        protocol,
        settings,
        cache_frames,
        truth_by_key,
        ekf_config=components["ekf_config"],
        measurement_noise=components["measurement_noise"],
        baseline_existence_config=components["baseline_existence_config"],
        baseline_bias=components["baseline_bias"],
        robust_existence_config=diagnostic_existence_config,
        robust_config=diagnostic_robust_config,
        robust_bias_frozen=components["robust_bias_frozen"],
        robust_bias_fitted=components["robust_bias_fitted"],
        observability_model=components["observability_model"],
    )
    diagnostic_metrics, _diagnostic_nis = summarize_f9c(diagnostic_rows, protocol=protocol)
    result["_diagnostics"] = {
        "all_switches_off_frozen_thresholds": {
            "purpose": (
                "Isolates the five named switches from the 'baseline' row's "
                "track-lifecycle-threshold override, for a clean "
                "apples-to-apples comparison against 'innovation_gate_only' "
                "(same frozen thresholds, gate on vs off) -- see "
                "ablation_known_limitations."
            ),
            "row_count": len(diagnostic_rows),
            "metrics": diagnostic_metrics,
        }
    }

    return result


def ablation_known_limitations(ablation: dict[str, Any]) -> dict[str, Any]:
    """Task 12 disclosures, computed fresh from ``ablation`` (never hand-
    typed) so they cannot drift from the numbers they describe. Mirrors,
    and must not contradict, ``known_limitations``'s
    ``conditional_detection_per_class_probabilities_are_inert`` finding
    (Task 11) -- restated here because a reader of the ablation table alone
    would otherwise attribute the ``conditional_detection_only`` row's
    effect to the fitted per-class detection probabilities, which
    contribute nothing on this run's data (see that finding: the miss
    branch is dominated by the I8 floor, and the detected branch saturates
    regardless of class).
    """

    def _range_rmse(name: str) -> float | None:
        stats = ablation[name]["metrics"]["ekf"]["robust_b"]["range"]
        return stats.get("rmse")

    diagnostic_range_rmse = (
        ablation.get("_diagnostics", {})
        .get("all_switches_off_frozen_thresholds", {})
        .get("metrics", {})
        .get("ekf", {})
        .get("robust_b", {})
        .get("range", {})
        .get("rmse")
    )
    baseline_range_rmse = _range_rmse("baseline")
    gate_only_range_rmse = _range_rmse("innovation_gate_only")
    association_only_range_rmse = _range_rmse("temporal_association_only")
    all_combined_range_rmse = _range_rmse("all_combined")

    return {
        "conditional_detection_per_class_probabilities_are_inert": (
            "Restated from artifacts/f9c_belief_metrics.json's "
            "known_limitations (Task 11), which this ablation does not "
            "contradict: conditional_detection's effect in the "
            "'conditional_detection_only' row comes from invariant I3's "
            "in/out-of-domain routing and the I8 miss-likelihood floor, "
            "NOT from the fitted per-class detection probabilities "
            "(detection_probability_center/mid_fov/edge_fov) -- the miss "
            "branch is dominated by the I8 floor "
            "(miss_likelihood_floor=0.37362469458201386, far above every "
            "in-domain class's implied likelihood ratio) and the detected "
            "branch saturates existence regardless of class."
        ),
        "innovation_gate_only_is_structurally_inert_without_temporal_association": {
            "statement": (
                "'innovation_gate_only' is expected to be metrically IDENTICAL "
                "to a config with all five switches off (comparable ONLY under "
                "the SAME frozen track-lifecycle thresholds -- see "
                "'why_not_compared_against_baseline' below), not merely close: "
                "when temporal_association is off, "
                "RobustPedestrianBeliefUpdater.update forces "
                "predicted_measurement=None into MeasurementAssociator.associate, "
                "which unconditionally returns mode='initialization' -- and the "
                "gate branch (step 7/8 in robust_updater.py) is only ever "
                "reached when association.mode != 'initialization'. With no "
                "active temporal association there is no innovation for the gate "
                "to threshold, so switches.innovation_gate controls nothing: "
                "kinematic_measurement_accepted is set True unconditionally by "
                "the 'initialization' branch regardless of the gate switch. This "
                "is a structural property of the frozen, already-reviewed "
                "belief.robust_updater coordinator (see its own [association] "
                "config comment in configs/f9c_robust_belief_v1.toml: 'Setting "
                "them equal makes the gate unreachable whenever "
                "temporal_association is on ... the innovation_gate ablation "
                "switch has no observable effect' describes the analogous "
                "threshold-ordering case; this is the mode-branching case), not "
                "a bug in this ablation harness."
            ),
            "why_not_compared_against_baseline": (
                "'baseline' overrides initialization_threshold/delete_threshold "
                "(see run_ablation's docstring) so it can match Baseline A "
                "exactly; every other named row, including 'innovation_gate_only', "
                "keeps the frozen (0.50/0.05) thresholds. Comparing "
                "'innovation_gate_only' against 'baseline' directly would "
                "therefore conflate 'gate on vs off' with 'thresholds "
                "overridden vs not'. ablation['_diagnostics']"
                "['all_switches_off_frozen_thresholds'] is a non-headline "
                "internal replay -- all five switches off, frozen thresholds, "
                "like every non-'baseline' row -- that isolates the gate "
                "switch alone."
            ),
            "all_switches_off_frozen_thresholds_range_rmse_m": diagnostic_range_rmse,
            "innovation_gate_only_range_rmse_m": gate_only_range_rmse,
            "rows_are_metrically_identical": (
                None
                if diagnostic_range_rmse is None or gate_only_range_rmse is None
                else diagnostic_range_rmse == gate_only_range_rmse
            ),
            "temporal_association_only_range_rmse_m_for_contrast": association_only_range_rmse,
        },
        "components_are_not_additively_separable": {
            "statement": (
                "Per-row deltas in the seven-row table must NOT be read as "
                "individual component contributions that sum. Two of the "
                "single-component rows are WORSE than 'baseline' on range "
                "RMSE, not better: 'innovation_gate_only' and "
                "'temporal_association_only' both increase range RMSE "
                "relative to 'baseline', while 'all_combined' (all five "
                "switches on together) is the best row of the seven. Adding "
                "a component in isolation can degrade the metric that "
                "adding all five together improves -- the components "
                "interact, they do not contribute independent, addable "
                "improvements."
            ),
            "baseline_range_rmse_m": baseline_range_rmse,
            "innovation_gate_only_range_rmse_m": gate_only_range_rmse,
            "temporal_association_only_range_rmse_m": association_only_range_rmse,
            "all_combined_range_rmse_m": all_combined_range_rmse,
            "innovation_gate_only_worse_than_baseline": (
                None
                if baseline_range_rmse is None or gate_only_range_rmse is None
                else gate_only_range_rmse > baseline_range_rmse
            ),
            "temporal_association_only_worse_than_baseline": (
                None
                if baseline_range_rmse is None or association_only_range_rmse is None
                else association_only_range_rmse > baseline_range_rmse
            ),
            "all_combined_best_of_the_seven": (
                None
                if any(
                    value is None
                    for value in (
                        baseline_range_rmse,
                        gate_only_range_rmse,
                        association_only_range_rmse,
                        all_combined_range_rmse,
                    )
                )
                else all_combined_range_rmse
                < min(baseline_range_rmse, gate_only_range_rmse, association_only_range_rmse)
            ),
            "innovation_gate_only_degradation_is_explained": (
                "Yes -- see "
                "'innovation_gate_only_is_structurally_inert_without_"
                "temporal_association' above: the gate contributes nothing "
                "in that row (it is unreachable code without temporal "
                "association), so its RMSE simply reflects the noise floor "
                "of 'all switches off, frozen thresholds' with no gate "
                "correction applied -- there is no mechanism by which the "
                "gate switch alone could improve on 'baseline' here."
            ),
            "temporal_association_only_degradation_is_an_open_question": (
                "NOT explained by this run. This ablation does not identify "
                "why turning on temporal_association alone increases range "
                "RMSE relative to 'baseline'. Recorded here as an open "
                "question rather than left for a reader to guess at."
            ),
            "temporal_association_only_degradation_hypothesis_UNTESTED": (
                "One HYPOTHESIS, explicitly not tested by this run: "
                "association selects the candidate with minimum NIS against "
                "S = H P^- H^T + lambda*R (invariant I1's single provider, "
                "belief/robust_updater.py's _innovation_covariance). In the "
                "'temporal_association_only' row, covariance_calibration is "
                "off, so lambda*R uses the UNINFLATED base R -- roughly 10x "
                "smaller than the fitted range_scale=9.96243043243885 this "
                "run's calibration found necessary (see "
                "artifacts/f9c_calibration_metrics.json). Selecting by "
                "minimum NIS against a covariance the calibration says is "
                "an order of magnitude too small could bias which candidate "
                "gets selected/corrected-against under ambiguity, coupling "
                "temporal_association's effect to covariance_calibration "
                "through invariant I1's shared S. This would predict "
                "'temporal_association_only' improving once "
                "covariance_calibration is also on (consistent with "
                "'all_combined' being the best row) -- but this run does "
                "not isolate that interaction (no row in ABLATION_ROWS "
                "turns on exactly {temporal_association, "
                "covariance_calibration} together), so it remains a "
                "hypothesis, not a finding."
            ),
        },
    }


def known_limitations(rows: list[dict[str, object]], protocol: F9cProtocol) -> dict[str, object]:
    """Fix-round-2 durable-record disclosures, computed fresh from ``rows``
    and the frozen config every time this is called -- never hand-typed
    into the JSON -- so they cannot silently drift from the data they
    describe. Adds no new metric and changes no existing one: this is
    purely descriptive, appended as a sibling of ``"metrics"`` in the
    report, never merged into it.
    """

    with protocol.config_path.open("rb") as stream:
        cfg = tomllib.load(stream)
    conditional_detection = cfg["conditional_detection"]
    p_fa = float(conditional_detection["false_positive_probability"])
    floor = float(conditional_detection["miss_likelihood_floor"])
    per_class_p_d_eff = {
        "center": float(conditional_detection["detection_probability_center"]),
        "mid_fov": float(conditional_detection["detection_probability_mid_fov"]),
        "edge_fov": float(conditional_detection["detection_probability_edge_fov"]),
        "outside_domain": float(conditional_detection["detection_probability_outside_domain"]),
    }
    implied_lr_by_class = {
        name: (1.0 - p_d) / (1.0 - p_fa) for name, p_d in per_class_p_d_eff.items()
    }
    in_domain_classes = ("center", "mid_fov", "edge_fov")

    detected_rows = [row for row in rows if bool(row["detector_detected"])]
    existence_by_class: dict[str, dict[str, object]] = {}
    for cls in ("center", "mid_fov", "edge_fov", "outside_domain"):
        subset = [
            float(row["robust_b_existence_probability"])
            for row in detected_rows
            if row["robust_b_observability_class"] == cls
        ]
        existence_by_class[cls] = {
            "count": len(subset),
            "minimum": None if not subset else min(subset),
            "mean": None if not subset else float(np.mean(subset)),
        }
    minimum_existence_among_detected = (
        None
        if not detected_rows
        else min(float(row["robust_b_existence_probability"]) for row in detected_rows)
    )

    initialization_rows = [row for row in rows if row["robust_b_frame_mode"] == "initialization"]
    initialization_outside_domain_count = sum(
        1 for row in initialization_rows if row["robust_b_observability_class"] == "outside_domain"
    )

    range_selected = [row for row in rows if bool(row["robust_b_belief_initialized"])]
    z = np.array(
        [
            (float(row["robust_b_belief_range_m"]) - float(row["gt_range_m"]))
            / float(row["robust_b_belief_range_std_m"])
            for row in range_selected
        ],
        dtype=float,
    )
    z_mean = float(np.mean(z)) if len(z) else None
    z_std = float(np.std(z, ddof=0)) if len(z) else None
    z_excess_kurtosis = (
        float(np.mean((z - z_mean) ** 4) / z_std**4 - 3.0)
        if len(z) and z_std not in (None, 0.0)
        else None
    )

    return {
        "conditional_detection_per_class_probabilities_are_inert": {
            "statement": (
                "conditional_detection contributes to Robust B's behaviour only "
                "through invariant I3's in-domain vs OUTSIDE_DOMAIN routing, NOT "
                "through the fitted per-class detection probabilities themselves. "
                "Toggling conditional_detection in an ablation should not be "
                "expected to reflect the fitted per-class magnitudes "
                "(detection_probability_center/mid_fov/edge_fov) -- both branches "
                "that would carry that information are saturated/floored inert on "
                "this run's data."
            ),
            "miss_branch_dominated_by_the_i8_floor": {
                "miss_likelihood_floor": floor,
                "implied_lr_by_class": implied_lr_by_class,
                "floor_dominates_every_in_domain_class": all(
                    floor > implied_lr_by_class[name] for name in in_domain_classes
                ),
                "note": (
                    "max(LR_nominal, miss_likelihood_floor) returns the floor for "
                    "center/mid_fov/edge_fov regardless of which in-domain class a "
                    "miss is predicted in, since every in-domain LR is far below "
                    "the floor. OUTSIDE_DOMAIN never reaches this branch at all "
                    "(invariant I3: no likelihood applied to an outside-domain "
                    "miss)."
                ),
            },
            "detected_branch_saturates_regardless_of_class": {
                # Named detected_frame_count, not detected_row_count: the
                # substring "row_count" collides with
                # verify_f9c_artifacts.py's blanket
                # every-key-named-row_count == len(rows) cross-check (see
                # the identical fix already applied to
                # f9c_belief.py's by_distance_bin/by_fov_region). This is a
                # SUBSET count, not the total.
                "detected_frame_count": len(detected_rows),
                "minimum_existence_probability_among_detected_rows": (
                    minimum_existence_among_detected
                ),
                "existence_by_class": existence_by_class,
                "note": (
                    "P_D^eff is not floored on the detected branch, but existence "
                    "saturates to a high value across every class on this run's "
                    "data, so the per-class magnitude is inert here too -- via "
                    "saturation, not flooring."
                ),
            },
            "initialization_frames_mostly_classify_outside_domain": {
                "initialization_frame_count": len(initialization_rows),
                "outside_domain_count": initialization_outside_domain_count,
                "reason": (
                    "The zero-prior default state (initial_belief()) has "
                    "range_mean_m=0.0, bearing_mean_rad=0.0, i.e. x_left=0, "
                    "y_forward=0. PredictedObservabilityModel.classify's "
                    "`if y_forward <= 0.0: return OUTSIDE_DOMAIN` branch fires on "
                    "that default state before any track exists, independent of "
                    "where the pedestrian actually is."
                ),
            },
        },
        "coverage_overshoot_is_not_a_tail_effect": {
            "statement": (
                "An initial hypothesis that heavy-tailed range errors (rather than "
                "a uniformly over-sized posterior sigma) explained "
                "coverage_68=0.852 overshooting the [0.60, 0.76] band was tested "
                "directly against this run's data and is REFUTED."
            ),
            "z_score_distribution": {
                "definition": (
                    "z = (robust_b_belief_range_m - gt_range_m) / "
                    "robust_b_belief_range_std_m, over every robust_b-initialized "
                    "row"
                ),
                "n": len(z),
                "mean": z_mean,
                "std": z_std,
                "excess_kurtosis_fisher": z_excess_kurtosis,
                "interpretation": (
                    "std < 1 and excess kurtosis < 0 (slightly LIGHTER-tailed than "
                    "Gaussian, not heavier) together are inconsistent with a "
                    "heavy-tail explanation. The overshoot is better described as "
                    "a fairly uniform ~25-30% over-sized posterior sigma across "
                    "the bulk of the distribution, not a small number of extreme "
                    "errors inflating a nominally-correct sigma."
                ),
            },
            "remaining_hypothesis_not_established_by_this_run": (
                "range_posterior_floor_m/bearing_posterior_floor_rad were "
                "calibrated from tau_seed/tau_episode estimated on 8 calibration "
                "seeds (6101-6108) and then applied unchanged to 4 different "
                "final-evaluation seeds (7101-7104); a floor fit from a small, "
                "noisy per-seed variance estimate producing a wider-than-needed "
                "interval on a disjoint seed set is plausible, but this run does "
                "not test that hypothesis -- it has not been established, only "
                "proposed."
            ),
        },
        "render_vs_replay_perception_fields_are_inert": {
            "statement": (
                "PerceptionObservation.ego/.road differ between the render path "
                "(live EgoObservation/RoadMeasurement from the simulator) and the "
                "replay path (EgoObservation(0.0, 0.0, "
                "ego_motion.linear_velocity_mps, ego_motion.yaw_rate_rad_s), "
                "road=None). This is inert for the pedestrian EKF and existence "
                "math: belief/updater.py:120-125 shows perception.ego is used "
                "only to set BeliefState.ego (echoed through, never read by the "
                "kinematic/existence update), and perception.road only sets "
                "BeliefState.road via _road_belief, which this module's "
                "row-building never reads. Verified directly against updater.py, "
                "not assumed."
            ),
        },
        "camera_calibration_gap_is_largely_inert_given_conditional_detection_inertness": (
            "The disclosed camera-calibration gap (nominal vs "
            "per-episode-randomized camera for PredictedObservabilityModel) is "
            "largely inert given the conditional_detection finding above: a "
            "CENTER<->MID_FOV<->EDGE_FOV misclassification changes nothing, since "
            "the miss-branch floor and the detected-branch saturation dominate "
            "regardless of which in-domain class is predicted. Only a "
            "misclassification that flips in-domain <-> OUTSIDE_DOMAIN "
            "(invariant I3's routing) could alter behaviour."
        ),
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    import csv

    if not rows:
        raise RuntimeError("F9c final evaluation produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f9c_robust_belief_v1.toml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--replay-from-cache",
        action="store_true",
        help=(
            "Reconstruct the CSV/metrics artifacts from the already-written "
            "runtime cache and evaluation truth instead of rendering. "
            "Touches no simulator, no detector, no GPU. Use only when the "
            "final-evaluation seeds have already been rendered exactly once "
            "and the render's own cache/truth files are on disk."
        ),
    )
    mode.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Task 12: replay the already-written runtime cache seven times, "
            "once per RobustObservationSwitches combination, and write "
            "artifacts/f9c_ablation_metrics.json. Mutually exclusive with "
            "the final run (--replay-from-cache or a live render): touches "
            "no simulator, no detector, no GPU (invariant I4)."
        ),
    )
    args = parser.parse_args()

    protocol = load_f9c_protocol(args.config, require_frozen=True)

    if args.ablation:
        ablation = run_ablation(protocol)
        report = {
            "schema_version": 1,
            "gate": "F9c",
            "status": "ablation_from_runtime_cache",
            "source": {
                "config_path": str(protocol.config_path),
                "config_sha256": protocol.config_sha256,
                "checkpoint_sha256": protocol.checkpoint_sha256,
                "final_evaluation_seeds": list(protocol.final_evaluation_seeds),
            },
            "runtime_cache": {
                "path": str(protocol.artifacts["runtime_cache"]),
                "sha256": ablation["_provenance"]["runtime_cache_sha256"],
            },
            "evaluation_truth": {
                "path": str(protocol.artifacts["evaluation_truth"]),
                "sha256": ablation["_provenance"]["evaluation_truth_sha256"],
            },
            "no_rerender_statement": (
                "This run REPLAYED the already-written runtime cache and "
                "evaluation truth from the original 2026-08 final-evaluation "
                "render, seven times (once per ABLATION_ROWS switch "
                "combination). No simulator, detector, or GPU was invoked "
                "(invariant I4); seeds 7101-7104 were rendered exactly once, "
                "by Task 11, and are not rendered again here."
            ),
            "rows": {name: ablation[name] for name, _ in ABLATION_ROWS},
            "diagnostics": ablation.get("_diagnostics", {}),
            "known_limitations": ablation_known_limitations(ablation),
        }
        ablation_path = protocol.artifacts["ablation_metrics_json"]
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ablation_metrics": str(ablation_path),
                    "runtime_cache_sha256": ablation["_provenance"]["runtime_cache_sha256"],
                    "row_names": [name for name, _ in ABLATION_ROWS],
                    "row_counts": {
                        name: ablation[name]["row_count"] for name, _ in ABLATION_ROWS
                    },
                },
                indent=2,
            )
        )
        return

    if args.replay_from_cache:
        rows, runtime_cache_sha256, evaluation_truth_sha256 = replay_from_cache(protocol)
        error_cases: list[str] = []  # already on disk from the original render; not regenerated.
        reconstructed = True
        no_rerender_statement = (
            "This run was RECONSTRUCTED from the runtime cache and evaluation "
            "truth written by the original 2026-08 final-evaluation render. "
            "No simulator, detector, or GPU was invoked to produce this "
            "artifact; seeds 7101-7104 were rendered exactly once."
        )
    else:
        settings = _ExperimentSettings(protocol.config_path)
        rows, cache_frames, truth_frames, error_cases = collect_final_rows(protocol, settings)
        runtime_cache_path = protocol.artifacts["runtime_cache"]
        evaluation_truth_path = protocol.artifacts["evaluation_truth"]
        runtime_cache_sha256 = write_runtime_cache(runtime_cache_path, cache_frames)
        evaluation_truth_sha256 = write_evaluation_truth(evaluation_truth_path, truth_frames)
        reconstructed = False
        no_rerender_statement = "This run rendered seeds 7101-7104 directly (not a replay)."

    metrics, nis = summarize_f9c(rows, protocol=protocol)

    report = {
        "schema_version": 1,
        "gate": "F9c",
        "status": "final_evaluation_complete_not_posthoc_tuned",
        "row_count": len(rows),
        "source": {
            "config_path": str(protocol.config_path),
            "config_sha256": protocol.config_sha256,
            "checkpoint_sha256": protocol.checkpoint_sha256,
            "final_evaluation_seeds": list(protocol.final_evaluation_seeds),
            "calibration_seeds_not_read": list(protocol.calibration_seeds),
            "runtime_path_baseline_a": (
                "front_rgb -> frozen_yolo -> highest_confidence_selection -> "
                "F9b_frozen_bias -> frozen_F7_ekf -> frozen_existence"
            ),
            "runtime_path_robust_b": (
                "front_rgb -> frozen_yolo -> candidates -> F9c_bias -> "
                "association -> innovation_gate -> lambda_inflated_R -> "
                "frozen_F7_ekf -> P_D_eff_existence(I8_floor) -> "
                "posterior_variance_floor"
            ),
            "privileged_truth_use": "post-update evaluation only, after both updaters are stepped",
        },
        "reconstruction": {
            "reconstructed_from_cache": reconstructed,
            "no_rerender_statement": no_rerender_statement,
            "runtime_cache_sha256": runtime_cache_sha256,
            "evaluation_truth_sha256": evaluation_truth_sha256,
            "disclosed_limitation": (
                None
                if not reconstructed
                else (
                    "Robust B's observability classification (robust_b_observability_class, "
                    "robust_b_effective_detection_probability, and any resulting "
                    "existence/track-lifecycle divergence) was computed using gym-duckietown's "
                    "NOMINAL (non-domain-randomized) camera calibration, not the per-episode "
                    "randomized one the original render used, because the per-episode value "
                    "is not part of the runtime-cache schema (I5). This affects ONLY Robust "
                    "B's existence-filter trajectory -- never the EKF kinematic state "
                    "(range/bearing/std/NIS/innovation), never gate/association decisions, "
                    "and never Baseline A, none of which depend on observability "
                    "classification. See the module docstring."
                )
            ),
        },
        "runtime_cache": {
            "path": str(protocol.artifacts["runtime_cache"]),
            "sha256": runtime_cache_sha256,
            "frame_count": None,
        },
        "evaluation_truth": {
            "path": str(protocol.artifacts["evaluation_truth"]),
            "sha256": evaluation_truth_sha256,
            "frame_count": None,
        },
        "metrics": metrics,
        # Fix-round-2: durable-record disclosures, a SIBLING of "metrics" --
        # computed fresh from `rows` in known_limitations(), never merged
        # into "metrics" and never used to alter a metric value.
        "known_limitations": known_limitations(rows, protocol),
        "error_case_images": error_cases,
    }

    validation_path = protocol.artifacts["validation_csv"]
    metrics_path = protocol.artifacts["belief_metrics_json"]
    nis_path = protocol.artifacts["nis_metrics_json"]
    _write_csv(rows, validation_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    nis_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "F9c",
                "config_sha256": protocol.config_sha256,
                "nis": nis,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "validation_csv": str(validation_path),
                "belief_metrics": str(metrics_path),
                "nis_metrics": str(nis_path),
                "rows": len(rows),
                "reconstructed_from_cache": reconstructed,
                "runtime_cache_sha256": runtime_cache_sha256,
                "support_check": metrics["support_check"],
                "ekf": metrics["ekf"],
                "robustness": metrics["robustness"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
