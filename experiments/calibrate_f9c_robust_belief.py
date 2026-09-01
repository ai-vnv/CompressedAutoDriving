"""Gate F9c calibration experiment: fit bias / covariance-scale / P_D^eff on
calibration-only seeds 6101-6108.

**Two coordinators run in lockstep, for two different jobs.**

- ``runtime_updater`` uses the CURRENT (as-checked-in, not-yet-fit)
  ``RobustObservationConfig`` loaded from ``configs/f9c_robust_belief_v1.toml``
  -- bias 0/0, lambda_r = lambda_beta = 1, posterior floor 0 -- fed the FULL
  candidate list every frame, exactly like the real runtime. Its
  ``RobustStepRecord`` supplies the CSV's ``detector_detected``,
  ``kinematic_measurement_accepted``, ``predicted_observability_class``, and
  ``predicted_nis`` columns: honest diagnostics of the config as it stands
  today, decoupled from anything this script goes on to fit.
- ``gt_tracking_updater`` is a SEPARATE diagnostic coordinator with
  ``temporal_association=False`` and ``innovation_gate=False``, fed ONLY the
  ground-truth-correct candidate each frame (or none). With those two
  switches off, ``MeasurementAssociator.associate`` always takes the
  "initialization" branch (see ``belief/robust_updater.py`` step 5/6),
  bypassing BOTH association's own chi-square gate and the explicit
  innovation gate -- so this track corrects on the GT-correct candidate
  EVERY time one exists, never rejecting it for having a large NIS. This is
  what invariant I6 requires: the lambda-fitting ingredients must be
  captured independently of any gate/association decision, not only on
  frames the (not-yet-calibrated) gate happened to accept. Its
  ``last_ekf_diagnostics.innovation_covariance`` is the EXACT invariant-I1
  ``S`` at lambda = 1 for that candidate;
  ``evaluation.f9c_calibration.lambda_fit_ingredients`` decomposes it into
  ``H P^- H^T`` and the base (pre-inflation) R, which is exact at lambda = 1
  (see that function's docstring for why re-scoring at a different fitted
  bias is a documented small-signal approximation).

Ground truth (silhouette assessment, IoU matching) is read only AFTER the
complete runtime chain (detector -> projection -> both coordinators) has
already produced its output for the frame, exactly as F9a/F9b do.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import (
    CovarianceCalibration,
    estimate_nested_variance_components,
    posterior_floor_from_components,
)
from duckie_pomdp.belief.measurement_association import CandidateMeasurement
from duckie_pomdp.belief.observability import PredictedObservabilityModel
from duckie_pomdp.belief.pedestrian_ekf import load_pedestrian_ekf_config
from duckie_pomdp.belief.robust_updater import (
    RobustObservationSwitches,
    RobustPedestrianBeliefUpdater,
)
from duckie_pomdp.belief.updater import initial_belief, load_existence_filter_config
from duckie_pomdp.dataset.annotations import assess_silhouette
from duckie_pomdp.dataset.config import load_dataset_config
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, ObjectClass, yolo_class_id
from duckie_pomdp.evaluation.f9c_calibration import (
    adjust_miss_likelihood_floor_for_false_positive_rate,
    detection_by_range_and_fov,
    eligible_distance_bin_counts,
    eligible_fov_region_counts,
    fit_covariance_scales,
    fit_effective_detection,
    fit_miss_likelihood_floor,
    frozen_bias_correction_from_fit,
    joint_nis_median,
    lambda_excluded_sample_report,
    lambda_fit_ingredients,
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
    observability_confusion_matrix,
    predicted_nis_diagnostic,
    range_bias_for_row,
    select_bias_model,
)
from duckie_pomdp.evaluation.f9c_protocol import F9cProtocol, load_f9c_protocol, sha256
from duckie_pomdp.evaluation.yolo_detection import intersection_over_union
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.f9_pipeline import YoloPedestrianMeasurementPipeline
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

# The GT-tracking updater must always accept its (single) candidate,
# regardless of NIS, so the lambda-fitting ingredients are captured on
# EVERY ground-truth-correct frame -- never only on frames the (not yet
# calibrated) gate/association would have accepted. See module docstring.
_GT_TRACKING_SWITCHES = RobustObservationSwitches(
    bias_refit=True,
    innovation_gate=False,
    temporal_association=False,
    covariance_calibration=True,
    conditional_detection=True,
)
_GT_TRACKING_COVARIANCE = CovarianceCalibration(1.0, 1.0, 0.0, 0.0)


def _optional(value: Any) -> Any:
    return "" if value is None else value


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


def _bounds(values: list[float]) -> tuple[float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("bounds must be an ordered pair")
    return bounds


class _ExperimentSettings:
    """Ancillary rendering/detector/calibration settings not exposed by the
    thin ``F9cProtocol`` (which exists for freeze-boundary guards, not
    rendering plumbing). Read directly from the same TOML file."""

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

        for section, key in (
            ("measurement_model", "parameters_frozen"),
            ("covariance_calibration", "parameters_frozen"),
            ("conditional_detection", "parameters_frozen"),
        ):
            if bool(data[section][key]):
                raise SystemExit(
                    f"refusing to overwrite already-frozen F9c calibration "
                    f"({section}.{key} is true)"
                )


def _scenario_for(settings: _ExperimentSettings, spec, seed: int, scenario_index: int):
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


def collect_calibration_rows(
    protocol: F9cProtocol,
    settings: _ExperimentSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Returns ``(rows, episode_warnings)``. ``episode_warnings`` records any
    episode that ended before its intended step count (e.g. an off-road
    "invalid-pose" excursion) -- every row collected up to that point is
    still genuine data and is kept; only the shortfall is reported."""

    episode_warnings: list[dict[str, object]] = []
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
    # Invariant I8: wire the (currently 0.0, no-op) miss-likelihood floor
    # from [conditional_detection] into the existence filter used by both
    # coordinators below. This run does not depend on its value -- L_mean,
    # the global P_D/P_FA, and every other fitted quantity are statistics of
    # the raw detector/GT columns, never of the existence filter's own
    # trajectory -- but the wiring is exercised here so it is ready for
    # whichever task freezes a non-zero value.
    existence_config = replace(
        load_existence_filter_config(protocol.config_path),
        miss_likelihood_floor=load_miss_likelihood_floor(protocol.config_path),
    )
    runtime_config = load_robust_observation_config(protocol.config_path)
    bias_frozen = load_frozen_bias_correction(
        protocol.config_path, section="baseline_measurement_model"
    )
    bias_fitted_placeholder = load_frozen_bias_correction(
        protocol.config_path, section="measurement_model"
    )
    gt_config = replace(
        runtime_config,
        switches=_GT_TRACKING_SWITCHES,
        covariance=_GT_TRACKING_COVARIANCE,
    )
    identity_bias = FrozenBiasCorrection.identity()

    rows: list[dict[str, object]] = []
    for seed in protocol.calibration_seeds:
        for scenario_index, spec in enumerate(protocol.scenarios):
            if not spec.use_for_calibration:
                continue
            scenario = _scenario_for(settings, spec, seed, scenario_index)
            episode = f"calibration_{seed}_{spec.name}"
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

                runtime_updater = RobustPedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    measurement_noise=measurement_noise,
                    existence_config=existence_config,
                    bias_frozen=bias_frozen,
                    bias_fitted=bias_fitted_placeholder,
                    observability_model=observability_model,
                    config=runtime_config,
                )
                gt_tracking_updater = RobustPedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    measurement_noise=measurement_noise,
                    existence_config=existence_config,
                    bias_frozen=identity_bias,
                    bias_fitted=identity_bias,
                    observability_model=observability_model,
                    config=gt_config,
                )

                runtime_belief = initial_belief(
                    observation.ego, existence_prior=existence_config.prior_probability
                )
                gt_belief = initial_belief(
                    observation.ego, existence_prior=existence_config.prior_probability
                )
                timestamp_s = 0.0
                dt_s = 1.0 / 30.0
                previous_action = PolicyAction(0.0, 0.0)

                for frame in range(spec.steps + 1):
                    # Complete runtime chain (both coordinators) finishes
                    # before any privileged read.
                    result = runtime_pipeline.observe(observation.front_rgb)

                    valid_candidates = [
                        candidate
                        for candidate in result.duckie_candidates
                        if candidate.projection_error is None
                    ]
                    runtime_candidates = [
                        CandidateMeasurement(
                            measurement=candidate.measurement,
                            confidence=candidate.detection.confidence,
                            bbox_key=_bbox_key(candidate.detection.bounding_box),
                        )
                        for candidate in valid_candidates
                    ]
                    ego_motion = observation.ego.motion
                    runtime_belief, runtime_record = runtime_updater.update(
                        runtime_belief,
                        previous_action,
                        ego_motion,
                        runtime_candidates,
                        dt_s,
                    )

                    # Evaluation-only boundary: ground truth read AFTER the
                    # complete runtime chain has produced its output.
                    privileged = integration.privileged.read()
                    silhouettes = {
                        item.object_kind: item
                        for item in integration.projection_validation.sample_object_silhouettes(
                            ("duckie",)
                        )
                    }
                    silhouette = silhouettes.get("duckie")
                    gt_box = None if silhouette is None else silhouette.bounding_box
                    visible_pixels = (
                        0 if silhouette is None else silhouette.visible_pixel_count
                    )
                    decision = assess_silhouette(
                        class_id=yolo_class_id(ObjectClass.DUCKIE),
                        box=gt_box,
                        visible_pixel_count=visible_pixels,
                        image_width_px=settings.image_width_px,
                        image_height_px=settings.image_height_px,
                        rules=annotation.duckie_rules,
                    )
                    truth = privileged.true_pomdp_state.pedestrian
                    if truth.range_m is None or truth.bearing_rad is None:
                        raise RuntimeError(f"{episode} lost pedestrian truth")

                    selected = result.selected_duckie
                    selected_box = None if selected is None else selected.bounding_box
                    iou = (
                        None
                        if not decision.accepted
                        or gt_box is None
                        or selected_box is None
                        else intersection_over_union(gt_box, selected_box)
                    )
                    correct = bool(
                        result.pedestrian.detected
                        and decision.accepted
                        and iou is not None
                        and iou >= settings.matching_iou_threshold
                    )
                    raw_range = result.pedestrian.range_m
                    raw_bearing = result.pedestrian.bearing_rad

                    # GT-tracking updater: fed ONLY the GT-verified-correct
                    # candidate (invariant I6). result.pedestrian IS the
                    # already-projected measurement for `selected`.
                    gt_candidates: list[CandidateMeasurement] = []
                    if correct:
                        assert selected is not None and raw_range is not None
                        gt_candidates = [
                            CandidateMeasurement(
                                measurement=result.pedestrian,
                                confidence=selected.confidence,
                                bbox_key=_bbox_key(selected.bounding_box),
                            )
                        ]
                    gt_belief, gt_record = gt_tracking_updater.update(
                        gt_belief, previous_action, ego_motion, gt_candidates, dt_s
                    )

                    ingredients = (
                        lambda_fit_ingredients(
                            gt_tracking_updater.last_ekf_diagnostics,
                            measurement_noise,
                            raw_range,
                        )
                        if correct
                        else None
                    )

                    row: dict[str, object] = {
                        "episode": episode,
                        "seed": seed,
                        "scenario": spec.name,
                        "frame": frame,
                        "eligible_visible": decision.accepted,
                        "detector_detected": runtime_record.detector_detected,
                        "kinematic_measurement_accepted": (
                            runtime_record.kinematic_measurement_accepted
                        ),
                        "duckie_detection_count": result.duckie_detection_count,
                        "candidate_count": len(runtime_candidates),
                        "selected_confidence": _optional(
                            None if selected is None else selected.confidence
                        ),
                        "selected_iou": _optional(iou),
                        "selected_correct_iou50": correct,
                        "raw_range_m": _optional(raw_range),
                        "raw_bearing_rad": _optional(raw_bearing),
                        "gt_range_m": truth.range_m,
                        "gt_bearing_rad": truth.bearing_rad,
                        "range_error_m": _optional(
                            raw_range - truth.range_m if correct else None
                        ),
                        "bearing_error_rad": _optional(
                            wrap_angle(raw_bearing - truth.bearing_rad)
                            if correct and raw_bearing is not None
                            else None
                        ),
                        "distance_bin": _distance_bin(
                            truth.range_m, settings.near_max_m, settings.medium_max_m
                        ),
                        "fov_region": _fov_region(
                            gt_box if decision.accepted else None,
                            settings.image_width_px,
                        ),
                        "predicted_observability_class": (
                            runtime_record.observability_class.value
                        ),
                        "predicted_nis": _optional(runtime_record.nis),
                        # Extra columns beyond the brief's required minimum:
                        # audit trail plus the lambda-fit ingredients.
                        "valid_projected_measurement": bool(result.pedestrian.detected),
                        "correct_class": True,
                        "gt_pedestrian_exists": bool(truth.exists),
                        "lambda_fit_eligible": ingredients is not None,
                        "hph_rr": _optional(None if ingredients is None else ingredients["hph_rr"]),
                        "hph_rb": _optional(None if ingredients is None else ingredients["hph_rb"]),
                        "hph_bb": _optional(None if ingredients is None else ingredients["hph_bb"]),
                        "base_range_variance_m2": _optional(
                            None if ingredients is None else ingredients["base_range_variance_m2"]
                        ),
                        "base_bearing_variance_rad2": _optional(
                            None
                            if ingredients is None
                            else ingredients["base_bearing_variance_rad2"]
                        ),
                        "range_innovation_m": _optional(
                            None if ingredients is None else ingredients["range_innovation_m"]
                        ),
                        "bearing_innovation_rad": _optional(
                            None if ingredients is None else ingredients["bearing_innovation_rad"]
                        ),
                    }
                    rows.append(row)

                    if frame == spec.steps:
                        break
                    transition = integration.agent.step(spec.action)
                    diagnostics = integration.diagnostics.read()
                    if transition.terminated or transition.truncated:
                        # A scenario ending early (e.g. an off-road/collision
                        # "invalid-pose" excursion from the randomized start
                        # perturbation) must not lose the other 79 episodes.
                        # Every frame already appended above is genuine data;
                        # only the intended remainder of this one episode is
                        # short. Reported prominently in the metrics artifact
                        # and the task report -- this is exactly the kind of
                        # near-range scenario-geometry finding Task 2 deferred
                        # to this task to measure.
                        warning = {
                            "episode": episode,
                            "seed": seed,
                            "scenario": spec.name,
                            "reached_frame": frame,
                            "intended_steps": spec.steps,
                            "done_code": diagnostics.done_code,
                        }
                        episode_warnings.append(warning)
                        print(
                            f"WARNING: {episode} ended early at frame {frame}/"
                            f"{spec.steps} ({diagnostics.done_code}); keeping "
                            f"{len(rows)} rows collected so far and moving on",
                            flush=True,
                        )
                        break
                    observation = transition.observation
                    dt_s = diagnostics.timestamp_s - timestamp_s
                    timestamp_s = diagnostics.timestamp_s
                    previous_action = spec.action
            finally:
                integration.close()
            print(f"completed {episode}: total_rows={len(rows)}", flush=True)
    return rows, episode_warnings


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    import csv

    if not rows:
        raise ValueError("cannot write empty F9c calibration CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_metrics_report(
    rows: list[dict[str, object]],
    protocol: F9cProtocol,
    settings: _ExperimentSettings,
    episode_warnings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    bias_model, bias_fit = select_bias_model(rows)
    fitted_bias = frozen_bias_correction_from_fit(
        bias_fit, near_max_m=settings.near_max_m, medium_max_m=settings.medium_max_m
    )

    lambda_r, lambda_beta = fit_covariance_scales(rows, bias=fitted_bias)
    joint_median = joint_nis_median(
        rows, bias=fitted_bias, lambda_r=lambda_r, lambda_beta=lambda_beta
    )
    excluded_report = lambda_excluded_sample_report(rows)
    gate_threshold = load_robust_observation_config(
        protocol.config_path
    ).gate.chi_square_threshold
    nis_diagnostic = predicted_nis_diagnostic(rows, gate_threshold=gate_threshold)

    # Invariant I8: the miss-likelihood floor. L_mean/LR_nominal/LR_floor
    # are statistics of the raw detector/GT columns, independent of any
    # existence-filter mechanism, so they are valid regardless of what
    # miss_likelihood_floor the render pass itself used.
    #
    # The self-fit false_positive_probability (from "naturally ineligible"
    # frames) is contaminated by borderline-visibility frames the detector
    # correctly still fires on -- this run collected no F9a-style
    # counterfactual hidden-pedestrian negatives. It is reported for
    # transparency, but the RECOMMENDED floor substitutes F9b's own
    # counterfactual-based frozen false_positive_probability
    # ([existence].false_positive_probability), a materially more
    # trustworthy estimate of a genuine false-alarm rate.
    miss_floor_fit = fit_miss_likelihood_floor(rows)
    frozen_false_positive_probability = load_existence_filter_config(
        protocol.config_path
    ).false_positive_probability
    miss_floor_adjusted = adjust_miss_likelihood_floor_for_false_positive_rate(
        miss_floor_fit, false_positive_probability=frozen_false_positive_probability
    )

    # The range confound behind the EDGE_FOV > CENTER P_D^eff inversion.
    range_fov_cross_tab = detection_by_range_and_fov(rows)

    matched = [row for row in rows if row["selected_correct_iou50"]]
    range_groups: dict[tuple[str, str], list[float]] = {}
    bearing_groups: dict[tuple[str, str], list[float]] = {}
    for row in matched:
        key = (str(row["seed"]), str(row["episode"]))
        range_residual = float(row["range_error_m"]) - range_bias_for_row(bias_fit, row)
        bearing_residual = float(row["bearing_error_rad"]) - float(bias_fit["bearing_bias_rad"])
        range_groups.setdefault(key, []).append(range_residual)
        bearing_groups.setdefault(key, []).append(bearing_residual)

    range_components = estimate_nested_variance_components(range_groups)
    bearing_components = estimate_nested_variance_components(bearing_groups)
    sigma_floor_r = posterior_floor_from_components(range_components)
    sigma_floor_beta = posterior_floor_from_components(bearing_components)

    detection = fit_effective_detection(rows)

    return {
        "schema_version": 1,
        "gate": "F9c",
        "status": "calibration_only_candidate_not_yet_frozen",
        "source": {
            "config_path": str(protocol.config_path),
            "prefreeze_config_sha256": protocol.config_sha256,
            "checkpoint_sha256": protocol.checkpoint_sha256,
            "calibration_seeds": list(protocol.calibration_seeds),
            "final_evaluation_seeds_reserved_not_read": list(
                protocol.final_evaluation_seeds
            ),
        },
        "row_count": len(rows),
        "episode_warnings_early_termination": episode_warnings or [],
        "eligible_distance_bin_counts": eligible_distance_bin_counts(rows),
        "eligible_fov_region_counts": eligible_fov_region_counts(rows),
        "minimum_support_target": protocol.minimum_support,
        "observability_confusion_matrix_predicted_x_gt_fov": (
            observability_confusion_matrix(rows)
        ),
        "bias": {
            "selected_model": bias_model,
            "fit": bias_fit,
        },
        "covariance_scales": {
            "lambda_r": lambda_r,
            "lambda_beta": lambda_beta,
            "joint_2dof_nis_median_at_fit": joint_median,
            "chi2_2dof_median_target": 1.3862943611198906,
            "excluded_sample_report": excluded_report,
        },
        "predicted_nis_diagnostic": nis_diagnostic,
        "miss_likelihood_floor": {
            "self_fit_contaminated_false_positive_rate": miss_floor_fit,
            "adjusted_using_frozen_f9b_false_positive_rate": miss_floor_adjusted,
        },
        "detection_by_range_and_fov": range_fov_cross_tab,
        "variance_components": {
            "range": {
                "seed_variance": range_components.seed_variance,
                "episode_variance": range_components.episode_variance,
                "within_variance": range_components.within_variance,
                "seed_count": range_components.seed_count,
                "episode_count": range_components.episode_count,
                "sample_count": range_components.sample_count,
                "sigma_floor_m": sigma_floor_r,
            },
            "bearing": {
                "seed_variance": bearing_components.seed_variance,
                "episode_variance": bearing_components.episode_variance,
                "within_variance": bearing_components.within_variance,
                "seed_count": bearing_components.seed_count,
                "episode_count": bearing_components.episode_count,
                "sample_count": bearing_components.sample_count,
                "sigma_floor_rad": sigma_floor_beta,
            },
        },
        "effective_detection": detection,
        "recommended_covariance_calibration": {
            "range_scale": lambda_r,
            "bearing_scale": lambda_beta,
            "range_posterior_floor_m": sigma_floor_r,
            "bearing_posterior_floor_rad": sigma_floor_beta,
        },
        "recommended_conditional_detection_miss_likelihood_floor": (
            miss_floor_adjusted["lr_floor"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f9c_robust_belief_v1.toml",
    )
    args = parser.parse_args()

    protocol = load_f9c_protocol(args.config, require_frozen=False)
    settings = _ExperimentSettings(protocol.config_path)

    rows, episode_warnings = collect_calibration_rows(protocol, settings)
    report = build_metrics_report(rows, protocol, settings, episode_warnings)

    csv_path = protocol.artifacts["calibration_csv"]
    metrics_path = protocol.artifacts["calibration_metrics_json"]
    _write_csv(rows, csv_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "calibration_csv": str(csv_path),
                "calibration_metrics_json": str(metrics_path),
                "calibration_metrics_json_sha256": sha256(metrics_path),
                "row_count": report["row_count"],
                "episode_warnings_early_termination": episode_warnings,
                "eligible_distance_bin_counts": report["eligible_distance_bin_counts"],
                "bias": report["bias"],
                "covariance_scales": report["covariance_scales"],
                "effective_detection": report["effective_detection"],
                "predicted_nis_diagnostic": report["predicted_nis_diagnostic"],
                "miss_likelihood_floor": report["miss_likelihood_floor"],
                "detection_by_range_and_fov": report["detection_by_range_and_fov"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
