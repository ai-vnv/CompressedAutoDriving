"""Task 11 fix round 1: the shared-builder equivalence guard.

Fix-round ruling: reconstruct the missing Task 11 artifacts by REPLAYING the
already-written runtime cache (no re-render), with the render path and the
replay path funnelled through ONE shared row-building function. This test is
the guard the ruling asked for: it proves ``_selection_from_render_result``
(render path) and ``_selection_from_cache_frame`` (replay path) produce
equivalent output for the same underlying candidates, and that
``build_row`` -- fed either one -- produces an identical row. All synthetic
data; no simulator, no detector, no GPU, no final-evaluation seeds.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import evaluate_f9c_robust_belief as evaluate_module  # noqa: E402

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.innovation_gate import InnovationGateConfig
from duckie_pomdp.belief.measurement_association import AssociationConfig
from duckie_pomdp.belief.observability import EffectiveDetectionModel, ObservabilityClass
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    load_pedestrian_ekf_config,
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
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, Detection, ObjectClass
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.evaluation.f9c_runtime_cache import RuntimeCacheFrame, TruthFrame
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector, CameraCalibration
from duckie_pomdp.perception.f9_pipeline import CandidateProjection, F9ImageObservation
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise

F9C_CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"


def _detection(confidence: float, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(
        object_class=ObjectClass.DUCKIE,
        confidence=confidence,
        bounding_box=BoundingBox(*bbox),
    )


def _measurement(range_m: float, bearing_rad: float, confidence: float) -> "ObjectMeasurement":
    from math import cos, sin

    from duckie_pomdp.domain.measurement import ObjectMeasurement

    return ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=confidence,
        x_left_m=range_m * sin(bearing_rad),
        y_forward_m=range_m * cos(bearing_rad),
        range_m=range_m,
        bearing_rad=bearing_rad,
    )


def _synthetic_render_result() -> F9ImageObservation:
    """Two raw candidates: a lower-confidence VALID one and a
    higher-confidence one whose projection FAILED -- deliberately exercises
    the "highest-confidence selection has no usable measurement" branch."""

    from duckie_pomdp.domain.measurement import ObjectMeasurement

    high_confidence_failed = _detection(0.91, (300.0, 0.0, 340.0, 80.0))
    low_confidence_valid = _detection(0.62, (120.0, 200.0, 180.0, 300.0))
    candidates = (
        CandidateProjection(
            detection=high_confidence_failed,
            measurement=ObjectMeasurement.missing(ObjectClass.DUCKIE),
            projection_error="near-horizon projection failure",
        ),
        CandidateProjection(
            detection=low_confidence_valid,
            measurement=_measurement(0.734, -0.12, 0.62),
            projection_error=None,
        ),
    )
    return F9ImageObservation(
        pedestrian=ObjectMeasurement.missing(ObjectClass.DUCKIE),
        duckie_detection_count=2,
        selected_duckie=high_confidence_failed,
        duplicate_selection=True,
        projection_error="near-horizon projection failure",
        stop_sign_detections=(),
        duckie_candidates=candidates,
    )


def _cache_frame_from_render_result(result: F9ImageObservation) -> RuntimeCacheFrame:
    """Mirrors exactly how collect_final_rows builds a RuntimeCacheFrame
    from a live F9ImageObservation (invariant I5: raw, pre-bias)."""

    return RuntimeCacheFrame(
        episode="synthetic_episode",
        seed=9999,
        scenario="synthetic_scenario",
        frame=3,
        dt_s=1.0 / 30.0,
        raw_candidate_range_m=tuple(
            float("nan") if c.projection_error is not None else float(c.measurement.range_m)
            for c in result.duckie_candidates
        ),
        raw_candidate_bearing_rad=tuple(
            float("nan") if c.projection_error is not None else float(c.measurement.bearing_rad)
            for c in result.duckie_candidates
        ),
        raw_candidate_confidence=tuple(float(c.detection.confidence) for c in result.duckie_candidates),
        raw_candidate_bbox=tuple(
            (
                c.detection.bounding_box.x_min_px,
                c.detection.bounding_box.y_min_px,
                c.detection.bounding_box.x_max_px,
                c.detection.bounding_box.y_max_px,
            )
            for c in result.duckie_candidates
        ),
        raw_candidate_projection_failed=tuple(
            c.projection_error is not None for c in result.duckie_candidates
        ),
        ego_linear_velocity_mps=0.2,
        ego_yaw_rate_rad_s=-0.05,
    )


def test_selection_from_cache_frame_matches_selection_from_render_result():
    result = _synthetic_render_result()
    cache_frame = _cache_frame_from_render_result(result)

    from_render = evaluate_module._selection_from_render_result(result)
    from_cache = evaluate_module._selection_from_cache_frame(cache_frame)

    assert from_render.duckie_detection_count == from_cache.duckie_detection_count == 2
    assert from_render.duplicate_selection == from_cache.duplicate_selection is True
    assert from_render.raw_pedestrian_measurement == from_cache.raw_pedestrian_measurement
    assert from_render.selected_confidence == from_cache.selected_confidence
    assert from_render.selected_bbox == from_cache.selected_bbox
    assert from_render.robust_candidates == from_cache.robust_candidates
    assert from_render.bbox_by_key == from_cache.bbox_by_key
    # The highest-confidence candidate's projection FAILED -- the shared
    # selection logic must report a missing raw measurement, not silently
    # fall back to the lower-confidence valid one.
    assert from_render.raw_pedestrian_measurement.detected is False
    # But the valid lower-confidence candidate must still be available for
    # Robust B's association to consider.
    assert len(from_render.robust_candidates) == 1


def _ekf_config():
    return load_pedestrian_ekf_config(F9C_CONFIG)


def _noise():
    return load_polar_measurement_noise(F9C_CONFIG)


def _existence_config():
    return load_existence_filter_config(F9C_CONFIG)


def _baseline_bias():
    return evaluate_module.AdditiveMeasurementBias(-0.045904804710162034, 0.00414567890700929)


def _robust_bias():
    return FrozenBiasCorrection(
        model="global_additive",
        range_bias_m=-0.045904804710162034,
        bearing_bias_rad=0.00414567890700929,
        range_bin_bias_m=None,
        near_max_m=0.55,
        medium_max_m=0.80,
    )


def _observability_model():
    projector = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )
    return evaluate_module.PredictedObservabilityModel(projector, image_width_px=640)


def _robust_observation_config() -> RobustObservationConfig:
    return RobustObservationConfig(
        switches=RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=True,
            temporal_association=True,
            covariance_calibration=True,
            conditional_detection=True,
        ),
        gate=InnovationGateConfig(9.21034037197618),
        association=AssociationConfig(
            chi_square_gate=13.815510557964274,
            initialization_rule="highest_confidence_then_bbox_lexicographic",
        ),
        covariance=CovarianceCalibration(1.0, 1.0, 0.0, 0.0),
        effective_detection=EffectiveDetectionModel(
            {
                ObservabilityClass.CENTER: 0.99,
                ObservabilityClass.MID_FOV: 0.95,
                ObservabilityClass.EDGE_FOV: 0.60,
                ObservabilityClass.OUTSIDE_DOMAIN: 0.05,
            },
            outside_domain_miss_policy="prediction_only",
        ),
        active_threshold=0.50,
        delete_threshold=0.05,
        initialization_threshold=0.50,
    )


def _new_updaters():
    baseline = PedestrianBeliefUpdater(
        ekf_config=_ekf_config(),
        existence_config=_existence_config(),
        measurement_noise=_noise(),
        measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
    )
    robust = RobustPedestrianBeliefUpdater(
        ekf_config=_ekf_config(),
        measurement_noise=_noise(),
        existence_config=_existence_config(),
        bias_frozen=_robust_bias(),
        bias_fitted=_robust_bias(),
        observability_model=_observability_model(),
        config=_robust_observation_config(),
    )
    return baseline, robust


def test_build_row_is_identical_whether_fed_the_render_or_cache_selection():
    """The end-to-end guard: same candidates, same updater states, same
    truth -> byte-identical row from either _FrameSelection source."""

    result = _synthetic_render_result()
    cache_frame = _cache_frame_from_render_result(result)
    ego_motion = EgoMotion(0.2, -0.05)
    ego = EgoObservation(0.0, 0.0, 0.2, -0.05)
    previous_action = PolicyAction(0.1, 0.0)
    truth = TruthFrame(
        episode="synthetic_episode",
        frame=3,
        gt_exists=True,
        gt_range_m=0.73,
        gt_bearing_rad=-0.11,
        gt_range_rate_mps=0.0,
        gt_bearing_rate_rad_s=0.0,
        eligible_visible=True,
        visible_pixel_count=812,
        gt_bbox=(118.0, 198.0, 182.0, 302.0),
        distance_bin="medium",
        fov_region="center",
    )

    rows = {}
    for label, selection in (
        ("render", evaluate_module._selection_from_render_result(result)),
        ("cache", evaluate_module._selection_from_cache_frame(cache_frame)),
    ):
        baseline_updater, robust_updater = _new_updaters()
        baseline_belief = initial_belief(ego, existence_prior=_existence_config().prior_probability)
        robust_belief = initial_belief(ego, existence_prior=_existence_config().prior_probability)

        baseline_belief, corrected_measurement, robust_belief, record = evaluate_module._step_both_systems(
            selection=selection,
            baseline_updater=baseline_updater,
            baseline_belief=baseline_belief,
            baseline_bias=_baseline_bias(),
            robust_updater=robust_updater,
            robust_belief=robust_belief,
            previous_action=previous_action,
            ego=ego,
            road=None,
            ego_motion=ego_motion,
            dt_s=1.0 / 30.0,
        )
        row = evaluate_module.build_row(
            episode="synthetic_episode",
            seed=9999,
            scenario="synthetic_scenario",
            pedestrian_mode="stationary",
            frame=3,
            timestamp_s=0.1,
            dt_s=1.0 / 30.0,
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
            matching_iou_threshold=0.5,
            minimum_range_m=_ekf_config().minimum_range_m,
        )
        rows[label] = row

    assert rows["render"].keys() == rows["cache"].keys()
    for key in rows["render"]:
        left, right = rows["render"][key], rows["cache"][key]
        if isinstance(left, float) and isinstance(right, float):
            assert left == pytest.approx(right, abs=1e-12), key
        else:
            assert left == right, key


def _known_limitations_row(
    *,
    detector_detected: bool,
    robust_b_observability_class: str,
    robust_b_existence_probability: float,
    robust_b_frame_mode: str = "temporal",
    robust_b_belief_initialized: bool = True,
    robust_b_belief_range_m: float = 1.0,
    robust_b_belief_range_std_m: float = 0.1,
    gt_range_m: float = 1.0,
) -> dict[str, object]:
    return {
        "detector_detected": detector_detected,
        "robust_b_observability_class": robust_b_observability_class,
        "robust_b_existence_probability": robust_b_existence_probability,
        "robust_b_frame_mode": robust_b_frame_mode,
        "robust_b_belief_initialized": robust_b_belief_initialized,
        "robust_b_belief_range_m": robust_b_belief_range_m,
        "robust_b_belief_range_std_m": robust_b_belief_range_std_m,
        "gt_range_m": gt_range_m,
    }


def test_known_limitations_is_purely_descriptive_and_computed_from_rows():
    from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol

    protocol = load_f9c_protocol(F9C_CONFIG, require_frozen=True)
    rows = [
        _known_limitations_row(
            detector_detected=True,
            robust_b_observability_class="center",
            robust_b_existence_probability=0.95,
            robust_b_belief_range_m=1.2,
            gt_range_m=1.0,
            robust_b_belief_range_std_m=0.2,
        ),
        _known_limitations_row(
            detector_detected=True,
            robust_b_observability_class="mid_fov",
            robust_b_existence_probability=0.93,
            robust_b_belief_range_m=0.8,
            gt_range_m=1.0,
            robust_b_belief_range_std_m=0.2,
        ),
        _known_limitations_row(
            detector_detected=False,
            robust_b_observability_class="outside_domain",
            robust_b_existence_probability=0.99,
            robust_b_frame_mode="initialization",
            robust_b_belief_initialized=False,
        ),
    ]
    result = evaluate_module.known_limitations(rows, protocol)

    conditional = result["conditional_detection_per_class_probabilities_are_inert"]
    detected_branch = conditional["detected_branch_saturates_regardless_of_class"]
    assert detected_branch["detected_frame_count"] == 2
    assert detected_branch["minimum_existence_probability_among_detected_rows"] == pytest.approx(0.93)
    assert detected_branch["existence_by_class"]["center"]["count"] == 1
    assert detected_branch["existence_by_class"]["mid_fov"]["mean"] == pytest.approx(0.93)

    miss_branch = conditional["miss_branch_dominated_by_the_i8_floor"]
    assert miss_branch["floor_dominates_every_in_domain_class"] is True
    assert set(miss_branch["implied_lr_by_class"]) == {"center", "mid_fov", "edge_fov", "outside_domain"}

    init = conditional["initialization_frames_mostly_classify_outside_domain"]
    assert init["initialization_frame_count"] == 1
    assert init["outside_domain_count"] == 1

    z_block = result["coverage_overshoot_is_not_a_tail_effect"]["z_score_distribution"]
    assert z_block["n"] == 2
    # z = (1.2-1.0)/0.2 = 1.0, (0.8-1.0)/0.2 = -1.0 -> mean 0, std 1
    assert z_block["mean"] == pytest.approx(0.0, abs=1e-12)
    assert z_block["std"] == pytest.approx(1.0, abs=1e-12)

    # Purely descriptive: no key here is named like a metric this function
    # could be mistaken for redefining.
    assert "metrics" not in result


def test_replay_from_cache_refuses_a_hash_that_does_not_match(tmp_path: Path):
    from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
    from duckie_pomdp.evaluation.f9c_runtime_cache import write_evaluation_truth, write_runtime_cache

    protocol = load_f9c_protocol(F9C_CONFIG, require_frozen=True)
    cache_path = tmp_path / "cache.npz"
    truth_path = tmp_path / "truth.npz"
    frame = RuntimeCacheFrame(
        episode="e",
        seed=1,
        scenario="s",
        frame=0,
        dt_s=1.0 / 30.0,
        raw_candidate_range_m=(),
        raw_candidate_bearing_rad=(),
        raw_candidate_confidence=(),
        raw_candidate_bbox=(),
        raw_candidate_projection_failed=(),
        ego_linear_velocity_mps=0.0,
        ego_yaw_rate_rad_s=0.0,
    )
    write_runtime_cache(cache_path, (frame,))
    write_evaluation_truth(
        truth_path,
        (
            TruthFrame(
                episode="e",
                frame=0,
                gt_exists=True,
                gt_range_m=1.0,
                gt_bearing_rad=0.0,
                gt_range_rate_mps=0.0,
                gt_bearing_rate_rad_s=0.0,
                eligible_visible=True,
                visible_pixel_count=10,
                gt_bbox=(0.0, 0.0, 1.0, 1.0),
                distance_bin="far",
                fov_region="center",
            ),
        ),
    )
    fake_protocol = replace(
        protocol,
        artifacts={**protocol.artifacts, "runtime_cache": cache_path, "evaluation_truth": truth_path},
    )
    with pytest.raises(ValueError):
        evaluate_module.replay_from_cache(
            fake_protocol,
            expected_runtime_cache_sha256="0" * 64,
        )
