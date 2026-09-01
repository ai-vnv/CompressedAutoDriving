"""Task 11: f9c_belief.py metrics, on synthetic data only.

Never touches the simulator, the detector, or seeds 7101-7104.
"""

from __future__ import annotations

import pytest

from duckie_pomdp.evaluation.f9c_belief import (
    augment_belief_metrics_with_calibration,
    belief_metrics_calibrated,
    miss_sequence_metrics,
    outlier_impact,
    robustness_metrics,
    safety_bias,
    support_check,
)


def _row(
    *,
    episode: str = "evaluation_6101_synthetic",
    frame: int,
    eligible_visible: bool = True,
    detector_detected: bool = True,
    gt_range_m: float = 1.0,
    gt_bearing_rad: float = 0.0,
    baseline_range_m: float = 1.0,
    robust_range_m: float = 1.0,
    baseline_initialized: bool = True,
    robust_initialized: bool = True,
    baseline_existence: float = 0.9,
    robust_existence: float = 0.9,
    robust_observability_class: str = "center",
    robust_gate_decision: str | None = "accepted",
    robust_frame_mode: str = "temporal",
    robust_track_deleted: bool = False,
    selected_correct_iou50: bool = True,
    distance_bin: str = "near",
    fov_region: str = "center",
    robust_associated_iou: float | None = 1.0,
    robust_association_differed: bool = False,
    duplicate_selection: bool = False,
    raw_range_m: float | None = 1.0,
) -> dict[str, object]:
    return {
        "episode": episode,
        "frame": frame,
        "eligible_visible": eligible_visible,
        "detector_detected": detector_detected,
        # f9_belief.belief_metrics/_existence_metrics reuse this generic
        # column name; F9c aliases it to the shared detector_detected flag.
        "measurement_detected": detector_detected,
        "gt_range_m": gt_range_m,
        "gt_bearing_rad": gt_bearing_rad,
        "gt_range_rate_mps": 0.0,
        "gt_bearing_rate_rad_s": 0.0,
        "distance_bin": distance_bin,
        "fov_region": fov_region,
        "selected_correct_iou50": selected_correct_iou50,
        "duplicate_selection": duplicate_selection,
        "raw_range_m": raw_range_m,
        "baseline_a_belief_initialized": baseline_initialized,
        "baseline_a_belief_range_m": baseline_range_m,
        "baseline_a_belief_range_std_m": 0.05,
        "baseline_a_belief_bearing_rad": 0.0,
        "baseline_a_belief_bearing_std_m": 0.05,
        "baseline_a_belief_bearing_std_rad": 0.05,
        "baseline_a_belief_range_rate_mps": 0.0,
        "baseline_a_belief_range_rate_std_mps": 0.1,
        "baseline_a_belief_bearing_rate_rad_s": 0.0,
        "baseline_a_belief_bearing_rate_std_rad_s": 0.1,
        "baseline_a_existence_probability": baseline_existence,
        "baseline_a_nis": 1.0,
        "robust_b_belief_initialized": robust_initialized,
        "robust_b_belief_range_m": robust_range_m,
        "robust_b_belief_range_std_m": 0.03,
        "robust_b_belief_bearing_rad": 0.0,
        "robust_b_belief_bearing_std_rad": 0.03,
        "robust_b_belief_range_rate_mps": 0.0,
        "robust_b_belief_range_rate_std_mps": 0.08,
        "robust_b_belief_bearing_rate_rad_s": 0.0,
        "robust_b_belief_bearing_rate_std_rad_s": 0.08,
        "robust_b_existence_probability": robust_existence,
        "robust_b_nis": 1.0,
        "robust_b_observability_class": robust_observability_class,
        "robust_b_gate_decision": robust_gate_decision,
        "robust_b_frame_mode": robust_frame_mode,
        "robust_b_track_deleted": robust_track_deleted,
        "robust_b_associated_iou": robust_associated_iou,
        "robust_b_association_differed_from_highest_confidence": robust_association_differed,
    }


# ---------------------------------------------------------------------------
# Coverage-error / std_over_rmse augmentation.
# ---------------------------------------------------------------------------


def test_coverage_error_is_the_absolute_gap_to_nominal():
    metrics = {
        "range": {"coverage_68": 0.60, "coverage_95": 0.90, "rmse": 0.1, "mean_predicted_std": 0.05},
    }
    augmented = augment_belief_metrics_with_calibration(metrics)
    assert augmented["range"]["coverage_error_68"] == pytest.approx(0.08)
    assert augmented["range"]["coverage_error_95"] == pytest.approx(0.05)
    assert augmented["range"]["std_over_rmse"] == pytest.approx(0.5)


def test_std_over_rmse_is_none_when_rmse_is_zero_or_missing():
    metrics = {"range": {"rmse": 0.0, "mean_predicted_std": 0.05}}
    augmented = augment_belief_metrics_with_calibration(metrics)
    assert augmented["range"]["std_over_rmse"] is None


def test_belief_metrics_calibrated_reuses_f9_belief_and_adds_calibration_fields():
    rows = [_row(frame=index, baseline_range_m=1.0 + 0.01 * index) for index in range(5)]
    result = belief_metrics_calibrated(rows, variant="baseline_a")
    assert result["count"] == 5
    assert "coverage_error_68" in result["range"]
    assert "std_over_rmse" in result["range"]


# ---------------------------------------------------------------------------
# miss_sequence_metrics.
# ---------------------------------------------------------------------------


def test_miss_sequence_metrics_only_counts_genuine_natural_misses():
    rows = [
        _row(frame=0, eligible_visible=True, detector_detected=True),
        _row(frame=1, eligible_visible=True, detector_detected=False),
        _row(frame=2, eligible_visible=True, detector_detected=False),
        _row(frame=3, eligible_visible=True, detector_detected=False),
        _row(frame=4, eligible_visible=True, detector_detected=True),
        # An ineligible (GT-invisible) frame must not extend or count as a
        # miss run even though the detector also failed to fire.
        _row(frame=5, eligible_visible=False, detector_detected=False),
    ]
    result = miss_sequence_metrics(rows, variant="robust_b")
    assert result["label"] == "natural"
    assert result["run_count"] == 1
    assert result["run_length_distribution"]["max"] == 3
    checkpoint_1 = result["run_length_checkpoints"]["1"]
    assert checkpoint_1["eligible_run_count"] == 1
    checkpoint_5 = result["run_length_checkpoints"]["5"]
    assert checkpoint_5["eligible_run_count"] == 0


def test_miss_sequence_metrics_reports_recovery_frames():
    rows = [
        _row(frame=0, eligible_visible=True, detector_detected=True, robust_existence=0.9),
        _row(frame=1, eligible_visible=True, detector_detected=False, robust_existence=0.4),
        _row(frame=2, eligible_visible=True, detector_detected=True, robust_existence=0.3),
        _row(frame=3, eligible_visible=True, detector_detected=True, robust_existence=0.8),
    ]
    result = miss_sequence_metrics(rows, variant="robust_b", active_probability_threshold=0.5)
    checkpoint_1 = result["run_length_checkpoints"]["1"]
    assert checkpoint_1["eligible_run_count"] == 1
    assert checkpoint_1["recovered_run_count"] == 1
    assert checkpoint_1["mean_frames_to_recovery"] == pytest.approx(2.0)


def test_miss_sequence_metrics_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        miss_sequence_metrics([], variant="robust_b", label="bogus")


def test_synthetic_label_never_reads_natural_miss_runs():
    rows = [_row(frame=0, eligible_visible=True, detector_detected=False)]
    result = miss_sequence_metrics(rows, variant="robust_b", label="synthetic")
    assert result["label"] == "synthetic"
    assert result["run_count"] == 0


# ---------------------------------------------------------------------------
# robustness_metrics: the three-way miss breakdown.
# ---------------------------------------------------------------------------


def test_three_way_miss_breakdown_is_disjoint_and_never_pooled():
    rows = [
        # detector_miss_in_domain: no candidate, predicted in-domain.
        _row(
            frame=0,
            detector_detected=False,
            robust_observability_class="center",
            robust_existence=0.05,
        ),
        # detector_miss_outside_domain: no candidate, predicted outside.
        _row(
            frame=1,
            detector_detected=False,
            robust_observability_class="outside_domain",
            robust_existence=0.95,
        ),
        # gated_rejection: candidate present, gate rejected -> a DETECTION.
        _row(
            frame=2,
            detector_detected=True,
            robust_gate_decision="rejected",
            robust_existence=0.85,
        ),
        # a normal accepted detection.
        _row(frame=3, detector_detected=True, robust_gate_decision="accepted"),
    ]
    result = robustness_metrics(rows)
    breakdown = result["miss_breakdown"]
    assert breakdown["detector_miss_in_domain"]["frame_count"] == 1
    assert breakdown["detector_miss_outside_domain"]["frame_count"] == 1
    assert breakdown["gated_rejection"]["frame_count"] == 1
    # Retention must be reported per class, not pooled: the in-domain frame
    # (existence 0.05) is well below active; the outside-domain frame
    # (0.95) is well above. A pooled figure would average them together.
    assert breakdown["detector_miss_in_domain"]["retention_fraction"] == pytest.approx(0.0)
    assert breakdown["detector_miss_outside_domain"]["retention_fraction"] == pytest.approx(1.0)
    assert breakdown["gated_rejection"]["retention_fraction"] == pytest.approx(1.0)
    assert result["gate_accept_reject"] == {"accepted": 1, "rejected": 1}
    assert "in_domain_control_readiness" in result
    assert result["in_domain_control_readiness"]["under_powered"] is True


def test_in_domain_control_readiness_flags_under_powered_support():
    rows = [
        _row(frame=index, detector_detected=False, robust_observability_class="center")
        for index in range(5)
    ]
    result = robustness_metrics(rows, in_domain_support_minimum=20)
    assert result["in_domain_control_readiness"]["frame_count"] == 5
    assert result["in_domain_control_readiness"]["under_powered"] is True


def test_robustness_metrics_tolerates_the_csv_empty_string_convention_for_associated_iou():
    """Regression test: the real evaluate_f9c_robust_belief.py row builder
    stores a missing ``robust_b_associated_iou`` as ``""`` (via its
    ``_optional`` helper, matching CSV-writer conventions), never a bare
    Python ``None``. This crashed the actual final-evaluation run
    (ValueError: could not convert string to float: '') the first time it
    was exercised on real 3328-row data -- no synthetic test had used the
    exact `_optional`-wrapped empty-string convention before this one."""

    rows = [
        _row(frame=0, robust_associated_iou=""),
        _row(frame=1, robust_associated_iou=0.2),
    ]
    result = robustness_metrics(rows)
    assert result["localization_outlier_count"] == 1


def test_wrong_association_events_require_both_a_differ_and_a_bad_iou():
    rows = [
        _row(
            frame=0,
            robust_association_differed=True,
            robust_associated_iou=0.1,
        ),
        _row(
            frame=1,
            robust_association_differed=True,
            robust_associated_iou=0.9,
        ),
        _row(
            frame=2,
            robust_association_differed=False,
            robust_associated_iou=0.1,
        ),
    ]
    result = robustness_metrics(rows)
    assert result["wrong_association_events"] == 1


def test_lifecycle_recoveries_exclude_the_first_ever_initialization():
    rows = [
        _row(frame=0, robust_frame_mode="initialization", selected_correct_iou50=True),
        _row(frame=1, robust_frame_mode="temporal"),
        _row(frame=2, robust_frame_mode="deleted", robust_track_deleted=True),
        _row(frame=3, robust_frame_mode="no_track"),
        _row(frame=4, robust_frame_mode="initialization", selected_correct_iou50=True),
    ]
    result = robustness_metrics(rows)
    assert result["track_deletions"] == 1
    # First initialization at frame 0 is not a recovery; frame 4's is.
    assert result["recoveries"] == 1


def test_false_track_initializations_require_a_wrong_match():
    rows = [
        _row(frame=0, robust_frame_mode="initialization", selected_correct_iou50=False),
    ]
    result = robustness_metrics(rows)
    assert result["false_track_initializations"] == 1


# ---------------------------------------------------------------------------
# outlier_impact / safety_bias / support_check.
# ---------------------------------------------------------------------------


def test_outlier_impact_restricted_to_localization_mismatch_frames():
    rows = [
        _row(
            frame=0,
            gt_range_m=1.0,
            raw_range_m=1.5,
            baseline_range_m=1.4,
            robust_range_m=1.1,
            selected_correct_iou50=False,
        ),
        # Not an outlier -- correctly matched, must be excluded.
        _row(frame=1, gt_range_m=1.0, raw_range_m=1.0, selected_correct_iou50=True),
    ]
    result = outlier_impact(rows)
    assert result["outlier_frame_count"] == 1
    assert result["measurement_range_rmse"]["count"] == 1
    assert result["baseline_a_belief_range_rmse"]["bias"] == pytest.approx(0.4)
    assert result["robust_b_belief_range_rmse"]["bias"] == pytest.approx(0.1)
    assert result["baseline_a_max_transient_belief_range_error_m"] == pytest.approx(0.4)


def test_safety_bias_sign_interpretation_present_and_correct_direction():
    rows = [
        _row(frame=0, gt_range_m=1.0, baseline_range_m=1.2, robust_range_m=0.8),
    ]
    result = safety_bias(rows)
    assert result["baseline_a_mean_signed_range_error_m"] == pytest.approx(0.2)
    assert result["robust_b_mean_signed_range_error_m"] == pytest.approx(-0.2)
    assert "FARTHER" in result["sign_interpretation"]
    assert "CLOSER" in result["sign_interpretation"]


def test_support_check_reports_shortfalls_and_satisfied_flag():
    rows = [
        _row(frame=0, distance_bin="near", fov_region="center"),
        _row(frame=1, distance_bin="near", fov_region="center"),
        _row(frame=2, distance_bin="medium", fov_region="edge_fov"),
    ]
    minimum_support = {"near": 2, "medium": 5, "far": 1, "edge_fov": 1}
    result = support_check(rows, minimum_support)
    assert result["counts"] == {"near": 2, "medium": 1, "far": 0, "edge_fov": 1}
    assert result["satisfied"] is False
    assert result["shortfalls"] == {"medium": 4, "far": 1}


def test_support_check_satisfied_when_every_bin_clears_the_minimum():
    rows = [
        _row(frame=0, distance_bin="near", fov_region="center"),
        _row(frame=1, distance_bin="medium", fov_region="mid_fov"),
        _row(frame=2, distance_bin="far", fov_region="edge_fov"),
    ]
    result = support_check(rows, {"near": 1, "medium": 1, "far": 1, "edge_fov": 1})
    assert result["satisfied"] is True
    assert result["shortfalls"] == {}


def test_support_check_only_counts_eligible_visible_frames():
    rows = [
        _row(frame=0, distance_bin="near", eligible_visible=False),
    ]
    result = support_check(rows, {"near": 1, "medium": 0, "far": 0, "edge_fov": 0})
    assert result["counts"]["near"] == 0
    assert result["satisfied"] is False
