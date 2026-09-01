import pytest
from pathlib import Path
import sys

from duckie_pomdp.evaluation.f9d_outlier_outcome import (
    gate_confusion,
    outlier_outcome_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_f9d_outlier_stress import _refuse_overwrite  # noqa: E402


def _row(frame, iou, baseline_error, robust_error, *, decision="accepted"):
    return {
        "episode": "e",
        "seed": 8201,
        "scenario": "s",
        "frame": frame,
        "eligible_visible": True,
        "selected_bbox_x1": 10,
        "selected_iou": iou,
        "gt_range_m": 1.0,
        "distance_bin": "far",
        "fov_region": "edge_fov",
        "baseline_a_belief_initialized": True,
        "baseline_a_belief_range_m": 1.0 + baseline_error,
        "robust_b_belief_initialized": True,
        "robust_b_belief_range_m": 1.0 + robust_error,
        "robust_b_associated_iou": iou,
        "robust_b_gate_decision": decision,
    }


def test_outlier_outcome_uses_the_locked_iou_frame_definition():
    rows = [
        _row(0, 0.4, 0.3, 0.1, decision="rejected"),
        _row(1, 0.8, 0.01, 0.01),
    ]
    metrics = outlier_outcome_metrics(rows, recovery_error_m=0.1)
    assert metrics["outlier_frame_count"] == 1
    assert metrics["baseline_a_range_error"]["rmse"] == pytest.approx(0.3)
    assert metrics["robust_b_range_error"]["rmse"] == pytest.approx(0.1)
    assert metrics["robust_to_baseline_rmse_ratio"] == pytest.approx(1.0 / 3.0)


def test_recovery_is_measured_after_each_contiguous_event():
    rows = [
        _row(0, 0.3, 0.3, 0.2),
        _row(1, 0.2, 0.2, 0.15),
        _row(2, 0.9, 0.15, 0.08),
        _row(3, 0.9, 0.08, 0.02),
    ]
    metrics = outlier_outcome_metrics(rows, recovery_error_m=0.1)
    assert metrics["robust_b_recovery"]["events"][0]["recovery_frames"] == 1
    assert metrics["baseline_a_recovery"]["events"][0]["recovery_frames"] == 2


def test_gate_confusion_matches_f9c_table_semantics():
    rows = [
        _row(0, 0.8, 0, 0, decision="accepted"),
        _row(1, 0.8, 0, 0, decision="rejected"),
        _row(2, 0.2, 0, 0, decision="accepted"),
        _row(3, 0.2, 0, 0, decision="rejected"),
    ]
    result = gate_confusion(rows)
    assert result["good_accept"] == result["good_reject"] == 1
    assert result["outlier_accept"] == result["outlier_reject"] == 1
    assert result["outlier_rejection_sensitivity"] == 0.5
    assert result["good_measurement_false_rejection_rate"] == 0.5


def test_final_run_refuses_to_overwrite_any_existing_evidence(tmp_path):
    existing = tmp_path / "cache.npz"
    existing.write_bytes(b"partial-final-render")
    with pytest.raises(RuntimeError, match="refusing to re-render"):
        _refuse_overwrite([tmp_path / "metrics.json", existing])
