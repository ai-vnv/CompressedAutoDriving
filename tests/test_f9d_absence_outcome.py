import pytest
from pathlib import Path
import sys

from duckie_pomdp.evaluation.f9d_absence_outcome import (
    absence_outcome_metrics,
    analytic_existence_probability,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_f9d_absence_stress import _read_absence_csv, _refuse_overwrite  # noqa: E402


def _row(
    frame,
    kind,
    probability,
    *,
    visible,
    exists=True,
    detected=False,
    obs="outside_domain",
    dropout=None,
):
    return {
        "episode": f"{kind}-episode",
        "seed": 8301,
        "scenario": f"{kind}-scenario",
        "frame": frame,
        "absence_kind": kind,
        "gt_exists": exists,
        "eligible_visible": visible,
        "dropout_frame": (kind == "B2" and frame > 0) if dropout is None else dropout,
        "robust_b_existence_probability": probability,
        "robust_b_observability_class": obs,
        "robust_b_track_active": probability >= 0.5,
        "robust_b_track_deleted": False,
        "detector_detected": detected,
    }


def test_prediction_only_recurrence_has_the_documented_halfway_fixed_point():
    value = analytic_existence_probability(
        0.99, 40, survival_probability=0.995, birth_probability=0.005
    )
    assert value == pytest.approx(0.8277961617, abs=1e-10)


def test_b1_checkpoints_compare_observed_with_closed_form_prediction():
    rows = [_row(0, "B1", 0.99, visible=True)]
    probability = 0.99
    for frame in range(1, 42):
        probability = 0.995 * probability + 0.005 * (1.0 - probability)
        rows.append(_row(frame, "B1", probability, visible=False))
    metrics = absence_outcome_metrics(
        rows,
        survival_probability=0.995,
        birth_probability=0.005,
        out_of_domain_floor=0.5,
        in_domain_ceiling=0.1,
        recurrence_tolerance=1e-9,
        recovery_frames_max=2,
    )
    assert metrics["B1"]["criterion"]["all_checkpoint_40_above_floor"] is True
    assert metrics["B1"]["criterion"]["recurrence_matches"] is True


def test_b1_recurrence_excludes_a_positive_detection_at_the_gt_visibility_boundary():
    rows = [_row(0, "B1", 0.99, visible=True, detected=True)]
    rows.append(_row(1, "B1", 0.999, visible=False, detected=True))
    probability = 0.999
    for frame in range(2, 43):
        probability = 0.995 * probability + 0.005 * (1.0 - probability)
        rows.append(_row(frame, "B1", probability, visible=False, detected=False))
    metrics = absence_outcome_metrics(
        rows,
        survival_probability=0.995,
        birth_probability=0.005,
        out_of_domain_floor=0.5,
        in_domain_ceiling=0.1,
        recurrence_tolerance=1e-9,
        recovery_frames_max=2,
    )
    assert metrics["B1"]["criterion"]["recurrence_matches"] is True
    assert metrics["B1"]["prediction_only_events"][0]["start_frame"] == 2


def test_b2_is_judged_at_twenty_suppressed_frames_and_recovers():
    rows = [_row(0, "B2", 0.99, visible=True, detected=True, obs="center")]
    for frame in range(1, 22):
        rows.append(_row(frame, "B2", 0.05, visible=True, obs="center"))
    rows.append(
        _row(
            22,
            "B2",
            0.9,
            visible=True,
            detected=True,
            obs="center",
            dropout=False,
        )
    )
    metrics = absence_outcome_metrics(
        rows,
        survival_probability=0.995,
        birth_probability=0.005,
        out_of_domain_floor=0.5,
        in_domain_ceiling=0.1,
        recurrence_tolerance=1e-9,
        recovery_frames_max=2,
    )
    assert metrics["B2"]["criterion"]["all_below_ceiling"] is True
    assert metrics["B2"]["recovery"]["max_frames"] == 1
    assert metrics["B2"]["criterion"]["all_observed_recoveries_within_limit"] is True


def test_b3_remains_separate_from_dropout_and_natural_absence():
    rows = [_row(0, "B3", 0.9, visible=True, exists=True, detected=True)]
    rows.extend(
        _row(frame, "B3", 0.4, visible=False, exists=False)
        for frame in range(1, 25)
    )
    metrics = absence_outcome_metrics(
        rows,
        survival_probability=0.995,
        birth_probability=0.005,
        out_of_domain_floor=0.5,
        in_domain_ceiling=0.1,
        recurrence_tolerance=1e-9,
        recovery_frames_max=2,
    )
    assert metrics["B3"]["event_count"] == 1
    assert metrics["B1"]["event_count"] == 0
    assert metrics["B2"]["event_count"] == 0


def test_absence_final_run_refuses_to_overwrite_existing_evidence(tmp_path):
    existing = tmp_path / "f9d_absence_stress.csv"
    existing.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to re-render"):
        _refuse_overwrite([existing, tmp_path / "metrics.json"])


def test_absence_csv_recompute_loader_restores_boolean_types(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "episode,seed,scenario,frame,absence_kind,eligible_visible,detector_detected,"
        "robust_b_track_active,robust_b_track_deleted,gt_exists,dropout_frame,"
        "robust_b_existence_probability,robust_b_observability_class\n"
        "e,8301,s,1,B2,True,False,False,True,True,True,0.05,center\n",
        encoding="utf-8",
    )
    row = _read_absence_csv(path)[0]
    assert row["eligible_visible"] is True
    assert row["detector_detected"] is False
    assert row["dropout_frame"] is True
    assert row["frame"] == 1
    assert row["robust_b_existence_probability"] == 0.05
