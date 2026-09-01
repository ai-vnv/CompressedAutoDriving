from pathlib import Path

from duckie_pomdp.evaluation.lane_policy import (
    LaneCheckpointScore,
    select_lane_checkpoint,
)


def score(
    step: int,
    *,
    success: float,
    invalid: float,
    yellow: float,
    departure: float,
    lateral: float,
    reward: float,
) -> LaneCheckpointScore:
    return LaneCheckpointScore(
        path=Path(f"checkpoint-{step}.pt"),
        global_step=step,
        sha256=str(step),
        summary={
            "lap_success_rate": success,
            "invalid_pose_rate": invalid,
            "yellow_crossing_rate": yellow,
            "lane_departure_rate": departure,
            "mean_abs_lateral_error_m": lateral,
            "mean_return": reward,
        },
    )


def test_lane_checkpoint_selection_applies_safety_before_return() -> None:
    unsafe_high_return = score(
        10_000,
        success=1.0,
        invalid=0.5,
        yellow=0.0,
        departure=0.0,
        lateral=0.02,
        reward=100.0,
    )
    safe = score(
        20_000,
        success=0.75,
        invalid=0.0,
        yellow=0.0,
        departure=0.0,
        lateral=0.04,
        reward=20.0,
    )
    selection = select_lane_checkpoint(
        (unsafe_high_return, safe),
        maximum_invalid_pose_rate=0.25,
        maximum_yellow_crossing_rate=0.25,
        maximum_lane_departure_rate=0.25,
    )
    assert selection.selected is safe
    assert selection.best_return is unsafe_high_return
    assert selection.safety_filter_passed


def test_lane_checkpoint_selection_tiebreaks_with_lane_error() -> None:
    worse_lane = score(
        10_000,
        success=1.0,
        invalid=0.0,
        yellow=0.0,
        departure=0.0,
        lateral=0.08,
        reward=30.0,
    )
    better_lane = score(
        20_000,
        success=1.0,
        invalid=0.0,
        yellow=0.0,
        departure=0.0,
        lateral=0.04,
        reward=20.0,
    )
    selection = select_lane_checkpoint(
        (worse_lane, better_lane),
        maximum_invalid_pose_rate=0.25,
        maximum_yellow_crossing_rate=0.25,
        maximum_lane_departure_rate=0.25,
    )
    assert selection.selected is better_lane
