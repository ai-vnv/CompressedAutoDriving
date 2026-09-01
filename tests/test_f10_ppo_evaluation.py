from pathlib import Path

from duckie_pomdp.control.ppo_protocol import (
    classify_curriculum_stage,
    evaluate_retention_change,
    load_ppo_curriculum_protocol,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "f10_ppo_v1.toml"


def test_stage_classification_and_progression_are_fail_closed():
    assert classify_curriculum_stage(safety_pass=True, skill_pass=True, retention_pass=True) == (
        "PASS",
        True,
    )
    assert classify_curriculum_stage(safety_pass=True, skill_pass=False, retention_pass=True) == (
        "LIMITED",
        False,
    )
    assert classify_curriculum_stage(safety_pass=True, skill_pass=True, retention_pass=False) == (
        "LIMITED",
        False,
    )
    assert classify_curriculum_stage(safety_pass=False, skill_pass=True, retention_pass=True) == (
        "FAILED",
        False,
    )


def test_pre_registered_retention_thresholds_are_enforced():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    comparison, passed = evaluate_retention_change(
        protocol,
        "c1",
        {"completion_rate": 1.0},
        {"completion_rate": 0.50},
    )
    assert comparison["drop"] == 0.50
    assert comparison["maximum_allowed"] == 0.25
    assert passed is False

    comparison, passed = evaluate_retention_change(
        protocol,
        "c3",
        {"collision_rate": 0.0},
        {"collision_rate": 0.05},
    )
    assert comparison["increase"] == 0.05
    assert passed is True


def test_retention_compares_matched_trajectory_performance_not_assumed_perfection():
    protocol = load_ppo_curriculum_protocol(CONFIG)
    comparison, passed = evaluate_retention_change(
        protocol,
        "c2",
        {"completion_rate": 0.25},
        {"completion_rate": 0.25},
    )
    assert comparison["baseline"] == 0.25
    assert comparison["current"] == 0.25
    assert comparison["drop"] == 0.0
    assert comparison["maximum_allowed"] == 0.25
    assert passed is True
