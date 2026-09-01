from pathlib import Path

import pytest

from duckie_pomdp.evaluation.f9d_protocol import load_f9d_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f9d_evidence_closure_v1.toml"

F9C_SHA = "359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e"


def test_f9d_pins_the_frozen_f9c_config_by_hash():
    protocol = load_f9d_protocol(CONFIG)
    assert protocol.f9c_config_sha256 == F9C_SHA


def test_f9d_rejects_a_drifted_f9c_config(tmp_path, monkeypatch):
    """If the F9c config ever changes, F9d must refuse to run rather than
    silently evaluate a different estimator than the one it claims to test."""
    text = CONFIG.read_text(encoding="utf-8").replace(F9C_SHA, "0" * 64)
    broken = ROOT / "configs" / "_tmp_f9d_drift_probe.toml"
    broken.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="frozen F9c"):
            load_f9d_protocol(broken)
    finally:
        broken.unlink(missing_ok=True)


def test_f9d_seeds_are_disjoint_from_every_earlier_band():
    protocol = load_f9d_protocol(CONFIG)
    development = set(protocol.development_seeds)
    outlier = set(protocol.outlier_final_seeds)
    absence = set(protocol.absence_final_seeds)
    assert development and outlier and absence
    assert not (development & outlier)
    assert not (development & absence)
    assert not (outlier & absence)
    forbidden = set(protocol.forbidden_seeds)
    assert {5101, 5102, 5103, 5104} <= forbidden
    assert {7101, 7102, 7103, 7104} <= forbidden
    assert {6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108} <= forbidden
    assert not (development | outlier | absence) & forbidden


def test_f9d_exposes_f9c_parameters_but_defines_none_of_its_own():
    """F9d may read every estimator parameter and write none. A parameter
    defined in the F9d config would be an estimator change wearing a
    stress-test costume."""
    import tomli

    with CONFIG.open("rb") as stream:
        data = tomli.load(stream)
    forbidden_sections = {
        "measurement_model",
        "covariance_calibration",
        "conditional_detection",
        "innovation_gate",
        "association",
        "ekf",
        "existence",
    }
    assert not (forbidden_sections & set(data)), (
        "F9d must not redefine any estimator section; it imports F9c's"
    )


def test_f9d_minima_are_pre_registered():
    protocol = load_f9d_protocol(CONFIG)
    assert protocol.minimum_outlier_frames >= 50
    assert protocol.minimum_outlier_events >= 12
    assert protocol.minimum_outlier_seeds >= 3
    assert protocol.insufficient_outlier_frames == 30
    assert protocol.minimum_absence_runs_20 >= 12
    assert protocol.minimum_absence_runs_40 >= 4


def test_outlier_support_requires_all_three_conditions():
    """Frames alone can be satisfied by two long bursts. The support check must
    fail when events or seed spread fall short, even with frames well clear."""
    from duckie_pomdp.evaluation.f9d_protocol import outlier_support_satisfied

    protocol = load_f9d_protocol(CONFIG)
    assert outlier_support_satisfied(protocol, frames=60, events=15, seeds=4)
    assert not outlier_support_satisfied(protocol, frames=60, events=2, seeds=2)
    assert not outlier_support_satisfied(protocol, frames=60, events=15, seeds=2)
    assert not outlier_support_satisfied(protocol, frames=40, events=15, seeds=4)


def test_absence_support_requires_both_conditions():
    """Task 4 fix round 1: the pre-registered minimum_absence_runs_20/_40
    check must live in exactly one place, not be re-implemented per caller
    -- the same reasoning that keeps outlier_support_satisfied as a single
    function."""
    from duckie_pomdp.evaluation.f9d_protocol import absence_support_satisfied

    protocol = load_f9d_protocol(CONFIG)
    assert absence_support_satisfied(protocol, runs_ge_20=12, runs_ge_40=4)
    assert absence_support_satisfied(protocol, runs_ge_20=42.0, runs_ge_40=28.0)
    assert not absence_support_satisfied(protocol, runs_ge_20=11, runs_ge_40=4)
    assert not absence_support_satisfied(protocol, runs_ge_20=12, runs_ge_40=3)
    assert not absence_support_satisfied(protocol, runs_ge_20=0, runs_ge_40=0)


def test_absence_support_is_checked_per_kind_not_only_combined():
    """Regression pin for the fix-round-1 correction: a caller must be able
    to check each absence kind (B1/B2/B3) independently against the same
    minima, not only a pre-pooled combined count -- the function itself is
    kind-agnostic (it just compares numbers), so this test pins that
    calling it three times with three different per-kind counts is the
    correct usage, using the actual fix-round-1 per-kind numbers."""
    from duckie_pomdp.evaluation.f9d_protocol import absence_support_satisfied

    protocol = load_f9d_protocol(CONFIG)
    per_kind = {
        "B1": {"runs_ge_20": 18.0, "runs_ge_40": 16.0},
        "B2": {"runs_ge_20": 24.0, "runs_ge_40": 12.0},
        "B3": {"runs_ge_20": 16.0, "runs_ge_40": 10.0},
    }
    for kind, counts in per_kind.items():
        assert absence_support_satisfied(protocol, **counts), kind
