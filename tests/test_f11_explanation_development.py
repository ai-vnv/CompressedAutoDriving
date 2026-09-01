from __future__ import annotations

from pathlib import Path

import numpy as np

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.development_protocol import (
    PhaseThresholds,
    apply_semantic_intervention,
    build_r002_baselines,
    public_phase,
    schema_valid_public_vector,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return load_ppo_curriculum_protocol(
        ROOT / "configs" / "f10_ppo_visual_objects_v30.toml"
    )


def neutral_physical() -> np.ndarray:
    p = protocol()
    neutral = p.raw["neutral"]
    values = {name: 0.0 for name in p.observation_order}
    values.update(
        lane_validity_probability=1.0,
        lane_lateral_error_std_m=0.05,
        lane_heading_error_std_rad=0.10,
        lane_curvature_std_inv_m=0.5,
        actual_linear_velocity_mps=0.2,
        stop_line_distance_m=float(neutral["stop_line_distance_m"]),
        stop_mode_none=1.0,
    )
    values.update({name: float(value) for name, value in neutral.items()})
    return np.asarray([values[name] for name in p.observation_order], dtype=np.float32)


def intervention(values: np.ndarray, name: str):
    return apply_semantic_intervention(
        values,
        name,
        protocol(),
        lane_low_confidence_validity=0.25,
        lane_low_confidence_min_lateral_std_m=0.20,
        lane_low_confidence_min_heading_std_rad=0.50,
        lane_low_confidence_min_curvature_std_inv_m=2.50,
    )


def test_public_phase_uses_only_registered_public_fields() -> None:
    p = protocol()
    values = neutral_physical()
    thresholds = PhaseThresholds(0.4, 1.2, 1.5, 0.5)
    assert public_phase(values, p.observation_order, thresholds) == "nominal"
    values[p.observation_order.index("lane_curvature_mean_inv_m")] = -2.0
    assert public_phase(values, p.observation_order, thresholds) == "lane_curve"
    values[p.observation_order.index("pedestrian_existence_probability")] = 0.8
    values[p.observation_order.index("pedestrian_range_mean_m")] = 0.7
    assert public_phase(values, p.observation_order, thresholds) == "pedestrian_relevant"
    values[p.observation_order.index("stop_mode_none")] = 0.0
    values[p.observation_order.index("stop_mode_required")] = 1.0
    assert public_phase(values, p.observation_order, thresholds) == "combined_pedestrian_stop"


def test_satisfied_stop_phase_is_local_not_persistent_forever() -> None:
    p = protocol()
    values = neutral_physical()
    thresholds = PhaseThresholds(0.4, 1.2, 1.5, 0.5)
    values[p.observation_order.index("stop_mode_none")] = 0.0
    values[p.observation_order.index("stop_mode_satisfied")] = 1.0
    values[p.observation_order.index("stop_line_distance_m")] = -0.2
    assert public_phase(values, p.observation_order, thresholds) == "stop_satisfied"
    values[p.observation_order.index("stop_line_distance_m")] = -2.0
    assert public_phase(values, p.observation_order, thresholds) == "nominal"


def test_r002_baselines_are_finite_and_schema_valid() -> None:
    p = protocol()
    physical = np.stack([neutral_physical(), neutral_physical()])
    observations = np.asarray(
        np.clip(
            physical / np.asarray(p.observation_scales, dtype=np.float32),
            -p.observation_clip,
            p.observation_clip,
        ),
        dtype=np.float32,
    )
    baselines = build_r002_baselines(
        observations, physical, np.asarray([1, 2], dtype=np.int64), p
    )
    assert tuple(baselines) == (
        "episode_reset",
        "public_median",
        "semantic_neutral_hazard",
    )
    for baseline in baselines.values():
        assert baseline.shape == (2, 29)
        assert np.isfinite(baseline).all()


def test_pedestrian_absent_uses_full_frozen_neutral_tuple() -> None:
    p = protocol()
    values = neutral_physical()
    for name, value in (
        ("pedestrian_existence_probability", 0.9),
        ("pedestrian_range_mean_m", 0.6),
        ("pedestrian_range_std_m", 0.1),
        ("pedestrian_bearing_mean_rad", 0.2),
    ):
        values[p.observation_order.index(name)] = value
    changed, intended = intervention(values, "pedestrian_absent")
    assert len(intended) == 9
    expected = neutral_physical() / np.asarray(p.observation_scales, dtype=np.float32)
    indexes = [p.observation_order.index(name) for name in intended]
    np.testing.assert_allclose(changed[indexes], expected[indexes])


def test_stop_absent_sets_stopline_and_valid_none_mode() -> None:
    p = protocol()
    values = neutral_physical()
    values[p.observation_order.index("stop_line_distance_m")] = 0.25
    values[p.observation_order.index("stop_sign_existence_probability")] = 0.9
    values[p.observation_order.index("stop_mode_none")] = 0.0
    values[p.observation_order.index("stop_mode_required")] = 1.0
    changed, intended = intervention(values, "stop_absent")
    scales = np.asarray(p.observation_scales, dtype=np.float32)
    physical = changed * scales
    schema_valid_public_vector(physical, p.observation_order)
    assert physical[p.observation_order.index("stop_line_distance_m")] == 2.0
    assert physical[p.observation_order.index("stop_mode_none")] == 1.0
    assert "stop_line_distance_m" in intended


def test_lane_centered_preserves_lane_uncertainty_and_curvature() -> None:
    p = protocol()
    values = neutral_physical()
    values[p.observation_order.index("lane_lateral_error_mean_m")] = 0.1
    values[p.observation_order.index("lane_heading_error_mean_rad")] = -0.2
    changed, intended = intervention(values, "lane_centered")
    physical = changed * np.asarray(p.observation_scales, dtype=np.float32)
    assert intended == (
        "lane_lateral_error_mean_m",
        "lane_heading_error_mean_rad",
    )
    assert physical[p.observation_order.index(intended[0])] == 0.0
    assert physical[p.observation_order.index(intended[1])] == 0.0
    assert np.isclose(
        physical[p.observation_order.index("lane_lateral_error_std_m")], 0.05
    )


def test_sham_is_bitwise_identical_after_normalization() -> None:
    p = protocol()
    values = neutral_physical()
    changed, intended = intervention(values, "sham")
    expected = np.asarray(
        values / np.asarray(p.observation_scales, dtype=np.float32), dtype=np.float32
    )
    np.testing.assert_array_equal(changed, expected)
    assert intended == ()


def test_spearman_agreement_has_expected_extremes() -> None:
    assert np.isclose(spearman([1, 2, 3], [1, 2, 3]), 1.0)
    assert np.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_development_script_has_no_privileged_runtime_input() -> None:
    source = (ROOT / "experiments" / "run_f11_r002_r003_development.py").read_text()
    assert ".privileged.read(" not in source
    assert "PrivilegedSimulatorState" not in source


def test_r003_sham_uses_exact_actor_input_not_physical_roundtrip() -> None:
    source = (ROOT / "experiments" / "run_f11_r002_r003_development.py").read_text()
    assert "changed = observations[row_index].copy()" in source


def test_r003_positive_gate_names_do_not_encode_false_as_failure() -> None:
    source = (ROOT / "experiments" / "run_f11_r002_r003_development.py").read_text()
    assert '"no_privileged_truth_stored": True' in source
    assert '"stored_privileged_truth": False' in source  # report field, outside criteria
