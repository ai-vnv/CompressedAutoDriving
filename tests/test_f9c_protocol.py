from pathlib import Path

import pytest

from duckie_pomdp.evaluation.f9c_calibration import load_robust_observation_config
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"


def test_f9c_seeds_are_disjoint_from_every_earlier_split():
    protocol = load_f9c_protocol(CONFIG)
    calibration = set(protocol.calibration_seeds)
    final = set(protocol.final_evaluation_seeds)
    assert calibration and final
    assert not calibration & final
    forbidden = set(protocol.forbidden_seeds)
    assert {5101, 5102, 5103, 5104} <= forbidden
    assert {4101, 4102, 4103, 4104} <= forbidden
    assert not (calibration | final) & forbidden


def test_f9c_may_not_change_frozen_f7_dynamics(tmp_path):
    protocol = load_f9c_protocol(CONFIG)
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    assert f9c["ekf"] == f7["ekf"]


def test_f9c_keeps_survival_and_birth_frozen():
    protocol = load_f9c_protocol(CONFIG)
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    for key in ("prior_probability", "survival_probability", "birth_probability"):
        assert f9c["existence"][key] == f7["existence"][key]


def test_f9c_rejects_a_config_that_edits_process_noise(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "position_process_std_m_per_sqrt_s = 0.001",
        "position_process_std_m_per_sqrt_s = 0.002",
    )
    # The loader resolves sibling paths (frozen_f7_config, checkpoint,
    # artifacts/...) relative to the config file's own directory, so a copy
    # under tmp_path fails path resolution before ever reaching the
    # process-noise check. Write the mutated copy into the real configs/
    # directory under a clearly temporary name instead, and always clean it
    # up.
    broken = ROOT / "configs" / "_tmp_process_noise_probe.toml"
    broken.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="frozen F7"):
            load_f9c_protocol(broken)
    finally:
        broken.unlink(missing_ok=True)


def test_f9c_acceptance_bands_are_pre_specified():
    protocol = load_f9c_protocol(CONFIG)
    bands = protocol.acceptance
    assert bands.coverage_68_low == 0.60 and bands.coverage_68_high == 0.76
    assert bands.coverage_95_low == 0.90 and bands.coverage_95_high == 0.98
    assert bands.max_std_over_rmse == 1.5
    assert protocol.minimum_support["near"] >= 100


def test_f9c_ablation_switches_default_to_all_enabled():
    protocol = load_f9c_protocol(CONFIG)
    switches = protocol.robust
    assert switches.bias_refit
    assert switches.innovation_gate
    assert switches.temporal_association
    assert switches.covariance_calibration
    assert switches.conditional_detection


def test_f9c_scenario_matrix_supports_near_range_final_evaluation():
    protocol = load_f9c_protocol(CONFIG)
    final = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
    calibration = [spec for spec in protocol.scenarios if spec.use_for_calibration]

    # A scenario reaches near range either by starting close, or by starting at
    # medium range and driving in. Classifying by start offset alone would
    # exclude the moving approach, whose whole purpose is to traverse the bin.
    def reaches_near(spec):
        return spec.ego_start_x_offset_m >= 0.35 or (
            spec.action.linear_velocity_mps > 0.0
            and spec.ego_start_x_offset_m >= 0.25
        )

    near_final = [spec for spec in final if reaches_near(spec)]
    near_calibration = [spec for spec in calibration if reaches_near(spec)]
    assert len(near_final) >= 2, "final evaluation must contain near-range scenarios"
    assert len(near_calibration) >= 2
    # A moving-ego approach is required so near range is traversed, not only sampled statically.
    assert any(
        spec.action.linear_velocity_mps > 0.0 for spec in near_final
    ), "at least one near-range final scenario must approach the pedestrian"


def test_load_robust_observation_config_builds_a_coordinator_config():
    """Task 9 is the first task that needs to build a RobustObservationConfig
    from configs/f9c_robust_belief_v1.toml; no loader existed before this.

    Task 10 froze covariance_calibration.range_scale to the final
    calibration re-run's fitted lambda_r (artifacts/f9c_calibration_metrics.json
    covariance_scales.lambda_r); this assertion was updated in lockstep with
    that freeze rather than left checking the pre-freeze placeholder 1.0."""

    config = load_robust_observation_config(CONFIG)
    assert config.switches.bias_refit is True
    assert config.gate.chi_square_threshold == pytest.approx(9.21034037197618)
    assert config.association.chi_square_gate == pytest.approx(13.815510557964274)
    assert config.covariance.range_scale == pytest.approx(9.96243043243885)
    assert config.active_threshold == pytest.approx(0.50)
    assert config.delete_threshold == pytest.approx(0.05)
    assert config.initialization_threshold == pytest.approx(0.50)
