from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.compression_diagnostics import (
    assign_complete_references,
    counterfactual_preservation_classification,
    evaluate_semantic_counterfactuals,
    load_f14_config,
    load_frozen_actors,
    load_policy_contract,
    normalized_to_physical,
    unresolved_evidence,
)
from duckie_pomdp.explain.group_shapley import (
    GROUP_ORDER,
    build_coalition_vectors,
    coalition_schema,
    exact_group_shapley,
    validate_group_partition,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f14_explainability_aware_compression_v1.toml"


def config():
    return load_f14_config(CONFIG)


def test_exact_29d_partition_has_six_non_overlapping_groups() -> None:
    _, names, indexes = load_policy_contract(config())
    assert tuple(indexes) == GROUP_ORDER
    flat = [index for values in indexes.values() for index in values]
    assert len(names) == 29
    assert sorted(flat) == list(range(29))


def test_all_64_coalitions_are_generated_once() -> None:
    coalitions = coalition_schema()
    assert coalitions.shape == (64, 6)
    assert len({tuple(row) for row in coalitions.tolist()}) == 64
    assert not coalitions[0].any()
    assert coalitions[-1].all()


def test_coalition_complement_uses_one_complete_reference_row() -> None:
    _, names, indexes = load_policy_contract(config())
    factual = np.arange(29, dtype=np.float32)
    reference = 1000.0 + np.arange(29, dtype=np.float32)
    coalitions = build_coalition_vectors(factual, reference, indexes)[0]
    mask = 1 << GROUP_ORDER.index("Pedestrian")
    row = coalitions[mask]
    for group, columns in indexes.items():
        expected = factual if group == "Pedestrian" else reference
        np.testing.assert_array_equal(row[list(columns)], expected[list(columns)])
    np.testing.assert_array_equal(coalitions[0], reference)
    np.testing.assert_array_equal(coalitions[-1], factual)


def test_exact_shapley_local_accuracy_and_repeatability() -> None:
    _, names, indexes = load_policy_contract(config())
    rng = np.random.default_rng(14)
    factual = rng.normal(0.0, 0.1, size=(5, 29)).astype(np.float32)
    references = rng.normal(0.0, 0.1, size=(2, 3, 5, 29)).astype(np.float32)
    # Preserve schema-bearing public columns.
    for matrix in (factual, references.reshape(-1, 29)):
        matrix[:, names.index("lane_validity_probability")] = 1.0
        matrix[:, names.index("pedestrian_existence_probability")] = 0.0
        matrix[:, names.index("stop_sign_existence_probability")] = 0.0
        for field in (
            "lane_lateral_error_std_m", "lane_heading_error_std_rad",
            "lane_curvature_std_inv_m", "pedestrian_range_std_m",
            "pedestrian_bearing_std_rad", "pedestrian_radial_velocity_std_mps",
            "pedestrian_bearing_rate_std_rad_s", "stop_sign_range_std_m",
            "stop_sign_bearing_std_rad",
        ):
            matrix[:, names.index(field)] = 0.1
        matrix[:, [names.index("stop_mode_none"), names.index("stop_mode_required"), names.index("stop_mode_satisfied")]] = (1.0, 0.0, 0.0)

    weights = np.linspace(-0.2, 0.2, 58, dtype=np.float32).reshape(29, 2)
    def actor(x):
        return np.asarray(x @ weights + np.asarray((0.2, 0.0), dtype=np.float32), dtype=np.float32)
    first = exact_group_shapley(actor, factual, references, indexes, names, observation_clip=3.0)
    second = exact_group_shapley(actor, factual, references, indexes, names, observation_clip=3.0)
    assert np.max(np.abs(first.efficiency_residual)) < 1.0e-6
    np.testing.assert_array_equal(first.mean_attribution, second.mean_attribution)


def test_fp32_and_int8_wrappers_are_deterministic_and_physical() -> None:
    actors = load_frozen_actors(config())
    x = np.zeros((3, 29), dtype=np.float32)
    for variant in ("A0", "A7"):
        first = actors[variant].physical(x)
        second = actors[variant].physical(x)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (3, 2)
        assert np.all((0.0 <= first[:, 0]) & (first[:, 0] <= 0.4))
        assert np.all((-4.0 <= first[:, 1]) & (first[:, 1] <= 4.0))
    assert str(actors["A7"].module.inlined_graph).count("quantized::linear") >= 3


def test_physical_action_mapping_is_frozen() -> None:
    actual = normalized_to_physical(np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(actual, [[0.0, -4.0], [0.2, 0.0], [0.4, 4.0]])


def test_reference_assignment_is_same_phase_cross_seed_and_nonself() -> None:
    observations = np.arange(8 * 2 * 29, dtype=np.float32).reshape(16, 29) / 1000.0
    phases = np.asarray(["nominal"] * 8 + ["lane_curve"] * 8)
    seeds = np.asarray(list(range(8)) + list(range(8)))
    factual = np.asarray([0, 8], dtype=np.int64)
    refs, indexes = assign_complete_references(
        observations, phases, seeds, factual,
        draw_seeds=(1, 2), references_per_draw=4,
    )
    assert refs.shape == (2, 4, 2, 29)
    for draw in range(2):
        for state, fact in enumerate(factual):
            chosen = indexes[draw, :, state]
            assert all(phases[index] == phases[fact] for index in chosen)
            assert all(seeds[index] != seeds[fact] for index in chosen)
            assert len(set(seeds[chosen].tolist())) == 4


def test_sham_is_exact_identity_and_intervention_masks_are_registered() -> None:
    cfg = config()
    protocol_path = (CONFIG.parent / cfg["frozen"]["policy_config"]).resolve()
    protocol = load_ppo_curriculum_protocol(protocol_path)
    actors = load_frozen_actors(cfg)
    source = np.load(
        (CONFIG.parent / cfg["development"]["source_dataset"]).resolve(), allow_pickle=False
    )
    normalized = np.asarray(source["observation"][:4], dtype=np.float32)
    physical = np.asarray(source["physical_observation"][:4], dtype=np.float32)
    factual, effects, intended = evaluate_semantic_counterfactuals(
        actors["A7"], normalized, physical, protocol,
        tuple(cfg["counterfactual"]["interventions"]),
        lane_low_confidence_validity=0.25,
        lane_low_confidence_min_lateral_std_m=0.20,
        lane_low_confidence_min_heading_std_rad=0.50,
        lane_low_confidence_min_curvature_std_inv_m=2.50,
    )
    assert factual.shape == (4, 2)
    sham = tuple(cfg["counterfactual"]["interventions"]).index("sham")
    np.testing.assert_array_equal(effects[sham], np.zeros((4, 2), dtype=np.float32))
    assert intended["pedestrian_absent"] == tuple(protocol.observation_order[10:19])
    assert intended["lane_centered"] == (
        "lane_lateral_error_mean_m", "lane_heading_error_mean_rad"
    )


def test_actor_hashes_match_frozen_registry() -> None:
    actors = load_frozen_actors(config())
    assert tuple(actors) == tuple(f"A{i}" for i in range(8))
    assert actors["A7"].sha256 == "f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e"


def test_unavailable_evidence_is_unresolved_not_zero() -> None:
    value = unresolved_evidence("no compatible saved per-step public trajectory")
    assert value["classification"] == "UNRESOLVED"
    assert value["value"] is None
    assert value["value"] != 0


def test_counterfactual_gate_uses_three_preregistered_cells_and_sham() -> None:
    rows = []
    for intervention, phase, action in (
        ("pedestrian_absent", "pedestrian_relevant", "v_cmd_mps"),
        ("stop_absent", "stop_required", "v_cmd_mps"),
        ("lane_centered", "lane_curve", "omega_cmd_rad_s"),
    ):
        rows.append({"intervention": intervention, "phase": phase, "action": action,
                     "paired_direction_agreement": 1.0, "normalized_mean_effect_drift": 0.0,
                     "normalized_p95_effect_drift": 0.0, "reference_mean_absolute": 0.1,
                     "candidate_mean_absolute": 0.1})
    rows.append({"intervention": "sham", "phase": "nominal", "action": "v_cmd_mps",
                 "paired_direction_agreement": 1.0, "normalized_mean_effect_drift": 0.0,
                 "normalized_p95_effect_drift": 0.0, "reference_mean_absolute": 0.0,
                 "candidate_mean_absolute": 0.0})
    result = counterfactual_preservation_classification(rows, {
        "sham_absolute_tolerance": 1e-7, "minimum_direction_agreement": 0.9,
        "maximum_normalized_mean_effect_drift": 0.1,
        "maximum_normalized_p95_effect_drift": 0.25,
    })
    assert result["classification"] == "PRESERVED"
    assert result["total_primary_cells"] == 3
    assert result["sham_gate"] == "PASS"


def test_serialization_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.npz"
    expected = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
    np.savez_compressed(path, attribution=expected, groups=np.asarray(GROUP_ORDER))
    actual = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(actual["attribution"], expected)
    assert tuple(actual["groups"]) == GROUP_ORDER


def test_historical_f11_f12_f13_reports_are_immutable() -> None:
    expected = {
        "refine-logs/EXPERIMENT_PLAN.md": "e257ede7e62791305924df9a2f1d38245e321dbc373fb2c83d432392585a5d62",
        "docs/F11_FINAL_EXPLANATION_SUMMARY.md": "6c31fb9bc3306e2e75b40c4117b4e49151f1d18617afe7c12115634bc1cc7249",
        "docs/F11_R004_REPORT_FOR_REVIEW.md": "c6194366fb8c018d3d12bf6ee3d3ec0de68099f8b147e05a3d1ca5ab9fa8e9eb",
        "docs/F12_COMPRESSION_PROTOCOL.md": "eb6b8c99f6f32ac8084d5917eadc6d0cc41e368e17281a601cf437daea386711",
        "docs/F12_COMPRESSION_RESULTS.md": "9b62c18f76b6f050690e6bfce07b12d731a41e24c18dfca852cc9a94145f6d3a",
        "docs/F12_COMPRESSION_ABLATION.md": "2de0488ccc75d2a37fbf8f9786d34aabcd6016f43f284e57efeb74e70fb29ddc",
        "docs/F13_EXPLAIN_AGAIN_PROTOCOL.md": "736af8710b25c763fa42599cae84dcd5c8821229c67bd65cf7743c22566578ca",
        "docs/F13_EXPLANATION_COMPARISON.md": "a63ac1fa2cbcbd05e5469545788f3b63a55df2a89e44fc2ad3700265a273d7b1",
        "docs/F13_FAILURE_MODE_REPORT.md": "132a7a1ca7a06c71ba4047a9287cf02967ba16a42cd621b6a737b3e9e235d90b",
        "docs/F13_FINAL_REPORT.md": "76ef408aecf0039408b6773d97d4cbdb4c8b26e930365e757f5133c5ec1ff420",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_no_privileged_names_can_enter_group_partition() -> None:
    names = [f"feature_{i}" for i in range(29)]
    names[0] = "evaluation_gt"
    groups = {name: [] for name in GROUP_ORDER}
    groups["Lane"] = names
    try:
        validate_group_partition(names, groups)
    except ValueError as error:
        assert "privileged" in str(error)
    else:  # pragma: no cover
        raise AssertionError("privileged field was accepted")
