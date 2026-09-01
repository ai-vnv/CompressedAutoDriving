"""Test collection rules for a checkout without the evaluation artifacts.

Part of this suite is a provenance suite: it re-derives numbers from the
evaluation ledgers under ``artifacts/``, hash-checks the frozen study
documents, and reads the training datasets. Those inputs are not distributed
with the repository, since the telemetry, rollout media, result ledgers, and
image datasets run to several gigabytes and are regenerated from ``configs/``
and ``experiments/``.

So that ``pytest tests -q`` is meaningful on a fresh clone, the modules that
need those inputs are collected only when the inputs are present. Everything
else -- the perception, belief, environment, action, and compression unit
tests -- runs from the repository alone.

Regenerate the artifacts and the full suite collects automatically; nothing
here needs to be edited to switch modes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
DATASETS = ROOT / "datasets"

# Modules that read the evaluation ledgers, the recorded media, the frozen
# study documents, or the training datasets. Established by running the suite
# against a checkout with those paths absent; every entry fails or errors
# without them.
REQUIRES_ARTIFACTS_OR_DOCS = (
    "test_evaluate_f9c_robust_belief.py",
    "test_experiment_loop_objects_v10.py",
    "test_experiment_loop_objects_v11.py",
    "test_experiment_loop_objects_v12.py",
    "test_f9_pipeline.py",
    "test_f9c_leakage.py",
    "test_f9c_protocol.py",
    "test_f9c_robust_updater.py",
    "test_f9d_association.py",
    "test_f9d_protocol.py",
    "test_f10_environment.py",
    "test_f10_evaluation.py",
    "test_f10_l2_environment.py",
    "test_f10_l2_protocol.py",
    "test_f10_l2_transfer.py",
    "test_f10_policy_interface.py",
    "test_f10_ppo_environment.py",
    "test_f10_ppo_evaluation.py",
    "test_f10_ppo_observation.py",
    "test_f10_ppo_protocol.py",
    "test_f10_ppo_stop_belief.py",
    "test_f10_pretraining.py",
    "test_f10_protocol.py",
    "test_f10_reward.py",
    "test_f11_explanation_development.py",
    "test_f11_ppo_observation_contract.py",
    "test_f11_r006_confirmatory_intervention.py",
    "test_f12_compression.py",
    "test_f13_compressed_explanation.py",
    "test_f14_explainability_aware_compression.py",
    "test_f14_explained_documentation.py",
    "test_f15_cross_curriculum_recovery.py",
    "test_lane_rgb_competence_v9.py",
    "test_lane_rgb_model.py",
    "test_ppo_behavior_warm_start_v13.py",
    "test_ppo_c2_rehearsal_v14.py",
    "test_ppo_c3_dagger_v18.py",
    "test_ppo_c3_on_policy_v19.py",
    "test_ppo_c3_warm_start_v17.py",
    "test_ppo_c4_combined_v21.py",
    "test_ppo_c4_cumulative_dagger_v23.py",
    "test_ppo_c4_dagger_v22.py",
    "test_ppo_c4_privileged_guidance_v24.py",
    "test_ppo_existence_gate_v30.py",
    "test_ppo_multitask_warm_start_v15.py",
    "test_v4_lane_belief_uncertainty.py",
    "test_v4_start_sampler.py",
    "test_verify_f9d_artifacts.py",
)

DISTRIBUTED_DATA = (ARTIFACTS, DOCS, DATASETS)

collect_ignore = []
if not all(p.is_dir() for p in DISTRIBUTED_DATA):
    collect_ignore.extend(REQUIRES_ARTIFACTS_OR_DOCS)


def pytest_report_header(config):
    if collect_ignore:
        missing = [p.name for p in DISTRIBUTED_DATA if not p.is_dir()]
        return (f"provenance suite not collected: {', '.join(missing)} absent "
                f"({len(collect_ignore)} modules skipped)")
    return "provenance suite collected: artifacts and study documents present"
