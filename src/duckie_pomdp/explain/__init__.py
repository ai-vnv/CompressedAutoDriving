"""Interpretability utilities for frozen Duckie-POMDP policies."""

from .ppo_integrated_gradients import (
    DistributionalIntegratedGradientsResult,
    IntegratedGradientsResult,
    PPOActionLimits,
    distributional_integrated_gradients,
    integrated_gradients,
    target_values,
)
from .ig_bev import (
    GroupAttribution,
    aggregate_groups,
    align_pose_to_samples,
    resolve_feature_groups,
    signed_total,
    validate_pose_trace,
)
from .observation_contract import (
    assert_policy_vector_precedes_privileged_read,
    deterministic_actor_statistics,
    reconstruct_normalized_observation,
    validate_feature_group_partition,
    validate_public_policy_mapping,
)
from .development_protocol import (
    PhaseThresholds,
    apply_semantic_intervention,
    build_r002_baselines,
    draw_phase_conditioned_references,
    group_absolute_shares,
    normalize_physical,
    public_phase,
    schema_valid_public_vector,
    spearman,
)

__all__ = [
    "DistributionalIntegratedGradientsResult",
    "IntegratedGradientsResult",
    "PPOActionLimits",
    "distributional_integrated_gradients",
    "integrated_gradients",
    "target_values",
    "GroupAttribution",
    "aggregate_groups",
    "align_pose_to_samples",
    "resolve_feature_groups",
    "signed_total",
    "validate_pose_trace",
    "assert_policy_vector_precedes_privileged_read",
    "deterministic_actor_statistics",
    "reconstruct_normalized_observation",
    "validate_feature_group_partition",
    "validate_public_policy_mapping",
    "PhaseThresholds",
    "apply_semantic_intervention",
    "build_r002_baselines",
    "draw_phase_conditioned_references",
    "group_absolute_shares",
    "normalize_physical",
    "public_phase",
    "schema_valid_public_vector",
    "spearman",
]
from duckie_pomdp.explain.compressed_policy_analysis import (
    actor_physical,
    classification_from_counterfactual,
    normalized_to_physical,
    paired_effect_metrics,
)

__all__ += [
    "actor_physical",
    "classification_from_counterfactual",
    "normalized_to_physical",
    "paired_effect_metrics",
]
