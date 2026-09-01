"""Deployment optimization for the frozen Belief-PPO actor."""

from duckie_pomdp.optimization.actor_compression import (
    DenseBeliefActor,
    QuantizableBeliefActor,
    build_pruned_actor,
    extract_original_actor,
)
from duckie_pomdp.optimization.cross_curriculum_recovery import (
    CURRICULA,
    HUMAN_NAMES,
    RetentionDecision,
    curriculum_balanced_probabilities,
    distill_multicurriculum_actor,
    first_objective_failure_event,
    retention_decision,
)

__all__ = [
    "DenseBeliefActor",
    "QuantizableBeliefActor",
    "build_pruned_actor",
    "extract_original_actor",
    "CURRICULA",
    "HUMAN_NAMES",
    "RetentionDecision",
    "curriculum_balanced_probabilities",
    "distill_multicurriculum_actor",
    "first_objective_failure_event",
    "retention_decision",
]
