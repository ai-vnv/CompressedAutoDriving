"""Belief-update port with explicit control and actual-motion inputs."""

from typing import Protocol

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState
from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.state import EgoMotion


class BeliefUpdater(Protocol):
    def update(
        self,
        previous_belief: BeliefState,
        previous_action: PolicyAction,
        ego_motion: EgoMotion,
        perception: PerceptionObservation,
        dt_s: float,
    ) -> BeliefState: ...
