"""Agent-visible environment transition without privileged simulator truth."""

from dataclasses import dataclass

from .observation import SensorObservation
from .reward import RewardTerms


@dataclass(frozen=True)
class Transition:
    observation: SensorObservation
    reward_terms: RewardTerms
    terminated: bool
    truncated: bool

    @property
    def reward(self) -> float:
        return self.reward_terms.total
