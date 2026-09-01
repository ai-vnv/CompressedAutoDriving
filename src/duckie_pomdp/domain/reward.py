"""Reward decomposition contract. No reward formula is implemented."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardTerms:
    progress: float
    lane: float
    stop: float
    pedestrian: float
    comfort: float
    collision: float

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.lane
            + self.stop
            + self.pedestrian
            + self.comfort
            + self.collision
        )
