"""Structured perception boundary between sensors and belief updating."""

from typing import Protocol

from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.observation import SensorObservation


class PerceptionPipeline(Protocol):
    def observe(self, sensors: SensorObservation) -> PerceptionObservation: ...
