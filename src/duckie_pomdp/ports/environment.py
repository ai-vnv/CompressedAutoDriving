"""Separate agent interaction from simulator-only truth access."""

from typing import Protocol

from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.observation import SensorObservation
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState
from duckie_pomdp.domain.transition import Transition


class AgentEnvironment(Protocol):
    """The only environment interface available to policy orchestration."""

    def reset(self, seed: int | None = None) -> SensorObservation: ...

    def step(self, action: PolicyAction) -> Transition: ...

    def close(self) -> None: ...


class PrivilegedStateSource(Protocol):
    """Evaluation-only interface kept outside AgentEnvironment."""

    def read(self) -> PrivilegedSimulatorState: ...
