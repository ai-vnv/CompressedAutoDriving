"""Port translating policy motion commands to simulator wheel commands."""

from typing import Protocol

from duckie_pomdp.domain.action import PolicyAction, WheelCommand


class ActionAdapter(Protocol):
    def to_wheels(self, action: PolicyAction) -> WheelCommand: ...

