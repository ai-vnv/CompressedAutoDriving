"""Ground-plane projection boundary."""

from typing import Protocol

from duckie_pomdp.domain.detection import ImagePoint
from duckie_pomdp.domain.measurement import GroundPoint


class GroundProjector(Protocol):
    def pixel_to_ground(self, point: ImagePoint) -> GroundPoint: ...
