"""Image-only object-detector port shared by runtime adapters."""

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.domain.detection import Detection


class ObjectDetector(Protocol):
    def detect(self, rgb: NDArray[np.uint8]) -> Sequence[Detection]: ...
