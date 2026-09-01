"""Camera source port."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class CameraSource(Protocol):
    def capture(self) -> NDArray[np.uint8]: ...

