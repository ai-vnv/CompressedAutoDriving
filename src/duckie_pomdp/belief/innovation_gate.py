"""Innovation-consistency gate. Consumes only filter statistics, never truth.

Hard rejection only. A soft-downweight branch would make the EKF correction use
a different measurement covariance than the one association and this gate
thresholded against, breaking the single-innovation-covariance invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InnovationGateConfig:
    chi_square_threshold: float

    def __post_init__(self) -> None:
        if not isfinite(self.chi_square_threshold) or self.chi_square_threshold <= 0.0:
            raise ValueError("chi-square threshold must be finite and positive")


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    nis: float
    threshold: float


def normalized_innovation_squared(
    innovation: NDArray[np.float64],
    innovation_covariance: NDArray[np.float64],
) -> float:
    vector = np.asarray(innovation, dtype=float)
    matrix = np.asarray(innovation_covariance, dtype=float)
    if vector.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError("innovation gate expects a 2D polar innovation")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        raise ValueError("innovation statistics must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError("innovation covariance must be positive definite")
    return float(vector @ np.linalg.solve(symmetric, vector))


class InnovationGate:
    def __init__(self, config: InnovationGateConfig) -> None:
        self.config = config

    def evaluate(
        self,
        innovation: NDArray[np.float64],
        innovation_covariance: NDArray[np.float64],
    ) -> GateDecision:
        nis = normalized_innovation_squared(innovation, innovation_covariance)
        threshold = self.config.chi_square_threshold
        return GateDecision(nis <= threshold, nis, threshold)
