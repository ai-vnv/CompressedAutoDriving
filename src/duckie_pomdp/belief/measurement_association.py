"""Temporal measurement association for the pedestrian track.

When the vision system reports several candidate detections in one frame,
this module picks the candidate most consistent with where the tracker
predicts the pedestrian is, scored by normalized innovation squared (NIS).

Association lives in the belief layer, not in perception: perception only
produces candidates, and this module decides which one (if any) corresponds
to the tracked pedestrian. That keeps the hexagonal boundary intact.

Invariant I1 — the single most important constraint on this module:
association, the innovation gate, and the EKF correction must all use the
*same* innovation covariance ``S = H P⁻ Hᵀ + λR`` for the same candidate on
the same frame. That is why ``associate`` takes ``innovation_covariance_for``
as an injected callable rather than building ``S`` itself: the coordinator
hands the identical provider to all three consumers. This module must never
construct an innovation covariance of its own — no ``np.diag(...)`` of
measurement noise, no recomputing ``H P Hᵀ``. It calls the injected callable
and uses what it returns. ``associate`` thresholds against exactly the
covariance its caller will later correct with.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.belief.innovation_gate import normalized_innovation_squared
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.perception.measurement_calibration import wrap_angle


@dataclass(frozen=True)
class AssociationConfig:
    chi_square_gate: float
    initialization_rule: str

    def __post_init__(self) -> None:
        if not isfinite(self.chi_square_gate) or self.chi_square_gate <= 0.0:
            raise ValueError("chi-square gate must be finite and positive")


@dataclass(frozen=True)
class CandidateMeasurement:
    measurement: ObjectMeasurement
    confidence: float
    bbox_key: tuple[int, int, int, int]


@dataclass(frozen=True)
class AssociationResult:
    selected_index: int | None
    selected: CandidateMeasurement | None
    mode: str
    candidate_nis: tuple[float | None, ...]
    highest_confidence_index: int | None
    differed_from_highest_confidence: bool


def _highest_confidence_index(candidates: Sequence[CandidateMeasurement]) -> int | None:
    """Same ordering as ``select_single_duckie``: ``(-confidence, bbox_key)``."""

    if not candidates:
        return None
    indices = range(len(candidates))
    return min(
        indices,
        key=lambda index: (-candidates[index].confidence, candidates[index].bbox_key),
    )


class MeasurementAssociator:
    def __init__(self, config: AssociationConfig) -> None:
        self.config = config

    def associate(
        self,
        candidates: Sequence[CandidateMeasurement],
        *,
        predicted_measurement: NDArray[np.float64] | None,
        innovation_covariance_for: Callable[[float], NDArray[np.float64]],
    ) -> AssociationResult:
        if not candidates:
            return AssociationResult(
                selected_index=None,
                selected=None,
                mode="no_candidate",
                candidate_nis=(),
                highest_confidence_index=None,
                differed_from_highest_confidence=False,
            )

        highest_confidence_index = _highest_confidence_index(candidates)

        if predicted_measurement is None:
            return AssociationResult(
                selected_index=highest_confidence_index,
                selected=candidates[highest_confidence_index],
                mode="initialization",
                candidate_nis=tuple(None for _ in candidates),
                highest_confidence_index=highest_confidence_index,
                differed_from_highest_confidence=False,
            )

        predicted_range = float(predicted_measurement[0])
        predicted_bearing = float(predicted_measurement[1])

        candidate_nis: list[float | None] = []
        for candidate in candidates:
            measurement = candidate.measurement
            range_innovation = measurement.range_m - predicted_range
            bearing_innovation = wrap_angle(measurement.bearing_rad - predicted_bearing)
            innovation = np.array([range_innovation, bearing_innovation], dtype=float)
            covariance = innovation_covariance_for(measurement.range_m)
            nis = normalized_innovation_squared(innovation, covariance)
            candidate_nis.append(nis)

        gated_indices = [
            index
            for index, nis in enumerate(candidate_nis)
            if nis is not None and nis <= self.config.chi_square_gate
        ]

        if not gated_indices:
            return AssociationResult(
                selected_index=None,
                selected=None,
                mode="all_gated_out",
                candidate_nis=tuple(candidate_nis),
                highest_confidence_index=highest_confidence_index,
                differed_from_highest_confidence=False,
            )

        selected_index = min(gated_indices, key=lambda index: candidate_nis[index])
        differed_from_highest_confidence = selected_index != highest_confidence_index

        return AssociationResult(
            selected_index=selected_index,
            selected=candidates[selected_index],
            mode="temporal",
            candidate_nis=tuple(candidate_nis),
            highest_confidence_index=highest_confidence_index,
            differed_from_highest_confidence=differed_from_highest_confidence,
        )
