"""Gate F9c final-evaluation runtime cache -- invariants I4 and I5.

**I4 (ablation runs zero inference).** The final evaluation (Task 11) is the
only process ever allowed to construct a detector or a simulator for the
7101-7104 seeds. Every candidate the runtime pipeline produced, on every
frame, is persisted here so a later ablation (Task 12) can replay the exact
same candidates through different `RobustObservationConfig` switch
combinations without re-rendering or re-running YOLO -- which would not only
waste the one-inference-pass guarantee but could produce *different*
candidates under domain randomization, silently invalidating the "same
frames" comparison the whole gate depends on.

**I5 (raw, pre-bias candidates only).** Every candidate field here is named
``raw_*`` and must be captured at the output of
``YoloPedestrianMeasurementPipeline.observe`` -- before either bias stage
(F9b's frozen ``AdditiveMeasurementBias`` or F9c's ``FrozenBiasCorrection``)
ever touches it. This is what lets one cached candidate feed both paths:

    raw candidate --+-- F9b bias --> Baseline A row
                     +-- F9c bias --> Robust B row

Had the cache stored a bias-corrected candidate, an ablation row with
``bias_refit=False`` would be replaying ``z_F9c_corrected - b_F9b`` --
neither Baseline A nor anything meaningful -- and would silently corrupt the
one ablation row (`baseline`) that is supposed to reproduce Baseline A
exactly.

This module is pure data plus I/O. It knows nothing about the detector, the
simulator, or the belief layer, and it must never import from
``perception`` or ``belief`` -- callers (the Task 11 experiment script)
build ``RuntimeCacheFrame`` instances from whatever richer objects they hold
and hand this module only their raw scalars/tuples.

Ragged per-frame candidate lists are stored as one flat array per field plus
an explicit ``raw_candidate_offsets`` index (CSR-style), never
``dtype=object`` -- so the ``.npz`` round-trips through ``np.load`` with
``allow_pickle=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from duckie_pomdp.evaluation.f9_protocol import sha256

_STRING_DTYPE = "<U256"


@dataclass(frozen=True)
class RuntimeCacheFrame:
    """One frame's runtime-visible, pre-bias state. See module docstring."""

    episode: str
    seed: int
    scenario: str
    frame: int
    dt_s: float
    raw_candidate_range_m: tuple[float, ...]
    raw_candidate_bearing_rad: tuple[float, ...]
    raw_candidate_confidence: tuple[float, ...]
    raw_candidate_bbox: tuple[tuple[float, float, float, float], ...]
    raw_candidate_projection_failed: tuple[bool, ...]
    ego_linear_velocity_mps: float
    ego_yaw_rate_rad_s: float

    def __post_init__(self) -> None:
        n = len(self.raw_candidate_range_m)
        for name, sequence in (
            ("raw_candidate_bearing_rad", self.raw_candidate_bearing_rad),
            ("raw_candidate_confidence", self.raw_candidate_confidence),
            ("raw_candidate_bbox", self.raw_candidate_bbox),
            ("raw_candidate_projection_failed", self.raw_candidate_projection_failed),
        ):
            if len(sequence) != n:
                raise ValueError(
                    f"RuntimeCacheFrame field length mismatch: "
                    f"raw_candidate_range_m has {n} entries but {name} has "
                    f"{len(sequence)}"
                )
        for box in self.raw_candidate_bbox:
            if len(box) != 4:
                raise ValueError(
                    "each raw_candidate_bbox entry must have exactly 4 values "
                    f"(x_min, y_min, x_max, y_max); got {box!r}"
                )

    @property
    def raw_candidate_count(self) -> int:
        return len(self.raw_candidate_range_m)


def write_runtime_cache(path: str | Path, frames: Sequence[RuntimeCacheFrame]) -> str:
    """Write ``frames`` to ``path`` as a ``.npz`` and return its SHA256.

    The returned hash is what the caller must record (Task 11 records it in
    ``artifacts/f9c_belief_metrics.json``) so a later replay can prove the
    cache it is reading is the one the headline run actually produced.
    """

    if not frames:
        raise ValueError("cannot write an empty runtime cache")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    counts = np.array([frame.raw_candidate_count for frame in frames], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    total = int(offsets[-1])

    episode = np.array([frame.episode for frame in frames], dtype=_STRING_DTYPE)
    scenario = np.array([frame.scenario for frame in frames], dtype=_STRING_DTYPE)
    seed = np.array([frame.seed for frame in frames], dtype=np.int64)
    frame_index = np.array([frame.frame for frame in frames], dtype=np.int64)
    dt_s = np.array([frame.dt_s for frame in frames], dtype=np.float64)
    ego_linear_velocity_mps = np.array(
        [frame.ego_linear_velocity_mps for frame in frames], dtype=np.float64
    )
    ego_yaw_rate_rad_s = np.array(
        [frame.ego_yaw_rate_rad_s for frame in frames], dtype=np.float64
    )

    range_m = np.empty(total, dtype=np.float64)
    bearing_rad = np.empty(total, dtype=np.float64)
    confidence = np.empty(total, dtype=np.float64)
    bbox = np.empty((total, 4), dtype=np.float64)
    projection_failed = np.empty(total, dtype=bool)

    cursor = 0
    for frame in frames:
        count = frame.raw_candidate_count
        if count:
            end = cursor + count
            range_m[cursor:end] = frame.raw_candidate_range_m
            bearing_rad[cursor:end] = frame.raw_candidate_bearing_rad
            confidence[cursor:end] = frame.raw_candidate_confidence
            bbox[cursor:end] = np.asarray(frame.raw_candidate_bbox, dtype=np.float64)
            projection_failed[cursor:end] = frame.raw_candidate_projection_failed
            cursor = end

    np.savez(
        path,
        episode=episode,
        scenario=scenario,
        seed=seed,
        frame=frame_index,
        dt_s=dt_s,
        raw_candidate_count=counts,
        raw_candidate_offsets=offsets,
        raw_candidate_range_m=range_m,
        raw_candidate_bearing_rad=bearing_rad,
        raw_candidate_confidence=confidence,
        raw_candidate_bbox=bbox,
        raw_candidate_projection_failed=projection_failed,
        ego_linear_velocity_mps=ego_linear_velocity_mps,
        ego_yaw_rate_rad_s=ego_yaw_rate_rad_s,
    )
    # np.savez appends .npz if the caller's path lacks the suffix.
    written_path = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return sha256(written_path)


def read_runtime_cache(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[RuntimeCacheFrame, ...]:
    """Read a cache written by ``write_runtime_cache``.

    If ``expected_sha256`` is given (the ablation's caller should always
    pass the value recorded in ``artifacts/f9c_belief_metrics.json``), the
    file's current SHA256 is checked against it first and a ``ValueError``
    is raised on any mismatch -- a silently regenerated cache must never be
    replayed as if it were the run that produced the recorded hash.
    """

    path = Path(path)
    if expected_sha256 is not None:
        actual = sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"runtime cache hash mismatch for {path}: expected "
                f"{expected_sha256}, got {actual} -- refusing to replay a "
                "cache that does not match the recorded hash"
            )

    with np.load(path, allow_pickle=False) as data:
        episode = data["episode"]
        scenario = data["scenario"]
        seed = data["seed"]
        frame_index = data["frame"]
        dt_s = data["dt_s"]
        counts = data["raw_candidate_count"]
        offsets = data["raw_candidate_offsets"]
        range_m = data["raw_candidate_range_m"]
        bearing_rad = data["raw_candidate_bearing_rad"]
        confidence = data["raw_candidate_confidence"]
        bbox = data["raw_candidate_bbox"]
        projection_failed = data["raw_candidate_projection_failed"]
        ego_linear_velocity_mps = data["ego_linear_velocity_mps"]
        ego_yaw_rate_rad_s = data["ego_yaw_rate_rad_s"]

        frames: list[RuntimeCacheFrame] = []
        for index in range(len(episode)):
            start, end = int(offsets[index]), int(offsets[index + 1])
            frames.append(
                RuntimeCacheFrame(
                    episode=str(episode[index]),
                    seed=int(seed[index]),
                    scenario=str(scenario[index]),
                    frame=int(frame_index[index]),
                    dt_s=float(dt_s[index]),
                    raw_candidate_range_m=tuple(
                        float(value) for value in range_m[start:end]
                    ),
                    raw_candidate_bearing_rad=tuple(
                        float(value) for value in bearing_rad[start:end]
                    ),
                    raw_candidate_confidence=tuple(
                        float(value) for value in confidence[start:end]
                    ),
                    raw_candidate_bbox=tuple(
                        tuple(float(value) for value in row) for row in bbox[start:end]
                    ),
                    raw_candidate_projection_failed=tuple(
                        bool(value) for value in projection_failed[start:end]
                    ),
                    ego_linear_velocity_mps=float(ego_linear_velocity_mps[index]),
                    ego_yaw_rate_rad_s=float(ego_yaw_rate_rad_s[index]),
                )
            )
            if int(counts[index]) != len(frames[-1].raw_candidate_range_m):
                raise ValueError(
                    f"runtime cache offsets are inconsistent with "
                    f"raw_candidate_count at row {index}"
                )
    return tuple(frames)


@dataclass(frozen=True)
class TruthFrame:
    """Ground truth for one ``(episode, frame)``, kept out of the runtime
    cache (invariant I4/I5: a replay consumer must be handable the runtime
    cache alone and never needs this file to reconstruct candidates)."""

    episode: str
    frame: int
    gt_exists: bool
    gt_range_m: float | None
    gt_bearing_rad: float | None
    gt_range_rate_mps: float | None
    gt_bearing_rate_rad_s: float | None
    eligible_visible: bool
    visible_pixel_count: int
    gt_bbox: tuple[float, float, float, float] | None
    distance_bin: str
    fov_region: str


def write_evaluation_truth(path: str | Path, frames: Sequence[TruthFrame]) -> str:
    """Write ground truth to ``path``, keyed by ``(episode, frame)``, and
    return its SHA256."""

    if not frames:
        raise ValueError("cannot write empty evaluation truth")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _nan_if_none(value: float | None) -> float:
        return float("nan") if value is None else float(value)

    episode = np.array([frame.episode for frame in frames], dtype=_STRING_DTYPE)
    frame_index = np.array([frame.frame for frame in frames], dtype=np.int64)
    gt_exists = np.array([frame.gt_exists for frame in frames], dtype=bool)
    gt_range_m = np.array([_nan_if_none(frame.gt_range_m) for frame in frames], dtype=np.float64)
    gt_bearing_rad = np.array(
        [_nan_if_none(frame.gt_bearing_rad) for frame in frames], dtype=np.float64
    )
    gt_range_rate_mps = np.array(
        [_nan_if_none(frame.gt_range_rate_mps) for frame in frames], dtype=np.float64
    )
    gt_bearing_rate_rad_s = np.array(
        [_nan_if_none(frame.gt_bearing_rate_rad_s) for frame in frames], dtype=np.float64
    )
    eligible_visible = np.array([frame.eligible_visible for frame in frames], dtype=bool)
    visible_pixel_count = np.array(
        [frame.visible_pixel_count for frame in frames], dtype=np.int64
    )
    has_gt_bbox = np.array([frame.gt_bbox is not None for frame in frames], dtype=bool)
    gt_bbox = np.array(
        [
            [float("nan")] * 4 if frame.gt_bbox is None else list(frame.gt_bbox)
            for frame in frames
        ],
        dtype=np.float64,
    )
    distance_bin = np.array([frame.distance_bin for frame in frames], dtype=_STRING_DTYPE)
    fov_region = np.array([frame.fov_region for frame in frames], dtype=_STRING_DTYPE)

    np.savez(
        path,
        episode=episode,
        frame=frame_index,
        gt_exists=gt_exists,
        gt_range_m=gt_range_m,
        gt_bearing_rad=gt_bearing_rad,
        gt_range_rate_mps=gt_range_rate_mps,
        gt_bearing_rate_rad_s=gt_bearing_rate_rad_s,
        eligible_visible=eligible_visible,
        visible_pixel_count=visible_pixel_count,
        has_gt_bbox=has_gt_bbox,
        gt_bbox=gt_bbox,
        distance_bin=distance_bin,
        fov_region=fov_region,
    )
    written_path = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return sha256(written_path)


def read_evaluation_truth(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[tuple[str, int], TruthFrame]:
    path = Path(path)
    if expected_sha256 is not None:
        actual = sha256(path)
        if actual != expected_sha256:
            raise ValueError(
                f"evaluation truth hash mismatch for {path}: expected "
                f"{expected_sha256}, got {actual}"
            )

    result: dict[tuple[str, int], TruthFrame] = {}
    with np.load(path, allow_pickle=False) as data:
        episode = data["episode"]
        frame_index = data["frame"]
        gt_exists = data["gt_exists"]
        gt_range_m = data["gt_range_m"]
        gt_bearing_rad = data["gt_bearing_rad"]
        gt_range_rate_mps = data["gt_range_rate_mps"]
        gt_bearing_rate_rad_s = data["gt_bearing_rate_rad_s"]
        eligible_visible = data["eligible_visible"]
        visible_pixel_count = data["visible_pixel_count"]
        has_gt_bbox = data["has_gt_bbox"]
        gt_bbox = data["gt_bbox"]
        distance_bin = data["distance_bin"]
        fov_region = data["fov_region"]

        def _none_if_nan(value: float) -> float | None:
            return None if np.isnan(value) else float(value)

        for index in range(len(episode)):
            key = (str(episode[index]), int(frame_index[index]))
            result[key] = TruthFrame(
                episode=key[0],
                frame=key[1],
                gt_exists=bool(gt_exists[index]),
                gt_range_m=_none_if_nan(gt_range_m[index]),
                gt_bearing_rad=_none_if_nan(gt_bearing_rad[index]),
                gt_range_rate_mps=_none_if_nan(gt_range_rate_mps[index]),
                gt_bearing_rate_rad_s=_none_if_nan(gt_bearing_rate_rad_s[index]),
                eligible_visible=bool(eligible_visible[index]),
                visible_pixel_count=int(visible_pixel_count[index]),
                gt_bbox=(
                    None
                    if not bool(has_gt_bbox[index])
                    else tuple(float(value) for value in gt_bbox[index])
                ),
                distance_bin=str(distance_bin[index]),
                fov_region=str(fov_region[index]),
            )
    return result
