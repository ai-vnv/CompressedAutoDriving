"""F9d-C: the association selection-rule diagnostic. Cache-only.

Two separable claims about why ``temporal_association_only`` was the worst
row of F9c's ablation (range RMSE 0.03776 vs baseline 0.02580):

    C1  does the wrong innovation covariance ``S`` (association scored
        candidates against ``lambda = 1`` when covariance_calibration was
        off, ~10x too small per the calibration) explain the ablation
        penalty?
    C2  even at the CORRECT, frozen lambda, does minimum-NIS selection pick
        worse boxes than highest-confidence selection?

**Holding the predicted state fixed (the whole point of this module).** A
naive comparison would re-run the belief updater once per selection rule and
compare the resulting range-RMSE trajectories. That measures drift, not
selection quality: the two runs' EKF corrections diverge after the first
frame the rules disagree, so frame 50's "disagreement" would really be
comparing two different point in ten-frame-old history, not the same
decision. This module instead runs the frozen Robust B configuration
EXACTLY ONCE over the cache (``_reference_snapshots``) to produce the real
predicted-state/covariance sequence, and at every frame where a track was
already active going in, snapshots ``(H(x-hat-), P-, h(x-hat-))`` via a
``copy.deepcopy`` of the coordinator's own EKF -- never touching the live
``updater.ekf`` the real trajectory is being driven from. Both counterfactual
scorings (``associate`` calls with different ``innovation_covariance_for``
providers) then run against that ONE frozen snapshot; neither one is ever
fed back into the updater, so the real trajectory that produces frame N+1's
snapshot is always the actual frozen Robust B trajectory, never a
counterfactual one.

This module:

- constructs no ``YoloObjectDetector``, no gym-duckietown simulator -- it
  only ever reads an already-produced ``RuntimeCacheFrame``/``TruthFrame``
  sequence and the frozen F9c config from disk;
- never assigns a new value to any F9c estimator parameter -- every
  ``RobustObservationConfig``/``CovarianceCalibration``/... object is loaded
  fresh from ``configs/f9c_robust_belief_v1.toml`` via the same frozen
  loaders F9c itself uses (``f9c_calibration.load_robust_observation_config``
  etc.), never constructed with a hand-picked numeric override;
- is purely diagnostic: ``compare_selection_rules``/``duplicate_frame_ranking``
  return counts; they never write anything back into the belief layer.

"Localization outlier" (both C1 and C2) means the selected candidate's GT
IoU is below ``configs/f9c_robust_belief_v1.toml``'s own
``calibration_protocol.matching_iou_threshold`` (0.50) -- the same
definition Task 3's ``outlier_yield`` uses, read from the same config
section rather than a second hard-coded literal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from math import cos, sin
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.existence_filter import ExistenceFilterConfig
from duckie_pomdp.belief.measurement_association import (
    AssociationResult,
    CandidateMeasurement,
    MeasurementAssociator,
)
from duckie_pomdp.belief.observability import PredictedObservabilityModel
from duckie_pomdp.belief.pedestrian_ekf import (
    PedestrianEKFConfig,
    load_pedestrian_ekf_config,
    measurement_function,
    measurement_jacobian,
)
from duckie_pomdp.belief.robust_updater import (
    RobustObservationConfig,
    RobustPedestrianBeliefUpdater,
)
from duckie_pomdp.belief.updater import initial_belief, load_existence_filter_config
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.evaluation.f9c_calibration import (
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.evaluation.f9c_runtime_cache import RuntimeCacheFrame, TruthFrame
from duckie_pomdp.evaluation.yolo_detection import intersection_over_union
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector, CameraCalibration
from duckie_pomdp.perception.measurement_noise import (
    PolarMeasurementNoiseModel,
    load_polar_measurement_noise,
)

# Repo root: src/duckie_pomdp/evaluation/f9d_association.py -> parents[3].
_DEFAULT_F9C_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "f9c_robust_belief_v1.toml"
)

# The default counterfactual pair for compare_selection_rules: the first
# entry is the ablation's identity covariance (lambda = 1, no inflation --
# what temporal_association_only actually scored against), the second is a
# marker resolved from the FROZEN config's own range_scale at call time (see
# compare_selection_rules), never a second hard-coded numeric literal.
_IDENTITY_RANGE_SCALE = 1.0

# Same nominal (unrandomized) camera constants
# experiments/evaluate_f9c_robust_belief.py's replay path uses -- see that
# module's "disclosed limitation" docstring section, and
# tests/test_f9c_robust_updater.py's observability_model() fixture, which
# uses the identical values. Duplicated deliberately rather than imported:
# this module must not depend on the experiments/ layer (only experiments/
# scripts import duckie_pomdp, never the reverse, everywhere else in this
# repo), and this constant is a fixed physical camera spec, not an F9c
# estimator parameter.
_NOMINAL_CAMERA_CALIBRATION = CameraCalibration(
    image_width_px=640,
    image_height_px=480,
    vertical_fov_deg=75.0,
    camera_height_m=0.108,
    camera_pitch_deg=19.15,
    camera_forward_offset_m=0.066,
)

_NO_ACTION = PolicyAction(0.0, 0.0)


# ---------------------------------------------------------------------------
# Frozen-config loading. Every value here is READ from
# configs/f9c_robust_belief_v1.toml via the same loaders F9c itself uses;
# nothing here is ever assigned a hand-picked numeric override.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReferenceConfig:
    ekf_config: PedestrianEKFConfig
    measurement_noise: PolarMeasurementNoiseModel
    existence_config: ExistenceFilterConfig
    bias_frozen: FrozenBiasCorrection
    bias_fitted: FrozenBiasCorrection
    observability_model: PredictedObservabilityModel
    robust_config: RobustObservationConfig
    matching_iou_threshold: float


def _read_matching_iou_threshold(path: Path) -> float:
    with path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    return float(data["calibration_protocol"]["matching_iou_threshold"])


def _load_reference_config(f9c_config_path: str | Path | None) -> _ReferenceConfig:
    path = (
        _DEFAULT_F9C_CONFIG_PATH if f9c_config_path is None else Path(f9c_config_path).resolve()
    )
    existence_config = replace(
        load_existence_filter_config(path),
        miss_likelihood_floor=load_miss_likelihood_floor(path),
    )
    observability_model = PredictedObservabilityModel(
        CalibratedGroundProjector(_NOMINAL_CAMERA_CALIBRATION),
        image_width_px=_NOMINAL_CAMERA_CALIBRATION.image_width_px,
    )
    return _ReferenceConfig(
        ekf_config=load_pedestrian_ekf_config(path),
        measurement_noise=load_polar_measurement_noise(path),
        existence_config=existence_config,
        bias_frozen=load_frozen_bias_correction(path, section="baseline_measurement_model"),
        bias_fitted=load_frozen_bias_correction(path, section="measurement_model"),
        observability_model=observability_model,
        robust_config=load_robust_observation_config(path),
        matching_iou_threshold=_read_matching_iou_threshold(path),
    )


def _bias_for_switches(reference: _ReferenceConfig) -> FrozenBiasCorrection:
    """Mirrors RobustPedestrianBeliefUpdater.update's own step 4b exactly."""

    return (
        reference.bias_fitted
        if reference.robust_config.switches.bias_refit
        else reference.bias_frozen
    )


# ---------------------------------------------------------------------------
# Cache-frame -> CandidateMeasurement marshaling. Mechanical data conversion
# only (no selection/NIS/EKF logic) -- mirrors
# experiments/evaluate_f9c_robust_belief.py's own
# _selection_from_cache_frame/_bbox_key exactly (same bbox_key rounding
# scheme, same raw_candidate_projection_failed filter) so bbox_key values
# from this module compare equal to the ones the real headline run recorded.
# ---------------------------------------------------------------------------


def _bbox_key(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    return (int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max)))


def _raw_candidates_from_cache_frame(
    cache_frame: RuntimeCacheFrame,
) -> tuple[CandidateMeasurement, ...]:
    candidates: list[CandidateMeasurement] = []
    for index in range(cache_frame.raw_candidate_count):
        if cache_frame.raw_candidate_projection_failed[index]:
            continue
        range_m = cache_frame.raw_candidate_range_m[index]
        bearing_rad = cache_frame.raw_candidate_bearing_rad[index]
        confidence = cache_frame.raw_candidate_confidence[index]
        measurement = ObjectMeasurement(
            object_class=ObjectClass.DUCKIE,
            detected=True,
            confidence=confidence,
            x_left_m=range_m * sin(bearing_rad),
            y_forward_m=range_m * cos(bearing_rad),
            range_m=range_m,
            bearing_rad=bearing_rad,
        )
        candidates.append(
            CandidateMeasurement(
                measurement=measurement,
                confidence=confidence,
                bbox_key=_bbox_key(cache_frame.raw_candidate_bbox[index]),
            )
        )
    return tuple(candidates)


def _bbox_by_key_from_cache_frame(
    cache_frame: RuntimeCacheFrame,
) -> dict[tuple[int, int, int, int], BoundingBox]:
    result: dict[tuple[int, int, int, int], BoundingBox] = {}
    for index in range(cache_frame.raw_candidate_count):
        if cache_frame.raw_candidate_projection_failed[index]:
            continue
        raw_bbox = cache_frame.raw_candidate_bbox[index]
        result[_bbox_key(raw_bbox)] = BoundingBox(*raw_bbox)
    return result


# ---------------------------------------------------------------------------
# The single reference replay. Runs the frozen Robust B configuration once
# over `cache`, driving the REAL trajectory forward with the REAL raw
# candidates on every frame, and snapshotting (H, P-, h(x-hat-)) at every
# frame where a track was already active BEFORE this frame's predict step --
# via a deepcopy of the coordinator's own EKF, so nothing computed from a
# snapshot can ever alter the trajectory the NEXT frame's snapshot is taken
# from.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameSnapshot:
    episode: str
    frame: int
    duplicate_selection: bool
    candidates: tuple[CandidateMeasurement, ...]  # bias-corrected
    bbox_by_key: dict[tuple[int, int, int, int], BoundingBox]
    predicted_measurement: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    predicted_covariance: NDArray[np.float64]
    gt_box: BoundingBox | None
    gt_eligible: bool


def _reference_snapshots(
    cache: Sequence[RuntimeCacheFrame],
    truth: Mapping[tuple[str, int], TruthFrame],
    *,
    f9c_config_path: str | Path | None = None,
) -> tuple[list[_FrameSnapshot], _ReferenceConfig]:
    reference = _load_reference_config(f9c_config_path)
    bias = _bias_for_switches(reference)

    frames_by_episode: dict[str, list[RuntimeCacheFrame]] = {}
    for cache_frame in cache:
        frames_by_episode.setdefault(cache_frame.episode, []).append(cache_frame)

    snapshots: list[_FrameSnapshot] = []
    for episode in sorted(frames_by_episode):
        episode_frames = sorted(frames_by_episode[episode], key=lambda item: item.frame)

        updater = RobustPedestrianBeliefUpdater(
            ekf_config=reference.ekf_config,
            measurement_noise=reference.measurement_noise,
            existence_config=reference.existence_config,
            bias_frozen=reference.bias_frozen,
            bias_fitted=reference.bias_fitted,
            observability_model=reference.observability_model,
            config=reference.robust_config,
        )
        placeholder_ego = EgoObservation(
            0.0, 0.0, episode_frames[0].ego_linear_velocity_mps, episode_frames[0].ego_yaw_rate_rad_s
        )
        belief = initial_belief(
            placeholder_ego, existence_prior=reference.existence_config.prior_probability
        )

        for cache_frame in episode_frames:
            key = (episode, cache_frame.frame)
            truth_frame = truth.get(key)
            if truth_frame is None:
                raise RuntimeError(f"{episode} frame {cache_frame.frame} missing from evaluation truth")

            ego_motion = EgoMotion(cache_frame.ego_linear_velocity_mps, cache_frame.ego_yaw_rate_rad_s)
            raw_candidates = _raw_candidates_from_cache_frame(cache_frame)

            if updater.ekf.initialized and raw_candidates:
                # Snapshot on a DEEPCOPY of the coordinator's own EKF -- the
                # live updater.ekf is never touched here, so the trajectory
                # that produces every later frame's snapshot is always the
                # real frozen Robust B trajectory, never a counterfactual one.
                ekf_snapshot = copy.deepcopy(updater.ekf)
                ekf_snapshot.predict(ego_motion, cache_frame.dt_s)
                predicted_state = ekf_snapshot.state
                predicted_measurement = measurement_function(
                    predicted_state, minimum_range_m=reference.ekf_config.minimum_range_m
                )
                jacobian = measurement_jacobian(
                    predicted_state, minimum_range_m=reference.ekf_config.minimum_range_m
                )
                predicted_covariance = ekf_snapshot.covariance
                corrected_candidates = tuple(
                    CandidateMeasurement(
                        measurement=bias.correct(candidate.measurement),
                        confidence=candidate.confidence,
                        bbox_key=candidate.bbox_key,
                    )
                    for candidate in raw_candidates
                )
                gt_box = None if truth_frame.gt_bbox is None else BoundingBox(*truth_frame.gt_bbox)
                snapshots.append(
                    _FrameSnapshot(
                        episode=episode,
                        frame=cache_frame.frame,
                        duplicate_selection=len(corrected_candidates) > 1,
                        candidates=corrected_candidates,
                        bbox_by_key=_bbox_by_key_from_cache_frame(cache_frame),
                        predicted_measurement=predicted_measurement,
                        jacobian=jacobian,
                        predicted_covariance=predicted_covariance,
                        gt_box=gt_box,
                        gt_eligible=truth_frame.eligible_visible,
                    )
                )

            # Advance the REAL trajectory. Nothing computed above is ever fed
            # back in here -- `raw_candidates`, not `corrected_candidates`,
            # matching production (update() applies the bias stage itself).
            belief, _record = updater.update(
                belief, _NO_ACTION, ego_motion, list(raw_candidates), cache_frame.dt_s
            )

    return snapshots, reference


def _innovation_covariance_provider(
    snapshot: _FrameSnapshot,
    base_noise: PolarMeasurementNoiseModel,
    *,
    range_scale: float,
    bearing_scale: float,
) -> Callable[[float], NDArray[np.float64]]:
    """The invariant-I1 formula ``S = H P- H^T + lambda R``, built from a
    FROZEN snapshot's own (H, P-) -- never RobustPedestrianBeliefUpdater's
    private ``_innovation_covariance`` (which reads the coordinator's LIVE
    ekf.state/covariance, unusable here without disturbing the real
    trajectory). ``CovarianceCalibration.inflate`` is the one frozen function
    that applies range_scale/bearing_scale; nothing here re-derives it."""

    calibration = CovarianceCalibration(range_scale, bearing_scale, 0.0, 0.0)

    def provider(range_m: float) -> NDArray[np.float64]:
        return (
            snapshot.jacobian @ snapshot.predicted_covariance @ snapshot.jacobian.T
            + calibration.inflate(base_noise.covariance(range_m))
        )

    return provider


def _unreachable_provider(range_m: float) -> NDArray[np.float64]:
    """Passed where predicted_measurement=None so mode="initialization" is
    guaranteed and the provider is never actually called -- see
    MeasurementAssociator.associate. Raises loudly if that guarantee is ever
    violated rather than silently returning a wrong covariance."""

    raise AssertionError(
        "innovation_covariance_for was called with predicted_measurement=None; "
        "MeasurementAssociator.associate should have short-circuited to the "
        "initialization branch before ever calling this"
    )


# ---------------------------------------------------------------------------
# Per-frame record building and aggregation -- shared by both comparisons.
# ---------------------------------------------------------------------------


def _selected_box(
    snapshot: _FrameSnapshot, result: AssociationResult
) -> tuple[tuple[int, int, int, int] | None, BoundingBox | None]:
    if result.selected is None:
        return None, None
    key = result.selected.bbox_key
    return key, snapshot.bbox_by_key.get(key)


def _selected_nis(result: AssociationResult) -> float | None:
    if result.selected_index is None:
        return None
    nis = result.candidate_nis[result.selected_index]
    return None if nis is None else float(nis)


def _gate_exceedance(
    result: AssociationResult, *, association_gate_threshold: float
) -> tuple[int, int]:
    """(candidates whose NIS exceeds the association chi-square gate, total
    scored candidates) for one AssociationResult. ``candidate_nis`` entries
    are None only in "initialization"/"no_candidate" mode (no predicted
    state to score against), which never occurs for the C1/C2 callers here
    since both always pass a real ``predicted_measurement`` for at least one
    of the two rules being compared -- included defensively regardless."""

    scored = [nis for nis in result.candidate_nis if nis is not None]
    over_gate = sum(1 for nis in scored if nis > association_gate_threshold)
    return over_gate, len(scored)


def _frame_record(
    snapshot: _FrameSnapshot,
    result_a: AssociationResult,
    result_b: AssociationResult,
    *,
    matching_iou_threshold: float,
    association_gate_threshold: float,
) -> dict[str, Any]:
    key_a, box_a = _selected_box(snapshot, result_a)
    key_b, box_b = _selected_box(snapshot, result_b)
    gt_available = snapshot.gt_eligible and snapshot.gt_box is not None

    iou_a = (
        None
        if not gt_available or box_a is None
        else float(intersection_over_union(snapshot.gt_box, box_a))
    )
    iou_b = (
        None
        if not gt_available or box_b is None
        else float(intersection_over_union(snapshot.gt_box, box_b))
    )

    over_gate_a, total_a = _gate_exceedance(
        result_a, association_gate_threshold=association_gate_threshold
    )
    over_gate_b, total_b = _gate_exceedance(
        result_b, association_gate_threshold=association_gate_threshold
    )

    return {
        "episode": snapshot.episode,
        "frame": int(snapshot.frame),
        "duplicate_selection": bool(snapshot.duplicate_selection),
        "gt_available": bool(gt_available),
        "selected_key_a": key_a,
        "selected_key_b": key_b,
        "agree": key_a == key_b,
        "mode_a": result_a.mode,
        "mode_b": result_b.mode,
        "abstained_a": result_a.mode == "all_gated_out",
        "abstained_b": result_b.mode == "all_gated_out",
        "iou_a": iou_a,
        "iou_b": iou_b,
        "outlier_a": None if iou_a is None else iou_a < matching_iou_threshold,
        "outlier_b": None if iou_b is None else iou_b < matching_iou_threshold,
        "nis_a": _selected_nis(result_a),
        "nis_b": _selected_nis(result_b),
        "candidates_over_gate_a": over_gate_a,
        "candidates_total_a": total_a,
        "candidates_over_gate_b": over_gate_b,
        "candidates_total_b": total_b,
    }


def _aggregate(
    records: Sequence[Mapping[str, Any]], *, rule_a_name: str, rule_b_name: str
) -> dict[str, Any]:
    frames_compared = len(records)
    agree = sum(1 for record in records if record["agree"])
    differ = frames_compared - agree
    differing = [record for record in records if not record["agree"]]

    paired = [record for record in differing if record["iou_a"] is not None and record["iou_b"] is not None]
    rule_a_higher = sum(1 for record in paired if record["iou_a"] > record["iou_b"])
    rule_b_higher = sum(1 for record in paired if record["iou_b"] > record["iou_a"])
    tie = sum(1 for record in paired if record["iou_a"] == record["iou_b"])

    one_sided = [
        record
        for record in differing
        if (record["iou_a"] is None) != (record["iou_b"] is None)
    ]
    neither_has_gt = [
        record for record in differing if record["iou_a"] is None and record["iou_b"] is None
    ]

    iou_a_values = [record["iou_a"] for record in records if record["iou_a"] is not None]
    iou_b_values = [record["iou_b"] for record in records if record["iou_b"] is not None]
    outlier_a = sum(1 for record in records if record["outlier_a"])
    outlier_b = sum(1 for record in records if record["outlier_b"])

    return {
        "rule_a": rule_a_name,
        "rule_b": rule_b_name,
        "frames_compared": frames_compared,
        "selections_agree": agree,
        "selections_differ": differ,
        "selections_agree_fraction": (agree / frames_compared) if frames_compared else None,
        "differing_frames_paired": {
            "rule_a_higher_iou": rule_a_higher,
            "rule_b_higher_iou": rule_b_higher,
            "tie": tie,
        },
        "differing_frames_excluded_from_pairing": {
            "one_side_made_no_selection": len(one_sided),
            "neither_side_has_gt": len(neither_has_gt),
        },
        "localization_outlier_count": {"rule_a": outlier_a, "rule_b": outlier_b},
        "mean_selected_iou": {
            "rule_a": float(np.mean(iou_a_values)) if iou_a_values else None,
            "rule_b": float(np.mean(iou_b_values)) if iou_b_values else None,
        },
        "median_selected_iou": {
            "rule_a": float(np.median(iou_a_values)) if iou_a_values else None,
            "rule_b": float(np.median(iou_b_values)) if iou_b_values else None,
        },
        "gt_available_frame_count": sum(1 for record in records if record["gt_available"]),
        "frames": [dict(record) for record in records],
    }


def _c1_abstention(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """C1-abstention: fix round 1. The paired ``differing_frames_paired``
    comparison in ``_aggregate`` EXCLUDES every frame where one rule
    selected nothing at all (``differing_frames_excluded_from_pairing
    .one_side_made_no_selection``) -- on this cache that exclusion bucket
    (19 of 23 differing frames) is most of the disagreement between
    ``lambda = 1`` and the frozen ``lambda_r``, not a small remainder. This
    function makes abstention itself a headline measure instead of an
    exclusion: how often each rule's own chi-square gate rejects every
    candidate outright (mode "all_gated_out"), and whether that traces to a
    correspondingly higher fraction of candidates exceeding the association
    gate (``chi_square_gate = 13.815510557964274``) under the tighter,
    uninflated ``S`` at ``lambda = 1``.
    """

    abstained_a = sum(1 for record in records if record["abstained_a"])
    abstained_b = sum(1 for record in records if record["abstained_b"])
    a_abstained_b_selected = [
        record for record in records if record["abstained_a"] and not record["abstained_b"]
    ]
    b_abstained_a_selected = [
        record for record in records if record["abstained_b"] and not record["abstained_a"]
    ]
    one_sided_abstention = a_abstained_b_selected + b_abstained_a_selected
    one_sided_abstention_with_gt = sum(
        1 for record in one_sided_abstention if record["gt_available"]
    )

    nis_a_values = [record["nis_a"] for record in records if record["nis_a"] is not None]
    nis_b_values = [record["nis_b"] for record in records if record["nis_b"] is not None]

    over_gate_a_total = sum(record["candidates_over_gate_a"] for record in records)
    total_a = sum(record["candidates_total_a"] for record in records)
    over_gate_b_total = sum(record["candidates_over_gate_b"] for record in records)
    total_b = sum(record["candidates_total_b"] for record in records)

    return {
        "frames_rule_a_selected_nothing": abstained_a,
        "frames_rule_b_selected_nothing": abstained_b,
        "rule_a_abstained_rule_b_selected": len(a_abstained_b_selected),
        "rule_b_abstained_rule_a_selected": len(b_abstained_a_selected),
        "one_sided_abstention_frame_count": len(one_sided_abstention),
        "one_sided_abstention_frames_with_gt_available": one_sided_abstention_with_gt,
        "mean_selected_nis": {
            "rule_a": float(np.mean(nis_a_values)) if nis_a_values else None,
            "rule_b": float(np.mean(nis_b_values)) if nis_b_values else None,
        },
        "candidate_gate_exceedance_fraction": {
            "rule_a": (over_gate_a_total / total_a) if total_a else None,
            "rule_b": (over_gate_b_total / total_b) if total_b else None,
        },
        "candidate_gate_exceedance_counts": {
            "rule_a": {"over_gate": over_gate_a_total, "total": total_a},
            "rule_b": {"over_gate": over_gate_b_total, "total": total_b},
        },
    }


def _c1_abstention_conclusion(abstention: Mapping[str, Any]) -> dict[str, Any]:
    """SUPPORTED iff lambda = 1 abstains (rejects every candidate) more
    often than the frozen lambda, AND the candidate-level gate-exceedance
    fraction moves in the SAME direction (lambda = 1 higher) -- the second
    condition is the direct mechanism check: a higher abstention count that
    was NOT accompanied by a higher gate-exceedance fraction would not
    actually support "the tighter S causes more rejection" as the
    explanation."""

    frames_a = abstention["frames_rule_a_selected_nothing"]
    frames_b = abstention["frames_rule_b_selected_nothing"]
    gate_a = abstention["candidate_gate_exceedance_fraction"]["rule_a"]
    gate_b = abstention["candidate_gate_exceedance_fraction"]["rule_b"]

    lambda_one_abstains_more_often = frames_a > frames_b
    gate_exceedance_moves_same_direction = (
        gate_a is not None and gate_b is not None and gate_a > gate_b
    )
    supported = lambda_one_abstains_more_often and gate_exceedance_moves_same_direction

    return {
        "question": (
            "does lambda=1 abstain (reject every candidate) materially more "
            "often than the frozen lambda, via a correspondingly higher "
            "association gate-exceedance fraction?"
        ),
        "verdict": "SUPPORTED" if supported else "UNSUPPORTED",
        "lambda_one_abstains_more_often": lambda_one_abstains_more_often,
        "gate_exceedance_moves_same_direction": gate_exceedance_moves_same_direction,
        "frames_rule_a_selected_nothing": frames_a,
        "frames_rule_b_selected_nothing": frames_b,
        "gate_exceedance_fraction_rule_a": gate_a,
        "gate_exceedance_fraction_rule_b": gate_b,
    }


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def compare_selection_rules(
    cache: Sequence[RuntimeCacheFrame],
    truth: Mapping[tuple[str, int], TruthFrame],
    *,
    lambda_scales: tuple[float, float] | None = None,
    f9c_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """C1: at every frame with a predicted state fixed by ONE reference
    replay (see module docstring), does minimum-NIS association select a
    different candidate under ``lambda = 1`` (rule A -- what
    ``temporal_association_only`` actually scored against, since
    ``covariance_calibration`` was off) than under the frozen ``lambda_r``
    (rule B -- read from the loaded config, never a second literal, unless
    ``lambda_scales`` overrides it)?

    ``lambda_scales`` is ``(range_scale_a, range_scale_b)``; bearing_scale is
    always the frozen config's own value (1.0) for both rules -- the
    ablation/calibration story is specifically about the RANGE scale.
    """

    snapshots, reference = _reference_snapshots(cache, truth, f9c_config_path=f9c_config_path)
    reference_bearing_scale = reference.robust_config.covariance.bearing_scale
    range_scale_a, range_scale_b = (
        (_IDENTITY_RANGE_SCALE, reference.robust_config.covariance.range_scale)
        if lambda_scales is None
        else lambda_scales
    )

    associator = MeasurementAssociator(reference.robust_config.association)
    records = []
    for snapshot in snapshots:
        result_a = associator.associate(
            snapshot.candidates,
            predicted_measurement=snapshot.predicted_measurement,
            innovation_covariance_for=_innovation_covariance_provider(
                snapshot,
                reference.measurement_noise,
                range_scale=range_scale_a,
                bearing_scale=reference_bearing_scale,
            ),
        )
        result_b = associator.associate(
            snapshot.candidates,
            predicted_measurement=snapshot.predicted_measurement,
            innovation_covariance_for=_innovation_covariance_provider(
                snapshot,
                reference.measurement_noise,
                range_scale=range_scale_b,
                bearing_scale=reference_bearing_scale,
            ),
        )
        records.append(
            _frame_record(
                snapshot,
                result_a,
                result_b,
                matching_iou_threshold=reference.matching_iou_threshold,
                association_gate_threshold=reference.robust_config.association.chi_square_gate,
            )
        )

    result = _aggregate(
        records,
        rule_a_name=f"lambda_range_scale={range_scale_a}",
        rule_b_name=f"lambda_range_scale={range_scale_b}",
    )
    result["lambda_range_scale_a"] = float(range_scale_a)
    result["lambda_range_scale_b"] = float(range_scale_b)
    result["bearing_scale"] = float(reference_bearing_scale)
    result["matching_iou_threshold"] = float(reference.matching_iou_threshold)
    result["association_gate_threshold"] = float(reference.robust_config.association.chi_square_gate)

    # Fix round 1: abstention is a headline C1 measure, not folded into the
    # paired selection-quality comparison's exclusion bucket -- see
    # _c1_abstention's own docstring for why.
    abstention = _c1_abstention(records)
    result["abstention"] = abstention
    result["conclusion_selection"] = _c1_conclusion(result)
    result["conclusion_abstention"] = _c1_abstention_conclusion(abstention)
    result["conclusion"] = {
        "selection": result["conclusion_selection"]["verdict"],
        "abstention": result["conclusion_abstention"]["verdict"],
    }
    # Resolution note (fix round 1): see duplicate_frame_ranking's identical
    # field for the interpretation -- a true selection-quality difference
    # smaller than this fraction is not detectable on this cache.
    result["minimum_detectable_selection_difference_fraction"] = (
        1.0 - result["selections_agree_fraction"]
        if result["selections_agree_fraction"] is not None
        else None
    )
    return result


def duplicate_frame_ranking(
    cache: Sequence[RuntimeCacheFrame],
    truth: Mapping[tuple[str, int], TruthFrame],
    *,
    f9c_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """C2: on duplicate frames only (candidate_count > 1, predicted state
    available), does minimum-NIS association at the frozen lambda_r (rule B)
    pick a worse candidate than highest-confidence-with-lexicographic-
    tie-break (rule A)?

    Rule A is produced by the REAL ``MeasurementAssociator.associate`` with
    ``predicted_measurement=None`` -- its own "initialization" branch is
    exactly the frozen highest-confidence-then-bbox-lexicographic rule, so
    no highest-confidence selection is reimplemented here.
    """

    snapshots, reference = _reference_snapshots(cache, truth, f9c_config_path=f9c_config_path)
    duplicate_snapshots = [snapshot for snapshot in snapshots if snapshot.duplicate_selection]

    associator = MeasurementAssociator(reference.robust_config.association)
    covariance = reference.robust_config.covariance
    records = []
    for snapshot in duplicate_snapshots:
        result_confidence = associator.associate(
            snapshot.candidates,
            predicted_measurement=None,
            innovation_covariance_for=_unreachable_provider,
        )
        result_min_nis = associator.associate(
            snapshot.candidates,
            predicted_measurement=snapshot.predicted_measurement,
            innovation_covariance_for=_innovation_covariance_provider(
                snapshot,
                reference.measurement_noise,
                range_scale=covariance.range_scale,
                bearing_scale=covariance.bearing_scale,
            ),
        )
        records.append(
            _frame_record(
                snapshot,
                result_confidence,
                result_min_nis,
                matching_iou_threshold=reference.matching_iou_threshold,
                association_gate_threshold=reference.robust_config.association.chi_square_gate,
            )
        )

    result = _aggregate(
        records, rule_a_name="highest_confidence", rule_b_name="min_nis_at_frozen_lambda"
    )
    result["frozen_range_scale"] = float(covariance.range_scale)
    result["frozen_bearing_scale"] = float(covariance.bearing_scale)
    result["matching_iou_threshold"] = float(reference.matching_iou_threshold)
    result["conclusion"] = _c2_conclusion(result)
    # Resolution note (fix round 1): with the two rules agreeing on this
    # many of the frames_compared, a true selection-quality difference
    # smaller than roughly (1 - selections_agree_fraction) is not
    # detectable on this cache regardless of which rule is actually better.
    result["minimum_detectable_selection_difference_fraction"] = (
        1.0 - result["selections_agree_fraction"]
        if result["selections_agree_fraction"] is not None
        else None
    )
    return result


# ---------------------------------------------------------------------------
# Conclusion rules -- fixed BEFORE the counts are seen (task-2-brief.md).
# Pure functions of the aggregate dict; no branch here may be edited to
# "soften" an outcome the counts do not support.
# ---------------------------------------------------------------------------


def _c1_conclusion(result: Mapping[str, Any]) -> dict[str, Any]:
    outliers_a = result["localization_outlier_count"]["rule_a"]
    outliers_b = result["localization_outlier_count"]["rule_b"]
    wins_a = result["differing_frames_paired"]["rule_a_higher_iou"]
    wins_b = result["differing_frames_paired"]["rule_b_higher_iou"]
    fewer_outliers_at_frozen_lambda = outliers_b < outliers_a
    more_wins_at_frozen_lambda = wins_b > wins_a
    supported = fewer_outliers_at_frozen_lambda and more_wins_at_frozen_lambda
    return {
        "question": "does the wrong S explain the ablation penalty?",
        "verdict": "SUPPORTED" if supported else "REFUTED",
        "fewer_outliers_at_frozen_lambda": fewer_outliers_at_frozen_lambda,
        "more_wins_at_frozen_lambda": more_wins_at_frozen_lambda,
        "outliers_rule_a": outliers_a,
        "outliers_rule_b": outliers_b,
        "wins_rule_a": wins_a,
        "wins_rule_b": wins_b,
    }


def _c2_conclusion(result: Mapping[str, Any]) -> dict[str, Any]:
    outliers_confidence = result["localization_outlier_count"]["rule_a"]
    outliers_min_nis = result["localization_outlier_count"]["rule_b"]
    wins_confidence = result["differing_frames_paired"]["rule_a_higher_iou"]
    wins_min_nis = result["differing_frames_paired"]["rule_b_higher_iou"]
    more_outliers = outliers_min_nis > outliers_confidence
    fewer_wins = wins_min_nis < wins_confidence
    inferior = more_outliers and fewer_wins
    return {
        "question": "is min-NIS a worse selection rule than confidence, at the correct lambda?",
        "verdict": "INFERIOR" if inferior else "NOT INFERIOR",
        "more_outliers_at_frozen_lambda": more_outliers,
        "fewer_wins_at_frozen_lambda": fewer_wins,
        "outliers_highest_confidence": outliers_confidence,
        "outliers_min_nis": outliers_min_nis,
        "wins_highest_confidence": wins_confidence,
        "wins_min_nis": wins_min_nis,
    }
