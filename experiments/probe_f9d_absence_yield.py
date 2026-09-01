"""Gate F9d Task 4: the absence-yield probe on development seeds only.

F9c's longest natural in-domain miss run was 10 frames -- far short of the
20+ this gate needs to test whether existence belief actually decays under
prolonged absence, and whether F9c's own miss-likelihood floor (added to
SLOW collapse) leaves a track alive on no evidence forever. This script
renders ONLY the eight development seeds (8101-8108) against the
``[[absence_scenario_matrix]]`` scenarios in
``configs/f9d_evidence_closure_v1.toml``, applies the two controlled
interventions (B2 detector dropout, B3 target removal) at exactly the
boundaries their own modules define, counts absence runs
(``duckie_pomdp.evaluation.f9d_absence_stress.absence_yield``), and projects
the observed per-seed rate onto the 4 final absence seeds (8301-8304).

**Three absence kinds, kept apart everywhere in this script**, per each
scenario's own ``kind`` field (never inferred from behaviour):

- B1 (genuine out-of-domain absence): no intervention. The scenario simply
  drives the ego straight past a stationary pedestrian; once the ego
  passes it, the pedestrian falls out of the camera's forward domain
  naturally.
- B2 (controlled in-domain detector dropout): the base detector always
  runs against the real rendered frame; a ``DetectorDropout``-derived
  schedule (fixed from the seed alone, before the episode runs) discards
  only its Duckie detections for a scheduled window, applied through
  ``DropoutAwareDetector`` wrapping the SAME detector instance every other
  scenario in this run uses.
- B3 (controlled target disappearance, optional/bonus): a
  ``TargetRemovalScheduler``-derived single switch frame (same seed-only
  rule) at which ``integration.projection_validation
  .remove_scenario_pedestrian()`` is called once; the pedestrian never
  returns for the rest of the episode.

**Reused, not re-derived.** Exactly like Task 3's ``probe_f9d_yield.py``,
this script reuses ``evaluate_f9c_robust_belief``'s ``_scenario_for``,
``_selection_from_render_result``, ``_step_both_systems``, ``build_row``,
``_distance_bin``, and ``_fov_region`` rather than re-implementing any of
that logic, so the estimator step this probe measures against cannot
silently diverge from what a later final-evaluation render will compute
for the exact same kind of frame.

**No estimator parameter is read, written, or overridden here.** Every
EKF/existence/bias/robust-observation object is loaded fresh from the
FROZEN F9c config, same as Task 3.

**Unlike Task 3's outlier definition, this probe's per-frame absence
predicate never depends on the detector or the estimator's own belief** --
B1/B3 depend only on privileged ground truth, B2 only on the dropout
schedule itself (see ``f9d_absence_stress.py``'s module docstring for why:
using the estimator's own predicted-observability classification would
mislabel every episode's pre-initialization warm-up frames as B1 absence).
The full estimator still runs on every frame so the artifact also carries
each row's ``robust_b_existence_probability`` trajectory -- useful
diagnostic context for whether belief actually decays, which is a
different, later question this script does not itself judge.

Development seeds are reusable: an early simulator termination is a
recorded warning, not a fatal error, exactly as Task 3's probe treats it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import evaluate_f9c_robust_belief as f9c_eval  # noqa: E402

from duckie_pomdp.adapters.gym_duckietown import (  # noqa: E402
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.belief.observability import PredictedObservabilityModel  # noqa: E402
from duckie_pomdp.belief.pedestrian_ekf import (  # noqa: E402
    MeasurementProfile,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.belief.robust_updater import RobustPedestrianBeliefUpdater  # noqa: E402
from duckie_pomdp.belief.updater import (  # noqa: E402
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.dataset.annotations import assess_silhouette  # noqa: E402
from duckie_pomdp.dataset.config import load_dataset_config  # noqa: E402
from duckie_pomdp.domain.action import PolicyAction  # noqa: E402
from duckie_pomdp.domain.detection import ObjectClass, yolo_class_id  # noqa: E402
from duckie_pomdp.evaluation.f9_protocol import F9ScenarioSpec  # noqa: E402
from duckie_pomdp.evaluation.f9c_calibration import (  # noqa: E402
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.evaluation.f9c_protocol import F9cProtocol  # noqa: E402
from duckie_pomdp.evaluation.f9c_runtime_cache import TruthFrame  # noqa: E402
from duckie_pomdp.evaluation.f9d_absence_stress import absence_yield  # noqa: E402
from duckie_pomdp.evaluation.f9d_protocol import (  # noqa: E402
    F9dProtocol,
    absence_support_satisfied,
    f9c_parameters,
    load_f9d_protocol,
)
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector  # noqa: E402
from duckie_pomdp.perception.detector_dropout import (  # noqa: E402
    DetectorDropout,
    DropoutAwareDetector,
    TargetRemovalScheduler,
)
from duckie_pomdp.perception.f9_pipeline import (  # noqa: E402
    AdditiveMeasurementBias,
    YoloPedestrianMeasurementPipeline,
)
from duckie_pomdp.perception.measurement_calibration import (  # noqa: E402
    LinearRangeCalibration,
    MeasurementCalibrator,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise  # noqa: E402
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector  # noqa: E402
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector  # noqa: E402
from duckie_pomdp.scenario import PedestrianMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NUMBER_OF_FINAL_ABSENCE_SEEDS = 4  # len(protocol.absence_final_seeds); asserted at runtime


# ---------------------------------------------------------------------------
# Scenario matrix loading -- mirrors probe_f9d_yield.py's
# _load_outlier_scenarios, extended with each row's absence `kind` and the
# kind-specific intervention parameters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsenceScenarioSpec:
    base: F9ScenarioSpec
    kind: str  # "B1" | "B2" | "B3"
    dropout_window_frames: int | None = None
    dropout_warmup_frames: int | None = None
    removal_warmup_frames: int | None = None
    removal_tail_frames: int | None = None


def _load_absence_scenarios(f9d_config_path: Path) -> list[AbsenceScenarioSpec]:
    with f9d_config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    items = data.get("absence_scenario_matrix", [])
    if not items:
        raise RuntimeError(f"{f9d_config_path} has no [[absence_scenario_matrix]] entries")

    specs: list[AbsenceScenarioSpec] = []
    for item in items:
        steps = int(item["steps"])
        if steps <= 0:
            raise ValueError("absence scenario steps must be positive")
        kind = str(item["kind"])
        if kind not in ("B1", "B2", "B3"):
            raise ValueError(f"unknown absence scenario kind: {kind!r}")
        base = F9ScenarioSpec(
            name=str(item["name"]),
            pedestrian_mode=PedestrianMode(str(item["pedestrian_mode"])),
            action=PolicyAction(
                float(item["linear_velocity_mps"]),
                float(item["angular_velocity_rad_s"]),
            ),
            steps=steps,
            ego_start_x_offset_m=float(item.get("ego_start_x_offset_m", 0.0)),
            ego_heading_offset_rad=float(item.get("ego_heading_offset_rad", 0.0)),
            use_for_calibration=False,
            use_for_final_evaluation=bool(item.get("use_for_final", False)),
        )
        specs.append(
            AbsenceScenarioSpec(
                base=base,
                kind=kind,
                dropout_window_frames=(
                    int(item["dropout_window_frames"]) if kind == "B2" else None
                ),
                dropout_warmup_frames=(
                    int(item["dropout_warmup_frames"]) if kind == "B2" else None
                ),
                removal_warmup_frames=(
                    int(item.get("removal_warmup_frames", 0)) if kind == "B3" else None
                ),
                removal_tail_frames=(
                    int(item.get("removal_tail_frames", 0)) if kind == "B3" else None
                ),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Plumbing settings -- identical shape to probe_f9d_yield.py's
# _ProbeSettings, duplicated for the same documented reason (importing a
# private class from a sibling script is more surprising than the small
# repetition).
# ---------------------------------------------------------------------------


class _ProbeSettings:
    def __init__(self, f9c_config_path: Path) -> None:
        with f9c_config_path.open("rb") as stream:
            data: dict[str, Any] = tomllib.load(stream)

        def relative(value: str) -> Path:
            return (f9c_config_path.parent / value).resolve()

        provenance = data["provenance"]
        detector = data["detector"]
        simulator = data["simulator"]
        perturbation = data["trajectory_perturbation"]
        calibration_protocol = data["calibration_protocol"]

        self.scenario_path = relative(str(provenance["scenario"]))
        self.annotation_config_path = relative(str(provenance["annotation_rules"]))
        self.detector_confidence_threshold = float(detector["confidence_threshold"])
        self.detector_nms_iou_threshold = float(detector["nms_iou_threshold"])
        self.detector_image_size = int(detector["image_size"])
        self.detector_device = detector["device"]
        self.detector_max_detections = int(detector["max_detections"])
        self.image_width_px = int(simulator["image_width_px"])
        self.image_height_px = int(simulator["image_height_px"])
        self.domain_randomization = bool(simulator["domain_randomization"])
        self.dynamics_randomization = bool(simulator["dynamics_randomization"])
        self.start_x_range_m = _bounds(perturbation["start_x_range_m"])
        self.lateral_offset_range_m = _bounds(perturbation["lateral_offset_range_m"])
        self.heading_range_rad = _bounds(perturbation["heading_range_rad"])
        self.matching_iou_threshold = float(calibration_protocol["matching_iou_threshold"])
        self.near_max_m = float(calibration_protocol["distance_bin_near_max_m"])
        self.medium_max_m = float(calibration_protocol["distance_bin_medium_max_m"])


def _bounds(values: list[float]) -> tuple[float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError("bounds must be an ordered pair")
    return bounds


# ---------------------------------------------------------------------------
# The render loop.
# ---------------------------------------------------------------------------


def collect_absence_rows(
    f9c_protocol: F9cProtocol,
    settings: _ProbeSettings,
    seeds: tuple[int, ...],
    specs: list[AbsenceScenarioSpec],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    annotation = load_dataset_config(settings.annotation_config_path)
    base_detector = YoloObjectDetector(
        f9c_protocol.checkpoint_path,
        confidence_threshold=settings.detector_confidence_threshold,
        iou_threshold=settings.detector_nms_iou_threshold,
        image_size=settings.detector_image_size,
        device=settings.detector_device,
        max_detections=settings.detector_max_detections,
    )
    ekf_config = load_pedestrian_ekf_config(f9c_protocol.config_path)
    measurement_noise = load_polar_measurement_noise(f9c_protocol.config_path)

    baseline_existence_config = load_existence_filter_config(f9c_protocol.config_path)
    robust_existence_config = replace(
        baseline_existence_config,
        miss_likelihood_floor=load_miss_likelihood_floor(f9c_protocol.config_path),
    )

    with f9c_protocol.config_path.open("rb") as stream:
        raw_config = tomllib.load(stream)
    baseline_bias = AdditiveMeasurementBias(
        float(raw_config["baseline_measurement_model"]["range_bias_m"]),
        float(raw_config["baseline_measurement_model"]["bearing_bias_rad"]),
    )

    robust_config = load_robust_observation_config(f9c_protocol.config_path)
    robust_bias_frozen = load_frozen_bias_correction(
        f9c_protocol.config_path, section="baseline_measurement_model"
    )
    robust_bias_fitted = load_frozen_bias_correction(
        f9c_protocol.config_path, section="measurement_model"
    )

    rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for seed in seeds:
        for scenario_index, spec in enumerate(specs):
            base = spec.base
            scenario = f9c_eval._scenario_for(settings, base, seed, scenario_index)
            episode = f"absence_probe_{seed}_{base.name}"
            episode_length = base.steps + 1

            dropout_wrapper: DropoutAwareDetector | None = None
            if spec.kind == "B2":
                dropout = DetectorDropout(
                    window_length=spec.dropout_window_frames,
                    warmup_frames=spec.dropout_warmup_frames,
                )
                schedule = dropout.schedule_for(seed=seed, episode_length=episode_length)
                dropout_wrapper = DropoutAwareDetector(base_detector, schedule)
                active_detector = dropout_wrapper
            else:
                active_detector = base_detector

            removal_schedule = None
            if spec.kind == "B3":
                scheduler = TargetRemovalScheduler(
                    warmup_frames=spec.removal_warmup_frames,
                    tail_frames=spec.removal_tail_frames,
                )
                removal_schedule = scheduler.schedule_for(seed=seed, episode_length=episode_length)

            integration = create_gym_duckietown(
                GymDuckietownConfig(
                    scenario=scenario,
                    domain_randomization=settings.domain_randomization,
                    dynamics_randomization=settings.dynamics_randomization,
                    maximum_steps=base.steps + 2,
                    camera_width=settings.image_width_px,
                    camera_height=settings.image_height_px,
                )
            )
            try:
                observation = integration.agent.reset(seed=seed)
                camera_calibration = integration.camera_calibration.read()
                projector = YoloMeasurementProjector(
                    CalibratedGroundProjector(camera_calibration),
                    MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
                )
                runtime_pipeline = YoloPedestrianMeasurementPipeline(active_detector, projector)
                observability_model = PredictedObservabilityModel(
                    CalibratedGroundProjector(camera_calibration),
                    image_width_px=settings.image_width_px,
                )

                baseline_updater = PedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    existence_config=baseline_existence_config,
                    measurement_noise=measurement_noise,
                    measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
                )
                robust_updater = RobustPedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    measurement_noise=measurement_noise,
                    existence_config=robust_existence_config,
                    bias_frozen=robust_bias_frozen,
                    bias_fitted=robust_bias_fitted,
                    observability_model=observability_model,
                    config=robust_config,
                )

                baseline_belief = initial_belief(
                    observation.ego, existence_prior=baseline_existence_config.prior_probability
                )
                robust_belief = initial_belief(
                    observation.ego, existence_prior=robust_existence_config.prior_probability
                )
                timestamp_s = 0.0
                dt_s = 1.0 / 30.0
                previous_action = PolicyAction(0.0, 0.0)
                removed_already = False

                # B3 boundary fix (Task 4 fix round 1). Removal must happen
                # BEFORE the render that produces switch_frame's image, or
                # privileged truth (which flips the instant
                # remove_scenario_pedestrian() is called) and the rendered
                # image (which only reflects removal starting the NEXT
                # render) would disagree about which frame is the first
                # absent one -- see RemovalSchedule's docstring. Frame 0 is
                # the one case with no prior step to place the call before;
                # every configured B3 scenario keeps switch_frame > 0 via
                # removal_warmup_frames, but this still keeps the contract
                # for switch_frame == 0.
                if removal_schedule is not None and removal_schedule.switch_frame == 0:
                    integration.projection_validation.remove_scenario_pedestrian()
                    removed_already = True

                for frame in range(base.steps + 1):
                    if dropout_wrapper is not None:
                        dropout_wrapper.frame = frame

                    result = runtime_pipeline.observe(observation.front_rgb)
                    selection = f9c_eval._selection_from_render_result(result)
                    ego_motion = observation.ego.motion

                    baseline_belief, corrected_measurement, robust_belief, record = (
                        f9c_eval._step_both_systems(
                            selection=selection,
                            baseline_updater=baseline_updater,
                            baseline_belief=baseline_belief,
                            baseline_bias=baseline_bias,
                            robust_updater=robust_updater,
                            robust_belief=robust_belief,
                            previous_action=previous_action,
                            ego=observation.ego,
                            road=observation.road,
                            ego_motion=ego_motion,
                            dt_s=dt_s,
                        )
                    )

                    # Runtime/privileged boundary: both updaters have been
                    # stepped; only now is ground truth read.
                    privileged = integration.privileged.read()
                    silhouettes = {
                        item.object_kind: item
                        for item in integration.projection_validation.sample_object_silhouettes(
                            ("duckie",)
                        )
                    }
                    silhouette = silhouettes.get("duckie")
                    gt_box = None if silhouette is None else silhouette.bounding_box
                    visible_pixels = 0 if silhouette is None else silhouette.visible_pixel_count
                    decision = assess_silhouette(
                        class_id=yolo_class_id(ObjectClass.DUCKIE),
                        box=gt_box,
                        visible_pixel_count=visible_pixels,
                        image_width_px=settings.image_width_px,
                        image_height_px=settings.image_height_px,
                        rules=annotation.duckie_rules,
                    )
                    truth_state = privileged.true_pomdp_state.pedestrian
                    gt_exists = bool(truth_state.exists)
                    has_kinematics = (
                        gt_exists
                        and truth_state.range_m is not None
                        and truth_state.bearing_rad is not None
                        and truth_state.radial_velocity_mps is not None
                        and truth_state.bearing_rate_rad_s is not None
                    )

                    if has_kinematics:
                        distance_bin = f9c_eval._distance_bin(
                            truth_state.range_m, settings.near_max_m, settings.medium_max_m
                        )
                        fov_region = f9c_eval._fov_region(
                            gt_box if decision.accepted else None, settings.image_width_px
                        )
                    else:
                        # No GT kinematics -- either a genuine B3 removal
                        # (expected, not a warning) or an unexpected privileged
                        # truth loss on a B1/B2 scenario (unexpected; recorded
                        # below so it can be reviewed).
                        distance_bin = "absent"
                        fov_region = "outside"
                        if spec.kind != "B3":
                            warnings.append(
                                {
                                    "episode": episode,
                                    "seed": seed,
                                    "scenario": base.name,
                                    "kind": spec.kind,
                                    "frame": frame,
                                    "reason": "unexpected_lost_pedestrian_truth",
                                }
                            )

                    truth = TruthFrame(
                        episode=episode,
                        frame=frame,
                        gt_exists=gt_exists,
                        gt_range_m=truth_state.range_m if has_kinematics else None,
                        gt_bearing_rad=truth_state.bearing_rad if has_kinematics else None,
                        gt_range_rate_mps=(
                            truth_state.radial_velocity_mps if has_kinematics else None
                        ),
                        gt_bearing_rate_rad_s=(
                            truth_state.bearing_rate_rad_s if has_kinematics else None
                        ),
                        eligible_visible=decision.accepted,
                        visible_pixel_count=visible_pixels,
                        gt_bbox=(
                            None
                            if gt_box is None
                            else (gt_box.x_min_px, gt_box.y_min_px, gt_box.x_max_px, gt_box.y_max_px)
                        ),
                        distance_bin=distance_bin,
                        fov_region=fov_region,
                    )

                    row = f9c_eval.build_row(
                        episode=episode,
                        seed=seed,
                        scenario=base.name,
                        pedestrian_mode=base.pedestrian_mode.value,
                        frame=frame,
                        timestamp_s=timestamp_s,
                        dt_s=dt_s,
                        previous_action=previous_action,
                        ego_motion=ego_motion,
                        truth=truth,
                        selection=selection,
                        baseline_updater=baseline_updater,
                        baseline_belief=baseline_belief,
                        corrected_measurement=corrected_measurement,
                        robust_updater=robust_updater,
                        robust_belief=robust_belief,
                        record=record,
                        matching_iou_threshold=settings.matching_iou_threshold,
                        minimum_range_m=ekf_config.minimum_range_m,
                    )
                    row["visibility_reason"] = decision.reason
                    row["absence_kind"] = spec.kind
                    row["gt_exists"] = gt_exists
                    row["dropout_frame"] = bool(
                        dropout_wrapper is not None and dropout_wrapper.schedule.is_dropout_frame(frame)
                    )
                    rows.append(row)

                    if frame == base.steps:
                        break

                    # Apply B3 removal now -- BEFORE the step() call below
                    # renders frame+1's image -- exactly when frame+1 is the
                    # scheduled switch_frame. This is what keeps the
                    # rendered image and privileged truth agreeing on
                    # switch_frame as the first absent frame (see
                    # RemovalSchedule's docstring); calling it any later
                    # (e.g. while processing switch_frame itself) would
                    # leave that frame's image one step stale relative to
                    # truth.
                    if (
                        removal_schedule is not None
                        and not removed_already
                        and removal_schedule.switch_frame == frame + 1
                    ):
                        integration.projection_validation.remove_scenario_pedestrian()
                        removed_already = True

                    transition = integration.agent.step(base.action)
                    diagnostics = integration.diagnostics.read()
                    if transition.terminated or transition.truncated:
                        warnings.append(
                            {
                                "episode": episode,
                                "seed": seed,
                                "scenario": base.name,
                                "kind": spec.kind,
                                "reached_frame": frame,
                                "intended_steps": base.steps,
                                "done_code": diagnostics.done_code,
                                "reason": "early_termination",
                            }
                        )
                        print(
                            f"WARNING: {episode} ended early at frame {frame}/"
                            f"{base.steps} ({diagnostics.done_code}); keeping "
                            f"{len(rows)} rows collected so far and moving on",
                            flush=True,
                        )
                        break
                    observation = transition.observation
                    dt_s = diagnostics.timestamp_s - timestamp_s
                    timestamp_s = diagnostics.timestamp_s
                    previous_action = base.action
            finally:
                integration.close()
            print(f"completed {episode} ({spec.kind}): total_rows={len(rows)}", flush=True)

    return rows, warnings


# ---------------------------------------------------------------------------
# Projection and resampling.
# ---------------------------------------------------------------------------


def _per_seed_run_counts(events: list[dict[str, Any]], kinds: tuple[str, ...]) -> dict[int, dict[str, int]]:
    """runs_ge_20 / runs_ge_40 per seed, counting only events of the given
    kinds (B1+B2 for the primary support count; a single kind for the
    per-kind breakdown)."""

    per_seed: dict[int, dict[str, int]] = {}
    for event in events:
        if event["kind"] not in kinds:
            continue
        seed = int(event["seed"])
        bucket = per_seed.setdefault(seed, {"runs_ge_20": 0, "runs_ge_40": 0})
        if event["length"] >= 20:
            bucket["runs_ge_20"] += 1
        if event["length"] >= 40:
            bucket["runs_ge_40"] += 1
    return per_seed


def project_to_final_seeds(
    events: list[dict[str, Any]],
    development_seeds: tuple[int, ...],
    number_of_final_seeds: int,
    kinds: tuple[str, ...],
) -> dict[str, Any]:
    per_seed = _per_seed_run_counts(events, kinds)
    runs_20 = [per_seed.get(seed, {"runs_ge_20": 0})["runs_ge_20"] for seed in development_seeds]
    runs_40 = [per_seed.get(seed, {"runs_ge_40": 0})["runs_ge_40"] for seed in development_seeds]

    mean_20 = statistics.mean(runs_20)
    mean_40 = statistics.mean(runs_40)

    return {
        "kinds": list(kinds),
        "development_seed_count": len(development_seeds),
        "final_seed_count": number_of_final_seeds,
        "per_seed_runs_ge_20": dict(zip(development_seeds, runs_20)),
        "per_seed_runs_ge_40": dict(zip(development_seeds, runs_40)),
        "projected_runs_ge_20": mean_20 * number_of_final_seeds,
        "projected_runs_ge_40": mean_40 * number_of_final_seeds,
    }


def resample_four_of_eight(
    events: list[dict[str, Any]],
    development_seeds: tuple[int, ...],
    kinds: tuple[str, ...],
    minimum_20: int,
    minimum_40: int,
) -> dict[str, Any]:
    """Exhaustive C(8,4)=70 resampling check (Task 3's own fix-round
    method, applied here to absence runs instead of outlier frames): every
    4-seed subset of the 8 development seeds stands in for one possible
    draw of the 4 unseen final seeds, so the fraction of subsets that fall
    below each pre-registered minimum is a direct, non-parametric estimate
    of tail risk -- not just a point projection."""

    per_seed = _per_seed_run_counts(events, kinds)
    runs_20_by_seed = {seed: per_seed.get(seed, {"runs_ge_20": 0})["runs_ge_20"] for seed in development_seeds}
    runs_40_by_seed = {seed: per_seed.get(seed, {"runs_ge_40": 0})["runs_ge_40"] for seed in development_seeds}

    subsets = list(combinations(development_seeds, 4))
    sums_20 = [sum(runs_20_by_seed[seed] for seed in subset) for subset in subsets]
    sums_40 = [sum(runs_40_by_seed[seed] for seed in subset) for subset in subsets]

    below_20 = sum(1 for value in sums_20 if value < minimum_20)
    below_40 = sum(1 for value in sums_40 if value < minimum_40)

    return {
        "kinds": list(kinds),
        "n_subsets": len(subsets),
        "runs_ge_20_subset_sums": {"min": min(sums_20), "median": statistics.median(sums_20), "max": max(sums_20)},
        "runs_ge_40_subset_sums": {"min": min(sums_40), "median": statistics.median(sums_40), "max": max(sums_40)},
        "fraction_below_minimum_20": below_20 / len(subsets),
        "fraction_below_minimum_40": below_40 / len(subsets),
        "minimum_20": minimum_20,
        "minimum_40": minimum_40,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to configs/f9d_evidence_closure_v1.toml")
    args = parser.parse_args()

    f9d_config_path = Path(args.config).resolve()
    protocol = load_f9d_protocol(f9d_config_path, require_frozen=True)
    f9c_protocol = f9c_parameters(protocol, require_frozen=True)
    settings = _ProbeSettings(f9c_protocol.config_path)
    specs = _load_absence_scenarios(f9d_config_path)

    if any(spec.base.use_for_final_evaluation for spec in specs):
        raise RuntimeError(
            "absence_scenario_matrix has an entry marked use_for_final = true "
            "before the yield probe has passed"
        )

    print(
        f"Rendering {len(specs)} absence scenarios "
        f"({sum(1 for s in specs if s.kind=='B1')} B1 / "
        f"{sum(1 for s in specs if s.kind=='B2')} B2 / "
        f"{sum(1 for s in specs if s.kind=='B3')} B3) x "
        f"{len(protocol.development_seeds)} development seeds "
        f"({protocol.development_seeds})...",
        flush=True,
    )
    rows, episode_warnings = collect_absence_rows(
        f9c_protocol, settings, protocol.development_seeds, specs
    )

    summary = absence_yield(rows)
    events = summary["events"]

    # Per-kind is the PRIMARY evidence structure (coordinator fix round 1):
    # B1 and B2 decay through mechanistically different paths (B1 through
    # the pure P_S/P_birth prediction recurrence with no likelihood applied
    # at all; B2 through the I8-floored likelihood update), so each kind's
    # support must be checked and recorded independently -- never only as a
    # number that has already pooled kinds together. absence_support_satisfied
    # (duckie_pomdp.evaluation.f9d_protocol) is the single place the
    # pre-registered minimum_absence_runs_20/_40 comparison happens; this
    # script never re-implements that comparison locally.
    per_kind_projection: dict[str, Any] = {}
    per_kind_resample: dict[str, Any] = {}
    per_kind_support_satisfied: dict[str, bool] = {}
    for kind in ("B1", "B2", "B3"):
        projection = project_to_final_seeds(
            events, protocol.development_seeds, len(protocol.absence_final_seeds), (kind,)
        )
        resample = resample_four_of_eight(
            events,
            protocol.development_seeds,
            (kind,),
            protocol.minimum_absence_runs_20,
            protocol.minimum_absence_runs_40,
        )
        per_kind_projection[kind] = projection
        per_kind_resample[kind] = resample
        per_kind_support_satisfied[kind] = absence_support_satisfied(
            protocol,
            runs_ge_20=projection["projected_runs_ge_20"],
            runs_ge_40=projection["projected_runs_ge_40"],
        )

    # B1+B2 combined is kept ONLY as a supplementary line, never the primary
    # judgement -- pooling kinds at the yield stage is exactly what the plan
    # forbids at the reporting stage. It is retained here only because it
    # was the original (superseded) primary figure and a reviewer may want
    # to see it move alongside the per-kind numbers, not because any
    # decision should be read from it.
    combined_projection = project_to_final_seeds(
        events, protocol.development_seeds, len(protocol.absence_final_seeds), ("B1", "B2")
    )
    combined_resample = resample_four_of_eight(
        events,
        protocol.development_seeds,
        ("B1", "B2"),
        protocol.minimum_absence_runs_20,
        protocol.minimum_absence_runs_40,
    )
    combined_support_satisfied_supplementary = absence_support_satisfied(
        protocol,
        runs_ge_20=combined_projection["projected_runs_ge_20"],
        runs_ge_40=combined_projection["projected_runs_ge_40"],
    )

    report = {
        "schema_version": 1,
        "gate": "f9d",
        "task": "task-4-absence-yield-probe",
        "config_sha256": protocol.config_sha256,
        "f9c_config_sha256": protocol.f9c_config_sha256,
        "development_seeds": list(protocol.development_seeds),
        "scenario_names": [spec.base.name for spec in specs],
        "episode_warnings": episode_warnings,
        "yield": summary,
        "per_kind_projection": per_kind_projection,
        "per_kind_resample": per_kind_resample,
        "per_kind_support_satisfied": per_kind_support_satisfied,
        "combined_b1_b2_projection_supplementary": combined_projection,
        "combined_b1_b2_resample_supplementary": combined_resample,
        "combined_b1_b2_support_satisfied_supplementary": combined_support_satisfied_supplementary,
        "minimum_absence_runs_20": protocol.minimum_absence_runs_20,
        "minimum_absence_runs_40": protocol.minimum_absence_runs_40,
    }

    output_path = protocol.artifacts["yield_probe_json"].with_name("f9d_absence_yield_probe.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
