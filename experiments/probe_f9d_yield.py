"""Gate F9d Task 3: the outlier-yield probe on development seeds only.

F9c rendered its final-evaluation seeds without knowing whether the
scenarios would produce enough natural localization outliers, and got 9
frames out of 3,328 -- far too few to conclude anything, on seeds that could
not be re-rendered. This script exists to answer that question BEFORE any
final seed (8201-8204/8301-8304) is ever rendered: it renders ONLY the eight
development seeds (8101-8108) against the ``[[outlier_scenario_matrix]]``
scenarios in ``configs/f9d_evidence_closure_v1.toml``, counts natural
localization-outlier frames and the contiguous events they form
(``duckie_pomdp.evaluation.f9d_stress.outlier_yield``), and projects the
observed per-seed rate onto the 4 final outlier seeds.

**No estimator parameter is read, written, or overridden here.** Every EKF/
existence/bias/robust-observation object below is loaded fresh from the
FROZEN F9c config (via ``f9d_protocol.f9c_parameters``, which itself reloads
``configs/f9c_robust_belief_v1.toml`` from disk on every call) through the
same loader functions ``evaluate_f9c_robust_belief.collect_final_rows``
uses. This script imports and reuses that module's helpers
(``_scenario_for``, ``_selection_from_render_result``, ``_step_both_systems``,
``build_row``, ``_distance_bin``, ``_fov_region``) rather than re-deriving
any of that logic, so the outlier definition this probe measures against
cannot silently diverge from what the eventual final-evaluation render will
compute for the exact same kind of frame.

**Natural outliers only.** Nothing here injects GT-derived error, adjusts
the IoU threshold, or touches the simulator/detector/annotation rules. Every
scenario in ``outlier_scenario_matrix`` only shapes *geometry* -- pedestrian
crossing direction, ego start offset/heading, ego motion, episode length.

**Development seeds are reusable; unlike ``collect_final_rows``, an early
simulator termination does not abort the whole run** -- it is recorded as a
warning and that episode's remaining frames are simply absent, mirroring
``calibrate_f9c_robust_belief.py``'s stance for the same reason (Task 3's
own docstring: "Iterate here freely" -- development-seed renders may be
repeated as many times as scenario design requires).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, MutableSequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# The script's own directory is already sys.path[0] when invoked as
# ``python experiments/probe_f9d_yield.py`` -- this insert is only a
# safety net for other invocation styles (e.g. via -m, or from pytest).
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
from duckie_pomdp.evaluation.f9c_runtime_cache import (  # noqa: E402
    RuntimeCacheFrame,
    TruthFrame,
)
from duckie_pomdp.evaluation.f9d_protocol import (  # noqa: E402
    F9dProtocol,
    f9c_parameters,
    load_f9d_protocol,
    outlier_support_satisfied,
)
from duckie_pomdp.evaluation.f9d_stress import outlier_yield  # noqa: E402
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector  # noqa: E402
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
NUMBER_OF_FINAL_OUTLIER_SEEDS = 4  # len(protocol.outlier_final_seeds); asserted at runtime


# ---------------------------------------------------------------------------
# Plumbing settings not exposed by F9cProtocol -- mirrors
# evaluate_f9c_robust_belief._ExperimentSettings, reading the SAME frozen F9c
# config sections (detector/simulator/trajectory_perturbation/
# calibration_protocol). Duplicated (not imported) for the same reason that
# module documents for its own duplication: importing a private class from a
# sibling script is more surprising than the small repetition.
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


def _load_outlier_scenarios(f9d_config_path: Path) -> list[F9ScenarioSpec]:
    """Read ``[[outlier_scenario_matrix]]`` straight from the F9d config.

    Deliberately NOT read through ``F9dProtocol.scenarios`` (Task 1's
    generic ``data.get("scenario_matrix", [])``): that key names a
    different, not-yet-populated section, and Task 4 will add its own
    ``[[absence_scenario_matrix]]`` alongside this one -- each stress
    dimension owns its own named matrix rather than sharing one generic
    ``scenario_matrix``.
    """

    with f9d_config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    items = data.get("outlier_scenario_matrix", [])
    if not items:
        raise RuntimeError(
            f"{f9d_config_path} has no [[outlier_scenario_matrix]] entries"
        )
    specs: list[F9ScenarioSpec] = []
    for item in items:
        steps = int(item["steps"])
        if steps <= 0:
            raise ValueError("outlier scenario steps must be positive")
        specs.append(
            F9ScenarioSpec(
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
        )
    return specs


# ---------------------------------------------------------------------------
# The render loop. Modeled directly on
# evaluate_f9c_robust_belief.collect_final_rows -- same per-frame shared
# path (_selection_from_render_result -> _step_both_systems -> build_row),
# same privileged-read-after-both-updaters-stepped ordering. Differs only in
# (a) no runtime-cache/error-case bookkeeping (this probe never needs to be
# replayed or inspected as images), and (b) an early termination is a
# recorded warning, not a fatal error -- development seeds may be rendered
# as many times as scenario design requires.
# ---------------------------------------------------------------------------


def collect_probe_rows(
    f9c_protocol: F9cProtocol,
    settings: _ProbeSettings,
    seeds: tuple[int, ...],
    specs: list[F9ScenarioSpec],
    *,
    runtime_cache: MutableSequence[RuntimeCacheFrame] | None = None,
    evaluation_truth: MutableSequence[TruthFrame] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect stress rows, optionally preserving replayable raw evidence.

    The optional sinks are used by Task 6's once-only final render.  They do
    not affect detector selection or either estimator: cache frames are
    captured directly from the raw projection result before bias correction,
    and truth is appended only after both systems have already stepped.
    """
    annotation = load_dataset_config(settings.annotation_config_path)
    detector = YoloObjectDetector(
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
            scenario = f9c_eval._scenario_for(settings, spec, seed, scenario_index)
            episode = f"probe_{seed}_{spec.name}"
            integration = create_gym_duckietown(
                GymDuckietownConfig(
                    scenario=scenario,
                    domain_randomization=settings.domain_randomization,
                    dynamics_randomization=settings.dynamics_randomization,
                    maximum_steps=spec.steps + 2,
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
                runtime_pipeline = YoloPedestrianMeasurementPipeline(detector, projector)
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

                for frame in range(spec.steps + 1):
                    result = runtime_pipeline.observe(observation.front_rgb)
                    selection = f9c_eval._selection_from_render_result(result)
                    ego_motion = observation.ego.motion

                    if runtime_cache is not None:
                        runtime_cache.append(
                            RuntimeCacheFrame(
                                episode=episode,
                                seed=seed,
                                scenario=spec.name,
                                frame=frame,
                                dt_s=dt_s,
                                raw_candidate_range_m=tuple(
                                    float("nan")
                                    if candidate.projection_error is not None
                                    else float(candidate.measurement.range_m)
                                    for candidate in result.duckie_candidates
                                ),
                                raw_candidate_bearing_rad=tuple(
                                    float("nan")
                                    if candidate.projection_error is not None
                                    else float(candidate.measurement.bearing_rad)
                                    for candidate in result.duckie_candidates
                                ),
                                raw_candidate_confidence=tuple(
                                    float(candidate.detection.confidence)
                                    for candidate in result.duckie_candidates
                                ),
                                raw_candidate_bbox=tuple(
                                    (
                                        candidate.detection.bounding_box.x_min_px,
                                        candidate.detection.bounding_box.y_min_px,
                                        candidate.detection.bounding_box.x_max_px,
                                        candidate.detection.bounding_box.y_max_px,
                                    )
                                    for candidate in result.duckie_candidates
                                ),
                                raw_candidate_projection_failed=tuple(
                                    candidate.projection_error is not None
                                    for candidate in result.duckie_candidates
                                ),
                                ego_linear_velocity_mps=ego_motion.linear_velocity_mps,
                                ego_yaw_rate_rad_s=ego_motion.yaw_rate_rad_s,
                            )
                        )

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
                    if (
                        not truth_state.exists
                        or truth_state.range_m is None
                        or truth_state.bearing_rad is None
                        or truth_state.radial_velocity_mps is None
                        or truth_state.bearing_rate_rad_s is None
                    ):
                        warnings.append(
                            {
                                "episode": episode,
                                "seed": seed,
                                "scenario": spec.name,
                                "reached_frame": frame,
                                "intended_steps": spec.steps,
                                "reason": "lost_pedestrian_truth",
                            }
                        )
                        print(
                            f"WARNING: {episode} lost pedestrian truth at frame "
                            f"{frame}/{spec.steps}; keeping {len(rows)} rows "
                            "collected so far and moving on",
                            flush=True,
                        )
                        break

                    distance_bin = f9c_eval._distance_bin(
                        truth_state.range_m, settings.near_max_m, settings.medium_max_m
                    )
                    fov_region = f9c_eval._fov_region(
                        gt_box if decision.accepted else None, settings.image_width_px
                    )

                    truth = TruthFrame(
                        episode=episode,
                        frame=frame,
                        gt_exists=bool(truth_state.exists),
                        gt_range_m=truth_state.range_m,
                        gt_bearing_rad=truth_state.bearing_rad,
                        gt_range_rate_mps=truth_state.radial_velocity_mps,
                        gt_bearing_rate_rad_s=truth_state.bearing_rate_rad_s,
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
                    if evaluation_truth is not None:
                        evaluation_truth.append(truth)

                    row = f9c_eval.build_row(
                        episode=episode,
                        seed=seed,
                        scenario=spec.name,
                        pedestrian_mode=spec.pedestrian_mode.value,
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
                    rows.append(row)

                    if frame == spec.steps:
                        break
                    transition = integration.agent.step(spec.action)
                    diagnostics = integration.diagnostics.read()
                    if transition.terminated or transition.truncated:
                        warnings.append(
                            {
                                "episode": episode,
                                "seed": seed,
                                "scenario": spec.name,
                                "reached_frame": frame,
                                "intended_steps": spec.steps,
                                "done_code": diagnostics.done_code,
                                "reason": "early_termination",
                            }
                        )
                        print(
                            f"WARNING: {episode} ended early at frame {frame}/"
                            f"{spec.steps} ({diagnostics.done_code}); keeping "
                            f"{len(rows)} rows collected so far and moving on",
                            flush=True,
                        )
                        break
                    observation = transition.observation
                    dt_s = diagnostics.timestamp_s - timestamp_s
                    timestamp_s = diagnostics.timestamp_s
                    previous_action = spec.action
            finally:
                integration.close()
            print(f"completed {episode}: total_rows={len(rows)}", flush=True)

    return rows, warnings


# ---------------------------------------------------------------------------
# Projection onto the 4 final outlier seeds and the pre-registered decision.
# ---------------------------------------------------------------------------


def _per_seed_stats(summary: dict[str, Any], seeds: tuple[int, ...]) -> dict[int, dict[str, int]]:
    return {
        seed: {
            "outlier_frames": summary["outlier_frames_per_seed"].get(seed, 0),
            "outlier_events": summary["events_per_seed"].get(seed, 0),
        }
        for seed in seeds
    }


def project_to_final_seeds(
    summary: dict[str, Any],
    development_seeds: tuple[int, ...],
    number_of_final_seeds: int,
) -> dict[str, Any]:
    """Naive linear projection: each development seed already gives an
    observed per-seed outlier rate (frames/events per seed); the projected
    total on the final band is that mean per-seed rate multiplied by HOW
    MANY final seeds there will be -- ``mean(per_dev_seed_rate) *
    number_of_final_seeds``. (The development-seed count only enters when
    computing the mean in the first place; it must not appear a second time
    as an extra divisor, or the projection silently shrinks by another
    factor of ``number_of_dev_seeds``.) Stated explicitly per the gate
    brief rather than silently assumed -- this is the "naive x 4/8 scaling"
    the brief allows, where 4/8 multiplies the *total*, not the *mean*.
    Per-seed spread (min/max/stdev) is reported alongside the point
    projection because the projection itself cannot capture how much that
    rate varies seed to seed.
    """

    per_seed = _per_seed_stats(summary, development_seeds)
    frame_counts = [stats["outlier_frames"] for stats in per_seed.values()]
    event_counts = [stats["outlier_events"] for stats in per_seed.values()]

    mean_frames = statistics.mean(frame_counts)
    mean_events = statistics.mean(event_counts)
    seeds_with_event_fraction = summary["seeds_with_event"] / len(development_seeds)

    return {
        "method": (
            "naive_linear: projected = mean(per_dev_seed_rate) * "
            "number_of_final_seeds (equivalently, total_dev_count * "
            f"({number_of_final_seeds}/{len(development_seeds)}))"
        ),
        "development_seed_count": len(development_seeds),
        "final_seed_count": number_of_final_seeds,
        "per_seed_outlier_frames": {str(seed): stats["outlier_frames"] for seed, stats in per_seed.items()},
        "per_seed_outlier_events": {str(seed): stats["outlier_events"] for seed, stats in per_seed.items()},
        "outlier_frames_spread": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": mean_frames,
            "stdev": statistics.pstdev(frame_counts),
        },
        "outlier_events_spread": {
            "min": min(event_counts),
            "max": max(event_counts),
            "mean": mean_events,
            "stdev": statistics.pstdev(event_counts),
        },
        "projected_outlier_frames": mean_frames * number_of_final_seeds,
        "projected_outlier_events": mean_events * number_of_final_seeds,
        "projected_seeds_with_event": round(seeds_with_event_fraction * number_of_final_seeds),
    }


def _judge(
    protocol: F9dProtocol, projection: dict[str, Any]
) -> dict[str, Any]:
    frames = projection["projected_outlier_frames"]
    events = projection["projected_outlier_events"]
    seeds = projection["projected_seeds_with_event"]
    satisfied = outlier_support_satisfied(protocol, frames=frames, events=events, seeds=seeds)
    if satisfied:
        decision = "PROCEED_TO_FREEZE"
    elif frames >= protocol.insufficient_outlier_frames:
        decision = "REVISE_SCENARIOS_AND_REPROBE"
    else:
        decision = "STOP_INSUFFICIENT_NATURAL_YIELD"
    return {
        "criteria": {
            "frames": {
                "value": frames,
                "minimum": protocol.minimum_outlier_frames,
                "pass": frames >= protocol.minimum_outlier_frames,
            },
            "events": {
                "value": events,
                "minimum": protocol.minimum_outlier_events,
                "pass": events >= protocol.minimum_outlier_events,
            },
            "seeds": {
                "value": seeds,
                "minimum": protocol.minimum_outlier_seeds,
                "pass": seeds >= protocol.minimum_outlier_seeds,
            },
        },
        "outlier_support_satisfied": satisfied,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to configs/f9d_evidence_closure_v1.toml")
    args = parser.parse_args()

    f9d_config_path = Path(args.config).resolve()
    protocol = load_f9d_protocol(f9d_config_path, require_frozen=True)
    f9c_protocol = f9c_parameters(protocol, require_frozen=True)
    settings = _ProbeSettings(f9c_protocol.config_path)
    specs = _load_outlier_scenarios(f9d_config_path)

    if any(spec.use_for_final_evaluation for spec in specs):
        raise RuntimeError(
            "outlier_scenario_matrix has an entry marked use_for_final = true "
            "before the yield probe has passed -- Task 3 must leave every "
            "entry false until the probe result justifies freezing it"
        )

    print(
        f"Rendering {len(specs)} outlier scenarios x {len(protocol.development_seeds)} "
        f"development seeds ({protocol.development_seeds})...",
        flush=True,
    )
    rows, episode_warnings = collect_probe_rows(
        f9c_protocol, settings, protocol.development_seeds, specs
    )

    summary = outlier_yield(rows, matching_iou_threshold=settings.matching_iou_threshold)
    projection = project_to_final_seeds(
        summary, protocol.development_seeds, len(protocol.outlier_final_seeds)
    )
    judgement = _judge(protocol, projection)

    report = {
        "schema_version": 1,
        "gate": "f9d",
        "task": "task-3-outlier-yield-probe",
        "config_sha256": protocol.config_sha256,
        "f9c_config_sha256": protocol.f9c_config_sha256,
        "development_seeds": list(protocol.development_seeds),
        "scenario_names": [spec.name for spec in specs],
        "episode_warnings": episode_warnings,
        "yield": summary,
        "projection_to_final_seeds": projection,
        "judgement": judgement,
    }

    output_path = protocol.artifacts["yield_probe_json"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
