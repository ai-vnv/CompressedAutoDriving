"""Run frozen YOLO measurements through the frozen F7 pedestrian EKF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.belief.updater import (
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.dataset.annotations import assess_silhouette
from duckie_pomdp.dataset.config import load_dataset_config
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import BoundingBox, ObjectClass, yolo_class_id
from duckie_pomdp.domain.measurement import ObjectMeasurement, PerceptionObservation
from duckie_pomdp.evaluation.f9_belief import summarize_f9
from duckie_pomdp.evaluation.f9_protocol import F9Protocol, load_f9_protocol, sha256
from duckie_pomdp.evaluation.yolo_detection import intersection_over_union
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.f9_pipeline import (
    AdditiveMeasurementBias,
    YoloPedestrianMeasurementPipeline,
)
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
    wrap_angle,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector


ROOT = Path(__file__).resolve().parents[1]


def _box_fields(prefix: str, box: BoundingBox | None) -> dict[str, object]:
    return {
        f"{prefix}_x1": None if box is None else box.x_min_px,
        f"{prefix}_y1": None if box is None else box.y_min_px,
        f"{prefix}_x2": None if box is None else box.x_max_px,
        f"{prefix}_y2": None if box is None else box.y_max_px,
    }


def _fov_region(box: BoundingBox | None, image_width_px: int) -> str:
    if box is None:
        return "outside"
    center = 0.5 * (box.x_min_px + box.x_max_px)
    normalized = abs(center - 0.5 * image_width_px) / (0.5 * image_width_px)
    if normalized < 1.0 / 3.0:
        return "center"
    if normalized < 2.0 / 3.0:
        return "mid_fov"
    return "edge_fov"


def _perception(observation, pedestrian: ObjectMeasurement) -> PerceptionObservation:
    return PerceptionObservation(
        ego=observation.ego,
        road=observation.road,
        stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
        pedestrian=pedestrian,
    )


def _belief_fields(prefix: str, updater, belief) -> dict[str, object]:
    initialized = updater.ekf.initialized
    pedestrian = belief.pedestrian
    diagnostics = updater.last_diagnostics.ekf
    state = updater.ekf.state if initialized else None
    return {
        f"{prefix}_belief_initialized": initialized,
        f"{prefix}_belief_range_m": pedestrian.range_mean_m,
        f"{prefix}_belief_range_std_m": pedestrian.range_std_m,
        f"{prefix}_belief_bearing_rad": pedestrian.bearing_mean_rad,
        f"{prefix}_belief_bearing_std_rad": pedestrian.bearing_std_rad,
        f"{prefix}_belief_range_rate_mps": pedestrian.radial_velocity_mean_mps,
        f"{prefix}_belief_range_rate_std_mps": pedestrian.radial_velocity_std_mps,
        f"{prefix}_belief_bearing_rate_rad_s": pedestrian.bearing_rate_mean_rad_s,
        f"{prefix}_belief_bearing_rate_std_rad_s": pedestrian.bearing_rate_std_rad_s,
        f"{prefix}_existence_probability": pedestrian.existence_probability,
        f"{prefix}_internal_x_left_m": None if state is None else float(state[0]),
        f"{prefix}_internal_y_forward_m": None if state is None else float(state[1]),
        f"{prefix}_internal_velocity_left_mps": None if state is None else float(state[2]),
        f"{prefix}_internal_velocity_forward_mps": None if state is None else float(state[3]),
        f"{prefix}_innovation_range_m": diagnostics.innovation_range_m,
        f"{prefix}_innovation_bearing_rad": diagnostics.innovation_bearing_rad,
        f"{prefix}_nis": diagnostics.nis,
        f"{prefix}_covariance_trace": diagnostics.covariance_trace,
    }


def _remember_case(
    cases: dict[str, list[dict[str, object]]],
    category: str,
    score: float | None,
    row: dict[str, object],
    rgb: np.ndarray,
    gt_box: BoundingBox | None,
    selected_box: BoundingBox | None,
    maximum: int,
) -> None:
    if score is None or not np.isfinite(score):
        return
    case = {
        "score": float(score),
        "row": dict(row),
        "rgb": np.asarray(rgb).copy(),
        "gt_box": gt_box,
        "selected_box": selected_box,
    }
    bucket = cases.setdefault(category, [])
    bucket.append(case)
    bucket.sort(
        key=lambda item: (
            -float(item["score"]),
            str(item["row"]["episode"]),
            int(item["row"]["frame"]),
        )
    )
    del bucket[maximum:]


def _save_error_cases(
    cases: dict[str, list[dict[str, object]]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for category in sorted(cases):
        for index, case in enumerate(cases[category]):
            row = case["row"]
            image = Image.fromarray(case["rgb"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box, color in (
                (case["gt_box"], (0, 255, 0)),
                (case["selected_box"], (255, 0, 0)),
            ):
                if box is None:
                    continue
                draw.rectangle(
                    (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                    outline=color,
                    width=3,
                )
                point = box.bottom_center
                draw.ellipse(
                    (point.x_px - 4, point.y_px - 4, point.x_px + 4, point.y_px + 4),
                    fill=color,
                )
            label = (
                f"{category} score={case['score']:.3f} conf={row['selected_confidence']} "
                f"z=({row['raw_measurement_range_m']},{row['raw_measurement_bearing_rad']}) "
                f"ekf=({row['corrected_belief_range_m']:.3f},"
                f"{row['corrected_belief_bearing_rad']:.3f}) "
                f"gt=({row['gt_range_m']:.3f},{row['gt_bearing_rad']:.3f})"
            )
            draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
            draw.text((4, 4), label, fill=(255, 255, 255))
            destination = output_dir / (
                f"{category}_{index:02d}_{row['episode']}_f{int(row['frame']):04d}.png"
            )
            image.save(destination)
            outputs.append(str(destination))
    return outputs


def collect_final_rows(
    protocol: F9Protocol,
) -> tuple[list[dict[str, object]], list[str]]:
    annotation = load_dataset_config(protocol.annotation_config_path)
    detector = YoloObjectDetector(
        protocol.checkpoint_path,
        confidence_threshold=protocol.detector.confidence_threshold,
        iou_threshold=protocol.detector.nms_iou_threshold,
        image_size=protocol.detector.image_size,
        device=protocol.detector.device,
        max_detections=protocol.detector.max_detections,
    )
    measurement_noise = load_polar_measurement_noise(protocol.config_path)
    ekf_config = load_pedestrian_ekf_config(protocol.config_path)
    existence_config = load_existence_filter_config(protocol.config_path)
    bias = AdditiveMeasurementBias(protocol.range_bias_m, protocol.bearing_bias_rad)
    rows: list[dict[str, object]] = []
    cases: dict[str, list[dict[str, object]]] = {}

    for seed in protocol.final_evaluation_seeds:
        for scenario_index, spec in enumerate(protocol.scenarios):
            if not spec.use_for_final_evaluation:
                continue
            scenario = protocol.scenario_for(seed, scenario_index)
            episode = f"evaluation_{seed}_{spec.name}"
            integration = create_gym_duckietown(
                GymDuckietownConfig(
                    scenario=scenario,
                    domain_randomization=protocol.domain_randomization,
                    dynamics_randomization=protocol.dynamics_randomization,
                    maximum_steps=spec.steps + 2,
                    camera_width=protocol.image_width_px,
                    camera_height=protocol.image_height_px,
                )
            )
            try:
                observation = integration.agent.reset(seed=seed)
                projector = YoloMeasurementProjector(
                    CalibratedGroundProjector(integration.camera_calibration.read()),
                    MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
                )
                runtime = YoloPedestrianMeasurementPipeline(detector, projector)
                raw_updater = PedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    existence_config=existence_config,
                    measurement_noise=measurement_noise,
                    measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
                )
                corrected_updater = PedestrianBeliefUpdater(
                    ekf_config=ekf_config,
                    existence_config=existence_config,
                    measurement_noise=measurement_noise,
                    measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
                )
                raw_belief = initial_belief(
                    observation.ego,
                    existence_prior=existence_config.prior_probability,
                )
                corrected_belief = initial_belief(
                    observation.ego,
                    existence_prior=existence_config.prior_probability,
                )
                timestamp_s = 0.0
                dt_s = 1.0 / 30.0
                previous_action = PolicyAction(0.0, 0.0)

                for frame in range(spec.steps + 1):
                    # The complete runtime chain ends before privileged truth is read.
                    runtime_result = runtime.observe(observation.front_rgb)
                    raw_measurement = runtime_result.pedestrian
                    corrected_measurement = bias.correct(raw_measurement)
                    raw_was_initialized = raw_updater.ekf.initialized
                    raw_belief = raw_updater.update(
                        previous_belief=raw_belief,
                        previous_action=previous_action,
                        ego_motion=observation.ego.motion,
                        perception=_perception(observation, raw_measurement),
                        dt_s=dt_s,
                    )
                    corrected_belief = corrected_updater.update(
                        previous_belief=corrected_belief,
                        previous_action=previous_action,
                        ego_motion=observation.ego.motion,
                        perception=_perception(observation, corrected_measurement),
                        dt_s=dt_s,
                    )

                    # Evaluation-only boundary starts here.
                    privileged = integration.privileged.read()
                    silhouettes = {
                        item.object_kind: item
                        for item in integration.projection_validation.sample_object_silhouettes(
                            ("duckie",)
                        )
                    }
                    silhouette = silhouettes.get("duckie")
                    gt_box = None if silhouette is None else silhouette.bounding_box
                    decision = assess_silhouette(
                        class_id=yolo_class_id(ObjectClass.DUCKIE),
                        box=gt_box,
                        visible_pixel_count=(
                            0 if silhouette is None else silhouette.visible_pixel_count
                        ),
                        image_width_px=protocol.image_width_px,
                        image_height_px=protocol.image_height_px,
                        rules=annotation.duckie_rules,
                    )
                    selected = runtime_result.selected_duckie
                    selected_box = None if selected is None else selected.bounding_box
                    iou = (
                        None
                        if gt_box is None or selected_box is None or not decision.accepted
                        else intersection_over_union(gt_box, selected_box)
                    )
                    selected_correct = bool(
                        decision.accepted
                        and iou is not None
                        and iou >= protocol.matching_iou_threshold
                    )
                    false_measurement = bool(
                        raw_measurement.detected and not selected_correct
                    )
                    truth = privileged.true_pomdp_state.pedestrian
                    if (
                        not truth.exists
                        or truth.range_m is None
                        or truth.bearing_rad is None
                        or truth.radial_velocity_mps is None
                        or truth.bearing_rate_rad_s is None
                    ):
                        raise RuntimeError(f"{episode} lost pedestrian truth")
                    stop_confidence = max(
                        (item.confidence for item in runtime_result.stop_sign_detections),
                        default=None,
                    )
                    row: dict[str, object] = {
                        "episode": episode,
                        "seed": seed,
                        "scenario": spec.name,
                        "pedestrian_mode": spec.pedestrian_mode.value,
                        "frame": frame,
                        "timestamp_s": timestamp_s,
                        "dt_s": dt_s,
                        "eligible_visible": decision.accepted,
                        "visibility_reason": decision.reason,
                        "visible_pixel_count": (
                            0 if silhouette is None else silhouette.visible_pixel_count
                        ),
                        "measurement_detected": raw_measurement.detected,
                        "duckie_detection_count": runtime_result.duckie_detection_count,
                        "duplicate_selection": runtime_result.duplicate_selection,
                        "selected_confidence": (
                            None if selected is None else selected.confidence
                        ),
                        "selected_iou": iou,
                        "selected_correct_iou50": selected_correct,
                        "projection_error": runtime_result.projection_error,
                        "stop_sign_detection_count": len(
                            runtime_result.stop_sign_detections
                        ),
                        "stop_sign_max_confidence": stop_confidence,
                        "false_measurement_event": false_measurement,
                        "false_track_initialization": (
                            false_measurement and not raw_was_initialized
                        ),
                        "gt_range_m": truth.range_m,
                        "gt_bearing_rad": truth.bearing_rad,
                        "gt_range_rate_mps": truth.radial_velocity_mps,
                        "gt_bearing_rate_rad_s": truth.bearing_rate_rad_s,
                        "raw_measurement_range_m": raw_measurement.range_m,
                        "raw_measurement_bearing_rad": raw_measurement.bearing_rad,
                        "corrected_measurement_range_m": corrected_measurement.range_m,
                        "corrected_measurement_bearing_rad": corrected_measurement.bearing_rad,
                        "raw_measurement_range_error_m": (
                            None
                            if raw_measurement.range_m is None
                            else raw_measurement.range_m - truth.range_m
                        ),
                        "raw_measurement_bearing_error_rad": (
                            None
                            if raw_measurement.bearing_rad is None
                            else wrap_angle(
                                raw_measurement.bearing_rad - truth.bearing_rad
                            )
                        ),
                        "corrected_measurement_range_error_m": (
                            None
                            if corrected_measurement.range_m is None
                            else corrected_measurement.range_m - truth.range_m
                        ),
                        "corrected_measurement_bearing_error_rad": (
                            None
                            if corrected_measurement.bearing_rad is None
                            else wrap_angle(
                                corrected_measurement.bearing_rad
                                - truth.bearing_rad
                            )
                        ),
                        "distance_bin": protocol.distance_bin(truth.range_m),
                        "fov_region": _fov_region(
                            gt_box if decision.accepted else None,
                            protocol.image_width_px,
                        ),
                        "actual_ego_velocity_mps": observation.ego.linear_velocity_mps,
                        "actual_ego_yaw_rate_rad_s": observation.ego.yaw_rate_rad_s,
                        "commanded_velocity_mps": previous_action.linear_velocity_mps,
                        "commanded_yaw_rate_rad_s": previous_action.angular_velocity_rad_s,
                    }
                    row.update(_box_fields("gt_bbox", gt_box))
                    row.update(_box_fields("selected_bbox", selected_box))
                    row.update(_belief_fields("raw", raw_updater, raw_belief))
                    row.update(
                        _belief_fields("corrected", corrected_updater, corrected_belief)
                    )
                    rows.append(row)

                    corrected_diag = corrected_updater.last_diagnostics.ekf
                    _remember_case(
                        cases,
                        "range_innovation",
                        None
                        if corrected_diag.innovation_range_m is None
                        else abs(corrected_diag.innovation_range_m),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selected_box,
                        protocol.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "bearing_innovation",
                        None
                        if corrected_diag.innovation_bearing_rad is None
                        else abs(corrected_diag.innovation_bearing_rad),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selected_box,
                        protocol.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "nis",
                        corrected_diag.nis,
                        row,
                        observation.front_rgb,
                        gt_box,
                        selected_box,
                        protocol.error_cases_per_category,
                    )
                    _remember_case(
                        cases,
                        "belief_range_error",
                        (
                            abs(corrected_belief.pedestrian.range_mean_m - truth.range_m)
                            if corrected_updater.ekf.initialized
                            else None
                        ),
                        row,
                        observation.front_rgb,
                        gt_box,
                        selected_box,
                        protocol.error_cases_per_category,
                    )

                    if frame == spec.steps:
                        break
                    transition = integration.agent.step(spec.action)
                    diagnostics = integration.diagnostics.read()
                    if transition.terminated or transition.truncated:
                        raise RuntimeError(
                            f"{episode} ended early: {diagnostics.done_code}"
                        )
                    observation = transition.observation
                    dt_s = diagnostics.timestamp_s - timestamp_s
                    timestamp_s = diagnostics.timestamp_s
                    previous_action = spec.action
            finally:
                integration.close()
            print(f"completed {episode}: total_rows={len(rows)}", flush=True)
    return rows, _save_error_cases(cases, protocol.artifacts["error_case_dir"])


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise RuntimeError("F9 final evaluation produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f9_yolo_ekf_v1.toml",
    )
    args = parser.parse_args()
    protocol = load_f9_protocol(args.config, require_frozen=True)
    rows, error_cases = collect_final_rows(protocol)
    metrics, nis = summarize_f9(
        rows,
        active_probability_threshold=protocol.active_belief_probability_threshold,
        nis_threshold=protocol.chi_square_2d_95_threshold,
    )
    oracle_path = ROOT / "artifacts" / "belief_calibration_metrics.json"
    report = {
        "schema_version": 1,
        "gate": "F9b",
        "status": "evaluation_complete_not_posthoc_tuned",
        "source": {
            "config_path": str(protocol.config_path),
            "config_sha256": protocol.config_sha256,
            "checkpoint_sha256": protocol.checkpoint_sha256,
            "calibration_artifact_sha256": protocol.calibration_artifact_sha256,
            "frozen_f7_config_sha256": sha256(protocol.frozen_f7_config_path),
            "final_evaluation_seeds": list(protocol.final_evaluation_seeds),
            "calibration_seeds_not_read": list(protocol.calibration_seeds),
            "runtime_path": "front_rgb -> frozen_yolo -> raw_projection -> optional_fixed_bias -> frozen_f7_ekf",
            "privileged_truth_use": "post-update evaluation only",
        },
        "measurement_model": {
            "range_bias_m": protocol.range_bias_m,
            "bearing_bias_rad": protocol.bearing_bias_rad,
            "f5b_range_calibration_used": False,
            "gaussian_assumption": "provisional_imperfect_for_yolo",
        },
        "metrics": metrics,
        "oracle_f7_context": json.loads(oracle_path.read_text(encoding="utf-8")),
        "error_case_images": error_cases,
    }
    validation_path = protocol.artifacts["validation_csv"]
    metrics_path = protocol.artifacts["belief_metrics_json"]
    nis_path = protocol.artifacts["nis_metrics_json"]
    _write_csv(rows, validation_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    nis_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "F9b",
                "config_sha256": protocol.config_sha256,
                "nis": nis,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "validation_csv": str(validation_path),
                "belief_metrics": str(metrics_path),
                "nis_metrics": str(nis_path),
                "rows": len(rows),
                "events": metrics["events"],
                "current_frame_yolo": metrics["current_frame_yolo"],
                "ekf": metrics["ekf"],
                "nis": nis["global"],
                "error_cases": len(error_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
