"""Estimate the F9 Duckie observation model on calibration-only simulator seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.dataset.annotations import assess_silhouette
from duckie_pomdp.dataset.config import load_dataset_config
from duckie_pomdp.domain.detection import BoundingBox, ObjectClass, yolo_class_id
from duckie_pomdp.evaluation.f9_calibration import summarize_calibration
from duckie_pomdp.evaluation.f9_protocol import F9Protocol, load_f9_protocol, sha256
from duckie_pomdp.evaluation.yolo_detection import intersection_over_union
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.f9_pipeline import YoloPedestrianMeasurementPipeline
from duckie_pomdp.perception.measurement_calibration import (
    LinearRangeCalibration,
    MeasurementCalibrator,
    wrap_angle,
)
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector


ROOT = Path(__file__).resolve().parents[1]


def _optional(value: Any) -> Any:
    return "" if value is None else value


def _box_fields(prefix: str, box: BoundingBox | None) -> dict[str, object]:
    return {
        f"{prefix}_x1": _optional(None if box is None else box.x_min_px),
        f"{prefix}_y1": _optional(None if box is None else box.y_min_px),
        f"{prefix}_x2": _optional(None if box is None else box.x_max_px),
        f"{prefix}_y2": _optional(None if box is None else box.y_max_px),
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


def collect_calibration_rows(protocol: F9Protocol) -> list[dict[str, object]]:
    annotation = load_dataset_config(protocol.annotation_config_path)
    detector_config = protocol.detector
    detector = YoloObjectDetector(
        protocol.checkpoint_path,
        confidence_threshold=detector_config.confidence_threshold,
        iou_threshold=detector_config.nms_iou_threshold,
        image_size=detector_config.image_size,
        device=detector_config.device,
        max_detections=detector_config.max_detections,
    )
    rows: list[dict[str, object]] = []
    for seed in protocol.calibration_seeds:
        for scenario_index, spec in enumerate(protocol.scenarios):
            if not spec.use_for_calibration:
                continue
            scenario = protocol.scenario_for(seed, scenario_index)
            episode = f"calibration_{seed}_{spec.name}"
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
                for frame in range(spec.steps + 1):
                    if frame % protocol.calibration_capture_every_steps == 0:
                        # Runtime inference is complete before any privileged read.
                        result = runtime.observe(observation.front_rgb)
                        privileged = integration.privileged.read()
                        silhouettes = {
                            item.object_kind: item
                            for item in integration.projection_validation.sample_object_silhouettes(
                                ("duckie",)
                            )
                        }
                        silhouette = silhouettes.get("duckie")
                        gt_box = None if silhouette is None else silhouette.bounding_box
                        visible_pixels = (
                            0 if silhouette is None else silhouette.visible_pixel_count
                        )
                        decision = assess_silhouette(
                            class_id=yolo_class_id(ObjectClass.DUCKIE),
                            box=gt_box,
                            visible_pixel_count=visible_pixels,
                            image_width_px=protocol.image_width_px,
                            image_height_px=protocol.image_height_px,
                            rules=annotation.duckie_rules,
                        )
                        truth = privileged.true_pomdp_state.pedestrian
                        if truth.range_m is None or truth.bearing_rad is None:
                            raise RuntimeError("calibration frame lacks pedestrian truth")
                        selected = result.selected_duckie
                        selected_box = None if selected is None else selected.bounding_box
                        iou = (
                            None
                            if not decision.accepted
                            or gt_box is None
                            or selected_box is None
                            else intersection_over_union(gt_box, selected_box)
                        )
                        correct = bool(
                            result.pedestrian.detected
                            and decision.accepted
                            and iou is not None
                            and iou >= protocol.matching_iou_threshold
                        )
                        raw_range = result.pedestrian.range_m
                        raw_bearing = result.pedestrian.bearing_rad
                        pose = privileged.ego_world_pose
                        row: dict[str, object] = {
                            "episode": episode,
                            "seed": seed,
                            "scenario": spec.name,
                            "opportunity_kind": "natural_scene",
                            "pedestrian_mode": spec.pedestrian_mode.value,
                            "frame": frame,
                            "timestamp_s": frame / 30.0,
                            "eligible_visible": decision.accepted,
                            "visibility_reason": decision.reason,
                            "visible_pixel_count": visible_pixels,
                            "duckie_detection_count": result.duckie_detection_count,
                            "duplicate_selection": result.duplicate_selection,
                            "selected_confidence": _optional(
                                None if selected is None else selected.confidence
                            ),
                            "metric_measurement_present": result.pedestrian.detected,
                            "projection_error": _optional(result.projection_error),
                            "selected_correct_iou50": correct,
                            "selected_iou": _optional(iou),
                            "stop_sign_detection_count": len(
                                result.stop_sign_detections
                            ),
                            "gt_range_m": truth.range_m,
                            "gt_bearing_rad": truth.bearing_rad,
                            "raw_range_m": _optional(raw_range),
                            "raw_bearing_rad": _optional(raw_bearing),
                            "range_error_m": _optional(
                                raw_range - truth.range_m if correct else None
                            ),
                            "bearing_error_rad": _optional(
                                wrap_angle(raw_bearing - truth.bearing_rad)
                                if correct and raw_bearing is not None
                                else None
                            ),
                            "distance_bin": protocol.distance_bin(truth.range_m),
                            "fov_region": _fov_region(
                                gt_box if decision.accepted else None,
                                protocol.image_width_px,
                            ),
                            "ego_x_m": pose.x_m,
                            "ego_z_m": pose.z_m,
                            "ego_heading_rad": pose.heading_rad,
                            "v_cmd_mps": spec.action.linear_velocity_mps,
                            "omega_cmd_rad_s": spec.action.angular_velocity_rad_s,
                        }
                        row.update(_box_fields("gt_bbox", gt_box))
                        row.update(_box_fields("selected_bbox", selected_box))
                        rows.append(row)

                        if frame % protocol.negative_capture_every_steps == 0:
                            # Privileged rendering creates an auditable
                            # class-negative RGB frame for offline P_FA
                            # calibration. The detector still receives RGB
                            # only, and this path is absent from F9 runtime.
                            negative_rgb = (
                                integration.projection_validation.render_without_objects(
                                    ("duckie",)
                                )
                            )
                            negative_result = runtime.observe(negative_rgb)
                            negative_selected = negative_result.selected_duckie
                            negative_box = (
                                None
                                if negative_selected is None
                                else negative_selected.bounding_box
                            )
                            negative_row = dict(row)
                            negative_row.update(
                                {
                                    "opportunity_kind": (
                                        "counterfactual_duckie_hidden"
                                    ),
                                    "eligible_visible": False,
                                    "visibility_reason": (
                                        "counterfactual_duckie_hidden"
                                    ),
                                    "visible_pixel_count": 0,
                                    "duckie_detection_count": (
                                        negative_result.duckie_detection_count
                                    ),
                                    "duplicate_selection": (
                                        negative_result.duplicate_selection
                                    ),
                                    "selected_confidence": _optional(
                                        None
                                        if negative_selected is None
                                        else negative_selected.confidence
                                    ),
                                    "metric_measurement_present": (
                                        negative_result.pedestrian.detected
                                    ),
                                    "projection_error": _optional(
                                        negative_result.projection_error
                                    ),
                                    "selected_correct_iou50": False,
                                    "selected_iou": "",
                                    "stop_sign_detection_count": len(
                                        negative_result.stop_sign_detections
                                    ),
                                    "raw_range_m": _optional(
                                        negative_result.pedestrian.range_m
                                    ),
                                    "raw_bearing_rad": _optional(
                                        negative_result.pedestrian.bearing_rad
                                    ),
                                    "range_error_m": "",
                                    "bearing_error_rad": "",
                                    "fov_region": "outside",
                                }
                            )
                            negative_row.update(_box_fields("gt_bbox", None))
                            negative_row.update(
                                _box_fields("selected_bbox", negative_box)
                            )
                            rows.append(negative_row)

                    if frame == spec.steps:
                        break
                    transition = integration.agent.step(spec.action)
                    observation = transition.observation
                    if transition.terminated or transition.truncated:
                        break
            finally:
                integration.close()
            print(f"completed {episode}: total_rows={len(rows)}", flush=True)
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty F9 calibration CSV")
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
    protocol = load_f9_protocol(args.config, require_frozen=False)
    if protocol.parameters_frozen:
        raise SystemExit("refusing to overwrite already-frozen F9 calibration")
    rows = collect_calibration_rows(protocol)
    report = summarize_calibration(rows, protocol)
    csv_path = protocol.artifacts["calibration_csv"]
    model_path = protocol.artifacts["calibration_model_json"]
    _write_csv(rows, csv_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "calibration_csv": str(csv_path),
                "measurement_model": str(model_path),
                "measurement_model_sha256": sha256(model_path),
                "opportunities": report["opportunities"],
                "bias_correction": report["bias_correction"],
                "covariance": report["covariance"],
                "existence_observation": report["existence_observation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
