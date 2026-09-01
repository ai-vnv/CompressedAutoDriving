"""Final frozen-test F8a detector and F8b metric-observation evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.detection import (
    YOLO_V1_CLASS_NAMES,
    BoundingBox,
    ObjectClass,
)
from duckie_pomdp.evaluation.yolo_detection import (
    DetectionOpportunity,
    EvaluatedOpportunity,
    evaluate_opportunity,
    summarize_detection,
)
from duckie_pomdp.evaluation.yolo_measurement import (
    MeasurementResidual,
    summarize_measurements,
)
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.measurement_calibration import (
    load_measurement_calibrator,
    wrap_angle,
)
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def optional_number(value: float | int | None) -> float | int | str:
    return "" if value is None else value


def box_or_none(row: dict[str, str]) -> BoundingBox | None:
    if row["object_visible"] != "1":
        return None
    return BoundingBox(
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def make_opportunity(row: dict[str, str]) -> DetectionOpportunity:
    return DetectionOpportunity(
        image_id=row["image_id"],
        episode_id=row["episode_id"],
        seed=int(row["seed"]),
        frame_index=int(row["frame_index"]),
        object_class=ObjectClass(row["object_class"]),
        eligible_visible=row["object_visible"] == "1",
        ground_truth_box=box_or_none(row),
        ground_truth_range_m=float(row["gt_range_origin"]),
        ground_truth_bearing_rad=float(row["gt_bearing"]),
        distance_bin=row["distance_bin"],
        fov_region=row["fov_region"],
    )


def detection_row(evaluation: EvaluatedOpportunity) -> dict[str, object]:
    opportunity = evaluation.opportunity
    gt = opportunity.ground_truth_box
    prediction = evaluation.matched_detection
    pred = prediction.bounding_box if prediction else None
    unmatched = [
        {
            "confidence": detection.confidence,
            "bbox_xyxy": asdict(detection.bounding_box),
            "iou_to_gt": iou,
        }
        for detection, iou in zip(
            evaluation.unmatched_detections,
            evaluation.unmatched_ious,
            strict=True,
        )
    ]
    gt_bottom = gt.bottom_center if gt else None
    pred_bottom = pred.bottom_center if pred else None
    return {
        "episode": opportunity.episode_id,
        "seed": opportunity.seed,
        "frame": opportunity.frame_index,
        "split": "test",
        "image_id": opportunity.image_id,
        "object_class": opportunity.object_class.value,
        "eligible_visible": opportunity.eligible_visible,
        "detector_output_present": bool(prediction or unmatched),
        "detected": evaluation.true_positive,
        "true_positive": evaluation.true_positive,
        "false_negative": evaluation.false_negative,
        "false_positive_count": len(evaluation.unmatched_detections),
        "prediction_count": len(evaluation.unmatched_detections) + int(prediction is not None),
        "confidence": optional_number(prediction.confidence if prediction else None),
        "gt_bbox_x1": optional_number(gt.x_min_px if gt else None),
        "gt_bbox_y1": optional_number(gt.y_min_px if gt else None),
        "gt_bbox_x2": optional_number(gt.x_max_px if gt else None),
        "gt_bbox_y2": optional_number(gt.y_max_px if gt else None),
        "pred_bbox_x1": optional_number(pred.x_min_px if pred else None),
        "pred_bbox_y1": optional_number(pred.y_min_px if pred else None),
        "pred_bbox_x2": optional_number(pred.x_max_px if pred else None),
        "pred_bbox_y2": optional_number(pred.y_max_px if pred else None),
        "iou": optional_number(evaluation.match_iou),
        "gt_bottom_u": optional_number(gt_bottom.x_px if gt_bottom else None),
        "gt_bottom_v": optional_number(gt_bottom.y_px if gt_bottom else None),
        "pred_bottom_u": optional_number(pred_bottom.x_px if pred_bottom else None),
        "pred_bottom_v": optional_number(pred_bottom.y_px if pred_bottom else None),
        "bottom_u_error_px": optional_number(
            pred_bottom.x_px - gt_bottom.x_px if pred_bottom and gt_bottom else None
        ),
        "bottom_v_error_px": optional_number(
            pred_bottom.y_px - gt_bottom.y_px if pred_bottom and gt_bottom else None
        ),
        "bottom_center_error_px": optional_number(
            math.hypot(
                pred_bottom.x_px - gt_bottom.x_px,
                pred_bottom.y_px - gt_bottom.y_px,
            )
            if pred_bottom and gt_bottom
            else None
        ),
        "gt_range": opportunity.ground_truth_range_m,
        "gt_bearing": opportunity.ground_truth_bearing_rad,
        "distance_bin": opportunity.distance_bin,
        "fov_region": opportunity.fov_region,
        "unmatched_predictions_json": json.dumps(unmatched, separators=(",", ":")),
    }


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def load_camera_projector(image_width: int, image_height: int) -> CalibratedGroundProjector:
    scenario = load_scenario(ROOT / "configs" / "scenario_pomdp_v1.toml")
    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            camera_width=image_width,
            camera_height=image_height,
        )
    )
    try:
        integration.agent.reset(seed=scenario.seed)
        calibration = integration.camera_calibration.read()
    finally:
        integration.close()
    return CalibratedGroundProjector(calibration)


def add_missing_strata(metrics: dict[str, object]) -> None:
    for class_metrics in metrics["by_class"].values():
        for key, label in (("by_distance", ("near", "medium", "far")), ("by_fov", ("center", "mid_fov", "edge_fov"))):
            table = class_metrics[key]
            for name in label:
                table.setdefault(
                    name,
                    {
                        "opportunities": 0,
                        "true_positives": 0,
                        "false_negatives": 0,
                        "recall": None,
                        "status": "N/A — no held-out eligible samples",
                    },
                )


def ap_metrics(checkpoint: Path, dataset_yaml: Path, config: dict[str, Any]) -> dict[str, object]:
    from ultralytics import YOLO

    detector_config = config["detector"]
    ap_config = config["average_precision"]
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=int(detector_config["image_size"]),
        batch=8,
        device=detector_config["device"],
        conf=float(ap_config["confidence_floor"]),
        iou=float(detector_config["nms_iou_threshold"]),
        max_det=int(detector_config["max_detections"]),
        plots=False,
        project=str(ROOT / "artifacts" / "yolo_v1" / "final_test_ap"),
        name="frozen_test",
        exist_ok=True,
        verbose=False,
    )
    return {
        "confidence_floor_for_pr_curve": float(ap_config["confidence_floor"]),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class": {
            name: {
                "map50": float(result.box.ap50[index]),
                "map50_95": float(result.box.maps[index]),
            }
            for index, name in enumerate(YOLO_V1_CLASS_NAMES)
        },
    }


def measurement_row(
    evaluation: EvaluatedOpportunity,
    projector: YoloMeasurementProjector,
) -> tuple[dict[str, object], MeasurementResidual | None]:
    base = detection_row(evaluation)
    row = {
        key: base[key]
        for key in (
            "episode",
            "seed",
            "frame",
            "split",
            "image_id",
            "object_class",
            "eligible_visible",
            "detected",
            "confidence",
            "gt_bbox_x1",
            "gt_bbox_y1",
            "gt_bbox_x2",
            "gt_bbox_y2",
            "pred_bbox_x1",
            "pred_bbox_y1",
            "pred_bbox_x2",
            "pred_bbox_y2",
            "iou",
            "gt_bottom_u",
            "gt_bottom_v",
            "pred_bottom_u",
            "pred_bottom_v",
            "gt_range",
            "gt_bearing",
            "distance_bin",
            "fov_region",
        )
    }
    row.update(
        {
            "projection_valid": False,
            "projection_error": "",
            "projection_bottom_u": "",
            "projection_bottom_v": "",
            "projection_pixel_clipped": "",
            "pred_x_left": "",
            "pred_y_forward": "",
            "raw_projected_range": "",
            "f5b_calibrated_range": "",
            "measured_bearing": "",
            "raw_range_error": "",
            "calibrated_range_error": "",
            "bearing_error": "",
        }
    )
    if evaluation.matched_detection is None:
        return row, None
    try:
        projected = projector.project(evaluation.matched_detection)
    except ValueError as error:
        row["projection_error"] = str(error)
        return row, None
    opportunity = evaluation.opportunity
    raw_error = projected.raw_polar.range_m - opportunity.ground_truth_range_m
    calibrated_error = (
        projected.calibrated_polar.range_m - opportunity.ground_truth_range_m
    )
    bearing_error = wrap_angle(
        projected.calibrated_polar.bearing_rad - opportunity.ground_truth_bearing_rad
    )
    row.update(
        {
            "projection_valid": True,
            "projection_bottom_u": projected.projection_pixel.x_px,
            "projection_bottom_v": projected.projection_pixel.y_px,
            "projection_pixel_clipped": projected.pixel_clipped_to_image,
            "pred_x_left": projected.ground_point.x_left_m,
            "pred_y_forward": projected.ground_point.y_forward_m,
            "raw_projected_range": projected.raw_polar.range_m,
            "f5b_calibrated_range": projected.calibrated_polar.range_m,
            "measured_bearing": projected.calibrated_polar.bearing_rad,
            "raw_range_error": raw_error,
            "calibrated_range_error": calibrated_error,
            "bearing_error": bearing_error,
        }
    )
    return row, MeasurementResidual(
        image_id=opportunity.image_id,
        object_class=opportunity.object_class.value,
        distance_bin=opportunity.distance_bin,
        fov_region=opportunity.fov_region,
        confidence=evaluation.matched_detection.confidence,
        raw_range_error_m=raw_error,
        calibrated_range_error_m=calibrated_error,
        bearing_error_rad=bearing_error,
    )


def write_candidate_noise(
    path: Path,
    metrics: dict[str, object],
    detection_metrics: dict[str, object],
    config: dict[str, Any],
    checkpoint_hash: str,
    evaluation_config_hash: str,
) -> dict[str, object]:
    raw_rmse = float(metrics["global"]["raw_range"]["rmse"])
    calibrated_rmse = float(
        metrics["global"]["f5b_calibrated_range"]["rmse"]
    )
    range_key = (
        "raw_range" if raw_rmse <= calibrated_rmse else "f5b_calibrated_range"
    )
    range_value = {
        "raw_range": "raw_projected_yolo_range",
        "f5b_calibrated_range": "f5b_calibrated_yolo_range",
    }[range_key]
    candidate = {
        "schema_version": 1,
        "status": "candidate_for_F9_not_applied_to_frozen_F7",
        "source": {
            "checkpoint_sha256": checkpoint_hash,
            "split": "frozen_test",
            "operating_confidence_threshold": config["detector"]["confidence_threshold"],
            "matching_iou_threshold": config["matching"]["iou_threshold"],
            "evaluation_config_sha256": evaluation_config_hash,
        },
        "measurement": {
            "range_semantics": "object_origin",
            "pixel_reference": "bbox_bottom_center",
            "range_value": range_value,
            "range_variant_key": range_key,
            "f5b_calibration_applied": range_key == "f5b_calibrated_range",
            "selection_rule": "lower_global_RMSE_without_refitting",
            "raw_range_rmse_m": raw_rmse,
            "f5b_calibrated_range_rmse_m": calibrated_rmse,
        },
        "detection_probability": {
            class_name: {
                "global": {
                    "sample_count": class_metrics["opportunities"],
                    "probability": class_metrics["recall"],
                },
                "by_distance": {
                    name: {
                        "sample_count": values["opportunities"],
                        "probability": values["recall"],
                    }
                    for name, values in class_metrics["by_distance"].items()
                },
                "by_fov": {
                    name: {
                        "sample_count": values["opportunities"],
                        "probability": values["recall"],
                    }
                    for name, values in class_metrics["by_fov"].items()
                },
            }
            for class_name, class_metrics in detection_metrics["by_class"].items()
        },
        "false_alarm": {
            class_name: {
                "negative_frame_count": class_metrics["class_negative_frames"],
                "false_positive_count": class_metrics["false_positives_on_negative_frames"],
                "false_positives_per_negative_frame": class_metrics[
                    "false_positives_per_negative_frame"
                ],
                "negative_frame_event_probability": class_metrics[
                    "false_alarm_event_probability"
                ],
            }
            for class_name, class_metrics in detection_metrics["by_class"].items()
        },
        "candidate_covariance": {
            "structure": "diagonal",
            "off_diagonal_enabled": False,
            "global": _noise_entry(metrics["global"], range_key),
            "by_class": {
                class_name: _noise_entry(class_metrics, range_key)
                for class_name, class_metrics in metrics["by_class"].items()
            },
            "by_class_and_distance": {
                class_name: {
                    name: _noise_entry(values, range_key)
                    for name, values in bins.items()
                }
                for class_name, bins in metrics["by_class_and_distance"].items()
            },
        },
        "empirical_range_bearing_residual": metrics["global"][
            "range_bearing_residual"
        ],
        "gaussian_assessment": metrics["global"]["gaussian_assessment"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    return candidate


def _noise_entry(
    metrics: dict[str, object], range_key: str = "f5b_calibrated_range"
) -> dict[str, object]:
    range_metrics = metrics[range_key]
    bearing_metrics = metrics["bearing"]
    return {
        "sample_count": metrics["count"],
        "range_bias_m": range_metrics["bias"],
        "sigma_range_m": range_metrics["residual_sd"],
        "bearing_bias_rad": bearing_metrics["bias"],
        "sigma_bearing_rad": bearing_metrics["residual_sd"],
        "R_diagonal": [
            float(range_metrics["residual_sd"]) ** 2,
            float(bearing_metrics["residual_sd"]) ** 2,
        ],
        "range_gaussian_assessment": metrics["gaussian_assessment"][
            range_key
        ],
        "bearing_gaussian_assessment": metrics["gaussian_assessment"]["bearing"],
    }


def write_candidate_toml(
    path: Path,
    noise: dict[str, object],
    metrics: dict[str, object],
    checkpoint_hash: str,
) -> None:
    global_noise = noise["candidate_covariance"]["global"]
    measurement = noise["measurement"]
    lines = [
        "schema_version = 1",
        'status = "candidate_for_F9_not_applied_to_frozen_F7"',
        "",
        "[source]",
        f'checkpoint_sha256 = "{checkpoint_hash}"',
        'split = "frozen_test"',
        'evaluation_config_sha256 = '
        f'"{noise["source"]["evaluation_config_sha256"]}"',
        "",
        "[measurement]",
        'range_semantics = "object_origin"',
        'pixel_reference = "bbox_bottom_center"',
        f'range_value = "{measurement["range_value"]}"',
        f'f5b_calibration_applied = {str(measurement["f5b_calibration_applied"]).lower()}',
        f'selection_rule = "{measurement["selection_rule"]}"',
        f"raw_range_rmse_m = {measurement['raw_range_rmse_m']!r}",
        "f5b_calibrated_range_rmse_m = "
        f"{measurement['f5b_calibrated_range_rmse_m']!r}",
        "",
        "[covariance]",
        'structure = "diagonal"',
        "use_off_diagonal = false",
        f"empirical_correlation = {metrics['global']['range_bearing_residual']['correlation']!r}",
        f"empirical_covariance_m_rad = {metrics['global']['range_bearing_residual']['covariance_m_rad']!r}",
        "",
        "[noise.global]",
        f"sample_count = {global_noise['sample_count']}",
        f"range_bias_m = {global_noise['range_bias_m']!r}",
        f"sigma_range_m = {global_noise['sigma_range_m']!r}",
        f"bearing_bias_rad = {global_noise['bearing_bias_rad']!r}",
        f"sigma_bearing_rad = {global_noise['sigma_bearing_rad']!r}",
    ]
    for class_name, class_noise in noise["candidate_covariance"]["by_class"].items():
        lines.extend(
            [
                "",
                f"[noise.{class_name}]",
                f"sample_count = {class_noise['sample_count']}",
                f"range_bias_m = {class_noise['range_bias_m']!r}",
                f"sigma_range_m = {class_noise['sigma_range_m']!r}",
                f"bearing_bias_rad = {class_noise['bearing_bias_rad']!r}",
                f"sigma_bearing_rad = {class_noise['sigma_bearing_rad']!r}",
            ]
        )
    for class_name, distance_bins in noise["candidate_covariance"][
        "by_class_and_distance"
    ].items():
        for distance_bin, bin_noise in distance_bins.items():
            lines.extend(
                [
                    "",
                    f"[noise.{class_name}.by_distance.{distance_bin}]",
                    f"sample_count = {bin_noise['sample_count']}",
                    f"range_bias_m = {bin_noise['range_bias_m']!r}",
                    f"sigma_range_m = {bin_noise['sigma_range_m']!r}",
                    f"bearing_bias_rad = {bin_noise['bearing_bias_rad']!r}",
                    f"sigma_bearing_rad = {bin_noise['sigma_bearing_rad']!r}",
                ]
            )
    for class_name, detection in noise["detection_probability"].items():
        lines.extend(
            [
                "",
                f"[detection.{class_name}]",
                f"sample_count = {detection['global']['sample_count']}",
                f"probability = {detection['global']['probability']!r}",
            ]
        )
        for distance_bin, values in detection["by_distance"].items():
            probability = values["probability"]
            if probability is None:
                continue
            lines.extend(
                [
                    "",
                    f"[detection.{class_name}.by_distance.{distance_bin}]",
                    f"sample_count = {values['sample_count']}",
                    f"probability = {probability!r}",
                ]
            )
        for fov_region, values in detection["by_fov"].items():
            probability = values["probability"]
            if probability is None:
                continue
            lines.extend(
                [
                    "",
                    f"[detection.{class_name}.by_fov.{fov_region}]",
                    f"sample_count = {values['sample_count']}",
                    f"probability = {probability!r}",
                ]
            )
        false_alarm = noise["false_alarm"][class_name]
        lines.extend(
            [
                "",
                f"[false_alarm.{class_name}]",
                f"negative_frame_count = {false_alarm['negative_frame_count']}",
                f"false_positive_count = {false_alarm['false_positive_count']}",
                "false_positives_per_negative_frame = "
                f"{false_alarm['false_positives_per_negative_frame']!r}",
                "negative_frame_event_probability = "
                f"{false_alarm['negative_frame_event_probability']!r}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_error_cases(
    evaluations: list[EvaluatedOpportunity],
    measurement_rows: list[dict[str, object]],
    frames_by_id: dict[str, dict[str, str]],
    dataset_root: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> list[str]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if output_dir.name != "yolo_error_cases" or output_dir.parent.name != "artifacts":
            raise RuntimeError(f"refusing to replace unexpected path: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    measurement_by_key = {
        (row["image_id"], row["object_class"]): row for row in measurement_rows
    }
    selected: list[tuple[str, EvaluatedOpportunity]] = []
    for class_name in YOLO_V1_CLASS_NAMES:
        class_rows = [
            row for row in evaluations if row.opportunity.object_class.value == class_name
        ]
        matched = [
            row
            for row in class_rows
            if row.true_positive
            and measurement_by_key[(row.opportunity.image_id, class_name)][
                "projection_valid"
            ]
        ]
        matched.sort(
            key=lambda row: abs(
                float(
                    measurement_by_key[
                        (row.opportunity.image_id, class_name)
                    ]["calibrated_range_error"]
                )
            ),
            reverse=True,
        )
        selected.extend(
            ("range_outlier", row)
            for row in matched[: int(config["outliers"]["worst_matched_per_class"])]
        )
        failures = [
            row
            for row in class_rows
            if row.false_negative
            or (not row.opportunity.eligible_visible and row.unmatched_detections)
        ]
        failures.sort(
            key=lambda row: max(
                (detection.confidence for detection in row.unmatched_detections),
                default=0.0,
            ),
            reverse=True,
        )
        selected.extend(
            ("detection_failure", row)
            for row in failures[
                : int(config["outliers"]["worst_false_or_missed_per_class"])
            ]
        )

    outputs: list[str] = []
    for index, (category, evaluation) in enumerate(selected):
        opportunity = evaluation.opportunity
        image_path = dataset_root / frames_by_id[opportunity.image_id]["image_path"]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        gt = opportunity.ground_truth_box
        if gt is not None:
            draw.rectangle(
                (gt.x_min_px, gt.y_min_px, gt.x_max_px, gt.y_max_px),
                outline=(0, 255, 0),
                width=3,
            )
            draw.ellipse(
                (
                    gt.bottom_center.x_px - 4,
                    gt.bottom_center.y_px - 4,
                    gt.bottom_center.x_px + 4,
                    gt.bottom_center.y_px + 4,
                ),
                fill=(0, 255, 0),
            )
        if evaluation.matched_detection is not None:
            prediction = evaluation.matched_detection
            box = prediction.bounding_box
            draw.rectangle(
                (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                outline=(255, 0, 0),
                width=3,
            )
            draw.ellipse(
                (
                    box.bottom_center.x_px - 4,
                    box.bottom_center.y_px - 4,
                    box.bottom_center.x_px + 4,
                    box.bottom_center.y_px + 4,
                ),
                fill=(255, 0, 0),
            )
        for prediction in evaluation.unmatched_detections:
            box = prediction.bounding_box
            draw.rectangle(
                (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                outline=(255, 165, 0),
                width=2,
            )
        measurement = measurement_by_key[(opportunity.image_id, opportunity.object_class.value)]
        label = (
            f"{category} {opportunity.object_class.value} "
            f"conf={measurement['confidence']} iou={measurement['iou']} "
            f"range_err={measurement['calibrated_range_error']}"
        )
        draw.rectangle((0, 0, min(image.width, 620), 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        destination = output_dir / (
            f"{index:02d}_{category}_{opportunity.object_class.value}_"
            f"{opportunity.image_id}.png"
        )
        image.save(destination)
        outputs.append(str(destination.relative_to(ROOT)))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "yolo_evaluation_v1.toml",
    )
    args = parser.parse_args()
    config = read_toml(args.config)
    evaluation_config_path = args.config.resolve()
    evaluation_config_hash = sha256(evaluation_config_path)
    checkpoint = resolve(config["checkpoint"]["path"])
    actual_hash = sha256(checkpoint)
    expected_hash = config["checkpoint"]["sha256"]
    if actual_hash != expected_hash:
        raise SystemExit(
            f"frozen checkpoint hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    model_manifest_path = ROOT / "artifacts" / "yolo_v1" / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest["checkpoint_sha256"] != actual_hash:
        raise SystemExit("checkpoint and model manifest hashes disagree")
    dataset_manifest_path = resolve(config["dataset"]["manifest"])
    if sha256(dataset_manifest_path) != model_manifest["dataset_manifest_sha256"]:
        raise SystemExit("frozen dataset manifest hash differs from model provenance")
    if tuple(config["checkpoint"]["class_names"]) != YOLO_V1_CLASS_NAMES:
        raise SystemExit("evaluation class mapping differs from Version 1")

    dataset_root = resolve(config["dataset"]["root"])
    frames = [
        row
        for row in read_csv(dataset_root / "metadata" / "frames.csv")
        if row["split"] == config["dataset"]["split"]
    ]
    if len(frames) != int(config["dataset"]["expected_images"]):
        raise SystemExit(f"unexpected frozen test image count: {len(frames)}")
    frames.sort(key=lambda row: row["image_id"])
    frames_by_id = {row["image_id"]: row for row in frames}

    detector_config = config["detector"]
    detector = YoloObjectDetector(
        checkpoint,
        confidence_threshold=float(detector_config["confidence_threshold"]),
        iou_threshold=float(detector_config["nms_iou_threshold"]),
        image_size=int(detector_config["image_size"]),
        device=detector_config["device"],
        max_detections=int(detector_config["max_detections"]),
    )

    # Inference is deliberately completed before privileged offline annotation
    # metadata is read. The detector receives each front RGB frame and nothing else.
    predictions_by_image: dict[str, tuple] = {}
    inference_started = time.perf_counter()
    for frame in frames:
        image_path = dataset_root / frame["image_path"]
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        predictions_by_image[frame["image_id"]] = tuple(detector.detect(rgb))
    inference_seconds = time.perf_counter() - inference_started

    object_rows = [
        row
        for row in read_csv(dataset_root / "metadata" / "objects.csv")
        if row["split"] == config["dataset"]["split"]
    ]
    object_rows.sort(key=lambda row: (row["image_id"], int(row["class_id"])))
    evaluations = [
        evaluate_opportunity(
            make_opportunity(row),
            predictions_by_image[row["image_id"]],
            iou_threshold=float(config["matching"]["iou_threshold"]),
        )
        for row in object_rows
    ]
    if len(evaluations) != len(frames) * len(YOLO_V1_CLASS_NAMES):
        raise RuntimeError("test metadata lacks one opportunity per frame and class")

    artifact_config = config["artifacts"]
    detection_rows = [detection_row(evaluation) for evaluation in evaluations]
    write_csv(detection_rows, resolve(artifact_config["detection_csv"]))
    detection_metrics: dict[str, Any] = {
        "gate": "F8a",
        "split": "frozen_test",
        "checkpoint_sha256": actual_hash,
        "evaluation_config_path": str(evaluation_config_path.relative_to(ROOT)),
        "evaluation_config_sha256": evaluation_config_hash,
        "test_images": len(frames),
        "evaluation_config": {
            "confidence_threshold": detector_config["confidence_threshold"],
            "nms_iou_threshold": detector_config["nms_iou_threshold"],
            "matching_iou_threshold": config["matching"]["iou_threshold"],
            "image_size": detector_config["image_size"],
            "device": detector_config["device"],
            "max_detections": detector_config["max_detections"],
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "torch_version": torch.__version__,
        },
        "inference": {
            "input": "uint8 front RGB only",
            "total_seconds": inference_seconds,
            "mean_ms_per_frame": inference_seconds * 1000.0 / len(frames),
        },
        "by_class": {
            class_name: summarize_detection(
                row
                for row in evaluations
                if row.opportunity.object_class.value == class_name
            )
            for class_name in YOLO_V1_CLASS_NAMES
        },
        "average_precision": ap_metrics(
            checkpoint, dataset_root / "dataset.yaml", config
        ),
    }
    add_missing_strata(detection_metrics)
    detection_metrics_path = resolve(artifact_config["detection_metrics_json"])
    detection_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    detection_metrics_path.write_text(
        json.dumps(detection_metrics, indent=2) + "\n", encoding="utf-8"
    )

    width = int(frames[0]["image_width_px"])
    height = int(frames[0]["image_height_px"])
    metric_projector = YoloMeasurementProjector(
        load_camera_projector(width, height),
        load_measurement_calibrator(resolve(config["measurement"]["f5b_model"])),
    )
    measurement_outputs = [
        measurement_row(evaluation, metric_projector) for evaluation in evaluations
    ]
    measurement_rows = [row for row, _ in measurement_outputs]
    residuals = [residual for _, residual in measurement_outputs if residual is not None]
    write_csv(measurement_rows, resolve(artifact_config["measurement_csv"]))
    measurement_metrics = summarize_measurements(residuals)
    f5b = json.loads(
        (ROOT / "artifacts" / "measurement_noise_v1.json").read_text(encoding="utf-8")
    )
    f5b_range = f5b["measurement_noise"]["range_global"]
    f5b_bearing = f5b["measurement_noise"]["bearing_global"]
    yolo_raw_range = measurement_metrics["global"]["raw_range"]
    yolo_range = measurement_metrics["global"]["f5b_calibrated_range"]
    yolo_bearing = measurement_metrics["global"]["bearing"]
    measurement_metrics.update(
        {
            "gate": "F8b",
            "split": "frozen_test",
            "checkpoint_sha256": actual_hash,
            "evaluation_config_path": str(evaluation_config_path.relative_to(ROOT)),
            "evaluation_config_sha256": evaluation_config_hash,
            "matched_detection_count": len(residuals),
            "projection_failure_count": sum(
                row["detected"] and not row["projection_valid"]
                for row in measurement_rows
            ),
            "canonical_range_semantics": "object_origin",
            "pixel_reference": "bbox_bottom_center; clamped only to calibrated image bounds",
            "f5b_baseline": {
                "calibrated_range": f5b_range,
                "bearing": f5b_bearing,
            },
            "degradation_vs_f5b": {
                "raw_range_rmse_ratio": yolo_raw_range["rmse"]
                / f5b_range["rmse"],
                "f5b_calibrated_range_rmse_ratio": yolo_range["rmse"]
                / f5b_range["rmse"],
                "bearing_rmse_ratio": yolo_bearing["rmse"] / f5b_bearing["rmse"],
            },
            "f5b_calibration_effect_on_yolo": {
                "raw_range_rmse": measurement_metrics["global"]["raw_range"]["rmse"],
                "calibrated_range_rmse": yolo_range["rmse"],
                "rmse_ratio_calibrated_over_raw": yolo_range["rmse"]
                / measurement_metrics["global"]["raw_range"]["rmse"],
                "improves_rmse": yolo_range["rmse"]
                < measurement_metrics["global"]["raw_range"]["rmse"],
            },
            "detection_context": {
                class_name: {
                    "opportunities": values["opportunities"],
                    "probability_of_detection": values["recall"],
                    "false_alarm_event_probability": values[
                        "false_alarm_event_probability"
                    ],
                }
                for class_name, values in detection_metrics["by_class"].items()
            },
        }
    )
    error_cases = render_error_cases(
        evaluations,
        measurement_rows,
        frames_by_id,
        dataset_root,
        resolve(artifact_config["error_case_dir"]),
        config,
    )
    measurement_metrics["error_case_visualizations"] = error_cases
    measurement_metrics_path = resolve(artifact_config["measurement_metrics_json"])
    measurement_metrics_path.write_text(
        json.dumps(measurement_metrics, indent=2) + "\n", encoding="utf-8"
    )
    candidate = write_candidate_noise(
        resolve(artifact_config["measurement_noise_json"]),
        measurement_metrics,
        detection_metrics,
        config,
        actual_hash,
        evaluation_config_hash,
    )
    write_candidate_toml(
        resolve(artifact_config["measurement_model_toml"]),
        candidate,
        measurement_metrics,
        actual_hash,
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": actual_hash,
                "test_images": len(frames),
                "detection": detection_metrics["by_class"],
                "average_precision": detection_metrics["average_precision"],
                "measurement_global": measurement_metrics["global"],
                "degradation_vs_f5b": measurement_metrics["degradation_vs_f5b"],
                "projection_failures": measurement_metrics["projection_failure_count"],
                "error_case_count": len(error_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
