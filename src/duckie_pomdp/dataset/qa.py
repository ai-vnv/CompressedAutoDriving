"""Automated and visual QA for the generated detection dataset."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from duckie_pomdp.dataset.annotations import YoloBox
from duckie_pomdp.dataset.config import DetectionDatasetConfig
from duckie_pomdp.dataset.split import assert_no_split_leakage
from duckie_pomdp.domain.detection import YOLO_V1_CLASS_NAMES


def validate_detection_dataset(config: DetectionDatasetConfig) -> dict[str, Any]:
    root = config.output_root
    frames = _read_csv(root / "metadata" / "frames.csv")
    objects = _read_csv(root / "metadata" / "objects.csv")
    if not frames:
        raise ValueError("dataset contains no frames")
    assert_no_split_leakage(frames)
    _validate_dataset_yaml(root / "dataset.yaml")
    _validate_frame_files_and_labels(config, frames)
    _validate_metadata(frames, objects)
    _validate_cross_split_image_hashes(frames)

    object_rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in objects:
        object_rows_by_image[row["image_id"]].append(row)
    for frame in frames:
        opportunities = object_rows_by_image[frame["image_id"]]
        if {row["object_class"] for row in opportunities} != set(
            YOLO_V1_CLASS_NAMES
        ):
            raise ValueError(
                f"frame {frame['image_id']} does not contain both object opportunities"
            )

    stats = _statistics(frames, objects)
    # The fixed V1 layout naturally supplies both, Duckie-only, and neither
    # frames. A clean stop-sign-only view is optional: accepting a tiny pole
    # fragment at the border merely to populate that category would weaken the
    # annotation semantics.
    required_categories = {"both", "duckie_only", "neither"}
    observed_categories = set(stats["visibility_categories"])
    missing = sorted(required_categories - observed_categories)
    if missing:
        raise ValueError(f"dataset lacks required visibility categories: {missing}")
    for split in ("train", "val", "test"):
        if stats["splits"][split]["negative_frames"] == 0:
            raise ValueError(f"split {split} contains no negative frames")
        for class_name in YOLO_V1_CLASS_NAMES:
            if stats["splits"][split]["boxes"][class_name] == 0:
                raise ValueError(f"split {split} contains no {class_name} boxes")

    qa_images = _write_visual_samples(config, frames, object_rows_by_image)
    stats.update(
        {
            "dataset_version": config.version,
            "qa_passed": True,
            "split_leakage": False,
            "cross_split_duplicate_images": 0,
            "visual_qa_images": [str(path) for path in qa_images],
        }
    )
    config.artifact_stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifact_stats_path.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def _validate_frame_files_and_labels(config, frames) -> None:
    for row in frames:
        image_path = config.output_root / row["image_path"]
        label_path = config.output_root / row["label_path"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing image/label pair for {row['image_id']}")
        with Image.open(image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"image is not RGB: {image_path}")
            if image.size != (config.image_width_px, config.image_height_px):
                raise ValueError(f"inconsistent image size: {image_path} {image.size}")
        labels = [
            YoloBox.parse(line)
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(labels) != int(row["label_count"]):
            raise ValueError(f"label count mismatch for {row['image_id']}")


def _validate_metadata(frames, objects) -> None:
    frame_numeric = (
        "seed",
        "frame_index",
        "timestamp_s",
        "image_width_px",
        "image_height_px",
        "ego_x",
        "ego_z",
        "ego_heading",
        "label_count",
    )
    object_numeric = (
        "seed",
        "frame_index",
        "class_id",
        "object_present_world",
        "silhouette_visible",
        "object_visible",
        "visible_pixel_count",
        "border_touches",
        "gt_range_origin",
        "gt_bearing",
    )
    for row in frames:
        _require_finite(row, frame_numeric)
    for row in objects:
        _require_finite(row, object_numeric)
        if int(row["class_id"]) not in (0, 1):
            raise ValueError("metadata contains an unsupported class id")
        if row["object_class"] != YOLO_V1_CLASS_NAMES[int(row["class_id"])]:
            raise ValueError("metadata class id/name mismatch")
        bbox_values = tuple(row[name] for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        if any(bbox_values) and not all(bbox_values):
            raise ValueError("metadata contains a partially missing bounding box")
        if all(bbox_values):
            values = tuple(float(value) for value in bbox_values)
            if not all(isfinite(value) for value in values):
                raise ValueError("metadata bounding box is non-finite")


def _require_finite(row, fields) -> None:
    for field in fields:
        value = float(row[field])
        if not isfinite(value):
            raise ValueError(f"metadata field {field} is non-finite")


def _validate_cross_split_image_hashes(frames) -> None:
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in frames:
        hash_splits[row["image_sha256"]].add(row["split"])
    duplicates = {
        digest: sorted(splits)
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }
    if duplicates:
        raise ValueError(f"identical images occur across splits: {duplicates}")


def _validate_dataset_yaml(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    normalized = {int(key): str(value) for key, value in names.items()}
    expected = {index: name for index, name in enumerate(YOLO_V1_CLASS_NAMES)}
    if normalized != expected:
        raise ValueError(f"dataset class mapping mismatch: {normalized} != {expected}")


def _statistics(frames, objects) -> dict[str, Any]:
    split_stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        selected_frames = [row for row in frames if row["split"] == split]
        selected_objects = [
            row
            for row in objects
            if row["split"] == split and int(row["object_visible"]) == 1
        ]
        split_stats[split] = {
            "images": len(selected_frames),
            "boxes": {
                name: sum(row["object_class"] == name for row in selected_objects)
                for name in YOLO_V1_CLASS_NAMES
            },
            "negative_frames": sum(
                int(row["label_count"]) == 0 for row in selected_frames
            ),
            "visibility_categories": dict(
                Counter(row["visibility_category"] for row in selected_frames)
            ),
        }

    visible = [row for row in objects if int(row["object_visible"]) == 1]
    return {
        "splits": split_stats,
        "visibility_categories": dict(
            Counter(row["visibility_category"] for row in frames)
        ),
        "box_distribution_by_distance": _nested_counts(
            visible, "object_class", "distance_bin"
        ),
        "box_distribution_by_fov": _nested_counts(
            visible, "object_class", "fov_region"
        ),
        "box_distribution_by_pedestrian_trajectory": _nested_counts(
            visible, "object_class", "pedestrian_mode"
        ),
        "annotation_decisions": _nested_counts(
            objects, "object_class", "annotation_decision"
        ),
    }


def _nested_counts(rows, first, second) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[row[first]][row[second]] += 1
    return {key: dict(value) for key, value in sorted(result.items())}


def _write_visual_samples(config, frames, objects_by_image) -> list[Path]:
    root = config.qa_output_root.resolve()
    if root.exists():
        if root.name != "dataset_qa" or root.parent.name != "artifacts":
            raise RuntimeError(f"refusing to replace unexpected QA path: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    ordered = sorted(frames, key=lambda row: row["image_id"])
    selected: list[dict[str, str]] = []
    seen_groups: set[tuple[str, str]] = set()
    for row in ordered:
        group = (row["split"], row["visibility_category"])
        if group not in seen_groups:
            selected.append(row)
            seen_groups.add(group)
    for row in ordered:
        if len(selected) >= config.visual_qa_samples:
            break
        if row not in selected:
            selected.append(row)
    selected = selected[: config.visual_qa_samples]

    outputs: list[Path] = []
    colors = {"stop_sign": "#ff3030", "duckie": "#00d7ff"}
    for row in selected:
        source = config.output_root / row["image_path"]
        with Image.open(source) as image:
            annotated = image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for obj in objects_by_image[row["image_id"]]:
            if int(obj["object_visible"]) != 1:
                continue
            box = tuple(
                float(obj[name])
                for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
            )
            label = obj["object_class"]
            draw.rectangle(box, outline=colors[label], width=3)
            draw.text((box[0] + 2, box[1] + 2), label, fill=colors[label])
        destination = root / f"{row['image_id']}_qa.png"
        annotated.save(destination)
        outputs.append(destination)
    return outputs


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
