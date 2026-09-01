"""Generate an auditable YOLO dataset from real Gym-Duckietown renders."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from math import hypot
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    ObjectSilhouetteAnnotation,
    create_gym_duckietown,
)
from duckie_pomdp.dataset.annotations import assess_silhouette
from duckie_pomdp.dataset.config import DetectionDatasetConfig
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import (
    ObjectClass,
    YOLO_V1_CLASS_NAMES,
    yolo_class_id,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


FRAME_FIELDS = (
    "image_id",
    "episode_id",
    "seed",
    "split",
    "frame_index",
    "timestamp_s",
    "image_path",
    "label_path",
    "image_sha256",
    "image_width_px",
    "image_height_px",
    "ego_x",
    "ego_z",
    "ego_heading",
    "pedestrian_mode",
    "scenario",
    "map",
    "label_count",
    "visibility_category",
)

OBJECT_FIELDS = (
    "image_id",
    "episode_id",
    "seed",
    "split",
    "frame_index",
    "object_class",
    "class_id",
    "object_present_world",
    "silhouette_visible",
    "object_visible",
    "visible_pixel_count",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "border_touches",
    "annotation_decision",
    "gt_range_origin",
    "gt_bearing",
    "distance_bin",
    "fov_region",
    "pedestrian_mode",
)


def generate_detection_dataset(
    config: DetectionDatasetConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    _prepare_output_root(config, overwrite=overwrite)
    _create_dataset_directories(config.output_root)
    base_scenario = load_scenario(config.scenario_path)
    frame_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []

    for split, seeds in config.split_seeds.items():
        for seed in seeds:
            for mode_index, mode in enumerate(config.pedestrian_modes):
                episode_id = f"{split}_{seed}_{mode.value}"
                rng = np.random.default_rng(seed + 100_003 * mode_index)
                scenario = _episode_scenario(base_scenario, config, seed, mode, rng)
                velocity = float(rng.uniform(*config.velocity_range_mps))
                integration = create_gym_duckietown(
                    GymDuckietownConfig(
                        scenario=scenario,
                        domain_randomization=config.domain_randomization,
                        dynamics_randomization=False,
                        maximum_steps=config.maximum_steps + 2,
                        camera_width=config.image_width_px,
                        camera_height=config.image_height_px,
                    )
                )
                try:
                    observation = integration.agent.reset(seed=seed)
                    last_capture_pose: tuple[float, float] | None = None
                    for step in range(config.maximum_steps + 1):
                        if step % config.capture_every_steps == 0:
                            privileged = integration.privileged.read()
                            pose = privileged.ego_world_pose
                            moved = (
                                last_capture_pose is None
                                or hypot(
                                    pose.x_m - last_capture_pose[0],
                                    pose.z_m - last_capture_pose[1],
                                )
                                >= config.minimum_capture_translation_m
                            )
                            if moved:
                                _capture_frame(
                                    config=config,
                                    integration=integration,
                                    observation=observation,
                                    privileged=privileged,
                                    split=split,
                                    seed=seed,
                                    episode_id=episode_id,
                                    frame_index=step,
                                    mode=mode,
                                    frame_rows=frame_rows,
                                    object_rows=object_rows,
                                )
                                last_capture_pose = (pose.x_m, pose.z_m)

                        if step == config.maximum_steps:
                            break
                        transition = integration.agent.step(
                            PolicyAction(velocity, 0.0)
                        )
                        observation = transition.observation
                        if transition.terminated or transition.truncated:
                            break
                finally:
                    integration.close()

    metadata_root = config.output_root / "metadata"
    _write_csv(metadata_root / "frames.csv", FRAME_FIELDS, frame_rows)
    _write_csv(metadata_root / "objects.csv", OBJECT_FIELDS, object_rows)
    _write_dataset_yaml(config.output_root)
    manifest = _build_manifest(config, frame_rows, object_rows)
    _write_json(config.output_root / "manifest.json", manifest)
    _write_json(config.artifact_manifest_path, manifest)
    return manifest


def _episode_scenario(base, config, seed, mode, rng):
    scenario = base.with_pedestrian_mode(mode)
    base_pose = scenario.ego_start_pose_m
    start_x = float(rng.uniform(*config.start_x_range_m))
    lateral_offset = float(rng.uniform(*config.lateral_offset_range_m))
    heading = float(rng.uniform(*config.heading_range_rad))
    return replace(
        scenario,
        seed=seed,
        ego_start_pose_m=(start_x, base_pose[1], base_pose[2] + lateral_offset),
        ego_heading_rad=heading,
    )


def _capture_frame(
    *,
    config,
    integration,
    observation,
    privileged,
    split,
    seed,
    episode_id,
    frame_index,
    mode,
    frame_rows,
    object_rows,
) -> None:
    image_id = f"{episode_id}_{frame_index:04d}"
    image_relative = Path("images") / split / f"{image_id}.png"
    label_relative = Path("labels") / split / f"{image_id}.txt"
    image_path = config.output_root / image_relative
    label_path = config.output_root / label_relative

    rgb = np.asarray(observation.front_rgb)
    expected_shape = (config.image_height_px, config.image_width_px, 3)
    if rgb.dtype != np.uint8 or rgb.shape != expected_shape:
        raise RuntimeError(
            f"unexpected simulator RGB format: {rgb.dtype} {rgb.shape}, "
            f"expected uint8 {expected_shape}"
        )
    Image.fromarray(rgb, mode="RGB").save(image_path)

    silhouettes = {
        sample.object_kind: sample
        for sample in integration.projection_validation.sample_object_silhouettes(
            ("sign_stop", "duckie")
        )
    }
    labels = []
    accepted_classes: set[ObjectClass] = set()
    state = privileged.true_pomdp_state
    for object_class, kind, object_state, world_position in (
        (
            ObjectClass.STOP_SIGN,
            "sign_stop",
            state.stop_sign,
            privileged.stop_sign_world_position,
        ),
        (
            ObjectClass.DUCKIE,
            "duckie",
            state.pedestrian,
            privileged.pedestrian_world_position,
        ),
    ):
        silhouette: ObjectSilhouetteAnnotation | None = silhouettes.get(kind)
        box = silhouette.bounding_box if silhouette is not None else None
        pixel_count = silhouette.visible_pixel_count if silhouette is not None else 0
        decision = assess_silhouette(
            class_id=yolo_class_id(object_class),
            box=box,
            visible_pixel_count=pixel_count,
            image_width_px=config.image_width_px,
            image_height_px=config.image_height_px,
            rules=config.annotation_rules(object_class.value),
        )
        if decision.yolo_box is not None:
            labels.append(decision.yolo_box)
            accepted_classes.add(object_class)
        gt_range = object_state.range_m
        gt_bearing = object_state.bearing_rad
        if world_position is None or gt_range is None or gt_bearing is None:
            raise RuntimeError(f"missing privileged annotation truth for {object_class.value}")
        object_rows.append(
            {
                "image_id": image_id,
                "episode_id": episode_id,
                "seed": seed,
                "split": split,
                "frame_index": frame_index,
                "object_class": object_class.value,
                "class_id": yolo_class_id(object_class),
                "object_present_world": 1,
                "silhouette_visible": int(silhouette is not None),
                "object_visible": int(decision.accepted),
                "visible_pixel_count": pixel_count,
                "bbox_x1": "" if box is None else box.x_min_px,
                "bbox_y1": "" if box is None else box.y_min_px,
                "bbox_x2": "" if box is None else box.x_max_px,
                "bbox_y2": "" if box is None else box.y_max_px,
                "border_touches": decision.border_touches,
                "annotation_decision": decision.reason,
                "gt_range_origin": gt_range,
                "gt_bearing": gt_bearing,
                "distance_bin": _distance_bin(gt_range),
                "fov_region": _fov_region(box, config.image_width_px),
                "pedestrian_mode": mode.value,
            }
        )

    labels.sort(key=lambda value: value.class_id)
    label_path.write_text(
        "".join(f"{label.to_line()}\n" for label in labels),
        encoding="utf-8",
    )
    pose = privileged.ego_world_pose
    frame_rows.append(
        {
            "image_id": image_id,
            "episode_id": episode_id,
            "seed": seed,
            "split": split,
            "frame_index": frame_index,
            "timestamp_s": frame_index / 30.0,
            "image_path": image_relative.as_posix(),
            "label_path": label_relative.as_posix(),
            "image_sha256": _sha256(image_path),
            "image_width_px": config.image_width_px,
            "image_height_px": config.image_height_px,
            "ego_x": pose.x_m,
            "ego_z": pose.z_m,
            "ego_heading": pose.heading_rad,
            "pedestrian_mode": mode.value,
            "scenario": "pomdp_v1",
            "map": config.scenario_path.name,
            "label_count": len(labels),
            "visibility_category": _visibility_category(accepted_classes),
        }
    )


def _visibility_category(classes: set[ObjectClass]) -> str:
    if classes == {ObjectClass.STOP_SIGN, ObjectClass.DUCKIE}:
        return "both"
    if classes == {ObjectClass.STOP_SIGN}:
        return "stop_sign_only"
    if classes == {ObjectClass.DUCKIE}:
        return "duckie_only"
    return "neither"


def _distance_bin(range_m: float) -> str:
    if range_m < 0.55:
        return "near"
    if range_m < 0.80:
        return "medium"
    return "far"


def _fov_region(box, image_width_px: int) -> str:
    if box is None:
        return "outside"
    center_x = 0.5 * (box.x_min_px + box.x_max_px)
    normalized_offset = abs(center_x - 0.5 * image_width_px) / (0.5 * image_width_px)
    if normalized_offset < 1.0 / 3.0:
        return "center"
    if normalized_offset < 2.0 / 3.0:
        return "mid_fov"
    return "edge_fov"


def _prepare_output_root(config: DetectionDatasetConfig, *, overwrite: bool) -> None:
    root = config.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"dataset output already exists: {root}")
        if root.name != config.version or root.parent.name != "datasets":
            raise RuntimeError(f"refusing to replace unexpected dataset path: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _create_dataset_directories(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(root: Path) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {name}" for class_id, name in enumerate(YOLO_V1_CLASS_NAMES)
    )
    (root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_manifest(config, frame_rows, object_rows) -> dict[str, Any]:
    import gym_duckietown

    split_counts = {
        split: sum(row["split"] == split for row in frame_rows)
        for split in ("train", "val", "test")
    }
    object_counts = {
        split: {
            name: sum(
                row["split"] == split
                and row["object_class"] == name
                and row["object_visible"] == 1
                for row in object_rows
            )
            for name in YOLO_V1_CLASS_NAMES
        }
        for split in ("train", "val", "test")
    }
    metadata_root = config.output_root / "metadata"
    source_hashes = {
        "dataset_config": _sha256(config.config_path),
        "scenario_config": _sha256(config.scenario_path),
        "map": _sha256(load_scenario(config.scenario_path).map_path),
        "dataset_yaml": _sha256(config.output_root / "dataset.yaml"),
        "frames_csv": _sha256(metadata_root / "frames.csv"),
        "objects_csv": _sha256(metadata_root / "objects.csv"),
    }
    return {
        "dataset_version": config.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gym_duckietown_version": gym_duckietown.__version__,
        "generator_git_commit": _git_commit(config.config_path.parent.parent),
        "scenario": "pomdp_v1",
        "map": str(load_scenario(config.scenario_path).map_path),
        "image_resolution": [config.image_width_px, config.image_height_px],
        "class_mapping": {
            str(class_id): name
            for class_id, name in enumerate(YOLO_V1_CLASS_NAMES)
        },
        "split_unit": "episode identified by disjoint seed and pedestrian mode",
        "split_seeds": {
            split: list(seeds) for split, seeds in config.split_seeds.items()
        },
        "pedestrian_modes": [mode.value for mode in config.pedestrian_modes],
        "image_counts": split_counts,
        "visible_box_counts": object_counts,
        "negative_frame_counts": {
            split: sum(
                row["split"] == split and row["label_count"] == 0
                for row in frame_rows
            )
            for split in ("train", "val", "test")
        },
        "annotation": {
            "method": "object-specific difference of visible/hidden simulator RGB renders",
            "minimum_visible_pixels": config.stop_sign_rules.minimum_visible_pixels,
            "minimum_bbox_width_px": config.stop_sign_rules.minimum_width_px,
            "minimum_bbox_height_px": config.stop_sign_rules.minimum_height_px,
            "class_specific_truncation": {
                name: {
                    "maximum_border_touches": config.annotation_rules(name).maximum_border_touches,
                    "minimum_truncated_height_px": config.annotation_rules(name).minimum_truncated_height_px,
                }
                for name in YOLO_V1_CLASS_NAMES
            },
            "mask_runtime_use": False,
        },
        "sampling": {
            "capture_every_steps": config.capture_every_steps,
            "minimum_capture_translation_m": config.minimum_capture_translation_m,
            "frame_rate_hz": 30,
        },
        "domain_randomization": config.domain_randomization,
        "source_hashes": source_hashes,
    }


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
