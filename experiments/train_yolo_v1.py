"""Train one controlled YOLO11n baseline and emit auditable provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.domain.detection import YOLO_V1_CLASS_NAMES
from duckie_pomdp.perception.yolo_detector import validate_v1_class_mapping


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def select_sanity_images(
    dataset_root: Path, max_images_per_class: int
) -> dict[str, list[Path]]:
    objects_path = dataset_root / "metadata" / "objects.csv"
    candidates: dict[str, list[tuple[float, str]]] = {
        class_name: [] for class_name in YOLO_V1_CLASS_NAMES
    }
    with objects_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "test" or row["object_visible"] != "1":
                continue
            width = float(row["bbox_x2"]) - float(row["bbox_x1"])
            height = float(row["bbox_y2"]) - float(row["bbox_y1"])
            candidates[row["object_class"]].append((width * height, row["image_id"]))
    return {
        class_name: [
            dataset_root / "images" / "test" / f"{image_id}.png"
            for _, image_id in sorted(entries, reverse=True)[:max_images_per_class]
        ]
        for class_name, entries in candidates.items()
    }


def scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "yolo_train_v1.toml"
    )
    args = parser.parse_args()
    config = load_toml(args.config)
    training = config["training"]
    augmentation = config["augmentation"]
    validation = config["validation"]
    sanity_config = config["readiness_sanity"]
    dataset_yaml = resolve(training["dataset_yaml"])
    dataset_root = dataset_yaml.parent
    dataset_manifest_path = ROOT / "artifacts" / "detection_dataset_v1_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    expected_mapping = {
        str(index): name for index, name in enumerate(YOLO_V1_CLASS_NAMES)
    }
    if dataset_manifest["class_mapping"] != expected_mapping:
        raise RuntimeError("dataset manifest class mapping is not Version 1")

    from ultralytics import YOLO

    seed = int(training["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    project_dir = resolve(training["project_dir"])
    run_name = str(training["run_name"])
    model = YOLO(str(config["model"]["architecture"]))
    train_result = model.train(
        data=str(dataset_yaml),
        epochs=int(training["epochs"]),
        imgsz=int(training["image_size"]),
        batch=int(training["batch_size"]),
        device=training["device"],
        workers=int(training["workers"]),
        seed=seed,
        deterministic=bool(training["deterministic"]),
        patience=int(training["patience"]),
        optimizer=str(training["optimizer"]),
        lr0=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        cache=bool(training["cache"]),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=bool(config["model"]["pretrained"]),
        hsv_h=float(augmentation["hsv_h"]),
        hsv_s=float(augmentation["hsv_s"]),
        hsv_v=float(augmentation["hsv_v"]),
        degrees=float(augmentation["degrees"]),
        translate=float(augmentation["translate"]),
        scale=float(augmentation["scale"]),
        shear=float(augmentation["shear"]),
        perspective=float(augmentation["perspective"]),
        flipud=float(augmentation["flip_up_down"]),
        fliplr=float(augmentation["flip_left_right"]),
        mosaic=float(augmentation["mosaic"]),
        mixup=float(augmentation["mixup"]),
        copy_paste=float(augmentation["copy_paste"]),
        close_mosaic=int(augmentation["close_mosaic"]),
        plots=True,
        verbose=True,
    )
    run_dir = Path(train_result.save_dir)
    source_best = run_dir / "weights" / "best.pt"
    source_last = run_dir / "weights" / "last.pt"
    if not source_best.is_file():
        raise RuntimeError(f"trainer did not produce {source_best}")
    artifact_dir = ROOT / "artifacts" / "yolo_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stable_best = artifact_dir / "best.pt"
    stable_last = artifact_dir / "last.pt"
    shutil.copy2(source_best, stable_best)
    if source_last.is_file():
        shutil.copy2(source_last, stable_last)

    selected = YOLO(str(stable_best))
    checkpoint_names = validate_v1_class_mapping(selected.names)
    metrics = selected.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=int(training["image_size"]),
        batch=int(training["batch_size"]),
        device=training["device"],
        conf=float(validation["confidence_threshold"]),
        iou=float(validation["iou_threshold"]),
        plots=True,
        project=str(artifact_dir / "validation"),
        name="heldout_val",
        exist_ok=True,
        verbose=False,
    )
    development_metrics = {
        "split": "validation",
        "precision": scalar(metrics.box.mp),
        "recall": scalar(metrics.box.mr),
        "map50": scalar(metrics.box.map50),
        "map50_95": scalar(metrics.box.map),
        "per_class_map50_95": {
            name: scalar(metrics.box.maps[index])
            for index, name in enumerate(checkpoint_names)
        },
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }
    (artifact_dir / "training_metrics.json").write_text(
        json.dumps(development_metrics, indent=2) + "\n", encoding="utf-8"
    )

    sanity_images = select_sanity_images(
        dataset_root, int(sanity_config["max_images_per_class"])
    )
    sanity_report: dict[str, Any] = {
        "purpose": "limited readiness check; not the F8a/F8b test evaluation",
        "confidence_threshold": float(sanity_config["confidence_threshold"]),
        "classes": {},
    }
    for class_id, class_name in enumerate(checkpoint_names):
        image_paths = sanity_images[class_name]
        results = selected.predict(
            source=[str(path) for path in image_paths],
            conf=float(sanity_config["confidence_threshold"]),
            iou=float(sanity_config["iou_threshold"]),
            imgsz=int(training["image_size"]),
            device=training["device"],
            verbose=False,
        )
        hit_images = []
        for image_path, result in zip(image_paths, results, strict=True):
            predicted_ids = (
                []
                if result.boxes is None
                else [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
            )
            if class_id in predicted_ids:
                hit_images.append(str(image_path.relative_to(ROOT)))
        sanity_report["classes"][class_name] = {
            "images_checked": len(image_paths),
            "images_with_target_detection": len(hit_images),
            "example_hits": hit_images[:3],
            "ready": bool(hit_images),
        }
    sanity_report["ready"] = all(
        entry["ready"] for entry in sanity_report["classes"].values()
    )
    (artifact_dir / "test_sanity.json").write_text(
        json.dumps(sanity_report, indent=2) + "\n", encoding="utf-8"
    )

    resolved_config = {
        "model": config["model"],
        "training": training,
        "augmentation": augmentation,
        "validation": validation,
        "readiness_sanity": sanity_config,
    }
    (artifact_dir / "training_config.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8"
    )
    model_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(stable_best.relative_to(ROOT)),
        "checkpoint_sha256": sha256(stable_best),
        "architecture": config["model"]["architecture"],
        "pretrained_initialization": bool(config["model"]["pretrained"]),
        "initial_weights_sha256": (
            sha256(ROOT / config["model"]["architecture"])
            if (ROOT / config["model"]["architecture"]).is_file()
            else None
        ),
        "class_mapping": {str(index): name for index, name in enumerate(checkpoint_names)},
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "training_seed": seed,
        "framework": {
            "python": platform.python_version(),
            "ultralytics": importlib.metadata.version("ultralytics"),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_config": str(args.config.resolve()),
        "training_config_sha256": sha256(args.config.resolve()),
        "environment_spec": "configs/yolo_env_v1.json",
        "environment_spec_sha256": sha256(ROOT / "configs" / "yolo_env_v1.json"),
        "run_dir": str(run_dir),
        "test_sanity_ready": sanity_report["ready"],
        "inference_input": "uint8 front RGB only",
        "privileged_inference_input": False,
        "environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        },
    }
    (artifact_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": development_metrics, "sanity": sanity_report}, indent=2))
    if not sanity_report["ready"]:
        raise SystemExit("checkpoint failed the limited untouched-test readiness check")


if __name__ == "__main__":
    main()
