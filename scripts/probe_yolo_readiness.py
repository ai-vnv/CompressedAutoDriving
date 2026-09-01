"""Audit a trained checkpoint and perform image-only test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from duckie_pomdp.domain.detection import YOLO_V1_CLASS_NAMES
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_images(dataset_root: Path) -> dict[int, Path]:
    representatives: dict[int, Path] = {}
    for label_path in sorted((dataset_root / "labels" / "test").glob("*.txt")):
        class_ids = {
            int(line.split()[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for class_id in class_ids:
            representatives.setdefault(
                class_id, dataset_root / "images" / "test" / f"{label_path.stem}.png"
            )
        if len(representatives) == len(YOLO_V1_CLASS_NAMES):
            break
    return representatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "artifacts" / "yolo_v1" / "best.pt"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "datasets" / "duckietown_detection_v1",
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "yolo_v1" / "readiness_examples",
    )
    args = parser.parse_args()
    report: dict[str, object] = {
        "weights": str(args.weights),
        "weights_exist": args.weights.is_file(),
        "required_class_mapping": {
            str(index): name for index, name in enumerate(YOLO_V1_CLASS_NAMES)
        },
        "inference_input": "uint8 front RGB only",
    }
    try:
        import ultralytics

        report["ultralytics_installed"] = True
        report["ultralytics_version"] = ultralytics.__version__
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        detector = YoloObjectDetector(
            args.weights, confidence_threshold=args.confidence, image_size=480, device=0
        )
        images = representative_images(args.dataset_root)
        if set(images) != set(range(len(YOLO_V1_CLASS_NAMES))):
            raise RuntimeError("test split lacks a representative image for each class")
        inference: dict[str, object] = {}
        all_fields_valid = True
        all_targets_seen = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for class_id, path in sorted(images.items()):
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            detections = detector.detect(rgb)
            predicted_classes = [
                detection.object_class.value for detection in detections
            ]
            target_seen = YOLO_V1_CLASS_NAMES[class_id] in predicted_classes
            inference[YOLO_V1_CLASS_NAMES[class_id]] = {
                "image": str(path),
                "detection_count": len(detections),
                "classes": predicted_classes,
                "target_seen": target_seen,
            }
            all_targets_seen = all_targets_seen and target_seen
            all_fields_valid = all_fields_valid and all(
                0.0 <= detection.confidence <= 1.0
                and detection.bounding_box.x_max_px >= detection.bounding_box.x_min_px
                and detection.bounding_box.y_max_px >= detection.bounding_box.y_min_px
                for detection in detections
            )
            overlay = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(overlay)
            for detection in detections:
                box = detection.bounding_box
                draw.rectangle(
                    (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                    outline=(255, 0, 0),
                    width=3,
                )
                draw.text(
                    (box.x_min_px + 2, box.y_min_px + 2),
                    f"{detection.object_class.value} {detection.confidence:.2f}",
                    fill=(255, 255, 0),
                )
            overlay.save(
                args.output_dir / f"{YOLO_V1_CLASS_NAMES[class_id]}_readiness.png"
            )
        report.update(
            {
                "checkpoint_sha256": file_sha256(args.weights),
                "checkpoint_loads": True,
                "class_count": len(YOLO_V1_CLASS_NAMES),
                "class_mapping_matches": True,
                "front_rgb_inference_succeeds": True,
                "detection_fields_valid": all_fields_valid,
                "both_target_classes_seen": all_targets_seen,
                "inference": inference,
            }
        )
        report["ready_for_inference_implementation"] = bool(
            all_fields_valid and all_targets_seen
        )
    except Exception as error:
        report["ready_for_inference_implementation"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(report, indent=2))
    if not report["ready_for_inference_implementation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
