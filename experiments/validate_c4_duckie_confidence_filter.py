"""Freeze the calibration-only evidence for C4's Duckie score floor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f10_ppo_visual_objects_v21.toml"),
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path("artifacts/f9_yolo_measurement_calibration.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/f10_ppo_visual_objects_v21/duckie_confidence_gate.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    protocol = load_ppo_curriculum_protocol(args.config)
    threshold = float(
        protocol.raw["runtime_detection"]["duckie_minimum_confidence"]
    )
    correct: list[float] = []
    incorrect: list[float] = []
    with args.calibration_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            confidence = row.get("selected_confidence", "")
            correctness = row.get("selected_correct_iou50", "")
            if not confidence or correctness not in {"True", "False"}:
                continue
            target = correct if correctness == "True" else incorrect
            target.append(float(confidence))
    if not correct or not incorrect:
        raise RuntimeError("calibration data needs correct and incorrect selections")

    minimum_correct = min(correct)
    maximum_incorrect = max(incorrect)
    passed = minimum_correct >= threshold and maximum_incorrect < threshold
    result = {
        "schema_version": 1,
        "passed": passed,
        "seed_role": "f9_calibration_only_no_c4_dev_or_final",
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "calibration_csv": str(args.calibration_csv.resolve()),
        "calibration_csv_sha256": file_sha256(args.calibration_csv),
        "threshold": threshold,
        "correct_detection_count": len(correct),
        "incorrect_detection_count": len(incorrect),
        "minimum_correct_confidence": minimum_correct,
        "maximum_incorrect_confidence": maximum_incorrect,
    }
    if not passed:
        raise RuntimeError(f"calibration-only confidence gate failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
