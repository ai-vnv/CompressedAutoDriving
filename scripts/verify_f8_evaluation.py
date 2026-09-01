"""Read-only integrity check for the completed frozen F8a/F8b evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    config_path = ROOT / "configs" / "yolo_evaluation_v1.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    expected_hash = config["checkpoint"]["sha256"]
    checkpoint = ROOT / config["checkpoint"]["path"]
    require(sha256(checkpoint) == expected_hash, "frozen checkpoint hash mismatch")

    detection = json.loads(
        (ROOT / "artifacts" / "yolo_detection_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    measurement = json.loads(
        (ROOT / "artifacts" / "yolo_measurement_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    noise = json.loads(
        (ROOT / "artifacts" / "yolo_measurement_noise_v1.json").read_text(
            encoding="utf-8"
        )
    )
    detection_rows = csv_rows(
        ROOT / "artifacts" / "yolo_detection_validation.csv"
    )
    measurement_rows = csv_rows(
        ROOT / "artifacts" / "yolo_measurement_validation.csv"
    )

    expected_images = int(config["dataset"]["expected_images"])
    expected_opportunities = expected_images * len(config["checkpoint"]["class_names"])
    config_hash = sha256(config_path)
    require(detection["checkpoint_sha256"] == expected_hash, "detection provenance mismatch")
    require(measurement["checkpoint_sha256"] == expected_hash, "measurement provenance mismatch")
    require(
        detection["evaluation_config_sha256"] == config_hash
        and measurement["evaluation_config_sha256"] == config_hash,
        "evaluation configuration provenance mismatch",
    )
    require(detection["test_images"] == expected_images, "test image count mismatch")
    require(len(detection_rows) == expected_opportunities, "detection CSV row count mismatch")
    require(len(measurement_rows) == expected_opportunities, "measurement CSV row count mismatch")
    require(measurement["projection_failure_count"] == 0, "metric projection contains failures")
    require(
        measurement["degradation_vs_f5b"]["raw_range_rmse_ratio"]
        < measurement["degradation_vs_f5b"][
            "f5b_calibrated_range_rmse_ratio"
        ],
        "raw and F5b-calibrated degradation ratios are mislabeled",
    )
    require(
        measurement["matched_detection_count"]
        == sum(row["detected"] == "True" for row in measurement_rows),
        "matched detection count mismatch",
    )
    require(
        noise["status"] == "candidate_for_F9_not_applied_to_frozen_F7",
        "candidate noise status is not isolated from F7",
    )
    require(
        noise["measurement"]["range_semantics"] == "object_origin",
        "canonical range semantics changed",
    )
    require(
        noise["measurement"]["range_value"] == "raw_projected_yolo_range"
        and noise["measurement"]["f5b_calibration_applied"] is False,
        "candidate range path disagrees with held-out raw-vs-F5b comparison",
    )
    candidate_toml_path = ROOT / "configs" / "measurement_model_yolo_v1.toml"
    require(candidate_toml_path.is_file(), "candidate TOML is missing")
    with candidate_toml_path.open("rb") as stream:
        candidate_toml = tomllib.load(stream)
    require(
        candidate_toml["measurement"]["range_value"]
        == noise["measurement"]["range_value"]
        and candidate_toml["measurement"]["f5b_calibration_applied"] is False,
        "candidate TOML and JSON measurement paths disagree",
    )
    error_cases = tuple((ROOT / "artifacts" / "yolo_error_cases").glob("*.png"))
    require(error_cases, "deterministic error-case visualizations are missing")

    print(
        json.dumps(
            {
                "verified": True,
                "checkpoint_sha256": expected_hash,
                "test_images": expected_images,
                "opportunity_rows": expected_opportunities,
                "matched_detections": measurement["matched_detection_count"],
                "projection_failures": measurement["projection_failure_count"],
                "error_case_images": len(error_cases),
                "candidate_range_value": noise["measurement"]["range_value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
