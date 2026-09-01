"""Read-only integrity checks for the completed F9 calibration/evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from duckie_pomdp.evaluation.f9_protocol import load_f9_protocol, sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN_BASELINE_HASHES = {
    "configs/oracle_ekf_v1.toml": (
        "a4815c8d0e17f1868d51619ae51d2183c72832a022edce88aa3c10302594d701"
    ),
    "artifacts/belief_calibration_metrics.json": (
        "1bf8d2bcd328a9703e9cb2797cd7a1cf54f8ddd9d231342cbc9718b58ffea5e5"
    ),
    "artifacts/yolo_measurement_metrics.json": (
        "b916ff2505dd9eb2e0041efda512f3ab27c2b969cc1a9e00c84e44de5bf29c39"
    ),
}


def main() -> None:
    protocol = load_f9_protocol(
        ROOT / "configs" / "f9_yolo_ekf_v1.toml",
        require_frozen=True,
    )
    with protocol.artifacts["validation_csv"].open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    metrics = json.loads(
        protocol.artifacts["belief_metrics_json"].read_text(encoding="utf-8")
    )
    nis = json.loads(
        protocol.artifacts["nis_metrics_json"].read_text(encoding="utf-8")
    )
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(protocol.final_evaluation_seeds):
        raise RuntimeError(f"unexpected F9 final seeds: {sorted(seeds)}")
    if seeds & set(protocol.calibration_seeds):
        raise RuntimeError("F9 final CSV contains calibration seed leakage")
    final_specs = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
    expected = {
        (seed, spec.name): spec.steps + 1
        for seed in protocol.final_evaluation_seeds
        for spec in final_specs
    }
    observed = Counter((int(row["seed"]), row["scenario"]) for row in rows)
    if observed != expected:
        raise RuntimeError("F9 final scenario/frame matrix is incomplete")
    if metrics["metrics"]["row_count"] != len(rows):
        raise RuntimeError("F9 metrics row count does not match CSV")
    if metrics["source"]["config_sha256"] != protocol.config_sha256:
        raise RuntimeError("F9 metrics config hash mismatch")
    if nis["config_sha256"] != protocol.config_sha256:
        raise RuntimeError("F9 NIS config hash mismatch")
    for row in rows:
        if row["measurement_detected"] == "False" and any(
            row[field]
            for field in (
                "raw_measurement_range_m",
                "raw_measurement_bearing_rad",
                "corrected_measurement_range_m",
                "corrected_measurement_bearing_rad",
            )
        ):
            raise RuntimeError("F9 miss is not represented by empty/None geometry")
    baseline_hashes = {
        relative: sha256(ROOT / relative)
        for relative in FROZEN_BASELINE_HASHES
    }
    if baseline_hashes != FROZEN_BASELINE_HASHES:
        raise RuntimeError("a frozen F5b/F6/F7/F8 baseline artifact changed")
    error_cases = sorted(protocol.artifacts["error_case_dir"].glob("*.png"))
    if len(error_cases) != 16:
        raise RuntimeError("F9 error-case image set is incomplete")
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "seeds": sorted(seeds),
                "scenarios": [spec.name for spec in final_specs],
                "checkpoint_sha256": protocol.checkpoint_sha256,
                "calibration_artifact_sha256": (
                    protocol.calibration_artifact_sha256
                ),
                "config_sha256": protocol.config_sha256,
                "baseline_hashes": baseline_hashes,
                "error_case_images": len(error_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
