"""Freeze the V1 Duckie bbox image-domain gate from calibration-only evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


def validate(config: Path, f9_csv: Path, c4_csv: Path) -> dict[str, object]:
    protocol = load_ppo_curriculum_protocol(config)
    maximum = float(protocol.raw["runtime_detection"]["duckie_maximum_bottom_y_px"])
    with f9_csv.open(newline="", encoding="utf-8") as stream:
        f9 = list(csv.DictReader(stream))
    f9_correct = [row for row in f9 if row["selected_correct_iou50"].lower() == "true"]
    f9_retained = [row for row in f9_correct if float(row["selected_bbox_y2"]) <= maximum]
    with c4_csv.open(newline="", encoding="utf-8") as stream:
        c4 = list(csv.DictReader(stream))
    visible = [row for row in c4 if row["pedestrian_present_after_inference"].lower() == "true"]
    absent = [row for row in c4 if row["pedestrian_present_after_inference"].lower() != "true"]
    visible_retained = [row for row in visible if float(row["y2"]) <= maximum]
    false_accepted = [row for row in absent if float(row["y2"]) <= maximum]
    false_rejected = [row for row in absent if float(row["y2"]) > maximum]
    passed = (
        len(f9_correct) >= 1000
        and len(f9_retained) == len(f9_correct)
        and visible
        and len(visible_retained) == len(visible)
        and len(false_rejected) > 0
        and len(false_accepted) <= 1
    )
    return {
        "schema_version": 1,
        "config_sha256": file_sha256(config),
        "maximum_bottom_y_px": maximum,
        "uses_privileged_truth_for_runtime_filter": False,
        "runtime_inputs": ["object_class", "bbox_y_max_px"],
        "truth_role": "offline post-inference audit labels only",
        "f9_calibration_csv_sha256": file_sha256(f9_csv),
        "c4_audit_csv_sha256": file_sha256(c4_csv),
        "f9_correct_rows": len(f9_correct),
        "f9_correct_rows_retained": len(f9_retained),
        "c4_visible_rows": len(visible),
        "c4_visible_rows_retained": len(visible_retained),
        "c4_absent_false_rows": len(absent),
        "c4_absent_false_rows_rejected": len(false_rejected),
        "c4_absent_false_rows_accepted": len(false_accepted),
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--f9-calibration", type=Path, required=True)
    parser.add_argument("--c4-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = validate(
        args.config.resolve(), args.f9_calibration.resolve(), args.c4_audit.resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
