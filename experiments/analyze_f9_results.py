"""Regenerate F9 metric JSON from the immutable final-evaluation CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from duckie_pomdp.evaluation.f9_belief import summarize_f9
from duckie_pomdp.evaluation.f9_protocol import load_f9_protocol


ROOT = Path(__file__).resolve().parents[1]
BOOLEAN_FIELDS = {
    "eligible_visible",
    "measurement_detected",
    "duplicate_selection",
    "selected_correct_iou50",
    "false_measurement_event",
    "false_track_initialization",
    "raw_belief_initialized",
    "corrected_belief_initialized",
}


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, object] = {}
            for key, value in source.items():
                if key in BOOLEAN_FIELDS:
                    row[key] = value == "True"
                elif value == "":
                    row[key] = None
                else:
                    row[key] = value
            rows.append(row)
    return rows


def main() -> None:
    protocol = load_f9_protocol(
        ROOT / "configs" / "f9_yolo_ekf_v1.toml",
        require_frozen=True,
    )
    metrics_path = protocol.artifacts["belief_metrics_json"]
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = load_rows(protocol.artifacts["validation_csv"])
    metrics, nis = summarize_f9(
        rows,
        active_probability_threshold=protocol.active_belief_probability_threshold,
        nis_threshold=protocol.chi_square_2d_95_threshold,
    )
    report["metrics"] = metrics
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    protocol.artifacts["nis_metrics_json"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate": "F9b",
                "config_sha256": protocol.config_sha256,
                "nis": nis,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "events": metrics["events"],
                "selection_diagnostics": metrics["selection_diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
