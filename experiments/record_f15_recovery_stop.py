#!/usr/bin/env python3
"""Record the F15 recovery outcome when no INT8 candidate satisfied the frozen gates.

``run_f15_recovery.py freeze-candidate`` writes ``recovery_experiments.csv`` and
``recovery_decision.json`` as a side effect of selecting a deployable INT8 actor. F15
stopped before that point: the FP32 recovered actor passed every frozen C0-C4 behavior,
fidelity, and safety gate, but neither PTQ nor multi-curriculum QAT+KD produced an
eligible INT8 actor at width 64.

This script therefore writes the same two machine-readable artifacts describing the
recovery experiments that were actually run and the decision to stop. It freezes no
candidate, opens no holdout seed, and changes no gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    CURRICULA,
    append_csv,
    artifact_root,
    load_config,
    provenance,
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    decision_path = root / "recovery/recovery_decision.json"
    experiments_csv = root / "recovery/recovery_experiments.csv"
    for path in (decision_path, experiments_csv):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")

    experiments = []
    for path in sorted((root / "recovery").glob("**/selection_result.json")):
        payload = read_json(path)
        entry = payload["entry"]
        statuses = {
            c: payload["behavior"]["decisions"][entry["variant"]][c]["status"] for c in CURRICULA
        }
        experiments.append({
            "result_path": str(path),
            "result_sha256": file_sha256(path),
            "model_id": entry["variant"],
            "model_name": entry["name"],
            "model_sha256": entry["sha256"],
            "width": int(entry["hidden_sizes"][0]),
            "int8": bool(entry["int8"]),
            "parameter_count": int(entry["parameter_count"]),
            **{f"status_{c}": statuses[c] for c in CURRICULA},
            "behavior_all_curricula_pass": bool(payload["all_curricula_behavior_pass"]),
            "fidelity_all_curricula_pass": bool(payload["fidelity"]["all_curricula_pass"]),
            "eligible": bool(payload["eligible"]),
        })
    if not experiments:
        raise RuntimeError("no F15 recovery selection results found")
    for row in experiments:
        append_csv(experiments_csv, row)

    by_id = {row["model_id"]: row for row in experiments}
    fp32 = next((r for r in experiments if not r["int8"] and r["width"] == 64), None)
    ptq = next((r for r in experiments if r["int8"] and "PTQ" in r["model_id"].upper()), None)
    qat = next((r for r in experiments if r["int8"] and "QAT" in r["model_id"].upper()), None)
    eligible_int8 = [r for r in experiments if r["int8"] and r["eligible"]]

    decision = {
        **provenance(config, CONFIG),
        "classification": "FROZEN",
        "outcome": "STOPPED_WITHOUT_ELIGIBLE_INT8_CANDIDATE",
        "localization_decision_sha256": file_sha256(root / "localization/failure_localization_decision.json"),
        "dataset_manifest_sha256": file_sha256(root / "recovery/datasets/dataset_manifest.json"),

        "multicurriculum_kd_recovered_64": bool(fp32 and fp32["eligible"]),
        "larger_width_required": "NOT_TESTED",
        "larger_width_not_tested_reason": (
            "the frozen rule run_larger_width_only_if_64_fails applies to the FP32 recovery, "
            "and the 64x64 FP32 student passed every gate, so widths 96/128/192 were not triggered"
        ),
        "ptq_preserved_recovery": bool(ptq and ptq["eligible"]),
        "qat_required": bool(ptq is not None and not ptq["eligible"]),
        "qat_restored_recovery": bool(qat and qat["eligible"]),
        "progressive_pruning_required": "NOT_TESTED",
        "progressive_pruning_not_tested_reason": (
            "direct target-width recovery succeeded in FP32; the observed loss is associated with "
            "the INT8 conversion rather than with pruning, and the study was stopped for reporting "
            "before opening a new experimental branch"
        ),

        "eligible_int8_candidates": len(eligible_int8),
        "final_candidate_frozen": False,
        "final_holdout_opened": False,
        "final_holdout_seeds_unopened": [int(v) for v in config["seeds"]["final_holdout"]],
        "stop_reason": (
            "No INT8 actor satisfied the frozen C0-C4 behavior, fidelity, and safety gates at "
            "width 64. freeze-candidate requires a deployable INT8 actor with eligible=true, so no "
            "final candidate could be frozen and the once-only holdout was deliberately left "
            "unopened. Substituting a different model after seeing these results is forbidden by "
            "the frozen protocol."
        ),
        "quantization_finding": (
            "Under the tested x86 static quantization procedure with curriculum-balanced C0-C4 "
            "calibration, INT8 conversion of the recovered 64x64 student introduced a retention "
            "failure in C3 and C4 while C0-C2 were preserved. Multi-curriculum QAT+KD improved "
            "same-state fidelity over PTQ but did not restore C3/C4 behavior. This is a statement "
            "about the tested procedure at this width, not a universal claim about quantization."
        ),
        "experiments": experiments,
        "experiments_csv": str(experiments_csv),
        "experiments_csv_sha256": file_sha256(experiments_csv),
    }
    write_json(decision_path, decision)
    print(json.dumps({
        "recovery_decision": str(decision_path),
        "experiments_csv": str(experiments_csv),
        "outcome": decision["outcome"],
        "multicurriculum_kd_recovered_64": decision["multicurriculum_kd_recovered_64"],
        "ptq_preserved_recovery": decision["ptq_preserved_recovery"],
        "qat_restored_recovery": decision["qat_restored_recovery"],
        "eligible_int8_candidates": decision["eligible_int8_candidates"],
        "final_holdout_opened": decision["final_holdout_opened"],
        "rows": len(experiments),
    }, indent=2))


if __name__ == "__main__":
    main()
