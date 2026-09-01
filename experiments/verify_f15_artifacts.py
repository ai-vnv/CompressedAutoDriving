#!/usr/bin/env python3
"""Fail-closed verifier for the completed F15 artifact/report package."""

from __future__ import annotations

import json
from pathlib import Path

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256

from run_f15_cross_curriculum_recovery import artifact_root, load_config, verify_protocol, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"


def main() -> None:
    config = load_config(CONFIG)
    verify_protocol(config, CONFIG)
    root = artifact_root(config, CONFIG)
    required = [
        ROOT / "docs/F15_PROTOCOL.md",
        ROOT / "docs/F15_FAILURE_LOCALIZATION_REPORT.md",
        ROOT / "docs/F15_RECOVERY_REPORT.md",
        ROOT / "docs/F15_FINAL_REPORT.md",
        ROOT / "docs/F15_OPTIMIZATION_RECOVERY_GUIDE_ID.md",
        ROOT / "docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md",
        ROOT / "docs/F15_VISUAL_EVIDENCE_OUTCOME.md",
        root / "localization/cross_curriculum_results.csv",
        root / "localization/cross_curriculum_results.json",
        root / "localization/matrix_results.json",
        root / "localization/pruning_width_retention.csv",
        root / "localization/open_loop_fidelity_by_curriculum.json",
        root / "localization/failure_event_registry.csv",
        root / "localization/failure_localization_decision.json",
        root / "recovery/datasets/dataset_manifest.json",
        root / "dataset_manifest.json",
        root / "recovery/recovery_experiments.csv",
        root / "recovery/recovery_decision.json",
        root / "final/final_candidate.json",
        root / "final/final_holdout_claim.json",
        root / "final/final_holdout.json",
        root / "final/efficiency_summary.json",
        root / "failure_telemetry/failure_telemetry_manifest.json",
        root / "success_telemetry/success_telemetry_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    for path in required[:5]:
        if path.exists() and "F15_RESULT_PLACEHOLDER" in path.read_text(encoding="utf-8"):
            missing.append(f"unresolved report placeholder: {path}")
    guide_path = ROOT / "docs/F15_OPTIMIZATION_RECOVERY_GUIDE_ID.md"
    guide_section_count = 0
    if guide_path.exists():
        guide_section_count = sum(
            line.startswith("## ") for line in guide_path.read_text(encoding="utf-8").splitlines()
        )
        if guide_section_count != 40:
            missing.append(f"Indonesian guide has {guide_section_count} numbered sections; expected 40")
    figure_stems = (
        "01_cross_curriculum_competence_across_compression_stages",
        "02_first_collapse_stage_by_curriculum",
        "03_pruning_width_vs_curriculum_retention",
        "04_action_fidelity_by_curriculum_and_stage",
        "05_original_vs_compressed_failure_timeline",
        "06_multi_curriculum_distillation_recovery",
        "07_quantization_after_recovery",
        "08_final_cross_curriculum_performance",
        "09_compression_retention_tradeoff",
    )
    for stem in figure_stems:
        for suffix in (".png", ".pdf"):
            path = root / "figures" / f"{stem}{suffix}"
            if not path.exists():
                missing.append(str(path))
    historical_manifest = json.loads((ROOT / "artifacts/f14_explainability_aware_compression_v1/integrity/historical_integrity_manifest.json").read_text(encoding="utf-8"))
    historical_mismatches = {
        relative: {"expected": expected, "actual": file_sha256(ROOT / relative)}
        for relative, expected in historical_manifest["files"].items()
        if not (ROOT / relative).exists() or file_sha256(ROOT / relative) != expected
    }
    final = json.loads((root / "final/final_holdout.json").read_text(encoding="utf-8")) if (root / "final/final_holdout.json").exists() else {}
    claim = json.loads((root / "final/final_holdout_claim.json").read_text(encoding="utf-8")) if (root / "final/final_holdout_claim.json").exists() else {}
    final_integrity = bool(final) and bool(claim) and final.get("claim_sha256") == file_sha256(root / "final/final_holdout_claim.json")
    manifest_path = root / "artifact_manifest.json"
    files = {}
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
            files[str(path.relative_to(ROOT))] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    source_paths = (
        ROOT / "configs/f15_cross_curriculum_recovery_v1.toml",
        ROOT / "docs/F15_PROTOCOL.md",
        ROOT / "docs/F15_FAILURE_LOCALIZATION_REPORT.md",
        ROOT / "docs/F15_RECOVERY_REPORT.md",
        ROOT / "docs/F15_FINAL_REPORT.md",
        ROOT / "docs/F15_OPTIMIZATION_RECOVERY_GUIDE_ID.md",
        ROOT / "src/duckie_pomdp/optimization/cross_curriculum_recovery.py",
        ROOT / "src/duckie_pomdp/optimization/actor_compression.py",
        ROOT / "src/duckie_pomdp/optimization/compression_metrics.py",
        ROOT / "experiments/run_f15_cross_curriculum_recovery.py",
        ROOT / "experiments/run_f15_recovery.py",
        ROOT / "experiments/render_f15_failure_traces.py",
        ROOT / "experiments/render_f15_telemetry_diagnostics.py",
        ROOT / "experiments/generate_f15_figures.py",
        ROOT / "experiments/verify_f15_artifacts.py",
        ROOT / "tests/test_f15_cross_curriculum_recovery.py",
    )
    source_files = {
        str(path.relative_to(ROOT)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
        if path.exists()
    }
    classification = "PASS" if not missing and not historical_mismatches and final_integrity else "FAIL"

    # Distinguish "the study deliberately stopped before the once-only holdout" from
    # "the pipeline is broken".  This annotates the result; it never changes it.
    holdout_trio = {
        str(root / "final/final_candidate.json"),
        str(root / "final/final_holdout_claim.json"),
        str(root / "final/final_holdout.json"),
    }
    recovery_decision_path = root / "recovery/recovery_decision.json"
    incompleteness_reason = None
    if classification == "FAIL" and not historical_mismatches and set(missing) <= holdout_trio:
        recovery_decision = (
            json.loads(recovery_decision_path.read_text(encoding="utf-8"))
            if recovery_decision_path.exists() else {}
        )
        incompleteness_reason = {
            "kind": "DELIBERATE_STOP_BEFORE_ONCE_ONLY_HOLDOUT",
            "detail": (
                "Every artifact except the final-candidate/holdout trio is present and every "
                "historical hash matches. No INT8 candidate satisfied the frozen C0-C4 gates, so "
                "no final candidate could be frozen and the once-only holdout was deliberately "
                "left unopened rather than spent on an ineligible or substituted model."
            ),
            "recovery_outcome": recovery_decision.get("outcome"),
            "eligible_int8_candidates": recovery_decision.get("eligible_int8_candidates"),
            "final_holdout_seeds_unopened": recovery_decision.get("final_holdout_seeds_unopened"),
            "study_classification": "LIMITED (see docs/F15_FINAL_REPORT.md)",
        }

    output = {
        "schema_version": 1,
        "classification": classification,
        "incompleteness_reason": incompleteness_reason,
        "config_sha256": config["_sha256"],
        "required_missing": missing,
        "historical_mismatches": historical_mismatches,
        "final_claim_integrity": final_integrity,
        "indonesian_guide_section_count": guide_section_count,
        "artifact_count": len(files),
        "files": files,
        "source_files": source_files,
    }
    write_json(manifest_path, output)
    print(json.dumps({key: output[key] for key in (
        "classification", "incompleteness_reason", "required_missing",
        "historical_mismatches", "final_claim_integrity", "artifact_count",
    )}, indent=2))
    if classification != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
