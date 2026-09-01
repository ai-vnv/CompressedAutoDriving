#!/usr/bin/env python3
"""Fail-closed verifier and final classifier for F13 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.explain.compressed_policy_analysis import (
    file_sha256,
    require_quantized_linear_graph,
    verify_hash,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"
REQUIRED_FIGURES = (
    "original_vs_a7_overall_attribution", "original_phase_heatmap_v",
    "a7_phase_heatmap_v", "attribution_drift_heatmap_v",
    "original_phase_heatmap_omega", "a7_phase_heatmap_omega",
    "attribution_drift_heatmap_omega", "semantic_structure_preservation",
    "counterfactual_original_vs_a7", "bev_original_vs_a7_representative_panels",
    "failure_mode_summary",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    if args.read_only:
        verify_existing_manifest()
        return
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    root = resolve(config["artifacts"]["directory"])
    final = root / "final"
    preservation_path = final / "explanation_preservation.json"
    registry_path = final / "failure_mode_registry.json"
    manifest_path = final / "artifact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    original = resolve(config["frozen"]["original"]["checkpoint"])
    a7_path = resolve(config["frozen"]["a7"]["checkpoint"])
    verify_hash(original, config["frozen"]["original"]["sha256"])
    verify_hash(a7_path, config["frozen"]["a7"]["sha256"])
    require_quantized_linear_graph(torch.jit.load(str(a7_path), map_location="cpu").eval())
    surrogate = read(root / "integrity/surrogate_equivalence.json")
    replay = read(root / "integrity/replay_integrity.json")
    cf = read(root / "counterfactual/counterfactual_metrics.json")
    stress = read(root / "failure_modes/exploratory/summary.json")
    confirm = read(root / "failure_modes/confirmatory_not_run.json")
    if surrogate["gradient_attribution_authorized"] or surrogate["approximate_surrogate_created"]:
        raise RuntimeError("blocked attribution branch integrity failed")
    if replay["classification"] != "PASS" or not cf["sham"]["pass"]:
        raise RuntimeError("replay or sham integrity failed")
    if stress["behavioral_classification"] != "PRESERVED":
        behavioral = "DEGRADED"
    else:
        behavioral = "PRESERVED"
    if confirm["seeds_opened"]:
        raise RuntimeError("confirmatory seeds were opened without an exploratory candidate")
    rows = list(csv.DictReader((root / "failure_modes/exploratory/paired_episodes.csv").open(newline="", encoding="utf-8")))
    keys = {(row["policy"], int(row["seed"])) for row in rows}
    expected = {(policy, seed) for policy in ("Original", "A7") for seed in config["stress"]["exploratory_seeds"]}
    if len(rows) != len(expected) or keys != expected:
        raise RuntimeError("paired stress episode cardinality mismatch")
    for stem in REQUIRED_FIGURES:
        for suffix in (".png", ".pdf"):
            path = root / "figures" / f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing figure {path}")
    reports = (
        ROOT / "docs/F13_EXPLAIN_AGAIN_PROTOCOL.md",
        ROOT / "docs/F13_SURROGATE_EQUIVALENCE_PROTOCOL.md",
        ROOT / "docs/F13_EXPLANATION_COMPARISON.md",
        ROOT / "docs/F13_FAILURE_MODE_REPORT.md",
        ROOT / "docs/F13_FINAL_REPORT.md",
    )
    if any(not path.exists() for path in reports):
        raise RuntimeError("required F13 report is missing")

    functional = cf["classification"]
    semantic = "UNRESOLVED"
    overall = "FAILED" if behavioral == "DEGRADED" else "LIMITED"
    attribution_dir = root / "attribution/comparison"
    attribution_dir.mkdir(parents=True, exist_ok=True)
    write(attribution_dir / "attribution_blocked.json", {
        "classification": "BLOCKED",
        "semantic_explanation_structure": semantic,
        "reason": surrogate["reason"],
        "original_r004_unchanged": True,
        "a7_values_imputed": False,
    })
    preservation = {
        "schema_version": 1,
        "original_sha256": file_sha256(original),
        "a7_sha256": file_sha256(a7_path),
        "config_sha256": file_sha256(CONFIG),
        "behavioral_c4": behavioral,
        "semantic_explanation_structure": semantic,
        "counterfactual_functional_sensitivity": functional,
        "overall_f13": overall,
        "attribution_comparison_authorized": False,
        "counterfactual_primary_checks": cf["primary_checks"],
        "closed_loop_differential_failure_count": stress["differential_failure_count"],
        "models_modified": False,
    }
    registry = {
        "schema_version": 1,
        "supported_findings": [
            {
                "level": 2,
                "name": "reduced stop-release velocity sensitivity",
                "trigger": "stop_absent during stop_required",
                "original_mean_delta_v_mps": cf["summary"]["stop_absent"]["stop_required"]["v_cmd_mps"]["original_mean"],
                "a7_mean_delta_v_mps": cf["summary"]["stop_absent"]["stop_required"]["v_cmd_mps"]["compressed_mean"],
                "normalized_mean_drift": cf["summary"]["stop_absent"]["stop_required"]["v_cmd_mps"]["normalized_mean_effect_drift"],
                "closed_loop_failure_observed": False,
            },
            {
                "level": 2,
                "name": "reduced pedestrian-removal yaw sensitivity (auxiliary)",
                "trigger": "pedestrian_absent during pedestrian_relevant",
                "original_mean_delta_omega_rad_s": cf["summary"]["pedestrian_absent"]["pedestrian_relevant"]["omega_cmd_rad_s"]["original_mean"],
                "a7_mean_delta_omega_rad_s": cf["summary"]["pedestrian_absent"]["pedestrian_relevant"]["omega_cmd_rad_s"]["compressed_mean"],
                "normalized_mean_drift": cf["summary"]["pedestrian_absent"]["pedestrian_relevant"]["omega_cmd_rad_s"]["normalized_mean_effect_drift"],
                "closed_loop_failure_observed": False,
            },
        ],
        "level_3_action_drift": stress["level3_action_drift"],
        "level_4_closed_loop_failure": stress["level4_closed_loop_failure"],
        "paired_exploratory_seeds": stress["seeds"],
        "confirmatory": confirm,
        "excluded_failed_attempts": [path.name for path in root.glob("_failed_*")],
    }
    if preservation_path.exists():
        if read(preservation_path) != preservation:
            raise RuntimeError("existing final classification differs from recomputation")
    else:
        write(preservation_path, preservation)
    if registry_path.exists():
        if read(registry_path) != registry:
            raise RuntimeError("existing failure registry differs from recomputation")
    else:
        write(registry_path, registry)
    active_files = [
        path for path in root.rglob("*")
        if path.is_file() and "_failed_" not in path.as_posix() and path != manifest_path
    ]
    source_files = [
        CONFIG,
        ROOT / "src/duckie_pomdp/explain/compressed_policy_analysis.py",
        ROOT / "experiments/verify_f13_explanation_boundary.py",
        ROOT / "experiments/run_f13_counterfactual_comparison.py",
        ROOT / "experiments/run_f13_failure_mode_probe.py",
        ROOT / "experiments/generate_f13_figures.py",
        Path(__file__).resolve(),
        *reports,
    ]
    manifest = {
        "schema_version": 1,
        "classification": overall,
        "active_artifacts": {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(active_files)},
        "source_and_report_hashes": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "excluded_failed_attempt_directories": registry["excluded_failed_attempts"],
        "verification_passed": True,
    }
    write(manifest_path, manifest)
    print(json.dumps(preservation, indent=2))


def verify_existing_manifest() -> None:
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    root = resolve(config["artifacts"]["directory"])
    manifest_path = root / "final/artifact_manifest.json"
    manifest = read(manifest_path)
    if manifest["classification"] != "LIMITED" or not manifest["verification_passed"]:
        raise RuntimeError("final manifest classification/integrity mismatch")
    for relative, expected in manifest["active_artifacts"].items():
        path = ROOT / relative
        if not path.exists() or file_sha256(path) != expected:
            raise RuntimeError(f"active artifact hash mismatch: {relative}")
    for relative, expected in manifest["source_and_report_hashes"].items():
        path = ROOT / relative
        if not path.exists() or file_sha256(path) != expected:
            raise RuntimeError(f"source/report hash mismatch: {relative}")
    print(json.dumps({
        "classification": manifest["classification"],
        "active_artifacts_verified": len(manifest["active_artifacts"]),
        "source_and_reports_verified": len(manifest["source_and_report_hashes"]),
        "read_only": True,
        "pass": True,
    }, indent=2))


def resolve(value: str) -> Path:
    return (CONFIG.parent / value).resolve()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
