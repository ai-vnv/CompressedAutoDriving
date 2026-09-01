#!/usr/bin/env python3
"""F14-D: A0-only reference robustness calibration and threshold freeze."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from duckie_pomdp.explain.compression_diagnostics import (
    compare_group_summaries,
    file_sha256,
    load_f14_config,
    load_frozen_actors,
    load_policy_contract,
    resolve_config_path,
    summarize_group_attribution,
)
from duckie_pomdp.explain.group_shapley import exact_group_shapley


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f14_explainability_aware_compression_v1.toml"


def run(config_path: Path) -> dict[str, Any]:
    config = load_f14_config(config_path)
    root = resolve_config_path(config, str(config["outputs"]["directory"]))
    if (root / "diagnostic/ablation_shapley.csv").exists():
        raise RuntimeError("A1-A7 explanation already exists; calibration order violated")
    states_path = root / "diagnostic/diagnostic_states.npz"
    references_path = root / "diagnostic/reference_assignments.npz"
    state_manifest = root / "diagnostic_state_manifest.json"
    reference_manifest = root / "reference_assignment_manifest.json"
    required = (states_path, references_path, state_manifest, reference_manifest)
    if not all(path.exists() for path in required):
        raise RuntimeError("F14 diagnostic states/references must be prepared first")
    targets = (
        root / "calibration/a0_reference_attribution.npz",
        root / "reference_calibration_metrics.json",
        root / "calibration/frozen_thresholds.json",
        ROOT / "docs/F14_REFERENCE_CALIBRATION.md",
    )
    if any(path.exists() for path in targets):
        raise RuntimeError("refusing to overwrite immutable F14 calibration")
    protocol, feature_names, group_indexes = load_policy_contract(config)
    del protocol
    actors = load_frozen_actors(config)
    states = dict(np.load(states_path, allow_pickle=False))
    references = dict(np.load(references_path, allow_pickle=False))
    if tuple(str(value) for value in states["feature_names"]) != feature_names:
        raise RuntimeError("diagnostic feature ordering mismatch")
    if tuple(str(value) for value in references["feature_names"]) != feature_names:
        raise RuntimeError("reference feature ordering mismatch")
    result = exact_group_shapley(
        actors["A0"].physical,
        np.asarray(states["observation"], dtype=np.float32),
        np.asarray(references["observation"], dtype=np.float32),
        group_indexes,
        feature_names,
        observation_clip=float(config["frozen"]["observation_clip"]),
        state_batch_size=int(config["shapley"]["state_batch_size"]),
    )
    maximum_residual = float(np.max(np.abs(result.efficiency_residual)))
    draw_summaries = [
        summarize_group_attribution(result.draw_attribution[draw], states["public_phase"])
        for draw in range(result.draw_attribution.shape[0])
    ]
    pairwise: list[dict[str, Any]] = []
    for left, right in combinations(range(len(draw_summaries)), 2):
        rows = compare_group_summaries(
            draw_summaries[left], draw_summaries[right],
            signed_deadband=float(config["shapley"]["signed_deadband"]),
        )
        for row in rows:
            if row["phase"] != "overall":
                pairwise.append({"draw_a": left, "draw_b": right, **row})
    spearman = np.asarray([row["group_spearman"] for row in pairwise])
    share_l1 = np.asarray([row["group_share_l1"] for row in pairwise])
    top = np.asarray([float(row["top_group_agreement"]) for row in pairwise])
    top2 = np.asarray([row["top_two_jaccard"] for row in pairwise])
    signed = np.asarray([row["signed_direction_agreement"] for row in pairwise])
    rules = config["reference_calibration"]
    thresholds = {
        "minimum_group_spearman": float(max(
            rules["minimum_group_spearman_floor"],
            np.quantile(spearman, rules["spearman_quantile"]),
        )),
        "maximum_group_share_l1": float(min(
            rules["maximum_group_share_l1_ceiling"],
            np.quantile(share_l1, rules["share_l1_quantile"]) + rules["share_l1_margin"],
        )),
        "minimum_top_group_agreement": float(max(
            rules["minimum_top_group_agreement_floor"],
            np.quantile(top, rules["agreement_quantile"]),
        )),
        "minimum_top_two_jaccard": float(max(
            rules["minimum_top_two_jaccard_floor"],
            np.quantile(top2, rules["agreement_quantile"]),
        )),
        "minimum_signed_agreement": float(max(
            rules["minimum_signed_agreement_floor"],
            np.quantile(signed, rules["agreement_quantile"]),
        )),
        "minimum_preserved_phase_action_cells": int(rules["minimum_preserved_phase_action_cells"]),
        "total_phase_action_cells": int(rules["total_phase_action_cells"]),
    }
    checks = {
        "spearman_p05": float(np.quantile(spearman, 0.05)) >= 0.50,
        "signed_median": float(np.median(signed)) >= 0.50,
        "top_group_median": float(np.median(top)) >= 0.50,
        "share_l1_p95": float(np.quantile(share_l1, 0.95)) <= 0.75,
        "local_accuracy": maximum_residual <= float(config["shapley"]["efficiency_absolute_tolerance"]),
    }
    classification = "PASS" if all(checks.values()) else "LIMITED"
    metrics = {
        "schema_version": 1,
        "classification": classification,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": str(config["_sha256"]),
        "actor": "A0",
        "actor_sha256": actors["A0"].sha256,
        "states_sha256": file_sha256(states_path),
        "references_sha256": file_sha256(references_path),
        "states": int(len(states["observation"])),
        "draws": int(result.draw_attribution.shape[0]),
        "references_per_draw": int(result.reference_attribution.shape[1]),
        "effective_references": int(result.reference_attribution.shape[0] * result.reference_attribution.shape[1]),
        "pairwise_phase_action_comparisons": len(pairwise),
        "maximum_local_accuracy_residual": maximum_residual,
        "metrics": {
            "group_spearman_p05": float(np.quantile(spearman, 0.05)),
            "group_spearman_median": float(np.median(spearman)),
            "group_share_l1_median": float(np.median(share_l1)),
            "group_share_l1_p95": float(np.quantile(share_l1, 0.95)),
            "top_group_agreement_median": float(np.median(top)),
            "top_two_jaccard_p05": float(np.quantile(top2, 0.05)),
            "signed_agreement_median": float(np.median(signed)),
        },
        "checks": checks,
        "frozen_thresholds": thresholds,
        "pairwise": pairwise,
        "a1_a7_results_inspected": False,
    }
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        targets[0],
        draw_attribution=result.draw_attribution,
        mean_attribution=result.mean_attribution,
        efficiency_residual=result.efficiency_residual,
        reference_delta=result.reference_delta,
        public_phase=np.asarray(states["public_phase"]),
        group_names=np.asarray(("Lane", "Ego", "StopLine", "Pedestrian", "Stop", "PreviousAction")),
        action_names=np.asarray(("v_cmd_mps", "omega_cmd_rad_s")),
    )
    targets[1].write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    freeze = {
        "schema_version": 1,
        "classification": "FROZEN" if classification == "PASS" else "UNRESOLVED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": str(config["_sha256"]),
        "calibration_metrics_sha256": file_sha256(targets[1]),
        "reference_calibration_classification": classification,
        "thresholds": thresholds,
        "threshold_source": "A0 self-consistency across six independent same-phase reference draws",
        "a1_a7_results_inspected": False,
    }
    targets[2].write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    report = _report(metrics)
    timestamped = ROOT / "docs/F14_REFERENCE_CALIBRATION_20260815_140652.md"
    if timestamped.exists():
        raise RuntimeError("timestamped calibration report already exists")
    timestamped.write_text(report, encoding="utf-8")
    targets[3].write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if classification != "PASS":
        raise RuntimeError("A0 reference robustness inadequate; A1-A7 remains blocked")
    return metrics


def _report(metrics: dict[str, Any]) -> str:
    m = metrics["metrics"]
    t = metrics["frozen_thresholds"]
    return f"""# F14 A0 Reference Calibration

Classification: **{metrics['classification']}**

This development-only stage used 500 public 29D states (100 per frozen phase),
six independent draws, and four complete same-phase cross-seed reference rows
per draw. No A1–A7 Shapley result was inspected before this result and threshold
freeze.

## A0 self-consistency

- group Spearman: median `{m['group_spearman_median']:.6f}`, P05 `{m['group_spearman_p05']:.6f}`;
- group-share L1: median `{m['group_share_l1_median']:.6f}`, P95 `{m['group_share_l1_p95']:.6f}`;
- top-group agreement median: `{m['top_group_agreement_median']:.6f}`;
- top-two Jaccard P05: `{m['top_two_jaccard_p05']:.6f}`;
- signed agreement median: `{m['signed_agreement_median']:.6f}`;
- maximum exact-Shapley local-accuracy residual: `{metrics['maximum_local_accuracy_residual']:.3e}`.

## Frozen preservation thresholds

- minimum group Spearman: `{t['minimum_group_spearman']:.6f}`;
- maximum group-share L1: `{t['maximum_group_share_l1']:.6f}`;
- minimum top-group agreement: `{t['minimum_top_group_agreement']:.6f}`;
- minimum top-two Jaccard: `{t['minimum_top_two_jaccard']:.6f}`;
- minimum signed agreement: `{t['minimum_signed_agreement']:.6f}`;
- structurally preserved phase/action cells: at least `{t['minimum_preserved_phase_action_cells']}` of `{t['total_phase_action_cells']}`.

These values are frozen before A1–A7 evaluation and will not be weakened after
compression results are observed.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()

