#!/usr/bin/env python3
"""Record the completed backend-matched 2x2 model-versus-evaluation-block diagnostic.

Writes a machine-readable result. Diagnostic only: no candidate, no eligibility, no gate
change. The sealed final holdout was never touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256, retention_decision,
)
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root, load_config, provenance, read_csv, summarize_episode_dicts, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
F15 = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
CUR = ["c0", "c1", "c2", "c3", "c4"]


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "results/model_vs_evalblock_2x2.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    loop = root / "closed_loop"

    def baseline(block: str) -> dict:
        if block == "NEW":
            rows = read_csv(loop / "baseline_episodes.csv")
        else:
            rows = [r for r in read_csv(F15 / "recovery/selection_baseline_episodes.csv")
                    if r["model_id"] == "A0"]
        return {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c]) for c in CUR}

    cells_spec = {
        ("F15R64", "OLD"): "transfer_F15R64_on_f15_selection_episodes.csv",
        ("F15R64", "NEW"): "transfer_F15R64_episodes.csv",
        ("D64", "OLD"): "fp32_D64_on_f15_selection_episodes.csv",
        ("D64", "NEW"): "fp32_D64_episodes.csv",
    }

    cells = {}
    for (model, block), filename in cells_spec.items():
        rows = read_csv(loop / filename)
        summaries = {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c]) for c in CUR}
        base = baseline(block)
        per_curriculum = {}
        for curriculum in CUR:
            decision = retention_decision(
                curriculum, summaries[curriculum], base[curriculum],
                config["retention"]["absolute"], config["retention"]["relative_to_original"],
                candidate_prior=summaries, original_prior=base)
            dd = decision if isinstance(decision, dict) else decision.__dict__
            per_curriculum[curriculum] = {
                "status": dd.get("status"),
                "failed_checks": [k for grp in ("absolute_checks", "relative_checks")
                                  for k, v in (dd.get(grp) or {}).items() if not v],
                "completion_rate": summaries[curriculum]["completion_rate"],
                "stop_violation_rate": summaries[curriculum]["stop_violation_rate"],
                "mean_progress_m": summaries[curriculum]["mean_progress_m"],
                "minimum_pedestrian_clearance_m": summaries[curriculum].get("minimum_pedestrian_clearance_m"),
            }
        cells[f"{model}|{block}"] = {
            "model": model, "evaluation_block": block, "episodes_csv": filename,
            "curricula": per_curriculum,
        }

    c4 = {k: v["curricula"]["c4"] for k, v in cells.items()}
    payload = {
        **provenance(config, CONFIG),
        "classification": "COMPLETE",
        "kind": "backend_matched_2x2_model_vs_evaluation_block_diagnostic",
        "backend": "cuda_strict_deterministic (all four cells)",
        "blocks": {
            "OLD": "F15 already-opened recovery-selection seeds 180201-180208",
            "NEW": "F16 selection seeds 181201-181208",
        },
        "sealed_holdout_touched": False,
        "is_candidate": False,
        "affects_eligibility": False,
        "cells": cells,
        "c4_summary_table": {
            k: {"status": v["status"], "completion": v["completion_rate"],
                "stop_violation": v["stop_violation_rate"],
                "clearance": v["minimum_pedestrian_clearance_m"],
                "failed": v["failed_checks"]}
            for k, v in c4.items()
        },
        "isolated_findings": {
            "f15_recovery_result_validated": (
                "The F15 recovered 64x64 checkpoint passes all five curricula on its own seed "
                "block under the deterministic backend, reproducing what F15 reported. The F15 "
                "conclusion was therefore not an artefact of the non-deterministic backend."
            ),
            "evaluation_block_sensitivity": (
                "With the model byte-identical and the backend fixed, changing only the eight "
                "evaluation seeds flips F15R64 C4 from PASS to FAIL, and does so solely through "
                "minimum_clearance (0.4513 m -> 0.4264 m). No other factor can account for it."
            ),
            "training_realization_sensitivity": (
                "D64 records completion 0.625 and stop-violation rate 0.500 on C4 on BOTH blocks, "
                "with identical values, while F15R64 records 1.000 and 0.000 on both. The stop "
                "defect is block-independent and is attributable to the training realization, not "
                "to the evaluation seeds."
            ),
            "the_two_effects_are_independent": (
                "They act on different metrics: the block effect appears in minimum_clearance, the "
                "training-realization effect in stop-violation and completion. Neither explains "
                "the other."
            ),
        },
        "metric_sensitivity_limitation": (
            "minimum_pedestrian_clearance_m compares the candidate's minimum over eight episodes "
            "against A0's minimum over the same eight episodes, with a 0.05 m allowance. It is a "
            "difference of two minimum statistics and is structurally the most seed-sensitive "
            "quantity in the frozen gate set. Reports must distinguish a clearance-only marginal "
            "failure from a behavioural or safety collapse. The gate is frozen and unmodified."
        ),
        "claim_limits": [
            "Not claimed: the training seed 'caused' the stop defect in a mechanistic sense; only "
            "that the training realization differs and the difference is block-independent.",
            "Not claimed: the clearance gate is wrong; only that it is seed-sensitive by construction.",
            "Not claimed: any conclusion about optimization sequence or width from this diagnostic.",
        ],
    }
    write_json(target, payload)
    print(json.dumps({
        "written": str(target),
        "c4": payload["c4_summary_table"],
    }, indent=2))


if __name__ == "__main__":
    main()
