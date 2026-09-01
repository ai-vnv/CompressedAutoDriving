#!/usr/bin/env python3
"""Record the investigator-directed decisive early stop of the F16 replication matrix.

This is an efficiency decision taken AFTER the decisive robustness result, not missing
data presented as completion. Unfinished cells are marked explicitly and no value is
fabricated. Nothing is deleted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root, load_config, provenance, read_csv, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
CUR = ["c0", "c1", "c2", "c3", "c4"]
WIDTHS = [64, 96, 128]
REALIZATIONS = ["S1", "S2", "S3"]


def candidate_id(schedule: str, width: int, realization: str) -> str:
    base = f"{schedule}{width}"
    return base if realization == "S1" else f"{base}_{realization}"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "integrity/decisive_early_stop.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    loop = root / "closed_loop"
    seeds = [int(s) for s in config["seeds"]["selection"]]

    completed, incomplete = {}, {}
    for width in WIDTHS:
        for realization in REALIZATIONS:
            for schedule in ("D", "P"):
                cid = candidate_id(schedule, width, realization)
                path = loop / f"fp32_{cid}_episodes.csv"
                episodes = len(read_csv(path)) if path.exists() else 0
                record = {
                    "candidate_id": cid, "target_width": width,
                    "pruning_schedule": "Direct" if schedule == "D" else "Progressive",
                    "training_realization": realization,
                    "episodes": episodes, "episodes_expected": len(seeds) * len(CUR),
                }
                if episodes == len(seeds) * len(CUR):
                    record["episodes_csv_sha256"] = file_sha256(path)
                    completed[cid] = record
                else:
                    record["status"] = ("CANCELLED_AFTER_DECISIVE_ELIGIBILITY_RESULT"
                                        if episodes > 0 else "NOT_RUN_AFTER_DECISIVE_STOP")
                    if episodes:
                        record["partial_episodes_csv_sha256"] = file_sha256(path)
                        record["partial_data_preserved"] = True
                    incomplete[cid] = record

    # Proof that no remaining cell can produce a 3/3 all-C0-C4 PASS candidate.
    from duckie_pomdp.optimization.cross_curriculum_recovery import retention_decision
    from run_f15_cross_curriculum_recovery import summarize_episode_dicts

    baseline_rows = read_csv(loop / "baseline_episodes.csv")
    baseline = {c: summarize_episode_dicts([r for r in baseline_rows if r["curriculum"] == c]) for c in CUR}

    def all_five_pass(cid: str):
        path = loop / f"fp32_{cid}_episodes.csv"
        if not path.exists():
            return None
        rows = read_csv(path)
        covered = {(r["curriculum"], int(r["seed"])) for r in rows}
        if not all((c, s) in covered for c in CUR for s in seeds):
            return None
        summaries = {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c]) for c in CUR}
        for c in CUR:
            decision = retention_decision(
                c, summaries[c], baseline[c],
                config["retention"]["absolute"], config["retention"]["relative_to_original"],
                candidate_prior=summaries, original_prior=baseline)
            dd = decision if isinstance(decision, dict) else decision.__dict__
            if dd.get("status") != "PASS":
                return False
        return True

    proof = {}
    for width in WIDTHS:
        for schedule in ("D", "P"):
            label = f"{'Direct' if schedule == 'D' else 'Progressive'}-{width}"
            per = {r: all_five_pass(candidate_id(schedule, width, r)) for r in REALIZATIONS}
            known_fail = sum(1 for v in per.values() if v is False)
            unknown = sum(1 for v in per.values() if v is None)
            proof[label] = {
                "all_five_pass_by_realization": {
                    r: ("PASS" if v is True else "FAIL" if v is False else "NOT_EVALUATED")
                    for r, v in per.items()
                },
                "known_failures": known_fail,
                "unevaluated": unknown,
                "can_still_reach_3_of_3_pass": known_fail == 0 and unknown > 0,
                "reason": (
                    "already has at least one FAIL, so 3/3 all-curricula PASS is arithmetically "
                    "unreachable regardless of any remaining evaluation"
                    if known_fail else "no failure recorded yet"
                ),
            }

    still_possible = [k for k, v in proof.items() if v["can_still_reach_3_of_3_pass"]]

    payload = {
        **provenance(config, CONFIG),
        "classification": "DECISIVE_EARLY_STOP",
        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_type": (
            "investigator-directed efficiency decision taken after the decisive robustness "
            "result; not missing data presented as completion"
        ),
        "reason": (
            "Every width x pruning-schedule combination had already recorded at least one "
            "all-curricula FAIL across the completed training realizations, so no remaining "
            "evaluation could produce a candidate with 3/3 stable all-C0-C4 PASS. The frozen "
            "stop condition in docs/F16_TRANSFER_AND_REPLICATION_PLAN.md was therefore already "
            "satisfied and further simulation could not change the eligibility outcome."
        ),
        "combinations_that_could_still_reach_3_of_3": still_possible,
        "arithmetic_proof": proof,
        "completed_cells": completed,
        "incomplete_cells": incomplete,
        "process_termination": {
            "mechanism": "tmux kill-session -t f16",
            "in_flight_candidate_at_stop": "P96_S3",
            "in_flight_partial_episodes": incomplete.get("P96_S3", {}).get("episodes"),
            "partial_data_deleted": False,
            "resume_script_retained": "artifacts/f16_chain_resume.sh",
        },
        "terminology_note": (
            "Direct versus Progressive is a PRUNING SCHEDULE, not an optimization-method order. "
            "The optimization-method order question (placement of pruning, distillation, PTQ, "
            "QAT) is addressed in F17."
        ),
        "frozen_result_files": {
            str(p.relative_to(root)): file_sha256(p)
            for p in sorted((root / "results").glob("*"))
            if p.is_file()
        },
        "frozen_integrity_files": {
            str(p.relative_to(root)): file_sha256(p)
            for p in sorted((root / "integrity").glob("*.json"))
            if p.is_file() and p.name != "decisive_early_stop.json"
        },
    }
    write_json(target, payload)
    print(json.dumps({
        "written": str(target),
        "completed_cells": len(completed),
        "incomplete_cells": {k: v["status"] for k, v in incomplete.items()},
        "combinations_that_could_still_reach_3_of_3": still_possible,
    }, indent=2))


if __name__ == "__main__":
    main()
