#!/usr/bin/env python3
"""Analyze F16 training-seed replication and classify the sequence question.

Applies the rule frozen in docs/F16_SEQUENCE_DISCORDANCE_RULE.md:

  same direction in all or nearly all realizations -> SEQUENCE EFFECT SUPPORTED
  direction changes between realizations           -> TRAINING-SEED SENSITIVE / INCONCLUSIVE
  Direct and Progressive never disagree            -> NO MATERIAL SEQUENCE EFFECT DETECTED

Also reports per-width FP32 verdict stability across realizations. A width whose
all-five-curricula verdict changes between training seeds is TRAINING-REALIZATION
SENSITIVE, and a later INT8 failure at that width may not be called purely
quantization-associated.

Failures are additionally classified by severity, because the frozen gates treat a
marginal clearance-only miss and a genuine safety collapse as the same FAIL. Severity is
descriptive only and never alters a verdict.

Emits training_realization_results.csv, width_results.csv, sequence_classification.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256, retention_decision,
)
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root, load_config, provenance, read_csv, read_json,
    summarize_episode_dicts, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
CUR = ["c0", "c1", "c2", "c3", "c4"]
WIDTHS = [64, 96, 128]
REALIZATIONS = ["S1", "S2", "S3"]

SAFETY_CHECKS = {
    "maximum_collision_rate", "new_collisions", "collision_rate",
    "maximum_stop_violation_rate", "stop_violation_rate",
    "maximum_unsafe_episode_rate", "unsafe_episode_rate",
}


def severity(failed: list[str]) -> str:
    """Descriptive severity of a FAIL. Never changes the verdict."""
    if not failed:
        return "none"
    if set(failed) <= {"minimum_clearance", "maximum_minimum_clearance_drop_m"}:
        return "marginal_clearance_only"
    if set(failed) & SAFETY_CHECKS:
        return "safety_relevant"
    return "behavioural"


def candidate_id(sequence: str, width: int, realization: str) -> str:
    base = f"{sequence}{width}"
    return base if realization == "S1" else f"{base}_{realization}"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    loop = root / "closed_loop"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_csv(loop / "baseline_episodes.csv")
    baseline = {c: summarize_episode_dicts([r for r in baseline_rows if r["curriculum"] == c]) for c in CUR}
    seeds = [int(s) for s in config["seeds"]["selection"]]

    def verdicts(cid: str):
        path = loop / f"fp32_{cid}_episodes.csv"
        if not path.exists():
            return None
        rows = read_csv(path)
        covered = {(r["curriculum"], int(r["seed"])) for r in rows}
        if not all((c, s) in covered for c in CUR for s in seeds):
            return None
        summaries = {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c]) for c in CUR}
        out = {}
        for c in CUR:
            decision = retention_decision(
                c, summaries[c], baseline[c],
                config["retention"]["absolute"], config["retention"]["relative_to_original"],
                candidate_prior=summaries, original_prior=baseline)
            dd = decision if isinstance(decision, dict) else decision.__dict__
            failed = [k for grp in ("absolute_checks", "relative_checks")
                      for k, v in (dd.get(grp) or {}).items() if not v]
            out[c] = {
                "status": dd.get("status"), "failed_checks": failed,
                "severity": severity(failed),
                "completion_rate": summaries[c]["completion_rate"],
                "mean_progress_m": summaries[c]["mean_progress_m"],
                "stop_violation_rate": summaries[c]["stop_violation_rate"],
                "collision_rate": summaries[c]["collision_rate"],
                "minimum_pedestrian_clearance_m": summaries[c].get("minimum_pedestrian_clearance_m"),
            }
        return out

    rows_out, width_rows = [], []
    per_cell: dict[tuple[int, str, str], dict] = {}
    for width in WIDTHS:
        for realization in REALIZATIONS:
            for sequence in ("D", "P"):
                cid = candidate_id(sequence, width, realization)
                v = verdicts(cid)
                if v is None:
                    continue
                per_cell[(width, realization, sequence)] = v
                for c in CUR:
                    rows_out.append({
                        "candidate_id": cid, "target_width": width,
                        "sequence": "Direct" if sequence == "D" else "Progressive",
                        "realization": realization, "curriculum": c.upper(),
                        "status": v[c]["status"], "severity": v[c]["severity"],
                        "failed_checks": "|".join(v[c]["failed_checks"]),
                        "completion_rate": v[c]["completion_rate"],
                        "mean_progress_m": v[c]["mean_progress_m"],
                        "stop_violation_rate": v[c]["stop_violation_rate"],
                        "collision_rate": v[c]["collision_rate"],
                        "minimum_pedestrian_clearance_m": v[c]["minimum_pedestrian_clearance_m"],
                    })

    # ---- sequence classification per width x curriculum, across realizations ----
    classification = {}
    for width in WIDTHS:
        per_curriculum = {}
        for c in CUR:
            observations = []
            for realization in REALIZATIONS:
                d = per_cell.get((width, realization, "D"))
                p = per_cell.get((width, realization, "P"))
                if d is None or p is None:
                    continue
                ds, ps = d[c]["status"], p[c]["status"]
                if ds == ps:
                    direction = "concordant"
                elif ds == "PASS":
                    direction = "direct_better"
                else:
                    direction = "progressive_better"
                observations.append({
                    "realization": realization, "direct": ds, "progressive": ps,
                    "direction": direction,
                    "direct_severity": d[c]["severity"], "progressive_severity": p[c]["severity"],
                })
            directions = {o["direction"] for o in observations}
            # A single training realization can never support a sequence effect: the frozen
            # rule requires a discordant cell to stay PROVISIONAL until it is replicated.
            if not observations:
                status = "NOT_YET_REPLICATED"
            elif len(observations) == 1:
                status = ("CONCORDANT_SINGLE_REALIZATION" if directions == {"concordant"}
                          else "PROVISIONAL_AWAITING_REPLICATION")
            elif directions == {"concordant"}:
                status = "NO MATERIAL SEQUENCE EFFECT DETECTED"
            elif len(directions) > 1:
                # Any mix — two opposite directions, or discordant in one realization and
                # concordant in another — means the direction is not stable.
                status = "TRAINING-SEED SENSITIVE / INCONCLUSIVE"
            else:
                status = "SEQUENCE EFFECT SUPPORTED"
            per_curriculum[c.upper()] = {
                "realizations_available": len(observations),
                "observations": observations, "classification": status,
            }
        classification[str(width)] = per_curriculum

    # ---- per-width FP32 stability across realizations ----
    stability = {}
    for width in WIDTHS:
        for sequence in ("D", "P"):
            verdict_by_realization = {}
            for realization in REALIZATIONS:
                v = per_cell.get((width, realization, sequence))
                if v is None:
                    continue
                all_pass = all(v[c]["status"] == "PASS" for c in CUR)
                verdict_by_realization[realization] = {
                    "all_five_pass": all_pass,
                    "failing_curricula": [c.upper() for c in CUR if v[c]["status"] != "PASS"],
                    "severities": {c.upper(): v[c]["severity"] for c in CUR if v[c]["severity"] != "none"},
                }
            values = {r["all_five_pass"] for r in verdict_by_realization.values()}
            if not verdict_by_realization:
                label = "NOT_YET_REPLICATED"
            elif len(verdict_by_realization) == 1:
                label = "SINGLE_REALIZATION_ONLY"
            elif len(values) > 1:
                label = "TRAINING-REALIZATION SENSITIVE"
            elif values == {True}:
                label = "STABLE_ALL_FIVE_PASS"
            else:
                label = "STABLE_FAILING"
            key = f"{'Direct' if sequence == 'D' else 'Progressive'}-{width}"
            stability[key] = {
                "target_width": width,
                "sequence": "Direct" if sequence == "D" else "Progressive",
                "per_realization": verdict_by_realization, "stability": label,
            }
            width_rows.append({
                "target_width": width, "sequence": stability[key]["sequence"],
                "realizations_evaluated": len(verdict_by_realization),
                "all_five_pass_by_realization": "|".join(
                    f"{r}:{'PASS' if v['all_five_pass'] else 'FAIL'}"
                    for r, v in sorted(verdict_by_realization.items())),
                "stability": label,
            })

    for name, rows in (("training_realization_results.csv", rows_out),
                       ("width_results.csv", width_rows)):
        if not rows:
            continue
        path = out_dir / name
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {name}: {len(rows)} rows")

    write_json(out_dir / "sequence_classification.json", {
        **provenance(config, CONFIG),
        "rule_document": "docs/F16_SEQUENCE_DISCORDANCE_RULE.md",
        "realizations": {"S1": 2026081701, "S2": 2026081801, "S3": 2026081802},
        "sequence_classification_by_width": classification,
        "fp32_stability_by_width_and_sequence": stability,
        "severity_is_descriptive_only": True,
        "no_pvalue_claimed": "patterns across a small number of training realizations are reported as patterns",
    })

    print()
    print("=== FP32 all-five-curricula verdict by realization ===")
    print(f"{'candidate':<16}{'S1':<7}{'S2':<7}{'S3':<7}  stability")
    for key, info in sorted(stability.items(), key=lambda kv: (kv[1]["target_width"], kv[0])):
        cells = []
        for realization in REALIZATIONS:
            v = info["per_realization"].get(realization)
            cells.append("-" if v is None else ("PASS" if v["all_five_pass"] else "FAIL"))
        print(f"{key:<16}" + "".join(c.ljust(7) for c in cells) + f"  {info['stability']}")

    print()
    print("=== sequence classification per width x curriculum ===")
    for width in WIDTHS:
        entries = classification[str(width)]
        interesting = {c: e for c, e in entries.items()
                       if e["classification"] not in ("NOT_YET_REPLICATED",)}
        if not interesting:
            continue
        print(f"  width {width}:")
        for c, e in interesting.items():
            dirs = ",".join(f"{o['realization']}:{o['direction']}" for o in e["observations"])
            print(f"    {c}: {e['classification']:<44} [{dirs}]")


if __name__ == "__main__":
    main()
