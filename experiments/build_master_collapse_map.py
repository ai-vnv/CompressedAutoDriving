#!/usr/bin/env python3
"""Build the master collapse map across F12, F15, and F16.

One machine-readable optimization history. Every row carries its own evaluation
provenance, because the three experiments used different seed blocks and different
evaluation backends:

  F15 localization   seeds 180001-180008   non-deterministic backend
  F15 recovery       seeds 180201-180208   non-deterministic backend
  F16 candidates     seeds 181201-181208   cuda_strict_deterministic
  F16 2x2 diagnostic seeds 180201-180208   cuda_strict_deterministic

Rows from different blocks are NOT directly comparable, and the map marks that instead of
hiding it. Parallel construction branches are never flattened into a fictitious linear
history: each row records its actual predecessor.

Reads only. Writes collapse_map.csv and collapse_map.json into the F16 namespace.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root, load_config, provenance, read_csv, read_json,
    summarize_episode_dicts, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
F15 = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
F12 = ROOT / "artifacts/f12_belief_ppo_compression_v1"
CUR = ["c0", "c1", "c2", "c3", "c4"]

METRICS = [
    "completion_rate", "mean_progress_m", "lane_failure_rate", "invalid_pose_rate",
    "collision_rate", "unsafe_episode_rate", "stop_completion_rate", "stop_violation_rate",
    "restart_rate", "timeout_rate", "minimum_pedestrian_clearance_m",
    "mean_v_cmd_mps", "mean_abs_omega_cmd_rad_s", "stationary_fraction",
]

# Actual construction lineage. Parallel branches keep their real predecessor.
LINEAGE = {
    "A0":  {"stage": "Original Policy", "predecessor": None, "branch": "reference"},
    "A1":  {"stage": "Pruning Only", "predecessor": "A0", "branch": "historical_prune"},
    "A2":  {"stage": "Pruning + historical C4-focused KD", "predecessor": "A1", "branch": "historical_main"},
    "A3":  {"stage": "PTQ of Original", "predecessor": "A0", "branch": "historical_quant_only"},
    "A4":  {"stage": "QAT + KD, unpruned", "predecessor": "A3", "branch": "historical_quant_only"},
    "A5":  {"stage": "Pruning + PTQ", "predecessor": "A1", "branch": "historical_prune_quant"},
    "A6":  {"stage": "Pruning + KD + PTQ", "predecessor": "A2", "branch": "historical_main"},
    "A7":  {"stage": "Historical final INT8 (PDQD)", "predecessor": "A6", "branch": "historical_main"},
    "P192": {"stage": "Pruning only, width 192", "predecessor": "A0", "branch": "historical_frontier"},
    "P128": {"stage": "Pruning only, width 128", "predecessor": "A0", "branch": "historical_frontier"},
    "P96": {"stage": "Pruning only, width 96", "predecessor": "A0", "branch": "historical_frontier"},
    "P64": {"stage": "Pruning only, width 64", "predecessor": "A0", "branch": "historical_frontier"},
    "PD192": {"stage": "Pruning + historical KD, width 192", "predecessor": "P192", "branch": "historical_frontier"},
    "PD128": {"stage": "Pruning + historical KD, width 128", "predecessor": "P128", "branch": "historical_frontier"},
    "PD96": {"stage": "Pruning + historical KD, width 96", "predecessor": "P96", "branch": "historical_frontier"},
    "PD64": {"stage": "Pruning + historical KD, width 64", "predecessor": "P64", "branch": "historical_frontier"},
    "R64": {"stage": "Pruning + balanced C0-C4 KD, width 64", "predecessor": "A1", "branch": "f15_recovery"},
    "R64_PTQ": {"stage": "Recovered 64 + PTQ", "predecessor": "R64", "branch": "f15_recovery"},
    "R64_QAT": {"stage": "Recovered 64 + balanced QAT+KD", "predecessor": "R64", "branch": "f15_recovery"},
}


def rows_from_f15_results(path: Path, block: str, backend: str, experiment: str) -> list[dict]:
    if not path.exists():
        return []
    payload = read_json(path)
    out = []
    for model_id, curricula in payload["decisions"].items():
        for curriculum, decision in curricula.items():
            summary = payload["summaries"][model_id][curriculum]
            lineage = LINEAGE.get(model_id, {})
            out.append({
                "experiment": experiment, "model_id": model_id,
                "stage": lineage.get("stage", model_id),
                "predecessor": lineage.get("predecessor"),
                "branch": lineage.get("branch", "unknown"),
                "sequence": "n/a", "target_width": None, "precision": None,
                "training_realization": "historical",
                "curriculum": curriculum.upper(), "status": decision["status"],
                "evaluation_seed_block": block, "evaluation_backend": backend,
                "episodes": summary.get("episodes"),
                **{m: summary.get(m) for m in METRICS},
            })
    return out


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    rows: list[dict] = []

    # --- F15 localization: historical A0-A7 and the pruning frontier ---
    rows += rows_from_f15_results(
        F15 / "localization/matrix_results.json",
        "180001-180008", "non_deterministic", "F15_localization")
    rows += rows_from_f15_results(
        F15 / "localization/pruning_results.json",
        "180001-180008", "non_deterministic", "F15_localization_frontier")

    # --- F15 recovery selection: balanced-KD recovery and its quantizations ---
    for name, experiment in (("selection_fp32_w64_results.json", "F15_recovery"),
                             ("selection_ptq_w64_results.json", "F15_recovery"),
                             ("selection_qat_w64_results.json", "F15_recovery")):
        rows += rows_from_f15_results(
            F15 / "recovery" / name, "180201-180208", "non_deterministic", experiment)

    # --- F16: recomputed from episodes under the frozen deterministic backend ---
    from duckie_pomdp.optimization.cross_curriculum_recovery import retention_decision

    loop = root / "closed_loop"
    baseline_rows = read_csv(loop / "baseline_episodes.csv")
    baseline = {c: summarize_episode_dicts([r for r in baseline_rows if r["curriculum"] == c]) for c in CUR}
    registry = read_json(root / "candidate_registry.json")["candidates"]

    f16_sources = [("baseline_episodes.csv", "A0", "Original Policy", "reference", None, "FP32", "n/a", "181201-181208")]
    for path in sorted(loop.glob("fp32_*_episodes.csv")):
        name = path.name[len("fp32_"):-len("_episodes.csv")]
        block = "180201-180208" if name.endswith("_on_f15_selection") else "181201-181208"
        candidate = name.replace("_on_f15_selection", "")
        record = registry.get(candidate, {})
        f16_sources.append((path.name, candidate, record.get("sequence", "?") + f" prune to {record.get('target_width')}",
                            "f16_" + str(record.get("sequence", "?")).lower(),
                            record.get("target_width"), "FP32",
                            record.get("realization", "S1"), block))
    for path in sorted(loop.glob("transfer_*_episodes.csv")):
        block = "180201-180208" if "_on_f15_selection" in path.name else "181201-181208"
        f16_sources.append((path.name, "F15R64", "F15 recovered 64 (transfer check)",
                            "f16_diagnostic", 64, "FP32", "historical", block))

    for filename, model_id, stage, branch, width, precision, realization, block in f16_sources:
        path = loop / filename
        if not path.exists():
            continue
        episodes = read_csv(path)
        present = [c for c in CUR if any(r["curriculum"] == c for r in episodes)]
        if not present:
            continue
        summaries = {c: summarize_episode_dicts([r for r in episodes if r["curriculum"] == c]) for c in present}
        seeds_expected = 8
        for curriculum in present:
            n = sum(1 for r in episodes if r["curriculum"] == curriculum)
            if model_id == "A0" and filename == "baseline_episodes.csv":
                status = "REFERENCE"
            elif n < seeds_expected:
                status = f"PARTIAL_{n}/{seeds_expected}"
            else:
                decision = retention_decision(
                    curriculum, summaries[curriculum], baseline[curriculum],
                    config["retention"]["absolute"], config["retention"]["relative_to_original"],
                    candidate_prior=summaries, original_prior=baseline)
                dd = decision if isinstance(decision, dict) else decision.__dict__
                status = dd.get("status")
                failed = [k for grp in ("absolute_checks", "relative_checks")
                          for k, v in (dd.get(grp) or {}).items() if not v]
            rows.append({
                "experiment": "F16", "model_id": model_id, "stage": stage,
                "predecessor": "A0", "branch": branch,
                "sequence": registry.get(model_id, {}).get("sequence", "n/a"),
                "target_width": width, "precision": precision,
                "training_realization": realization,
                "curriculum": curriculum.upper(), "status": status,
                "evaluation_seed_block": block,
                "evaluation_backend": "cuda_strict_deterministic",
                "episodes": n,
                "failed_checks": "|".join(failed) if status == "FAIL" else "",
                **{m: summaries[curriculum].get(m) for m in METRICS},
            })

    # --- first collapse per curriculum along each real branch ---
    def status_of(model_id, curriculum, block):
        for r in rows:
            if r["model_id"] == model_id and r["curriculum"] == curriculum.upper() \
                    and r["evaluation_seed_block"] == block:
                return r["status"]
        return None

    first_collapse = {}
    historical_path = ["A0", "A1", "A2", "A7"]
    for curriculum in CUR:
        chain, collapse = [], None
        for model_id in historical_path:
            chain.append((model_id, status_of(model_id, curriculum, "180001-180008")))
        for (prev_id, prev), (next_id, nxt) in zip(chain, chain[1:]):
            if prev in {"REFERENCE", "PASS"} and nxt == "FAIL":
                collapse = {"transition": f"{prev_id}->{next_id}",
                            "after_stage": LINEAGE[next_id]["stage"]}
                break
        first_collapse[curriculum.upper()] = {
            "branch": "historical_main", "chain": [c[0] for c in chain],
            "statuses": [c[1] for c in chain], "first_collapse": collapse,
            "evaluation_seed_block": "180001-180008",
            "evaluation_backend": "non_deterministic",
        }

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "collapse_map.csv"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_json(out_dir / "collapse_map.json", {
        **provenance(config, CONFIG),
        "rows": len(rows),
        "csv": str(csv_path), "csv_sha256": file_sha256(csv_path),
        "first_collapse_by_curriculum": first_collapse,
        "comparability_warning": (
            "Rows from different evaluation_seed_block or evaluation_backend values are NOT "
            "directly comparable. F15 rows were produced under a non-deterministic backend "
            "whose run-to-run variation was measured at up to one flipped outcome label per "
            "eight episodes; F16 rows are bit-exactly reproducible. Cross-block differences "
            "must not be attributed to an optimization factor."
        ),
        "branches_not_flattened": sorted({r["branch"] for r in rows}),
        "evaluation_blocks": sorted({r["evaluation_seed_block"] for r in rows}),
    })

    print(f"wrote {csv_path.name}: {len(rows)} rows")
    print(f"branches: {sorted({r['branch'] for r in rows})}")
    print(f"evaluation blocks: {sorted({r['evaluation_seed_block'] for r in rows})}")
    print()
    print("=== FIRST COLLAPSE, historical main path A0->A1->A2->A7 (F15 localization block) ===")
    for curriculum, info in first_collapse.items():
        collapse = info["first_collapse"]
        text = f"{collapse['transition']}  after {collapse['after_stage']}" if collapse else "no collapse observed"
        print(f"  {curriculum}: {info['statuses']}  ->  {text}")


if __name__ == "__main__":
    main()
