#!/usr/bin/env python3
"""Record the F17 eligibility outcome. No candidate is frozen; the holdout stays sealed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
from run_f15_cross_curriculum_recovery import load_config, provenance, read_json, write_json  # noqa: E402

CONFIG = ROOT / "configs/f17_optimization_method_order_v1.toml"


def main() -> None:
    config = load_config(CONFIG)
    root = (CONFIG.parent / config["artifacts"]["directory"]).resolve()
    target = root / "results/eligibility_outcome.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    summary = read_json(root / "results/pathway_summary.json")
    pathways = summary["pathways"]

    eligible = [p for p, v in pathways.items() if v["eligible"]]
    payload = {
        **provenance(config, CONFIG),
        "classification": "NO_ELIGIBLE_FINAL_CANDIDATE" if not eligible else "CANDIDATE_AVAILABLE",
        "eligible_pathways": eligible,
        "eligibility_rule": ("INT8 + all frozen behaviour gates + all frozen same-state fidelity "
                             "gates + safety + provenance; diagnostics never affect it"),
        "per_pathway": {
            p: {"behaviour_all": v["behaviour_all_curricula_pass"],
                "fidelity_all": v["fidelity_all_curricula_pass"],
                "int8": v["precision"] == "INT8",
                "eligible": v["eligible"],
                "statuses": v["statuses"]}
            for p, v in sorted(pathways.items())
        },
        "closest_candidates": {
            "A4": ("behaviour PASS on all five curricula as INT8 (width 256; 2.7x file and ~2.6x "
                   "latency compression vs FP32 Original per F12 benchmarks) but fails the frozen "
                   "same-state fidelity gate on correlation checks in 4/5 curricula, replicating "
                   "the historical F12 result for unpruned PTQ"),
            "A6_A8": ("width-64 INT8 candidates retain C0-C2 but fail C3/C4 behaviour with "
                      "opposite phenotypes (stop-freeze under PTQ, stop-violation under QAT+KD)"),
        },
        "headline_finding": (
            "The INT8 C3/C4 retention failure is an interaction between narrow width and "
            "quantization, not an effect of quantization alone: pruning+balanced-KD passes all "
            "five curricula in FP32 (A3), PTQ of the unpruned Original passes all five in INT8 "
            "behaviour (A4), and only their combination fails (A6/A8)."
        ),
        "fidelity_gate_note": (
            "A4's failing checks are Pearson/Spearman correlation components, whose fragility on "
            "low-variance signals is a documented metric limitation. The gates are frozen and "
            "were not modified; eligibility is reported under the frozen gates."
        ),
        "final_holdout_opened": False,
        "final_holdout_seeds_sealed": [int(s) for s in config["seeds"]["sealed_final_holdout"]],
        "no_endless_tuning": "the quantization procedure was fixed; no additional tuning was run to force a PASS",
        "results_sha256": {
            name: file_sha256(root / "results" / name)
            for name in ("pathway_results.csv", "same_state_fidelity.csv", "pathway_summary.json")
        },
    }
    write_json(target, payload)
    print(json.dumps({"classification": payload["classification"],
                      "eligible_pathways": eligible,
                      "holdout_opened": False}, indent=2))


if __name__ == "__main__":
    main()
