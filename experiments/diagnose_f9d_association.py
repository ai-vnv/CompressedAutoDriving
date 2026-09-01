"""Gate F9d Task 2: the association selection-rule diagnostic.

Cache-only. Constructs no ``YoloObjectDetector``, no gym-duckietown simulator
-- every candidate this script scores came from the ORIGINAL 2026-08
final-evaluation render (seeds 7101-7104, 3328 frames), replayed from
``artifacts/f9c_runtime_cache.npz``/``artifacts/f9c_evaluation_truth.npz``.
Both hashes are checked BEFORE any read, against the same values
``experiments/evaluate_f9c_robust_belief.py`` records from that render, so a
silently regenerated or corrupted cache is refused rather than replayed as
if it were the run that actually happened.

Answers two separable questions (see ``duckie_pomdp.evaluation.f9d_association``
module docstring for the full argument):

    C1  does the wrong innovation covariance (lambda = 1, what
        temporal_association_only actually scored against with
        covariance_calibration off) explain the ablation penalty, once the
        predicted state is held fixed to the real frozen Robust B
        trajectory?
    C2  even at the correct, frozen lambda_r, does minimum-NIS selection
        pick worse boxes than highest-confidence selection, on duplicate
        frames?

This is a diagnostic. It cannot change gate F9d's or F9c's status by itself
-- it never writes to any belief-layer module or frozen config -- and it
prints/records counts only, no subjective language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duckie_pomdp.evaluation.f9c_runtime_cache import read_evaluation_truth, read_runtime_cache
from duckie_pomdp.evaluation.f9d_association import compare_selection_rules, duplicate_frame_ranking
from duckie_pomdp.evaluation.f9_protocol import sha256

ROOT = Path(__file__).resolve().parents[1]

# Recorded in task-11-report.md from the ORIGINAL 2026-08 final-evaluation
# render -- identical to experiments/evaluate_f9c_robust_belief.py's own
# EXPECTED_RUNTIME_CACHE_SHA256/EXPECTED_EVALUATION_TRUTH_SHA256 constants.
# Duplicated here rather than imported: this script must not construct
# anything evaluate_f9c_robust_belief.py's module scope would (it imports
# YoloObjectDetector at module scope), so it never imports that module at
# all -- see test_the_diagnostic_constructs_no_detector_and_no_simulator.
EXPECTED_RUNTIME_CACHE_SHA256 = (
    "fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d"
)
EXPECTED_EVALUATION_TRUTH_SHA256 = (
    "26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9"
)

RUNTIME_CACHE_PATH = ROOT / "artifacts" / "f9c_runtime_cache.npz"
EVALUATION_TRUTH_PATH = ROOT / "artifacts" / "f9c_evaluation_truth.npz"
OUTPUT_PATH = ROOT / "artifacts" / "f9d_association_diagnostic.json"


def _print_comparison(title: str, result: dict[str, Any]) -> None:
    print(f"\n=== {title}: {result['rule_a']} vs {result['rule_b']} ===")
    print(f"frames compared:      {result['frames_compared']}")
    print(f"selections agree:     {result['selections_agree']}")
    print(f"selections differ:    {result['selections_differ']}")
    paired = result["differing_frames_paired"]
    print("among the DIFFERING frames only (paired):")
    print(f"  rule A ({result['rule_a']}) picks the higher-IoU box: {paired['rule_a_higher_iou']}")
    print(f"  rule B ({result['rule_b']}) picks the higher-IoU box: {paired['rule_b_higher_iou']}")
    print(f"  tie:                                                  {paired['tie']}")
    excluded = result["differing_frames_excluded_from_pairing"]
    print(
        "differing frames excluded from pairing (one side made no selection: "
        f"{excluded['one_side_made_no_selection']}, neither side has GT: "
        f"{excluded['neither_side_has_gt']})"
    )
    outliers = result["localization_outlier_count"]
    print(f"localization-outlier count: rule A / rule B = {outliers['rule_a']} / {outliers['rule_b']}")
    mean_iou = result["mean_selected_iou"]
    print(f"mean selected IoU:          rule A / rule B = {mean_iou['rule_a']} / {mean_iou['rule_b']}")
    median_iou = result["median_selected_iou"]
    print(
        "median selected IoU:        rule A / rule B = "
        f"{median_iou['rule_a']} / {median_iou['rule_b']}"
    )
    print(
        "resolution: with rules agreeing on "
        f"{result['selections_agree_fraction']:.4f} of frames_compared, a true "
        "selection-quality difference smaller than roughly "
        f"{result['minimum_detectable_selection_difference_fraction']:.4f} is not "
        "detectable on this cache regardless of which rule is actually better"
    )

    if "abstention" in result:
        # C1 only (fix round 1): abstention is a headline measure, not an
        # exclusion -- see f9d_association._c1_abstention's docstring.
        abstention = result["abstention"]
        print("\nabstention (fix round 1):")
        print(
            "  frames where association selected NOTHING: rule A / rule B = "
            f"{abstention['frames_rule_a_selected_nothing']} / "
            f"{abstention['frames_rule_b_selected_nothing']}"
        )
        print(
            "  rule A abstained, rule B selected: "
            f"{abstention['rule_a_abstained_rule_b_selected']}"
        )
        print(
            "  rule B abstained, rule A selected: "
            f"{abstention['rule_b_abstained_rule_a_selected']}"
        )
        print(
            "  of those, GT-comparable box available: "
            f"{abstention['one_sided_abstention_frames_with_gt_available']} / "
            f"{abstention['one_sided_abstention_frame_count']}"
        )
        mean_nis = abstention["mean_selected_nis"]
        print(f"  mean NIS of the winning candidate: rule A / rule B = {mean_nis['rule_a']} / {mean_nis['rule_b']}")
        gate = abstention["candidate_gate_exceedance_fraction"]
        print(
            "  fraction of candidates exceeding the association gate: "
            f"rule A / rule B = {gate['rule_a']} / {gate['rule_b']}"
        )
        conclusion_selection = result["conclusion_selection"]
        conclusion_abstention = result["conclusion_abstention"]
        print(
            f"conclusion (C1-selection):  {conclusion_selection['verdict']} -- "
            f"{conclusion_selection['question']}"
        )
        print(
            f"conclusion (C1-abstention): {conclusion_abstention['verdict']} -- "
            f"{conclusion_abstention['question']}"
        )
    else:
        conclusion = result["conclusion"]
        print(f"conclusion: {conclusion['verdict']} -- {conclusion['question']}")


def main() -> None:
    runtime_cache_sha256 = sha256(RUNTIME_CACHE_PATH)
    if runtime_cache_sha256 != EXPECTED_RUNTIME_CACHE_SHA256:
        raise ValueError(
            f"runtime cache {RUNTIME_CACHE_PATH} (sha256={runtime_cache_sha256}) does not "
            f"match the expected hash {EXPECTED_RUNTIME_CACHE_SHA256} from the original "
            "final-evaluation render -- refusing to replay a cache that may not be the "
            "one that render produced"
        )
    evaluation_truth_sha256 = sha256(EVALUATION_TRUTH_PATH)
    if evaluation_truth_sha256 != EXPECTED_EVALUATION_TRUTH_SHA256:
        raise ValueError(
            f"evaluation truth {EVALUATION_TRUTH_PATH} (sha256={evaluation_truth_sha256}) "
            f"does not match the expected hash {EXPECTED_EVALUATION_TRUTH_SHA256}"
        )

    cache = read_runtime_cache(RUNTIME_CACHE_PATH, expected_sha256=runtime_cache_sha256)
    truth = read_evaluation_truth(EVALUATION_TRUTH_PATH, expected_sha256=evaluation_truth_sha256)

    c1 = compare_selection_rules(cache, truth)
    c2 = duplicate_frame_ranking(cache, truth)

    _print_comparison("C1 (S-scale hypothesis)", c1)
    _print_comparison("C2 (min-NIS vs highest-confidence, at the frozen lambda)", c2)

    report = {
        "schema_version": 1,
        "gate": "F9d",
        "task": "association diagnostic (cache-only)",
        "provenance": {
            "runtime_cache_path": str(RUNTIME_CACHE_PATH),
            "runtime_cache_sha256": runtime_cache_sha256,
            "evaluation_truth_path": str(EVALUATION_TRUTH_PATH),
            "evaluation_truth_sha256": evaluation_truth_sha256,
            "frame_count": len(cache),
        },
        "c1_lambda_scale_hypothesis": c1,
        "c2_min_nis_vs_highest_confidence": c2,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "c1_selection_verdict": c1["conclusion_selection"]["verdict"],
                "c1_abstention_verdict": c1["conclusion_abstention"]["verdict"],
                "c2_verdict": c2["conclusion"]["verdict"],
                "output": str(OUTPUT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
