"""F9d-A once-only natural localization-outlier stress evaluation.

The live render reuses Task 3's proven collection loop, which itself routes
every frame through F9c's shared ``_step_both_systems`` and ``build_row``.
Final seeds are protected by an artifact-existence guard: a completed or
partially cached final render is never silently overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import evaluate_f9c_robust_belief as f9c_eval  # noqa: E402
from probe_f9d_yield import (  # noqa: E402
    _ProbeSettings,
    _load_outlier_scenarios,
    collect_probe_rows,
)

from duckie_pomdp.evaluation.f9c_runtime_cache import (  # noqa: E402
    RuntimeCacheFrame,
    TruthFrame,
    write_evaluation_truth,
    write_runtime_cache,
)
from duckie_pomdp.evaluation.f9d_outlier_outcome import (  # noqa: E402
    outlier_outcome_metrics,
)
from duckie_pomdp.evaluation.f9d_protocol import (  # noqa: E402
    f9c_parameters,
    load_f9d_protocol,
    outlier_support_satisfied,
)
from duckie_pomdp.evaluation.f9d_stress import outlier_yield  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f9d_evidence_closure_v1.toml"


def _refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            "F9d-A final artifacts already exist; refusing to re-render final "
            f"seeds or overwrite recovery evidence: {joined}"
        )


def _support(protocol, summary: dict) -> dict:
    frames = int(summary["outlier_frames"])
    events = int(summary["outlier_events"])
    seeds = int(summary["seeds_with_event"])
    checks = {
        "frames": {
            "value": frames,
            "minimum": protocol.minimum_outlier_frames,
            "pass": frames >= protocol.minimum_outlier_frames,
        },
        "events": {
            "value": events,
            "minimum": protocol.minimum_outlier_events,
            "pass": events >= protocol.minimum_outlier_events,
        },
        "seeds": {
            "value": seeds,
            "minimum": protocol.minimum_outlier_seeds,
            "pass": seeds >= protocol.minimum_outlier_seeds,
        },
    }
    satisfied = outlier_support_satisfied(
        protocol, frames=frames, events=events, seeds=seeds
    )
    return {
        "checks": checks,
        "insufficient_frame_floor": protocol.insufficient_outlier_frames,
        "satisfied": satisfied,
        "decision": "SUPPORTED" if satisfied else "INSUFFICIENT_EVIDENCE",
    }


def _paths(protocol, development_dir: Path | None) -> dict[str, Path]:
    if development_dir is None:
        return {
            "csv": protocol.artifacts["outlier_csv"],
            "metrics": protocol.artifacts["outlier_metrics_json"],
            "cache": protocol.artifacts["outlier_runtime_cache"],
            "truth": protocol.artifacts["outlier_evaluation_truth"],
        }
    development_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": development_dir / "f9d_outlier_development.csv",
        "metrics": development_dir / "f9d_outlier_development_metrics.json",
        "cache": development_dir / "f9d_outlier_development_cache.npz",
        "truth": development_dir / "f9d_outlier_development_truth.npz",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--development-output", type=Path, default=Path("/tmp/f9d-task6-smoke"))
    args = parser.parse_args()

    protocol = load_f9d_protocol(args.config, require_frozen=True)
    f9c_protocol = f9c_parameters(protocol)
    settings = _ProbeSettings(f9c_protocol.config_path)
    specs = [spec for spec in _load_outlier_scenarios(protocol.config_path) if spec.use_for_final_evaluation]
    if args.scenario_limit is not None:
        if not args.development:
            raise ValueError("--scenario-limit is development-only")
        specs = specs[: args.scenario_limit]
    if not specs:
        raise RuntimeError("frozen F9d config has no final outlier scenarios")

    seeds = (protocol.development_seeds[0],) if args.development else protocol.outlier_final_seeds
    output = _paths(protocol, args.development_output if args.development else None)
    if not args.development:
        _refuse_overwrite(list(output.values()))

    cache: list[RuntimeCacheFrame] = []
    truth: list[TruthFrame] = []
    rows, warnings = collect_probe_rows(
        f9c_protocol,
        settings,
        seeds,
        specs,
        runtime_cache=cache,
        evaluation_truth=truth,
    )
    if not rows or len(cache) != len(rows) or len(truth) != len(rows):
        raise RuntimeError(
            f"incomplete F9d-A evidence: rows={len(rows)}, cache={len(cache)}, truth={len(truth)}"
        )

    cache_hash = write_runtime_cache(output["cache"], cache)
    truth_hash = write_evaluation_truth(output["truth"], truth)
    f9c_eval._write_csv(rows, output["csv"])

    yield_summary = outlier_yield(
        rows, matching_iou_threshold=settings.matching_iou_threshold
    )
    support = _support(protocol, yield_summary)
    outcome = outlier_outcome_metrics(
        rows,
        recovery_error_m=protocol.criteria["outlier_recovery_error_m"],
    )
    ratio = outcome["robust_to_baseline_rmse_ratio"]
    outcome_pass = (
        support["satisfied"]
        and ratio is not None
        and ratio <= protocol.criteria["outlier_rmse_ratio_max"]
    )
    report = {
        "schema_version": 1,
        "gate": "F9d-A",
        "status": "development_smoke" if args.development else "final_once_only_complete",
        "config_sha256": protocol.config_sha256,
        "f9c_config_sha256": protocol.f9c_config_sha256,
        "checkpoint_sha256": protocol.checkpoint_sha256,
        "seeds": list(seeds),
        "scenario_names": [spec.name for spec in specs],
        "row_count": len(rows),
        "support_check_read_first": support,
        "yield": yield_summary,
        "outcome": outcome,
        "criterion": {
            "rmse_ratio_max": protocol.criteria["outlier_rmse_ratio_max"],
            "pass": outcome_pass,
            "interpretation": (
                "descriptive_only_because_support_is_insufficient"
                if not support["satisfied"]
                else "primary_outcome_on_supported_natural_outliers"
            ),
        },
        "runtime_cache": {"path": str(output["cache"]), "sha256": cache_hash},
        "evaluation_truth": {"path": str(output["truth"]), "sha256": truth_hash},
        "episode_warnings": warnings,
        "privileged_truth_use": "post-update evaluation and annotation only",
    }
    output["metrics"].parent.mkdir(parents=True, exist_ok=True)
    output["metrics"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Intentionally print support only. Outcome must be inspected only after
    # the independent support judgement has been read and recorded.
    print(
        json.dumps(
            {
                "metrics": str(output["metrics"]),
                "csv": str(output["csv"]),
                "row_count": len(rows),
                "support_check_read_first": support,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
