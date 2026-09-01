"""F9d-B once-only long-absence stress evaluation on seeds 8301-8304."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import evaluate_f9c_robust_belief as f9c_eval  # noqa: E402
from probe_f9d_absence_yield import (  # noqa: E402
    _ProbeSettings,
    _load_absence_scenarios,
    collect_absence_rows,
)

from duckie_pomdp.evaluation.f9d_absence_outcome import absence_outcome_metrics  # noqa: E402
from duckie_pomdp.evaluation.f9d_absence_stress import absence_yield  # noqa: E402
from duckie_pomdp.evaluation.f9d_protocol import (  # noqa: E402
    absence_support_satisfied,
    f9c_parameters,
    load_f9d_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f9d_evidence_closure_v1.toml"


def _refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            "F9d-B final artifacts already exist; refusing to re-render final "
            "seeds or overwrite evidence: " + ", ".join(str(path) for path in existing)
        )


def _paths(protocol, development_dir: Path | None) -> dict[str, Path]:
    if development_dir is None:
        return {
            "csv": protocol.artifacts["absence_csv"],
            "metrics": protocol.artifacts["absence_metrics_json"],
        }
    development_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": development_dir / "f9d_absence_development.csv",
        "metrics": development_dir / "f9d_absence_development_metrics.json",
    }


def _support(protocol, summary: dict) -> dict:
    result = {}
    for kind in ("B1", "B2", "B3"):
        values = summary["per_kind"][kind]
        satisfied = absence_support_satisfied(
            protocol,
            runs_ge_20=values["runs_ge_20"],
            runs_ge_40=values["runs_ge_40"],
        )
        result[kind] = {
            "runs_ge_20": values["runs_ge_20"],
            "minimum_runs_ge_20": protocol.minimum_absence_runs_20,
            "runs_ge_40": values["runs_ge_40"],
            "minimum_runs_ge_40": protocol.minimum_absence_runs_40,
            "satisfied": satisfied,
        }
    return result


def _existence_physics(f9c_config_path: Path) -> tuple[float, float]:
    with f9c_config_path.open("rb") as stream:
        data = tomllib.load(stream)
    existence = data["existence"]
    return float(existence["survival_probability"]), float(existence["birth_probability"])


def _read_absence_csv(path: Path) -> list[dict]:
    """Load only the typed fields the pure absence analysis consumes."""

    boolean_fields = {
        "eligible_visible",
        "detector_detected",
        "robust_b_track_active",
        "robust_b_track_deleted",
        "gt_exists",
        "dropout_frame",
    }
    integer_fields = {"seed", "frame"}
    float_fields = {"robust_b_existence_probability"}
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for field in boolean_fields:
                row[field] = row[field].strip().lower() == "true"
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--development", action="store_true")
    parser.add_argument(
        "--recompute-from-csv",
        action="store_true",
        help="Rebuild metrics after a pure-analysis fix; performs no render or inference.",
    )
    parser.add_argument("--kind", choices=("B1", "B2", "B3"), default=None)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--development-output", type=Path, default=Path("/tmp/f9d-task7-smoke"))
    args = parser.parse_args()
    if args.recompute_from_csv and (
        args.development or args.kind is not None or args.scenario_limit is not None
    ):
        raise ValueError("--recompute-from-csv cannot be combined with development selectors")

    protocol = load_f9d_protocol(args.config, require_frozen=True)
    f9c_protocol = f9c_parameters(protocol)
    settings = _ProbeSettings(f9c_protocol.config_path)
    specs = [
        spec
        for spec in _load_absence_scenarios(protocol.config_path)
        if spec.base.use_for_final_evaluation
    ]
    if args.kind is not None:
        if not args.development:
            raise ValueError("--kind is development-only")
        specs = [spec for spec in specs if spec.kind == args.kind]
    if args.scenario_limit is not None:
        if not args.development:
            raise ValueError("--scenario-limit is development-only")
        specs = specs[: args.scenario_limit]
    if not specs:
        raise RuntimeError("frozen F9d config has no selected absence scenarios")

    seeds = (protocol.development_seeds[0],) if args.development else protocol.absence_final_seeds
    output = _paths(protocol, args.development_output if args.development else None)
    if not args.development and not args.recompute_from_csv:
        _refuse_overwrite(list(output.values()))

    if args.recompute_from_csv:
        if not output["csv"].exists() or not output["metrics"].exists():
            raise RuntimeError("cannot recompute F9d-B metrics without the once-only CSV and prior metrics")
        previous_report = json.loads(output["metrics"].read_text(encoding="utf-8"))
        rows = _read_absence_csv(output["csv"])
        warnings = previous_report.get("episode_warnings", [])
    else:
        rows, warnings = collect_absence_rows(f9c_protocol, settings, seeds, specs)
    if not rows:
        raise RuntimeError("F9d-B produced no evidence rows")
    if not args.recompute_from_csv:
        f9c_eval._write_csv(rows, output["csv"])

    summary = absence_yield(rows)
    support = _support(protocol, summary)
    survival, birth = _existence_physics(f9c_protocol.config_path)
    outcome = absence_outcome_metrics(
        rows,
        survival_probability=survival,
        birth_probability=birth,
        out_of_domain_floor=protocol.criteria["absence_out_of_domain_floor"],
        in_domain_ceiling=protocol.criteria["absence_in_domain_ceiling"],
        recurrence_tolerance=protocol.criteria["absence_recurrence_abs_tolerance"],
        recovery_frames_max=int(protocol.criteria["recovery_frames_max"]),
    )
    b1_pass = (
        support["B1"]["satisfied"]
        and outcome["B1"]["criterion"]["all_checkpoint_40_above_floor"]
        and outcome["B1"]["criterion"]["recurrence_matches"]
    )
    b2_pass = (
        support["B2"]["satisfied"]
        and outcome["B2"]["criterion"]["all_below_ceiling"]
        and outcome["B2"]["criterion"]["all_observed_recoveries_within_limit"]
    )
    report = {
        "schema_version": 1,
        "gate": "F9d-B",
        "status": (
            "development_smoke"
            if args.development
            else (
                "final_once_only_complete_postprocess_recomputed_from_csv"
                if args.recompute_from_csv
                else "final_once_only_complete"
            )
        ),
        "config_sha256": protocol.config_sha256,
        "f9c_config_sha256": protocol.f9c_config_sha256,
        "checkpoint_sha256": protocol.checkpoint_sha256,
        "seeds": list(seeds),
        "scenario_names": [spec.base.name for spec in specs],
        "row_count": len(rows),
        "support": support,
        "yield": summary,
        "outcome": outcome,
        "criteria": {
            "B1_pass": b1_pass,
            "B2_pass": b2_pass,
            "B3_support": support["B3"]["satisfied"],
            "long_absence_evidence_pass": b1_pass and b2_pass,
        },
        "existence_prediction_physics": {
            "survival_probability": survival,
            "birth_probability": birth,
            "factor": survival - birth,
            "fixed_point": birth / (1.0 - survival + birth),
        },
        "episode_warnings": warnings,
        "privileged_truth_use": "post-update evaluation and B3 annotation only",
        "no_rerender_statement": (
            "Metrics recomputed from the existing once-only CSV; no simulator, detector, "
            "or estimator was invoked."
            if args.recompute_from_csv
            else "Rows came from the live once-only render."
        ),
    }
    output["metrics"].parent.mkdir(parents=True, exist_ok=True)
    output["metrics"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": str(output["metrics"]),
                "csv": str(output["csv"]),
                "row_count": len(rows),
                "support": support,
                "criteria": report["criteria"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
