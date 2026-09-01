"""Read-only integrity verifier for the F9d evidence-closure gate.

Development mode verifies every artifact that should already exist and
reports future final-run artifacts as ``SKIP``.  ``--final`` is strict: the
Task 6/7 CSV and metrics artifacts must all exist and be bound to the frozen
F9d config hash.  The verifier never renders, runs YOLO, or steps an EKF.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
from duckie_pomdp.evaluation.f9d_protocol import load_f9d_protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f9d_evidence_closure_v1.toml"


@dataclass
class VerificationResults:
    items: list[dict[str, str]] = field(default_factory=list)

    def record(self, name: str, status: str, message: str) -> None:
        self.items.append({"check": name, "status": status, "message": message})
        print(f"[{status}] {name}: {message}")

    def check(self, name: str, function: Callable[[], str]) -> None:
        try:
            message = function()
        except Exception as error:
            self.record(name, "FAIL", str(error))
        else:
            self.record(name, "PASS", message)

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.items)


def verify_required_artifacts(
    results: VerificationResults,
    paths: dict[str, Path],
    *,
    final: bool,
) -> None:
    """Record presence of final-run artifacts under graceful/strict policy."""

    for name, path in paths.items():
        if path.exists():
            results.record(name, "PASS", str(path))
        elif final:
            results.record(name, "FAIL", f"required final artifact missing: {path}")
        else:
            results.record(name, "SKIP", f"not produced yet: {path}")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def verify(config_path: Path = DEFAULT_CONFIG, *, final: bool = False) -> VerificationResults:
    results = VerificationResults()
    protocol = None

    def load_frozen() -> str:
        nonlocal protocol
        protocol = load_f9d_protocol(config_path, require_frozen=True)
        return f"config_sha256={protocol.config_sha256}"

    results.check("frozen_config_loads", load_frozen)
    if protocol is None:
        return results

    frozen_path = protocol.artifacts["frozen_config_json"]

    def frozen_hashes() -> str:
        frozen = _json(frozen_path)
        _assert_equal(frozen["config_sha256"], sha256(protocol.config_path), "F9d config hash")
        _assert_equal(frozen["f9c_config_sha256"], sha256(protocol.f9c_config_path), "F9c config hash")
        f9c = load_f9c_protocol(protocol.f9c_config_path, require_frozen=True)
        _assert_equal(frozen["checkpoint_sha256"], sha256(f9c.checkpoint_path), "checkpoint hash")
        _assert_equal(frozen["seed_groups"]["development"], list(protocol.development_seeds), "development seeds")
        _assert_equal(frozen["seed_groups"]["outlier_final"], list(protocol.outlier_final_seeds), "outlier seeds")
        _assert_equal(frozen["seed_groups"]["absence_final"], list(protocol.absence_final_seeds), "absence seeds")
        _assert_equal(frozen["minima"]["minimum_outlier_frames"], protocol.minimum_outlier_frames, "outlier frame minimum")
        _assert_equal(frozen["minima"]["minimum_absence_runs_40"], protocol.minimum_absence_runs_40, "absence run minimum")
        return "F9d, F9c, checkpoint, seed, and minimum claims verified"

    results.check("frozen_hashes_and_claims", frozen_hashes)

    def outlier_probe() -> str:
        data = _json(protocol.artifacts["yield_probe_json"])
        _assert_equal(data["f9c_config_sha256"], protocol.f9c_config_sha256, "outlier probe F9c hash")
        _assert_equal(data["judgement"]["outlier_support_satisfied"], True, "outlier probe support")
        return f"projected_frames={data['projection_to_final_seeds']['projected_outlier_frames']}"

    results.check("outlier_yield_probe", outlier_probe)

    def absence_probe() -> str:
        data = _json(protocol.artifacts["absence_yield_probe_json"])
        _assert_equal(data["f9c_config_sha256"], protocol.f9c_config_sha256, "absence probe F9c hash")
        _assert_equal(data["per_kind_support_satisfied"], {"B1": True, "B2": True, "B3": True}, "absence support")
        _assert_equal(data["yield"]["b2_frames_with_gt_invisible"], 0, "B2 contamination")
        return "B1/B2/B3 support true; B2 GT-invisible contamination=0"

    results.check("absence_yield_probe", absence_probe)

    def association_diagnostic() -> str:
        data = _json(protocol.artifacts["association_diagnostic_json"])
        _assert_equal(data["provenance"]["frame_count"], 3328, "association cache frame count")
        return "cache-only C1/C2 diagnostic present"

    results.check("association_diagnostic", association_diagnostic)

    final_paths = {
        "outlier_csv_present": protocol.artifacts["outlier_csv"],
        "outlier_metrics_present": protocol.artifacts["outlier_metrics_json"],
        "outlier_runtime_cache_present": protocol.artifacts["outlier_runtime_cache"],
        "outlier_truth_present": protocol.artifacts["outlier_evaluation_truth"],
        "absence_csv_present": protocol.artifacts["absence_csv"],
        "absence_metrics_present": protocol.artifacts["absence_metrics_json"],
    }
    verify_required_artifacts(results, final_paths, final=final)

    for label, path in (
        ("outlier", protocol.artifacts["outlier_metrics_json"]),
        ("absence", protocol.artifacts["absence_metrics_json"]),
    ):
        if not path.exists():
            continue

        def metrics_hash(path: Path = path, label: str = label) -> str:
            data = _json(path)
            _assert_equal(data["config_sha256"], protocol.config_sha256, f"{label} metrics config hash")
            _assert_equal(data["f9c_config_sha256"], protocol.f9c_config_sha256, f"{label} metrics F9c hash")
            return f"{label} metrics bound to frozen F9d/F9c hashes"

        results.check(f"{label}_metrics_hashes", metrics_hash)

    for label, path in (
        ("outlier", protocol.artifacts["outlier_csv"]),
        ("absence", protocol.artifacts["absence_csv"]),
    ):
        if not path.exists():
            continue

        def csv_nonempty(path: Path = path, label: str = label) -> str:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                count = sum(1 for _ in reader)
            if count == 0:
                raise AssertionError(f"{label} CSV has no data rows")
            return f"{count} data rows"

        results.check(f"{label}_csv_nonempty", csv_nonempty)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    results = verify(args.config, final=args.final)
    print(json.dumps({"final_mode": args.final, "results": results.items}, indent=2))
    raise SystemExit(1 if results.failed else 0)


if __name__ == "__main__":
    main()
