"""Read-only integrity checks for the F9c robust-belief calibration/freeze,
modeled on ``experiments/verify_f9_artifacts.py``.

Unlike F9's verifier, this script runs at two different points in F9c's
lifecycle and must behave correctly at both:

1. **After Task 10** (this task) -- only the calibration artifacts and the
   frozen config exist. ``artifacts/f9c_validation.csv``,
   ``artifacts/f9c_belief_metrics.json``, and ``artifacts/f9c_nis_metrics.json``
   are Task 11's output and do not exist yet.
2. **After Task 11** -- the full final-evaluation artifact set exists, and
   this script additionally re-derives every headline metric in
   ``f9c_belief_metrics.json`` directly from ``f9c_validation.csv`` --
   without running any inference (no simulator, no EKF, no detector) -- and
   cross-checks the recomputed numbers against the artifact's own reported
   numbers.

Because Task 11 has not run yet as of Task 10, this script does not import
anything from a not-yet-written ``evaluation.f9c_belief`` module. Instead it
recomputes RMSE/coverage directly from CSV columns using the same column
naming convention already established by ``experiments/validate_f9_yolo_ekf.py``
(``{prefix}_belief_range_m`` / ``{prefix}_belief_range_std_m`` / ``gt_range_m``,
etc.) and by ``robust_updater.RobustStepRecord``
(``reported_range_std_m`` / ``reported_bearing_std_rad`` / ``existence_probability``).
If Task 11 ends up using different column or key names, the affected checks
below report SKIPPED with the reason, rather than raising -- exactly the
degrade-gracefully behavior this script is required to have until Task 11's
artifacts exist. Re-run this script after Task 11 lands; if any check flips
from SKIPPED to a genuine mismatch, that is the bug to fix, not this script.

Exit code is non-zero iff at least one check that actually ran (i.e. not
SKIPPED) reports FAIL.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
from duckie_pomdp.perception.measurement_calibration import wrap_angle

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "f9c_robust_belief_v1.toml"

# Frozen well before any F9c task; F9c may not perturb these regardless of
# which task is running. Identical list to experiments/verify_f9_artifacts.py.
FROZEN_BASELINE_HASHES = {
    "configs/oracle_ekf_v1.toml": (
        "a4815c8d0e17f1868d51619ae51d2183c72832a022edce88aa3c10302594d701"
    ),
    "artifacts/belief_calibration_metrics.json": (
        "1bf8d2bcd328a9703e9cb2797cd7a1cf54f8ddd9d231342cbc9718b58ffea5e5"
    ),
    "artifacts/yolo_measurement_metrics.json": (
        "b916ff2505dd9eb2e0041efda512f3ab27c2b969cc1a9e00c84e44de5bf29c39"
    ),
}

Status = str  # "PASS" | "FAIL" | "SKIP"


class SchemaSkip(Exception):
    """Raised by a check function when it cannot even attempt its
    comparison because the artifact's schema does not match what this
    verifier expects (e.g. a guessed CSV column name that Task 11 ended up
    not using). This is distinct from AssertionError: it means "I could not
    check this", not "I checked this and it is wrong", and must never be
    reported as a FAIL -- that would misattribute a stale guess in this
    verifier to a defect in Task 11's artifact."""


class Results:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def record(self, name: str, status: Status, message: str) -> None:
        self.items.append({"check": name, "status": status, "message": message})
        print(f"[{status}] {name}: {message}")

    def check(self, name: str, fn: Callable[[], str]) -> None:
        """Run ``fn``; it returns a success message, raises AssertionError
        for a genuine mismatch, or raises SchemaSkip if it cannot even be
        attempted against the artifact's actual schema."""
        try:
            message = fn()
        except SchemaSkip as error:
            self.record(name, "SKIP", str(error))
        except AssertionError as error:
            self.record(name, "FAIL", str(error))
        except Exception as error:  # pragma: no cover - defensive
            self.record(name, "FAIL", f"unexpected error: {error!r}")
        else:
            self.record(name, "PASS", message)

    def skip(self, name: str, reason: str) -> None:
        self.record(name, "SKIP", reason)

    @property
    def failed(self) -> bool:
        return any(item["status"] == "FAIL" for item in self.items)


def _find_key_paths(data: Any, substrings: tuple[str, ...]) -> list[tuple[list[str], Any]]:
    """Recursively find every (path, value) whose final key contains any of
    ``substrings`` (case-insensitive) and whose value is a plain number."""

    found: list[tuple[list[str], Any]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if any(sub in lowered for sub in substrings):
                        found.append((path + [key], value))
                walk(value, path + [key])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + [str(index)])

    walk(data, [])
    return found


def _first_present_column(fieldnames: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def main() -> None:
    results = Results()

    # ------------------------------------------------------------------
    # 1. The config loads under require_frozen=True (this IS the freeze
    #    boundary this script exists to police).
    # ------------------------------------------------------------------
    protocol = None

    def _load_frozen_protocol() -> str:
        nonlocal protocol
        protocol = load_f9c_protocol(CONFIG_PATH, require_frozen=True)
        return f"config_sha256={protocol.config_sha256}"

    results.check("frozen_config_loads", _load_frozen_protocol)
    if protocol is None:
        # Nothing else below is meaningful without a loaded protocol.
        print(json.dumps(results.items, indent=2))
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 2. Every hash the frozen-config artifact claims is independently
    #    re-verified against the files it names.
    # ------------------------------------------------------------------
    frozen_config_path = protocol.artifacts["frozen_config_json"]

    def _frozen_artifact_hashes() -> str:
        frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        checks = {
            "config_sha256": (protocol.config_sha256, sha256(protocol.config_path)),
            "checkpoint_sha256": (
                frozen["checkpoint_sha256"],
                sha256(protocol.checkpoint_path),
            ),
            "calibration_artifact_sha256": (
                frozen["calibration_artifact_sha256"],
                sha256(protocol.artifacts["calibration_metrics_json"]),
            ),
            "frozen_f7_config_sha256": (
                frozen["frozen_f7_config_sha256"],
                sha256(protocol.frozen_f7_config_path),
            ),
        }
        mismatches = [
            f"{name}: claimed={claimed} actual={actual}"
            for name, (claimed, actual) in checks.items()
            if claimed != actual
        ]
        assert not mismatches, "; ".join(mismatches)
        assert frozen.get("config_sha256") == protocol.config_sha256
        assert frozen.get("final_evaluation_seeds_not_yet_rendered") is True
        assert list(frozen["calibration_seeds"]) == list(protocol.calibration_seeds)
        assert list(frozen["final_evaluation_seeds"]) == list(protocol.final_evaluation_seeds)
        return f"4 hashes verified against {frozen_config_path.name}"

    results.check("frozen_config_artifact_hashes", _frozen_artifact_hashes)

    # ------------------------------------------------------------------
    # 3. Every fitted parameter written into the config is re-derived
    #    (not just hashed) from the calibration artifact, bit-for-bit.
    # ------------------------------------------------------------------
    calibration_metrics_path = protocol.artifacts["calibration_metrics_json"]

    def _fitted_parameters_match_calibration_artifact() -> str:
        with protocol.config_path.open("rb") as stream:
            cfg = tomllib.load(stream)
        art = json.loads(calibration_metrics_path.read_text(encoding="utf-8"))

        pairs = [
            ("measurement_model.bias_model", cfg["measurement_model"]["bias_model"], art["bias"]["selected_model"]),
            ("measurement_model.range_bias_m", cfg["measurement_model"]["range_bias_m"], art["bias"]["fit"]["range_bias_m"]),
            ("measurement_model.bearing_bias_rad", cfg["measurement_model"]["bearing_bias_rad"], art["bias"]["fit"]["bearing_bias_rad"]),
            ("covariance_calibration.range_scale", cfg["covariance_calibration"]["range_scale"], art["covariance_scales"]["lambda_r"]),
            ("covariance_calibration.bearing_scale", cfg["covariance_calibration"]["bearing_scale"], art["covariance_scales"]["lambda_beta"]),
            ("covariance_calibration.range_posterior_floor_m", cfg["covariance_calibration"]["range_posterior_floor_m"], art["variance_components"]["range"]["sigma_floor_m"]),
            ("covariance_calibration.bearing_posterior_floor_rad", cfg["covariance_calibration"]["bearing_posterior_floor_rad"], art["variance_components"]["bearing"]["sigma_floor_rad"]),
            ("conditional_detection.detection_probability_center", cfg["conditional_detection"]["detection_probability_center"], art["effective_detection"]["detection_probability_center"]),
            ("conditional_detection.detection_probability_mid_fov", cfg["conditional_detection"]["detection_probability_mid_fov"], art["effective_detection"]["detection_probability_mid_fov"]),
            ("conditional_detection.detection_probability_edge_fov", cfg["conditional_detection"]["detection_probability_edge_fov"], art["effective_detection"]["detection_probability_edge_fov"]),
            ("conditional_detection.detection_probability_outside_domain", cfg["conditional_detection"]["detection_probability_outside_domain"], art["effective_detection"]["detection_probability_outside_domain"]),
            (
                "conditional_detection.false_positive_probability",
                cfg["conditional_detection"]["false_positive_probability"],
                art["miss_likelihood_floor"]["adjusted_using_frozen_f9b_false_positive_rate"]["false_positive_probability_used"],
            ),
            (
                "conditional_detection.miss_likelihood_floor",
                cfg["conditional_detection"]["miss_likelihood_floor"],
                art["miss_likelihood_floor"]["adjusted_using_frozen_f9b_false_positive_rate"]["lr_floor"],
            ),
        ]
        mismatches = [
            f"{name}: config={cfg_val!r} artifact={art_val!r}"
            for name, cfg_val, art_val in pairs
            if cfg_val != art_val
        ]
        assert not mismatches, "; ".join(mismatches)
        for section in ("measurement_model", "covariance_calibration", "conditional_detection"):
            assert cfg[section]["parameters_frozen"] is True, f"{section}.parameters_frozen is not true"
        return f"{len(pairs)}/{len(pairs)} fitted values match {calibration_metrics_path.name} exactly"

    results.check(
        "fitted_parameters_match_calibration_artifact",
        _fitted_parameters_match_calibration_artifact,
    )

    # ------------------------------------------------------------------
    # 4. Frozen F7 physics untouched by the freeze.
    # ------------------------------------------------------------------
    def _ekf_and_existence_unperturbed() -> str:
        with protocol.config_path.open("rb") as stream:
            cfg = tomllib.load(stream)
        with protocol.frozen_f7_config_path.open("rb") as stream:
            f7 = tomllib.load(stream)
        assert cfg["ekf"] == f7["ekf"], "[ekf] block diverged from frozen F7"
        for key in ("prior_probability", "survival_probability", "birth_probability"):
            assert cfg["existence"][key] == f7["existence"][key], f"existence.{key} diverged"
        return "[ekf] and 3 frozen [existence] keys byte-identical to oracle_ekf_v1.toml"

    results.check("frozen_f7_physics_unperturbed", _ekf_and_existence_unperturbed)

    # ------------------------------------------------------------------
    # 5. Invariant I7: association gate strictly looser than the
    #    innovation gate.
    # ------------------------------------------------------------------
    def _invariant_i7() -> str:
        with protocol.config_path.open("rb") as stream:
            cfg = tomllib.load(stream)
        assoc = cfg["association"]["chi_square_gate"]
        gate = cfg["innovation_gate"]["chi_square_threshold"]
        assert assoc > gate, f"association gate {assoc} is not looser than innovation gate {gate}"
        return f"association {assoc} > innovation_gate {gate}"

    results.check("invariant_i7_association_looser_than_gate", _invariant_i7)

    # ------------------------------------------------------------------
    # 6. Upstream frozen baselines (F5b/F6/F7) still byte-identical.
    # ------------------------------------------------------------------
    def _baseline_hashes_unchanged() -> str:
        actual = {relative: sha256(ROOT / relative) for relative in FROZEN_BASELINE_HASHES}
        assert actual == FROZEN_BASELINE_HASHES, f"mismatch: {actual}"
        return f"{len(FROZEN_BASELINE_HASHES)} upstream frozen artifacts unchanged"

    results.check("upstream_baseline_hashes_unchanged", _baseline_hashes_unchanged)

    # ------------------------------------------------------------------
    # 7. Task 11 artifacts: validation CSV / belief metrics / NIS metrics.
    #    These do not exist as of Task 10 -- skip cleanly and say so.
    # ------------------------------------------------------------------
    validation_csv = protocol.artifacts["validation_csv"]
    belief_metrics_path = protocol.artifacts["belief_metrics_json"]
    nis_metrics_path = protocol.artifacts["nis_metrics_json"]
    error_case_dir = protocol.artifacts["error_case_dir"]

    missing = [
        path
        for path in (validation_csv, belief_metrics_path, nis_metrics_path)
        if not path.exists()
    ]
    if missing:
        for path in (validation_csv, belief_metrics_path, nis_metrics_path):
            name = f"task11_artifact_present:{path.name}"
            if path.exists():
                results.skip(name, "present, but a sibling Task 11 artifact is missing; see below")
            else:
                results.skip(
                    name,
                    "Task 11 (final evaluation on seeds 7101-7104) has not produced "
                    f"this artifact yet ({path}); re-run this script after Task 11 lands.",
                )
        results.skip(
            "final_evaluation_scenario_frame_matrix",
            "requires validation CSV -- not yet produced by Task 11",
        )
        results.skip(
            "belief_metrics_config_hash",
            "requires belief metrics JSON -- not yet produced by Task 11",
        )
        results.skip(
            "nis_metrics_config_hash",
            "requires NIS metrics JSON -- not yet produced by Task 11",
        )
        results.skip(
            "miss_rows_have_empty_geometry",
            "requires validation CSV -- not yet produced by Task 11",
        )
        results.skip(
            "error_case_images_present",
            "requires validation CSV run to generate error-case images -- not yet produced by Task 11",
        )
        results.skip(
            "rederive_belief_metrics_from_csv",
            "requires both validation CSV and belief metrics JSON -- not yet produced by Task 11",
        )
    else:
        with validation_csv.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        metrics = json.loads(belief_metrics_path.read_text(encoding="utf-8"))
        nis = json.loads(nis_metrics_path.read_text(encoding="utf-8"))

        def _seed_leakage() -> str:
            seeds = {int(row["seed"]) for row in rows}
            assert seeds == set(protocol.final_evaluation_seeds), (
                f"unexpected F9c final seeds: {sorted(seeds)}"
            )
            assert not seeds & set(protocol.calibration_seeds), (
                "F9c final CSV contains calibration seed leakage"
            )
            assert not seeds & set(protocol.forbidden_seeds), (
                "F9c final CSV contains forbidden-seed leakage"
            )
            return f"seeds {sorted(seeds)} == final_evaluation_seeds, disjoint from calibration/forbidden"

        results.check("no_seed_leakage_in_final_csv", _seed_leakage)

        def _scenario_frame_matrix() -> str:
            final_specs = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
            expected = {
                (seed, spec.name): spec.steps + 1
                for seed in protocol.final_evaluation_seeds
                for spec in final_specs
            }
            observed = Counter((int(row["seed"]), row["scenario"]) for row in rows)
            assert observed == expected, "F9c final scenario/frame matrix is incomplete"
            return f"{len(expected)} (seed, scenario) cells complete"

        results.check("final_evaluation_scenario_frame_matrix", _scenario_frame_matrix)

        def _belief_metrics_hash() -> str:
            found = _find_key_paths(metrics, ("config_sha256",))
            candidates = [value for _, value in found] + [
                metrics.get("source", {}).get("config_sha256")
            ]
            candidates = [value for value in candidates if isinstance(value, str)]
            assert candidates, "no config_sha256 field found in belief metrics JSON"
            assert protocol.config_sha256 in candidates, (
                f"belief metrics config_sha256 {candidates} != {protocol.config_sha256}"
            )
            row_count_paths = _find_key_paths(metrics, ("row_count",))
            if row_count_paths:
                for path, value in row_count_paths:
                    assert int(value) == len(rows), (
                        f"row_count at {'.'.join(path)}={value} != len(rows)={len(rows)}"
                    )
            return "config_sha256 matches; row_count fields (if any) match CSV length"

        results.check("belief_metrics_config_hash", _belief_metrics_hash)

        def _nis_metrics_hash() -> str:
            claimed = nis.get("config_sha256")
            assert claimed == protocol.config_sha256, (
                f"NIS metrics config_sha256 {claimed} != {protocol.config_sha256}"
            )
            return "NIS metrics config_sha256 matches"

        results.check("nis_metrics_config_hash", _nis_metrics_hash)

        def _miss_rows_empty_geometry() -> str:
            geometry_columns = [
                name
                for name in rows[0]
                if name in (
                    "raw_measurement_range_m",
                    "raw_measurement_bearing_rad",
                    "corrected_measurement_range_m",
                    "corrected_measurement_bearing_rad",
                    "raw_range_m",
                    "raw_bearing_rad",
                )
            ]
            detected_columns = [
                name for name in ("measurement_detected", "detector_detected") if name in rows[0]
            ]
            assert detected_columns, "no detected/measurement_detected column found"
            detected_column = detected_columns[0]
            violations = [
                row
                for row in rows
                if row[detected_column] in ("False", "false", "0")
                and any(row[column] for column in geometry_columns)
            ]
            assert not violations, f"{len(violations)} miss rows carry non-empty geometry"
            return f"0/{len(rows)} miss rows carry stray geometry ({detected_column})"

        results.check("miss_rows_have_empty_geometry", _miss_rows_empty_geometry)

        def _error_case_images() -> str:
            images = sorted(error_case_dir.glob("*.png")) if error_case_dir.exists() else []
            assert images, f"no error-case images found in {error_case_dir}"
            return f"{len(images)} error-case images present"

        results.check("error_case_images_present", _error_case_images)

        def _rederive_belief_metrics() -> str:
            """Best-effort, schema-tolerant re-derivation of range/bearing
            RMSE and 68%/95% coverage directly from the CSV, cross-checked
            against whatever the belief metrics JSON reports. Column/key
            names are guessed from the established F9/F9c conventions
            (validate_f9_yolo_ekf.py's ``_belief_fields`` and
            ``RobustStepRecord``); if Task 11 used different names, this
            reports which specific piece could not be matched rather than
            silently passing or crashing."""

            if not rows:
                raise AssertionError("validation CSV is empty")
            fieldnames = set(rows[0].keys())

            range_col = _first_present_column(
                fieldnames,
                ("corrected_belief_range_m", "belief_range_m", "reported_range_m"),
            )
            range_std_col = _first_present_column(
                fieldnames,
                (
                    "corrected_belief_range_std_m",
                    "belief_range_std_m",
                    "reported_range_std_m",
                ),
            )
            bearing_col = _first_present_column(
                fieldnames,
                ("corrected_belief_bearing_rad", "belief_bearing_rad", "reported_bearing_rad"),
            )
            bearing_std_col = _first_present_column(
                fieldnames,
                (
                    "corrected_belief_bearing_std_rad",
                    "belief_bearing_std_rad",
                    "reported_bearing_std_rad",
                ),
            )
            gt_range_col = "gt_range_m" if "gt_range_m" in fieldnames else None
            gt_bearing_col = "gt_bearing_rad" if "gt_bearing_rad" in fieldnames else None

            missing_columns = [
                label
                for label, column in (
                    ("belief range mean", range_col),
                    ("belief range std", range_std_col),
                    ("belief bearing mean", bearing_col),
                    ("belief bearing std", bearing_std_col),
                    ("gt range", gt_range_col),
                    ("gt bearing", gt_bearing_col),
                )
                if column is None
            ]
            if missing_columns:
                raise SchemaSkip(
                    "cannot locate CSV columns for: "
                    + ", ".join(missing_columns)
                    + " -- Task 11's exact column names differ from the guessed "
                    "convention; update this verifier's candidate lists to match "
                    "(this is a stale guess in the verifier, not necessarily a "
                    "defect in the artifact)"
                )

            range_errors: list[float] = []
            range_stds: list[float] = []
            bearing_errors: list[float] = []
            bearing_stds: list[float] = []
            for row in rows:
                mean_r = _to_float(row[range_col])
                std_r = _to_float(row[range_std_col])
                mean_b = _to_float(row[bearing_col])
                std_b = _to_float(row[bearing_std_col])
                gt_r = _to_float(row[gt_range_col])
                gt_b = _to_float(row[gt_bearing_col])
                if None in (mean_r, std_r, gt_r):
                    continue
                range_errors.append(mean_r - gt_r)
                range_stds.append(std_r)
                if None not in (mean_b, std_b, gt_b):
                    bearing_errors.append(wrap_angle(mean_b - gt_b))
                    bearing_stds.append(std_b)

            if not range_errors:
                raise AssertionError(
                    "no rows had both a belief range estimate and ground truth "
                    "to compute RMSE from"
                )

            range_errors_arr = np.asarray(range_errors)
            range_stds_arr = np.asarray(range_stds)
            recomputed_range_rmse = float(np.sqrt(np.mean(range_errors_arr**2)))
            recomputed_coverage_68 = float(
                np.mean(np.abs(range_errors_arr) <= range_stds_arr)
            )
            recomputed_coverage_95 = float(
                np.mean(np.abs(range_errors_arr) <= 1.96 * range_stds_arr)
            )

            rmse_hits = _find_key_paths(metrics, ("rmse",))
            coverage_68_hits = _find_key_paths(metrics, ("coverage_68", "coverage68"))
            coverage_95_hits = _find_key_paths(metrics, ("coverage_95", "coverage95"))

            checked: list[str] = [
                f"recomputed range RMSE={recomputed_range_rmse:.6f} m "
                f"over {len(range_errors)} rows"
            ]
            if rmse_hits:
                close = any(
                    abs(float(value) - recomputed_range_rmse) < 1e-6
                    for _, value in rmse_hits
                )
                checked.append(
                    f"{'a' if close else 'NO'} reported *rmse* field matched the "
                    f"recomputed range RMSE within 1e-6 (candidates: {rmse_hits})"
                )
                if not close:
                    raise AssertionError(
                        f"recomputed range RMSE {recomputed_range_rmse!r} matches none "
                        f"of the reported rmse fields {rmse_hits}"
                    )
            else:
                checked.append("no *rmse*-named field found in belief metrics JSON to cross-check")

            if coverage_68_hits or coverage_95_hits:
                checked.append(
                    f"coverage_68 candidates found: {coverage_68_hits}, "
                    f"recomputed={recomputed_coverage_68:.4f}; "
                    f"coverage_95 candidates found: {coverage_95_hits}, "
                    f"recomputed={recomputed_coverage_95:.4f} "
                    "(reported for review; exact threshold/eligibility filtering may "
                    "differ from this best-effort recomputation)"
                )
            else:
                checked.append("no coverage_68/coverage_95-named field found to cross-check")

            return "; ".join(checked)

        results.check("rederive_belief_metrics_from_csv", _rederive_belief_metrics)

    # ------------------------------------------------------------------
    print()
    summary = Counter(item["status"] for item in results.items)
    print(json.dumps({"summary": dict(summary), "checks": results.items}, indent=2))

    if results.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
