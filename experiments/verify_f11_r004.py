#!/usr/bin/env python3
"""Strict read-only verifier for once-only F11 R004."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_explanation_r004_v1.toml",
    )
    args = parser.parse_args()
    verify(args.config.resolve())


def verify(config_path: Path) -> None:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    output = (config_path.parent / str(config["output"]["directory"])).resolve()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    metrics = json.loads(
        (output / str(config["output"]["metrics"])).read_text()
    )
    claim = json.loads(
        (output / str(config["output"]["launch_claim"])).read_text()
    )
    trace_manifest = json.loads(
        (output / str(config["output"]["trace_manifest"])).read_text()
    )
    if manifest["classification"] != "PASS" or metrics["classification"] != "PASS":
        raise RuntimeError("R004 is not PASS")
    if not all(bool(value) for value in metrics["criteria"].values()):
        raise RuntimeError("a frozen R004 criterion failed")
    if not claim["once_only"] or claim["rerun_permitted"]:
        raise RuntimeError("once-only claim is invalid")
    if claim["config_sha256"] != sha256(config_path):
        raise RuntimeError("claim/config hash mismatch")
    if (output / "once_only_failure.json").exists():
        raise RuntimeError("once-only failure marker exists")
    for relative, expected in manifest["files"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"hash mismatch: {relative}")
    for field in (
        "r002b_config", "r002b_metrics", "r002b_manifest", "r002b_protocol",
        "r003_metrics", "r001_result", "r002b_runner",
    ):
        path = (config_path.parent / str(config["frozen"][field])).resolve()
        if sha256(path) != str(config["frozen"][f"{field}_sha256"]):
            raise RuntimeError(f"frozen input drift: {field}")

    trace = dict(np.load(output / str(config["output"]["trace"]), allow_pickle=False))
    references = dict(
        np.load(output / str(config["output"]["references"]), allow_pickle=False)
    )
    draws = dict(
        np.load(output / str(config["output"]["draw_attributions"]), allow_pickle=False)
    )
    final = dict(
        np.load(output / str(config["output"]["final_attributions"]), allow_pickle=False)
    )
    if trace["observation"].shape != (17600, 29):
        raise RuntimeError("locked trace shape mismatch")
    if references["reference_observation"].shape != (6, 4, 4400, 29):
        raise RuntimeError("reference shape mismatch")
    if draws["attribution"].shape != (6, 2, 4400, 29):
        raise RuntimeError("draw attribution shape mismatch")
    if final["attribution"].shape != (2, 4400, 29):
        raise RuntimeError("final attribution shape mismatch")
    if int(final["effective_reference_count"]) != 24:
        raise RuntimeError("final estimator does not record 24 references")
    np.testing.assert_array_equal(
        final["attribution"], np.mean(draws["attribution"], axis=0, dtype=np.float32)
    )
    index = references["reference_index"]
    phases = references["public_phase"]
    seeds = references["seed"]
    reference_seeds = references["reference_seed"]
    if not np.all(phases[index] == phases[None, None, :]):
        raise RuntimeError("reference phase mismatch")
    if not np.all(reference_seeds != seeds[None, None, :]):
        raise RuntimeError("factual seed reused as reference")
    if min(
        len(np.unique(reference_seeds[draw, :, row]))
        for draw in range(6)
        for row in range(len(seeds))
    ) != 4:
        raise RuntimeError("reference seeds are not distinct within draw")
    expected_seeds = set(int(value) for value in config["data"]["locked_evaluation_seeds"])
    if set(int(value) for value in np.unique(trace["seed"])) != expected_seeds:
        raise RuntimeError("locked trace seed set mismatch")
    if set(int(value) for value in np.unique(seeds)) != expected_seeds:
        raise RuntimeError("attribution seed set mismatch")
    forbidden_development = set(
        int(value) for value in config["data"]["forbidden_development_seeds"]
    )
    if set(int(value) for value in np.unique(seeds)) & forbidden_development:
        raise RuntimeError("development seed entered R004")
    if any(not value["sufficient"] or value["seed_count"] != 8 for value in metrics["phase_seed_support"].values()):
        raise RuntimeError("phase seed support is insufficient")
    forbidden = ("privileged", "evaluation_gt", "ground_truth", "world_pose", "bbox", "iou")
    for artifact in (trace, references, draws, final):
        if any(
            any(token == part for part in key.lower().split("_"))
            for key in artifact
            for token in forbidden
        ):
            raise RuntimeError("privileged/evaluation schema found")
        if not all(np.isfinite(value).all() for value in artifact.values() if value.dtype.kind in "fiu"):
            raise RuntimeError("non-finite numeric artifact")
    if trace_manifest["trace_sha256"] != sha256(output / str(config["output"]["trace"])):
        raise RuntimeError("trace manifest hash mismatch")
    if metrics["reference_protocol"]["fallback_used"]:
        raise RuntimeError("an unregistered reference fallback was used")
    if metrics["rerun_permitted"] or metrics["r006_r007_started"]:
        raise RuntimeError("post-R004 boundary violated")
    for later in ("r006", "r007"):
        if (ROOT / "artifacts" / "f11_ppo_explanation_v2" / later).exists():
            raise RuntimeError(f"{later} was started automatically")

    tests = manifest["full_tests"]
    test_log = ROOT / str(tests["log"])
    if sha256(test_log) != str(tests["sha256"]):
        raise RuntimeError("test log hash mismatch")
    text = test_log.read_text(errors="replace")
    match = re.search(r"(\d+) passed, (\d+) warnings", text)
    if not match or int(match.group(1)) != 660:
        raise RuntimeError("active test summary mismatch")
    if re.search(r"\d+ failed|\d+ skipped|ERROR collecting|=+ ERRORS =+", text, re.I):
        raise RuntimeError("active test transcript contains failure/skip")
    print(
        json.dumps(
            {
                "classification": "PASS",
                "r004": metrics["classification"],
                "once_only": True,
                "locked_seeds": sorted(expected_seeds),
                "trace_rows": len(trace["seed"]),
                "attribution_samples": len(seeds),
                "effective_references": int(final["effective_reference_count"]),
                "r006_r007_started": False,
                "full_tests": {"passed": 660, "failed": 0, "skipped": 0},
            },
            indent=2,
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

