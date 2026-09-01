#!/usr/bin/env python3
"""Read-only verifier for the completed F11 R002b artifacts."""

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
        default=ROOT / "configs" / "f11_ppo_explanation_r002b_v1.toml",
    )
    args = parser.parse_args()
    verify(args.config.resolve())


def verify(config_path: Path) -> None:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    output = (config_path.parent / str(config["output"]["directory"])).resolve()
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    metrics = json.loads((output / str(config["output"]["metrics"])).read_text())
    if manifest["classification"] != "PASS" or metrics["classification"] != "PASS":
        raise RuntimeError("R002b is not PASS")
    if not all(bool(value) for value in metrics["criteria"].values()):
        raise RuntimeError("one or more preregistered R002b criteria failed")
    if metrics["r002c_permitted"] or not metrics["r004_unlocked"]:
        raise RuntimeError("R002b decision rule was not applied")

    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"artifact/source hash mismatch: {relative}")
    if sha256(config_path) != metrics["config_sha256"]:
        raise RuntimeError("R002b config hash mismatch")
    for field in (
        "r002_r003_config",
        "r002_metrics",
        "r003_metrics",
        "development_trace",
        "r001_result",
    ):
        path = (config_path.parent / str(config["frozen"][field])).resolve()
        if sha256(path) != str(config["frozen"][f"{field}_sha256"]):
            raise RuntimeError(f"frozen input drift: {field}")

    references = dict(
        np.load(output / str(config["output"]["references"]), allow_pickle=False)
    )
    attributions = dict(
        np.load(output / str(config["output"]["attributions"]), allow_pickle=False)
    )
    if references["reference_observation"].shape != (6, 4, 2200, 29):
        raise RuntimeError("reference artifact shape mismatch")
    if attributions["attribution"].shape != (6, 2, 2200, 29):
        raise RuntimeError("attribution artifact shape mismatch")
    if attributions["completeness_delta"].shape != (6, 2, 2200):
        raise RuntimeError("completeness artifact shape mismatch")
    if not np.isfinite(references["reference_observation"]).all():
        raise RuntimeError("non-finite reference")
    if not np.isfinite(attributions["attribution"]).all():
        raise RuntimeError("non-finite attribution")
    index = references["reference_index"]
    phase = references["public_phase"]
    seed = references["seed"]
    if not np.all(phase[index] == phase[None, None, :]):
        raise RuntimeError("reference phase mismatch")
    if not np.all(references["reference_seed"] != seed[None, None, :]):
        raise RuntimeError("same-seed reference found")
    if np.any(index == np.arange(len(seed))[None, None, :]):
        raise RuntimeError("self reference found")
    development = set(int(value) for value in config["data"]["development_seeds"])
    locked = set(int(value) for value in config["data"]["locked_evaluation_seeds"])
    if set(int(value) for value in np.unique(seed)) != development:
        raise RuntimeError("unexpected development seed set")
    if set(int(value) for value in np.unique(seed)) & locked:
        raise RuntimeError("locked seed leakage")
    forbidden = ("privileged", "evaluation_gt", "ground_truth", "world_pose", "bbox", "iou")
    for artifact in (references, attributions):
        if any(
            any(token == part for part in key.lower().split("_"))
            for key in artifact
            for token in forbidden
        ):
            raise RuntimeError("privileged/evaluation schema found")
    if (ROOT / "artifacts" / "f11_ppo_explanation_v2" / "r004").exists():
        raise RuntimeError("R004 was run automatically")

    tests = manifest["full_tests"]
    test_log = ROOT / str(tests["log"])
    if sha256(test_log) != str(tests["sha256"]):
        raise RuntimeError("full-test log hash mismatch")
    text = test_log.read_text(errors="replace")
    match = re.search(r"(\d+) passed, (\d+) warnings", text)
    if not match or int(match.group(1)) != 655:
        raise RuntimeError("full-test summary mismatch")
    if re.search(
        r"\d+ failed|\d+ skipped|ERROR collecting|=+ ERRORS =+",
        text,
        re.IGNORECASE,
    ):
        raise RuntimeError("full active-suite transcript reports failure/skip")

    print(
        json.dumps(
            {
                "classification": "PASS",
                "r002b": metrics["classification"],
                "samples": metrics["sample_count"],
                "draws": metrics["reference_protocol"]["draw_count"],
                "references_per_input": metrics["reference_protocol"]["references_per_input"],
                "locked_evaluation_seeds_opened": False,
                "r004_exists": False,
                "full_tests": {"passed": 655, "failed": 0, "skipped": 0},
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
