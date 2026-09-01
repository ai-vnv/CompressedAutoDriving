#!/usr/bin/env python3
"""Audit F13 model boundaries, QAT provenance, and development replay tolerances."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.explain.compressed_policy_analysis import (
    actor_physical,
    file_sha256,
    require_quantized_linear_graph,
    scalar_metrics,
    verify_hash,
)
from duckie_pomdp.optimization.actor_compression import extract_original_actor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/f13_explain_compressed_v1.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preflight", "audit", "calibrate-replay"), required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.mode == "preflight":
        print(json.dumps(preflight(config, config_path), indent=2))
    elif args.mode == "audit":
        print(json.dumps(audit(config, config_path), indent=2))
    else:
        print(json.dumps(calibrate_replay(config, config_path), indent=2))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        result = tomllib.load(stream)
    result["_sha256"] = file_sha256(path)
    return result


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def artifact_root(config: dict[str, Any], config_path: Path) -> Path:
    return resolve(config_path, str(config["artifacts"]["directory"]))


def frozen_models(config: dict[str, Any], config_path: Path):
    original_path = resolve(config_path, config["frozen"]["original"]["checkpoint"])
    a7_path = resolve(config_path, config["frozen"]["a7"]["checkpoint"])
    verify_hash(original_path, config["frozen"]["original"]["sha256"])
    verify_hash(a7_path, config["frozen"]["a7"]["sha256"])
    original, _, payload = extract_original_actor(
        original_path, expected_sha256=config["frozen"]["original"]["sha256"]
    )
    a7 = torch.jit.load(str(a7_path), map_location="cpu").eval()
    require_quantized_linear_graph(a7)
    probe = np.zeros((2, 29), dtype=np.float32)
    if actor_physical(original, probe).shape != (2, 2) or actor_physical(a7, probe).shape != (2, 2):
        raise RuntimeError("frozen actor interface mismatch")
    return original_path, a7_path, original.eval(), a7, payload


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    original_path, a7_path, _, _, payload = frozen_models(config, config_path)
    policy_path = resolve(config_path, config["frozen"]["contract"]["policy_config"])
    f12_path = resolve(config_path, config["frozen"]["contract"]["f12_config"])
    verify_hash(policy_path, config["frozen"]["contract"]["policy_config_sha256"])
    verify_hash(f12_path, config["frozen"]["contract"]["f12_config_sha256"])
    protocol = load_ppo_curriculum_protocol(policy_path)
    if len(protocol.observation_order) != 29:
        raise RuntimeError("public observation contract is not 29D")
    for key, value in config["frozen"]["f11"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        verify_hash(resolve(config_path, config["frozen"]["f11"][path_key]), value)
    return {
        "classification": "PASS",
        "mode": "preflight",
        "config_sha256": config["_sha256"],
        "original_sha256": file_sha256(original_path),
        "a7_sha256": file_sha256(a7_path),
        "original_global_step": int(payload["global_step"]),
        "observation_dimension": 29,
        "action_dimension": 2,
        "a7_quantized_linear_graph": True,
        "artifact_root_exists": artifact_root(config, config_path).exists(),
    }


def audit(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    preflight_result = preflight(config, config_path)
    root = artifact_root(config, config_path)
    integrity = root / "integrity"
    if integrity.exists():
        raise FileExistsError("F13 integrity audit already exists")
    integrity.mkdir(parents=True)
    original_path, a7_path, _, _, _ = frozen_models(config, config_path)
    search_root = resolve(config_path, config["surrogate"]["search_roots"][0])
    all_model_files = sorted(
        path for suffix in ("*.pt", "*.pth", "*.ckpt") for path in search_root.rglob(suffix)
    )
    plausible = [
        path for path in all_model_files
        if any(token in path.name.lower() for token in ("qat", "fake_quant", "preconvert", "pre_convert"))
        and path != a7_path
    ]
    history = search_root / "prune_distill_quant_distill/qat_distillation_history.json"
    conversion_source = ROOT / "experiments/run_f12_compression.py"
    source_text = conversion_source.read_text(encoding="utf-8")
    direct_convert_without_qat_save = (
        "converted = convert_qat(qat" in source_text
        and "save_quantized_actor(path, converted)" in source_text
        and "qat_distillation_history.json" in source_text
    )
    surrogate_pass = len(plausible) == 1 and not direct_convert_without_qat_save
    result = {
        "schema_version": 1,
        "classification": "PASS" if surrogate_pass else "BLOCKED",
        "gradient_attribution_authorized": surrogate_pass,
        "required_kind": config["surrogate"]["required_kind"],
        "search_root": str(search_root),
        "model_files_examined": len(all_model_files),
        "plausible_exact_qat_state_candidates": [str(path) for path in plausible],
        "qat_history_exists": history.exists(),
        "qat_history_is_model_state": False,
        "f12_conversion_source": str(conversion_source),
        "f12_conversion_source_sha256": file_sha256(conversion_source),
        "conversion_was_direct_without_persisting_qat_state": direct_convert_without_qat_save,
        "reason": (
            "exact pre-conversion A7 fake-quantized QAT state is available"
            if surrogate_pass
            else "F12 persisted only deployable INT8 A7 plus QAT loss history; no exact pre-conversion model state exists"
        ),
        "fallback": "direct deployed-INT8 counterfactual plus paired C4 diagnostics",
        "approximate_surrogate_created": False,
    }
    write_json(integrity / "surrogate_equivalence.json", result)
    hashes = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["_sha256"],
        "original": {"path": str(original_path), "sha256": file_sha256(original_path)},
        "a7": {"path": str(a7_path), "sha256": file_sha256(a7_path)},
        "policy_config_sha256": config["frozen"]["contract"]["policy_config_sha256"],
        "f12_config_sha256": config["frozen"]["contract"]["f12_config_sha256"],
        "models_modified": False,
    }
    write_json(integrity / "hashes.json", hashes)
    return result


def calibrate_replay(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = artifact_root(config, config_path)
    target = root / "integrity/replay_integrity.json"
    if target.exists():
        raise FileExistsError("F13 replay calibration already exists")
    if not (root / "integrity/surrogate_equivalence.json").exists():
        raise RuntimeError("run boundary audit before replay calibration")
    _, _, original, a7, _ = frozen_models(config, config_path)
    trace_path = resolve(config_path, config["frozen"]["f11"]["development_trace"])
    with np.load(trace_path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observation"], dtype=np.float32)
        stored_physical = np.asarray(archive["physical_action"], dtype=np.float32)
        phases = np.asarray(archive["public_phase"], dtype="U40")
    original_action = actor_physical(original, observations)
    a7_first = actor_physical(a7, observations)
    a7_second = actor_physical(a7, observations)
    original_max = float(np.max(np.abs(original_action - stored_physical)))
    a7_repeat_max = float(np.max(np.abs(a7_first - a7_second)))
    original_tolerance = max(float(config["replay"]["original_minimum_tolerance"]), 2.0 * original_max)
    a7_tolerance = max(float(config["replay"]["a7_minimum_tolerance"]), 2.0 * a7_repeat_max)
    phase_counts = {str(phase): int(np.sum(phases == phase)) for phase in np.unique(phases)}
    result = {
        "schema_version": 1,
        "classification": "PASS",
        "calibration_source": config["replay"]["calibration_source"],
        "source_sha256": file_sha256(trace_path),
        "rows": int(len(observations)),
        "phase_counts": phase_counts,
        "original_vs_stored": {
            "v_cmd_mps": scalar_metrics(stored_physical[:, 0], original_action[:, 0]),
            "omega_cmd_rad_s": scalar_metrics(stored_physical[:, 1], original_action[:, 1]),
            "maximum_absolute_error": original_max,
        },
        "a7_repeat_maximum_absolute_error": a7_repeat_max,
        "frozen_original_replay_tolerance": original_tolerance,
        "frozen_a7_repeat_tolerance": a7_tolerance,
        "r006_modified_or_recovered": False,
        "models_modified": False,
    }
    write_json(target, result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
