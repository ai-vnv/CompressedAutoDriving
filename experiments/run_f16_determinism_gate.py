#!/usr/bin/env python3
"""F16 determinism gate.

F15 established that repeated closed-loop runs of the same (model, curriculum, seed)
could differ, because the frozen perception front-end ran nondeterministic CUDA kernels.
F16 may not begin its sequence comparison until a reproducible evaluation path is frozen.

This script measures reproducibility for one candidate backend by repeating the same
preflight cells and comparing every repeat against the first. The backend is selected
ONLY on reproducibility; model performance plays no part in the choice.

Backends are selected by environment before torch initialises, so this script is invoked
once per backend by artifacts/f16_determinism.sh:

  cuda_strict_deterministic : CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms,
                              cudnn.deterministic, cudnn.benchmark off
  cpu_deterministic         : CUDA_VISIBLE_DEVICES="" so every "auto" device resolves to
                              CPU, plus deterministic algorithms

Usage:
  run_f16_determinism_gate.py measure --backend <name>
  run_f16_determinism_gate.py decide
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment  # noqa: E402
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256,
    verify_registry,
)
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    ActorPolicy,
    artifact_root,
    frozen_paths,
    load_actor,
    load_config,
    provenance,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"


def apply_determinism_flags(config) -> dict:
    """Apply in-process determinism settings and report what actually took effect."""
    report = {
        "requested_use_deterministic_algorithms": bool(config["determinism"]["torch_use_deterministic_algorithms"]),
        "cublas_workspace_config_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        torch.use_deterministic_algorithms(True)
        report["use_deterministic_algorithms"] = "enabled_strict"
    except Exception as error:  # pragma: no cover - environment dependent
        torch.use_deterministic_algorithms(True, warn_only=True)
        report["use_deterministic_algorithms"] = f"enabled_warn_only ({type(error).__name__})"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = bool(config["determinism"]["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = bool(config["determinism"]["cudnn_benchmark"])
        report["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        report["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    torch.manual_seed(0)
    np.random.seed(0)
    return report


def run_cell(config, config_path, registry_entry, curriculum: str, seed: int) -> dict:
    """One episode. Returns the per-step signals the gate compares."""
    paths = frozen_paths(config, config_path)
    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    actor = load_actor(registry_entry)
    policy = ActorPolicy("Original Policy", actor)
    environment = PPOCurriculumEnvironment(
        paths["policy_config"], stage=curriculum,
        split=f"f16_determinism_{curriculum}", seeds=(seed,),
    )
    actions: list[np.ndarray] = []
    progress: list[float] = []
    started = time.time()
    try:
        observation, _ = environment.reset(seed=seed)
        policy.reset(seed)
        terminated = truncated = False
        info: dict = {}
        for _ in range(protocol.stage(curriculum).episode_horizon_steps):
            action = policy.act(observation)
            actions.append(np.asarray(action, dtype=np.float32).copy())
            observation, _, terminated, truncated, info = environment.step(action)
            progress.append(float(info["progress_m"]))
            if terminated or truncated:
                break
    finally:
        environment.close()
    elapsed = time.time() - started
    return {
        "steps": len(actions),
        "actions": np.asarray(actions, dtype=np.float32),
        "progress": np.asarray(progress, dtype=np.float32),
        "completed": bool(info.get("completed", False)),
        "collision": bool(info.get("collision", False)),
        "lane_failure": bool(info.get("lane_failure", False)),
        "invalid_pose": bool(info.get("invalid_pose", False)),
        "stop_violation": bool(info.get("stop_violation", False)),
        "termination_reason": str(info.get("termination_reason") or ""),
        "truncation_reason": str(info.get("truncation_reason") or ""),
        "wall_seconds": elapsed,
    }


def measure(backend: str) -> dict:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    flags = apply_determinism_flags(config)
    det = config["determinism"]
    paths = frozen_paths(config, CONFIG)
    matrix = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    entry = matrix[str(det["preflight_model"])]

    repeats = int(det["repeats_per_cell"])
    seeds = [int(s) for s in config["seeds"]["determinism_preflight"][:2]]
    curricula = [str(c) for c in det["preflight_curricula"]]

    csv_path = root / "integrity/determinism_test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))

    for curriculum in curricula:
        for seed in seeds:
            baseline = None
            for repeat in range(repeats):
                result = run_cell(config, CONFIG, entry, curriculum, seed)
                if baseline is None:
                    baseline = result
                    action_delta = 0.0
                    progress_delta = 0.0
                else:
                    common = min(len(baseline["actions"]), len(result["actions"]))
                    action_delta = (
                        float(np.max(np.abs(baseline["actions"][:common] - result["actions"][:common])))
                        if common else float("inf")
                    )
                    common_p = min(len(baseline["progress"]), len(result["progress"]))
                    progress_delta = (
                        float(np.max(np.abs(baseline["progress"][:common_p] - result["progress"][:common_p])))
                        if common_p else float("inf")
                    )
                rows.append({
                    "backend": backend,
                    "model_id": str(det["preflight_model"]),
                    "curriculum": curriculum,
                    "seed": seed,
                    "repeat": repeat,
                    "steps": result["steps"],
                    "steps_match_first": result["steps"] == baseline["steps"],
                    "max_abs_action_delta": action_delta,
                    "max_abs_progress_delta_m": progress_delta,
                    "completed": result["completed"],
                    "completed_match_first": result["completed"] == baseline["completed"],
                    "failure_label": "|".join(
                        name for name in ("collision", "lane_failure", "invalid_pose", "stop_violation")
                        if result[name]
                    ),
                    "failure_label_match_first": (
                        "|".join(n for n in ("collision", "lane_failure", "invalid_pose", "stop_violation") if result[n])
                        == "|".join(n for n in ("collision", "lane_failure", "invalid_pose", "stop_violation") if baseline[n])
                    ),
                    "termination_reason": result["termination_reason"],
                    "termination_reason_match_first": result["termination_reason"] == baseline["termination_reason"],
                    "wall_seconds": round(result["wall_seconds"], 3),
                })
                print(
                    f"  {backend:<28} {curriculum} seed={seed} repeat={repeat} "
                    f"steps={result['steps']:<5} d_action={action_delta:.3e} "
                    f"d_progress={progress_delta:.3e} {result['wall_seconds']:.1f}s",
                    flush=True,
                )

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    backend_rows = [r for r in rows if r["backend"] == backend and int(r["repeat"]) > 0]
    summary = summarize(backend, backend_rows, config)
    summary["determinism_flags"] = flags
    print(json.dumps(summary, indent=2, default=str))
    return summary


def _as_float(value) -> float:
    return float(value)


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def summarize(backend: str, rows: list[dict], config) -> dict:
    det = config["determinism"]
    if not rows:
        return {"backend": backend, "reproducible": False, "reason": "no repeat rows"}
    action_tol = float(det["maximum_absolute_normalized_action_difference"])
    progress_tol = float(det["maximum_absolute_progress_difference_m"])
    max_action = max(_as_float(r["max_abs_action_delta"]) for r in rows)
    max_progress = max(_as_float(r["max_abs_progress_delta_m"]) for r in rows)
    checks = {
        "action_within_tolerance": max_action <= action_tol,
        "progress_within_tolerance": max_progress <= progress_tol,
        "steps_identical": all(_as_bool(r["steps_match_first"]) for r in rows),
        "completion_identical": all(_as_bool(r["completed_match_first"]) for r in rows),
        "failure_label_identical": all(_as_bool(r["failure_label_match_first"]) for r in rows),
        "termination_reason_identical": all(_as_bool(r["termination_reason_match_first"]) for r in rows),
    }
    wall = [_as_float(r["wall_seconds"]) for r in rows]
    return {
        "backend": backend,
        "repeat_comparisons": len(rows),
        "max_abs_action_delta": max_action,
        "max_abs_progress_delta_m": max_progress,
        "action_tolerance": action_tol,
        "progress_tolerance_m": progress_tol,
        "checks": checks,
        "reproducible": all(checks.values()),
        "median_episode_wall_seconds": float(np.median(wall)),
        "mean_episode_wall_seconds": float(np.mean(wall)),
    }


def decide() -> dict:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    csv_path = root / "integrity/determinism_test.csv"
    target = root / "integrity/determinism_gate.json"
    if target.exists():
        raise RuntimeError("refusing to overwrite the frozen F16 determinism gate")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    backends = sorted({r["backend"] for r in rows})
    summaries = {
        backend: summarize(backend, [r for r in rows if r["backend"] == backend and int(r["repeat"]) > 0], config)
        for backend in backends
    }
    # Selection is on reproducibility only. Wall time breaks ties between backends
    # that are equally reproducible; it never promotes a non-reproducible backend.
    reproducible = [b for b, s in summaries.items() if s["reproducible"]]
    selected = None
    if reproducible:
        selected = min(reproducible, key=lambda b: summaries[b]["median_episode_wall_seconds"])
    output = {
        **provenance(config, CONFIG),
        "classification": "PASS" if selected else "FAIL",
        "selected_backend": selected,
        "selection_rule": (
            "reproducibility first; wall time only breaks ties among backends that are "
            "already fully reproducible; model performance is never an input"
        ),
        "backends_tested": backends,
        "summaries": summaries,
        "determinism_test_csv": str(csv_path),
        "determinism_test_csv_sha256": file_sha256(csv_path),
        "consequence_if_fail": (
            "F16 closed-loop scientific evaluation must not proceed; matched-sequence claims "
            "would be indistinguishable from run-to-run noise"
        ),
    }
    write_json(target, output)
    print(json.dumps({k: output[k] for k in ("classification", "selected_backend", "backends_tested")}, indent=2))
    for backend, summary in summaries.items():
        print(f"  {backend}: reproducible={summary['reproducible']} checks={summary['checks']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("measure", "decide"))
    parser.add_argument("--backend")
    args = parser.parse_args()
    if args.command == "measure":
        if not args.backend:
            parser.error("measure requires --backend")
        measure(args.backend)
    else:
        decide()


if __name__ == "__main__":
    main()
