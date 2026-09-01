#!/usr/bin/env python3
"""F18 FP16 control runner.

F18 deliberately adds no evaluation logic of its own. It points the frozen F17 evaluation
code at the F18 config and extends only the checkpoint loader, so closed-loop gating,
fidelity computation, phenotype assignment and media capture run through exactly the same
code that produced the F17 verdicts. The loader detects half-precision checkpoints from
their stored tensor dtype, so no F17 source is modified and no F17 artifact is touched
(F18 writes into its own artifact namespace).

Commands:
  fp16-determinism-check   preregistered bit-exactness addendum (must PASS before evaluate)
  evaluate                 the single new candidate, resumable per (curriculum, seed)
  benchmark                serialized size, logical parameter memory, actor-only CPU latency
  results                  comparison table + fidelity + eligibility outcome
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment  # noqa: E402
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402
import run_f15_cross_curriculum_recovery as f15  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    ActorPolicy, load_config, provenance, read_json, write_json,
)
import run_f17_optimization_method_order as f17  # noqa: E402
from freeze_f18_protocol import load_fp16_actor  # noqa: E402

CONFIG = ROOT / "configs/f18_fp16_control_v1.toml"
CUR = ["c0", "c1", "c2", "c3", "c4"]


class Fp16Actor(torch.nn.Module):
    """Half-precision core behind an FP32 I/O boundary.

    Weights and activations are float16 (verified by the frozen validity gate); inputs are
    cast down and outputs cast back up so the surrounding evaluation code is unchanged.
    """

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x.half()).float()


def is_fp16_checkpoint(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        return False
    return any(t.dtype == torch.float16 for t in payload["state_dict"].values()
               if torch.is_tensor(t))


def load_actor_f18(entry: Any) -> torch.nn.Module:
    """Loader used in place of f15.load_actor: dispatches half-precision checkpoints."""
    if bool(entry.get("int8")):
        return f15.load_actor(entry)
    path = Path(entry["model_path"])
    if is_fp16_checkpoint(path):
        return Fp16Actor(load_fp16_actor(path))
    return f15.load_actor(entry)


def install() -> None:
    """Repoint the frozen F17 evaluation code at F18's config and loader."""
    f17.CONFIG = CONFIG
    f17.load_actor = load_actor_f18


def require_addendum(root: Path) -> None:
    target = root / "integrity/fp16_determinism_addendum.json"
    if not target.exists() or read_json(target)["classification"] != "PASS":
        raise RuntimeError("run `fp16-determinism-check` first; it must PASS before evaluation")


def fp16_determinism_check() -> dict[str, Any]:
    """Same criteria as the frozen F16 determinism gate and the F17 INT8 addendum."""
    install()
    config = load_config(CONFIG)
    determinism = f17.apply_determinism(config)
    root = f17.artifact_root(config)
    target = root / "integrity/fp16_determinism_addendum.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    record = read_json(root / "pathway_registry.json")["pathways"]["F16H"]
    if file_sha256(record["checkpoint"]) != record["sha256"]:
        raise RuntimeError("F16H checkpoint changed since freeze")
    policy = ActorPolicy(record["label"], load_actor_f18(
        {"model_path": record["checkpoint"], "int8": False,
         "hidden_sizes": [record["width"]] * 2}))

    policy_config = f17.resolve(config, "f10_ppo_visual_objects_v30.toml")
    protocol = load_ppo_curriculum_protocol(policy_config)
    seed = int(config["seeds"]["primary_evaluation"][0])

    cells = {}
    for curriculum in ("c0", "c3"):
        repeats = []
        for repeat in range(3):
            environment = PPOCurriculumEnvironment(
                policy_config, stage=curriculum,
                split=f"f18_fp16_determinism_{curriculum}", seeds=(seed,))
            actions, progress = [], []
            info: dict[str, Any] = {}
            try:
                observation, _ = environment.reset(seed=seed)
                policy.reset(seed)
                for _ in range(protocol.stage(curriculum).episode_horizon_steps):
                    action = policy.act(observation)
                    actions.append(np.asarray(action, dtype=np.float32).copy())
                    observation, _, terminated, truncated, info = environment.step(action)
                    progress.append(float(info["progress_m"]))
                    if terminated or truncated:
                        break
            finally:
                environment.close()
            repeats.append({
                "steps": len(actions),
                "actions": np.asarray(actions, dtype=np.float32),
                "progress": np.asarray(progress, dtype=np.float32),
                "completed": bool(info.get("completed", False)),
                "failure_label": "|".join(n for n in ("collision", "lane_failure",
                                                      "invalid_pose", "stop_violation")
                                          if info.get(n)),
                "termination_reason": str(info.get("termination_reason") or ""),
            })
            print(f"  F16H {curriculum} seed={seed} repeat={repeat} steps={len(actions)}",
                  flush=True)
        first = repeats[0]
        checks = {
            "steps_identical": all(r["steps"] == first["steps"] for r in repeats),
            "actions_bit_identical": all(np.array_equal(r["actions"], first["actions"])
                                         for r in repeats),
            "progress_bit_identical": all(np.array_equal(r["progress"], first["progress"])
                                          for r in repeats),
            "completion_identical": all(r["completed"] == first["completed"] for r in repeats),
            "failure_label_identical": all(r["failure_label"] == first["failure_label"]
                                           for r in repeats),
            "termination_reason_identical": all(
                r["termination_reason"] == first["termination_reason"] for r in repeats),
        }
        cells[curriculum] = {"seed": seed, "repeats": 3, "steps": first["steps"],
                             "checks": checks, "reproducible": all(checks.values())}

    passed = all(c["reproducible"] for c in cells.values())
    payload = {
        **provenance(config, CONFIG),
        "classification": "PASS" if passed else "FAIL",
        "kind": "fp16_determinism_addendum",
        "criteria_source": "identical to the frozen F16 determinism gate and F17 INT8 addendum",
        "determinism": determinism,
        "representative_model": {"pathway": "F16H", "sha256": record["sha256"]},
        "cells": cells,
        "consequence_if_fail": "the FP16 evaluation block is barred",
    }
    write_json(target, payload)
    print(json.dumps({"classification": payload["classification"],
                      "cells": {k: v["reproducible"] for k, v in cells.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)
    return payload


def benchmark() -> dict[str, Any]:
    """Serialized size, logical parameter memory and actor-only CPU latency."""
    import time

    install()
    config = load_config(CONFIG)
    root = f17.artifact_root(config)
    settings = config["benchmark"]
    torch.set_num_threads(int(settings["threads"]))
    registry = read_json(root / "pathway_registry.json")["pathways"]

    rows = {}
    for pid in ("A3", "F16H", "A6"):
        record = registry[pid]
        path = Path(record["checkpoint"])
        actor = load_actor_f18({"model_path": str(path), "int8": record["int8"],
                                "hidden_sizes": [record["width"]] * 2})
        actor = actor.cpu().eval()
        if pid == "F16H":
            parameters = sum(p.numel() for p in actor.parameters())
            bytes_per = 2
        elif record["int8"]:
            parameters = None
            bytes_per = 1
        else:
            parameters = sum(p.numel() for p in actor.parameters())
            bytes_per = 4
        probe = torch.zeros((int(settings["batch_size"]), 29), dtype=torch.float32)

        medians, p95s, p99s = [], [], []
        with torch.inference_mode():
            for _ in range(int(settings["warmup_iterations"])):
                actor(probe)
            for _ in range(int(settings["repeats"])):
                samples = np.empty(int(settings["iterations"]), dtype=np.float64)
                for i in range(int(settings["iterations"])):
                    start = time.perf_counter()
                    actor(probe)
                    samples[i] = time.perf_counter() - start
                micro = samples * 1e6
                medians.append(float(np.median(micro)))
                p95s.append(float(np.percentile(micro, 95)))
                p99s.append(float(np.percentile(micro, 99)))
        rows[pid] = {
            "label": record["label"], "precision": record["precision"],
            "width": record["width"], "serialized_bytes": path.stat().st_size,
            "parameters": parameters,
            "logical_parameter_memory_bytes": None if parameters is None else parameters * bytes_per,
            "latency_median_us": float(np.median(medians)),
            "latency_p95_us": float(np.median(p95s)),
            "latency_p99_us": float(np.median(p99s)),
            "latency_median_us_per_repeat": medians,
            "throughput_inferences_per_s": 1e6 / float(np.median(medians)),
        }
        print(f"  {pid:5s} {record['precision']:5s} bytes={path.stat().st_size:6d} "
              f"median={rows[pid]['latency_median_us']:.2f}us "
              f"P95={rows[pid]['latency_p95_us']:.2f} P99={rows[pid]['latency_p99_us']:.2f}",
              flush=True)

    reference = rows["A3"]
    for pid, row in rows.items():
        row["size_ratio_vs_fp32_anchor"] = reference["serialized_bytes"] / row["serialized_bytes"]
        row["speedup_vs_fp32_anchor"] = reference["latency_median_us"] / row["latency_median_us"]
    payload = {
        **provenance(config, CONFIG),
        "settings": {k: settings[k] for k in settings},
        "rows": rows,
        "note": ("actor-only CPU inference, single thread, batch 1; the INT8 row is a traced "
                 "quantized graph so logical parameter memory is not directly comparable"),
    }
    write_json(root / "results/precision_benchmark.json", payload)
    print(json.dumps({pid: {"bytes": r["serialized_bytes"],
                            "median_us": round(r["latency_median_us"], 2),
                            "speedup_vs_fp32": round(r["speedup_vs_fp32_anchor"], 3)}
                      for pid, r in rows.items()}, indent=2))
    return payload


def record_outcome() -> dict[str, Any]:
    install()
    config = load_config(CONFIG)
    root = f17.artifact_root(config)
    summary = read_json(root / "results/pathway_summary.json")["pathways"]
    gate = read_json(root / "integrity/fp16_validity_gate.json")
    candidate = summary.get("F16H")
    if candidate is None:
        raise RuntimeError("F16H has no results yet")

    behaviour = candidate["behaviour_all_curricula_pass"]
    payload = {
        **provenance(config, CONFIG),
        "classification": "FP16_PRESERVES_COMPETENCE" if behaviour else "FP16_FAILS_RETENTION",
        "decision_rule_source": "docs/F18_FP16_CONTROL_PROTOCOL.md (frozen before results)",
        "fp16_validity": {"classification": gate["classification"],
                          "execution_label": gate["execution_label"],
                          "accumulation_width": gate["accumulation_width"]},
        "statuses": {pid: summary[pid]["statuses"] for pid in sorted(summary)},
        "behaviour_all_curricula_pass": {pid: summary[pid]["behaviour_all_curricula_pass"]
                                         for pid in sorted(summary)},
        "fidelity_all_curricula_pass": {pid: summary[pid]["fidelity_all_curricula_pass"]
                                        for pid in sorted(summary)},
        "eligible_under_frozen_selection_rule": {pid: summary[pid]["eligible"]
                                                 for pid in sorted(summary)},
        "selection_rule_note": ("the frozen selection rule requires INT8; an FP16 candidate "
                                "cannot satisfy it by construction, so eligibility is reported "
                                "unchanged rather than redefined"),
        "stop_after_this_stage": True,
        "further_sweeps_started": False,
        "final_holdout_opened": False,
        "final_holdout_seeds_sealed": [int(s) for s in config["seeds"]["sealed_final_holdout"]],
    }
    write_json(root / "results/f18_outcome.json", payload)
    print(json.dumps({"classification": payload["classification"],
                      "F16H": payload["statuses"].get("F16H")}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="F18 FP16 control")
    parser.add_argument("command", choices=("fp16-determinism-check", "evaluate",
                                            "benchmark", "results", "outcome"))
    parser.add_argument("--pathway", default="F16H")
    args = parser.parse_args()
    if args.command == "fp16-determinism-check":
        fp16_determinism_check()
    elif args.command == "evaluate":
        install()
        require_addendum(f17.artifact_root(load_config(CONFIG)))
        print(json.dumps(f17.evaluate_pathway(args.pathway), indent=2, default=str))
    elif args.command == "benchmark":
        benchmark()
    elif args.command == "results":
        install()
        f17.build_results()
    else:
        record_outcome()


if __name__ == "__main__":
    main()
