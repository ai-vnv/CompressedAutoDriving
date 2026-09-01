#!/usr/bin/env python3
"""Actor-only efficiency for the F15 recovery candidates.

The built-in ``run_f15_recovery.py efficiency`` stage sources its models from
``final/final_holdout.json``.  F15 stopped before the once-only holdout because no INT8
candidate satisfied the frozen gates, so that stage cannot run.  Prompt section 25 still
requires efficiency to be reported, so this script measures the same quantities with the
same ``benchmark_actor`` implementation and the same frozen ``[benchmark]`` parameters,
sourcing models from the recovery selection results instead.

It writes ``final/efficiency_summary.json``.  It performs no simulation, opens no holdout
seed, and selects no candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.actor_compression import ActorSpec  # noqa: E402
from duckie_pomdp.optimization.compression_metrics import benchmark_actor  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256,
    verify_registry,
)
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    CURRICULA,
    artifact_root,
    frozen_paths,
    load_actor,
    load_config,
    provenance,
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/f15_cross_curriculum_recovery_v1.toml"


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    target = root / "final/efficiency_summary.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    paths = frozen_paths(config, CONFIG)
    matrix = verify_registry(
        paths["ablation_registry"],
        expected_registry_sha256=config["frozen"]["f12_ablation_registry_sha256"],
        collection_key="variants",
    )
    benchmark = config["benchmark"]

    # Original reference, the historical INT8 endpoint, and every F15 recovery candidate.
    entries: list[tuple[str, dict, dict | None]] = [
        ("Original Policy", matrix["A0"], None),
        ("Final INT8 Policy (historical A7)", matrix["A7"], None),
    ]
    for method, label in (("fp32", "Recovered 64x64 + Multi-Curriculum KD (FP32)"),
                          ("ptq", "Recovered 64x64 + PTQ (INT8)"),
                          ("qat", "Recovered 64x64 + Multi-Curriculum QAT+KD (INT8)")):
        result_path = root / f"recovery/{method}/w64/selection_result.json"
        if not result_path.exists():
            continue
        payload = read_json(result_path)
        entries.append((label, payload["entry"], payload))

    models = []
    for label, entry, payload in entries:
        width = int(entry["hidden_sizes"][0])
        actor = load_actor(entry)
        metrics = benchmark_actor(
            actor, ActorSpec(hidden_sizes=(width, width)), entry["model_path"],
            warmup=int(benchmark["warmup_iterations"]), iterations=int(benchmark["timed_iterations"]),
            repeats=int(benchmark["repeats"]), threads=int(benchmark["threads"]), int8=bool(entry["int8"]),
        )
        record = {
            "label": label,
            "model_id": entry["variant"],
            "sha256": entry["sha256"],
            "hidden_sizes": entry["hidden_sizes"],
            "int8": bool(entry["int8"]),
            "parameter_count": metrics["dense_parameter_count"],
            **metrics,
        }
        if payload is not None:
            record["eligible"] = bool(payload["eligible"])
            record["all_curricula_behavior_pass"] = bool(payload["all_curricula_behavior_pass"])
            record["fidelity_all_curricula_pass"] = bool(payload["fidelity"]["all_curricula_pass"])
            record["curricula_behavior_passed"] = sum(
                payload["behavior"]["decisions"][entry["variant"]][c]["status"] == "PASS"
                for c in CURRICULA
            )
            record["evaluation_split"] = "recovery selection seeds 180201-180208"
        else:
            record["evaluation_split"] = "not evaluated by this script"
        models.append(record)

    original = models[0]

    def ratios(record: dict) -> dict:
        return {
            "parameter_reduction_fraction": 1.0 - record["parameter_count"] / original["parameter_count"],
            "file_size_reduction_fraction": 1.0 - record["actor_checkpoint_size_bytes"] / original["actor_checkpoint_size_bytes"],
            "actor_only_cpu_speedup": original["batch1_latency_us_median"] / record["batch1_latency_us_median"],
        }

    for record in models[1:]:
        record["versus_original"] = ratios(record)

    output = {
        **provenance(config, CONFIG),
        "scope": "actor-only CPU benchmark; perception (MobileNet/YOLO/belief) is unchanged",
        "claim_boundary": (
            "actor-only latency is not end-to-end visuomotor latency; no end-to-end speedup is claimed"
        ),
        "benchmark_parameters": {
            "device": benchmark["device"], "batch_size": benchmark["batch_size"],
            "threads": benchmark["threads"], "warmup_iterations": benchmark["warmup_iterations"],
            "timed_iterations": benchmark["timed_iterations"], "repeats": benchmark["repeats"],
        },
        "final_holdout_opened": False,
        "final_candidate_frozen": (root / "final/final_candidate.json").exists(),
        "note": (
            "F15 stopped before the once-only holdout: the FP32 recovered actor satisfied every "
            "frozen C0-C4 gate, but neither PTQ nor multi-curriculum QAT+KD produced an eligible "
            "INT8 actor at width 64. Efficiency is therefore reported for all measured candidates "
            "rather than for a single selected deployment model."
        ),
        "models": models,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, output)
    print(json.dumps(
        {
            "written": str(target),
            "models": [
                {
                    "label": m["label"],
                    "params": m["parameter_count"],
                    "bytes": m["actor_checkpoint_size_bytes"],
                    "latency_us_median": round(m["batch1_latency_us_median"], 3),
                    "eligible": m.get("eligible"),
                }
                for m in models
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
