#!/usr/bin/env python3
"""F17 optimization-method-order runner.

Evaluates every frozen optimization pathway on one matched deterministic block. No pathway
member is retrained: each is an existing frozen checkpoint from F12 or F15, so training
realization cannot contaminate the method-order comparison.

Reuses the validated F16 machinery: the deterministic backend that passed the determinism
gate, and the in-rollout RGB ring buffer that passed the media gate.

Subcommands:
  evaluate --pathway <ID>   run one pathway on the primary block (resumable)
  results                   build pathway results, fidelity, comparisons
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment  # noqa: E402
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import (  # noqa: E402
    file_sha256, first_objective_failure_event, retention_decision,
)
import run_f15_cross_curriculum_recovery as f15  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    ActorPolicy, append_csv, load_actor, load_config, phase_thresholds, provenance,
    read_csv, read_json, run_episode_with_telemetry, summarize_episode_dicts, write_json,
)
from run_f16_sequence_int8 import RGBRecordingEnvironment, encode_media  # noqa: E402

CONFIG = ROOT / "configs/f17_optimization_method_order_v1.toml"
CUR = ["c0", "c1", "c2", "c3", "c4"]

SAFETY = {"maximum_collision_rate", "new_collisions", "collision_rate",
          "maximum_stop_violation_rate", "stop_violation_rate",
          "maximum_unsafe_episode_rate", "unsafe_episode_rate",
          "minimum_stop_completion_rate", "minimum_restart_rate"}
MARGINAL = {"minimum_clearance", "maximum_minimum_clearance_drop_m"}


class ScopedNondeterministicOps:
    """Temporarily lift strict-determinism enforcement for CPU quantized-actor ops.

    PyTorch's strict mode rejects ``quantized_resize_cpu_`` because the op lacks a
    formal deterministic tag, even though fixed-point integer arithmetic on CPU has no
    nondeterministic reduction path. This context is applied ONLY around INT8 actor
    loading and inference; the CUDA perception path inside environment.step keeps the
    frozen strict configuration. Whether the hybrid configuration is actually
    reproducible is NOT assumed: it is verified empirically by the preregistered
    int8-determinism addendum check, using the same criteria as the frozen gate.
    """

    def __enter__(self):
        self._was = torch.are_deterministic_algorithms_enabled()
        if self._was:
            torch.use_deterministic_algorithms(False)
        return self

    def __exit__(self, *exc):
        if self._was:
            torch.use_deterministic_algorithms(True)
        return False


class Int8ActorPolicy(ActorPolicy):
    """ActorPolicy whose forward runs inside the scoped-nondeterminism context."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        with ScopedNondeterministicOps():
            return super().act(observation)


def make_policy(entry: dict[str, Any]) -> ActorPolicy:
    if entry.get("int8"):
        with ScopedNondeterministicOps():
            actor = load_actor(entry)
        return Int8ActorPolicy(entry["name"], actor)
    return ActorPolicy(entry["name"], load_actor(entry))


def resolve(config, value: str) -> Path:
    return (CONFIG.parent / value).resolve()


def artifact_root(config) -> Path:
    return resolve(config, config["artifacts"]["directory"])


def phenotype(failed: list[str]) -> str:
    if not failed:
        return "none"
    if set(failed) & SAFETY:
        return "safety_relevant"
    if set(failed) - MARGINAL:
        return "behavioural"
    return "marginal_clearance_only"


def apply_determinism(config) -> dict[str, Any]:
    import os
    gate_path = resolve(config, config["determinism"]["inherited_gate"])
    gate = read_json(gate_path)
    if gate["classification"] != "PASS":
        raise RuntimeError("inherited determinism gate did not pass")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != config["determinism"]["cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before torch initialises")
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("frozen backend requires CUDA")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    return {"backend": gate["selected_backend"], "gate_sha256": file_sha256(gate_path)}


def evaluate_pathway(pathway_id: str) -> dict[str, Any]:
    config = load_config(CONFIG)
    determinism = apply_determinism(config)
    root = artifact_root(config)
    registry = read_json(root / "pathway_registry.json")["pathways"]
    if pathway_id not in registry:
        raise KeyError(f"unknown pathway: {pathway_id}")
    record = registry[pathway_id]
    if file_sha256(record["checkpoint"]) != record["sha256"]:
        raise RuntimeError(f"checkpoint changed since freeze: {pathway_id}")

    entry = {
        "variant": pathway_id, "name": record["label"],
        "model_path": record["checkpoint"], "sha256": record["sha256"],
        "hidden_sizes": [record["width"], record["width"]], "int8": record["int8"],
    }

    policy_config = resolve(config, "f10_ppo_visual_objects_v30.toml")
    f12_config = resolve(config, "f12_belief_ppo_compression_v1.toml")
    protocol = load_ppo_curriculum_protocol(policy_config)
    thresholds = phase_thresholds(f12_config)
    seeds = [int(s) for s in config["seeds"]["primary_evaluation"]]
    sealed = {int(s) for s in config["seeds"]["sealed_final_holdout"]}
    if sealed & set(seeds):
        raise RuntimeError("refusing to evaluate on sealed holdout seeds")

    ring = int(config["media"]["rgb_ring_buffer_steps"])
    before = int(config["evaluation"]["failure_window_steps_before"])
    after = int(config["evaluation"]["failure_window_steps_after"])

    episode_csv = root / "closed_loop" / f"{pathway_id}_episodes.csv"
    episode_csv.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["curriculum"], int(r["seed"])) for r in read_csv(episode_csv)}

    policy = make_policy(entry)
    f15._CURRENT_MODEL_PATH = Path(entry["model_path"])
    if entry["int8"]:
        addendum = root / "integrity/int8_determinism_addendum.json"
        if not addendum.exists() or read_json(addendum)["classification"] != "PASS":
            raise RuntimeError(
                "INT8 pathways require the passed int8-determinism addendum check; "
                "run `int8-determinism-check` first"
            )

    for curriculum in CUR:
        pending = [s for s in seeds if (curriculum, s) not in done]
        if not pending:
            continue
        environment = PPOCurriculumEnvironment(
            policy_config, stage=curriculum,
            split=f"f17_{pathway_id}_{curriculum}", seeds=tuple(seeds),
        )
        recorder = RGBRecordingEnvironment(environment, ring)
        try:
            for seed in pending:
                target = root / "telemetry" / pathway_id / curriculum / f"seed_{seed}" / "trace.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                row = run_episode_with_telemetry(
                    recorder, seed=seed, policy=policy,
                    protocol=protocol, thresholds=thresholds, target=target,
                )
                append_csv(episode_csv, {"model_id": pathway_id, "model_name": entry["name"],
                                         "curriculum": curriculum, **row})

                with np.load(target, allow_pickle=False) as archive:
                    flags = {n: np.asarray(archive[n], dtype=bool) for n in
                             ("collision", "unsafe", "stop_violation", "lane_failure",
                              "invalid_pose", "timeout", "terminated", "truncated", "completed")}
                    progress = np.asarray(archive["progress_m"], dtype=np.float32)
                    physical = np.asarray(archive["physical_action"], dtype=np.float32)
                step_rows = [{"step": i, **{n: bool(v[i]) for n, v in flags.items()}}
                             for i in range(len(progress))]
                event = first_objective_failure_event(step_rows)
                if event is not None:
                    frames = [(i, f) for i, f in recorder.frames
                              if max(0, event["step"] - before) <= i <= event["step"] + after
                              and i < len(progress)]
                    if frames:
                        encode_media(
                            {"curriculum": curriculum, "seed": seed, "frames": frames,
                             "failure_step": event["step"], "episode_length": len(step_rows),
                             "steps": [{"step": r["step"],
                                        "progress_m": float(progress[r["step"]]),
                                        "v_cmd": float(physical[r["step"], 0]),
                                        "omega_cmd": float(physical[r["step"], 1]),
                                        **{k: r[k] for k in ("collision", "unsafe", "stop_violation",
                                                             "lane_failure", "invalid_pose", "timeout")}}
                                       for r in step_rows]},
                            root / "primary_media" / pathway_id / curriculum / f"seed_{seed}",
                            {"model_id": pathway_id, "model_name": entry["name"],
                             "sequence": record["optimization_method_order"],
                             "width": record["width"], "precision": record["precision"],
                             "model_sha256": record["sha256"]},
                        )
                print(f"  {pathway_id} {curriculum} seed={seed} completed={row['completed']} "
                      f"steps={row['steps']}", flush=True)
        finally:
            environment.close()

    return {"pathway": pathway_id, "episode_csv": str(episode_csv), "determinism": determinism}


def int8_determinism_check() -> dict[str, Any]:
    """Verify that the scoped-nondeterminism INT8 configuration is reproducible.

    The frozen gate criteria are reapplied unchanged: three repeats of the same
    (model, curriculum, seed) must produce bit-identical actions, identical progress,
    identical episode length, completion, failure labels, and termination reason. The
    representative model is A6 (INT8, width 64) — the class whose ops required the scoped
    context. FAIL bars every INT8 pathway evaluation.
    """
    config = load_config(CONFIG)
    determinism = apply_determinism(config)
    root = artifact_root(config)
    target = root / "integrity/int8_determinism_addendum.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")

    registry = read_json(root / "pathway_registry.json")["pathways"]
    record = registry["A6"]
    if file_sha256(record["checkpoint"]) != record["sha256"]:
        raise RuntimeError("A6 checkpoint changed since freeze")
    entry = {"variant": "A6", "name": record["label"], "model_path": record["checkpoint"],
             "sha256": record["sha256"], "hidden_sizes": [record["width"]] * 2, "int8": True}
    policy = make_policy(entry)

    policy_config = resolve(config, "f10_ppo_visual_objects_v30.toml")
    protocol = load_ppo_curriculum_protocol(policy_config)
    seed = int(config["seeds"]["primary_evaluation"][0])

    cells = {}
    for curriculum in ("c0", "c3"):
        repeats = []
        for repeat in range(3):
            environment = PPOCurriculumEnvironment(
                policy_config, stage=curriculum,
                split=f"f17_int8_determinism_{curriculum}", seeds=(seed,),
            )
            actions, progress = [], []
            try:
                observation, _ = environment.reset(seed=seed)
                policy.reset(seed)
                info: dict[str, Any] = {}
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
            print(f"  A6 {curriculum} seed={seed} repeat={repeat} steps={len(actions)}", flush=True)
        first = repeats[0]
        checks = {
            "steps_identical": all(r["steps"] == first["steps"] for r in repeats),
            "actions_bit_identical": all(np.array_equal(r["actions"], first["actions"]) for r in repeats),
            "progress_bit_identical": all(np.array_equal(r["progress"], first["progress"]) for r in repeats),
            "completion_identical": all(r["completed"] == first["completed"] for r in repeats),
            "failure_label_identical": all(r["failure_label"] == first["failure_label"] for r in repeats),
            "termination_reason_identical": all(r["termination_reason"] == first["termination_reason"] for r in repeats),
        }
        cells[curriculum] = {"seed": seed, "repeats": 3, "steps": first["steps"], "checks": checks,
                             "reproducible": all(checks.values())}

    passed = all(c["reproducible"] for c in cells.values())
    payload = {
        **provenance(config, CONFIG),
        "classification": "PASS" if passed else "FAIL",
        "kind": "int8_determinism_addendum",
        "mechanism": (
            "torch strict deterministic mode rejects quantized_resize_cpu_ for lacking a "
            "deterministic tag; INT8 actor load/forward therefore run inside a scoped "
            "nondeterminism-allowed context while the CUDA perception path inside "
            "environment.step keeps the frozen strict configuration"
        ),
        "criteria_source": "identical to the frozen F16 determinism gate",
        "determinism": determinism,
        "representative_model": {"pathway": "A6", "sha256": record["sha256"]},
        "cells": cells,
        "consequence_if_fail": "every INT8 pathway evaluation is barred",
    }
    write_json(target, payload)
    print(json.dumps({"classification": payload["classification"],
                      "cells": {k: v["reproducible"] for k, v in cells.items()}}, indent=2))
    if not passed:
        raise SystemExit(1)
    return payload


def build_results() -> dict[str, Any]:
    from duckie_pomdp.optimization.compression_metrics import (
        action_fidelity, actor_physical_predictions,
    )
    from duckie_pomdp.optimization.cross_curriculum_recovery import fidelity_pass

    config = load_config(CONFIG)
    root = artifact_root(config)
    registry = read_json(root / "pathway_registry.json")["pathways"]
    seeds = [int(s) for s in config["seeds"]["primary_evaluation"]]
    loop = root / "closed_loop"

    base_csv = loop / "A0_episodes.csv"
    if not base_csv.exists():
        raise RuntimeError("A0 reference must be evaluated before results can be built")
    base_rows = read_csv(base_csv)
    baseline = {c: summarize_episode_dicts([r for r in base_rows if r["curriculum"] == c]) for c in CUR}

    observations = {}
    for curriculum in CUR:
        chunks = []
        for row in base_rows:
            if row["curriculum"] != curriculum:
                continue
            with np.load(Path(row["trace_path"]), allow_pickle=False) as archive:
                chunks.append(np.asarray(archive["public_normalized_29d"], dtype=np.float32))
        observations[curriculum] = np.concatenate(chunks)
    original = load_actor({"model_path": registry["A0"]["checkpoint"], "int8": False,
                           "hidden_sizes": [256, 256]})
    original_pred = {c: actor_physical_predictions(original, observations[c]) for c in CUR}

    pathway_rows, fidelity_rows, summary = [], [], {}
    for pid in sorted(registry):
        record = registry[pid]
        path = loop / f"{pid}_episodes.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        covered = {(r["curriculum"], int(r["seed"])) for r in rows}
        complete = all((c, s) in covered for c in CUR for s in seeds)
        present = [c for c in CUR if any(r["curriculum"] == c for r in rows)]
        if not present:
            continue
        summaries = {c: summarize_episode_dicts([r for r in rows if r["curriculum"] == c])
                     for c in present}
        actor = load_actor({"model_path": record["checkpoint"], "int8": record["int8"],
                            "hidden_sizes": [record["width"]] * 2})
        statuses, behaviour_pass, fidelity_all = {}, complete, complete
        for curriculum in present:
            n = sum(1 for r in rows if r["curriculum"] == curriculum)
            if pid == "A0":
                status, failed = "REFERENCE", []
            elif n < len(seeds):
                status, failed = f"PARTIAL_{n}/{len(seeds)}", []
            else:
                decision = retention_decision(
                    curriculum, summaries[curriculum], baseline[curriculum],
                    config["retention"]["absolute"], config["retention"]["relative_to_original"],
                    candidate_prior=summaries, original_prior=baseline)
                dd = decision if isinstance(decision, dict) else decision.__dict__
                status = dd.get("status")
                failed = [k for grp in ("absolute_checks", "relative_checks")
                          for k, v in (dd.get(grp) or {}).items() if not v]
            statuses[curriculum.upper()] = status
            behaviour_pass &= status in {"PASS", "REFERENCE"}
            s = summaries[curriculum]
            pathway_rows.append({
                "pathway_id": pid, "label": record["label"],
                "optimization_method_order": record["optimization_method_order"],
                "target_width": record["width"], "precision": record["precision"],
                "pruning_schedule": record["pruning_schedule"],
                "curriculum": curriculum.upper(), "status": status,
                "failure_phenotype": phenotype(failed), "failed_checks": "|".join(failed),
                "episodes": n, "completion_rate": s["completion_rate"],
                "mean_progress_m": s["mean_progress_m"], "collision_rate": s["collision_rate"],
                "unsafe_episode_rate": s["unsafe_episode_rate"],
                "stop_completion_rate": s["stop_completion_rate"],
                "stop_violation_rate": s["stop_violation_rate"],
                "restart_rate": s["restart_rate"], "timeout_rate": s["timeout_rate"],
                "lane_failure_rate": s["lane_failure_rate"],
                "invalid_pose_rate": s["invalid_pose_rate"],
                "minimum_pedestrian_clearance_m": s.get("minimum_pedestrian_clearance_m"),
                "mean_v_cmd_mps": s["mean_v_cmd_mps"],
                "mean_abs_omega_cmd_rad_s": s["mean_abs_omega_cmd_rad_s"],
                "stationary_fraction": s["stationary_fraction"],
            })

            metrics = action_fidelity(
                original_pred[curriculum],
                actor_physical_predictions(actor, observations[curriculum]),
                omega_deadband=float(config["evaluation"]["omega_sign_deadband_rad_s"]))
            passed, checks = fidelity_pass(metrics, config["fidelity"])
            fidelity_all &= passed
            v, om = metrics["v_cmd_mps"], metrics["omega_cmd_rad_s"]
            fidelity_rows.append({
                "pathway_id": pid, "label": record["label"],
                "optimization_method_order": record["optimization_method_order"],
                "target_width": record["width"], "precision": record["precision"],
                "curriculum": curriculum.upper(), "pass": passed,
                "v_mae_mps": v["mae"], "v_rmse_mps": v["rmse"],
                "v_p95_mps": v["p95_absolute_error"], "v_p99_mps": v["p99_absolute_error"],
                "v_bias_mps": v["bias"],
                "omega_mae_rad_s": om["mae"], "omega_rmse_rad_s": om["rmse"],
                "omega_p95_rad_s": om["p95_absolute_error"], "omega_p99_rad_s": om["p99_absolute_error"],
                "omega_bias_rad_s": om["bias"], "omega_pearson": om["pearson"],
                "omega_spearman": om["spearman"],
                "omega_sign_disagreement": metrics["omega_sign"]["disagreement_frequency"],
                "saturation_disagreement": metrics["action_bound_saturation_frequency"]["disagreement"],
                "failed_checks": "|".join(k for k, v2 in checks.items() if not v2),
            })
        summary[pid] = {
            "label": record["label"],
            "optimization_method_order": record["optimization_method_order"],
            "target_width": record["width"], "precision": record["precision"],
            "episodes_complete": complete, "statuses": statuses,
            "behaviour_all_curricula_pass": behaviour_pass,
            "fidelity_all_curricula_pass": fidelity_all,
            "eligible": bool(complete and behaviour_pass and fidelity_all and record["int8"]),
        }

    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pathway_results.csv", pathway_rows),
                       ("same_state_fidelity.csv", fidelity_rows)):
        if not rows:
            continue
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        print(f"  wrote {name}: {len(rows)} rows")

    comparisons = {}
    for label, question in config["comparisons"].items():
        if not isinstance(question, str) or " vs " not in label:
            continue
        left, right = [x.strip() for x in label.split("vs")]
        if left in summary and right in summary:
            comparisons[label] = {
                "question": question,
                left: summary[left]["statuses"], right: summary[right]["statuses"],
                "changed_curricula": [c for c in [x.upper() for x in CUR]
                                      if summary[left]["statuses"].get(c) != summary[right]["statuses"].get(c)],
            }
    write_json(out / "pathway_summary.json", {
        **provenance(config, CONFIG), "pathways": summary, "comparisons": comparisons,
        "eligibility_note": "eligibility requires INT8 plus all frozen behaviour, fidelity and safety gates; diagnostics never affect it",
    })

    print()
    print(f"{'id':<4}{'prec':<6}{'W':>5}  " + "".join(c.upper().ljust(9) for c in CUR) + " pathway")
    for pid in sorted(summary):
        s = summary[pid]
        cells = [s["statuses"].get(c.upper(), "-") for c in CUR]
        print(f"{pid:<4}{s['precision']:<6}{s['target_width']:>5}  "
              + "".join(x.ljust(9) for x in cells) + f" {s['optimization_method_order']}")
    return {"pathways": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "results", "int8-determinism-check"))
    parser.add_argument("--pathway")
    args = parser.parse_args()
    if args.command == "evaluate":
        if not args.pathway:
            parser.error("evaluate requires --pathway")
        print(json.dumps(evaluate_pathway(args.pathway), indent=2, default=str))
    elif args.command == "int8-determinism-check":
        int8_determinism_check()
    else:
        build_results()


if __name__ == "__main__":
    main()
