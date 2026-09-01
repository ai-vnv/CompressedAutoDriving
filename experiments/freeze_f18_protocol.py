#!/usr/bin/env python3
"""Freeze the F18 FP16 control: build the candidate, emit the config, run the validity gate.

Everything here happens BEFORE any curriculum result exists. The gates, seeds, fidelity
thresholds and media rules are copied verbatim out of the frozen F17 config (programmatic
copy, not retyped) so F18 verdicts stay directly comparable to F17's.

The A0 reference and the A3/A6 comparator rows are reused from F17 — identical block,
identical backend, identical reference checkpoint — so F18 evaluates exactly one new
candidate. Their source CSV hashes are recorded.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import platform
import shutil
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    import tomli as tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.actor_compression import load_dense_actor  # noqa: E402
from duckie_pomdp.optimization.cross_curriculum_recovery import file_sha256  # noqa: E402

F17_CONFIG = ROOT / "configs/f17_optimization_method_order_v1.toml"
F18_CONFIG = ROOT / "configs/f18_fp16_control_v1.toml"
PROTOCOL = ROOT / "docs/F18_FP16_CONTROL_PROTOCOL.md"
F17_ART = ROOT / "artifacts/f17_optimization_method_order_v1"
F18_ART = ROOT / "artifacts/f18_fp16_control_v1"
ANCHOR_SHA = "64c84cd0bad44ddaa564a5895c88b82254950752b322030ce67df912a3667276"
CUR = ["c0", "c1", "c2", "c3", "c4"]


def toml_dump(value) -> str:
    """Minimal TOML serializer for the copied literal sections."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_dump(v) for v in value) + "]"
    raise TypeError(type(value))


def emit_section(name: str, table: dict, lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in table.items() if isinstance(v, dict)}
    if scalars or not nested:
        lines.append(f"[{name}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {toml_dump(value)}")
        lines.append("")
    for key, value in nested.items():
        emit_section(f"{name}.{key}", value, lines)


def build_fp16_candidate(anchor: Path, target: Path) -> dict:
    """Cast the anchor's weights to float16 and serialize them as 2-byte tensors."""
    payload = torch.load(anchor, map_location="cpu", weights_only=False)
    half_state = {k: v.half() if torch.is_floating_point(v) else v
                  for k, v in payload["state_dict"].items()}
    for name, tensor in half_state.items():
        if torch.is_floating_point(tensor) and tensor.element_size() != 2:
            raise RuntimeError(f"{name} did not serialize as 2-byte float")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": payload["spec"], "state_dict": half_state,
                "provenance": {"parent_checkpoint": str(anchor), "parent_sha256": ANCHOR_SHA,
                               "transform": "state_dict cast to torch.float16",
                               "retrained": False, "width_changed": False}}, target)
    return {"parameters": int(sum(v.numel() for v in half_state.values()))}


def load_fp16_actor(path: Path) -> torch.nn.Module:
    """Rebuild the actor and put it in genuine half precision.

    load_state_dict silently widens the stored fp16 tensors to the module's fp32
    parameters, so the module is explicitly halved afterwards. The round trip is
    value-exact (every value came from fp16) and is asserted bitwise below.
    """
    stored = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    actor, _ = load_dense_actor(path)
    actor = actor.half().eval()
    for name, parameter in actor.state_dict().items():
        if not torch.equal(parameter, stored[name].half()):
            raise RuntimeError(f"fp16 round trip changed {name}")
    return actor


def validity_gate(anchor: Path, candidate: Path) -> dict:
    """V1/V2/V3 plus an accumulation-width characterization. Criteria frozen in the protocol."""
    torch.set_num_threads(1)
    fp32 = load_dense_actor(anchor)[0].eval()
    half = load_fp16_actor(candidate)

    rng = np.random.default_rng(1801)
    probe = torch.as_tensor(rng.standard_normal((512, 29)), dtype=torch.float32)

    checks: dict[str, object] = {}
    with torch.inference_mode():
        out16 = half(probe.half())
        out32 = fp32(probe)
    checks["v1_half_forward_executes"] = True
    checks["v2_output_dtype_is_float16"] = out16.dtype == torch.float16
    checks["v3_parameters_are_2_byte"] = all(
        p.element_size() == 2 for p in half.parameters())
    checks["v3_outputs_differ_from_fp32"] = not torch.equal(out16.float(), out32)
    checks["v3_not_silent_fp32_upcast"] = bool(checks["v3_outputs_differ_from_fp32"])

    # Accumulation width: sum N ones through a half Linear. Sequential fp16 accumulation
    # saturates at 2048 (spacing 2; +1.0 is an exact tie, round-half-to-even holds); a wider
    # accumulator returns N, and N is exactly representable in fp16 so the output cast to
    # fp16 cannot hide the difference.
    accumulation = {}
    for n in (4096, 8192):
        linear = torch.nn.Linear(n, 1, bias=False).eval()
        with torch.no_grad():
            linear.weight.fill_(1.0)
        ones = torch.ones(1, n)
        with torch.inference_mode():
            accumulation[str(n)] = {
                "fp32": float(linear(ones).item()),
                "fp16": float(copy.deepcopy(linear).half()(ones.half()).item()),
                "exact": float(n),
            }
    wide = all(v["fp16"] == v["exact"] for v in accumulation.values())

    difference = (out16.float() - out32).abs()
    passed = all(bool(v) for v in checks.values())
    return {
        "classification": "PASS" if passed else "FAIL",
        "checks": {k: bool(v) for k, v in checks.items()},
        "accumulation_probe": accumulation,
        "accumulation_width": "wider_than_fp16" if wide else "fp16_narrow",
        "execution_label": (
            "FP16 weights and activations with FP32-wide accumulation (standard fp16 "
            "inference semantics), FP32 I/O boundary"
            if wide else "FP16 weights, activations and accumulation"),
        "silent_upcast_stop_condition_triggered": bool(torch.equal(out16.float(), out32)),
        "action_difference_vs_fp32": {
            "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
            "probe_rows": int(probe.shape[0]),
        },
        "criteria_source": "docs/F18_FP16_CONTROL_PROTOCOL.md (frozen before this probe ran)",
    }


def main() -> None:
    if F18_CONFIG.exists():
        raise RuntimeError(f"refusing to overwrite {F18_CONFIG}")
    with F17_CONFIG.open("rb") as stream:
        f17 = tomllib.load(stream)
    def frozen_path(pid: str) -> Path:
        return (F17_CONFIG.parent / f17["pathways"][pid]["checkpoint"]).resolve()

    anchor = frozen_path("A3")
    if file_sha256(anchor) != ANCHOR_SHA:
        raise RuntimeError("anchor checkpoint does not match the frozen F17 hash")

    candidate = F18_ART / "candidates/actor_fp16.pt"
    info = build_fp16_candidate(anchor, candidate)

    # ---- config: gates and seeds copied verbatim from the frozen F17 config ----
    lines = [
        "schema_version = 1",
        'experiment = "F18 FP16 Control v1"',
        "",
        "# Gates, seeds, fidelity thresholds, evaluation and media rules below are a",
        "# PROGRAMMATIC VERBATIM COPY of configs/f17_optimization_method_order_v1.toml,",
        "# so F18 verdicts are directly comparable to F17's. Nothing was retyped.",
        "",
        "[scope]",
        'primary_question = "does FP32 -> FP16 precision reduction preserve C0-C4 competence"',
        'manipulated_factor = "actor_numeric_precision"',
        'held_fixed = ["actor_width", "pruning_schedule", "teacher", "kd_dataset", "evaluation_block", "gates"]',
        "no_kd_retraining = true",
        "no_width_change = true",
        "no_width_sweep = true",
        "no_pruning_schedule_study = true",
        "no_training_seed_study = true",
        "new_candidates = 1",
        "",
    ]
    for section in ("frozen", "anchor", "seeds", "determinism", "retention", "fidelity",
                    "evaluation", "failure_phenotype", "media", "selection", "final_holdout"):
        emit_section(section, f17[section], lines)

    lines += [
        "# ---------------------------------------------------------------------------",
        "# Members. Only F16H is evaluated by F18. A0/A3/A6 rows are reused from F17 on",
        "# the identical block and backend; re-running them could only add noise.",
        "# ---------------------------------------------------------------------------",
        "[pathways.A0]",
        'label = "Original Policy"',
        'pathway = "none"',
        f'checkpoint = {toml_dump(str(frozen_path("A0")))}',
        "width = 256", 'precision = "FP32"', 'role = "reference"', "reused_from_f17 = true", "",
        "[pathways.A3]",
        'label = "Recovered FP32 anchor (prune + balanced C0-C4 KD)"',
        'pathway = "prune -> KD(balanced C0-C4)"',
        f"checkpoint = {toml_dump(str(anchor))}",
        "width = 64", 'precision = "FP32"', "is_anchor = true", "reused_from_f17 = true", "",
        "[pathways.A6]",
        'label = "Anchor after frozen PTQ (INT8 comparator)"',
        'pathway = "prune -> KD(balanced) -> PTQ"',
        f'checkpoint = {toml_dump(str(frozen_path("A6")))}',
        "width = 64", 'precision = "INT8"', 'parent = "A3"', "reused_from_f17 = true", "",
        "[pathways.F16H]",
        'label = "Anchor cast to FP16 (new candidate)"',
        'pathway = "prune -> KD(balanced) -> FP16"',
        f"checkpoint = {toml_dump(str(candidate))}",
        "width = 64", 'precision = "FP16"', 'parent = "A3"',
        "fp16 = true", "retrained = false", "",
        "[comparisons]",
        '"A3 vs F16H" = "PRIMARY: does reduced floating-point precision preserve competence on this fixed checkpoint"',
        '"F16H vs A6" = "does FP16 retain what the tested INT8 procedure lost (both from the same FP32 parent)"',
        "",
        "[comparisons.forbidden]",
        'deployment_readiness = "no deployment claim; latency and size are reported as measured"',
        'generalizing_precision = "claims apply to this checkpoint, block and backend only"',
        "",
        "[benchmark]",
        "threads = 1", "batch_size = 1", "warmup_iterations = 1000",
        "iterations = 10000", "repeats = 5", "",
        "[artifacts]",
        'directory = "../artifacts/f18_fp16_control_v1"',
        "",
    ]
    F18_CONFIG.write_text("\n".join(lines), encoding="utf-8")
    with F18_CONFIG.open("rb") as stream:
        parsed = tomllib.load(stream)
    for section in ("retention", "fidelity", "seeds", "media", "determinism"):
        if parsed[section] != f17[section]:
            raise RuntimeError(f"verbatim copy failed for [{section}]")

    # ---- reuse F17's A0 / A3 / A6 episode rows ----
    (F18_ART / "closed_loop").mkdir(parents=True, exist_ok=True)
    reused = {}
    for pid in ("A0", "A3", "A6"):
        source = F17_ART / "closed_loop" / f"{pid}_episodes.csv"
        shutil.copy2(source, F18_ART / "closed_loop" / f"{pid}_episodes.csv")
        reused[pid] = {"source": str(source), "sha256": file_sha256(source)}

    # ---- registry ----
    registry = {}
    for pid, entry in parsed["pathways"].items():
        path = Path(entry["checkpoint"])
        registry[pid] = {
            "pathway_id": pid, "label": entry["label"],
            "optimization_method_order": entry["pathway"],
            "checkpoint": str(path), "sha256": file_sha256(path),
            "bytes": path.stat().st_size, "width": entry["width"],
            "precision": entry["precision"], "pruning_schedule": "Direct",
            "int8": entry["precision"] == "INT8", "fp16": bool(entry.get("fp16", False)),
            "parent": entry.get("parent"), "role": entry.get("role", "pathway"),
            "reused_from_f17": bool(entry.get("reused_from_f17", False)),
            "retrained_by_f18": False,
        }
    (F18_ART / "integrity").mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    (F18_ART / "pathway_registry.json").write_text(json.dumps(
        {"schema_version": 1, "created_at_utc": stamp, "config_path": str(F18_CONFIG),
         "config_sha256": file_sha256(F18_CONFIG), "torch": torch.__version__,
         "numpy": np.__version__, "python": sys.version.split()[0],
         "platform": platform.platform(), "pathways": registry,
         "reused_episode_csvs": reused}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- validity gate ----
    gate = validity_gate(anchor, candidate)
    gate.update({"created_at_utc": stamp, "candidate_sha256": file_sha256(candidate),
                 "parent_sha256": ANCHOR_SHA, "parameters": info["parameters"]})
    (F18_ART / "integrity/fp16_validity_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- protocol manifest ----
    (F18_ART / "integrity/protocol_manifest.json").write_text(json.dumps({
        "schema_version": 1, "created_at_utc": stamp,
        "frozen_before_any_result": True,
        "documents": {str(p.relative_to(ROOT)): file_sha256(p) for p in
                      (PROTOCOL, F18_CONFIG, F17_CONFIG)},
        "candidate": {"path": str(candidate), "sha256": file_sha256(candidate),
                      "bytes": candidate.stat().st_size, "parent_sha256": ANCHOR_SHA},
        "reused_episode_csvs": reused,
        "sealed_final_holdout": [int(s) for s in parsed["seeds"]["sealed_final_holdout"]],
        "holdout_opened": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "config": str(F18_CONFIG.relative_to(ROOT)),
        "candidate_bytes": candidate.stat().st_size,
        "anchor_bytes": anchor.stat().st_size,
        "validity_gate": gate["classification"],
        "accumulation_width": gate["accumulation_width"],
        "execution_label": gate["execution_label"],
        "max_abs_action_difference_vs_fp32": gate["action_difference_vs_fp32"]["max_abs"],
    }, indent=2))


if __name__ == "__main__":
    main()
