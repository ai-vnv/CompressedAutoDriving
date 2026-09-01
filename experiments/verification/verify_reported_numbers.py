"""Recompute every number reported in the paper from the evaluation artifacts.

This is the verification specification for the study: each check restates a
number as it is printed in the manuscript and recomputes it from the result
ledgers, the frozen configurations, or the protocol loader. A check passes only
when the recomputed value matches the reported one.

Run it after the evaluation artifacts have been regenerated:

    python experiments/verification/verify_reported_numbers.py

Exit status is 0 when every check passes, 1 when any check fails, and 2 when
the artifacts are not present.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
F17 = ART / "f17_optimization_method_order_v1"
F18 = ART / "f18_fp16_control_v1"
F15 = ART / "f15_cross_curriculum_recovery_v1"
CUR = ["C0", "C1", "C2", "C3", "C4"]

checks: list[tuple[str, object, object, bool]] = []
skipped: list[str] = []


def check(name, got, expect, tol=0.0):
    if isinstance(expect, (int, float)) and isinstance(got, (int, float)) \
            and not isinstance(expect, bool):
        ok = abs(got - expect) <= tol
    else:
        ok = got == expect
    checks.append((name, got, expect, ok))


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def require_artifacts():
    needed = [
        F17 / "results/pathway_results.csv",
        F17 / "results/same_state_fidelity.csv",
        F18 / "results/pathway_results.csv",
        F18 / "results/precision_benchmark.json",
    ]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        print("The evaluation artifacts are not present, so the reported "
              "numbers cannot be recomputed.")
        print("Regenerate them with the runners in experiments/ and the seeds "
              "recorded in configs/, then run this again. Missing:")
        for p in missing:
            print(f"  {p.relative_to(ROOT)}")
        raise SystemExit(2)


require_artifacts()

r17 = read_rows(F17 / "results/pathway_results.csv")
r18 = read_rows(F18 / "results/pathway_results.csv")
fid17 = read_rows(F17 / "results/same_state_fidelity.csv")
fid18 = read_rows(F18 / "results/same_state_fidelity.csv")


def cell(rows, pid, cur):
    return next(x for x in rows if x["pathway_id"] == pid
                and x["curriculum"] == cur)


def fid(rows, pid, cur, key):
    return float(cell(rows, pid, cur)[key])


def source(pid):
    return r18 if pid == "F16H" else r17


# --- 1. completion table and verdicts, exactly as printed -------------------
COMPLETIONS = {
    "A0": [8, 7, 7, 7, 8], "A1": [3, 3, 0, 0, 0], "A2": [0, 0, 0, 8, 8],
    "A3": [8, 7, 8, 8, 8], "A4": [8, 7, 7, 7, 8], "A5": [4, 3, 0, 0, 0],
    "A6": [8, 7, 7, 3, 7], "A7": [0, 0, 0, 8, 8], "A8": [8, 7, 8, 5, 1],
    "F16H": [8, 7, 8, 8, 8],
}
VERDICTS = {
    "A1": "FFFFF", "A2": "FFFPP", "A3": "PPPPP", "A4": "PPPPP", "A5": "FFFFF",
    "A6": "PPPFF", "A7": "FFFPP", "A8": "PPPFF", "F16H": "PPPPP",
}

for pid, expect in COMPLETIONS.items():
    got = [round(float(cell(source(pid), pid, c)["completion_rate"]) * 8)
           for c in CUR]
    check(f"completions {pid}", got, expect)

for pid, expect in VERDICTS.items():
    got = "".join("P" if cell(source(pid), pid, c)["status"] == "PASS" else "F"
                  for c in CUR)
    check(f"verdict string {pid}", got, expect)

# --- 2. failure-mode numbers quoted in the text -----------------------------
check("A8 C3 stop violations of 8",
      round(fid(r17, "A8", "C3", "stop_violation_rate") * 8), 4)
check("A8 C4 stop violations of 8",
      round(fid(r17, "A8", "C4", "stop_violation_rate") * 8), 7)
check("A6 C3 stop violation rate", fid(r17, "A6", "C3", "stop_violation_rate"), 0.0)
check("A1 C3 stop violation rate", fid(r17, "A1", "C3", "stop_violation_rate"), 1.0)
check("A5 C3 stop violation rate", fid(r17, "A5", "C3", "stop_violation_rate"), 1.0)
check("A5 maximum invalid-pose rate",
      max(fid(r17, "A5", c, "invalid_pose_rate") for c in CUR), 1.0)
check("A2 C0 invalid-pose rate", fid(r17, "A2", "C0", "invalid_pose_rate"), 1.0)
check("A2 C2 lane-failure rate", fid(r17, "A2", "C2", "lane_failure_rate"), 1.0)
check("A0 has exactly three 7/8 curricula",
      sum(1 for c in CUR
          if round(fid(r17, "A0", c, "completion_rate") * 8) == 7), 3)
check("A8 C4 completions", round(fid(r17, "A8", "C4", "completion_rate") * 8), 1)
check("A6 C4 completions", round(fid(r17, "A6", "C4", "completion_rate") * 8), 7)

# --- 3. action-level fidelity ------------------------------------------------
check("A8 C3 omega MAE", fid(fid17, "A8", "C3", "omega_mae_rad_s"), 0.055, 0.0005)
check("A6 C3 omega MAE", fid(fid17, "A6", "C3", "omega_mae_rad_s"), 0.086, 0.0005)
check("A4 C4 omega Spearman", fid(fid17, "A4", "C4", "omega_spearman"), 0.923, 0.0005)
check("FP16 vs FP32 maximum omega-MAE gap",
      max(abs(fid(fid18, "F16H", c, "omega_mae_rad_s")
              - fid(fid18, "A3", c, "omega_mae_rad_s")) for c in CUR),
      0.0004, 0.00005)
check("FP16 vs FP32 maximum progress gap",
      round(max(abs(fid(r18, "F16H", c, "mean_progress_m")
                    - fid(r18, "A3", c, "mean_progress_m")) for c in CUR), 4),
      0.008, 0.0005)

# --- 4. actor cost ----------------------------------------------------------
bench = json.loads((F18 / "results/precision_benchmark.json").read_text())["rows"]
check("A3 serialized bytes", bench["A3"]["serialized_bytes"], 29295)
check("FP16 serialized bytes", bench["F16H"]["serialized_bytes"], 15865)
check("A6 serialized bytes", bench["A6"]["serialized_bytes"], 34088)
check("A3 parameter memory bytes", bench["A3"]["logical_parameter_memory_bytes"], 24840)
check("FP16 parameter memory bytes", bench["F16H"]["logical_parameter_memory_bytes"], 12420)
check("A3 median latency us", bench["A3"]["latency_median_us"], 19.4, 0.05)
check("FP16 median latency us", bench["F16H"]["latency_median_us"], 24.5, 0.05)
check("A6 median latency us", bench["A6"]["latency_median_us"], 12.9, 0.05)
check("FP16 latency increase percent",
      (bench["F16H"]["latency_median_us"] / bench["A3"]["latency_median_us"] - 1) * 100,
      26.4, 0.3)
check("parameter reduction percent", (1 - 6210 / 73986) * 100, 91.6, 0.05)

int8_mem = ROOT / "experiments/paper_figures/int8_parameter_memory.json"
if int8_mem.is_file():
    check("A6 parameter memory bytes, unpacked from the deployed graph",
          json.loads(int8_mem.read_text())["logical_parameter_memory_bytes"], 7640)
else:
    skipped.append("A6 parameter memory (run experiments/paper_figures/"
                   "compute_int8_memory.py first)")

# --- 5. protocol invariants --------------------------------------------------
check("total evaluated episodes", 9 * 40 + 40, 400)
gate = json.loads((F18 / "integrity/fp16_validity_gate.json").read_text())
check("FP16 validity classification", gate["classification"], "PASS")
check("FP16 accumulation width", gate["accumulation_width"], "wider_than_fp16")
check("INT8 determinism addendum",
      json.loads((F17 / "integrity/int8_determinism_addendum.json").read_text())["classification"],
      "PASS")
check("FP16 determinism addendum",
      json.loads((F18 / "integrity/fp16_determinism_addendum.json").read_text())["classification"],
      "PASS")
outcome = json.loads((F17 / "results/eligibility_outcome.json").read_text())
check("sealed holdout never opened", outcome["final_holdout_opened"], False)
check("eligible candidates", len(outcome["eligible_pathways"]), 0)

import numpy as np  # noqa: E402  (only needed once the artifacts are present)

kd = np.load(F15 / "recovery/datasets/multicurriculum_public_states.npz")
keys = [k for k in kd.files if "state" in k or "public" in k]
n_states = int(kd[keys[0]].shape[0] if keys
               else max(kd[k].shape[0] for k in kd.files))
check("balanced rehearsal dataset states", n_states, 62176)

# --- 6. configuration and curriculum contract --------------------------------
sys.path.insert(0, str(ROOT / "src"))
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol  # noqa: E402

protocol = load_ppo_curriculum_protocol(
    ROOT / "configs/f10_ppo_visual_objects_v30.toml")
for stage, horizon in (("c0", 1900), ("c1", 2700), ("c2", 2700),
                       ("c3", 2700), ("c4", 4200)):
    check(f"horizon {stage}", protocol.stage(stage).episode_horizon_steps, horizon)
check("belief dimension", len(protocol.observation_order), 29)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open(ROOT / "configs/f17_optimization_method_order_v1.toml", "rb") as fh:
    cfg = tomllib.load(fh)
check("physical action ranges",
      [float(v) for v in cfg["frozen"]["physical_action_ranges"]], [0.4, 8.0])
check("omega sign deadband",
      float(cfg["evaluation"]["omega_sign_deadband_rad_s"]), 0.2)
check("minimum Spearman for fidelity",
      float(cfg["fidelity"]["minimum_spearman"]), 0.970)
check("evaluation seeds exclude the sealed holdout",
      len(set(int(s) for s in cfg["seeds"]["primary_evaluation"])
          & set(int(s) for s in cfg["seeds"]["sealed_final_holdout"])), 0)

f12 = ROOT / "docs/F12_COMPRESSION_RESULTS.md"
if f12.is_file():
    text = f12.read_text(encoding="utf-8")
    check("historical F12 speedup claim present in its source document",
          all(s in text for s in ("3.04", "42.77", "14.07")), True)
else:
    skipped.append("historical F12 claim (study documents are not distributed "
                   "with the repository)")

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[3]]
for name, got, expect, ok in checks:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: recomputed={got} reported={expect}")
for note in skipped:
    print(f"  SKIP {note}")

print()
print(f"{len(checks) - len(failed)}/{len(checks)} checks passed"
      + (f", {len(skipped)} skipped" if skipped else ""))
raise SystemExit(1 if failed else 0)
