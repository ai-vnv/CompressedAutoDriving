#!/usr/bin/env python3
"""F16 stop-phase INT8 divergence diagnostics.

Three populations, deliberately kept separate:

  A. teacher-visited states  — same-state mapping change on Original-distribution states
  B. self-visited states     — which phases each actor actually occupies (occupancy and
                               behaviour only; never step-aligned across actors)
  C. same-state cross-policy — the exact 29D rows recorded along the INT8 trajectories,
                               replayed offline through FP32/PTQ/QAT

C is the discriminating one: it separates "the mapping already differs on this very
state" from "the trajectory drifted off-distribution and failure amplified".

Offline and CPU-only. No simulator episode is run. Diagnostic localization, not causal
explanation — see docs/F16_QUANTIZATION_SCOPE_LIMITATION.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.compression_metrics import actor_physical_predictions  # noqa: E402
from run_f15_cross_curriculum_recovery import (  # noqa: E402
    artifact_root,
    load_actor,
    load_config,
    provenance,
    read_json,
    write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
F15 = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
STOP_OFFICIAL = {"stop_required", "combined_pedestrian_stop"}


def sub_phase(official: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Apply the frozen sub-phase rules. Only the SIGN of delta_v is used."""
    delta = np.diff(velocity, prepend=velocity[:1])
    out = np.full(len(official), "NOMINAL", dtype="U14")
    in_stop = np.isin(official, list(STOP_OFFICIAL))
    out[in_stop & (delta >= 0)] = "APPROACH"
    out[in_stop & (delta < 0)] = "DECELERATION"
    satisfied = official == "stop_satisfied"
    out[satisfied & (delta <= 0)] = "STOP_HOLD"
    out[satisfied & (delta > 0)] = "RESTART"
    return out


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(reference: np.ndarray, candidate: np.ndarray, deadband: float) -> dict:
    """Action-mapping and temporal-control metrics for one phase bucket."""
    if len(reference) == 0:
        return {}
    rv, ro = reference[:, 0], reference[:, 1]
    cv, co = candidate[:, 0], candidate[:, 1]
    d_rv, d_ro = np.diff(rv, prepend=rv[:1]), np.diff(ro, prepend=ro[:1])
    d_cv, d_co = np.diff(cv, prepend=cv[:1]), np.diff(co, prepend=co[:1])
    eligible = np.abs(ro) > deadband
    sign_dis = (
        float(np.mean(np.sign(ro[eligible]) != np.sign(co[eligible]))) if eligible.any() else 0.0
    )
    sat_ref = float(np.mean((np.abs(rv - 0.4) < 1e-6) | (np.abs(rv) < 1e-6)))
    sat_can = float(np.mean((np.abs(cv - 0.4) < 1e-6) | (np.abs(cv) < 1e-6)))
    return {
        "state_count": int(len(reference)),
        "v_mae_mps": float(np.mean(np.abs(rv - cv))),
        "omega_mae_rad_s": float(np.mean(np.abs(ro - co))),
        "omega_pearson": _corr(ro, co),
        "omega_spearman": _corr(_rank(ro), _rank(co)),
        "omega_sign_disagreement": sign_dis,
        "delta_v_mae": float(np.mean(np.abs(d_rv - d_cv))),
        "delta_omega_mae": float(np.mean(np.abs(d_ro - d_co))),
        "saturation_disagreement": abs(sat_ref - sat_can),
    }


def load_actors(config) -> dict:
    entries = {
        "FP32": (F15 / "recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt", False),
        "PTQ": (F15 / "recovery/ptq/w64/actor_int8.pt", True),
        "QAT": (F15 / "recovery/qat/w64/actor_int8.pt", True),
    }
    out = {}
    for name, (path, int8) in entries.items():
        out[name] = load_actor({"model_path": str(path), "int8": int8, "hidden_sizes": [64, 64]})
    return out


def episode_traces(split: str) -> list[Path]:
    return sorted((F15 / "telemetry" / split).rglob("trace.npz"))


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    definition = read_json(root / "protocol/stop_phase_definition.json")
    if definition["classification"] != "FROZEN":
        raise RuntimeError("stop-phase definition is not frozen")
    deadband = float(config["evaluation"]["omega_sign_deadband_rad_s"])
    out_dir = root / "stop_phase"
    out_dir.mkdir(parents=True, exist_ok=True)
    actors = load_actors(config)
    PHASES = ["NOMINAL", "APPROACH", "DECELERATION", "STOP_HOLD", "RESTART"]

    # ---------------- Population A: teacher-visited states ----------------
    with np.load(F15 / "recovery/datasets/multicurriculum_public_states.npz", allow_pickle=False) as a:
        ds = {k: a[k] for k in a.files}
    names = [str(v) for v in ds["feature_names"]]
    vel_index = names.index("actual_linear_velocity_mps")
    stop_index = names.index("stop_line_distance_m")

    rows_a = []
    for curriculum in ("c3", "c4"):
        mask = ds["curriculum"] == curriculum
        obs = ds["observation"][mask]
        phys = ds["physical_observation"][mask]
        official = ds["public_phase"][mask]
        seeds, steps = ds["seed"][mask], ds["step"][mask]
        order = np.lexsort((steps, seeds))
        obs, phys, official = obs[order], phys[order], official[order]
        sub = sub_phase(official, phys[:, vel_index])
        preds = {name: actor_physical_predictions(actor, obs) for name, actor in actors.items()}
        for phase in PHASES:
            sel = sub == phase
            if not sel.any():
                continue
            for candidate in ("PTQ", "QAT"):
                metrics = compare(preds["FP32"][sel], preds[candidate][sel], deadband)
                rows_a.append({
                    "population": "A_teacher_states", "curriculum": curriculum,
                    "reference": "FP32", "candidate": candidate, "phase": phase,
                    "state_fraction": float(sel.mean()),
                    "mean_stop_line_distance_m": float(np.mean(phys[sel, stop_index])),
                    "mean_actual_velocity_mps": float(np.mean(phys[sel, vel_index])),
                    **metrics,
                })

    # ---------------- Population B: self-visited occupancy ----------------
    rows_b = []
    for label, split in (("FP32", "selection_fp32_w64"), ("PTQ", "selection_ptq_w64"), ("QAT", "selection_qat_w64")):
        for trace in episode_traces(split):
            curriculum = trace.parent.parent.name
            if curriculum not in ("c3", "c4"):
                continue
            with np.load(trace, allow_pickle=False) as archive:
                phys = archive["public_physical_29d"]
                official = np.asarray([str(v) for v in archive["public_phase"]])
                action = archive["physical_action"]
            sub = sub_phase(official, phys[:, vel_index])
            for phase in PHASES:
                sel = sub == phase
                if not sel.any():
                    continue
                rows_b.append({
                    "population": "B_self_visited", "actor": label, "curriculum": curriculum,
                    "seed": int(trace.parent.name.split("_")[-1]), "phase": phase,
                    "state_count": int(sel.sum()), "state_fraction": float(sel.mean()),
                    "episode_length": int(len(sub)),
                    "mean_v_cmd_mps": float(np.mean(action[sel, 0])),
                    "mean_abs_omega_cmd_rad_s": float(np.mean(np.abs(action[sel, 1]))),
                    "mean_actual_velocity_mps": float(np.mean(phys[sel, vel_index])),
                    "mean_stop_line_distance_m": float(np.mean(phys[sel, stop_index])),
                })

    # ------- Population C: same-state cross-policy on INT8-visited states -------
    rows_c = []
    for visited_by, split in (("PTQ", "selection_ptq_w64"), ("QAT", "selection_qat_w64")):
        for curriculum in ("c3", "c4"):
            chunks_obs, chunks_phys, chunks_official = [], [], []
            for trace in episode_traces(split):
                if trace.parent.parent.name != curriculum:
                    continue
                with np.load(trace, allow_pickle=False) as archive:
                    chunks_obs.append(archive["public_normalized_29d"])
                    chunks_phys.append(archive["public_physical_29d"])
                    chunks_official.append(np.asarray([str(v) for v in archive["public_phase"]]))
            if not chunks_obs:
                continue
            # Sub-phase is computed per episode so delta_v never crosses an episode boundary.
            sub = np.concatenate([sub_phase(o, p[:, vel_index]) for o, p in zip(chunks_official, chunks_phys)])
            obs = np.concatenate(chunks_obs)
            phys = np.concatenate(chunks_phys)
            preds = {name: actor_physical_predictions(actor, obs) for name, actor in actors.items()}
            for phase in PHASES:
                sel = sub == phase
                if not sel.any():
                    continue
                for candidate in ("PTQ", "QAT"):
                    metrics = compare(preds["FP32"][sel], preds[candidate][sel], deadband)
                    rows_c.append({
                        "population": "C_same_state_cross_policy",
                        "states_visited_by": visited_by, "curriculum": curriculum,
                        "reference": "FP32", "candidate": candidate, "phase": phase,
                        "state_fraction": float(sel.mean()),
                        "mean_stop_line_distance_m": float(np.mean(phys[sel, stop_index])),
                        "mean_actual_velocity_mps": float(np.mean(phys[sel, vel_index])),
                        **metrics,
                    })

    for name, rows in (("stop_phase_teacher_states.csv", rows_a),
                       ("stop_phase_self_visited.csv", rows_b),
                       ("stop_phase_same_state_cross_policy.csv", rows_c)):
        path = out_dir / name
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {name}: {len(rows)} rows")

    write_json(out_dir / "stop_phase_diagnostics_manifest.json", {
        **provenance(config, CONFIG),
        "definition_sha256": None,
        "populations": {"A": len(rows_a), "B": len(rows_b), "C": len(rows_c)},
        "reference_policy": "recovered FP32 64x64 (multi-curriculum KD)",
        "offline_only": True,
        "simulator_episodes_run": 0,
        "claim_limit": definition["claim_limit"],
    })

    print("\n=== C (same-state, on INT8-visited states) — FP32 vs PTQ ===")
    print(f"{'cur':<5}{'phase':<14}{'n':>7}{'om_MAE':>10}{'pearson':>10}{'spearman':>10}{'sign_dis':>10}")
    for r in rows_c:
        if r["candidate"] == "PTQ" and r["states_visited_by"] == "PTQ":
            print(f"{r['curriculum']:<5}{r['phase']:<14}{r['state_count']:>7}"
                  f"{r['omega_mae_rad_s']:>10.4f}{r['omega_pearson']:>10.4f}"
                  f"{r['omega_spearman']:>10.4f}{r['omega_sign_disagreement']:>10.4f}")


if __name__ == "__main__":
    main()
