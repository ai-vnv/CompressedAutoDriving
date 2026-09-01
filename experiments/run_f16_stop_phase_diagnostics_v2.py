#!/usr/bin/env python3
"""F16 stop-phase INT8 divergence diagnostics, v2.

Corrections over v1, all methodological:

1. **Tie-aware Spearman.** v1 used ordinal ranks, which are ill-defined under heavy ties;
   two sort kinds gave different answers on the same data. v2 uses average ranks, and
   still demotes Spearman to a secondary descriptive metric with a low-dynamic-range
   warning, because a correct Spearman is still uninformative when the reference signal
   is nearly constant.

2. **Two normalisations.** R_sigma = MAE/SD and the outlier-robust R_iqr = MAE/IQR.
   A phase is flagged LOW_DYNAMIC_RANGE when the error exceeds the entire interquartile
   spread of the reference (IQR < MAE) — a self-referential test with no tuned constant.

3. **Episode-level aggregation.** Timesteps within an episode are strongly
   autocorrelated and are not independent replicates. Metrics are computed per episode,
   then summarised across episodes/seeds as median with interquartile range.

Primary metrics are absolute error, the two normalisations, and the temporal command
deltas. Offline, CPU-only, zero simulator episodes. Diagnostic localization, not causal
explanation.
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
    artifact_root, load_actor, load_config, provenance, read_json, write_json,
)

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
F15 = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
STOP_OFFICIAL = {"stop_required", "combined_pedestrian_stop"}
PHASES = ["NOMINAL", "APPROACH", "DECELERATION", "STOP_HOLD", "RESTART"]


def sub_phase(official: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Frozen rules. delta_v == 0 resolves to APPROACH (>=0) and STOP_HOLD (<=0)."""
    delta = np.diff(velocity, prepend=velocity[:1])
    out = np.full(len(official), "NOMINAL", dtype="U14")
    in_stop = np.isin(official, list(STOP_OFFICIAL))
    out[in_stop & (delta >= 0)] = "APPROACH"
    out[in_stop & (delta < 0)] = "DECELERATION"
    satisfied = official == "stop_satisfied"
    out[satisfied & (delta <= 0)] = "STOP_HOLD"
    out[satisfied & (delta > 0)] = "RESTART"
    return out


def average_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks: the correct ranking for Spearman under ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop)
        start = stop + 1
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def episode_metrics(reference: np.ndarray, candidate: np.ndarray, deadband: float) -> dict:
    """Metrics for one phase inside ONE episode."""
    rv, ro = reference[:, 0], reference[:, 1]
    cv, co = candidate[:, 0], candidate[:, 1]
    d_rv, d_ro = np.diff(rv), np.diff(ro)
    d_cv, d_co = np.diff(cv), np.diff(co)

    sd = float(np.std(ro))
    iqr = float(np.percentile(ro, 75) - np.percentile(ro, 25))
    mae_omega = float(np.mean(np.abs(ro - co)))
    unique_fraction = float(len(np.unique(np.round(ro, 6))) / len(ro))

    eligible = np.abs(ro) > deadband
    sign_dis = float(np.mean(np.sign(ro[eligible]) != np.sign(co[eligible]))) if eligible.any() else None

    return {
        "state_count": int(len(reference)),
        "v_mae_mps": float(np.mean(np.abs(rv - cv))),
        "omega_mae_rad_s": mae_omega,
        "omega_reference_sd": sd,
        "omega_reference_iqr": iqr,
        "r_sigma": mae_omega / sd if sd > 0 else None,
        "r_iqr": mae_omega / iqr if iqr > 0 else None,
        "low_dynamic_range": bool(iqr < mae_omega),
        "delta_v_mae": float(np.mean(np.abs(d_rv - d_cv))) if len(d_rv) else None,
        "delta_omega_mae": float(np.mean(np.abs(d_ro - d_co))) if len(d_ro) else None,
        "omega_sign_disagreement": sign_dis,
        "reference_unique_value_fraction": unique_fraction,
        "omega_pearson_secondary": _corr(ro, co),
        "omega_spearman_secondary": _corr(average_rank(ro), average_rank(co)),
    }


def summarise(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Median and IQR ACROSS EPISODES — never across timesteps."""
    numeric = [
        "v_mae_mps", "omega_mae_rad_s", "omega_reference_sd", "omega_reference_iqr",
        "r_sigma", "r_iqr", "delta_v_mae", "delta_omega_mae", "omega_sign_disagreement",
        "reference_unique_value_fraction", "omega_pearson_secondary", "omega_spearman_secondary",
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(keys, key))
        record["episodes"] = len(members)
        record["total_states"] = int(sum(m["state_count"] for m in members))
        record["low_dynamic_range_episodes"] = int(sum(bool(m["low_dynamic_range"]) for m in members))
        for name in numeric:
            values = [m[name] for m in members if m.get(name) is not None]
            if values:
                record[f"{name}_median"] = float(np.median(values))
                record[f"{name}_q25"] = float(np.percentile(values, 25))
                record[f"{name}_q75"] = float(np.percentile(values, 75))
            else:
                record[f"{name}_median"] = record[f"{name}_q25"] = record[f"{name}_q75"] = None
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name}: {len(rows)} rows")


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    definition = read_json(root / "protocol/stop_phase_definition.json")
    if definition["classification"] != "FROZEN":
        raise RuntimeError("stop-phase definition is not frozen")
    deadband = float(config["evaluation"]["omega_sign_deadband_rad_s"])
    out_dir = root / "stop_phase_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    actors = {
        "FP32": load_actor({"model_path": str(F15 / "recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt"),
                            "int8": False, "hidden_sizes": [64, 64]}),
        "PTQ": load_actor({"model_path": str(F15 / "recovery/ptq/w64/actor_int8.pt"),
                           "int8": True, "hidden_sizes": [64, 64]}),
        "QAT": load_actor({"model_path": str(F15 / "recovery/qat/w64/actor_int8.pt"),
                           "int8": True, "hidden_sizes": [64, 64]}),
    }

    # ---- Population C, per episode: same-state cross-policy on INT8-visited states ----
    per_episode: list[dict] = []
    occupancy: list[dict] = []
    for visited_by, split in (("PTQ", "selection_ptq_w64"), ("QAT", "selection_qat_w64")):
        for trace in sorted((F15 / "telemetry" / split).rglob("trace.npz")):
            curriculum = trace.parent.parent.name
            if curriculum not in ("c3", "c4"):
                continue
            seed = int(trace.parent.name.split("_")[-1])
            with np.load(trace, allow_pickle=False) as archive:
                obs = archive["public_normalized_29d"]
                phys = archive["public_physical_29d"]
                official = np.asarray([str(v) for v in archive["public_phase"]])
                names = [str(v) for v in archive["feature_names"]]
            vel = names.index("actual_linear_velocity_mps")
            stop = names.index("stop_line_distance_m")
            sub = sub_phase(official, phys[:, vel])
            preds = {name: actor_physical_predictions(actor, obs) for name, actor in actors.items()}

            for phase in PHASES:
                sel = sub == phase
                if sel.sum() < 3:
                    continue
                occupancy.append({
                    "states_visited_by": visited_by, "curriculum": curriculum, "seed": seed,
                    "phase": phase, "state_count": int(sel.sum()),
                    "state_fraction": float(sel.mean()), "episode_length": int(len(sub)),
                    "mean_stop_line_distance_m": float(np.mean(phys[sel, stop])),
                    "mean_actual_velocity_mps": float(np.mean(phys[sel, vel])),
                })
                for candidate in ("PTQ", "QAT"):
                    per_episode.append({
                        "states_visited_by": visited_by, "curriculum": curriculum,
                        "candidate": candidate, "phase": phase, "seed": seed,
                        **episode_metrics(preds["FP32"][sel], preds[candidate][sel], deadband),
                    })

    write_csv(out_dir / "stop_phase_per_episode.csv", per_episode)
    write_csv(out_dir / "stop_phase_occupancy.csv", occupancy)
    summary = summarise(per_episode, ("states_visited_by", "curriculum", "candidate", "phase"))
    write_csv(out_dir / "stop_phase_summary_by_episode.csv", summary)

    write_json(out_dir / "stop_phase_v2_manifest.json", {
        **provenance(config, CONFIG),
        "supersedes": "stop_phase/ (v1 used ordinal ranks and treated timesteps as replicates)",
        "corrections": [
            "tie-aware average-rank Spearman, demoted to secondary",
            "R_sigma = MAE/SD and robust R_iqr = MAE/IQR",
            "LOW_DYNAMIC_RANGE flag when IQR < MAE (no tuned constant)",
            "per-episode metrics summarised across episodes/seeds, not across timesteps",
        ],
        "independent_replicate_unit": "episode (seed), not timestep",
        "primary_metrics": ["omega_mae_rad_s", "r_sigma", "r_iqr", "delta_omega_mae", "delta_v_mae"],
        "secondary_metrics": ["omega_pearson_secondary", "omega_spearman_secondary"],
        "simulator_episodes_run": 0,
        "claim_limit": definition["claim_limit"],
    })

    print()
    print("=== Population C: FP32 vs PTQ on PTQ-visited states (median across episodes) ===")
    print(f"{'cur':<5}{'phase':<14}{'eps':>5}{'omMAE':>9}{'refSD':>9}{'refIQR':>9}"
          f"{'R_sig':>8}{'R_iqr':>8}{'dOm':>9}{'LDR':>5}")
    for r in summary:
        if r["candidate"] == "PTQ" and r["states_visited_by"] == "PTQ":
            ri = r["r_iqr_median"]
            print(f"{r['curriculum']:<5}{r['phase']:<14}{r['episodes']:>5}"
                  f"{r['omega_mae_rad_s_median']:>9.4f}{r['omega_reference_sd_median']:>9.4f}"
                  f"{r['omega_reference_iqr_median']:>9.4f}{r['r_sigma_median']:>8.3f}"
                  f"{ri if ri is None else round(ri, 3):>8}"
                  f"{r['delta_omega_mae_median']:>9.4f}"
                  f"{r['low_dynamic_range_episodes']:>3}/{r['episodes']}")


if __name__ == "__main__":
    main()
