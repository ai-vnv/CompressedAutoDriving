#!/usr/bin/env python3
"""Confound check for the phase-specific Spearman drop.

Spearman rank correlation degrades easily when the underlying signal has little
variance: if omega is nearly constant during a phase, even a tiny absolute error
reshuffles ranks. A low Spearman would then be an artefact of low signal variance
rather than evidence of a control-relevant mapping change.

This script reports, per phase, the reference omega spread alongside the error, plus a
normalised error (MAE divided by reference standard deviation). It writes nothing into
the frozen diagnostics; it is an interpretation guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from duckie_pomdp.optimization.compression_metrics import actor_physical_predictions  # noqa: E402
from run_f15_cross_curriculum_recovery import artifact_root, load_actor, load_config  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from run_f16_stop_phase_diagnostics import sub_phase  # noqa: E402

CONFIG = ROOT / "configs/f16_sequence_int8_recovery_v1.toml"
F15 = ROOT / "artifacts/f15_cross_curriculum_recovery_v1"
PHASES = ["NOMINAL", "APPROACH", "DECELERATION", "STOP_HOLD", "RESTART"]


def main() -> None:
    config = load_config(CONFIG)
    root = artifact_root(config, CONFIG)
    actors = {
        "FP32": load_actor({"model_path": str(F15 / "recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt"),
                            "int8": False, "hidden_sizes": [64, 64]}),
        "PTQ": load_actor({"model_path": str(F15 / "recovery/ptq/w64/actor_int8.pt"),
                           "int8": True, "hidden_sizes": [64, 64]}),
    }

    print(f"{'cur':<5}{'phase':<14}{'n':>7}{'ref_om_std':>12}{'om_MAE':>10}"
          f"{'MAE/std':>10}{'ref_om_iqr':>12}{'spearman':>10}")
    summary = []
    for curriculum in ("c3", "c4"):
        chunks_obs, chunks_phys, chunks_official = [], [], []
        for trace in sorted((F15 / "telemetry/selection_ptq_w64").rglob("trace.npz")):
            if trace.parent.parent.name != curriculum:
                continue
            with np.load(trace, allow_pickle=False) as a:
                chunks_obs.append(a["public_normalized_29d"])
                chunks_phys.append(a["public_physical_29d"])
                chunks_official.append(np.asarray([str(v) for v in a["public_phase"]]))
                names = [str(v) for v in a["feature_names"]]
        vel = names.index("actual_linear_velocity_mps")
        sub = np.concatenate([sub_phase(o, p[:, vel]) for o, p in zip(chunks_official, chunks_phys)])
        obs = np.concatenate(chunks_obs)
        ref = actor_physical_predictions(actors["FP32"], obs)[:, 1]
        can = actor_physical_predictions(actors["PTQ"], obs)[:, 1]

        for phase in PHASES:
            sel = sub == phase
            if not sel.any():
                continue
            r, c = ref[sel], can[sel]
            std = float(np.std(r))
            mae = float(np.mean(np.abs(r - c)))
            iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
            order_r = np.argsort(np.argsort(r))
            order_c = np.argsort(np.argsort(c))
            sp = float(np.corrcoef(order_r, order_c)[0, 1]) if len(r) > 1 else float("nan")
            print(f"{curriculum:<5}{phase:<14}{sel.sum():>7}{std:>12.4f}{mae:>10.4f}"
                  f"{mae/std if std else float('nan'):>10.3f}{iqr:>12.4f}{sp:>10.4f}")
            summary.append({"curriculum": curriculum, "phase": phase, "n": int(sel.sum()),
                            "ref_omega_std": std, "omega_mae": mae,
                            "mae_over_std": mae / std if std else None,
                            "ref_omega_iqr": iqr, "spearman": sp})

    print()
    print("Interpretation guard:")
    print("  If a phase shows LOW ref_om_std together with LOW spearman, the rank drop is")
    print("  plausibly a low-variance artefact and must NOT be read as a control-relevant")
    print("  divergence. A control-relevant divergence needs the error to be large RELATIVE")
    print("  to the reference spread (MAE/std), not merely a reshuffled rank order.")
    (root / "stop_phase").mkdir(parents=True, exist_ok=True)
    (root / "stop_phase" / "spearman_confound_check.json").write_text(
        json.dumps({"rows": summary}, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
