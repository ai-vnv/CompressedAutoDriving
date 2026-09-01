# F15 Recovery Report

## Controlled recovery question

F15 first changes only distillation coverage: the Original teacher, historical
survivor indices, architecture, normalized physical-action Smooth-L1 loss, Adam
optimizer family, and training budget remain frozen. Public C0–C4 states receive equal
curriculum mass and equal supported-phase mass within curriculum.

Everything held fixed against the historical F12 procedure: teacher
(`a0_original_actor.pt`, SHA256 `713d26d9…`), the 64×64 survivor indices inherited from
`pruning/p75/actor_pruned_fp32.pt` (SHA256 `6e4ff154…`), the `29→64→64→2` Tanh
architecture, Smooth-L1 on deterministic physical actions normalized by `[0.4, 8.0]`,
Adam, 80 epochs, batch size 512, learning rate 0.001, weight decay 1e-6, and the copied
state-independent `log_std`. No PPO retraining, no reward, no critic target, no
ground-truth action labels.

## Recovery dataset

`artifacts/f15_cross_curriculum_recovery_v1/recovery/datasets/multicurriculum_public_states.npz`
(manifest `dataset_manifest.json`, dataset SHA256 `385e2a3a…`) contains **62,176 rows**
collected with the frozen Original teacher on the recovery-dataset seeds 180101–180108.

| Curriculum | Rows | Share of raw collection | Public phases present |
|---|---:|---:|---|
| C0 | 6,855 | 11.0% | nominal, lane_curve |
| C1 | 6,750 | 10.9% | nominal, lane_curve |
| C2 | 12,871 | 20.7% | nominal, lane_curve, pedestrian_relevant |
| C3 | 16,628 | 26.7% | nominal, lane_curve, stop_required, stop_satisfied |
| C4 | 19,072 | 30.7% | nominal, lane_curve, pedestrian_relevant, stop_required, stop_satisfied |

The raw collection is intrinsically imbalanced — C4 produces 2.8× more frames than C0
simply because its horizon is longer. Training therefore samples with equal curriculum
mass first and equal supported-phase mass within curriculum, so neither C4 nor nominal
driving can dominate by frame count alone.

The dataset contains public 29D inputs and deterministic Original actions only. It does
not contain privileged simulator truth, reward targets, critic targets, or GT actions
(`contains_privileged_truth: false`).

## FP32 multi-curriculum KD

**FP32 CROSS-CURRICULUM RECOVERY = PASS.** The 64×64 student
(`recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`, SHA256 `64c84cd0…`) passed every
frozen C0–C4 behavior, fidelity, and safety gate on the recovery-selection seeds
180201–180208: `eligible: true`, `all_curricula_behavior_pass: true`,
`fidelity.all_curricula_pass: true`, with no failed sub-check.

| Curriculum | Status | Completion (recovered) | Completion (Original) | Progress m (recovered) | Progress m (Original) |
|---|---|---:|---:|---:|---:|
| C0 | PASS | 1.000 | 1.000 | 5.345 | 5.337 |
| C1 | PASS | 0.875 | 0.750 | 7.245 | 6.454 |
| C2 | PASS | 0.875 | 0.875 | 7.906 | 7.897 |
| C3 | PASS | 1.000 | 0.875 | 7.250 | 6.818 |
| C4 | PASS | 1.000 | 1.000 | 7.209 | 7.200 |

The controlled comparison is against the historical A2 actor, which uses the **same
width, same survivor indices, same loss, same optimizer, and same budget**, and differs
only in that its rehearsal states were drawn from C4 development seeds. A2 records
completion **0.000** in C0, C1, and C2. The recovered student records 1.000, 0.875, and
0.875. Same-state action fidelity moved with it: C0 omega MAE fell from 0.31409 rad/s
(A2) and 0.26946 rad/s (A7) to **0.03172 rad/s**, with omega sign disagreement 0.000 and
Pearson 0.99962 against a gate of 0.980.

Under this protocol, curriculum-balanced public rehearsal coverage is sufficient to
restore C0–C2 retention at the smallest historical width, without changing capacity,
loss, optimizer, or training budget. The frozen width rule
(`run_larger_width_only_if_64_fails`) was therefore **not triggered**, and widths 96,
128, and 192 were not trained.

## Quantization after recovery

**PTQ recovery path: FAIL.** PTQ used a curriculum-balanced C0–C4 calibration set of
16,384 rows drawn from the same recovery dataset (`recovery/ptq/w64/conversion.json`),
with the recovered FP32 weights frozen. The converted INT8 actor
(SHA256 `7ac05518…`) is **not eligible**.

| Curriculum | FP32 recovered | + PTQ | + QAT+KD |
|---|---|---|---|
| C0 | PASS | PASS | PASS |
| C1 | PASS | PASS | PASS |
| C2 | PASS | PASS | PASS |
| C3 | **PASS** | **FAIL** | **FAIL** |
| C4 | **PASS** | **FAIL** | **FAIL** |

PTQ completion collapsed on C3 from 1.000 to **0.375** and mean progress from 7.250 m to
**3.836 m**; C4 completion fell to 0.875. The failing fidelity checks were **Pearson and
Spearman**, not MAE — C0 omega MAE was 0.16179 rad/s, inside the 0.200 gate, while
Pearson was 0.97780 against a 0.980 gate. The distortion is in the rank ordering of the
steering command rather than in its magnitude.

**QAT was therefore required, and was run.** Multi-curriculum fake-quantized QAT+KD used
the same frozen teacher, the same balanced C0–C4 rehearsal, and the same x86 static INT8
backend (SHA256 `c943e34f…`). It measurably improved fidelity over PTQ — C2 and C3 both
passed the full fidelity gate, C3 omega MAE fell from 0.08639 to 0.05796 rad/s, and omega
sign disagreement reached 0.000 on C3 and C4 — but it did **not** restore behavior:

- **C3**: completion 0.500, and the absolute gate `maximum_stop_violation_rate` failed.
  This is a safety-relevant regression on the stop task, not merely reduced performance.
- **C4**: completion 0.750, failing `completion_rate`, `mean_progress_m`,
  `minimum_clearance`, `restart_rate`, and `stop_violation_rate` against Original.

Both INT8 paths fail in the same place, and it is the mirror image of the historical
failure: C0–C2 survive quantization while C3–C4 break. The curricula that break are the
ones requiring precise modulation — decelerating to a stop line and restarting.

Stated conservatively: **under the tested x86 static quantization procedure, with
curriculum-balanced calibration, at width 64, INT8 conversion introduced a C3/C4
retention failure that multi-curriculum QAT+KD did not repair.** This is a statement
about the tested procedure at this width. It is not a claim that quantization
universally destroys curriculum retention.

## Progressive pruning

**Not run.** The frozen rule
(`run_progressive_pruning_only_if_direct_recovery_is_insufficient`) reserves progressive
prune–distill for failure of direct target-width recovery. Direct 256→64 pruning plus
multi-curriculum KD **succeeded** in FP32. The observed loss is associated with the INT8
conversion, not with pruning, so progressive pruning addresses a different mechanism.
Whether optimization order matters therefore remains **UNRESOLVED** in F15; no ordering
claim is made.

## Frozen candidate

**No final candidate was frozen, and the once-only holdout was not opened.**

`freeze-candidate` requires a deployable INT8 actor with `eligible: true`. Zero INT8
candidates qualified:

| Candidate | Width | INT8 | Behavior all-pass | Fidelity all-pass | Eligible |
|---|---:|---|---|---|---|
| Recovered + Multi-Curriculum KD | 64 | no | **true** | **true** | **true** |
| Recovered + PTQ | 64 | yes | false | false | false |
| Recovered + Multi-Curriculum QAT+KD | 64 | yes | false | false | false |

The only eligible actor is FP32, and the frozen protocol requires the final candidate to
be INT8. Substituting a different model after seeing these results is forbidden. The
holdout seeds **180301–180308 remain unopened** and are recorded as such in
`recovery/recovery_decision.json` (`final_holdout_opened: false`). They remain available
for a future preregistered attempt.

Machine-readable outcome: `recovery/recovery_decision.json`
(`outcome: STOPPED_WITHOUT_ELIGIBLE_INT8_CANDIDATE`) and `recovery/recovery_experiments.csv`.

## Efficiency cost of recovery

Actor-only, one CPU thread, batch 1, 1,000 warmups, 10,000 timed iterations, five
repeats — the frozen `[benchmark]` parameters
(`final/efficiency_summary.json`).

| Actor | Params | Bytes | Median latency | Params vs Original | Bytes vs Original | Actor-only speedup |
|---|---:|---:|---:|---:|---:|---:|
| Original Policy | 73,986 | 299,667 | 40.428 µs | — | — | 1.00× |
| **Recovered 64×64 + Multi-Curriculum KD (FP32)** | **6,210** | **29,295** | **35.840 µs** | **−91.61%** | **−90.22%** | **1.13×** |
| Recovered 64×64 + PTQ (INT8) | 6,210 | 34,088 | 15.984 µs | −91.61% | −88.62% | 2.53× |
| Recovered 64×64 + QAT+KD (INT8) | 6,210 | 34,152 | 15.383 µs | −91.61% | −88.60% | 2.63× |
| Final INT8 Policy (historical A7) | 6,210 | 36,880 | 15.313 µs | −91.61% | −87.69% | 2.64× |

The cost of recovery is visible here and should not be understated. The only actor that
preserves all five curricula is FP32, so it keeps the **91.61% parameter reduction** and
**90.22% file-size reduction** but delivers only a **1.13× actor-only speedup**. The
2.5–2.6× speedups all belong to INT8 actors that fail C3/C4. F15 therefore did not
achieve compression *and* full retention simultaneously in a deployable INT8 form; it
achieved parameter and size compression with full retention in FP32.

All speed claims are actor-only, one-thread CPU measurements. Perception
(MobileNet/YOLO/belief) is unchanged and remains the dominant deployment cost; no
end-to-end visuomotor speedup is claimed.
