# F15 Failure Localization Report

## Evidence boundary

This report uses only the new paired localization seeds 180001–180008. Historical
F10–F14 artifacts remain immutable. All A0–A7 and pruning-frontier checkpoints were
SHA256-verified before simulation. No attribution or explainability method is part of
F15.

The analysis separates same-state action fidelity from closed-loop behavior. A
curriculum is `UNRESOLVED`, rather than a compression failure, when Original Policy
does not pass its absolute gate on the same seeds.

## Cross-stage competence

The paired eight-seed matrix is:

| Optimization stage | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Original Policy | REFERENCE | REFERENCE | REFERENCE | REFERENCE | REFERENCE |
| Pruning Only | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + Knowledge Distillation | FAIL | FAIL | FAIL | PASS | PASS |
| Post-Training Quantization (PTQ) | PASS | PASS | PASS | FAIL | PASS |
| QAT + Distillation | PASS | FAIL | FAIL | PASS | PASS |
| Pruning + PTQ | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + Distillation + PTQ | FAIL | FAIL | FAIL | PASS | FAIL |
| Final INT8 Policy | FAIL | FAIL | FAIL | PASS | PASS |

Original completion was 8/8 in C0, C2, C3, and C4 and 7/8 in C1; it passed every
applicable absolute gate. Pruning Only reduced completion to 4/8 in C0/C1 and 0/8 in
C2–C4. Historical pruning+KD recovered completion to 8/8 in C3/C4 but remained 0/8 in
C0–C2. Thus the historical C0–C2 loss replicated on eight new seeds and was not an
artifact of the two-seed F12 retention warning.

Machine-readable evidence:

- `artifacts/f15_cross_curriculum_recovery_v1/localization/cross_curriculum_results.csv`
- `artifacts/f15_cross_curriculum_recovery_v1/localization/matrix_results.json`

## First collapse by curriculum

For **C0, C1, C2, C3, and C4**, the first observed PASS→FAIL transition was
**Original Policy → Pruning Only**. The later trajectory differed:

- C0–C2: historical C4-focused KD, PTQ, and final QAT/KD did not restore retention on
  the actual final 64×64 path;
- C3–C4: KD after pruning restored closed-loop retention;
- C4: PTQ after pruning+KD produced a new failure in A6, and the final QAT/KD branch
  restored it in A7;
- C3: PTQ-only failed, while the unpruned QAT+distillation variant restored retention.

These are operation-associated transitions under the frozen branches, not claims that
a particular removed neuron caused a later event.

The final historical construction path is represented as Original Policy → Pruning
Only → Pruning + Knowledge Distillation → Final INT8 Policy. PTQ/QAT alternatives are
reported as parallel diagnostics and are not presented as a fictitious linear history.

## Same-state action evidence

Direct pruning caused the largest same-state errors across curricula. For example, C0
Pruning Only had v/omega MAE 0.11880 m/s and 1.04927 rad/s. Historical KD reduced those
to 0.00554 m/s and 0.31409 rad/s, but C0 still failed the full fidelity gate and all
eight closed-loop episodes failed. In C4, the same KD actor reached 0.00139 m/s and
0.02310 rad/s and passed both fidelity and behavior.

The Final INT8 Policy similarly passed C4 fidelity (0.00235 m/s and 0.03958 rad/s MAE)
but failed C0–C3 fidelity; C0 omega MAE was 0.26946 rad/s. These data show that fidelity
is strongly curriculum-dependent and that a global C4 result cannot stand in for
cross-curriculum equivalence.

Across non-reference actors, the frozen full fidelity gate passed only for: Pruning+KD
in C4; PTQ in C2; QAT+distillation in C4; Pruning+KD+PTQ in C4; and Final INT8 in C4.

The exact normalized public 29D rows from Original trajectories were passed offline to
all actors. These metrics therefore diagnose actor-mapping drift before trajectory
feedback.

## Pruning-width evidence

No historical width preserved all five curricula. Pruning-only 192×192 and 96×96
passed C0/C1 but failed C2–C4; 128×128 passed C0 only; 64×64 failed all five. All
historically distilled widths (192, 128, 96, 64) failed C0–C2. They passed C3, and the
64/96/128 distilled actors passed C4, whereas PD192 failed C4 on this seed block.

The non-monotonic cells and the uniform C0–C2 failure after historical KD do not support
a simple universal width threshold. They motivate the controlled rehearsal-coverage
test before increasing capacity.

Width comparisons support association under the historical pruning/KD procedure; they
do not establish a neuron-level causal capacity threshold.

## Objective failure traces

The frozen registry contains 50 objectively selected failing cells. All 50 now have a
PNG contact sheet, GIF, MP4 timeline, CSV window, and hash-bound JSON generated directly
from the immutable primary telemetry under
`artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/`. The aggregate manifest
is `failure_telemetry/failure_telemetry_manifest.json`.

The active simulator could not reproduce the old closed-loop RGB episodes within the
frozen visual-replay criteria: in the first two audited attempts, recorded-action replay
terminated before the frozen failure window. Those media are explicitly quarantined as
`UNRESOLVED` under `failure_traces/A1/c0/seed_180002/unresolved/` and
`failure_traces/A1/c1/seed_180002/unresolved/`; they are not evidence. Because the
primary telemetry did not store RGB, the valid failure animation shows recorded
progress, physical actions, and lane error rather than inventing camera frames.

For each failing cell, the selected trace remains the first objective failure event in
the lowest failing seed. The Original-versus-compressed telemetry comparison is a
same-seed recorded comparison, not a causal paired trajectory. A separate historical A7
C4 simulator video remains available as a qualitative example only at
`artifacts/f12_belief_ppo_compression_v1/final/a7_c4_front_bev.mp4`.

## Frozen localization decision

**Localization classification: FROZEN / complete.** Every A0–A7 cell contains eight
paired episodes; every unique pruning-frontier cell contains eight paired episodes.
The decision records pruning as the first observed collapse stage for all curricula,
historical KD recovery for C3/C4 only, quantization-associated branch changes, and
multi-curriculum rehearsal coverage as the next controlled hypothesis. Optimization
order remains unresolved unless direct multi-curriculum recovery proves insufficient.

Recovery was not started until
`artifacts/f15_cross_curriculum_recovery_v1/localization/failure_localization_decision.json`
had been written and hash-bound to the competence, pruning-width, fidelity, and failure
event artifacts.

## Run-to-run reproducibility of the closed-loop measurements

Closed-loop rollouts in this runtime are **not reproducible across runs**, and every cell
in the tables above inherits that. The actor is deterministic (`actor.cpu().eval()`), but
the frozen F10 perception front-end resolves to CUDA with no determinism flags set, so
identical seeds do not yield identical trajectories. Measured directly from repeated
`(model, curriculum, seed)` cells that F15 happened to run more than once:

| | count |
|---|---:|
| repeated cells | 150 |
| differing numerically | 43 |
| differing in an **objective outcome label** | 7 |

The label flips include `A0/c2/180001`, where Original Policy completes in one run and
times out in another, and `A6/c2/180002`, where `lane_failure` and `invalid_pose` exchange
places. Evidence is retained under
`integrity/superseded_pre_shard_localization_csv/`; mechanism and consequences are
documented in `docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md`.

The practical noise floor is therefore about **one episode in eight, i.e. 0.125** — which
is exactly the size of several frozen relative margins
(`maximum_completion_rate_drop`, `maximum_lane_failure_rate_increase`,
`maximum_invalid_pose_rate_increase`, all `0.125`).

This does not threaten the principal findings, which are far outside that band: Pruning +
Knowledge Distillation and Final INT8 Policy record **0/8** completion in C0–C2 against
Original's **8/8**, a separation of eight episodes, and every historically distilled width
from 192 down to 64 records 0/8 in C0–C2. No plausible ordering of floating-point
reductions converts 0/8 into 8/8.

It does mean that cells separated by one or two episodes must be read as **not
conclusive**. Specifically:

- **PTQ in C3** (`FAIL`): completion 0.875 against Original's 1.000 — a difference of a
  single episode, exactly at the frozen margin.
- **Pruning-only C1 across widths**: 192 `PASS`, 128 `FAIL` (0.375), 96 `PASS`. The
  non-monotonicity is consistent with noise rather than a width effect.
- **Pruning Only in C0/C1** (4/8): sits directly on the `minimum_completion_rate = 0.50`
  absolute threshold.

No threshold, seed, or gate was altered after these results were seen. The cells above are
reported as measured and flagged as borderline rather than reclassified.

## What cannot yet be concluded

Localization can identify the first observed PASS→FAIL transition and whether
quantization adds a failure under the tested branch. It cannot by itself prove that a
specific removed neuron caused a failure or that incomplete rehearsal coverage is the
sole mechanism. The latter is tested as a controlled recovery factor in F15.
