# F17 — Optimization-Method Order and Quantization Recovery Protocol

**Frozen before any F17 scientific run.** Config:
`configs/f17_optimization_method_order_v1.toml`.

## Primary question

> After pruning-induced competence collapse and successful multi-curriculum distillation
> recovery, how does the **placement and order** of distillation and quantization affect
> cross-curriculum retention?

This is the investigator's intended meaning of "does optimization sequence matter". It is
**not** Direct-versus-Progressive pruning, which is a **pruning schedule** and was studied
separately in F16.

| Term | Meaning |
|---|---|
| actor width | 64 / 96 / 128 / 192 |
| pruning schedule | Direct / Progressive — **F16 subject, held fixed here** |
| optimization-method order (optimization pathway) | placement/order of pruning, distillation, PTQ, QAT — **the F17 subject** |

## Inherited position

Already answered and not rerun:

1. **First collapse** — Original → structured pruning. All C0–C4 changed from PASS to FAIL
   after pruning.
2. **Historical partial recovery** — C4-focused KD recovered C3/C4 but not C0–C2.
3. **Multi-curriculum recovery is achievable** — the validated F15 balanced C0–C4 KD
   checkpoint passed all five curricula on seeds 180201–180208 under deterministic
   re-evaluation.
4. **Robustness qualification (F16, secondary)** — that recovery is training-realization
   sensitive, no monotonic width–retention relationship was observed, and no stable
   pruning-schedule advantage was established.

Point 3 is the primary recovery finding. Point 4 qualifies it; it does not erase it.

## Design

Width and pruning schedule are **held fixed**. Only the optimization pathway changes. This
is what makes the comparison causally interpretable.

### Canonical recovered FP32 anchor

`artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`,
SHA256 `64c84cd0bad44ddaa564a5895c88b82254950752b322030ce67df912a3667276`, width 64,
Direct pruning schedule.

Not retrained, not modified. It has already been independently verified to PASS all five
curricula on the primary evaluation block under the deterministic backend
(`artifacts/f16_sequence_int8_recovery_v1/closed_loop/transfer_F15R64_on_f15_selection_episodes.csv`,
40/40 episodes).

Because the parent is a single fixed checkpoint, any A3 → A6 change is associated with the
frozen PTQ procedure applied to **that exact checkpoint** — not with a training draw.

### No large sweeps

F17 runs **no width sweep**, **no pruning-schedule replication**, and **no S1/S2/S3
training-seed matrix**. Every pathway member already exists as a frozen checkpoint from F12
or F15, so F17 is an evaluation study on a matched block and no new training enters the
primary comparison at all.

### Primary evaluation block

Seeds **180201–180208** under `cuda_strict_deterministic`, inheriting the F16 determinism
gate (zero action delta, zero progress delta, identical episode length, completion,
failure labels, termination reason on the preregistered set).

These are already-opened F15 recovery-selection seeds. **The sealed final holdout
180301–180308 stays closed.** Every F17 branch uses the identical block.

## Optimization-pathway matrix

| ID | Pathway | Width | Precision |
|---|---|---:|---|
| A0 | Original Policy | 256 | FP32 |
| A1 | prune | 64 | FP32 |
| A2 | prune → KD (C4-focused) | 64 | FP32 |
| **A3** | **prune → KD (balanced C0–C4) — anchor** | 64 | FP32 |
| A4 | PTQ, no pruning — quantization-only control | 256 | INT8 |
| A5 | prune → PTQ, no distillation | 64 | INT8 |
| A6 | prune → KD (balanced) → PTQ | 64 | INT8 |
| A7 | prune → KD (C4) → PTQ → QAT+KD (C4) — historical final | 64 | INT8 |
| A8 | prune → KD (balanced) → QAT+KD (balanced) → INT8 | 64 | INT8 |

Only practically valid pathways are included. No "static INT8 → ordinary FP32 KD" route is
invented, because the framework cannot train that graph meaningfully. Parallel branches are
reported as branches, never flattened into a linear chain.

## What each comparison answers

| Comparison | Question |
|---|---|
| A0 vs A1 | does pruning introduce the first collapse (historical anchor) |
| A1 vs A2 | how much does historical C4-focused distillation recover |
| **A2 vs A3** | **PRIMARY RECOVERY** — does broader KD rehearsal improve cross-curriculum recovery |
| A0 vs A4 | **quantization-only control** — quantization without pruning |
| A1 vs A5 | quantizing a pruned, unrecovered actor |
| **A3 vs A6** | **PRIMARY QUANTIZATION** — does PTQ introduce a new PASS→FAIL after a known FP32 recovery |
| **A5 vs A6** | **PRIMARY METHOD ORDER** — does balanced KD before PTQ preserve more competence than quantizing the pruned actor directly |
| A6 vs A8 | can balanced QAT with teacher guidance recover competence lost under quantization |
| A7 vs A8 | does balanced rehearsal in the QAT stage outperform historical C4-focused QAT |

Method-order claims are made only among width-64 members. A0 and A4 are width-256
controls and never license a method-order claim.

## Measurement, reported separately

**Same-state action fidelity** — identical public normalized 29D rows replayed offline
through every candidate: v MAE, omega MAE, RMSE, P95, P99, signed bias, omega sign
disagreement, tie-aware Spearman, saturation disagreement.

**Closed-loop curriculum retention** — completion, progress, restart, stop completion,
stop violation, collision, unsafe episode, minimum pedestrian clearance, termination
reason, for C0–C4.

Closed-loop competence is never inferred from low action MAE alone. F16 established that
every FP32 candidate passed the fidelity gate including candidates that failed closed-loop
behaviour.

## Failure phenotype

Descriptive annotation with deterministic precedence, alongside every verdict:

1. collision / unsafe / stop violation / stop-completion / restart → `safety_relevant`
2. else completion / progress / timeout / lane failure / invalid pose → `behavioural`
3. else `minimum_clearance` only → `marginal_clearance_only`

This never changes the official verdict. PASS/FAIL remains based only on the frozen gates.
It exists because F16 showed the gates treat a 9 mm clearance margin and an actual
collision as the same FAIL.

## Primary visual evidence

RGB is captured **during the primary scientific rollout** through the validated in-memory
ring buffer, proven not to perturb policy execution. For every objective failure: MP4, GIF,
PNG contact sheet, telemetry, episode JSON, failure-event JSON, model SHA256, config
SHA256, curriculum, seed, precision, and optimization pathway.

No failure is reconstructed later by replay. Paired videos are labelled **Same-Seed Primary
Rollouts**, never "causal counterfactual trajectory".

## Decision tree

```
A3 recovered FP32 anchor  (already verified PASS C0-C4)
        |
        v
A6 = anchor -> PTQ  ->  evaluate C0-C4
        |
        +-- retains all required competence  ->  record PTQ retention success
        |
        +-- new PASS->FAIL  ->  document curriculum and failure phenotype
                                        |
                                        v
                            A8 = anchor -> balanced QAT+KD -> INT8
                                        |
                                        +-- recovered      -> INT8 RECOVERY ACHIEVED UNDER TESTED QAT/KD PROCEDURE
                                        +-- not recovered  -> INT8 RECOVERY NOT ACHIEVED UNDER TESTED PROCEDURE
```

No endless tuning until a PASS appears. The quantization procedure is fixed.

## Final holdout

**180301–180308 stays closed** during pathway comparison, PTQ calibration, QAT, KD, model
selection, failure analysis, and figure generation. It is opened only if a final candidate
is frozen under this protocol. If F17 produces no eligible candidate, the holdout is not
opened and the negative result is reported.

## Claim discipline

Permitted when supported:

- "Pruning was the first observed optimization stage at which curriculum Cx changed from
  PASS to FAIL."
- "Balanced C0–C4 KD demonstrated full recovery in the validated recovered run."
- "INT8 conversion under the frozen quantization procedure was associated with a new
  retention failure for the fixed recovered FP32 checkpoint."
- "Performing balanced distillation before PTQ preserved more competence than quantizing
  the pruned actor directly, under the tested procedure."
- "QAT with teacher guidance restored / failed to restore the INT8 retention loss."

Not permitted:

- "Balanced KD always recovers C0–C4."
- "Direct or Progressive pruning is better."
- "INT8 caused failure" without a fixed-checkpoint PASS→FAIL under the same block.
- "Quantization resolution caused the stop failure" — not tested here.
- "The training seed caused the mechanism."
- Any generalization beyond the tested quantization procedure.

## Stop rules

Stop before evaluation if any checkpoint hash, registry hash, config hash, 29D contract,
action map, INT8 invocation, or the inherited determinism gate fails. Stop before final
access if candidate provenance or the once-only claim is missing. Never change gates,
seeds, curricula, the anchor, or the quantization procedure after results are seen.

## Artifact namespace

All outputs are new under `artifacts/f17_optimization_method_order_v1/`. F10–F16 paths are
read-only.
