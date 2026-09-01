# F16 Final Report — Scope Corrected

## Classification

**F16 = SECONDARY ROBUSTNESS / PRUNING-SCHEDULE STUDY**

F16 is **not** the answer to "does optimization-method order matter?". That question — the
placement and order of pruning, distillation, PTQ, and QAT — is addressed in F17. F16
answers three narrower, secondary questions.

## Terminology

| Term | Meaning in this project |
|---|---|
| **actor width** | 64 / 96 / 128 / 192 |
| **pruning schedule** | Direct (prune straight to target width) versus Progressive (prune in stages with distillation between) |
| **optimization-method order / optimization pathway** | the order and placement of pruning, distillation, PTQ, QAT — the F17 subject |

The earlier F16 use of "sequence" for Direct versus Progressive was a misnomer. Figure
titles and this report use **pruning schedule**. Historical CSV column names are retained
where renaming them would break provenance; new reporting artifacts use
`pruning_schedule`.

## What F16 answered

### A. Does actor width show a simple monotonic retention relationship?

**No.** No monotonic width–retention relationship was observed.

| Width | Pruning schedule | S1 | S2 | S3 | Stability |
|---|---|---|---|---|---|
| 64 | Direct | FAIL | FAIL | FAIL | `STABLE_FAILING` |
| 64 | Progressive | FAIL | FAIL | FAIL | `STABLE_FAILING` |
| 96 | Direct | PASS | PASS | FAIL | `TRAINING-REALIZATION SENSITIVE` |
| 96 | Progressive | PASS | FAIL | cancelled | `TRAINING-REALIZATION SENSITIVE` |
| 128 | Direct | FAIL | PASS | not run | `TRAINING-REALIZATION SENSITIVE` |
| 128 | Progressive | FAIL | FAIL | not run | `STABLE_FAILING` |
| 192 | Direct ≡ Progressive | FAIL (C2) | — | — | single realization |

Width 128 Direct failed on the first realization and passed on the second. Width 192
failed C2 while width 96 passed everything on the same realization. The one statement that
survives replication is narrower and stronger:

> **Width 64 failed consistently across both pruning schedules and all three completed
> training realizations** — six independent models, with the majority of failures
> safety-relevant (stop violations, and one collision).

No capacity threshold is asserted.

### B. Does the Direct versus Progressive pruning schedule provide a stable advantage?

**No stable pruning-schedule advantage was established.**

Five width x curriculum cells ever showed a Direct/Progressive verdict difference. **All
five** were classified `TRAINING-SEED SENSITIVE / INCONCLUSIVE`; **none** reached
`SEQUENCE EFFECT SUPPORTED`.

| Width | Curriculum | S1 | S2 |
|---|---|---|---|
| 64 | C2 | Direct better | concordant |
| 64 | C3 | **Direct better** | **Progressive better** |
| 96 | C2 | concordant | Direct better |
| 128 | C3 | **Progressive better** | concordant |
| 128 | C4 | concordant | Direct better |

The width-64 C3 cell reversed outright between training realizations. Progressive also
used 4x the cumulative distillation epochs at width 64 (320 versus 80) and 3x at width 96
(240 versus 80) without a stable behavioural gain; that asymmetry is reported, not hidden.

This is not merely absence of evidence. A difference was observed twice, in opposite
directions, under bit-exact deterministic evaluation.

### C. How sensitive is balanced-KD recovery to training realization?

**Recovery is achievable but not guaranteed across independent training realizations.**

The most direct demonstration is width 96 Direct: two realizations reproduced the Original
Policy almost exactly (completion identical to A0 on all five curricula, progress within
2 cm, zero stop violations, zero collisions), and the third collapsed on the stop
curricula — completion 0.000 on C3 and 0.125 on C4 with stop-violation rate 1.000, from an
identical procedure differing only in the distillation seed.

A backend-matched 2x2 diagnostic separated two effects that had been entangled:

|  | OLD block 180201–208 | NEW block 181201–208 |
|---|---|---|
| F15 recovered checkpoint | **PASS all five** | FAIL C4 — `minimum_clearance` only |
| Newly trained D64 | FAIL C4 — stop violations | FAIL C4 — stop violations |

- **Training-realization sensitivity**: D64 recorded completion 0.625 and stop-violation
  rate 0.500 on C4 on *both* blocks with identical values, while the F15 checkpoint
  recorded 1.000 and 0.000 on both. The defect is block-independent.
- **Evaluation-block sensitivity**: with the checkpoint byte-identical and the backend
  fixed, changing only the eight evaluation seeds flipped C4 from PASS to FAIL, solely
  through `minimum_clearance` (0.4513 m to 0.4264 m).

The two effects act on different metrics and neither explains the other.

## What F16 did not change

**The primary recovery finding stands.** The validated F15 balanced C0–C4 KD checkpoint
passed all five curricula on its own recovery-selection block under deterministic
re-evaluation, reproducing what F15 reported. The F15 conclusion was therefore not an
artefact of the non-deterministic backend.

> Expanding rehearsal coverage from the historical C4-focused distribution to balanced
> C0–C4 coverage restored all five curricula in the tested recovered 64x64 FP32 actor.

Secondary qualification, which must not erase the primary finding:

> The recovery was achievable but was not guaranteed across all independent training
> realizations tested later in F16.

## Integrity gates passed

1. **Determinism.** Strict deterministic CUDA produced exact repeatability on all
   preregistered determinism-gate comparisons: zero normalized-action difference, zero
   progress difference, identical episode length, completion outcome, failure labels, and
   termination reason. Scoped to those measured fields on that preflight set. CPU fallback
   was attempted, aborted before any episode for a device-configuration reason, and never
   measured — CUDA was selected because it was the first backend in the preregistered
   fallback chain to satisfy the gate, not because it beat CPU.
2. **Primary camera evidence.** RGB is captured during the primary scientific rollout via
   an in-memory ring buffer, proven not to perturb policy execution (zero action delta,
   identical episode length with and without capture). This closes the F15 gap where no
   valid primary front-camera footage existed.

## Eligibility outcome

**`NO_ELIGIBLE_FINAL_CANDIDATE`.**

No width x pruning-schedule combination achieved a stable all-C0–C4 FP32 pass across
training realizations, so the frozen stop condition triggered. No candidate was frozen, no
INT8 stage was run in F16, and the sealed holdout **180301–180308 was never opened**.

The replication matrix was stopped early once the decision became arithmetically
unreachable; see `docs/F16_DECISIVE_EARLY_STOP.md`.

## Metric-sensitivity limitation

`minimum_pedestrian_clearance_m` compares the candidate's minimum over eight episodes
against A0's minimum over the same eight episodes, with a 0.05 m allowance. It is a
difference of two minimum statistics and is structurally the most seed-sensitive quantity
in the frozen gate set. Three otherwise-healthy models failed on it alone with zero unsafe
episodes and zero collisions, and at width 96 all four models clustered within
0.168–0.245 m around a gate line at 0.184 m — one pass held by 5.6 mm.

The gate is frozen and unmodified. A descriptive severity annotation
(`marginal_clearance_only` / `behavioural` / `safety_relevant`) is reported alongside every
verdict so a marginal clearance miss is not read as a safety collapse. Severity never
alters a verdict.

## Stop-phase diagnostic outcome

The C3/C4 stop-phase diagnostic produced a **negative result** and is retained as such.
Absolute steering discrepancy was of similar magnitude across phases; the two
normalisations (MAE/SD and MAE/IQR) disagreed; the C3 approach phase was low-dynamic-range
in 6 of 8 episodes; and delta-omega error was *lower* during approach than during nominal
driving.

> The tested diagnostic did not provide robust evidence that action divergence was
> concentrated specifically in approach or deceleration.

The earlier v1 analysis is marked `SUPERSEDED` because ordinal ranking mishandled ties and
timestep-level aggregation overstated the effective sample size. These diagnostics are
descriptive only and never gated candidate eligibility.

## Claim discipline

Supported:

- "No monotonic width–retention relationship was observed."
- "Width 64 failed consistently across both pruning schedules and all three completed
  training realizations."
- "No stable pruning-schedule advantage was established."
- "Recovery showed training-realization sensitivity."
- "Balanced C0–C4 KD demonstrated full recovery in the validated recovered run."

Not claimed:

- that Direct or Progressive is better;
- that larger width is safer;
- that the training seed *caused* any mechanism;
- that balanced KD always recovers C0–C4;
- anything about optimization-method order — that is F17.

## Artifacts

`artifacts/f16_sequence_int8_recovery_v1/` — `integrity/` (determinism gate, media gate,
scope clarification, discordance rule, 2x2 diagnostic, decisive early stop),
`results/` (collapse map, training-realization results, width results, sequence
classification, same-state fidelity, 2x2), `closed_loop/` (episodes), `primary_media/`
(primary-rollout MP4/GIF/contact sheets), `figures/` (01–05, 08).

Historical F10–F15 artifacts were not modified.
