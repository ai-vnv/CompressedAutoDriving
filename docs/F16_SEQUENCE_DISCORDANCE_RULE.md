# F16 Sequence-Discordance Rule and Cross-Seed Transfer Check

**Frozen before any matched Direct/Progressive pair completed.** At the time of writing,
D64 had finished (40/40) and P64 was still running, so no matched comparison had been
observed. This note adds interpretation rules only; it changes no gate, threshold, seed,
candidate, or selection rule in `docs/F16_PROTOCOL.md`
(SHA256 `629cd8285a8665285ce9dda3473cd61c26b555351bb5bb0820291b0122715b81`), which remains
unmodified.

## Why this is needed

The determinism gate removed **evaluation** noise: repeated runs of the same
(model, curriculum, seed) now reproduce exactly. It did **not** remove **training** noise.
Each F16 candidate is trained once, so a Direct-versus-Progressive difference at a matched
endpoint could be caused by the optimization sequence, by the training-seed draw, or by
both. With one training run per candidate these cannot be separated.

This became concrete before any matched pair existed. D64 — a fresh Direct-sequence
training at width 64 under the identical frozen procedure — failed C4 in FP32
(completion 0.625, stop-violation rate 0.500), whereas the F15 recovered 64×64 actor had
passed all five curricula.

**The F15 64×64 recovery result did not reproduce in the newly trained D64 model under the
new deterministic F16 evaluation block. Because both the distillation seed and the
evaluation seed block differ, training-seed sensitivity cannot yet be separated from
evaluation-distribution sensitivity.**

| | F15 | F16 |
|---|---|---|
| distillation seed | 2026081601 | 2026081701 |
| evaluation seeds | 180201–180208 | 181201–181208 |
| evaluation backend | non-deterministic | `cuda_strict_deterministic` |

Three factors differ, so no single-factor attribution is permitted from this observation
alone.

## Check 1 — cross-seed checkpoint transfer

Take the **existing** F15 recovered FP32 64×64 checkpoint
(`artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`,
SHA256 `64c84cd0bad44ddaa564a5895c88b82254950752b322030ce67df912a3667276`) and evaluate it
**unchanged** on the F16 deterministic evaluation seeds 181201–181208. No training occurs.

This holds the model fixed while changing only the evaluation block and backend:

| Outcome | Reading |
|---|---|
| F15 checkpoint PASSES C4 while D64 FAILS C4 on the same seeds | training-run / model sensitivity is strongly supported |
| F15 checkpoint also FAILS C4 on the same seeds | evaluation-distribution sensitivity is the more likely factor, not the training seed |
| Same verdict but different margins | verdicts agree; report the robustness gap quantitatively |

This is registered as a **diagnostic transfer check**. It is not a candidate, it is not
eligible for selection, and it never enters the final-candidate hierarchy.

## Check 2 — discordance rule for sequence claims

> **If Direct and Progressive disagree in verdict at a matched width × curriculum, the
> sequence effect is recorded as `PROVISIONAL` and triggers confirmatory training-seed
> replication of that discordant cell only.**

Replication is deliberately narrow so F16's cost does not explode: only discordant cells
are replicated, never the whole matrix. Each replication uses an additional frozen
training seed, and **Direct and Progressive use the same training seed as a matched
pair** within each replication.

Classification after replication:

| Pattern across independent training realizations | Status |
|---|---|
| same direction in all or nearly all training seeds | `SEQUENCE EFFECT SUPPORTED` |
| direction changes between training seeds | `TRAINING-SEED SENSITIVE / INCONCLUSIVE` |
| Direct and Progressive never disagree | `NO MATERIAL SEQUENCE EFFECT DETECTED` |

No formal p-value is claimed from a handful of training realizations. The report states
the observed pattern across independent training runs and nothing stronger.

## Wording constraints

Permitted:

- "The F15 64×64 recovery result did not reproduce in the newly trained D64 model under
  the new deterministic F16 evaluation block."
- "Training-seed sensitivity cannot yet be separated from evaluation-distribution
  sensitivity."
- "Direct and Progressive disagreed at width W on curriculum C under one training
  realization; this is provisional pending replication."

Not permitted:

- "Balanced KD recovery at 64×64 is not reproducible." (two factors changed)
- "The training seed caused the C4 failure." (not isolated)
- "Progressive is better than Direct." (single training run per sequence)
- Any sequence claim from unmatched widths.

## Effect on the F16 research question

This does not weaken the F16 design; it adds a factor the study must account for.
Robustness to the training realization is now recognised as a property a recovery must
demonstrate before being called stable, and it can be probed with targeted checks rather
than by repeating the whole experiment.
