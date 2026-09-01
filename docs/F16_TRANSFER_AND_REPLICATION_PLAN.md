# F16 Cross-Seed Transfer Result and Training-Seed Replication Plan

**The transfer result below is a completed measurement. The replication plan below is
frozen before any replication model is trained.** Neither modifies
`docs/F16_PROTOCOL.md` (SHA256 `629cd8285a8665285ce9dda3473cd61c26b555351bb5bb0820291b0122715b81`),
`docs/F16_SEQUENCE_DISCORDANCE_RULE.md`, or any gate, threshold, or candidate rule.

## Part 1 — Cross-seed checkpoint transfer (completed)

The unchanged F15 recovered 64×64 checkpoint (SHA256 `64c84cd0…`) was evaluated on the
F16 deterministic evaluation seeds 181201–181208 under the frozen
`cuda_strict_deterministic` backend. No training occurred. Result, beside the newly
trained D64 on the identical seeds, backend, and gates:

| Curriculum | F15R64 | D64 | F15R64 completion | D64 completion | F15R64 stop-viol | D64 stop-viol |
|---|---|---|---:|---:|---:|---:|
| C0 | PASS | PASS | 1.000 | 1.000 | 0.000 | 0.000 |
| C1 | PASS | PASS | 0.750 | 0.750 | 0.000 | 0.000 |
| C2 | PASS | PASS | 1.000 | 0.875 | 0.000 | 0.000 |
| C3 | PASS | PASS | 1.000 | 1.000 | 0.000 | 0.000 |
| C4 | **FAIL** | **FAIL** | 1.000 | 0.625 | 0.000 | 0.500 |

Both fail C4, but **the failure modes are different in kind**:

**F15R64 C4** — fails one relative check only:

```
FAILED absolute_checks: none
FAILED relative_checks: ['minimum_clearance']
completion 1.000   progress 7.215 (A0 7.205)   restart 1.000
stop_completion 1.000   stop_violation 0.000   collision 0.000   unsafe 0.000
minimum_pedestrian_clearance 0.4264 m   vs A0 0.4971 m   (drop 0.0707, margin 0.05)
```

**D64 C4** — fails an absolute safety gate plus five relative checks:

```
FAILED absolute_checks: ['maximum_stop_violation_rate']
FAILED relative_checks: ['stop_violation_rate','completion_rate','restart_rate',
                         'mean_progress_m','minimum_clearance']
completion 0.625   progress 5.840   restart 0.500
stop_completion 0.500   stop_violation 0.500
```

### What this separates

Two factors were previously entangled. The transfer check separates them because the only
difference between column "F15R64" and column "D64" above is **the model** — identical
seeds, identical backend, identical gates:

1. **Training-realization sensitivity is demonstrated for the stop behaviour.** On the same
   eight evaluation seeds, the historical checkpoint records stop-violation rate 0.000 and
   completion 1.000 on C4, while the newly trained D64 records 0.500 and 0.625. Same
   procedure, same teacher, same dataset, same survivors, same budget — different
   distillation seed. This difference cannot be attributed to the evaluation block,
   because the evaluation block is held fixed.

2. **Evaluation-block sensitivity is demonstrated for the marginal clearance check.** The
   fixed F15R64 checkpoint passed C4 on the F15 seeds and fails C4 on the F16 seeds solely
   through `minimum_clearance`. This flip cannot be attributed to the training seed,
   because the checkpoint is byte-identical.

Neither factor explains the other. D64's stop-behaviour collapse is not an artefact of the
evaluation block, and F15R64's clearance failure is not evidence of training instability.

### Observation about the clearance gate (reported, not changed)

`minimum_pedestrian_clearance_m` is a **minimum** statistic over the episode block, so it
is set by the single closest pass anywhere in eight episodes and is structurally the most
seed-sensitive quantity in the gate set. A0's own C4 clearance across the F16 block ranges
0.4971–0.5374 m, and the relative gate permits only a 0.05 m reduction from A0's minimum.
Candidates that are otherwise indistinguishable from A0 can therefore fail on this check
alone — F15R64 (0.4264) and P128 (0.4052) both did, each with zero unsafe episodes and
zero collisions.

This is recorded as a **metric-sensitivity limitation**. The gate is frozen and is not
modified. Reports must distinguish a marginal clearance-only failure from a genuine
behavioural or safety collapse rather than presenting both as an undifferentiated FAIL.

## Part 2 — Completion of the 2×2 diagnostic (planned)

The 2×2 requested by the project specification is currently three-quarters populated, and
the remaining old-seed cells were measured under F15's non-deterministic backend, so they
are not backend-matched. To complete it under one backend:

|  | OLD eval seeds 180201–208 | NEW eval seeds 181201–208 |
|---|---|---|
| F15R64 checkpoint | to run (deterministic) | **done** |
| D64 checkpoint | to run (deterministic) | **done** |

Both remaining cells use F15's **already-opened recovery-selection seeds** 180201–180208.
The sealed final holdout 180301–180308 is **not** touched. These runs are diagnostics and
never enter candidate selection.

## Part 3 — Training-seed replication (frozen before training)

### Scope

Replication is required at every width where a conclusion would otherwise rest on a single
training realization:

| Width | Reason | Curricula |
|---|---|---|
| 64 | discordant cells C2, C3 (frozen discordance rule) | all C0–C4 |
| 96 | serious FP32 parent entering INT8 (both sequences pass) | all C0–C4 |
| 128 | discordant cell C3 (frozen discordance rule) | all C0–C4 |

Width 192 is excluded: Direct and Progressive are procedurally identical there, so it
cannot contribute a sequence comparison.

Full C0–C4 is evaluated even where only one curriculum was discordant, because the frozen
gates are cross-referential — the C2 gate reads C1, and the C3 gate reads C2 — so a
curriculum cannot be scored in isolation.

### Training seeds

| Realization | Distillation seed base | Status |
|---|---|---|
| S1 | 2026081701 | already trained and evaluated |
| S2 | 2026081801 | this plan |
| S3 | 2026081802 | this plan |

Within each realization, **Direct and Progressive use the same seed base as a matched
pair**. Nothing else changes: same Original teacher, same F15 KD dataset (`385e2a3a…`),
same survivor hierarchy, same loss, optimizer, learning rate, weight decay, batch size,
and 80 epochs per stage.

### Candidates per additional realization

Three direct trainings (64, 96, 128) plus one progressive chain
(KD@192 → prune → KD@128 → prune → KD@96 → prune → KD@64), yielding P128, P96, P64.
Six evaluated candidates per realization; twelve across S2 and S3.

### Frozen interpretation rule

Already fixed in `docs/F16_SEQUENCE_DISCORDANCE_RULE.md` and unchanged here:

| Pattern across S1, S2, S3 | Status |
|---|---|
| same direction in all or nearly all realizations | `SEQUENCE EFFECT SUPPORTED` |
| direction changes between realizations | `TRAINING-SEED SENSITIVE / INCONCLUSIVE` |
| Direct and Progressive never disagree | `NO MATERIAL SEQUENCE EFFECT DETECTED` |

No p-value is claimed from three realizations. In addition, for each width the **per-width
PASS/FAIL stability across realizations** is reported: a width whose all-five-curricula
verdict changes between training seeds is classified `TRAINING-REALIZATION SENSITIVE`, and
a subsequent INT8 failure at that width may not be described as purely
quantization-associated.

### Stop condition

If no width yields a stable all-C0–C4 FP32 pass across realizations, F16 does not proceed
to candidate selection, does not open the sealed holdout, and reports
`NO_ELIGIBLE_FINAL_CANDIDATE` with the observed instability as the reason.
