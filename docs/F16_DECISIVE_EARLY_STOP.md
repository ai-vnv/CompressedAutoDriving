# F16 Decisive Early Stop

**Investigator-directed efficiency decision taken after the decisive robustness result.**
This is not missing data presented as completion. Unfinished cells are marked explicitly,
no value is fabricated, and nothing was deleted.

Machine-readable record: `artifacts/f16_sequence_int8_recovery_v1/integrity/decisive_early_stop.json`

## Terminology correction applied from here on

| Term | Meaning |
|---|---|
| **actor width** | 64 / 96 / 128 / 192 |
| **pruning schedule** | Direct / Progressive |
| **optimization-method order (optimization pathway)** | the placement/order of pruning, distillation, PTQ, QAT |

Direct versus Progressive is a **pruning schedule**. It is no longer described as an
"optimization-method sequence". The optimization-method-order question is addressed in
F17, not F16.

## Completed cells (15)

All at 40/40 episodes, 8 seeds x 5 curricula, deterministic backend, SHA256 recorded in
the machine-readable record.

| Width | Pruning schedule | S1 | S2 | S3 |
|---|---|---|---|---|
| 64 | Direct | 40/40 | 40/40 | 40/40 |
| 64 | Progressive | 40/40 | 40/40 | 40/40 |
| 96 | Direct | 40/40 | 40/40 | 40/40 |
| 96 | Progressive | 40/40 | 40/40 | — |
| 128 | Direct | 40/40 | 40/40 | — |
| 128 | Progressive | 40/40 | 40/40 | — |

## Incomplete cells (3), explicitly marked

| Cell | Episodes | Status |
|---|---|---|
| `P96_S3` | 34/40 | `CANCELLED_AFTER_DECISIVE_ELIGIBILITY_RESULT` |
| `D128_S3` | 0/40 | `NOT_RUN_AFTER_DECISIVE_STOP` |
| `P128_S3` | 0/40 | `NOT_RUN_AFTER_DECISIVE_STOP` |

The 34 completed `P96_S3` episodes (C0–C3 complete, C4 at 2/8) are **retained** with their
partial-CSV hash. They are not used to compute any all-curricula verdict.

## Why the remaining cells could not reverse the decision

The eligibility question was whether any width x pruning-schedule combination achieves
**3/3 all-C0–C4 PASS across independent training realizations**. Once a combination has
recorded a single all-curricula FAIL, 3/3 is arithmetically unreachable no matter what any
remaining evaluation returns.

| Width / schedule | S1 | S2 | S3 | Known FAILs | Can still reach 3/3? |
|---|---|---|---|---:|---|
| Direct-64 | FAIL | FAIL | FAIL | 3 | no |
| Progressive-64 | FAIL | FAIL | FAIL | 3 | no |
| Direct-96 | PASS | PASS | **FAIL** | 1 | no |
| Progressive-96 | PASS | **FAIL** | not evaluated | 1 | no |
| Direct-128 | **FAIL** | PASS | not evaluated | 1 | no |
| Progressive-128 | **FAIL** | **FAIL** | not evaluated | 2 | no |

`combinations_that_could_still_reach_3_of_3` = **[]** — computed programmatically, not
asserted.

The frozen stop condition in `docs/F16_TRANSFER_AND_REPLICATION_PLAN.md` was therefore
already satisfied:

> If no width yields a stable all-C0–C4 FP32 pass across realizations, F16 does not
> proceed to candidate selection, does not open the sealed holdout, and reports
> `NO_ELIGIBLE_FINAL_CANDIDATE` with the observed instability as the reason.

## Process termination

- Mechanism: `tmux kill-session -t f16`
- In-flight candidate at stop: `P96_S3` (34/40, partial data preserved)
- Partial data deleted: **no**
- Resume script retained: `artifacts/f16_chain_resume.sh` — re-running it would continue
  from the recorded counts, but doing so is not required for any F16 conclusion.

## Sealed holdout

Seeds **180301–180308 were never opened** and remain sealed. No candidate was frozen and
no holdout claim was written.

## What is frozen

Hashes of every file under `results/` and `integrity/` are recorded in
`integrity/decisive_early_stop.json` at the moment of stopping. Historical F10–F15
artifacts were not touched.
