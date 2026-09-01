# F17 Final Report — Optimization-Method Order and Quantization Recovery

## Classification

**F17 = COMPLETE. Primary questions answered. `NO_ELIGIBLE_FINAL_CANDIDATE` under the
frozen gates; the sealed holdout 180301–180308 was never opened.**

Design integrity: all nine pathway members were pre-existing frozen checkpoints from F12
and F15 — **no training occurred anywhere in the primary comparison** — and every pathway
was evaluated on the identical deterministic block (seeds 180201–180208,
`cuda_strict_deterministic`, 8 seeds × 5 curricula = 40 episodes each, 360 total). The
INT8 pathways ran only after a preregistered addendum verified that the scoped-
nondeterminism INT8 configuration reproduces bit-exactly under the frozen gate criteria.

## The pathway matrix

| ID | Optimization pathway | W | Prec | C0 | C1 | C2 | C3 | C4 |
|---|---|---:|---|---|---|---|---|---|
| A0 | Original Policy | 256 | FP32 | REF | REF | REF | REF | REF |
| A1 | prune | 64 | FP32 | FAIL | FAIL | FAIL | FAIL | FAIL |
| A2 | prune → KD (C4-focused) | 64 | FP32 | FAIL | FAIL | FAIL | PASS | PASS |
| **A3** | prune → KD (balanced C0–C4) — anchor | 64 | FP32 | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| **A4** | PTQ only, no pruning — control | 256 | INT8 | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| A5 | prune → PTQ | 64 | INT8 | FAIL | FAIL | FAIL | FAIL | FAIL |
| A6 | prune → KD (balanced) → PTQ | 64 | INT8 | PASS | PASS | PASS | FAIL | FAIL |
| A7 | prune → KD (C4) → PTQ → QAT+KD (C4) | 64 | INT8 | FAIL | FAIL | FAIL | PASS | PASS |
| A8 | prune → KD (balanced) → QAT+KD (balanced) | 64 | INT8 | PASS | PASS | PASS | FAIL | FAIL |

Figures: `06_optimization_method_pathways`, `07_distillation_coverage_recovery`,
`08_fp32_to_ptq_transition`, `09_qat_recovery`, `10_method_order_comparison`
(PNG + PDF each).

## Answers, in the frozen licensed wording

**1. Recovery (A2 vs A3) — rehearsal coverage is the operative factor.** Same pruned
parent, teacher, loss, optimizer, budget; only the distillation data coverage differs.
C4-focused KD leaves C0–C2 at completion 0.000 (invalid-pose 100% on C0, lane-failure
100% on C2); balanced C0–C4 KD passes all five. *Expanding rehearsal coverage from the
historical C4-focused distribution to balanced C0–C4 coverage restored all five curricula
in the tested recovered 64×64 FP32 actor* — now verified bit-exactly, with the F16
qualification that this recovery is training-realization sensitive.

**2. Quantization after recovery (A3 vs A6) — PTQ reintroduces failure.** The byte-fixed
anchor passes all five in FP32 and fails C3 (safety-relevant: completion 0.375, restart
0.375 — the robot stops correctly and never restarts) and C4 (behavioural: progress only)
after the frozen PTQ. *INT8 conversion under the frozen PTQ procedure was associated with
a new retention failure for this fixed recovered FP32 checkpoint.* No training and no
evaluation noise can account for it.

**3. Placement (A5 vs A6) — distillation before quantization is strongly beneficial.**
A5 (prune → PTQ, no distillation) fails all five with invalid-pose up to 100% and C3
stop-violation 100%. A6 (balanced KD inserted before PTQ) retains C0–C2 in full.
*Inserting balanced distillation before PTQ preserved substantially more cross-curriculum
competence than quantizing the pruned actor directly, under the tested pathway.* Not a
factorial proof of operation ordering.

**4. QAT route (A6 vs A8) — not a recovery, and an opposite phenotype.** Both routes from
the same FP32 parent fail C3/C4. The PTQ route freezes at the stop line (stop-violation
0.000, restart 0.375); the QAT+KD route violates stops (0.500 on C3, 0.875 on C4,
completion 0.125). *Under the tested procedures, the QAT+KD quantization route did not
preserve the retention that the PTQ route lost.* A8's same-state fidelity is better than
A6's on every curriculum while its C4 behaviour is far worse — the sharpest
fidelity-versus-behaviour dissociation observed in the project.

**5. Quantization-only control (A0 vs A4) — the headline.** PTQ of the unpruned Original
passes **all five curricula in INT8 behaviour**. Therefore:

> **The INT8 C3/C4 retention failure is an interaction between narrow width and
> quantization, not an effect of quantization alone.** Pruning + balanced KD passes all
> five in FP32 (A3); quantization alone passes all five in INT8 behaviour (A4); only their
> combination fails (A6, A8).

**6. A coverage-alignment pattern, reported descriptively.** Every tested INT8 pathway
retains a subset aligned with its distillation emphasis: C4-focused pathways (A7) retain
{C3, C4}; balanced pathways (A6, A8) retain {C0, C1, C2}. INT8 *can* pass every
curriculum — A7 proves C3/C4 in INT8, A6/A8 prove C0–C2 in INT8, A4 proves all five at
width 256 — but no tested width-64 pathway retains all five simultaneously. The mechanism
is outside F17's frozen scope (see `docs/F16_QUANTIZATION_SCOPE_LIMITATION.md`); the
untested quantization representation remains the cleanly-motivated next variable.

## Eligibility and the holdout

Zero of nine pathways are eligible under the frozen rule (INT8 + behaviour + fidelity +
safety + provenance):

- **A4** passes all behaviour but fails the frozen same-state fidelity gate on
  Pearson/Spearman correlation components in 4/5 curricula (e.g. C4 Spearman 0.9229 vs
  gate 0.970) — replicating the historical F12 finding for unpruned PTQ. Its v/ω MAE are
  tiny (≤0.005 / ≤0.126). This is a fidelity-gate failure with intact behaviour — the
  reverse dissociation — and the correlation components' fragility on low-variance
  signals is a documented metric limitation. The gates are frozen and were not modified.
- **A6/A8** fail C3/C4 behaviour.

`results/eligibility_outcome.json` records `NO_ELIGIBLE_FINAL_CANDIDATE`. No candidate was
frozen, no holdout claim was written, and **180301–180308 remain sealed**. No additional
tuning was run to force a PASS; the quantization procedure was fixed throughout.

## Evidence

- `results/pathway_results.csv` (45 rows: 9 pathways × 5 curricula, with failure
  phenotype), `results/same_state_fidelity.csv`, `results/pathway_summary.json`,
  `results/eligibility_outcome.json`
- `integrity/int8_determinism_addendum.json` (PASS, bit-exact ×3 repeats),
  `integrity/comparison_interpretation_amendment.json`
- Primary-rollout camera media for every objective failure under `primary_media/`
  (MP4/GIF/contact sheet, overlays from the same episode's telemetry)
- Figures 06–10, PNG + PDF

## Limitations

1. One evaluation block (8 seeds); the block-sensitivity of `minimum_clearance` documented
   in F16 applies here too (it played no role in any F17 verdict flip).
2. The anchor is one training realization; F16 showed balanced-KD recovery is
   training-realization sensitive. F17's comparisons are conditional on this anchor.
3. The quantization representation was fixed (x86 static, per-tensor activations); no
   claim extends beyond it.
4. Width-64 conclusions do not automatically transfer to other widths; A4 shows width 256
   behaves qualitatively differently under PTQ.
5. Fidelity-gate correlation components are fragile on low-variance signals; both
   dissociation directions (fidelity-pass/behaviour-fail and behaviour-pass/fidelity-fail)
   were observed in this project.
