# F14 Final Report — Explainability-Aware Compression Diagnostics

## Result

Overall F14 classification: **LIMITED**.

| Axis | Classification |
|---|---|
| Integrity / explanation validity | PASS |
| Efficiency | PASS |
| C4 behavioral preservation | PRESERVED |
| Semantic attribution preservation | SHIFTED |
| Counterfactual functional sensitivity | SHIFTED |
| Historical retention | NOT PRESERVED |
| Semantic retention explanation | UNRESOLVED |

LIMITED is the scientifically appropriate result: A7 is a valid, efficient,
C4-capable deployment actor, but it is not semantically or functionally
equivalent to A0 under the frozen model-agnostic diagnostics.

## Verified policies and boundary

All registered A0--A7 paths and hashes match the frozen F12 registry. A0 actor
SHA256 is `713d26d93488a17fae246b227e1de38f51501dc87a3d20ac6176036a8a8e64c5`;
the original PPO checkpoint remains
`02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`;
A7 is `f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e`.
The six semantic groups partition the same normalized public 29D input exactly
once. RGB, perception, belief filters, normalization, and action mapping are
unchanged; privileged state is absent from explanation inputs.

## Reference calibration

A0-only calibration passed before A1--A7 inspection using 500 states, six
independent draws, and four complete same-phase cross-seed references per draw.
Median/P05 group Spearman was 1.000/0.943, median/P95 group-share L1 was
0.031/0.081, and maximum local-accuracy residual was `2.384e-07`. Thresholds
were frozen from this variability and were not weakened afterward.

## What compression changed

- Pruning-only A1 caused the largest combined degradation: semantic and
  functional shifts, large action error, and 0% C4 completion.
- Distillation A2 restored action fidelity, all primary counterfactual cells,
  and C4 completion, but not A0's exact attribution structure.
- PTQ-only A3 introduced both action-equivalence failure and partial semantic /
  functional drift while retaining tested C4 behavior.
- QAT/KD A4 improved functional preservation modestly but did not pass frozen
  fidelity or semantic-equivalence gates.
- A5 confirms that quantizing the directly pruned actor did not repair the
  pruning-dominated failure.
- A6/A7 preserve C4 behavior and F12 action fidelity after distillation, while
  retaining meaningful attribution and counterfactual drift.

## Final A0 vs A7

On the frozen R004 4,400-state, 24-reference comparison, A7 preserves several
phase-level top groups—Pedestrian during pedestrian-relevant states, Stop for
stop-required velocity, and Lane after stop satisfaction—but redistributes
substantial mass toward PreviousAction. Stop-required yaw changes top group
from Lane to PreviousAction. Only 1/10 semantic phase/action cells and 1/3
preregistered primary functional cells satisfy all strict frozen preservation criteria.

F12 efficiency remains valid: 91.61% parameter reduction, 87.69% actor-file
reduction, and 3.04x actor-only CPU speedup. F12 C4 remains 8/8 complete and
safe. These results support C4 deployment preservation, not universal policy
equivalence.

## Limitations and immutable history

- Exact Group Shapley validates schema and local accuracy but coalition rows can
  still be physically off-manifold.
- No objective failed-event panel is claimed because frozen F12 lacks paired
  per-step public 29D failed traces.
- Semantic retention outside C4 is unavailable; historical behavior already
  shows C0--C2 retention loss.
- F11, F12, and F13 artifacts/statuses are unchanged. In particular, F13 A7
  gradient attribution remains unresolved; F14 does not repair it.

No actor was retrained, replaced, reconstructed, or selected in F14.

## Verification

- F14 targeted suite: **14 passed, 0 failed**;
- active full suite: **705 passed, 0 failed, 0 skipped** (458 warnings);
- artifact verifier: all provenance, row-count, sham, local-accuracy, frozen-source,
  figure-pair, and no-rerun checks PASS.

The first full-suite invocation omitted the repository root from `PYTHONPATH`
and stopped during collection with eight `experiments` import errors. Its log is
retained. The corrected historical overlay (`.:src:...`) produced the clean
705-test result above.
