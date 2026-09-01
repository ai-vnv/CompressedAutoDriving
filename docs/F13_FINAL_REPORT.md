# F13 Final Report — Explain Again and Compression Diagnostics

## 1. Frozen models

- Original Belief-PPO: 29-256-256-2 FP32,
  `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`.
- A7 / B-PPO-PDQD: 29-64-64-2 static INT8,
  `f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e`.

The public semantic 29D contract, normalization, perception, belief filters,
and physical action mapping were unchanged.

## 2. Attribution-surrogate integrity

The exact pre-conversion A7 QAT state does not exist in the frozen F12
artifacts. Standard IG was not applied to the deployable quantized graph, and
no approximate surrogate was constructed. A7 gradient attribution is therefore
**BLOCKED** and semantic attribution preservation is **UNRESOLVED**.

Original R004 remains the frozen reference: lane dominates normal/curve and
post-stop control, pedestrian belief dominates pedestrian-relevant states, and
stop belief is largest for stop-required velocity. F13 makes no claim that A7
preserves or changes those attribution rankings.

## 3. Counterfactual functional sensitivity

Direct actor evaluation is valid without gradients. Across 4,400 public R004
states, sham was exactly zero and numerical replay passed. Pedestrian-removal
velocity response was preserved (`+0.3037` Original vs `+0.2749 m/s` A7).
Lane-centering yaw response passed the frozen magnitude rule. Stop-removal
velocity response kept its direction but was smaller (`+0.2188` vs
`+0.1770 m/s`), producing normalized mean drift `0.1045` against limit `0.10`.
Pedestrian-removal yaw response was also materially lower as an auxiliary
finding.

Counterfactual functional sensitivity: **PARTIALLY PRESERVED**.

## 4. Explanation-guided C4 probe

On four accepted paired exploratory seeds, both policies achieved 4/4
completion, 0 collision, 0 unsafe episodes, 4/4 stop completion, 0 stop
violation, 4/4 restart, 0 lane failure, and 0 invalid pose. Minimum pedestrian
clearance was 0.4961 m for Original and 0.4904 m for A7. No A7-only failure was
observed; confirmatory seeds remained unopened.

Behavioral C4: **PRESERVED** for this fixed diagnostic set.

## 5. What compression changed

The supported change is **Level 2 functional-sensitivity drift**, principally a
slightly reduced release in commanded velocity when stop semantics are removed.
This did not become Level 3 action drift or Level 4 closed-loop failure in the
accepted paired C4 probe. Attribution drift itself cannot be assessed because
the exact differentiable A7 source was not persisted.

## 6. Final classifications

| Axis | Classification |
|---|---|
| Behavioral C4 | **PRESERVED** |
| Semantic explanation structure | **UNRESOLVED** |
| Counterfactual functional sensitivity | **PARTIALLY PRESERVED** |
| Overall F13 | **LIMITED** |

Overall is LIMITED because behavior remains intact but the semantic-attribution
comparison is unavailable and a small functional sensitivity drift is present.
This result does not trigger retraining or repair.

## 7. Historical integrity and stop rule

F11 remains R002 LIMITED, R002b/R003/R004 PASS, R006 FAILED, R007 BLOCKED.
F12 remains PASS for C4-only deployment. No earlier result was rewritten.

F13 stops here. No A7 repair, pruning change, quantization change, additional
distillation, reward/perception/belief modification, or F14 optimization was
started.

## 8. Regression tests

The active repository suite completed with **691 passed, 0 failed, 0 skipped**
and 443 warnings. An earlier collection attempt omitted the repository root
from `PYTHONPATH` and produced eight `experiments` import errors; that failed
command log is retained separately and is not the active-suite witness.
