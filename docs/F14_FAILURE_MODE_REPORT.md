# F14 Compression Failure-Mode Report

## Descriptive hierarchy

| Variant | L0 integrity | L1 attribution | L2 sensitivity | L3 fidelity | L4 C4 control |
|---|---|---|---|---|---|
| A0 | PASS | No diagnosed drift | No diagnosed drift | Preserved | Preserved |
| A1 | PASS | Shifted | Shifted | Drifted | Not preserved |
| A2 | PASS | Shifted | Preserved | Preserved | Preserved |
| A3 | PASS | Partial | Partial | Drifted | Preserved |
| A4 | PASS | Partial | Preserved | Drifted | Preserved |
| A5 | PASS | Shifted | Shifted | Drifted | Not preserved |
| A6 | PASS | Shifted | Partial | Preserved | Preserved |
| A7 | PASS | Shifted | Shifted | Preserved | Preserved |

## Failed branches

### A1: pruning only

A1 changed the top velocity group from Stop to Lane in nominal/lane-curve
states, and pedestrian-relevant yaw changed from Pedestrian to Lane. Its
pedestrian and stop counterfactual effects also changed materially. Frozen F12
evidence records 0% completion, 25% lane failure, 12.5% unsafe episodes, and
only 1.62 m mean progress. This is a mixed semantic, functional, action, and
closed-loop degradation—not merely a calibration error.

### A5: pruning plus PTQ

A5 retained essentially the same explanation/fidelity failure pattern as A1:
1/10 semantic cells and 3/8 functional cells were preserved, with 0% C4
completion. This supports a pruning-dominated branch failure with quantization
failing to repair it; it does not establish that quantization caused the
failure.

### A3 and A4: fidelity-gate failures without C4 failure

A3 and A4 failed the frozen F12 action-equivalence gate while completing all C4
episodes safely. Both semantic results are PARTIAL; functional sensitivity is
PARTIAL for A3 and PRESERVED for A4. They are
examples of explanation/action drift without demonstrated C4 behavioral
failure on the tested scenarios.

## Objective failed-event trace limitation

The frozen F12 failed-ablation artifacts retain episode-level summaries but no
provenance-bound per-step public 29D trace for an objective paired event window.
F14 therefore marks the requested failed-ablation trace **UNRESOLVED** rather
than cherry-picking a frame or rerunning historical evaluations.

## Pruning frontier and retention

All existing 192, 128, 96, and 64-width P/PD actors were provenance-verified and
received a lightweight one-draw diagnostic on the same 500 states. No missing
checkpoint was reconstructed and no post-hoc model selection was performed.

Historical A7 retention remains **NOT PRESERVED** outside the designated C4
scope: C0--C2 completion was 0%, whereas C3/C4 completion was 100%. Semantic
retention diagnosis is **UNRESOLVED** because no compatible saved per-step
public 29D C0--C3 rows exist. F12 remains PASS only for its frozen C4 deployment
scope.
