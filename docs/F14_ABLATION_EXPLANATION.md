# F14 Ablation Explanation

## Scope and method

F14 compares the frozen A0--A7 actors on the same 500 development public-policy
states: 100 states in each of the five frozen phases. Exact six-player Group
Shapley uses all 64 coalitions, six independent draws, and four complete
same-phase cross-seed reference rows per draw. The semantic groups are Lane,
Ego, StopLine, Pedestrian, Stop, and PreviousAction. Semantic counterfactuals
reuse the frozen R003 operators. Neither analysis uses simulator truth.

The A0-only calibration passed before any A1--A7 explanation was read. Its
frozen preservation thresholds are Spearman >= 0.942857, group-share L1 <=
0.130886, unchanged top group, top-two Jaccard >= 0.633333, and at least 8/10
preserved phase/action cells. These deliberately strict thresholds describe
preservation relative to A0 reference variability; they are not behavioral
safety thresholds.

## Same-state findings

| Variant | Operation | Semantic attribution | Cells | Counterfactual sensitivity | Cells | F12 fidelity | F12 C4 behavior |
|---|---|---|---:|---|---:|---|---|
| A0 | Original FP32 | PRESERVED | 10/10 | PRESERVED | 3/3 | Reference | Preserved |
| A1 | Pruning | SHIFTED | 0/10 | SHIFTED | 1/3 | Failed | Not preserved |
| A2 | Pruning + distillation | SHIFTED | 0/10 | PRESERVED | 3/3 | Passed | Preserved |
| A3 | INT8 PTQ | PARTIAL | 7/10 | PARTIAL | 2/3 | Failed | Preserved |
| A4 | INT8 QAT/KD | PARTIAL | 5/10 | PRESERVED | 3/3 | Failed | Preserved |
| A5 | Pruning + INT8 PTQ | SHIFTED | 1/10 | SHIFTED | 1/3 | Failed | Not preserved |
| A6 | Pruning + distillation + PTQ | SHIFTED | 0/10 | PARTIAL | 2/3 | Passed | Preserved |
| A7 | Pruning + distillation + QAT/KD + INT8 | SHIFTED | 0/10 | SHIFTED | 1/3 | Passed | Preserved |

## Operation-level diagnosis

- **A0 to A1:** direct 64x64 pruning changed attribution, functional sensitivity,
  action fidelity, and closed-loop behavior together. It is not supported to
  describe the removed neurons as useless.
- **A1 to A2:** distillation recovered all three preregistered primary counterfactual cells,
  action fidelity, and C4 behavior. It did not restore A0's exact Group Shapley
  share/ranking structure under the frozen threshold. This is functional and
  behavioral recovery without semantic-attribution equivalence.
- **A0 to A3:** PTQ preserved C4 completion but produced a frozen action-fidelity
  failure and partial attribution/counterfactual drift. Thus the effect was not
  purely numerical, although it did not become a C4 failure.
- **A3 to A4:** QAT/KD improved counterfactual preservation from 2/3 to 3/3 and
  reduced historical action error, but did not pass the frozen fidelity gate or
  restore full semantic structure.
- **A1 to A5:** A5 closely retained the A1 failure pattern. Quantization did not
  rescue direct-pruning damage and added no evidence that pruning was harmless.
- **A2 to A6:** PTQ retained C4 behavior and acceptable F12 fidelity after
  distillation, while counterfactual preservation fell from 3/3 to 2/3.
- **A6 to A7:** the second fake-quant/KD stage improved F12 action fidelity but
  did not restore the A0 Group Shapley structure; only 1/3 primary
  counterfactual cells passed all frozen diagnostic thresholds.

These levels are co-occurring diagnostics. F14 does not claim that attribution
drift caused any closed-loop failure.
