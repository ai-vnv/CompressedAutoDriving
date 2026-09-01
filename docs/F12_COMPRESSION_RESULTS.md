# F12 Belief-PPO Compression Results

## Classification

**F12: PASS for the designated C4 combined-scenario deployment scope.**

The selected 64×64 INT8 A7 actor preserved frozen C4 behavior. Cross-curriculum
retention exposed loss of C0--C2 driving competence, so this PASS must not be
generalized beyond C4. No post-hoc candidate replacement, threshold change,
retraining, or final-holdout reuse was performed.

## Frozen model and protocol

- Original checkpoint SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`
- Selection-time candidate: A7 / B-PPO-PDQD, 29→64→64→2, INT8
- A7 SHA256: `f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e`
- Quantization: PyTorch static eager x86, qint8 per-channel weights and quint8 per-tensor activations
- Actor-only scope; MobileNet, YOLO, belief filters, normalization, and 29D ordering are unchanged

## Efficiency

- Parameters: 73,986 → 6,210 (91.61% reduction; 11.91×)
- Actor file: 299,667 → 36,880 bytes (87.69% reduction)
- Logical parameter memory: 295,944 → 7,640 bytes (38.74×)
- CPU batch-1 median: 42.77 → 14.07 µs (3.04× speedup)
- End-to-end visuomotor latency was not claimed; perception is unchanged and remains the dominant deployment cost.
- Process peak-RSS delta was below measurement resolution for these tiny actors and is not used as an efficiency claim.

## Action fidelity

On the untouched final public-state dataset (17,600 rows), A7 passed all frozen
overall and phase-wise gates:

- v MAE/RMSE/P95: 0.002228 / 0.003404 / 0.006118 m/s
- ω MAE/RMSE/P95: 0.035594 / 0.050869 / 0.085389 rad/s
- ω sign disagreement above the 0.2 rad/s deadband: 0.012750%

## Final C4 closed loop

| Metric | Original A0 | Compressed A7 |
|---|---:|---:|
| Completion | 100% | 100% |
| Collision | 0% | 0% |
| Unsafe episode | 0% | 0% |
| Stop violation | 0% | 0% |
| Lane failure | 0% | 0% |
| Restart | 100% | 100% |
| Minimum pedestrian clearance | 0.450 m | 0.480 m |
| Mean progress | 7.207 m | 7.272 m |

## Retention failure

| Stage | A0 completion | A7 completion | A0 lane failure | A7 lane failure | A0 progress | A7 progress |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 100% | 0% | 0% | 0% | 5.373 m | 0.952 m |
| C1 | 100% | 0% | 0% | 50% | 7.667 m | 1.156 m |
| C2 | 100% | 0% | 0% | 100% | 7.452 m | 0.964 m |
| C3 | 100% | 100% | 0% | 0% | 7.250 m | 7.274 m |
| C4 | 100% | 100% | 0% | 0% | 7.207 m | 7.272 m |


A7 remains competent on C3/C4 but loses C0--C2. The exact next technical need is
multi-stage/retention-aware distillation with frozen rehearsal coverage, followed
by a new preregistered compression gate. That recovery was not attempted in F12.

## Scientific answers

1. Structured pruning produced the largest parameter/file reduction.
2. Pruning without distillation caused the largest fidelity and control loss.
3. At width 64, distillation reduced pruning-only v/ω MAE by about 96.5%/94.7% on selection states and restored C4 completion from 0% to 100%.
4. INT8 PTQ was sufficient for A6 on C4, but not for unpruned A3 under the frozen fidelity gates.
5. QAT improved normalized A6 action MAE by 10.654%, meeting the frozen selection threshold, but did not prevent cross-stage forgetting.
6. The selected actor achieved 91.61% parameter reduction, 87.69% file reduction, and 3.04× actor-only CPU speedup.
7. Safety-critical C4 behavior was preserved; broad C0--C4 behavior was not.

## Stop rule

F12 stops here. Explain-again, perception compression, post-hoc recovery, and policy
optimization were not started.
