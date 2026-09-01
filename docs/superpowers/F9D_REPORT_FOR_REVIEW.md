# F9d Control-Readiness Evidence Closure — Report for Review

Date: 2026-08-09  
Frozen F9d config SHA256: `7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6`  
Frozen F9c config SHA256: `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`

## Decision

**F9d classification: `LIMITED`.** Long-absence evidence (B) passed, but
natural localization-outlier evidence (A) did not clear its pre-registered
50-frame minimum. The final run produced 43 outlier frames, 29 independent
events, and events on all four seeds. The outcome on those 43 frames strongly
favoured Robust B, but it remains descriptive because support failed. The
cache-only association diagnostic (C) cannot change the classification.

No F9c estimator parameter was modified. Seeds 8201–8204 and 8301–8304 were
each rendered exactly once and must not be rendered again.

## Functional criteria

| Area | Functional result |
|---|---|
| Frozen estimator | PASS — F9c config and YOLO checkpoint hashes unchanged |
| Runtime boundary | PASS — inference/filter update precedes privileged evaluation reads |
| A outlier stress | COMPLETE, but statistically under-supported |
| B1 natural out-of-domain absence | PASS |
| B2 controlled in-domain dropout | PASS |
| B3 controlled target removal | PASS |
| C cache-only diagnostic | COMPLETE; no render, detector, or estimator change |
| Strict artifact verification | PASS — 15/15, no skip |

## A — Natural localization-outlier stress

Final seeds: `8201, 8202, 8203, 8204`; 7,658 rows. The support decision was
read before any RMSE value.

| Support component | Observed | Minimum | Result |
|---|---:|---:|---|
| Natural outlier frames | 43 | 50 | FAIL |
| Contiguous outlier events | 29 | 12 | PASS |
| Seeds with an event | 4 | 3 | PASS |

The 43 frames are above the hard insufficient floor of 30, but all three
support components were required. Therefore the primary verdict is
`INSUFFICIENT_EVIDENCE`, not PASS.

Descriptive outcome on the 43 supported frames:

| Range belief | Bias (m) | MAE (m) | RMSE (m) | Max abs (m) |
|---|---:|---:|---:|---:|
| Baseline A | 0.04641 | 0.04682 | 0.06827 | 0.16434 |
| Robust B | 0.00520 | 0.01194 | 0.01648 | 0.07475 |

Robust/Baseline RMSE ratio was `0.2414`. Robust B recovered within the frozen
0.10 m error definition in one frame for every recoverable event (28/29;
the remaining event ended with the episode). Baseline A recovered in 28/29,
mean 1.39 frames, maximum 8.

Gate confusion, using the same associated-IoU definition as F9c:

| GT localization | Gate accept | Gate reject |
|---|---:|---:|
| IoU >= 0.5 | 5,484 | 83 |
| IoU < 0.5 | 16 | 0 |

This diagnostic shows that Robust B's favourable outcome did not come from
innovation-gate rejection of the 16 associated outliers. Association and
temporal smoothing are plausible contributors, but F9d does not introduce a
new post-hoc mechanism claim.

There were 46 documented early `invalid-pose` terminations. The development
projection predicted 119.5 outlier frames, but the disjoint final seeds
yielded 43. That seed/termination sensitivity is the exact reason the support
minimum remains binding.

## B — Long absence

Final seeds: `8301, 8302, 8303, 8304`; 7,260 rows; zero episode warnings.

| Kind | Runs >=20 | Minimum | Runs >=40 | Minimum | Support |
|---|---:|---:|---:|---:|---|
| B1 natural out-of-domain | 20 | 12 | 15 | 4 | PASS |
| B2 controlled dropout | 24 | 12 | 12 | 4 | PASS |
| B3 controlled removal | 16 | 12 | 12 | 4 | PASS |

### B1 — natural out-of-domain routing

All 20 events kept an active track and none deleted it. At frame 40 the mean
existence probability was `0.83741` (range `0.83448–0.84127`), above the
frozen 0.50 floor. GT-out-of-FOV vs runtime observability was 1,114
`outside_domain` frames and one `edge_fov` frame.

The prediction-only recurrence was checked only on frames satisfying all
three conditions: GT out of FOV, runtime `outside_domain`, and no Duckie
detection. This matters because a positive detection at the visibility
boundary applies detection evidence and is not prediction-only. Across 20
pure runs, maximum absolute deviation from
`p_n = 0.5 + (p_0 - 0.5) * 0.99^n` was `4.44e-16`.

### B2 — controlled in-domain detector dropout

Mean existence probability by suppressed-frame checkpoint:

| Frame | N | Mean P(exists) |
|---:|---:|---:|
| 1 | 24 | 0.986718 |
| 5 | 24 | 0.478222 |
| 10 | 24 | 0.009507 |
| 20 | 24 | 0.002989 |
| 30 | 12 | 0.002988 |
| 40 | 12 | 0.002988 |

All 24 runs were below 0.10 by frame 20. All tracks deleted during the
dropout window and all 24 recovered one frame after a real detection
returned. This is correct forgetting of an unsupported in-domain track, and
also a lost true positive because the real pedestrian remains present in B2;
both sides of that safety trade-off must be retained.

### B3 — controlled target removal

All 16 events deleted their tracks; no recovery occurred or was expected
because the pedestrian was removed from both rendered RGB and privileged
existence truth. P(exists) followed the same evidence-decay shape as B2,
reaching mean `0.002989` at frame 20. B3 supports the narrower claim that the
frozen estimator can forget an object that genuinely no longer exists.

## C — Cache-only association diagnostic

The 3,328-frame frozen F9c runtime cache was reused; no simulator, detector,
or estimator rerun occurred.

**C1, lambda=1 vs frozen lambda.** Selections agreed on 3,057/3,080 frames
and differed on 23. Among paired differing frames, lambda=1 won 0 and frozen
lambda won 3; both produced four localization outliers. The pre-registered
selection hypothesis is therefore **REFUTED**. A separate abstention
diagnostic was **SUPPORTED**: lambda=1 selected nothing on 42 frames vs 22
for the frozen lambda.

**C2, highest confidence vs min-NIS at frozen lambda.** On 81 duplicate
frames, selections agreed on 53 and differed on 28. Paired wins tied 12–12;
localization outliers were 1 vs 2. Under the pre-registered conjunctive rule,
min-NIS is **NOT INFERIOR**, but the comparison is weakly powered and does
not establish superiority.

## Closure of F9c open evidence

| F9c open item | F9d result |
|---|---|
| Gross localization-outlier impact | Still open: favourable descriptive result, but 43 < 50 frames |
| 20-consecutive-miss existence behaviour | Closed and passed by B1/B2; B3 adds genuine-removal evidence |

## Strict verifier output

Executed before this report was written:

```text
[PASS] frozen_config_loads: config_sha256=7bbe6525c24e294b55a46808301249633236658814e906a68d0d804d5e8a8ca6
[PASS] frozen_hashes_and_claims: F9d, F9c, checkpoint, seed, and minimum claims verified
[PASS] outlier_yield_probe: projected_frames=119.5
[PASS] absence_yield_probe: B1/B2/B3 support true; B2 GT-invisible contamination=0
[PASS] association_diagnostic: cache-only C1/C2 diagnostic present
[PASS] outlier_csv_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_outlier_stress.csv
[PASS] outlier_metrics_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_outlier_metrics.json
[PASS] outlier_runtime_cache_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_outlier_runtime_cache.npz
[PASS] outlier_truth_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_outlier_evaluation_truth.npz
[PASS] absence_csv_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_absence_stress.csv
[PASS] absence_metrics_present: /home/pannntastic/aivnv/duckie-pomdp/artifacts/f9d_absence_metrics.json
[PASS] outlier_metrics_hashes: outlier metrics bound to frozen F9d/F9c hashes
[PASS] absence_metrics_hashes: absence metrics bound to frozen F9d/F9c hashes
[PASS] outlier_csv_nonempty: 7658 data rows
[PASS] absence_csv_nonempty: 7260 data rows
```

Exit code: `0`; 15 PASS, 0 FAIL, 0 SKIP.

## Regression tests

The complete active suite was run through the documented Duckietown virtual
environment: **351 passed, 0 failed, 0 skipped**. Pytest reported 264
dependency/runtime warnings; none changed the gate result.

## Final classification

The pre-registered rule requires both A and B to pass. B passes, A does not
clear support, and C is diagnostic-only. F9d therefore remains **LIMITED**.
This is an evidence limitation, not an estimator refit request. No stop-state
logic, reward, SAC, TD3, PPO, or other F10 work begins here.
