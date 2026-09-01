# F9c — Robust Observation and Belief Calibration: Final Report

Answers the required final summary of the F9c specification (§43), plus the explicit
questions. Every number is taken from the frozen artifacts, not retyped from an
intermediate run.

---

## 1. Provenance

```
calibration seeds        6101–6108   (6,656 rows, 80 episodes)
final evaluation seeds   7101–7104   (3,328 frames, 40/40 episodes, rendered EXACTLY ONCE)
frozen config SHA256     359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e
runtime cache SHA256     fe425c55aadd45af88d072c256010f5bddcbb82d952669e8fa988bd70722526d
evaluation truth SHA256  26663ebb85ebd2ed9fd00ffc0903679b298b9333ef3e10f58ff253d7fd1e2ae9
frozen YOLO checkpoint   3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c  (unchanged)
F7 [ekf] block / Q       byte-identical to configs/oracle_ekf_v1.toml  (unchanged)
seeds never read         5101–5104 (F9b test data), enforced by config + a source-scan test
full test suite          251 passed, 0 failed, 0 skipped
```

The config was frozen **before** any 7101-series frame was rendered, with
`final_evaluation_seeds_not_yet_rendered: true` recorded in `artifacts/f9c_frozen_config.json`.

---

## 1b. Protocol amendment — declared

**The miss-likelihood floor was not in the pre-registered plan.** The plan specified the
existence problem would be handled by `P_D^eff(predicted observability)` plus the
outside-domain prediction-only rule, and named the global miss likelihood ratio as the
problem with `P_D^eff` as the lever.

**Protocol amendment, made after F9c calibration and before the frozen final evaluation:**
calibration on seeds 6101–6108 showed that no single-frame `P_D^eff` — however conditioned —
could represent temporally correlated detector misses. Measured `P_D` by range gives implied
miss likelihood ratios of 0.003 / 0.0075 / 0.028 (near/medium/far) against F9b's global
0.0233, so conditioning makes collapse *faster* in the near and medium bins. A
miss-likelihood floor derived solely from calibration-set run-length statistics was therefore
introduced, and frozen — together with every other parameter — before any 7101–7104 frame was
rendered.

There is **no test leakage**: the amendment was motivated entirely by calibration-set
evidence, and `artifacts/f9c_frozen_config.json` records
`final_evaluation_seeds_not_yet_rendered: true` alongside the config hash. The status is a
*pre-final protocol amendment*, not a post-hoc adjustment, but the method as executed differs
from the method as pre-registered and that difference is stated here rather than left for a
reader to discover.

---

## 2. Frozen parameters

```
gate                hard reject only, chi-square 2-DOF 99%      = 9.21034037197618
association         minimum-NIS, chi-square 2-DOF 99.9%         = 13.815510557964274
                    (deliberately LOOSER than the gate — see finding 6)
bias model          global_additive
                    b_r    = -0.02986607430110723 m
                    b_beta = +0.0012336629252072933 rad
                    per-range-bin was REJECTED: leave-one-seed-out improved by only -0.9%,
                    far below the pre-specified +10% bar
covariance          lambda_r = 9.96243043243885,  lambda_beta = 1.0
posterior floors    sigma_floor_r    = 0.02041790926900693 m
                    sigma_floor_beta = 0.012546331734068323 rad
                    from a two-level nested (seed, episode) variance-component fit
miss-likelihood     LR_floor = 0.37362469458201386
floor               = LR_nominal ** (1 / L_mean), L_mean = 4.0333 measured on 6101–6108,
                    LR_nominal = 0.018858
existence           P_S = 0.995 and P_birth = 0.005 UNCHANGED (never the lever)
                    P_D^eff per predicted observability class: center 0.9490 / mid_fov 0.9801
                    / edge_fov 0.9973 / outside_domain 0.5587
```

### Why the posterior floor exists

A variance decomposition of the F9a calibration data showed ~78% of range error is a
**per-episode/per-seed offset**, not per-frame noise. A Kalman filter averages away independent
noise; a constant offset within an episode survives averaging untouched. With `Q` frozen and
tiny, posterior variance keeps shrinking with more frames **no matter how much `R` is
inflated** — so `R` inflation alone provably cannot fix coverage on long episodes.

Hence two mechanisms with two jobs: `lambda` makes the innovation covariance honest so the
gate/association thresholds are calibrated; `sigma_floor` makes the *reported* uncertainty
honest by adding back the offset variance the filter structurally cannot see.

F9c calibration confirmed both structural predictions: range offset is **seed**-carried
(variance 3.36e-4 vs episode 3.48e-5), bearing offset is **episode**-carried (1.23e-4 vs
2.70e-5). A one-level fit would have divided the seed component by the episode count and
understated the floor roughly 3x.

---

## 3. Baseline A vs Robust B (seeds 7101–7104, 3,328 frames)

Baseline A = F9b-corrected YOLO → highest-confidence selection → frozen F7 EKF → frozen
existence. Robust B = candidates → F9c bias → association → gate → lambda-inflated R → frozen
F7 EKF → P_D^eff + I3 routing + I8 floor → posterior floor. Both run on the **same frames and
the same single YOLO inference per frame**.

| variable | metric | Baseline A | Robust B |
|---|---|---|---|
| range | bias | +0.016263 | **+0.001531** |
| | MAE | 0.019645 | **0.017292** |
| | RMSE | 0.025796 | **0.020242** |
| bearing | bias | -0.002203 | **+0.000395** |
| | MAE | 0.009278 | **0.008500** |
| | RMSE | 0.015904 | **0.013556** |
| range-rate | RMSE | 0.018436 | 0.019561 |
| bearing-rate | RMSE | 0.037537 | 0.038648 |

### Calibration

| variable | coverage_68 | coverage_95 | mean NLL | mean_std / RMSE |
|---|---|---|---|---|
| range — Baseline A | 0.2470 | 0.3881 | +31.085 | 0.191 |
| range — **Robust B** | **0.8522** | **0.9885** | **−2.439** | **1.279** |
| bearing — Baseline A | 0.4536 | 0.6957 | +1.136 | 0.315 |
| bearing — **Robust B** | **0.8513** | **0.9403** | **−2.881** | **1.009** |
| range-rate — Robust B | 0.8759 | 0.9686 | −1.165 | 2.660 |
| bearing-rate — Robust B | 0.8569 | 0.9281 | +10.853 | 0.793 |

Pre-registered acceptance bands were `coverage_68 ∈ [0.60, 0.76]` and
`coverage_95 ∈ [0.90, 0.98]`, with an anti-inflation guard `mean_std/RMSE ≤ 1.5`.

**Three pre-registered acceptance targets were missed:** range coverage_68 (0.852), range
coverage_95 (0.989) and bearing coverage_68 (0.851) all fall outside their bands, on the
conservative side. Bearing coverage_95 (0.940) is the one coverage value inside its band.

The posterior floor **deliberately increases** reported uncertainty — that is its purpose, and
saying otherwise would be misleading. The defensible claim is narrower: *the calibration
improvement was not achieved through excessive uncertainty inflation, because the pre-specified
anti-inflation guard remained satisfied* — `mean_std/RMSE` is 1.279 on range and 1.009 on
bearing against a ≤1.5 limit.

The accurate summary is therefore **conservative but not excessively inflated**, and
**no longer overconfident**, rather than "in band".

### Safety-relevant bias

`E[μ_r − r_GT]` went from **+0.016263 m to +0.001531 m**. Positive means the pedestrian is
believed *farther* than reality. F9b's persistent +0.01879 m is essentially eliminated.

---

## 4. Robustness and track behaviour

```
support (pre-specified minima in brackets)
  near   616 [100]     medium 671 [200]     far 1887 [200]     edge_fov 543 [50]   -> SATISFIED

miss breakdown (three disjoint classes, NEVER pooled)
  detector_miss_in_domain        55 frames   retention  Baseline A 18.2% (10/55)
                                                        Robust B   61.8% (34/55)
  gated_rejection                23 frames   retention  Robust B  100.0% (23/23)
  detector_miss_outside_domain               prediction step only, no likelihood applied

gate                accepted 3,033   rejected 23
duplicate frames    84               wrong association events 2
false track initializations          0        track deletions 8      recoveries 3
recovery after re-detection          1.0 frames (both systems)
```

### Localization-outlier counts — the three numbers, defined

Three different counts appear in the artifacts and they measure different things. Naming them
precisely:

```
9  frames   eligible + detected where BASELINE A's highest-confidence selection had GT IoU < 0.5
            -> this is outlier_impact.outlier_frame_count, the set the impact comparison uses
6  events   contiguous runs those 9 frames form, per episode
4  frames   eligible + detected where ROBUST B's ASSOCIATED candidate had GT IoU < 0.5
            -> this is robustness.localization_outlier_count
```

They are not inconsistent: 9 and 4 describe **different systems' selections**, and 6 counts
**events** rather than frames. Earlier phrasing ("4 outliers and n=9 impact frames") was
correct but under-specified.

### Gate confusion table (post-hoc, diagnostic only — no threshold was retuned)

Over the 3,056 frames where the gate ran on a candidate with a GT-comparable box:

| GT localization | gate accept | gate reject |
|---|---:|---:|
| IoU ≥ 0.5 (good) | 3,032 | 20 |
| IoU < 0.5 (outlier) | 1 | 3 |

```
outlier rejection sensitivity     3/4     = 0.750
good-measurement false rejection  20/3052 = 0.0066   (0.66%)
rejection precision               3/23    = 0.130
```

**Reading this honestly.** The 13% precision looks alarming in isolation but is what Bayes
requires: outliers are 4 in 3,056 frames, a base rate of 0.13%, so even a well-calibrated
gate will have most of its rejections be good measurements. The number that judges the gate's
calibration is the false-rejection rate — **0.66% against the ~1% a χ²₂ 99% gate should
produce**, which is close to nominal. The cost of a false rejection is also low by design:
under invariant I2 a rejected measurement still counts as a **detection** for existence, and
the EKF simply runs prediction-only for that frame.

### What Robust B actually did on the 9 Baseline-A localization failures

| Robust B behaviour | frames |
|---|---:|
| association selected no candidate at all → prediction-only | 7 |
| association selected a candidate, gate rejected it (IoU 0.27) → prediction-only | 1 |
| association selected and gate accepted (IoU 0.47, marginal) | 1 |

**8 of 9 gross localization failures resulted in prediction-only.** The one that got through
had IoU 0.47 — just under the 0.5 matching threshold, a marginal case rather than a gross
error.

So the robust layer *did* protect against localization failure, but through a mechanism
different from the one originally hypothesised: it protected by **refusing to use any box**
(association's own gate filtered 7, the innovation gate caught 1), not by selecting a better
box. See finding 1 for why this did not translate into a better outlier-impact RMSE.

### NIS diagnostics (measurements that reached the EKF)

| | count | mean | median | p95 | fraction > χ²₂ 95% |
|---|---|---|---|---|---|
| Baseline A | 3,083 | 1.5420 | 0.0452 | 5.6050 | 0.0331 |
| Robust B | 3,035 | 0.2393 | 0.0100 | 1.0273 | 0.0000 |

Robust B's zero exceedance is by construction — rejected measurements never reach the EKF.
The 23 rejected measurements are reported separately above.

---

## 5. Ablation (same cache, zero inference, seven configurations)

| row | range RMSE | coverage_68 | in-domain retention |
|---|---|---|---|
| baseline | 0.02580 | 0.2470 | 18.2% |
| + bias refit only | 0.02407 | 0.0752 | 18.2% |
| + innovation gate only | 0.02987 | 0.2588 | 18.2% |
| + temporal association only | 0.03776 | 0.2566 | 18.2% |
| + covariance calibration only | 0.02989 | 0.6403 | 18.2% |
| + conditional detection only | 0.02569 | 0.2589 | **61.8%** |
| **all combined** | **0.02024** | **0.8522** | **61.8%** |

Both endpoints were verified equal to their headline systems field-by-field to 1e-12.

---

## 6. Findings — including the ones that do not flatter the result

1. **Outlier handling: the mechanism worked, the outcome metric did not — and n=9 cannot
   settle it.** On the 9 Baseline-A localization-failure frames, Robust B ran prediction-only
   on 8 (7 filtered by association, 1 by the gate) and accepted just one marginal box at
   IoU 0.47. The gate's own statistics are near nominal: outlier rejection sensitivity 3/4,
   false-rejection rate 0.66% against the ~1% a 99% gate should produce.

   Yet the outlier-impact RMSE went the wrong way: Baseline A 0.0218 vs Robust B 0.0346. The
   likely reason is that a box with IoU < 0.5 is not necessarily wrong in *range*, while
   several consecutive prediction-only frames accumulate drift — Baseline A's EKF had already
   attenuated the raw measurement error (0.171 m RMSE) down to 0.0218 m on its own. So
   refusing an untrustworthy measurement can cost more than using it, when the filter was
   already suppressing it effectively. **With n=9 this is not a conclusion, and it is the
   single clearest gap in the gate's evidence.**

2. **Three coverage bands were missed**, all on the conservative side. Stated plainly rather
   than rounded into a pass. The anti-inflation guard passes, so both facts stand together.

3. **The heavy-tail explanation for the overshoot was tested and REFUTED.** Measured
   `z = error/σ` over n=3,214 gives std ≈ 0.79 and excess kurtosis ≈ **−0.43** — slightly
   *lighter*-tailed than Gaussian. The overshoot is a roughly uniform 25–30% over-sized
   posterior σ, not a tail effect. The remaining hypothesis — `σ_floor` calibrated on
   6101–6108 transferring wide to 7101–7104 — is consistent but **untested by this run**, and
   `τ_seed` from only 8 seeds is itself noisy.

4. **`conditional_detection` is inert through its per-class probabilities, on both branches.**
   The I8 floor (0.373625) dominates every implied miss likelihood ratio (center 0.050991,
   mid_fov 0.019882, edge_fov 0.002791). On the detected branch existence saturates ≥0.918287
   across all 3,122 detected rows regardless of class. The component contributes **only** via
   the I3 in/out-domain routing. Also, 39/42 track initializations classify `outside_domain`
   because the zero-prior default state has `y_forward = 0`.

5. **The components are NOT additively separable.** Gate-alone (0.02987) and
   association-alone (0.03776) are each *worse* than baseline (0.02580), while the combination
   is best. Per-row deltas must not be read as individual contributions. An untested
   hypothesis for association-alone: it selects by minimum NIS against
   `S = H P⁻ Hᵀ + λR`, and with covariance calibration off that `S` uses uninflated `R` — a
   covariance the calibration says is ~10x too small.

   **UPDATE — this was subsequently tested in F9d-C and the mechanism is now established,
   though not the one first proposed.** Holding the predicted state fixed and asking both
   scorings which candidate they would pick:

   - *Selection quality is not the mechanism.* Localization-outlier selections tie at 4/4
     between `λ=1` and the frozen `λ`, so `λ=1` does not make association pick worse boxes
     among the candidates that survive its gate.
   - *Abstention is.* Frames where association selected nothing: **42 at `λ=1` versus 22 at
     the frozen `λ`**, with one-sided abstention **20 versus 0** — fully asymmetric. The
     association-gate exceedance fraction moves the same way, 2.87% versus 1.70%.

   With `S = H P⁻ Hᵀ + λR` under-scaled, every NIS is inflated, so association's own
   chi-square gate returns *no selection* far more often, converting correction frames into
   prediction-only frames whose drift is what the ablation RMSE measured.

   **The `temporal_association_only` row must therefore not be read as "Mahalanobis
   association is intrinsically harmful".** The defensible statement is:

   > The temporal-association-only ablation is not evidence that Mahalanobis association is
   > intrinsically harmful. Its penalty is predominantly explained by increased association
   > abstention under the uncalibrated innovation covariance.

   This sharpens finding 5 rather than replacing it: association and covariance calibration
   are **mechanistically non-separable** — association requires a calibrated `S` — which is
   precisely what a one-component-at-a-time ablation is for surfacing.

6. **The innovation gate is structurally inert without temporal association.** With
   association off, `predicted_measurement` is forced to `None`, association returns
   `mode="initialization"`, and the gate branch only runs otherwise — there is no innovation
   to threshold. `innovation_gate_only` is *exactly* identical to an all-off diagnostic
   (RMSE `0.029874031173970667` in both). Separately, an earlier draft had the association and
   gate thresholds equal, which made the gate unreachable whenever association was on; the
   association gate was widened to 99.9% to give the two components distinct roles.

7. **Reproducibility evidence is n=2** full calibration runs under `domain_randomization =
   true`, agreeing to ~0.16% (`lambda_r` 9.9779 → 9.9624). The frozen values are one sample
   from a distribution.

8. **A P_FA scare that resolved.** A self-fit false-alarm rate of 31.2% was diagnosed as an
   artefact: all 65 flagged frames carry a GT range, so they are correct detections of a real
   pedestrian that the conservative `eligible_visible` rule excluded. The frozen F9b `P_FA`
   (measured properly from counterfactual renders with the Duckie hidden) was used instead.

9. **The final run crashed after rendering.** The render of 7101–7104 completed 100% (40/40
   episodes, zero early terminations) and wrote a hash-verified runtime cache; the metrics
   step then crashed on a CSV empty-string sentinel. **No metric had been computed, so the bug
   fix carries no risk of having been shaped by results.** Artifacts were reconstructed by
   replaying the cache — no simulator, no detector, no GPU, no re-render — through a shared
   row-builder used by both the render and replay paths so fidelity is structural.

---

## 7. Why no single-frame P_D conditioning could fix belief collapse

This is the central scientific result, and it is a negative one.

F9c calibration measured `P_D` by range: near 0.9969, medium 0.9925, far 0.9717. The implied
miss likelihood ratios are **0.003 / 0.0075 / 0.028** against F9b's global **0.0233** — so
conditioning on range makes collapse *faster* in the near and medium bins. The apparent
"edge_fov beats center" inversion in the FOV-conditioned fit is a **range confound**: `center`
is 77.9% far-range while `edge_fov` is 1.9% far-range, and within each range bin detection
rates are flat (0.946–1.000).

The mis-specification is therefore **not the value of `P_D` but the independence assumption
behind it**. Real detector misses arrive in bursts — F9b's mean run was 7.125 frames, F9c's
4.033 — and an independent-Bernoulli likelihood cannot express that at any `P_D`.

That is what motivated the miss-likelihood floor: a burst of correlated misses carries roughly
the evidence of **one** independent miss, not `L_mean` of them. `LR_floor = LR_nominal **
(1/L_mean)` spreads the nominal single-miss evidence across the run's expected length, so a
run of typical length reproduces exactly the nominal evidence while longer runs still decay.
Derived from measured run-length statistics, never tuned to a retention target.

---

## 8. Criteria tally — reported in two layers

These are two different kinds of criterion and pooling them would flatter the result. Both
layers are reported.

### Layer 1 — the specification's 17 functional PASS criteria

```
15  met
 2  insufficient evidence
 0  failed
```

The two insufficient:
- **criterion 5** (gross localization outliers have reduced influence) — mechanism verified
  (8/9 failures went prediction-only) but the outcome metric went the wrong way at n=9
- **criterion 10** (long absence still allows decay) — never exercised; the longest natural
  miss run in the final data was 10 frames, so the ≥20-consecutive-miss checkpoint was never
  reached

### Layer 2 — the pre-registered statistical acceptance targets

```
coverage targets     3 of 4 OUTSIDE band, all on the conservative side
                       range   coverage_68  0.852  vs [0.60, 0.76]   outside
                       range   coverage_95  0.989  vs [0.90, 0.98]   outside
                       bearing coverage_68  0.851  vs [0.60, 0.76]   outside
                       bearing coverage_95  0.940  vs [0.90, 0.98]   INSIDE
anti-inflation guard PASSED   mean_std/RMSE 1.279 range, 1.009 bearing  (limit 1.5)
accuracy guard       PASSED   robust RMSE below 1.15x baseline on range and bearing
support guards       PASSED   near 616 / medium 671 / far 1887 / edge_fov 543
```

### Overall gate

```
LIMITED
```

**On the relationship between the two layers.** The coverage bands were added during planning
to operationalise functional criterion 8 ("not achieved by absurdly inflating uncertainty").
That purpose is independently guarded by `mean_std/RMSE ≤ 1.5`, which passes. Functional
criteria 6 ("no longer severely overconfident") and 7 ("coverage materially improves") are
both met. The bands are symmetric while the risk is asymmetric — an over-conservative
posterior is safe, an overconfident one is dangerous — so a symmetric band was arguably the
wrong instrument for this quantity. That is an argument about instrument design, offered for
the reader to weigh; it is **not** grounds for treating a missed target as met, and the misses
are reported as misses.

---

## 9. The eight questions, answered

1. **Did robust observation handling reduce localization-outlier impact?**
   **The mechanism fired; the outcome metric did not confirm it; n=9 cannot settle it.**
   Robust B ran prediction-only on 8 of the 9 Baseline-A localization failures, and the gate's
   own statistics are near nominal (sensitivity 3/4, false-rejection 0.66% vs ~1% expected).
   But belief RMSE on those frames was worse than Baseline A (0.0346 vs 0.0218), plausibly
   because prediction-only drift cost more than the mislocalized measurements did once the EKF
   had attenuated them. Not a pass.

2. **Did temporal association improve duplicate frames?**
   **No — on the evidence available it did the opposite.** On the 3,098 frames where both
   selections have a GT-comparable box, the highest-confidence selection produced 2
   localization outliers and temporal association produced 4. Association **rescued 0 frames**
   and **introduced 2**. Restricting to the 80 duplicate frames — the exact case it was built
   for — highest-confidence produced 0 outliers and association produced 2.

   This is consistent with the ablation, where association-alone worsens range RMSE (0.03776
   vs 0.02580). The one thing association demonstrably did well was *refuse* bad frames: it
   selected nothing on 7 of the 9 Baseline-A localization failures. So its value in this data
   came from abstention, not from better selection.

   A labelled hypothesis for the degradation (untested): association scores candidates by
   minimum NIS against `S = H P⁻ Hᵀ + λR`, and the selection quality therefore depends on
   `λ` being right. This is cheap to test on the existing cache and is listed in §10.

3. **Did range uncertainty become realistically calibrated?**
   **Yes, and then some.** coverage_68 0.247 → 0.852, coverage_95 0.388 → 0.988, NLL +31.08 →
   −2.44. It overshoots the pre-registered bands on the conservative side, with
   mean_std/RMSE = 1.279, confirming that the conservative coverage was not achieved through
   excessive uncertainty inflation. The posterior floor raises uncertainty by design; the
   guard establishes that it did not do so excessively.

4. **Did persistence improve belief through natural misses?**
   **Yes.** In-domain natural-miss retention 18.2% → 61.8% (10/55 → 34/55), clearing the ≥60%
   criterion. Gated-rejection retention is 100% (23/23) — the direct payoff of separating
   detection evidence from kinematic acceptance.

5. **Did separating detection evidence from kinematic acceptance prevent the gate from
   worsening existence collapse?**
   **Yes, measurably.** 23 frames had a detection whose bbox the gate rejected. All 23 retained
   an active belief. Had those been scored as misses — as an earlier design did — each would
   have applied the miss likelihood ratio to a frame where the detector *did* see the
   pedestrian, and the gate would have accelerated collapse rather than preventing it.

6. **Was RMSE materially worsened to achieve calibration?**
   **No — it improved.** Range RMSE 0.02580 → 0.02024 (21.5% lower), bearing 0.01590 → 0.01356
   (14.8% lower), and range bias nearly eliminated. Rate RMSEs are marginally worse
   (range-rate 0.0184 → 0.0196, bearing-rate 0.0375 → 0.0386).

7. **Is EKF + robust observation handling sufficient for Version-1 POMDP?**
   For accuracy and uncertainty, yes. For the two unproven criteria, not demonstrated. The
   estimator is accurate, well-behaved and honestly uncertain, but its outlier robustness and
   its long-absence behaviour have no supporting evidence.

8. **Is the system control-ready?**
   **Recommended: LIMITED.** Not because the calibration overshoots — a conservative posterior
   is the safe direction — but because criteria 5 and 10 have no evidence, and those are
   precisely the conditions under which a stop policy must act: a gross localization error at
   close range, and a pedestrian that stays out of view for a long time.

---

## 10. What a follow-up gate would need to close

- **Outlier robustness**, with enough GT-labelled localization outliers to conclude anything.
  4 outliers and n=9 impact frames is not a test.
- **Long-absence decay**, with scenarios that actually produce ≥20-frame absences.
- **Whether the σ_floor overshoot is a transfer effect**, testable by re-fitting the floor on
  a held-out seed group and comparing.
- **Whether association degrades alone because of the uninflated `S`** — currently a labelled
  hypothesis, cheap to test on the existing cache.

Reward, stop logic, SAC/TD3/PPO were **not** implemented, per the specification's instruction
to stop and report first.
