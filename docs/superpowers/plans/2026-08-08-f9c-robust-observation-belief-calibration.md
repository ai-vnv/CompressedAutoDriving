# F9c — Robust Observation and Belief Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the LIMITED F9b belief estimator into an accurate *and calibrated* probabilistic pedestrian belief by refitting the YOLO observation model, representing unmodelled per-episode localization bias as honest uncertainty, gating and associating measurements on innovation statistics, and conditioning detection probability on predicted observability — without touching the frozen F7 dynamics.

**Architecture:** The frozen F7 EKF stays exactly as-is (state, `F`, `Q`, ego-motion compensation, Jacobians, polar transform). Everything F9c changes sits *around* it: perception now emits a candidate list instead of a pre-selected detection; a new belief-layer coordinator performs temporal association → innovation gating → EKF correction → existence update; and a new reporting boundary adds a variance floor before the belief leaves the estimator. Four calibrated components (`A` bias refit, `B` covariance calibration, `C` robust measurement handling, `D` conditional detection model) are independently switchable for ablation.

**Tech Stack:** Python 3.10, NumPy (no new dependencies), Ultralytics 8.4.116 (frozen checkpoint, inference only), gym-duckietown 6.2.0 source overlay, pytest, TOML config via `tomllib`/`tomli`.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Environment (exact, do not vary):**

The project lives in WSL (`Ubuntu-Baru`) but the agent harness runs on Windows. The venv is a Linux venv and **cannot** be invoked directly from a Windows shell. Every command in this plan must be wrapped:

```bash
wsl.exe -d Ubuntu-Baru -- bash -lc '<command>'
```

Inside that shell:

```bash
cd /home/pannntastic/aivnv/duckie-pomdp
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=8123
export CUDA_VISIBLE_DEVICES=0
PY=/home/pannntastic/aivnv/duckie/.venv/bin/python
```

Full suite command (must stay green after every task), as one line:

```bash
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

File **edits** use the Windows UNC path `\\wsl.localhost\Ubuntu-Baru\home\pannntastic\aivnv\duckie-pomdp\...`; only **execution** goes through `wsl.exe`. Note that `mkdir -p` on a relative path fails from Git Bash under this UNC working directory — create directories with Python's `os.makedirs` or inside the WSL shell.

Baseline verified 2026-08-08 before Task 1: **108 passed, 0 failed, 0 skipped, 260 warnings**.

**This project is not a git repository and will not become one** (operator decision). Task checkpoints are ledger entries plus a green suite. Per-task rollback points come from the SDD workspace snapshot tool, not from git:

```bash
python .superpowers/sdd/2026-08-08-f9c-robust-observation-belief-calibration/snap.py save|diff|restore
```

**This directory is NOT a git repository.** `git rev-parse` fails. Do not invent commit steps. Each task ends with a **Checkpoint** step: append a dated entry to `IMPLEMENTATION_NOTES.md` under a new `## F9c` section and re-run the full suite. If the operator later runs `git init`, replace Checkpoint steps with commits — do not run `git init` yourself.

**Freeze boundary (hard):**

| FROZEN — must not change | UNFROZEN for F9c calibration | KEEP FIXED INITIALLY |
|---|---|---|
| `artifacts/yolo_v1/best.pt` (SHA `3d4f816d…d7482c`) | measurement bias `b_r`, `b_β` | `survival_probability = 0.995` |
| EKF state definition `[x_left, y_forward, v_left, v_forward]` | `R` inflation `λ_r`, `λ_β` | `birth_probability = 0.005` |
| `compensated_transition` / `F` / offset | posterior variance floor `σ_floor,r`, `σ_floor,β` | `prior_probability = 0.50` |
| `Q` (`position_process_std…=0.001`, `velocity_process_std…=0.005`) | innovation gate type + threshold | detector operating point (conf 0.10, NMS 0.70, imgsz 480) |
| `measurement_function`, `measurement_jacobian` | association rule + association gate | matching IoU 0.50 (evaluation only) |
| `cartesian_to_polar_moments` | effective detection probability model `P_D^eff` | |
| all remaining `[ekf]` keys in `configs/oracle_ekf_v1.toml` | existence activate/delete/init thresholds | |

`P_S` is **already implemented** (`src/duckie_pomdp/belief/existence_filter.py:35-38`) and is **not** the lever. Do not tune it in F9c. If a later gate proves it must change, that is a separate decision.

**Seeds (pre-specified, disjoint):**

```text
detector train      1101–1106
detector val        2101–2102
detector test       3101–3102
F9a calibration     4101–4104   (used, frozen)
F9b final test      5101–5104   (FROZEN — TEST ONLY, never read during F9c development)
F9c calibration     6101–6108   (8 seeds — see §"Why 8 seeds" below)
F9c final           7101–7104
```

Reading, plotting, or fitting anything on 5101–5104 or 7101–7104 during development is a gate failure. `f9c_protocol` must enforce disjointness programmatically (Task 1).

**Runtime purity:** the runtime chain may consume only `front_rgb`, detector output, projected candidate measurements, predicted EKF state/covariance, and camera calibration. It must never consume `privileged.*`, `sample_object_silhouettes`, `eligible_visible`, GT range/bearing, or GT IoU. Offline *calibration* may use GT for statistics (as F9a did). Evaluation may use GT only after the runtime step has completed. Task 13 adds explicit leakage tests.

**Pre-specified acceptance bands (locked BEFORE the final run — do not renegotiate after seeing 7101–7104):**

```text
coverage_68  ∈ [0.60, 0.76]      for range (primary), bearing
coverage_95  ∈ [0.90, 0.98]      for range (primary), bearing
anti-inflation guard: mean_predicted_std ≤ 1.5 × RMSE   (per variable)
accuracy guard:       robust range RMSE ≤ 1.15 × Baseline-A range RMSE
                      robust bearing RMSE ≤ 1.15 × Baseline-A bearing RMSE
existence:            REPORTED SEPARATELY, NEVER POOLED for control-readiness:
                        detector_miss_in_domain      → active-belief retention
                        detector_miss_outside_domain → active-belief retention
                        gated_rejection              → active-belief retention
                      primary criterion (in-domain only):
                        ≥ 0.60 of detector_miss_in_domain frames retain active belief
                        (F9b pooled achieved 8/57 = 0.140; F9b did not distinguish
                         the classes, so this is the closest available comparison)
                      after ≥ 20 consecutive predicted-in-domain misses, P(e) < 0.10
recovery:             mean frames to re-activate after re-detection ≤ 2
false tracks:         false_track_initializations ≤ 1 (F9b achieved 0)
```

**Pre-specified minimum final-evaluation support (locked before the final run):**

```text
near   (r < 0.55 m)          ≥ 100 eligible-visible frames
medium (0.55 ≤ r < 0.80 m)   ≥ 200 eligible-visible frames
far    (r ≥ 0.80 m)          ≥ 200 eligible-visible frames
edge_fov                     ≥  50 eligible-visible frames
```

If near < 100 in the final run, the gate **cannot** be labelled `CONTROL_READY` regardless of every other metric.

**Cross-task invariants (each has a dedicated regression test; none may be relaxed):**

**I1 — one innovation covariance.** Association, the innovation gate, and the EKF correction must all use the *same* `S` for the same candidate on the same frame:

```text
S = H P⁻ Hᵀ + λR        with the SAME λ-inflated R in all three places
```

Never `raw R` for association and gating but `λR` for correction. `λ` is fitted so the calibration NIS median matches χ²₂; if association and gating threshold against a different `S` than the one `λ` was fitted to, the calibrated decision boundary does not apply to the decisions it was calibrated for. Enforced by `test_association_gate_and_correction_share_one_innovation_covariance` (Task 8).

**I2 — detection evidence ≠ measurement acceptance.** See Finding 6. `detector_detected` drives existence; `kinematic_measurement_accepted` drives EKF correction. A gated-out bbox on an active track is not an existence miss. Enforced by `test_a_rejected_localization_is_not_an_existence_miss` (Task 8).

**I3 — outside-domain absence applies no likelihood.** See Finding 7. Enforced by `test_outside_domain_miss_decays_only_through_survival` (Task 7).

**I4 — the ablation runs zero inference.** The final run persists an immutable runtime cache to disk; the ablation replays that file and is forbidden from constructing a detector or a simulator. Enforced by `test_ablation_performs_no_inference_and_no_render` (Task 12).

**I5 — the runtime cache holds raw, pre-bias candidates.** Every cached candidate field is named `raw_*` and is written before the Task 3b bias stage, so one cache can feed both the F9b-bias baseline and the F9c-bias robust path. Enforced by `test_runtime_cache_contains_pre_bias_raw_candidates` (Task 11).

**I6 — `λ` is fitted on GT-matched nominal detections, not on the runtime accepted set.** Selecting the fitting set by gate acceptance would be circular: acceptance depends on `S`, which depends on the `λ` being fitted. Enforced by `test_lambda_fitting_set_is_selected_by_ground_truth_not_by_the_gate` (Task 9).

**I8 — a single miss may not carry unbounded evidence of absence.** F9c calibration (seeds 6101–6108) established that no single-frame `P_D` conditioning can fix belief collapse: measured `P_D` by range is near 0.997 / medium 0.993 / far 0.972, giving miss likelihood ratios of 0.003 / 0.0075 / 0.028 against F9b's global 0.0233 — conditioning makes collapse *faster* in the near and medium bins. The mis-specification is not the value of `P_D` but the independence assumption behind it: real detector misses arrive in bursts (F9b mean run 7.125 frames), and an independent-Bernoulli likelihood cannot express that.

The floor is therefore applied to the miss likelihood ratio itself:

```text
LR_miss = (1 - P_D_eff) / (1 - P_FA)
LR_used = max(LR_miss, LR_floor)
```

`LR_floor` is **derived from the observed miss-run length**, never hand-picked and never tuned to hit a retention target:

```text
LR_floor = LR_nominal ** (1 / L_mean)
```

where `L_mean` is the mean length of consecutive genuine-miss runs measured on the calibration seeds, and `LR_nominal` is the global miss likelihood ratio. The reasoning: a burst of correlated misses carries roughly the evidence of **one** independent miss, not `L_mean` of them, so spreading the nominal single-miss evidence across the run's expected length is the correct discount. A full run of typical length then reproduces exactly the nominal single-miss evidence, and runs materially longer than typical still drive existence down.

This is the same species of correction as the posterior variance floor, with the same justification: a model that is over-confident about its own reliability has its influence bounded, using a quantity measured from calibration data.

**I7 — the association gate must stay strictly looser than the innovation gate.** Association filters candidates before the gate sees them, so equal thresholds make `InnovationGate` unreachable whenever `temporal_association` is enabled — the `innovation_gate` ablation switch would then have no observable effect, and any contribution attributed to it in Task 12 would be spurious. Enforced by `test_association_gate_is_looser_than_the_innovation_gate` (Task 8) and by a runtime assertion in the coordinator's construction.

**Do not implement in F9c:** reward, stop state machine, SAC/TD3/PPO, side cameras, multi-pedestrian tracking, particle filter, UKF, Student-t filter, DeepSORT/ByteTrack/transformer trackers, or any change to `Q`. **STOP and report after F9c.**

---

## Empirical Basis — Why This Plan Differs From The Original F9c Prompt

All numbers below come from `artifacts/f9_yolo_measurement_calibration.csv` (F9a calibration seeds 4101–4104, 1193 matched measurements) and the already-published F9b summary. No F9b row-level test data was mined to produce them.

### Finding 1 — the residual is dominated by *per-episode* bias, not per-frame noise

Unbalanced one-way random-effects decomposition of the raw error over 30 calibration episodes with ≥15 matched samples — computed with the **same estimator Task 6 specifies**, `τ̂² = max(0, (MS_B − MS_W)/n_eff)`, not the SD of the episode means:

```text
                k    n_eff    τ̂ (between)   σ̂_w (within)   total SD
range          30    39.57    0.01425 m     0.00739 m      0.01581 m
bearing        30    39.57    0.01203 rad   0.00455 rad    0.01265 rad
```

Per-seed raw range bias:

```text
4101  −0.03028      4102  −0.05796      4103  −0.03591      4104  −0.06118
```

That is a **0.031 m spread between seeds**, against a within-episode noise SD of only 0.0074 m.

**The grouping is not the same for the two variables.** Refitting with seeds as groups, and separately measuring how much of each episode offset is shared inside a seed:

```text
                seed-level τ̂    SD of episode means ACROSS seeds    WITHIN seed
range           0.01550 m       0.01526 m                           0.00480 m
bearing         0.00501 rad     0.00708 rad                         0.01459 rad
```

Range offset is almost entirely **seed-level**: episodes inside one seed share nearly the whole offset (0.01526 across vs 0.00480 within). Bearing is the opposite — **episode-level**, driven by scenario geometry rather than by the seed.

This has one consequence the fit must respect: **`SE(b̂)` must divide by the number of independent units at the level that actually carries the offset.** For range that unit is the *seed*, so `SE ≈ τ̂_seed/√n_seeds`, not `τ̂/√n_episodes`. Using the episode count would claim an accuracy the data does not support — with 80 episodes it would shrink that term by roughly 3× against the truth.

**Corrected floor model, used by Task 6 and Task 9 — a two-level nested fit, seed → episode|seed:**

```text
σ_floor²  =  τ̂_seed²  +  τ̂_episode|seed²  +  SE(b̂)²

SE(b̂)²   =  τ̂_seed²/n_seeds  +  τ̂_episode|seed²/n_episodes
```

**Honesty note on the F9a projection.** The seed-level numbers in the table above come from only `k = 4` groups, and the ACROSS/WITHIN split is a descriptive unweighted decomposition, not the random-effects estimator — the same class of error that produced the discarded `0.01562 rad` figure. Those magnitudes are therefore indicative, not reliable: for range the one-level episode fit (`0.01425`) and the seed-level fit (`0.01550`) are noisy estimates of overlapping quantities. What the data *does* support is the structural claim — range offset is largely shared within a seed, bearing offset is not.

Accordingly the projection is stated as a band, not a point:

```text
σ_floor,r  expected ≈ 0.015 – 0.018 m
σ_floor,β  expected ≈ 0.012 – 0.016 rad
```

Task 9 computes the authoritative values from 6101–6108 with the nested estimator. This is the second, independent reason for eight seeds rather than four: with `k = 4` the seed-level component is barely estimable at all.

**Consequences, all of which are visible in F9b:**

1. It explains the F9b bias failure exactly. F9a fitted `b_r = −0.04590` on 4101–4104; F9b's raw range bias on 5101–5104 was `−0.02662`. The 0.019 m gap is a *group-level* shift entirely consistent with a per-episode bias SD of 0.014. A globally frozen additive constant fitted on four seeds cannot transfer.
2. It explains why the EKF "does not help". An EKF averages measurements it assumes are independent. A constant per-episode offset survives averaging untouched, so more frames buy nothing on range.
3. **It proves `R` inflation alone cannot fix coverage.** With `Q` frozen and tiny (`0.001 m/√s`), the steady-state posterior variance shrinks roughly like `λσ_w²/N_eff`. Whatever `λ` is chosen, sufficient frames drive the reported `σ` below the irreducible per-episode offset. The original prompt's §13–16 (`R' = λR`, optional floor) therefore cannot reach the coverage target on long episodes.

**Correction #4 (new):** the per-episode bias must be represented as a **posterior variance floor applied at the belief-reporting boundary**, not only as an `R` scale:

```text
σ_report,r² = σ_EKF,r² + σ_floor,r²
σ_report,β² = σ_EKF,β² + σ_floor,β²
```

with `σ_floor` estimated from the two-level nested random-effects fit specified below.

This is not arbitrary inflation. `τ̂` is a measured quantity, and the floor is an explicit statement that the belief handed to the POMDP layer carries an unmodelled slowly-varying observation bias. The alternative — augmenting the EKF with a bias state — would change the frozen state definition and is out of scope.

**Predicted effect (state this before running, then check it):** with `σ_EKF,r ≈ 0.0058` and `σ_floor,r` in the projected band `0.015 – 0.018`, `σ_report,r ≈ 0.016 – 0.019`, against a per-frame error SD of comparable size once the refit is unbiased. Taking the band midpoint, `σ_report,r ≈ 0.0181` against an error SD of `≈ 0.0171` gives `coverage_68 ≈ 0.71`, `coverage_95 ≈ 0.96`, and `mean_predicted_std / RMSE ≈ 1.05` — inside both the coverage bands and the anti-inflation guard. If the achieved numbers land far outside this, the model is wrong; say so rather than retuning.

### Finding 2 — `R` inflation still has a job, just a different one

`λ` should not be fitted to coverage. It should be fitted so the *innovation* statistics are consistent, because the innovation covariance `S` is what the gate and the association rule threshold against. Fit `λ` so calibration NIS matches χ²₂ (median 1.386), then verify the tail. Separation of concerns:

```text
λ (R inflation)      → correct S → correct gate/association thresholds
σ_floor (posterior)  → correct reported uncertainty → correct coverage
```

### Finding 3 — per-bin bias is tempting and probably wrong

F9a per-bin raw range bias: near `−0.02549` (n=42), medium `−0.04172` (n=239), far `−0.04794` (n=912). The spread across bins (0.022 m) is *comparable to the spread across seeds* (0.031 m), and near has only 42 samples from a single scenario. A per-bin fit risks encoding seed identity as if it were range dependence. **Decision rule, pre-specified:** use per-bin bias only if both hold on calibration data — (i) every bin has ≥ 100 matched samples from ≥ 3 distinct scenarios, and (ii) leave-one-seed-out cross-validation across the 8 calibration seeds shows per-bin reduces held-out range RMSE by ≥ 10% relative to global additive. Otherwise use global additive. LOSO is the right instrument here precisely because it measures the transfer failure that killed F9b.

### Finding 4 — near range does not exist outside two calibration-only scenarios

Eligible-frame distance distribution per F9a scenario:

```text
calibration_near_stationary      near=42  medium=0    far=0      (use_for_final_evaluation = false)
calibration_medium_stationary    near=0   medium=84   far=0      (use_for_final_evaluation = false)
cross_left_to_right              near=0   medium=0    far=264
cross_right_to_left              near=0   medium=0    far=264
crossing_moving_turning_ego      near=0   medium=83   far=76
stationary_ped_moving_ego        near=0   medium=72   far=76
stationary_ped_stationary_ego    near=0   medium=0    far=148
stationary_ped_turning_ego       near=0   medium=0    far=112
```

GT range across all eligible calibration frames spans only `[0.464, 1.031] m`. Near-range frames exist *only* in the scenario deliberately excluded from final evaluation. Task 2 fixes this by adding final-evaluation-enabled near/medium approach scenarios.

### Finding 5 — existence collapse arithmetic

With `P_D = 0.9766776`, `P_FA = 0.00078003`, `P_S = 0.995`, `P_birth = 0.005`, starting from `p = 0.99`:

```text
1 miss → 0.614      2 misses → 0.036      (mean natural miss run in F9b = 7.125 frames)
```

Raising `P_S` cannot rescue this; the miss likelihood ratio `(1−P_D)/(1−P_FA) ≈ 0.0233` dominates. The lever is `P_D^eff` conditioned on **predicted** observability, as locked by the operator.

### Finding 6 — detection evidence and kinematic acceptance are different quantities

An earlier draft of this plan fed the existence filter with `detected = "an accepted measurement exists"`. That is wrong, and it would have made the innovation gate actively harmful.

The detector answering "there is a Duckie in this image" and the geometry answering "I trust these bbox coordinates" are independent claims. F9b recorded 16 visible frames where YOLO found the pedestrian but localized it badly enough to produce measurement range RMSE `0.1511 m` and NIS P95 `282`. Those frames are exactly the ones the gate is built to reject — and under the earlier draft each rejection would have been scored as a miss, applying the `0.0233` miss likelihood ratio to a frame where the detector *did* see the pedestrian. The gate would have accelerated existence collapse instead of preventing it.

**Correction: two independent flags flow out of the runtime step.**

```text
detector_detected              a Duckie candidate of sufficient confidence exists in this frame
kinematic_measurement_accepted association found a candidate AND the gate passed it
```

```text
existence observation update  ← driven by detector_detected
EKF correction vs prediction  ← driven by kinematic_measurement_accepted
```

A rejected localization on an **already active track** is therefore "I believe the pedestrian is there, I do not believe these coordinates": existence is updated as a detection, the EKF runs prediction-only.

Track **initialization** stays strict, because there the two claims cannot be separated — with no prior track there is no innovation to test against, so a candidate that fails projection or falls outside the association gate must not create a track. Initialization requires `kinematic_measurement_accepted`.

### Finding 7 — an outside-domain miss carries almost no information

Forcing every class to satisfy `P_D^eff > P_FA` and then running the ordinary Bayesian miss update is the wrong model for a pedestrian the belief predicts is not in frame. No detection there is *expected*; it is not evidence of non-existence. Squeezing this into a very small `P_D^eff` is a workaround that still applies a likelihood ratio where none is warranted, and it makes the fitted number a tuning knob rather than a measured probability.

**Correction: outside-domain frames skip the observation update entirely.**

```text
CENTER / MID_FOV / EDGE_FOV  + no detection   → Bayesian existence observation update with P_D^eff(class)
OUTSIDE_DOMAIN               + no detection   → prediction step only; no likelihood applied
any class                    + detection      → Bayesian observation update (positive evidence always counts)
```

The asymmetry is deliberate and is the POMDP-correct reading: a sensor cannot inform you about a region it does not observe, but a detection from a region you predicted was unobservable is still a real observation — and is precisely the signal that the prediction was wrong. Under this rule `P_D^eff(OUTSIDE_DOMAIN)` is never used for misses; it is retained in the artifact as a diagnostic only.

Consequence for existence decay: outside-domain absence now decays `P(e)` only through the survival term `P_S = 0.995`, i.e. a half-life of ≈ 138 frames. That is intended. A pedestrian that leaves the field of view should be forgotten slowly, and the acceptance criterion "after ≥ 20 consecutive **predicted-in-domain** misses, `P(e) < 0.10`" is worded against in-domain misses for exactly this reason.

---

## File Structure

**New source modules** (each one responsibility, all NumPy-only, all runtime-pure):

```text
src/duckie_pomdp/belief/innovation_gate.py         NIS gate: hard accept/reject decision + record
src/duckie_pomdp/belief/bias_correction.py         frozen F9c additive bias applied to raw candidates
src/duckie_pomdp/belief/measurement_association.py candidate → predicted-measurement Mahalanobis selection
src/duckie_pomdp/belief/covariance_calibration.py  λ_r, λ_β, σ_floor_r, σ_floor_β; R inflation + posterior floor
src/duckie_pomdp/belief/observability.py           predicted image-plane observability → P_D^eff (no GT)
src/duckie_pomdp/belief/robust_updater.py          coordinator: associate → gate → EKF → existence → report
                                                   also owns the single innovation-covariance provider (I1)
src/duckie_pomdp/evaluation/f9c_runtime_cache.py   immutable per-frame runtime cache: write once, replay many (I4)
src/duckie_pomdp/evaluation/f9c_protocol.py        F9c config load + freeze-boundary + seed-disjointness guards
src/duckie_pomdp/evaluation/f9c_belief.py          coverage bands, coverage_error, robustness + miss-sequence metrics
src/duckie_pomdp/evaluation/f9c_calibration.py     bias refit (global vs per-bin + LOSO), variance components, P_D^eff fit
```

**Modified source:**

```text
src/duckie_pomdp/perception/f9_pipeline.py         add candidate-list output; keep select_single_duckie intact
src/duckie_pomdp/belief/existence_filter.py        accept per-step detection probability; default preserves F9b behaviour
src/duckie_pomdp/belief/__init__.py                export new names
src/duckie_pomdp/perception/__init__.py            export new names
```

**Config:**

```text
configs/f9c_robust_belief_v1.toml                  new; explicit [frozen_reference] + [robust_observation] switches
```

**Experiments:**

```text
experiments/calibrate_f9c_robust_belief.py         seeds 6101–6108 → artifacts/f9c_calibration*.{csv,json}
experiments/evaluate_f9c_robust_belief.py          seeds 7101–7104 → baseline A + robust B + ablations
experiments/verify_f9c_artifacts.py                read-only artifact/hash verifier (no inference)
```

**Tests (new):**

```text
tests/test_f9c_innovation_gate.py
tests/test_f9c_bias_correction.py
tests/test_f9c_association.py
tests/test_f9c_covariance_calibration.py
tests/test_f9c_observability.py
tests/test_f9c_existence.py
tests/test_f9c_robust_updater.py
tests/test_f9c_protocol.py
tests/test_f9c_leakage.py
```

**Artifacts produced:**

```text
artifacts/f9c_calibration.csv              artifacts/f9c_validation.csv
artifacts/f9c_calibration_metrics.json     artifacts/f9c_belief_metrics.json
artifacts/f9c_frozen_config.json           artifacts/f9c_ablation_metrics.json
                                           artifacts/f9c_nis_metrics.json
                                           artifacts/f9c_error_cases/
                                           artifacts/f9c_runtime_cache.npz   (written once by the final run)
                                           artifacts/f9c_evaluation_truth.npz (GT, kept in a separate file)
```

**Why 8 calibration seeds:** `SE(b̂) ≈ τ̂/√n_seeds`. With τ̂ ≈ 0.014 m, 4 seeds gives SE ≈ 0.0070 m; 8 seeds gives ≈ 0.0050 m. That directly shrinks the transfer error that broke F9b, and it costs one extra calibration run. Additional seeds do not remove the per-episode offset itself — only `σ_floor` handles that.

---

## Task 1: F9c protocol, config, and freeze-boundary guards

**Files:**
- Create: `configs/f9c_robust_belief_v1.toml`
- Create: `src/duckie_pomdp/evaluation/f9c_protocol.py`
- Test: `tests/test_f9c_protocol.py`

**Interfaces:**
- Consumes: `duckie_pomdp.evaluation.f9_protocol.sha256`, `F9ScenarioSpec`, `load_scenario`.
- Produces: `F9cProtocol` dataclass with fields `config_path`, `checkpoint_path`, `checkpoint_sha256`, `frozen_f7_config_path`, `calibration_seeds: tuple[int, ...]`, `final_evaluation_seeds: tuple[int, ...]`, `forbidden_seeds: tuple[int, ...]`, `scenarios: tuple[F9ScenarioSpec, ...]`, `robust: RobustObservationSwitches`, `acceptance: AcceptanceBands`, `minimum_support: dict[str, int]`, `artifacts: dict[str, Path]`, plus `config_sha256` property; `RobustObservationSwitches(bias_refit: bool, innovation_gate: bool, temporal_association: bool, covariance_calibration: bool, conditional_detection: bool)`; `AcceptanceBands(coverage_68_low, coverage_68_high, coverage_95_low, coverage_95_high, max_std_over_rmse, max_rmse_ratio_vs_baseline)`; `load_f9c_protocol(path, *, require_frozen: bool = False) -> F9cProtocol`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_protocol.py
from pathlib import Path

import pytest

from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"


def test_f9c_seeds_are_disjoint_from_every_earlier_split():
    protocol = load_f9c_protocol(CONFIG)
    calibration = set(protocol.calibration_seeds)
    final = set(protocol.final_evaluation_seeds)
    assert calibration and final
    assert not calibration & final
    forbidden = set(protocol.forbidden_seeds)
    assert {5101, 5102, 5103, 5104} <= forbidden
    assert {4101, 4102, 4103, 4104} <= forbidden
    assert not (calibration | final) & forbidden


def test_f9c_may_not_change_frozen_f7_dynamics(tmp_path):
    protocol = load_f9c_protocol(CONFIG)
    import tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    assert f9c["ekf"] == f7["ekf"]


def test_f9c_keeps_survival_and_birth_frozen():
    protocol = load_f9c_protocol(CONFIG)
    import tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    for key in ("prior_probability", "survival_probability", "birth_probability"):
        assert f9c["existence"][key] == f7["existence"][key]


def test_f9c_rejects_a_config_that_edits_process_noise(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "position_process_std_m_per_sqrt_s = 0.001",
        "position_process_std_m_per_sqrt_s = 0.002",
    )
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen F7"):
        load_f9c_protocol(broken)


def test_f9c_acceptance_bands_are_pre_specified():
    protocol = load_f9c_protocol(CONFIG)
    bands = protocol.acceptance
    assert bands.coverage_68_low == 0.60 and bands.coverage_68_high == 0.76
    assert bands.coverage_95_low == 0.90 and bands.coverage_95_high == 0.98
    assert bands.max_std_over_rmse == 1.5
    assert protocol.minimum_support["near"] >= 100


def test_f9c_ablation_switches_default_to_all_enabled():
    protocol = load_f9c_protocol(CONFIG)
    switches = protocol.robust
    assert switches.bias_refit
    assert switches.innovation_gate
    assert switches.temporal_association
    assert switches.covariance_calibration
    assert switches.conditional_detection
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.evaluation.f9c_protocol'`

- [ ] **Step 3: Write `configs/f9c_robust_belief_v1.toml`**

Copy `configs/f9_yolo_ekf_v1.toml` and change only the sections below. The `[ekf]` block and the three `[existence]` keys `prior_probability`, `survival_probability`, `birth_probability` must be byte-identical to `configs/oracle_ekf_v1.toml`.

```toml
schema_version = 1

[split]
calibration_seeds = [6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108]
final_evaluation_seeds = [7101, 7102, 7103, 7104]
forbidden_seeds = [1101, 1102, 1103, 1104, 1105, 1106, 2101, 2102, 3101, 3102, 4101, 4102, 4103, 4104, 5101, 5102, 5103, 5104]

[robust_observation]
bias_refit = true
innovation_gate = true
temporal_association = true
covariance_calibration = true
conditional_detection = true

[innovation_gate]
# Hard reject only in F9c v1. A soft-downweight variant would make the
# correction use 25*lambda*R while association and the gate used lambda*R,
# breaking invariant I1. See "Why there is no downweight mode".
mode = "hard_reject"
chi_square_threshold = 9.21034037197618   # 2 DOF, 99%
# Invariant I2: a gated-out bbox suppresses the EKF correction only. Existence
# evidence comes from the detector, never from the gate decision.
existence_evidence_source = "detector"

[association]
rule = "minimum_nis"
# DELIBERATELY LOOSER than [innovation_gate].chi_square_threshold. The two
# answer different questions: association asks "which candidate is plausibly
# the track?" and should reject only wild outliers, while the gate asks "do I
# trust these coordinates enough to correct with them?" and is the actual
# accept/reject decision. Setting them equal makes the gate unreachable
# whenever temporal_association is on -- association filters first, so every
# surviving candidate trivially passes the gate, and the innovation_gate
# ablation switch has no observable effect.
chi_square_gate = 13.815510557964274   # 2 DOF, 99.9%
initialization_rule = "highest_confidence_then_bbox_lexicographic"

[covariance_calibration]
# Filled by experiments/calibrate_f9c_robust_belief.py, then frozen.
parameters_frozen = false
range_scale = 1.0
bearing_scale = 1.0
range_posterior_floor_m = 0.0
bearing_posterior_floor_rad = 0.0

[measurement_model]
# F9c-fitted bias, used when robust_observation.bias_refit = true.
parameters_frozen = false
bias_model = "global_additive"   # or "per_range_bin" if the LOSO rule in Task 9 selects it
range_bias_m = 0.0
bearing_bias_rad = 0.0

[baseline_measurement_model]
# F9b frozen bias. This is Baseline A's correction and is also what every
# ablation row with bias_refit = false uses. Never refit; copied verbatim from
# configs/f9_yolo_ekf_v1.toml so "all switches off == Baseline A" holds by
# construction rather than by coincidence.
bias_model = "global_additive"
range_bias_m = -0.045904804710162034
bearing_bias_rad = 0.00414567890700929

[conditional_detection]
# Filled by calibration; keys are predicted-observability classes.
parameters_frozen = false
detection_probability_center = 0.9766775777414075
detection_probability_mid_fov = 0.9766775777414075
detection_probability_edge_fov = 0.9766775777414075
# Invariant I3: this value is a reported diagnostic only. It is never applied to
# a miss, because an outside-domain absence carries no likelihood.
detection_probability_outside_domain = 0.9766775777414075
outside_domain_miss_policy = "prediction_only"
false_positive_probability = 0.00078003120124805

[existence_track]
active_threshold = 0.50
delete_threshold = 0.05
initialization_threshold = 0.50

[acceptance]
coverage_68_low = 0.60
coverage_68_high = 0.76
coverage_95_low = 0.90
coverage_95_high = 0.98
max_std_over_rmse = 1.5
max_rmse_ratio_vs_baseline = 1.15

[minimum_support]
near = 100
medium = 200
far = 200
edge_fov = 50

[artifacts]
calibration_csv = "../artifacts/f9c_calibration.csv"
calibration_metrics_json = "../artifacts/f9c_calibration_metrics.json"
frozen_config_json = "../artifacts/f9c_frozen_config.json"
validation_csv = "../artifacts/f9c_validation.csv"
belief_metrics_json = "../artifacts/f9c_belief_metrics.json"
ablation_metrics_json = "../artifacts/f9c_ablation_metrics.json"
nis_metrics_json = "../artifacts/f9c_nis_metrics.json"
error_case_dir = "../artifacts/f9c_error_cases"
runtime_cache = "../artifacts/f9c_runtime_cache.npz"
evaluation_truth = "../artifacts/f9c_evaluation_truth.npz"
calibration_log = "../artifacts/f9c_calibration.log"
validation_log = "../artifacts/f9c_validation.log"
```

- [ ] **Step 4: Write `src/duckie_pomdp/evaluation/f9c_protocol.py`**

Model it on `f9_protocol.py`. The `_validate` function must:
1. verify `sha256(checkpoint_path) == checkpoint_sha256`;
2. raise `ValueError("F9c changes frozen F7 dynamics")` if `data["ekf"] != frozen_f7["ekf"]` — message must contain the literal substring `frozen F7`;
3. raise if any of `prior_probability`, `survival_probability`, `birth_probability` differ from the frozen F7 config;
4. raise if `calibration_seeds` and `final_evaluation_seeds` are empty or overlapping, or if either intersects `forbidden_seeds`, or if either intersects the detector manifest seeds;
5. when `require_frozen=True`, raise unless `measurement_model.parameters_frozen`, `covariance_calibration.parameters_frozen`, and `conditional_detection.parameters_frozen` are all true and `artifacts/f9c_frozen_config.json` exists with a matching `config_sha256`.

Note that F9c **deliberately does not** re-validate `data["existence"]["detection_probability"]` against F7 — that is the unfrozen parameter. Add a comment saying so.

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 114 passed. Append to `IMPLEMENTATION_NOTES.md` under `## F9c`: the freeze-boundary table, the seed allocation, and the pre-specified acceptance bands, stating explicitly that they were written before any 7101-series frame was rendered.

---

## Task 2: Final-evaluation near-range scenarios

**Files:**
- Modify: `configs/f9c_robust_belief_v1.toml` (`[[scenario_matrix]]` entries)
- Test: `tests/test_f9c_protocol.py` (add one test)

**Interfaces:**
- Consumes: `F9cProtocol.scenarios` from Task 1.
- Produces: scenario names `approach_near_stationary_ego`, `approach_near_moving_ego`, `cross_near_left_to_right` usable by both calibration and final evaluation.

The existing near scenario uses `ego_start_x_offset_m = 0.50` with `use_for_final_evaluation = false`, which is exactly why F9b had N=0 near. F9c needs near frames on **both** sides of the split.

- [ ] **Step 1: Write the failing test**

```python
def test_f9c_scenario_matrix_supports_near_range_final_evaluation():
    protocol = load_f9c_protocol(CONFIG)
    final = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
    calibration = [spec for spec in protocol.scenarios if spec.use_for_calibration]
    # A scenario reaches near range either by starting close, or by starting at
    # medium range and driving in. Classifying by start offset alone would
    # exclude the moving approach, whose whole purpose is to traverse the bin.
    def reaches_near(spec):
        return spec.ego_start_x_offset_m >= 0.35 or (
            spec.action.linear_velocity_mps > 0.0
            and spec.ego_start_x_offset_m >= 0.25
        )

    near_final = [spec for spec in final if reaches_near(spec)]
    near_calibration = [spec for spec in calibration if reaches_near(spec)]
    assert len(near_final) >= 2, "final evaluation must contain near-range scenarios"
    assert len(near_calibration) >= 2
    # A moving-ego approach is required so near range is traversed, not only sampled statically.
    assert any(
        spec.action.linear_velocity_mps > 0.0 for spec in near_final
    ), "at least one near-range final scenario must approach the pedestrian"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_f9c_protocol.py::test_f9c_scenario_matrix_supports_near_range_final_evaluation -q`
Expected: FAIL — `assert 0 >= 2`

- [ ] **Step 3: Add the scenarios to `configs/f9c_robust_belief_v1.toml`**

Keep the six original F9b scenarios byte-identical (they are Baseline A's trajectories), flip the two old `calibration_*` scenarios to `use_for_final_evaluation = true` under new names, and add a moving approach:

```toml
[[scenario_matrix]]
name = "approach_near_stationary_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 60
ego_start_x_offset_m = 0.50
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "approach_medium_stationary_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 60
ego_start_x_offset_m = 0.25
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "approach_near_moving_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.20
angular_velocity_rad_s = 0.0
steps = 90
ego_start_x_offset_m = 0.30
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "cross_near_left_to_right"
pedestrian_mode = "cross_left_to_right"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 110
ego_start_x_offset_m = 0.40
use_for_calibration = true
use_for_final_evaluation = true
```

`approach_near_moving_ego` sweeps range continuously downward, which is what makes the near bin a traversed regime rather than a single static pose — and it is the regime a future stop policy will actually operate in.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Verify the support minima are actually reachable — dry run on ONE calibration seed**

Run:

```bash
$PY - <<'PY'
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
protocol = load_f9c_protocol(Path("configs/f9c_robust_belief_v1.toml"))
final = [s for s in protocol.scenarios if s.use_for_final_evaluation]
print(sum(s.steps + 1 for s in final) * len(protocol.final_evaluation_seeds), "final frames")
PY
```

Then render seed 6101 only through the Task 11 collector once it exists, and count `distance_bin`. **If near < 100/4 seeds proportionally, increase `steps` or `ego_start_x_offset_m` NOW, before any 7101 frame is rendered.** Record the dry-run counts in `IMPLEMENTATION_NOTES.md`. Adjusting scenario geometry after seeing final-seed results is a gate failure; adjusting it from a calibration-seed dry run is correct practice.

- [ ] **Step 6: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 115 passed. Note the dry-run distance-bin counts in `IMPLEMENTATION_NOTES.md`.

---

## Task 3: Innovation gate

**Files:**
- Create: `src/duckie_pomdp/belief/innovation_gate.py`
- Test: `tests/test_f9c_innovation_gate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except NumPy.
- Produces:
  - `InnovationGateConfig(chi_square_threshold: float)`.
  - `GateDecision(accepted: bool, nis: float, threshold: float)`.
  - `normalized_innovation_squared(innovation: NDArray, innovation_covariance: NDArray) -> float`.
  - `InnovationGate.evaluate(innovation: NDArray, innovation_covariance: NDArray) -> GateDecision`.

The gate consumes only the innovation and `S`. It never sees a measurement object, a confidence, or ground truth — that is enforced by the signature.

**Why there is no downweight mode.** An earlier draft offered `GateMode.DOWNWEIGHT`, accepting an inconsistent measurement with `covariance_scale = 25`. That is incompatible with invariant I1. Association and the gate would threshold against `S = HP⁻Hᵀ + λR`, while the correction would then use `S = HP⁻Hᵀ + 25λR` — so `S_gate ≠ S_correction`, and the `λ` fitted to make the NIS median match χ²₂ would no longer describe the covariance the decision boundary was calibrated on. Since the frozen config selects `hard_reject` anyway, F9c v1 implements **only** hard rejection:

```text
NIS ≤ threshold  → accept
NIS >  threshold → reject, EKF runs prediction only
```

A soft-downweighting variant remains a legitimate follow-up experiment if hard rejection turns out to be too aggressive — but it needs its own gate design that keeps one `S`, and it is out of scope here. Do not add a `mode` field "for future flexibility"; an unused branch that violates a stated invariant is worse than no branch.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_innovation_gate.py
import numpy as np
import pytest

from duckie_pomdp.belief.innovation_gate import (
    InnovationGate,
    InnovationGateConfig,
    normalized_innovation_squared,
)

THRESHOLD = 9.21034037197618


def _gate():
    return InnovationGate(InnovationGateConfig(chi_square_threshold=THRESHOLD))


def test_nis_matches_the_closed_form_for_a_diagonal_covariance():
    innovation = np.array([0.03, 0.01])
    covariance = np.diag([0.0004, 0.0001])
    expected = 0.03**2 / 0.0004 + 0.01**2 / 0.0001
    assert normalized_innovation_squared(innovation, covariance) == pytest.approx(expected)


def test_nis_uses_the_full_matrix_not_only_the_diagonal():
    innovation = np.array([0.02, -0.02])
    covariance = np.array([[4.0e-4, 3.0e-4], [3.0e-4, 4.0e-4]])
    expected = float(innovation @ np.linalg.solve(covariance, innovation))
    assert normalized_innovation_squared(innovation, covariance) == pytest.approx(expected)
    assert expected != pytest.approx(0.02**2 / 4.0e-4 + 0.02**2 / 4.0e-4)


def test_consistent_measurement_is_accepted():
    decision = _gate().evaluate(np.array([0.005, 0.002]), np.diag([0.0004, 0.0001]))
    assert decision.accepted
    assert decision.nis < THRESHOLD


def test_gross_outlier_is_rejected():
    decision = _gate().evaluate(np.array([0.30, 0.0]), np.diag([0.0004, 0.0001]))
    assert not decision.accepted
    assert decision.nis > THRESHOLD


def test_the_gate_exposes_no_covariance_scaling_knob():
    """Invariant I1: the gate must not be able to hand the correction a
    different R than association and the gate itself thresholded against."""
    decision = _gate().evaluate(np.array([0.30, 0.0]), np.diag([0.0004, 0.0001]))
    assert not hasattr(decision, "covariance_scale")
    assert not hasattr(decision, "downweighted")
    import duckie_pomdp.belief.innovation_gate as module

    assert not hasattr(module, "GateMode")


def test_gate_is_exactly_inclusive_at_the_threshold():
    # Build the boundary exactly rather than round-tripping through sqrt:
    # (THRESHOLD**0.5)**2 lands ~2e-15 above THRESHOLD, so an inclusive gate
    # would appear to reject its own boundary for purely floating-point reasons.
    covariance = np.diag([1.0, 1.0])
    innovation = np.array([3.0, 0.0])
    nis = normalized_innovation_squared(innovation, covariance)
    assert nis == 9.0
    gate = InnovationGate(InnovationGateConfig(chi_square_threshold=nis))
    decision = gate.evaluate(innovation, covariance)
    assert decision.nis == nis
    assert decision.accepted, "NIS exactly at the threshold must be accepted"
    just_above = gate.evaluate(np.array([3.0 + 1.0e-6, 0.0]), covariance)
    assert not just_above.accepted, "the boundary must still be a boundary"


def test_gate_rejects_a_non_positive_definite_innovation_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        _gate().evaluate(np.array([0.01, 0.01]), np.diag([0.0004, -1.0e-9]))


def test_gate_signature_cannot_receive_ground_truth():
    import inspect

    parameters = set(inspect.signature(InnovationGate.evaluate).parameters)
    assert parameters == {"self", "innovation", "innovation_covariance"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_innovation_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.innovation_gate'`

- [ ] **Step 3: Write the implementation**

```python
"""Innovation-consistency gate. Consumes only filter statistics, never truth.

Hard rejection only. A soft-downweight branch would make the EKF correction use
a different measurement covariance than the one association and this gate
thresholded against, breaking the single-innovation-covariance invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InnovationGateConfig:
    chi_square_threshold: float

    def __post_init__(self) -> None:
        if not isfinite(self.chi_square_threshold) or self.chi_square_threshold <= 0.0:
            raise ValueError("chi-square threshold must be finite and positive")


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    nis: float
    threshold: float


def normalized_innovation_squared(
    innovation: NDArray[np.float64],
    innovation_covariance: NDArray[np.float64],
) -> float:
    vector = np.asarray(innovation, dtype=float)
    matrix = np.asarray(innovation_covariance, dtype=float)
    if vector.shape != (2,) or matrix.shape != (2, 2):
        raise ValueError("innovation gate expects a 2D polar innovation")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        raise ValueError("innovation statistics must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError("innovation covariance must be positive definite")
    return float(vector @ np.linalg.solve(symmetric, vector))


class InnovationGate:
    def __init__(self, config: InnovationGateConfig) -> None:
        self.config = config

    def evaluate(
        self,
        innovation: NDArray[np.float64],
        innovation_covariance: NDArray[np.float64],
    ) -> GateDecision:
        nis = normalized_innovation_squared(innovation, innovation_covariance)
        threshold = self.config.chi_square_threshold
        return GateDecision(nis <= threshold, nis, threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_innovation_gate.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 123 passed.

---

## Task 3b: Frozen F9c bias correction as an explicit runtime stage

*(Numbered 3b so the later task numbers referenced throughout this document stay stable.)*

**Files:**
- Create: `src/duckie_pomdp/belief/bias_correction.py`
- Test: `tests/test_f9c_bias_correction.py`

**Interfaces:**
- Consumes: `ObjectMeasurement`, `wrap_angle`.
- Produces:
  - `FrozenBiasCorrection(model: str, range_bias_m: float, bearing_bias_rad: float, range_bin_bias_m: Mapping[str, float] | None, near_max_m: float, medium_max_m: float)` with `correct(measurement: ObjectMeasurement) -> ObjectMeasurement` and classmethods `identity()` and `from_config(data: Mapping) -> FrozenBiasCorrection`.
  - `model ∈ {"identity", "global_additive", "per_range_bin"}`.

**Why this task exists.** An earlier draft fitted `b_r`, `b_β` in Task 9 but never wired them into the Robust-B runtime — the coordinator went straight from candidate to association. Estimating a correction and not applying it is the quietest possible failure: every artifact would look right and the correction would do nothing. The bias stage is therefore a named, separately tested runtime component.

**Locked runtime order:**

```text
z_raw → FROZEN F9c BIAS CORRECTION → z_corr → association → gate → EKF
```

Bias correction runs **before** association, not after, because association thresholds candidates on innovation against `h(x̂⁻)` — comparing an uncorrected measurement against a corrected prediction would inject the full bias into every NIS and would systematically mis-rank candidates in duplicate frames.

**Baseline A keeps the F9b class.** `AdditiveMeasurementBias` in `perception/f9_pipeline.py` stays untouched and is what Baseline A uses, with the F9b constants `b_r = −0.045904804710162034`, `b_β = +0.00414567890700929`. Robust B uses `FrozenBiasCorrection` with the F9c-fitted values. Keeping them as two classes is what makes `bias_refit = false` mean *exactly* "Baseline A's bias", by construction rather than by coincidence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_bias_correction.py
import pytest

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement


def _measurement(range_m, bearing_rad):
    from math import cos, sin

    return ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=0.8,
        x_left_m=range_m * sin(bearing_rad),
        y_forward_m=range_m * cos(bearing_rad),
        range_m=range_m,
        bearing_rad=bearing_rad,
    )


def test_global_additive_subtracts_the_frozen_bias():
    correction = FrozenBiasCorrection(
        model="global_additive",
        range_bias_m=-0.0459,
        bearing_bias_rad=0.0041,
        range_bin_bias_m=None,
        near_max_m=0.55,
        medium_max_m=0.80,
    )
    corrected = correction.correct(_measurement(0.900, 0.050))
    assert corrected.range_m == pytest.approx(0.900 + 0.0459)
    assert corrected.bearing_rad == pytest.approx(0.050 - 0.0041)


def test_cartesian_fields_stay_consistent_with_the_corrected_polar_pair():
    from math import cos, sin

    correction = FrozenBiasCorrection(
        "global_additive", -0.0459, 0.0041, None, 0.55, 0.80
    )
    corrected = correction.correct(_measurement(0.900, 0.050))
    assert corrected.x_left_m == pytest.approx(
        corrected.range_m * sin(corrected.bearing_rad)
    )
    assert corrected.y_forward_m == pytest.approx(
        corrected.range_m * cos(corrected.bearing_rad)
    )


def test_per_range_bin_selects_the_bin_from_the_measured_range():
    correction = FrozenBiasCorrection(
        model="per_range_bin",
        range_bias_m=0.0,
        bearing_bias_rad=0.0041,
        range_bin_bias_m={"near": -0.0255, "medium": -0.0417, "far": -0.0479},
        near_max_m=0.55,
        medium_max_m=0.80,
    )
    assert correction.correct(_measurement(0.50, 0.0)).range_m == pytest.approx(
        0.50 + 0.0255
    )
    assert correction.correct(_measurement(0.70, 0.0)).range_m == pytest.approx(
        0.70 + 0.0417
    )
    assert correction.correct(_measurement(0.95, 0.0)).range_m == pytest.approx(
        0.95 + 0.0479
    )


def test_identity_correction_is_a_no_op():
    correction = FrozenBiasCorrection.identity()
    original = _measurement(0.900, 0.050)
    corrected = correction.correct(original)
    assert corrected.range_m == pytest.approx(0.900)
    assert corrected.bearing_rad == pytest.approx(0.050)


def test_correction_leaves_a_missing_measurement_untouched():
    correction = FrozenBiasCorrection("global_additive", -0.0459, 0.0041, None, 0.55, 0.80)
    missing = ObjectMeasurement.missing(ObjectClass.DUCKIE)
    assert correction.correct(missing) is missing


def test_corrected_range_is_clamped_at_zero():
    correction = FrozenBiasCorrection("global_additive", 0.50, 0.0, None, 0.55, 0.80)
    assert correction.correct(_measurement(0.20, 0.0)).range_m == 0.0


def test_bearing_correction_wraps_across_pi():
    from math import pi

    correction = FrozenBiasCorrection("global_additive", 0.0, -0.02, None, 0.55, 0.80)
    corrected = correction.correct(_measurement(0.90, pi - 0.01))
    assert corrected.bearing_rad == pytest.approx(-pi + 0.01)


def test_per_range_bin_requires_all_three_bins():
    with pytest.raises(ValueError, match="near, medium, far"):
        FrozenBiasCorrection(
            "per_range_bin", 0.0, 0.0, {"near": -0.02}, 0.55, 0.80
        )


def test_correction_never_receives_ground_truth():
    import inspect

    assert set(inspect.signature(FrozenBiasCorrection.correct).parameters) == {
        "self",
        "measurement",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_bias_correction.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.bias_correction'`

- [ ] **Step 3: Implement**

Follow the shape of `AdditiveMeasurementBias.correct` in `perception/f9_pipeline.py:42-61` — same clamping (`max(0.0, r − b)`), same `wrap_angle`, same rebuild of the Cartesian fields from the corrected polar pair. The only additions are the `per_range_bin` lookup and `identity()`. Do **not** modify `AdditiveMeasurementBias`; Baseline A depends on it being exactly what F9b ran.

Bin selection uses the **measured** range, never a predicted or true range — a bias correction that consumed the filter's own prediction would be a feedback loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_bias_correction.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 132 passed.

---

## Task 4: Temporal measurement association

**Files:**
- Create: `src/duckie_pomdp/belief/measurement_association.py`
- Test: `tests/test_f9c_association.py`

**Interfaces:**
- Consumes: `duckie_pomdp.domain.measurement.ObjectMeasurement`, `normalized_innovation_squared` (Task 3).
- Produces:
  - `AssociationConfig(chi_square_gate: float, initialization_rule: str)`.
  - `CandidateMeasurement(measurement: ObjectMeasurement, confidence: float, bbox_key: tuple[int, int, int, int])`.
  - `AssociationResult(selected_index: int | None, selected: CandidateMeasurement | None, mode: str, candidate_nis: tuple[float | None, ...], highest_confidence_index: int | None, differed_from_highest_confidence: bool)` where `mode ∈ {"initialization", "temporal", "no_candidate", "all_gated_out"}`.
  - `MeasurementAssociator.associate(candidates, *, predicted_measurement, innovation_covariance_for) -> AssociationResult`, where `predicted_measurement: NDArray | None` is `h(x̂⁻)` (None means no active track) and `innovation_covariance_for: Callable[[float], NDArray]` returns `S` for a candidate range.

Association lives in the belief layer, not in perception — perception only produces candidates. This keeps the hexagonal boundary intact.

**Invariant I1:** `innovation_covariance_for` is injected, not constructed here, precisely so that the coordinator can hand the *same* provider to the associator, the gate, and the EKF correction. The associator must never build an `S` of its own. Document this in the module docstring: `associate` thresholds against exactly the covariance its caller will later correct with.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_association.py
import numpy as np
import pytest

from duckie_pomdp.belief.measurement_association import (
    AssociationConfig,
    CandidateMeasurement,
    MeasurementAssociator,
)
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement

GATE = 9.21034037197618


def _candidate(range_m, bearing_rad, confidence, bbox=(0, 0, 10, 10)):
    from math import cos, sin

    return CandidateMeasurement(
        measurement=ObjectMeasurement(
            object_class=ObjectClass.DUCKIE,
            detected=True,
            confidence=confidence,
            x_left_m=range_m * sin(bearing_rad),
            y_forward_m=range_m * cos(bearing_rad),
            range_m=range_m,
            bearing_rad=bearing_rad,
        ),
        confidence=confidence,
        bbox_key=bbox,
    )


def _covariance_for(_range_m):
    return np.diag([4.0e-4, 1.6e-4])


def _associator():
    return MeasurementAssociator(
        AssociationConfig(
            chi_square_gate=GATE,
            initialization_rule="highest_confidence_then_bbox_lexicographic",
        )
    )


def test_without_an_active_track_the_highest_confidence_candidate_initializes():
    result = _associator().associate(
        [_candidate(0.90, 0.01, 0.42), _candidate(0.70, 0.05, 0.81)],
        predicted_measurement=None,
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "initialization"
    assert result.selected_index == 1


def test_initialization_breaks_exact_confidence_ties_deterministically():
    result = _associator().associate(
        [
            _candidate(0.90, 0.01, 0.50, bbox=(30, 0, 40, 10)),
            _candidate(0.70, 0.05, 0.50, bbox=(10, 0, 20, 10)),
        ],
        predicted_measurement=None,
        innovation_covariance_for=_covariance_for,
    )
    assert result.selected_index == 1, "lexicographically smallest bbox wins ties"


def test_with_an_active_track_the_most_consistent_candidate_wins_over_confidence():
    result = _associator().associate(
        [_candidate(0.62, 0.30, 0.95), _candidate(0.90, 0.02, 0.31)],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "temporal"
    assert result.selected_index == 1
    assert result.highest_confidence_index == 0
    assert result.differed_from_highest_confidence


def test_association_wraps_bearing_across_pi():
    from math import pi

    result = _associator().associate(
        [_candidate(0.90, -pi + 0.01, 0.40)],
        predicted_measurement=np.array([0.90, pi - 0.01]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "temporal"
    assert result.selected_index == 0
    assert result.candidate_nis[0] == pytest.approx(0.02**2 / 1.6e-4, rel=1e-6)


def test_every_candidate_outside_the_gate_yields_no_selection():
    result = _associator().associate(
        [_candidate(1.80, 0.50, 0.90)],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "all_gated_out"
    assert result.selected is None


def test_no_candidates_is_reported_distinctly_from_all_gated_out():
    result = _associator().associate(
        [],
        predicted_measurement=np.array([0.90, 0.02]),
        innovation_covariance_for=_covariance_for,
    )
    assert result.mode == "no_candidate"
    assert result.selected is None


def test_associate_signature_admits_no_privileged_argument():
    import inspect

    parameters = set(inspect.signature(MeasurementAssociator.associate).parameters)
    assert parameters == {
        "self",
        "candidates",
        "predicted_measurement",
        "innovation_covariance_for",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_association.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.measurement_association'`

- [ ] **Step 3: Write the implementation**

Key points: the innovation is `[z_r − ẑ_r, wrap_angle(z_β − ẑ_β)]` using `duckie_pomdp.perception.measurement_calibration.wrap_angle`; `S` comes from `innovation_covariance_for(candidate_range)`; selection is `argmin` over NIS subject to `nis <= chi_square_gate`; `highest_confidence_index` is computed with the same `(-confidence, bbox_key)` ordering as `select_single_duckie` so the diagnostic comparison is apples-to-apples. When `predicted_measurement is None`, skip NIS entirely and return `mode="initialization"` with `candidate_nis` all `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_association.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 139 passed.

---

## Task 5: Perception emits candidates

**Files:**
- Modify: `src/duckie_pomdp/perception/f9_pipeline.py`
- Test: `tests/test_f9_pipeline.py` (add tests; do not weaken existing ones)

**Interfaces:**
- Consumes: existing `YoloPedestrianMeasurementPipeline`, `Detection`, `ObjectClass`.
- Produces: `F9ImageObservation` gains a field `duckie_candidates: tuple[CandidateProjection, ...]` where `CandidateProjection(detection: Detection, measurement: ObjectMeasurement, projection_error: str | None)`. All existing fields keep their meaning and values so Baseline A is bit-identical.

- [ ] **Step 1: Write the failing tests**

```python
def test_pipeline_projects_every_duckie_candidate_not_only_the_selected_one():
    # Build a stub detector returning two Duckie boxes and one stop-sign box.
    observation = pipeline.observe(front_rgb)
    assert len(observation.duckie_candidates) == 2
    assert all(candidate.measurement.detected for candidate in observation.duckie_candidates)
    assert observation.duckie_detection_count == 2


def test_selected_duckie_still_matches_highest_confidence_for_baseline_parity():
    observation = pipeline.observe(front_rgb)
    best = max(observation.duckie_candidates, key=lambda item: item.detection.confidence)
    assert observation.selected_duckie is best.detection
    assert observation.pedestrian.range_m == best.measurement.range_m


def test_a_candidate_whose_projection_fails_is_kept_with_its_error_and_no_measurement():
    observation = pipeline.observe(front_rgb_with_horizon_box)
    failed = [c for c in observation.duckie_candidates if c.projection_error is not None]
    assert failed
    assert all(not c.measurement.detected for c in failed)


def test_no_duckie_detections_yields_no_candidates_and_a_missing_measurement():
    observation = pipeline.observe(empty_rgb)
    assert observation.duckie_candidates == ()
    assert not observation.pedestrian.detected
```

Reuse whatever stub-detector fixture `tests/test_f9_pipeline.py` already defines rather than writing a new one; read that file first and follow its existing pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9_pipeline.py -q`
Expected: FAIL — `AttributeError: 'F9ImageObservation' object has no attribute 'duckie_candidates'`

- [ ] **Step 3: Implement**

In `observe()`, project every Duckie detection (wrapping each `project_raw` call in its own `try/except ValueError`), build the candidate tuple, then keep the existing selection logic untouched to populate `pedestrian`, `selected_duckie`, `duplicate_selection`, and `projection_error`. Add `duckie_candidates` as the last field of `F9ImageObservation` with a default of `()` so any positional construction elsewhere keeps working.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9_pipeline.py -q`
Expected: PASS — all pre-existing tests plus 4 new ones.

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 143 passed. Confirm in the notes that `selected_duckie` semantics are unchanged, because Baseline A depends on it.

---

## Task 6: Covariance calibration — R inflation and posterior floor

**Files:**
- Create: `src/duckie_pomdp/belief/covariance_calibration.py`
- Test: `tests/test_f9c_covariance_calibration.py`

**Interfaces:**
- Consumes: NumPy only.
- Produces:
  - `CovarianceCalibration(range_scale: float, bearing_scale: float, range_posterior_floor_m: float, bearing_posterior_floor_rad: float)` with methods `inflate(measurement_covariance: NDArray) -> NDArray` and `floor_polar_standard_deviation(range_std_m: float, bearing_std_rad: float) -> tuple[float, float]`.
  - `VarianceComponents(between_group_variance: float, within_group_variance: float, group_count: int, sample_count: int)`.
  - `estimate_variance_components(residuals_by_group: Mapping[str, Sequence[float]]) -> VarianceComponents` — the one-level primitive.
  - `NestedVarianceComponents(seed_variance: float, episode_variance: float, within_variance: float, seed_count: int, episode_count: int, sample_count: int)`.
  - `estimate_nested_variance_components(residuals_by_seed_episode: Mapping[tuple[str, str], Sequence[float]]) -> NestedVarianceComponents` — applies the one-level estimator twice: once with seeds as groups to get `seed_variance`, once on the seed-centred residuals with episodes as groups to get `episode_variance`.
  - `posterior_floor_from_components(components: NestedVarianceComponents) -> float` implementing

    ```text
    sqrt( τ_seed² + τ_episode² + τ_seed²/n_seeds + τ_episode²/n_episodes )
    ```

**Methodological label — state this in the module docstring and in the artifact.** This is an *approximate nested variance-component estimator*: the one-level ANOVA moment estimator applied twice, to seeds and then to seed-centred residuals by episode. It is **not** a REML mixed-effects fit, and must not be described as one. It gives no standard errors on the components themselves and can be biased under strong imbalance. That is acceptable here because the target is an uncertainty *floor* — a quantity that only needs to be right to within a modest factor to fix coverage — and because the constraint is NumPy-only with no new dependencies. Anyone reading the artifact should be able to see the approximation rather than infer a rigor that is not there.

**Why nested rather than one-level.** Finding 1 shows range offset is carried almost entirely at the seed level while bearing offset is carried at the episode level. A single-level fit grouped by episode would use `τ̂/√n_episodes` for `SE(b̂)` on range, understating that term by roughly 3× because episodes inside one seed are not independent draws of the range offset. The nested fit gets both variables right with one estimator and no per-variable special-casing — which matters, since the correct grouping is an empirical property that may differ again on 6101–6108.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_covariance_calibration.py
import numpy as np
import pytest

from duckie_pomdp.belief.covariance_calibration import (
    CovarianceCalibration,
    estimate_variance_components,
    posterior_floor_from_components,
)


def test_inflation_scales_range_and_bearing_variance_independently():
    calibration = CovarianceCalibration(4.0, 1.0, 0.0, 0.0)
    inflated = calibration.inflate(np.diag([1.0e-4, 4.0e-4]))
    assert inflated[0, 0] == pytest.approx(4.0e-4)
    assert inflated[1, 1] == pytest.approx(4.0e-4)


def test_inflated_covariance_stays_positive_semidefinite():
    calibration = CovarianceCalibration(7.3, 2.1, 0.0, 0.0)
    inflated = calibration.inflate(np.diag([1.0e-6, 1.0e-8]))
    assert float(np.linalg.eigvalsh(inflated).min()) > 0.0


def test_scales_below_one_are_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        CovarianceCalibration(0.5, 1.0, 0.0, 0.0)


def test_posterior_floor_adds_in_quadrature_and_never_shrinks_uncertainty():
    calibration = CovarianceCalibration(1.0, 1.0, 0.016, 0.004)
    range_std, bearing_std = calibration.floor_polar_standard_deviation(0.006, 0.005)
    assert range_std == pytest.approx((0.006**2 + 0.016**2) ** 0.5)
    assert bearing_std == pytest.approx((0.005**2 + 0.004**2) ** 0.5)
    assert range_std > 0.006 and bearing_std > 0.005


def test_variance_components_recover_a_known_between_group_offset():
    rng = np.random.default_rng(20260808)
    groups = {}
    for index in range(40):
        offset = rng.normal(0.0, 0.014)
        groups[f"episode_{index}"] = list(rng.normal(offset, 0.0073, size=40))
    components = estimate_variance_components(groups)
    assert components.between_group_variance**0.5 == pytest.approx(0.014, abs=0.004)
    assert components.within_group_variance**0.5 == pytest.approx(0.0073, abs=0.002)
    assert components.group_count == 40


def test_variance_components_do_not_report_negative_between_group_variance():
    rng = np.random.default_rng(7)
    groups = {f"e{i}": list(rng.normal(0.0, 0.01, size=50)) for i in range(20)}
    components = estimate_variance_components(groups)
    assert components.between_group_variance >= 0.0


def test_posterior_floor_includes_both_levels_and_the_bias_estimation_error():
    from duckie_pomdp.belief.covariance_calibration import NestedVarianceComponents

    components = NestedVarianceComponents(
        seed_variance=0.0155**2,
        episode_variance=0.0048**2,
        within_variance=0.0074**2,
        seed_count=8,
        episode_count=80,
        sample_count=3200,
    )
    floor = posterior_floor_from_components(components)
    expected = (
        0.0155**2 + 0.0048**2 + 0.0155**2 / 8 + 0.0048**2 / 80
    ) ** 0.5
    assert floor == pytest.approx(expected, rel=1e-9)
    assert floor > 0.0155, "the floor must exceed the seed component alone"


def test_the_bias_estimation_error_uses_the_seed_count_not_the_episode_count():
    """Finding 1: episodes inside one seed are not independent draws of the
    range offset. Dividing the seed component by the episode count would
    understate the floor."""
    from duckie_pomdp.belief.covariance_calibration import NestedVarianceComponents

    correct = NestedVarianceComponents(0.0155**2, 0.0048**2, 0.0074**2, 8, 80, 3200)
    wrong_if_flat = (0.0155**2 + 0.0048**2 + 0.0155**2 / 80 + 0.0048**2 / 80) ** 0.5
    assert posterior_floor_from_components(correct) > wrong_if_flat


def test_nested_components_recover_a_known_two_level_structure():
    from duckie_pomdp.belief.covariance_calibration import (
        estimate_nested_variance_components,
    )

    rng = np.random.default_rng(20260808)
    groups = {}
    for seed_index in range(8):
        seed_offset = rng.normal(0.0, 0.0155)
        for episode_index in range(10):
            episode_offset = seed_offset + rng.normal(0.0, 0.0048)
            groups[(f"s{seed_index}", f"s{seed_index}_e{episode_index}")] = list(
                rng.normal(episode_offset, 0.0074, size=40)
            )
    components = estimate_nested_variance_components(groups)
    assert components.seed_variance**0.5 == pytest.approx(0.0155, abs=0.006)
    assert components.episode_variance**0.5 == pytest.approx(0.0048, abs=0.003)
    assert components.within_variance**0.5 == pytest.approx(0.0074, abs=0.001)
    assert components.seed_count == 8 and components.episode_count == 80
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_covariance_calibration.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`estimate_variance_components` uses the standard unbalanced one-way random-effects estimator: within-group variance is the pooled residual mean square about each group mean, `MS_within = ΣᵢΣⱼ(x_ij − x̄ᵢ)²/(N−k)`; between-group variance is `max(0, (MS_between − MS_within)/n_effective)` with `MS_between = Σᵢnᵢ(x̄ᵢ − x̄)²/(k−1)` and `n_effective = (N − Σnᵢ²/N)/(k−1)`. Clamp at zero and document why (Step-1 test `test_variance_components_do_not_report_negative_between_group_variance`).

`estimate_nested_variance_components` calls that primitive twice — first with seeds as groups, then on seed-centred residuals with episodes as groups — and reports `within_variance` from the episode-level pass. Do not substitute the SD of the group means anywhere: that quantity is inflated by unbalanced group sizes and by the sampling noise of each mean, and it is precisely the error that produced a discarded `0.01562 rad` figure during plan review.

`floor_polar_standard_deviation` is applied at the belief-reporting boundary only. Add a module docstring stating explicitly:

> The posterior floor represents observation bias that varies slowly within an episode and is therefore not averaged away by the EKF. Version 1 does not augment the frozen EKF state with a bias term; the floor is a documented approximation of that missing state, calibrated from measured between-episode variance.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_covariance_calibration.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 152 passed.

---

## Task 7: Predicted observability and effective detection probability

**Files:**
- Create: `src/duckie_pomdp/belief/observability.py`
- Modify: `src/duckie_pomdp/belief/existence_filter.py`
- Test: `tests/test_f9c_observability.py`, `tests/test_f9c_existence.py`

**Interfaces:**
- Consumes: `duckie_pomdp.perception.camera_geometry` (read it first to reuse the existing intrinsics/extrinsics types — do not re-derive the projection), the frozen `PedestrianEKF` predicted state.
- Produces:
  - `ObservabilityClass` str-enum: `CENTER`, `MID_FOV`, `EDGE_FOV`, `OUTSIDE_DOMAIN`.
  - `PredictedObservability(observability_class: ObservabilityClass, normalized_horizontal_offset: float | None, predicted_range_m: float)`.
  - `PredictedObservabilityModel(projector, image_width_px: int).classify(predicted_state: NDArray) -> PredictedObservability`.
  - `EffectiveDetectionModel(probability_by_class: Mapping[ObservabilityClass, float], *, outside_domain_miss_policy: str)` with two methods: `probability(observability) -> float` and `miss_is_informative(observability) -> bool`, the latter returning `False` for `OUTSIDE_DOMAIN` when the policy is `"prediction_only"` (invariant I3).
  - `ExistenceFilterConfig` gains `detection_probability` as a *default*; `ExistenceFilter.update(detected: bool, *, detection_probability: float | None = None, observation_informative: bool = True) -> float`. With `observation_informative=False` the filter runs the `P_S`/`P_birth` prediction step and returns without applying any likelihood. Defaults (`None`, `True`) reproduce F9b behaviour exactly.

The `probability` / `miss_is_informative` split is what keeps invariant I3 honest: `P_D^eff(OUTSIDE_DOMAIN)` is still *estimated* and reported as a diagnostic, but it is never *applied* to a miss. That prevents the number from silently becoming a tuning knob.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_observability.py
import numpy as np
import pytest

from duckie_pomdp.belief.observability import (
    EffectiveDetectionModel,
    ObservabilityClass,
    PredictedObservabilityModel,
)


def test_pedestrian_predicted_straight_ahead_is_center(model):
    predicted = model.classify(np.array([0.0, 0.85, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.CENTER
    assert predicted.normalized_horizontal_offset == pytest.approx(0.0, abs=1e-6)


def test_pedestrian_predicted_far_to_the_side_is_edge_fov(model):
    predicted = model.classify(np.array([0.45, 0.60, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.EDGE_FOV


def test_pedestrian_predicted_beyond_the_image_is_outside_domain(model):
    predicted = model.classify(np.array([3.0, 0.40, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.OUTSIDE_DOMAIN


def test_pedestrian_predicted_behind_the_camera_is_outside_domain(model):
    predicted = model.classify(np.array([0.0, -0.50, 0.0, 0.0]))
    assert predicted.observability_class is ObservabilityClass.OUTSIDE_DOMAIN
    assert predicted.normalized_horizontal_offset is None


def test_classification_uses_no_privileged_input(model):
    import inspect

    parameters = set(inspect.signature(PredictedObservabilityModel.classify).parameters)
    assert parameters == {"self", "predicted_state"}


def _detection_model(policy="prediction_only"):
    return EffectiveDetectionModel(
        {
            ObservabilityClass.CENTER: 0.99,
            ObservabilityClass.MID_FOV: 0.97,
            ObservabilityClass.EDGE_FOV: 0.72,
            ObservabilityClass.OUTSIDE_DOMAIN: 0.05,
        },
        outside_domain_miss_policy=policy,
    )


def test_effective_detection_probability_is_lower_at_the_edge_of_the_field_of_view():
    from duckie_pomdp.belief.observability import PredictedObservability

    model = _detection_model()
    center = PredictedObservability(ObservabilityClass.CENTER, 0.0, 0.85)
    edge = PredictedObservability(ObservabilityClass.EDGE_FOV, 0.8, 0.85)
    assert model.probability(center) == 0.99
    assert model.probability(edge) == 0.72


def test_an_outside_domain_miss_is_declared_uninformative():
    from duckie_pomdp.belief.observability import PredictedObservability

    model = _detection_model()
    outside = PredictedObservability(ObservabilityClass.OUTSIDE_DOMAIN, None, 0.85)
    edge = PredictedObservability(ObservabilityClass.EDGE_FOV, 0.8, 0.85)
    assert not model.miss_is_informative(outside)
    assert model.miss_is_informative(edge)
    # The probability is still reported, it is simply never applied to a miss.
    assert model.probability(outside) == 0.05


def test_effective_detection_model_rejects_a_missing_class():
    with pytest.raises(ValueError, match="every observability class"):
        EffectiveDetectionModel(
            {ObservabilityClass.CENTER: 0.99},
            outside_domain_miss_policy="prediction_only",
        )
```

```python
# tests/test_f9c_existence.py
import pytest

from duckie_pomdp.belief.existence_filter import ExistenceFilter, ExistenceFilterConfig

CONFIG = ExistenceFilterConfig(
    prior_probability=0.50,
    detection_probability=0.9766775777414075,
    false_positive_probability=0.00078003120124805,
    survival_probability=0.995,
    birth_probability=0.005,
)


def test_default_update_reproduces_the_frozen_f9b_collapse():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    assert existence.update(False) == pytest.approx(0.614, abs=0.01)
    assert existence.update(False) == pytest.approx(0.036, abs=0.01)


def test_a_low_effective_detection_probability_preserves_belief_through_misses():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(5):
        existence.update(False, detection_probability=0.10)
    assert existence.probability > 0.60


def test_existence_still_decays_monotonically_under_repeated_misses():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    values = [existence.update(False, detection_probability=0.30) for _ in range(20)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert values[-1] < 0.10


def test_belief_recovers_rapidly_after_re_detection():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(5):
        existence.update(False, detection_probability=0.30)
    recovered = existence.update(True, detection_probability=0.98)
    assert recovered > 0.95


def test_detection_probability_override_must_exceed_the_false_alarm_rate():
    existence = ExistenceFilter(CONFIG)
    with pytest.raises(ValueError, match="false-positive"):
        existence.update(False, detection_probability=0.0001)


def test_an_uninformative_observation_applies_no_likelihood_at_all():
    """Invariant I3. An outside-domain miss must move P(e) by the survival
    prediction only, never by the miss likelihood ratio."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.90
    expected = 0.995 * 0.90 + 0.005 * (1.0 - 0.90)
    assert existence.update(False, observation_informative=False) == pytest.approx(
        expected
    )


def test_outside_domain_miss_decays_only_through_survival():
    """Invariant I3, over a long absence: 40 uninformative misses must leave
    P(e) far above the in-domain collapse, decaying at the P_S half-life."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.99
    for _ in range(40):
        existence.update(False, observation_informative=False)
    assert existence.probability > 0.80
    informative = ExistenceFilter(CONFIG)
    informative.probability = 0.99
    for _ in range(40):
        informative.update(False, detection_probability=0.97)
    assert informative.probability < 0.01


def test_an_uninformative_observation_ignores_any_detection_probability_passed():
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.90
    quiet = existence.update(
        False, detection_probability=0.0001, observation_informative=False
    )
    assert quiet == pytest.approx(0.995 * 0.90 + 0.005 * 0.10)


def test_a_detection_still_counts_when_the_belief_predicted_outside_domain():
    """A detection is always evidence, even from a region we predicted was
    unobservable -- that is exactly the signal the prediction was wrong."""
    existence = ExistenceFilter(CONFIG)
    existence.probability = 0.20
    updated = existence.update(True, detection_probability=0.97)
    assert updated > 0.90


def test_ps_is_untouched_by_f9c():
    assert CONFIG.survival_probability == 0.995
```

Build the `model` fixture from the real calibration in `configs/scenario_pomdp_v1.toml` via `CalibratedGroundProjector`, mirroring how `tests/test_camera_geometry.py` constructs one — read that file first.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q`
Expected: FAIL — `ModuleNotFoundError` for observability; `TypeError: update() got an unexpected keyword argument 'detection_probability'` for existence.

- [ ] **Step 3: Implement**

`PredictedObservabilityModel.classify` takes the *predicted* EKF state `[x_left, y_forward, …]`, converts to an image column with the existing calibrated projection, and bins `|u − W/2| / (W/2)` with the same thresholds already used by `_fov_region` in `experiments/validate_f9_yolo_ekf.py:62-71` (`<1/3` center, `<2/3` mid, else edge). Return `OUTSIDE_DOMAIN` when `y_forward <= 0` (behind camera) or the projected column falls outside `[0, W)`. Reuse the thresholds by importing or duplicating them with a comment pointing at the source — the evaluation binning and the runtime binning must agree or the calibrated `P_D^eff` will not apply.

In `ExistenceFilter.update`:

```python
def update(
    self,
    detected: bool,
    *,
    detection_probability: float | None = None,
    observation_informative: bool = True,
) -> float:
    config = self.config
    predicted = (
        config.survival_probability * self.probability
        + config.birth_probability * (1.0 - self.probability)
    )
    if not observation_informative:
        # Invariant I3: the camera cannot inform us about a region the belief
        # predicts is unobservable. Prediction only, no likelihood applied.
        self.probability = min(1.0, max(0.0, predicted))
        return self.probability
    probability = (
        config.detection_probability
        if detection_probability is None
        else detection_probability
    )
    if not 0.0 <= probability <= 1.0 or probability <= config.false_positive_probability:
        raise ValueError(
            "effective detection probability must exceed the false-positive rate"
        )
    ...  # existing numerator/denominator arithmetic, unchanged
```

Note the ordering: the validation lives *after* the `observation_informative` early return, so `test_an_uninformative_observation_ignores_any_detection_probability_passed` passes. Do **not** touch the `P_S`/`P_birth` prediction step in either branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_observability.py tests/test_f9c_existence.py -q`
Expected: PASS (18 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 170 passed. The existing `tests/test_pedestrian_ekf.py` existence tests must still pass unmodified — that is the proof the default path is unchanged.

---

## Task 8: Robust belief coordinator

**Files:**
- Create: `src/duckie_pomdp/belief/robust_updater.py`
- Modify: `src/duckie_pomdp/belief/__init__.py`
- Test: `tests/test_f9c_robust_updater.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 4, 6, 7, plus the frozen `PedestrianEKF`, `ExistenceFilter`, `PedestrianBelief`.
- Produces:
  - `RobustObservationConfig(switches, gate, association, covariance, effective_detection, active_threshold, delete_threshold, initialization_threshold)`.
  - `RobustStepRecord(frame_mode: str, detector_detected: bool, kinematic_measurement_accepted: bool, association: AssociationResult, gate: GateDecision | None, effective_detection_probability: float, observation_informative: bool, observability_class: ObservabilityClass, existence_probability: float, track_active: bool, track_deleted: bool, nis: float | None, reported_range_std_m: float, reported_bearing_std_rad: float)`.
  - `RobustPedestrianBeliefUpdater.update(previous_belief, previous_action, ego_motion, candidates, dt_s) -> tuple[BeliefState, RobustStepRecord]`.
  - Private `_innovation_covariance(range_m) -> NDArray` — the **single** provider satisfying invariant I1, returning `H P⁻ Hᵀ + λR(range_m)` from the current predicted covariance.

Order of operations, fixed:

```text
 1. EKF predict with actual ego motion                          (frozen)
 2. classify predicted observability from x̂⁻                    (Task 7)
 3. resolve P_D^eff and miss_is_informative for this frame      (Task 7)
 4. detector_detected = any raw candidate exists in this frame  <- existence evidence
4b. apply the FROZEN F9c BIAS CORRECTION to every candidate     (Task 3b)
       z_raw -> z_corr, before anything reads a candidate range
 5. build ONE innovation-covariance provider from λR and P⁻     (I1)
 6. associate corrected candidates against h(x̂⁻) using it       (Task 4)
 7. gate the associated candidate on NIS using that provider    (Task 3)
 8. kinematic_measurement_accepted = association selected AND gate accepted
 9. if accepted -> EKF correct using that same provider's R     (Tasks 3+6)
    else        -> prediction only, filter state already advanced by step 1
10. existence update:
       observation_informative = miss_is_informative(observability) OR detector_detected
       existence.update(detected=detector_detected,
                        detection_probability=P_D^eff,
                        observation_informative=observation_informative)
11. delete the track if P(e) < delete_threshold
12. report belief with the posterior floor applied              (Task 6)
```

**Step 10 is the fix for invariant I2 and is the single most consequential line in this plan.** Existence is driven by `detector_detected` — the detector's answer to "is a Duckie in this image" — *not* by `kinematic_measurement_accepted`. A frame where YOLO finds the pedestrian but the gate rejects the bbox is scored as a **detection** for existence and as **prediction-only** for the EKF. That is the estimator saying "I believe it is there, I do not believe these coordinates."

The `OR detector_detected` in the `observation_informative` expression implements the asymmetry of invariant I3: a miss predicted outside the domain applies no likelihood, but a *detection* always does, whatever the predicted class was.

**Step 4b placement matters.** `detector_detected` is read from the **raw** candidate list at step 4, because whether the detector saw a Duckie cannot depend on a metric correction. Everything downstream — association, gating, correction — consumes only corrected candidates, so a candidate range never reaches an innovation computation uncorrected.

The updater holds exactly one `self._bias: FrozenBiasCorrection`, resolved from the switch:

```text
bias_refit = false  ->  F9c bias stage loaded with the F9b FROZEN constants
                        b_r = -0.045904804710162034, b_beta = +0.00414567890700929
bias_refit = true   ->  F9c bias stage loaded with the F9c FITTED constants
```

Not `identity()` when the switch is off — Baseline A applies the F9b correction, so an identity fallback would make `all switches off` differ from Baseline A by the entire F9b bias and would silently invalidate `test_switching_every_robust_component_off_reproduces_the_f9b_path`. `identity()` exists for unit tests only and must not appear in any ablation configuration.

**Initialization is the deliberate exception.** With no active track there is no innovation to test, so the two claims cannot be separated. A track is created only when `kinematic_measurement_accepted` is true — a candidate that fails projection or falls outside the association gate must not create a track. State transition, to be documented verbatim in `IMPLEMENTATION_NOTES.md`:

```text
no track  + accepted candidate + P(e) >= initialization_threshold  -> track created (frame_mode="initialization")
no track  + detection only                                          -> existence updates, no track
active    + accepted candidate                                      -> correct        (frame_mode="temporal")
active    + detection, gate rejected                                -> predict only, existence counts a DETECTION
active    + no detection, in-domain                                 -> predict only, existence counts a MISS
active    + no detection, outside domain                            -> predict only, existence prediction step only
active    + P(e) < delete_threshold                                 -> track deleted (frame_mode="deleted")
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_robust_updater.py
def test_a_gross_outlier_does_not_move_the_belief_more_than_the_baseline_does(...):
    """Feed 10 consistent measurements then one 0.30 m outlier.
    Robust belief range error after the outlier must be < 0.5 x baseline error."""


def test_reported_range_std_is_never_below_the_posterior_floor(...):
    """After 60 consistent updates the EKF std collapses; the reported std must
    still be >= range_posterior_floor_m."""


def test_a_duplicate_frame_selects_the_temporally_consistent_candidate(...):
    """Two candidates, the higher-confidence one 0.25 m off track.
    record.association.differed_from_highest_confidence is True and the selected
    measurement is the consistent one."""


def test_a_rejected_localization_is_not_an_existence_miss(...):
    """Invariant I2. Establish a track, then feed 6 frames whose single candidate
    is a 0.30 m gross outlier. Every frame must record
    detector_detected=True, kinematic_measurement_accepted=False, and
    track_active=True; existence must stay above the active threshold.
    Contrast case: the same 6 frames with NO candidate at all must drive
    existence below the active threshold. The two sequences must differ."""


def test_association_gate_and_correction_share_one_innovation_covariance(...):
    """Invariant I1. Monkeypatch the updater's _innovation_covariance to record
    every matrix it returns within one update() call, and assert that the
    matrix used by the associator, the matrix passed to the gate, and the
    matrix implied by the EKF correction's R are identical to 1e-15.
    Additionally assert the returned R equals calibration.inflate(base_R),
    not base_R -- a lambda fitted on inflated S must be applied to inflated S."""


def test_belief_survives_five_consecutive_misses_when_predicted_edge_fov(...):
    """record.track_active stays True through 5 misses at EDGE_FOV."""


def test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one(...):
    """Invariant I3 at the coordinator level. 30 missing frames with the belief
    predicted OUTSIDE_DOMAIN must leave P(e) > 0.80 and track_deleted False;
    the same 30 frames predicted CENTER must delete the track."""


def test_belief_still_dies_after_a_long_absence(...):
    """After 40 consecutive misses at CENTER, P(e) < delete_threshold and
    record.track_deleted becomes True exactly once."""


def test_a_deleted_track_reinitializes_from_the_next_valid_candidate(...):
    """After deletion, one good candidate produces frame_mode == 'initialization'
    and an initialized EKF."""


def test_switching_every_robust_component_off_reproduces_the_f9b_path(...):
    """With all five switches False, the belief sequence must equal
    PedestrianBeliefUpdater's output to within 1e-12 on the same inputs.
    This test is the Baseline-A regression guard."""


def test_updater_never_receives_privileged_state(...):
    import inspect
    parameters = set(
        inspect.signature(RobustPedestrianBeliefUpdater.update).parameters
    )
    assert parameters == {
        "self", "previous_belief", "previous_action", "ego_motion",
        "candidates", "dt_s",
    }
```

Fill each docstring stub with a concrete arrangement using the same synthetic-measurement helpers `tests/test_pedestrian_ekf.py` already uses — read that file and reuse its fixtures rather than inventing new ones.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.belief.robust_updater'`

- [ ] **Step 3: Implement**

Call the *existing* `PedestrianEKF.predict` / `PedestrianEKF.correct`; do not reimplement filter mathematics. For the λ-inflated correction, wrap the EKF's `measurement_noise` in a small adapter whose `covariance()` returns `calibration.inflate(base.covariance(r))` — this keeps `pedestrian_ekf.py` untouched, which is what "frozen" must mean in practice. Apply the posterior floor only in the reported `PedestrianBelief`, never to `self.ekf._covariance`; add a comment saying so, because writing the floor back into the filter state would corrupt the next prediction.

**Invariant I1 in code.** Construct the inflated noise adapter *once* in `__init__` and install it on the EKF. Then `_innovation_covariance(range_m)` reads `H` and `P⁻` from the EKF after `predict()` and returns `H @ P⁻ @ H.T + adapter.covariance(range_m)`. Hand that one bound method to the associator as `innovation_covariance_for`, use its output for the gate, and let the EKF correction consume the same adapter. There must be exactly one call site constructing `S` in the whole module:

```bash
grep -n "covariance(" src/duckie_pomdp/belief/robust_updater.py
```

If more than one expression builds `H P Hᵀ + R`, the invariant is already broken.

Two tests carry the weight of this task. `test_switching_every_robust_component_off_reproduces_the_f9b_path` — if it does not pass exactly, the ablation in Task 12 is meaningless. `test_a_rejected_localization_is_not_an_existence_miss` — if it does not pass, the gate makes existence collapse *worse* than F9b, which is the specific failure mode this whole gate exists to prevent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 181 passed.

---

## Task 9: Calibration experiment on seeds 6101–6108

**Files:**
- Create: `experiments/calibrate_f9c_robust_belief.py`
- Create: `src/duckie_pomdp/evaluation/f9c_calibration.py`
- Test: covered by Task 13 leakage tests plus unit tests below

**Interfaces:**
- Consumes: `F9cProtocol`, `estimate_nested_variance_components`, `posterior_floor_from_components`. **Use the nested estimator, not the one-level `estimate_variance_components`** — the latter is a primitive that the nested one calls internally, and reaching for it directly here would silently reintroduce the `SE(b̂) = τ̂/√n_episodes` error that Finding 1 rejects. Group the residuals by `(seed, episode)`.
- Produces: `fit_bias(rows, *, model: str) -> dict`, `leave_one_seed_out_range_rmse(rows, *, model: str) -> float`, `select_bias_model(rows) -> tuple[str, dict]`, `fit_covariance_scales(rows, *, bias) -> tuple[float, float]`, `fit_effective_detection(rows) -> dict[str, float]`, and `artifacts/f9c_calibration.csv` + `artifacts/f9c_calibration_metrics.json`.

The CSV must carry, per frame: `episode, seed, scenario, frame, eligible_visible, detector_detected, kinematic_measurement_accepted, duckie_detection_count, candidate_count, selected_confidence, selected_iou, selected_correct_iou50, raw_range_m, raw_bearing_rad, gt_range_m, gt_bearing_rad, range_error_m, bearing_error_rad, distance_bin, fov_region, predicted_observability_class, predicted_nis`.

`predicted_observability_class` is computed at runtime from the belief; `fov_region` stays the GT-derived evaluation label. Both are needed: one to fit `P_D^eff`, one to audit whether the runtime prediction agrees with reality. Report the confusion between them in the calibration metrics — a predicted-observability model that disagrees badly with the GT FOV region is a finding, not a detail.

`detector_detected` and `kinematic_measurement_accepted` are separate columns because invariant I2 makes them separate quantities, and because `fit_effective_detection` must condition on the **detector** flag. Fitting `P_D^eff` on the acceptance flag would fold localization quality into a detection probability and re-introduce exactly the conflation Finding 6 rejects.

- [ ] **Step 1: Write the failing unit tests for the fitting rules**

```python
# add to tests/test_f9c_covariance_calibration.py
def test_bias_model_selection_prefers_global_when_per_bin_does_not_generalize():
    """Synthesize rows whose per-bin differences are pure seed artefacts.
    select_bias_model must return "global_additive"."""


def test_bias_model_selection_picks_per_bin_only_on_a_genuine_range_dependence():
    """Synthesize a true linear bias(r) with >=100 samples in each of 3 bins from
    >=3 scenarios. select_bias_model must return "per_range_bin"."""


def test_leave_one_seed_out_holds_out_whole_seeds_not_frames():
    """A rule that shuffles frames would report a far lower RMSE; assert the
    LOSO RMSE exceeds the in-sample RMSE on seed-correlated synthetic data."""


def test_lambda_is_fitted_to_the_nis_median_not_to_coverage():
    # "lambda" here is the R-inflation scale, not the gate covariance knob --
    # that knob no longer exists. See "Why there is no downweight mode".
    """With residuals whose true variance is 4x the modelled variance,
    fit_covariance_scales must return range_scale ~ 4 (rel 0.25)."""


def test_lambda_fitting_set_is_selected_by_ground_truth_not_by_the_gate():
    """Invariant I6. Build calibration rows where one sample has a large
    innovation but selected_correct_iou50=True, and another has a small
    innovation but selected_correct_iou50=False. fit_covariance_scales must
    include the first and exclude the second -- proving selection is by GT
    match, not by any NIS threshold. Then assert the returned lambda is
    unchanged when the rows are re-fed in a different order, and that fitting
    twice with lambda seeded at 1.0 and at 10.0 converges to the same value:
    a gate-conditioned set would not be seed-independent."""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_covariance_calibration.py -q`
Expected: FAIL — `ImportError: cannot import name 'select_bias_model'`

- [ ] **Step 3: Implement `f9c_calibration.py`**

`select_bias_model` applies the pre-specified rule verbatim: per-bin wins only if every bin has ≥100 matched samples drawn from ≥3 distinct scenarios **and** LOSO held-out range RMSE improves by ≥10% relative.

`fit_covariance_scales` solves for `λ_r`, `λ_β` such that the calibration NIS median matches the χ²₂ median `1.3862943611198906`, using the same `S` the runtime would compute — the invariant-I1 provider, not a separately assembled matrix.

**The fitting set is selected by ground truth, not by the gate (invariant I6).** An earlier draft said "fit `λ` on accepted-and-visible frames", which is circular: acceptance depends on `S`, `S` depends on `λ`, and `λ` is what is being fitted. A frame whose NIS is 12 at `λ = 1` would be excluded, even though at the fitted `λ = 4` its NIS would be 3 and it would have been accepted — so the fit would be conditioned on a decision boundary that the fit itself invalidates. The set must therefore be defined by an external criterion:

```text
lambda fitting set (calibration only, offline):
    eligible_visible                       (GT silhouette)
    AND valid projected measurement
    AND correct class
    AND selected_correct_iou50             (GT IoU >= 0.50)
```

Both columns already exist in the CSV (`selected_iou`, `selected_correct_iou50`), so no new machinery is required. Using GT here is legitimate — this is offline calibration, exactly as F9a used GT to fit its bias and sigmas; nothing in this path reaches runtime.

The intent is unchanged: gross localization mismatches stay out of the `λ` fit, because they are what the gate exists to reject and folding them in would inflate `λ` until the gate admitted them. Only the *selection rule* changes, from a self-referential one to an external one.

Report the excluded-sample count and their NIS distribution in `artifacts/f9c_calibration_metrics.json`, so the fit's blind spot is visible rather than implied.

`fit_effective_detection` computes a Beta(0.5, 0.5) posterior mean of `P(detector_detected | pedestrian exists, predicted_observability_class = c)` for each of the four classes, where existence comes from offline GT and the class comes from the runtime prediction. Record trial counts per class. If `OUTSIDE_DOMAIN` has fewer than 30 trials, say so — the value is a reported diagnostic and under invariant I3 it is never applied to a miss, so a thin count there is disclosed rather than blocking.

- [ ] **Step 4: Run the calibration**

```bash
$PY experiments/calibrate_f9c_robust_belief.py --config configs/f9c_robust_belief_v1.toml
```

Expected: `artifacts/f9c_calibration.csv` with roughly 8 seeds × 10 scenarios of frames, and `artifacts/f9c_calibration_metrics.json` containing the fitted `b_r`, `b_β`, chosen bias model with its LOSO evidence, `λ_r`, `λ_β`, variance components `τ̂`/`σ̂_w`, `σ_floor,r`, `σ_floor,β`, and the four `P_D^eff` values with their trial counts.

- [ ] **Step 5: Sanity-check the fit against the plan's predictions**

Compare against the predictions in "Empirical Basis":

```text
τ̂_seed,range      expected ≈ 0.012 – 0.018 m   (F9a random-effects, k=4, noisy)
τ̂_episode,range   expected small vs τ̂_seed     (range offset is seed-carried)
σ̂_w,range         expected ≈ 0.0074 m          (F9a: 0.00739)
σ_floor,r         expected ≈ 0.015 – 0.018 m

τ̂_seed,bearing    expected small vs τ̂_episode  (bearing offset is episode-carried)
σ̂_w,bearing       expected ≈ 0.0046 rad        (F9a: 0.00455)
σ_floor,β         expected ≈ 0.012 – 0.016 rad

λ_r               expected ≈ 3 – 8
P_D^eff EDGE_FOV  expected materially below P_D^eff CENTER
```

The **structural** predictions — range offset seed-carried, bearing offset episode-carried — are the ones to take seriously; the F9a magnitudes come from only four seeds. If the structure inverts on 6101–6108, that is a real finding: report it and let the nested estimator produce whatever floor the data supports, since the formula handles either structure without modification.

If any value is wildly off, **stop and diagnose** — do not adjust the target to match the result. Record the comparison in `IMPLEMENTATION_NOTES.md`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_covariance_calibration.py -q`
Expected: PASS (14 tests)

- [ ] **Step 7: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 186 passed.

---

## Task 10: Freeze the configuration

**Files:**
- Modify: `configs/f9c_robust_belief_v1.toml`
- Create: `artifacts/f9c_frozen_config.json`
- Create: `experiments/verify_f9c_artifacts.py`

- [ ] **Step 1: Write the fitted values into the config and set all three `parameters_frozen = true`**

Copy every number from `artifacts/f9c_calibration_metrics.json` into `[measurement_model]`, `[covariance_calibration]`, and `[conditional_detection]`. Set `parameters_frozen = true` in all three sections.

- [ ] **Step 2: Write `artifacts/f9c_frozen_config.json`**

Contains `config_sha256`, `checkpoint_sha256`, `calibration_artifact_sha256`, `frozen_f7_config_sha256`, `calibration_seeds`, `final_evaluation_seeds`, the full fitted parameter set, the pre-specified acceptance bands, and the pre-specified minimum support — plus an ISO timestamp and the literal statement `"final_evaluation_seeds_not_yet_rendered": true`.

- [ ] **Step 3: Verify `require_frozen=True` now loads**

```bash
$PY -c "
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
p = load_f9c_protocol(Path('configs/f9c_robust_belief_v1.toml'), require_frozen=True)
print('frozen config sha256:', p.config_sha256)
"
```

Expected: prints a hash and does not raise.

- [ ] **Step 4: Write `experiments/verify_f9c_artifacts.py`**

Model it on `experiments/verify_f9_artifacts.py`. It must re-derive every metric in `artifacts/f9c_belief_metrics.json` from `artifacts/f9c_validation.csv` without running inference, and re-check every hash. It must exit non-zero on any mismatch.

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 186 passed. Record the frozen `config_sha256` in `IMPLEMENTATION_NOTES.md` and in `GATES.md` as the pre-final-run witness. **From this point the config is read-only until F9c reports.**

---

## Task 11: Final evaluation on seeds 7101–7104

**Files:**
- Create: `experiments/evaluate_f9c_robust_belief.py`
- Create: `src/duckie_pomdp/evaluation/f9c_belief.py`
- Create: `src/duckie_pomdp/evaluation/f9c_runtime_cache.py`

**Interfaces:**
- Consumes: `F9cProtocol` (`require_frozen=True`), `RobustPedestrianBeliefUpdater`, `PedestrianBeliefUpdater`.
- Produces: `artifacts/f9c_validation.csv`, `artifacts/f9c_belief_metrics.json`, `artifacts/f9c_nis_metrics.json`, `artifacts/f9c_error_cases/`, `artifacts/f9c_runtime_cache.npz`, `artifacts/f9c_evaluation_truth.npz`; and `summarize_f9c(rows, *, protocol) -> tuple[dict, dict]`; and `write_runtime_cache(path, frames) -> str` / `read_runtime_cache(path) -> tuple[RuntimeCacheFrame, ...]`.

Two systems run side by side on the *same rendered frames and the same single YOLO inference per frame*, exactly as F9b ran raw and corrected in parallel:

```text
Baseline A: corrected YOLO (F9b frozen bias) → highest-confidence selection → frozen F7 EKF → frozen existence
Robust  B: candidates → association → gate → λ-inflated R → frozen F7 EKF → P_D^eff existence → posterior floor
```

Baseline A must be constructed from the *unmodified* `PedestrianBeliefUpdater` and the F9b bias constants, so it is a genuine control.

- [ ] **Step 1: Write `f9c_belief.py`**

Reuse `duckie_pomdp.evaluation.f9_belief.scalar_error_metrics` and `belief_metrics` for the shared quantities. Add:
- `coverage_error_68 = |coverage_68 − 0.68|`, `coverage_error_95 = |coverage_95 − 0.95|` (named exactly that — not "ECE");
- `std_over_rmse` per variable;
- `miss_sequence_metrics(rows)` reporting `P(exists)`, reported range std, `track_active`, and recovery frames at miss-run lengths 1, 3, 5, and 10 — using genuine natural misses only, labelled `natural`; any injected sequence must be labelled `synthetic` and reported separately;
- `robustness_metrics(rows)` reporting localization-outlier count (post-hoc GT IoU < 0.5), how many were accepted vs rejected by the gate, duplicate frames, wrong-association events, natural misses, belief maintained, false-track initializations, track deletions, and recoveries. Misses must be broken out into three disjoint counts, because invariant I2 and invariant I3 make them different events and collapsing them would hide whether the corrections worked:

```text
detector_miss_in_domain        no candidate, predicted CENTER/MID/EDGE   -> likelihood applied
detector_miss_outside_domain   no candidate, predicted OUTSIDE_DOMAIN    -> prediction only
gated_rejection                candidate present, gate rejected          -> counted as a DETECTION
```

  Report, explicitly, how many `gated_rejection` frames occurred and what fraction of them retained an active belief. Under the earlier draft of this plan every one of those frames would have been an existence miss; that number is the direct measure of what Finding 6 bought.

  **Report active-belief retention per class and never pool it.** Invariant I3 makes outside-domain misses nearly free to survive — they apply no likelihood at all — so a pooled retention figure is dominated by whichever class happens to be more common. A run with 80 outside-domain and 20 in-domain misses would report 80% retention while the belief collapsed on essentially every genuine detector miss. Pooled retention must not appear as a control-readiness criterion anywhere; the primary criterion is `detector_miss_in_domain` retention alone, because that is the only class where `P_D^eff` is actually doing work. Outside-domain retention is still reported, but its role is a sanity check on a different question — that belief does not collapse merely because the camera is not looking at the region.

  If `detector_miss_in_domain` has fewer than 20 frames in the final run, say so and treat the in-domain criterion as under-powered rather than passed.
- `outlier_impact(rows)` reporting measurement RMSE, Baseline-A RMSE, Robust-B RMSE, and max transient belief error over the GT-labelled outlier frames;
- `safety_bias(rows)` reporting `E[μ_r − r_GT]` separately for Baseline A and Robust B with an explicit `sign_interpretation` string — positive means the pedestrian is believed farther than reality;
- `support_check(rows, minimum_support)` returning per-bin counts and a boolean `satisfied`.

- [ ] **Step 2: Write `f9c_runtime_cache.py`**

Invariant I4 requires the ablation to be a pure replay, so the cache must survive process exit. An in-memory cache would silently let a second process re-render and re-infer, which would break the "one inference pass" guarantee and could even produce different candidates under domain randomization.

`RuntimeCacheFrame` is a frozen dataclass holding **only runtime-visible, pre-bias quantities**. Every candidate field is named `raw_*` — the naming is load-bearing, not cosmetic:

```text
episode, seed, scenario, frame, dt_s
raw_candidate_count
raw_candidate_range_m[], raw_candidate_bearing_rad[], raw_candidate_confidence[]
raw_candidate_bbox[4][], raw_candidate_projection_failed[]
ego_linear_velocity_mps, ego_yaw_rate_rad_s, ego_motion payload as consumed by predict()
```

**Invariant I5 — the runtime cache is written BEFORE any bias correction.** The ablation must be able to send the same candidate down two paths:

```text
raw candidate ─┬─ F9b bias → baseline row
               └─ F9c bias → robust row
```

If the cache stored F9c-corrected candidates, `bias_refit = false` would replay `z_F9c_corrected − b_F9b`, which is neither Baseline A nor anything else meaningful, and `test_switching_every_robust_component_off_reproduces_the_f9b_path` would be comparing against a corrupted control. Cache at the output of `YoloPedestrianMeasurementPipeline.observe`, before step 4b of Task 8's order of operations.

Ground truth goes to `artifacts/f9c_evaluation_truth.npz`, keyed by `(episode, frame)`, in a **separate file** so that a replay consumer can be handed the runtime cache alone. `write_runtime_cache` returns the SHA256 of the written file; record it in `artifacts/f9c_belief_metrics.json`. `read_runtime_cache` must verify that hash and raise on mismatch — an ablation replaying a cache that has been regenerated is not an ablation.

Ragged per-frame candidate lists must be stored with an explicit offsets array rather than `dtype=object`, so the `.npz` loads without `allow_pickle`.

- [ ] **Step 3: Write `evaluate_f9c_robust_belief.py`**

Structure it on `experiments/validate_f9_yolo_ekf.py`. Preserve the runtime/privileged boundary comment placement: both updaters must be stepped **before** `integration.privileged.read()` is called. Write the runtime cache and the truth file as part of this run. Log per gated measurement, as required: `frame, confidence, predicted range/bearing, measurement range/bearing, innovation, NIS, gate threshold, decision` — and, per invariant I2, also log `detector_detected` and `kinematic_measurement_accepted` so the gate log itself shows that a rejection did not become a miss.

- [ ] **Step 4: Run the final evaluation — ONCE**

```bash
$PY experiments/evaluate_f9c_robust_belief.py --config configs/f9c_robust_belief_v1.toml
```

- [ ] **Step 5: Check the pre-specified support minima FIRST, before reading any accuracy metric**

```bash
$PY -c "
import json
m = json.load(open('artifacts/f9c_belief_metrics.json'))
print(json.dumps(m['metrics']['support_check'], indent=2))
"
```

If `satisfied` is false, the gate is at best `LIMITED` and `CONTROL_READY` is unavailable. Do not re-run with different scenarios to fix it — record the shortfall and report it.

- [ ] **Step 6: Verify artifacts without re-running inference**

Run: `$PY experiments/verify_f9c_artifacts.py`
Expected: exit 0. The verifier must also confirm that the runtime-cache SHA256 recorded in `artifacts/f9c_belief_metrics.json` matches the file on disk.

- [ ] **Step 7: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 186 passed. Record row count, support counts, runtime-cache SHA256, and the frozen config hash in `IMPLEMENTATION_NOTES.md`.

---

## Task 12: Ablation

**Files:**
- Modify: `experiments/evaluate_f9c_robust_belief.py` (add `--ablation`)
- Produces: `artifacts/f9c_ablation_metrics.json`

Seven configurations replayed from `artifacts/f9c_runtime_cache.npz` — invariant I4 means the ablation constructs no detector and no simulator. **The bias stage is always present; only which constants it carries varies.** Writing the table this way is what makes the baseline row equal Baseline A by definition:

```text
row                         bias stage      other components
─────────────────────────── ─────────────── ──────────────────────────────────
baseline                    F9b frozen      none        (== Baseline A exactly)
+ bias refit only           F9c fitted      none
+ innovation gate only      F9b frozen      gate
+ temporal association only F9b frozen      association
+ covariance calibration    F9b frozen      lambda R + posterior floor
+ conditional detection     F9b frozen      P_D^eff + I3 outside-domain policy
all combined                F9c fitted      all of the above  (== Robust B exactly)
```

Note that `+ covariance calibration only` still uses `λR` in the invariant-I1 provider, so its association and gate — where present in other rows — remain internally consistent. No row mixes a raw-`R` provider with an inflated-`R` correction.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_robust_updater.py
def test_ablation_endpoints_match_the_two_headline_systems():
    """ablation['baseline'] must equal the Baseline-A metrics and
    ablation['all_combined'] must equal the Robust-B metrics, field by field,
    to within 1e-12."""


def test_ablation_performs_no_inference_and_no_render(monkeypatch, tmp_path):
    """Invariant I4. Import experiments.evaluate_f9c_robust_belief, monkeypatch
    YoloObjectDetector.__init__ and create_gym_duckietown to raise
    AssertionError('ablation must not run inference or render'), then call the
    ablation entry point on a small hand-written runtime cache written with
    write_runtime_cache. It must complete and produce seven result sets."""


def test_runtime_cache_contains_pre_bias_raw_candidates():
    """Invariant I5. Write a cache from a known pipeline observation, read it
    back, and assert raw_candidate_range_m equals the projector's raw output --
    not that value minus any bias constant. Then replay the same cache twice,
    once with bias_refit=False and once with True, and assert the two runs see
    candidate ranges differing by exactly (f9c_range_bias_m - f9b_range_bias_m).
    A cache written post-correction would make that difference zero."""


def test_ablation_refuses_a_runtime_cache_whose_hash_does_not_match():
    """read_runtime_cache must raise when the recorded SHA256 disagrees with the
    file, so a silently regenerated cache cannot be replayed as if it were the
    final run's."""


def test_bias_ablation_uses_f9b_bias_when_switch_off():
    """Every ablation row with bias_refit=False must carry a bias stage whose
    range_bias_m equals -0.045904804710162034 and bearing_bias_rad equals
    0.00414567890700929 -- the F9b frozen constants, not identity and not the
    F9c fitted values. Assert on the constructed updater's _bias, so the check
    cannot be satisfied by a coincidentally matching metric."""


def test_bias_refit_switch_applies_f9c_frozen_bias_before_association():
    """With bias_refit=True, feed one candidate and capture the range the
    associator receives. It must equal raw_range - f9c_range_bias_m, proving
    the correction happens upstream of association rather than after the gate
    or not at all. Run the same frame with bias_refit=False and assert the
    associator receives raw_range - f9b_range_bias_m instead."""
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_f9c_robust_updater.py -k "ablation or bias" -q`
Expected: FAIL

- [ ] **Step 3: Implement**

Add `--ablation` to `evaluate_f9c_robust_belief.py` as a **mutually exclusive** mode with the final run: `--ablation` loads `artifacts/f9c_runtime_cache.npz` plus `artifacts/f9c_evaluation_truth.npz` and never touches the detector or the simulator. Structure the module so the detector and the environment are constructed inside the final-run branch only — that is what makes the monkeypatch test meaningful rather than decorative.

Replaying candidates rather than images is what makes one inference pass sufficient. Record the runtime-cache SHA256 in `artifacts/f9c_ablation_metrics.json` so the ablation and the headline result are provably the same frames.

- [ ] **Step 4: Run and verify**

Run: `$PY experiments/evaluate_f9c_robust_belief.py --ablation` then `$PY -m pytest tests/test_f9c_robust_updater.py -k "ablation or bias" -q`
Expected: PASS

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 192 passed.

---

## Task 13: Leakage tests, regression guard, and gate report

**Files:**
- Create: `tests/test_f9c_leakage.py`
- Modify: `GATES.md`, `README.md`, `IMPLEMENTATION_NOTES.md`, `.aris/compute/local.md`

- [ ] **Step 1: Write the leakage and regression tests**

```python
# tests/test_f9c_leakage.py
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = [
    "src/duckie_pomdp/belief/innovation_gate.py",
    "src/duckie_pomdp/belief/measurement_association.py",
    "src/duckie_pomdp/belief/covariance_calibration.py",
    "src/duckie_pomdp/belief/observability.py",
    "src/duckie_pomdp/belief/robust_updater.py",
]
FORBIDDEN = (
    "privileged", "PrivilegedState", "true_pomdp_state",
    "sample_object_silhouettes", "eligible_visible",
    "gt_range_m", "gt_bearing_rad", "selected_iou", "intersection_over_union",
)


def test_no_runtime_module_references_privileged_state():
    for relative in RUNTIME_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in source, f"{relative} references {token}"


def test_no_runtime_module_imports_the_evaluation_package():
    for relative in RUNTIME_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", None) or ""] + [
                    alias.name for alias in node.names
                ]
                assert not any("evaluation" in str(name) for name in names), relative


def test_the_evaluator_steps_both_beliefs_before_reading_privileged_truth():
    source = (ROOT / "experiments" / "evaluate_f9c_robust_belief.py").read_text(
        encoding="utf-8"
    )
    privileged_at = source.index("integration.privileged.read()")
    baseline_at = source.index("baseline_updater.update(")
    robust_at = source.index("robust_updater.update(")
    assert baseline_at < privileged_at
    assert robust_at < privileged_at


def test_f9b_frozen_artifacts_are_untouched():
    from duckie_pomdp.evaluation.f9_protocol import sha256

    assert sha256(ROOT / "artifacts" / "f9_measurement_model.json") == (
        "eb09ea6c64b6cbf3306057092e254a0e049776b38581e5b873a8ef9e2e91b278"
    )


def test_f9c_source_never_names_a_frozen_test_seed():
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for seed in ("5101", "5102", "5103", "5104"):
            assert seed not in source, f"{path} hardcodes frozen test seed {seed}"
```

- [ ] **Step 2: Run to verify they fail or pass honestly**

Run: `$PY -m pytest tests/test_f9c_leakage.py -q`
Expected: any failure here is a real leak — fix the source module, never the test.

- [ ] **Step 3: Run the complete suite**

Run: `$PY -m pytest tests -q`
Expected: PASS, ≈ 197 tests, 0 failed, 0 skipped. Record the exact count. (Per-task counts in this plan are expectations, not contracts — if your fixture layout yields a different total, reconcile it once here and move on. Zero failures and zero skips is the contract.)

- [ ] **Step 4: Write the gate report**

Add the F9c row to `GATES.md` and a `## F9c` section to `README.md` with the reproduction commands. Add an `### env:` witness block to `.aris/compute/local.md` following the existing format (`how` / `tier` / `validated` / `gotcha`).

The report must contain, in this order:

```text
calibration seeds / final seeds / frozen config SHA256
gate type + threshold
association rule
covariance inflation parameters (λ_r, λ_β) and posterior floors (σ_floor,r, σ_floor,β)
variance components τ̂ and σ̂_w that justify the floors
bias model chosen + LOSO evidence for the choice
effective detection probabilities per observability class
support counts near/medium/far/edge_fov vs the pre-specified minima

Baseline A vs Robust B:
  range bias / MAE / RMSE
  bearing bias / MAE / RMSE
  range-rate RMSE / bearing-rate RMSE
  coverage_68 / coverage_95 / coverage_error_68 / coverage_error_95 / NLL
  mean_predicted_std and std_over_rmse

natural misses maintained / duplicate handling / outlier handling
miss breakdown: detector_miss_in_domain / detector_miss_outside_domain / gated_rejection
fraction of gated_rejection frames that retained an active belief   (invariant I2 payoff)
false tracks / track deletions / recovery time
NIS diagnostics for accepted measurements AND for rejected ones, separately
ablation table
runtime-cache SHA256 shared by the headline run and the ablation    (invariant I4)
predicted-observability vs GT FOV-region confusion
full test count
```

Then answer, explicitly:

1. Did robust observation handling reduce localization-outlier impact?
2. Did temporal association improve duplicate frames?
3. Did range uncertainty become realistically calibrated?
4. Did the conditional detection model improve belief through natural misses?
5. Did separating detection evidence from kinematic acceptance prevent the gate from worsening existence collapse? Quantify against the counterfactual: how many `gated_rejection` frames were there, and what would `P(e)` have done had each been scored as a miss?
6. Was RMSE materially worsened to achieve calibration?
7. Is EKF + robust observation handling sufficient for Version-1 POMDP?
8. Is the system control-ready?

- [ ] **Step 5: Classify and STOP**

Classify as `CONTROL_READY`, `LIMITED`, or `FAILED` against the pre-specified acceptance bands and the minimum-support requirement. `CONTROL_READY` is unavailable if near-range support < 100.

**Then STOP.** Do not begin stop logic, reward, or SAC. Report and wait.

---

## Self-Review

**Spec coverage.** Every locked decision maps to a task: `P_S` untouched → Task 7 (`test_ps_is_untouched_by_f9c`) and the freeze table; existence unfrozen but only observation parameters → Task 1 validator (which checks `prior`/`survival`/`birth` but deliberately not `detection_probability`); bias refit on new calibration seeds before covariance → Task 9 ordering, with the per-bin decision rule pre-specified; 5101–5104 frozen → Task 1 `forbidden_seeds` plus Task 13 source scan; near-range required for `CONTROL_READY` → Task 2, Task 11 Step 4, Task 13 Step 5; `P_D → P_D^eff(predicted observability)` → Task 7; acceptance bands rather than exact 0.68/0.95 → Task 1 `[acceptance]`; candidates in perception, association in the belief layer → Tasks 5 and 4; four experiments A/B/C/D → Tasks 9, 7, 3+4, 6; ablation → Task 12; artifacts and required tests → Tasks 9–13.

**Additions beyond the operator's brief, flagged deliberately:** (i) the posterior variance floor and its random-effects justification, because `R` inflation provably cannot fix steady-state coverage under a frozen tiny `Q`; (ii) eight calibration seeds instead of four, to halve `SE(b̂)`; (iii) the LOSO rule for the per-bin bias decision; (iv) the anti-inflation guard `mean_predicted_std ≤ 1.5 × RMSE`, which operationalizes "not achieved by absurdly inflating uncertainty" as a number fixed before the run.

**Review round 4 — four corrections applied, plan then marked ready.** (1) *Cache holds raw candidates* — invariant I5, every cached field renamed `raw_*`, cache written at the output of `observe()` before Task 8 step 4b, and `test_runtime_cache_contains_pre_bias_raw_candidates` which asserts the two replay paths see ranges differing by exactly `b_F9c − b_F9b` (zero would prove the cache was written post-correction). (2) *`λ` fit de-circularized* — invariant I6; the fitting set is now selected by `eligible_visible AND valid projection AND correct class AND selected_correct_iou50`, an external GT criterion, rather than by gate acceptance which depends on the `λ` being fitted; excluded-sample count and NIS distribution are reported so the blind spot is visible. (3) *Existence retention un-pooled* — reported separately for `detector_miss_in_domain`, `detector_miss_outside_domain`, and `gated_rejection`, with the in-domain figure alone as the control-readiness criterion, since invariant I3 makes outside-domain retention nearly free and a pooled number could read 80% while the belief collapsed on every genuine detector miss; under-powered if in-domain misses < 20. (4) *Task 9 calls the nested estimator explicitly*, with the one-level function labelled as an internal primitive. The estimator is also now labelled an approximate nested variance-component estimator rather than a REML mixed-effects fit.

**Review round 3 — three corrections applied.** (1) *The bias refit must actually reach the runtime* — new Task 3b adds `FrozenBiasCorrection` as a named runtime stage locked at position 4b in Task 8, before association rather than after the gate, since association thresholds candidates against `h(x̂⁻)` and an uncorrected candidate would inject the full bias into every NIS. The switch selects between the F9b frozen constants and the F9c fitted ones — never `identity()` — and the ablation table now states the bias column for every row, so "all switches off == Baseline A" holds by construction; guarded by `test_bias_ablation_uses_f9b_bias_when_switch_off` and `test_bias_refit_switch_applies_f9c_frozen_bias_before_association`. (2) *`DOWNWEIGHT` removed* — it would have made the correction use `25λR` while association and the gate used `λR`, so `S_gate ≠ S_correction` and invariant I1 would fail; F9c v1 is hard-reject only, and `test_the_gate_exposes_no_covariance_scaling_knob` asserts the knob does not exist rather than merely going unused. (3) *Random-effects statistics corrected* — the `0.01562 rad` bearing figure was the SD of the episode means, not `τ̂`; recomputed with Task 6's own estimator the components are `τ̂_r = 0.01425 / σ̂_w,r = 0.00739` and `τ̂_β = 0.01203 / σ̂_w,β = 0.00455`. Verifying that also surfaced a structural fact that changes the floor: range offset is carried at the *seed* level and bearing offset at the *episode* level, so `SE(b̂)` must divide by the seed count for range. Task 6 now specifies a two-level nested fit and the projections are stated as bands, since F9a's four seeds cannot pin the split.

**Review round 2 — four corrections applied.** (1) *Detection evidence ≠ measurement acceptance* — Finding 6, invariant I2, Task 8 step 10 and its state-transition table, `detector_detected` / `kinematic_measurement_accepted` split through the record, the calibration CSV, the gate log, and the miss breakdown; guarded by `test_a_rejected_localization_is_not_an_existence_miss`. The earlier draft would have turned every gated bbox into an existence miss, making the gate a net negative — this was a real defect, not a wording issue. (2) *One `S` everywhere* — invariant I1, the single `_innovation_covariance` provider in Task 8, the injected `innovation_covariance_for` contract in Task 4, the `grep` check, and `test_association_gate_and_correction_share_one_innovation_covariance`; Task 9 additionally fits `λ` against that same provider. (3) *Outside-domain misses apply no likelihood* — Finding 7, invariant I3, `miss_is_informative` and `observation_informative`, the `outside_domain_miss_policy` config key, and the reworded in-domain decay criterion; `P_D^eff(OUTSIDE_DOMAIN)` survives as a reported diagnostic only, so it cannot become a tuning knob. (4) *Ablation replays an on-disk cache* — invariant I4, `f9c_runtime_cache.py`, hash-verified load, `--ablation` as a mutually exclusive mode with the detector and environment constructed only in the final-run branch, and `test_ablation_performs_no_inference_and_no_render`.

**Placeholder scan.** Task 8 Step 1 and Task 9 Step 1 carry docstring-only test stubs. That is deliberate and bounded: both instruct the implementer to reuse the fixtures already present in `tests/test_pedestrian_ekf.py`, which must be read first; inventing fixture code here that conflicts with the existing ones would be worse than pointing at them. Every assertion those tests must make is stated. No other step defers content.

**Type consistency.** `GateDecision.accepted` (Task 3) is the sole gate output consumed by `RobustPedestrianBeliefUpdater` (Task 8); the gate carries no covariance-scaling field, by invariant I1. `FrozenBiasCorrection.correct` (Task 3b) is applied to every candidate before `CandidateMeasurement` construction in Task 8, and its parameters come from `[measurement_model]` (Task 1) as fitted in Task 9. `AssociationResult.differed_from_highest_confidence` (Task 4) is read by Task 8's duplicate test and Task 11's `robustness_metrics`. `ObservabilityClass` (Task 7) keys `EffectiveDetectionModel`, drives `miss_is_informative`, and appears as `predicted_observability_class` in both the calibration CSV (Task 9) and the validation CSV (Task 11). `RobustObservationSwitches` field names (Task 1) are the ablation keys (Task 12). `CandidateMeasurement` (Task 4) is produced from `CandidateProjection` (Task 5) by the coordinator in Task 8, and is what `RuntimeCacheFrame` (Task 11) serializes for replay in Task 12. `RobustStepRecord.detector_detected` / `.kinematic_measurement_accepted` (Task 8) are the same two column names used by the calibration CSV (Task 9), the validation CSV and gate log (Task 11), and the miss breakdown in `robustness_metrics` (Task 11). `ExistenceFilter.update`'s `observation_informative` keyword (Task 7) is supplied from `EffectiveDetectionModel.miss_is_informative(...) or detector_detected` in Task 8 step 10.
