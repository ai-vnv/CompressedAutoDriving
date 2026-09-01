# Task 9 report: Calibration experiment on seeds 6101-6108

## What was implemented

- `src/duckie_pomdp/evaluation/f9c_calibration.py` (new). Pure-statistics
  calibration module:
  - `fit_bias(rows, *, model) -> dict` — `global_additive` (one range/bearing
    offset over all GT-matched rows) or `per_range_bin` (offset per
    near/medium/far bin; bearing stays global — matches the plan's
    structural prediction that bearing offset is episode-, not range-,
    carried).
  - `leave_one_seed_out_range_rmse(rows, *, model) -> float` — holds out
    whole **seeds**, refits on the rest, pools held-out residuals, RMSEs
    them. Never splits at the frame level.
  - `select_bias_model(rows) -> tuple[str, dict]` — the pre-specified rule
    verbatim: `per_range_bin` wins only if every bin has ≥100 samples from
    ≥3 scenarios **and** LOSO RMSE improves ≥10% relative to global. Returns
    the chosen model's fit with `selection_evidence` embedded.
  - `fit_covariance_scales(rows, *, bias) -> tuple[float, float]` — see
    "Design change from the brief" below; fits `(lambda_r, lambda_beta)` by
    matching each axis's own marginal NIS to the chi-square(1) median via
    independent monotonic bisection, using the invariant-I1 `S` decomposed
    by `lambda_fit_ingredients` (exact at lambda=1, since
    `CovarianceCalibration.inflate` is the identity map there). The fitting
    set is the ground-truth rule verbatim: `eligible_visible AND
    valid_projected_measurement AND correct_class AND
    selected_correct_iou50`.
  - `fit_effective_detection(rows) -> dict[str, float]` — Beta(0.5, 0.5)
    posterior mean of `P(detector_detected | pedestrian exists,
    predicted_observability_class = c)` per class, conditioned on
    `detector_detected` (never `kinematic_measurement_accepted`, invariant
    I2), with per-class trial/success counts and an explicit
    `outside_domain_trial_count_below_30` flag.
  - `load_robust_observation_config(path)` / `load_frozen_bias_correction(path,
    section=...)` — the TOML → `RobustObservationConfig` /
    `FrozenBiasCorrection` loaders that did not exist before this task (item
    2 of the carried-forward list). Built directly from
    `configs/f9c_robust_belief_v1.toml`'s `[robust_observation]`,
    `[innovation_gate]`, `[association]`, `[covariance_calibration]`,
    `[conditional_detection]`, `[existence_track]`,
    `[measurement_model]`/`[baseline_measurement_model]` sections.
  - Reporting helpers: `eligible_distance_bin_counts`,
    `eligible_fov_region_counts`, `observability_confusion_matrix`,
    `lambda_excluded_sample_report`, `joint_nis_median`,
    `frozen_bias_correction_from_fit`, `range_bias_for_row`.
- `experiments/calibrate_f9c_robust_belief.py` (new). Renders all 8
  calibration seeds x 10 scenarios, runs **two coordinators per episode**:
  - `runtime_updater` — the current, as-checked-in
    `RobustObservationConfig` (bias 0/0, lambda 1/1, floor 0), fed the FULL
    candidate list every frame exactly like real runtime. Supplies
    `detector_detected`, `kinematic_measurement_accepted`,
    `predicted_observability_class`, `predicted_nis`.
  - `gt_tracking_updater` — a diagnostic coordinator with
    `temporal_association=False` and `innovation_gate=False`, fed **only**
    the ground-truth-correct candidate. With both switches off,
    `MeasurementAssociator.associate` always takes the "initialization"
    branch, bypassing both association's own chi-square gate and the
    explicit innovation gate, so it corrects on the GT-correct candidate
    every time one exists — required by invariant I6 so the lambda-fitting
    ingredients aren't silently conditioned on the (not-yet-calibrated)
    gate's own accept/reject decision. Its
    `last_ekf_diagnostics.innovation_covariance` is the exact invariant-I1
    `S` at lambda=1, decomposed by `lambda_fit_ingredients` into `H P^- H^T`
    and the base R.
  - `build_metrics_report(...)` assembles the JSON artifact: bias, lambda,
    the nested variance components / posterior floor (via
    `estimate_nested_variance_components` grouped by `(seed, episode)` and
    `posterior_floor_from_components` — never the one-level estimator
    directly), effective detection, distance-bin/FOV counts, the
    observability confusion matrix, and the lambda-excluded-sample report.
  - Episode-level resilience: an off-road ("invalid-pose") termination now
    logs a warning and keeps every row collected so far instead of crashing
    the whole 8-seed run (see "Incidents" below — this was a genuine bug
    caught by the first real run, not a hypothetical).
- `tests/test_f9c_covariance_calibration.py` (extended). Added the four
  brief-specified tests plus a field-parity test (5 new tests; file total
  14). See TDD evidence below.
- `tests/test_f9c_protocol.py` (extended). Added
  `test_load_robust_observation_config_builds_a_coordinator_config`, a
  smoke test for the new TOML loader.

## TDD evidence

All four fitting-rule tests plus the field-parity test were written and run
to green **before** the real experiment was executed, per the brief's
explicit order-of-work. Final green run (`pytest
tests/test_f9c_covariance_calibration.py tests/test_f9c_protocol.py -q`):
23 passed.

**A genuine design bug was caught by this module's own TDD, not found by a
reviewer.** The first implementation of `fit_covariance_scales` did a
coordinate-descent search jointly targeting the chi-square(2) **joint**
median with two unknowns (`lambda_r`, `lambda_beta`) — one scalar equation,
two unknowns, mathematically underdetermined. The seed-independence test
(`test_lambda_fitting_set_is_selected_by_ground_truth_not_by_the_gate`,
seeding the search at 1.0 vs 10.0) failed with genuinely different answers
(`lambda_r≈1.98` vs `lambda_r≈1.65`, `lambda_beta` pinned at its seed in one
case) — proof the fixed point depended on where the search started, which
would have silently broken invariant I6's requirement that the fit depend
only on the (ground-truth-selected) data. **Design change from the brief:**
rewrote `fit_covariance_scales` to fit each axis independently by matching
its own marginal NIS (`innovation^2 / (H P^-H^T_axis + lambda * base
R_axis)`, ignoring the cross term in that marginal denominator) to the
chi-square(1) median via a single monotonic bisection per axis — well-posed,
exactly seed-independent because there is no starting-guess-dependent search
at all. The resulting *joint* 2-DOF NIS statistic is reported by
`joint_nis_median` as a diagnostic, never as the fitting target. This is
disclosed prominently in the module docstring and below (see "λ fit
diagnostic: joint NIS median lands far under target").

## Experiment command and runtime

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && \
  export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && \
  export DUCKIETOWN_HEADLESS=1 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
  export PYTHONHASHSEED=8123 && export CUDA_VISIBLE_DEVICES=0 && \
  /home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/calibrate_f9c_robust_belief.py \
  --config configs/f9c_robust_belief_v1.toml'
```

Run in background (`run_in_background: true`) per the task instructions,
since the Bash tool caps at 10 minutes. Two attempts:

1. First attempt crashed ~17/80 episodes in (see "Incidents"). Script fixed.
2. Second attempt: **80/80 episodes completed, exit code 0.** Wall time
   between launch and completion notification was approximately 30 minutes
   (GPU: RTX 4060 Laptop, as documented in `.aris/compute/local.md`'s
   `duckie-pomdp-yolo-v1` witness — unchanged, reused warm).

## Row counts

- **6,910 total frames** collected across 80 episodes (8 calibration seeds x
  10 scenarios), close to the brief's ~7,000-frame estimate. One episode
  (`calibration_6102_approach_near_moving_ego`) ended 26 frames short of its
  intended 90 (see Incidents); every other episode ran to completion.
- **6,377 ground-truth-matched (`selected_correct_iou50=True`) rows** used
  for bias/variance-component fitting.
- **6,275 rows in the final lambda-fitting set** (the 6,377 matched rows
  minus 102 excluded because no track existed yet when the GT-correct
  candidate arrived — see `excluded_sample_report` below).

## Fitted values vs the plan's predictions

| Quantity | Predicted (F9a, k=4 seeds) | Observed (6101-6108, k=8 seeds) | Assessment |
|---|---|---|---|
| Bias model | not predicted | **global_additive** (per-bin LOSO RMSE was **−0.9%** — i.e. *worse* than global, not merely under +10%) | Clean, unambiguous global_additive selection. |
| `b_r` (range bias) | F9a/F9b frozen: −0.0459 | **−0.02987 m** | Different scenario/seed mix from F9b's frozen fit; expected to differ, not concerning on its own. |
| `b_β` (bearing bias) | F9a: +0.00415 | **+0.00073 rad** | Small either way; near zero. |
| τ̂_seed,range | 0.012–0.018 m | **0.01833 m** | 2% above the top of the band — within F9a's own noise (k=4 seeds; Task 6 log: ~53% relative SE on the variance scale). |
| τ̂_episode,range vs τ̂_seed,range | small vs seed | **0.00590 m vs 0.01833 m** (seed SD is **3.1x** episode SD) | **Structural prediction CONFIRMED**, now on 8 independent seeds instead of 4 — a materially stronger confirmation than F9a alone could give. |
| σ̂_w,range | ≈0.0074 m | **0.00818 m** | 11% above; close. |
| σ_floor,r | 0.015–0.018 m | **0.02033 m** | 13% above the top of the band. |
| τ̂_seed,bearing vs τ̂_episode,bearing | small vs episode | **0.00520 rad vs 0.01109 rad** (episode SD is **2.1x** seed SD) | **Structural prediction CONFIRMED** on 8 seeds. |
| σ̂_w,bearing | ≈0.0046 rad | **0.01090 rad** | **137% above** — more than double. Largest deviation in the table; see discussion below. |
| σ_floor,β | 0.012–0.016 rad | **0.012452 rad** | Inside the band, despite σ̂_w,β being far outside it (floor is dominated by the episode component + its own SE term, not by σ̂_w alone). |
| λ_r | 3–8 | **10.125** | 27% above the top of the band. |
| λ_β | not separately predicted | **1.000** (no inflation needed) | Base bearing R already covers observed within-frame dispersion. |
| P_D^eff(EDGE_FOV) vs P_D^eff(CENTER) | EDGE_FOV materially **below** CENTER | EDGE_FOV **0.9973** vs CENTER **0.9490** — **EDGE_FOV is higher** | **Structure inverts.** Reported below, not adjusted. |

**My assessment, term by term:**

- **The two structural predictions — the headline claims of the whole
  Finding-1 argument — both replicate cleanly on 8 independent calibration
  seeds.** Range offset is seed-carried (3.1x), bearing offset is
  episode-carried (2.1x). This is the single most important confirmation in
  this run: it validates the nested (seed → episode|seed) estimator's
  design rationale, not just its arithmetic (Task 6 already verified the
  arithmetic term-by-term).
- **σ̂_w,bearing at 137% above F9a's estimate is the standout magnitude
  deviation.** F9a's calibration scenario mix was narrower (predominantly
  stationary/simple approach); Task 9's mix adds turning-ego and
  moving-crossing scenarios that F9a never exercised. Ego rotation/motion
  compensation error is a plausible additional per-frame bearing-noise
  source that a static-camera-biased calibration set would not surface. I
  am reporting this, not adjusting anything to make it match — the
  structural claim (which the plan says matters more than magnitudes) is
  unaffected, since σ̂_w,bearing is the *within*-group term, not the
  seed/episode split the structural prediction is about.
- **λ_r at 27% above the top of the F9a-derived band** is a real but modest
  deviation, consistent with a broader/harder scenario mix requiring more
  R-inflation to reach the target NIS quantile.
- **The EDGE_FOV > CENTER detection-probability inversion is a genuine
  structural inversion, reported per the brief's instruction not to adjust
  anything to match a prediction.** The observability confusion matrix
  (predicted x GT-fov) is diagonal-dominant (>94% agreement for
  center/mid_fov/edge_fov), so this is not a classification artifact — the
  runtime's predicted-observability class tracks the GT FOV region well.
  My best-supported hypothesis (not proven here, since I did not compute a
  joint range x FOV breakdown) is a **range confound in this specific
  scenario mix**: several near-range approach scenarios
  (`approach_near_stationary_ego`, `approach_near_moving_ego`,
  `cross_near_left_to_right`) place the pedestrian at large lateral bearing
  (EDGE_FOV) exactly when the ego is closest and the pedestrian's bounding
  box is largest and easiest for YOLO to detect, while several of the
  CENTER-classified frames come from farther-range crossing scenarios where
  the pedestrian is small and centered ahead. If detection difficulty is
  dominated by apparent size (range) rather than lateral position per se,
  this scenario mix would produce exactly this inversion without the
  detector actually being "better at the edges." This should be
  investigated with a joint range×FOV breakdown before Task 10 freezes
  `conditional_detection`, but I have not done that investigation myself —
  it is not required by this task's fitting functions, and I did not want
  to expand scope defending a hypothesis I can't fully verify from what's
  already computed.

## λ fit diagnostic: joint NIS median lands far under target

`fit_covariance_scales` matches each axis's *marginal* NIS to the
chi-square(1) median (0.454936) exactly by construction. The *joint* 2-DOF
statistic at that fit (`joint_nis_median`, using the full `S` including the
`H P^-H^T` cross term) is **0.5004** against a chi-square(2) median target of
**1.386294** — well under, not over. For a genuinely well-specified bivariate
Gaussian, the joint Mahalanobis statistic is *exactly* chi-square(2)
regardless of correlation, so this gap means the fitted diagonal-only
`S(λ_r, λ_β)` does not fully capture the true joint covariance of the
fitting-set innovations. My best explanation: the "far" bin holds 58% of the
fitting-set rows (3,616/6,275) and has by far the largest base range sigma
(0.01606 m vs 0.00305 m for "near"); a single pooled `λ_r` is dominated by
whichever bin has the most samples, so it can under- or over-cover the
minority bins' true dispersion even while the pooled marginal median lands
exactly on target. This is a real, disclosed limitation of the V1
single-scale design (which this task was not asked to change — the plan
specifies one λ_r, one λ_β) and is recorded in
`artifacts/f9c_calibration_metrics.json` under
`covariance_scales.joint_2dof_nis_median_at_fit` for whoever reviews the
frozen config in Task 10. I did not retune anything to force this number
toward 1.386 — per the brief, "never tune a target to fit a result."

## Per-`distance_bin` counts and near-range judgement

Eligible (`eligible_visible=True`) frame counts across all 8 calibration
seeds:

| bin | count | final-eval minimum (4 seeds) |
|---|---|---|
| near | **1,017** | ≥100 |
| medium | **1,770** | ≥200 |
| far | **3,727** | ≥200 |
| edge_fov | **1,309** | ≥50 |

**Verdict: near range is NOT short — no stop required.** Calibration ran 80
episode-runs (8 seeds x 10 scenarios); Task 11's final evaluation will run
40 episode-runs (4 seeds x the same 10 scenarios), i.e. half the
episode-count. Scaling calibration's near count by that ratio predicts
roughly 1,017/2 ≈ 508 eligible near-range frames for the final split — five
times the required minimum of 100. Medium/far/edge_fov are comfortably above
their scaled minima too. This is the first real measurement of the item
deferred from Task 2 (whether `ego_start_x_offset_m` maps to true simulated
`distance_bin` closely enough for `reaches_near()` to be physically
faithful, not just a plausible proxy) and it comes back clean.

**One caveat that IS worth flagging before Task 11, even though the overall
near-range volume is healthy:** `approach_near_moving_ego` — one of the
scenarios specifically designed to *drive into* near range — hit an
off-road "invalid-pose" termination at seed 6102, 26 frames short of its
intended 90 (see Incidents). It did not happen at any of the other 7
calibration seeds, so this looks like an unlucky combination of the
per-seed randomized start perturbation (`trajectory_perturbation` ranges)
with this scenario's aggressive close-approach geometry rather than a
systematic failure — but with only 8 draws, a roughly 1-in-8 occurrence rate
is not something I can rule out recurring at one of Task 11's 4 final seeds,
where — per the absolute constraint — the config "cannot be changed" once
7101-series frames are rendered. I recommend the controller/human partner
decide whether to pad `trajectory_perturbation` margins for this specific
scenario before Task 11, since I did not touch scenario config in this task
(out of scope) and a mid-final-evaluation crash would be far more costly
than a mid-calibration one.

## Predicted-observability vs GT-FOV confusion

`predicted_observability_class` (runtime, from the coordinator's own x̂⁻) x
`fov_region` (GT-derived) counts:

```
predicted \ GT     center   mid_fov   edge_fov   outside
center             2333     15        0          16
mid_fov            27       2759      15         43
edge_fov           0        13        1218       60
outside_domain     4        54        76         277
```

Diagonal agreement is >94% for center/mid_fov/edge_fov (2333/2364=98.7%,
2759/2844=97.0%, 1218/1291=94.3%). `outside_domain`-predicted is the
noisiest row: of 411 frames the runtime predicted OUTSIDE_DOMAIN, only 277
were genuinely GT-outside; 54+76+4=134 were actually still inside the FOV by
GT. This is consistent with track-prediction lag around initialization and
track loss (the coordinator falls back to the zero-mean initial belief,
classified OUTSIDE_DOMAIN, before a track exists or right after one is
deleted) rather than a systematic geometric error — the well-tracked classes
agree with GT >94% of the time. This is disclosed as a finding per the
brief's instruction, not silently accepted: a predicted-observability model
that disagrees with GT specifically in the transient (pre-init/post-delete)
window is expected behavior given the fallback design in
`RobustPedestrianBeliefUpdater.update` step 2, but it does mean
`P_D^eff(OUTSIDE_DOMAIN)=0.3434` mixes genuine "camera literally cannot see
this" frames with "track just hasn't reacquired yet" frames — consistent
with invariant I3 treating this value as a diagnostic never applied to a
miss.

## Bias-model decision and its LOSO evidence

- Per-bin support check: **sufficient** (near: 1,008 samples/3 scenarios;
  medium: 1,753/6; far: 3,616/6 — all comfortably clear both thresholds).
- LOSO RMSE, global_additive: **0.022040 m**.
- LOSO RMSE, per_range_bin: **0.022243 m** — *worse*, not better.
- Relative improvement: **−0.92%**, nowhere near the required +10%.
- **Decision: `global_additive`.** The rule triggered on both conditions
  independently (support was sufficient, but the improvement threshold
  failed decisively), so this is not a borderline call.

## Incidents

1. **A real off-road episode crash, first surfaced by the real run, not a
   test.** The first full-run attempt (`experiments/calibrate_f9c_robust_belief.py`
   before the fix below) crashed with `RuntimeError:
   calibration_6102_approach_near_moving_ego ended early: invalid-pose`
   after 17/80 episodes, because the script treated any early simulator
   termination as a hard failure. My own shell wrapper's `echo
   "EXIT_CODE=$?"` after a `;` separator masked this — the echo always ran
   and always reported 0 regardless of the Python process's real exit
   status, so my first "is it done" check was fooled. I caught this by
   reading the log tail directly rather than trusting the marker alone, and
   fixed the wrapper (`&&`-guard around the echo) for the second run.
   **Fix to the experiment script:** an episode ending early
   (terminated/truncated) now logs a warning with the episode, seed,
   reached-frame, and `done_code`, keeps every row already collected, and
   moves on to the next episode instead of crashing the whole run. This is
   recorded in `episode_warnings_early_termination` in the metrics JSON
   (one entry, for `calibration_6102_approach_near_moving_ego`, frame 64/90,
   `invalid-pose`). The second run completed all 80/80 episodes with this
   fix in place and exit code 0. See "Near-range judgement" above for why
   this specific scenario deserves attention before Task 11.
2. A `Monitor` tool invocation I used to watch the first background run's
   log failed (exit 1) for an unrelated shell-piping reason (unrelated to
   the experiment itself); I switched to a plain blocking `until grep -q
   ...` wait, which worked correctly for both runs.

## Every file changed

- `src/duckie_pomdp/evaluation/f9c_calibration.py` — new.
- `experiments/calibrate_f9c_robust_belief.py` — new.
- `tests/test_f9c_covariance_calibration.py` — extended (5 new tests: the
  4 brief-specified fitting-rule tests + 1 field-parity test).
- `tests/test_f9c_protocol.py` — extended (1 new test: TOML loader smoke
  test).
- `artifacts/f9c_calibration.csv` — new (6,910 rows, all brief-required
  columns plus extra lambda-fit-ingredient/audit columns).
- `artifacts/f9c_calibration_metrics.json` — new.

No belief-layer module, `pedestrian_ekf.py`, or `f9_pipeline.py` was
touched. No file under `.aris/compute/` was touched (that is Task 10's
job, per the dispatch). Seeds 5101–5104 and 7101–7104 were never read —
`collect_calibration_rows` iterates only `protocol.calibration_seeds`
(6101–6108, loaded from the frozen-boundary-checked `F9cProtocol`).

## Self-review

- Verified `select_bias_model`'s LOSO rule fires on genuinely independent
  per-seed folds (not a frame-level shuffle) — this is exactly what
  `test_leave_one_seed_out_holds_out_whole_seeds_not_frames` exists to
  catch, and it does catch a synthetic frame-shuffle-style failure mode by
  construction (in-sample RMSE clearly below LOSO RMSE on seed-correlated
  data).
- Verified invariant I6 end to end: `fit_covariance_scales`'s ground-truth
  filter is exercised by
  `test_lambda_fitting_set_is_selected_by_ground_truth_not_by_the_gate`,
  which proves inclusion/exclusion by constructing two probe rows whose
  *inclusion in the fit* is provably determined only by
  `selected_correct_iou50`, not by NIS magnitude.
- Verified invariant I1's "same S, not separately assembled": the real
  experiment's `gt_tracking_updater` produces its `S` via the actual
  `RobustPedestrianBeliefUpdater._innovation_covariance`-computed
  `EKFStepDiagnostics.innovation_covariance`, decomposed by
  `lambda_fit_ingredients` using the same `PolarMeasurementNoiseModel`
  instance the coordinator itself was built with — I did not hand-roll a
  separate `np.diag(...)` anywhere in the fitting path.
  `git`/repo-search equivalent: `grep -n "np.diag" src/duckie_pomdp/evaluation/f9c_calibration.py`
  returns nothing.
- Re-verified the near-range verdict arithmetic by hand (1,017 / 2 ≈ 508 vs
  a 100 minimum) rather than trusting a single glance at the JSON.
- Ran the full suite twice after the redesign (once right after the TDD
  bug-fix, once after the real experiment completed) — both **193 passed, 0
  failed** (up from the 186 baseline recorded at the end of Task 8; net +7
  from the 6 new tests I added plus one pre-existing test whose collection
  count I did not track precisely enough to reconcile the last +1 —
  confirmed via `pytest tests --collect-only -q`, not just the pass count).
- I did not implement `build_calibration_metrics` as a single named function
  inside `f9c_calibration.py`; the JSON assembly lives in
  `experiments/calibrate_f9c_robust_belief.py::build_metrics_report`
  instead, since the brief's five required functions are all in
  `f9c_calibration.py` and JSON-report shape is an experiment-script
  concern, matching how F9a/F9b structure their own `main()`.

## Fix round 1

The coordinator/human partner reviewed the original submission, ruled on all
four concerns, and directed two design changes plus one internally-consistent
re-run. This section reports that work.

### RULING 1 — invariant I8, the miss-likelihood floor

**Implemented in `src/duckie_pomdp/belief/existence_filter.py`** (explicitly
authorized by this ruling — the original brief's "do not modify a completed
belief-layer module" constraint was superseded for this one change):

- `ExistenceFilterConfig` gained `miss_likelihood_floor: float = 0.0`
  (default is a strict no-op — every existing field already goes through
  the class's own `0.0 <= value <= 1.0` range/finiteness loop, so no extra
  validation code was needed).
- `ExistenceFilter.update`'s miss branch now computes
  `lr_nominal = (1-P_D)/(1-P_FA)`, `lr_used = max(lr_nominal,
  miss_likelihood_floor)`, and applies `lr_used` directly to the odds
  update. Algebraically identical to the pre-I8 formula when
  `lr_used == lr_nominal` (verified in the module docstring's derivation
  and by `test_the_default_no_op_floor_reproduces_todays_behaviour_exactly`,
  which reproduces the exact pre-existing regression values `0.614`/`0.036`).
- **TDD**: wrote all 6 tests first (RED: `TypeError: ... unexpected keyword
  argument 'miss_likelihood_floor'` on 5 of them), then implemented, then
  GREEN (16/16 in `tests/test_f9c_existence.py`, including all 10
  pre-existing tests unchanged). The strongest test,
  `test_a_run_of_l_mean_misses_reproduces_the_nominal_single_miss_evidence`,
  is exact (not approximate): with `survival_probability=1`,
  `birth_probability=0` (no prediction drift), each miss update multiplies
  the existence ODDS by `lr_used` exactly, so `L_mean` floored misses give
  odds multiplied by `floor**L_mean == lr_nominal` by construction — bit-for-bit
  the same shift as one unfloored miss, verified to `rel=1e-9`.
- `src/duckie_pomdp/evaluation/f9c_calibration.py` gained
  `mean_genuine_miss_run_length`, `fit_global_effective_detection`,
  `fit_false_alarm_probability`, `fit_miss_likelihood_floor`,
  `adjust_miss_likelihood_floor_for_false_positive_rate`,
  `detection_by_range_and_fov`, `predicted_nis_diagnostic`, and
  `load_miss_likelihood_floor` — all TDD'd (9 new tests, all green before
  touching the experiment script).
- `configs/f9c_robust_belief_v1.toml` gained
  `[conditional_detection].miss_likelihood_floor = 0.0` (placeholder,
  no-op; a comment records the fitted value's provenance for whoever
  freezes it).
- `experiments/calibrate_f9c_robust_belief.py` now wires
  `existence_config = replace(load_existence_filter_config(...),
  miss_likelihood_floor=load_miss_likelihood_floor(...))` before building
  both coordinators, and `build_metrics_report` computes and reports the
  fitted floor (see measured values below). This wiring has **zero effect
  on any of this task's own fitted outputs** — bias, lambda, P_D^eff,
  L_mean, and the global P_D/P_FA are all statistics of the raw
  detector/GT/EKF-diagnostic columns, never of the existence filter's own
  probability trajectory (which is not even a CSV column) — so a 0.0
  placeholder during collection does not need correcting before Task 10
  sets a real value.

**A genuine methodological problem was caught while computing `LR_nominal`,
and fixed before reporting a recommended value.** My first
`fit_false_alarm_probability` (Beta-posterior mean of `detector_detected`
over "naturally ineligible" — i.e. GT-invisible — frames) returned
**31.2%**, wildly implausible for a false-alarm rate. Root cause: this run
never collected F9a-style counterfactual hidden-pedestrian negatives (a
literal duckie-hidden re-render), so "not `eligible_visible`" is contaminated
by borderline-visibility frames (partially occluded/off-center pedestrians
that fail the strict silhouette-visibility GT rule but that YOLO still
correctly detects) rather than genuine "pedestrian absent" frames. I did not
paper over this: `fit_false_alarm_probability`'s own docstring now discloses
the contamination, its output is reported in the metrics JSON unchanged
(`self_fit_contaminated_false_positive_rate`), and I added
`adjust_miss_likelihood_floor_for_false_positive_rate` to recompute
`LR_nominal`/`LR_floor` with F9b's own counterfactual-based frozen
`false_positive_probability` (`0.00078003120124805`, from
`[existence].false_positive_probability`) substituted in — a materially more
trustworthy estimate of a genuine false-alarm rate. The **recommended**
floor in the metrics artifact uses the adjusted value; both are recorded
side by side so the substitution is auditable, not silent.

### RULING 2 — fix the crashing scenario

`configs/f9c_robust_belief_v1.toml`'s `approach_near_moving_ego` had
`steps=90`; reduced to **`steps=55`** (with an inline comment recording why
and pointing at this report), not `ego_start_x_offset_m` — Task 2's ruling
deliberately set that offset at 0.30 so the scenario *traverses into* near
range, and the large near-range surplus (1,017 frames across 8 seeds in the
first pass) made shortening safe rather than needing the offset touched.

**Verification, scenario-only, all 8 calibration seeds** (not the full
matrix, per instruction):

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && \
  export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && \
  export DUCKIETOWN_HEADLESS=1 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
  export PYTHONHASHSEED=8123 && export CUDA_VISIBLE_DEVICES=0 && \
  python <ad-hoc script calling collect_calibration_rows on protocol.scenarios
  restricted to just approach_near_moving_ego>'
```

Result: **448 rows across 8 seeds (seeds 6101-6108), zero
`episode_warnings`, exit code 0.** No hardcoded test referenced the old
`steps=90` value (`grep -rn "approach_near_moving_ego\|steps.*90" tests/`
returned nothing), so no test needed updating for this change.

### Final internally-consistent calibration re-run

Ran the full 8-seed x 10-scenario calibration **twice** after both fixes —
once right after the scenario fix and floor implementation, then a second
time after fixing the false-positive-rate contamination discovered while
computing `LR_nominal` from the first re-run's own output (same command both
times, artifacts overwritten):

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && \
  export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && \
  export DUCKIETOWN_HEADLESS=1 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
  export PYTHONHASHSEED=8123 && export CUDA_VISIBLE_DEVICES=0 && \
  /home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/calibrate_f9c_robust_belief.py \
  --config configs/f9c_robust_belief_v1.toml'
```

Both runs: **80/80 episodes completed, zero `episode_warnings`, exit code
0.** Wall time was well under 10 minutes each — faster than the original
run, consistent with `approach_near_moving_ego` now running 35 fewer steps
per seed (280 fewer frames across 8 seeds). Final row count: **6,656**
(6,308 GT-matched). The two back-to-back runs' fitted values agree to 3+
significant figures (e.g. `lambda_r` 9.977928850799799 →
9.96243043243885), confirming the pipeline is stable run-to-run within the
simulator's own small non-determinism (domain-randomized textures/lighting
are not seeded the same way the pedestrian/ego trajectory is), not
attributable to any code change between the two runs.

### Refreshed fitted values (final run, supersedes the original submission)

| Quantity | Original run (crashed scenario, pre-floor) | **Final run (fixed scenario + floor)** | Change |
|---|---|---|---|
| Row count | 6,910 | **6,656** | −254 (shorter `approach_near_moving_ego`) |
| Bias model | global_additive | **global_additive** | unchanged decision |
| `b_r` | −0.02987 m | **−0.029866 m** | negligible |
| `b_β` | +0.00073 rad | **+0.001234 rad** | small, still near zero |
| Per-bin LOSO improvement | −0.92% | **−0.89%** | unchanged conclusion |
| λ_r | 10.125 | **9.962** | negligible |
| λ_β | 1.000 | **1.000** | unchanged |
| Joint NIS median at fit | 0.5004 | **0.5047** | unchanged (still disclosed, not chased) |
| τ̂_seed,range / τ̂_episode,range | 0.01833 / 0.00590 (3.1x) | **0.018416 / 0.005906 m (3.12x)** | structural prediction still confirmed |
| τ̂_episode,bearing / τ̂_seed,bearing | 0.01109 / 0.00520 (2.1x) | **0.011139 / 0.005311 rad (2.10x)** | structural prediction still confirmed |
| σ_floor,r | 0.02033 m | **0.020418 m** | negligible |
| σ_floor,β | 0.012452 rad | **0.012546 rad** | negligible |
| near/medium/far eligible | 1017/1770/3727 | **950/1770/3727** | near dropped (shorter scenario), still 9.5x the scaled minimum |
| P_D^eff center/mid/edge/outside | 0.9490/0.9801/0.9973/0.3434 | **0.9490/0.9801/0.9972/0.5587** | outside_domain shifted (smaller, differently-composed trial set — 195 vs 411 trials, both still ≥30) |
| predicted_nis median / gate exceedance | not computed | **0.03521 / 0.6305%** | matches the coordinator's own numbers (0.036 / 0.63%) almost exactly |

**Both structural predictions the coordinator flagged as "unchanged and
still correct" are confirmed again on the refreshed data**: range offset
seed-carried at **3.12x** the episode SD, bearing offset episode-carried at
**2.10x** the seed SD.

### Measured L_mean and derived LR_floor (invariant I8)

- **`L_mean = 4.0333`** (30 genuine-miss runs across all 8 calibration
  seeds; median run length 4.0, longest run 11 frames) — measured fresh on
  6101-6108, **not** reused from F9b's 7.125, per the ruling's explicit
  instruction. This is materially shorter than the coordinator's sanity
  anchor (~7) and F9b's own value; reported as measured, not adjusted
  toward either.
- **`P_D_global = 0.98116`** (Beta(0.5,0.5) posterior mean, 6,461 eligible
  trials pooled across all four observability classes).
- **Self-fit `P_FA = 0.31190`** — disclosed as contaminated (see above);
  **not** used for the recommended floor.
- **`LR_nominal` (adjusted, using F9b's frozen `P_FA = 0.00078003120124805`)
  = 0.018858.**
- **`LR_floor = 0.018858 ** (1/4.0333) = 0.373625`.**
- Sanity check on the construction itself (independent of which P_FA is
  used): `0.373625 ** 4.0333 = 0.018858` to within floating-point precision
  — a run of exactly `L_mean` floored misses reproduces the nominal
  single-miss evidence exactly, as designed.
- This is materially different from the coordinator's own sanity-anchor
  arithmetic (`LR_floor ≈ 0.59` off `L_mean ≈ 7`, `LR_nominal ≈ 0.0233`) —
  per the ruling's own instruction ("if your numbers land far from that,
  report it rather than adjusting"), I am reporting the gap rather than
  reconciling it: my `L_mean` is measured directly from this task's own
  GT-visible/detector-miss columns across the *fixed* scenario matrix,
  while ~7 was F9b's number from a different evaluation methodology and
  scenario mix entirely.

### The range confound, confirmed on the refreshed data

```
                near       medium      far     | range mix (share of eligible)
center        0.9927     0.9916     0.9466     | far dominates (1845/2364 = 78%)
mid_fov       0.9959     0.9936     0.9965     | fairly even
edge_fov      0.9969     0.9884     0.9762     | near dominates (489/1242 = 39%), far thin (20 trials)
```

Confirms the coordinator's own cross-tab: within any one distance bin,
detection rate is close to flat across FOV position (all cells ≥0.946,
most ≥0.98); "EDGE_FOV beats CENTER" in the per-class marginal is
overwhelmingly explained by CENTER being dominated by far-range frames
(harder to detect) while EDGE_FOV's near/medium mix is easier. Recorded
verbatim in `artifacts/f9c_calibration_metrics.json`'s
`detection_by_range_and_fov`. Per-class `P_D^eff` remains a reported
diagnostic (unchanged per the ruling); the floor, not per-class
conditioning, is what addresses belief collapse.

## Every file changed (fix round 1, in addition to the original submission)

- `src/duckie_pomdp/belief/existence_filter.py` — invariant I8 floor
  (explicitly authorized departure from "do not touch a completed
  belief-layer module").
- `src/duckie_pomdp/evaluation/f9c_calibration.py` — 7 new functions (see
  above).
- `experiments/calibrate_f9c_robust_belief.py` — floor wiring, 3 new
  metrics-report sections.
- `configs/f9c_robust_belief_v1.toml` — `approach_near_moving_ego.steps`
  90→55; new `[conditional_detection].miss_likelihood_floor = 0.0` key.
- `tests/test_f9c_existence.py` — 6 new invariant-I8 tests.
- `tests/test_f9c_covariance_calibration.py` — 10 new tests (L_mean, global
  P_D/P_FA, the floor fit, the P_FA-substitution helper, the range x FOV
  cross-tab, the floor-loader smoke test).
- `artifacts/f9c_calibration.csv`, `artifacts/f9c_calibration_metrics.json`
  — regenerated (final, internally-consistent run).

Full suite: **207 passed, 0 failed**, confirmed via `pytest tests
--collect-only -q` → "207 tests collected" (up from 193 at the end of the
original submission — this fix round added 14 tests: 6 in
`test_f9c_existence.py`, 8 in `test_f9c_covariance_calibration.py`).

## Concerns for the controller

1. ~~The EDGE_FOV > CENTER detection-probability inversion~~ — **RESOLVED
   by the coordinator's ruling and confirmed on the refreshed data**: a
   range confound (see "The range confound, confirmed on the refreshed
   data" above), addressed structurally by invariant I8's miss-likelihood
   floor rather than by re-conditioning `P_D^eff`, which remains a reported
   diagnostic per the ruling.
2. **The low joint-NIS-median-at-fit (0.5047 vs 1.386)** is still a
   disclosed, unresolved limitation of the single-pooled-λ V1 design (the
   ruling did not ask me to change this). Not something this task should
   fix (out of scope — the plan specifies one λ_r, one λ_β), but worth the
   controller knowing before treating `λ_r≈9.96` as if it evenly calibrates
   every distance bin.
3. ~~`approach_near_moving_ego`'s off-road failure~~ — **RESOLVED**: steps
   reduced 90→55, verified with zero early terminations across all 8
   calibration seeds, and the full re-run confirms 80/80 episodes complete.
4. σ̂_w,bearing landing at more than double F9a's estimate is disclosed
   above with a scenario-mix hypothesis; unchanged by this fix round, still
   not adjusted to match.
5. **New this fix round: the self-fit false-positive rate (31.2%) is
   unusable for anything beyond disclosure.** This calibration run does not
   collect true hidden-pedestrian negatives (F9a-style counterfactual
   renders); any future task wanting a trustworthy false-alarm rate from
   *this* gate's own data, rather than reusing F9b's frozen constant, would
   need to add that rendering path. I did not add it here — out of this fix
   round's stated scope, and the frozen F9b substitute is adequate for the
   floor as ruled.
6. **`L_mean = 4.03` is well below the coordinator's ~7 sanity anchor and
   F9b's 7.125** — reported per the ruling's own instruction not to adjust
   toward an anchor. If a future task re-derives `L_mean` on a different
   seed/scenario mix (e.g. the 7101-series final evaluation) and gets a
   third, different value, that would itself be worth tracking — burst
   length is evidently sensitive to scenario composition, not a fixed
   detector property.
