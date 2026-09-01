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

