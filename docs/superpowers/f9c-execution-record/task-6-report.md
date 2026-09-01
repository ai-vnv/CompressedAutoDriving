# Task 6 report: Covariance calibration — R inflation and posterior floor

## What was implemented

`src/duckie_pomdp/belief/covariance_calibration.py` (new file):

- `CovarianceCalibration` — frozen dataclass, positional fields
  `range_scale, bearing_scale, range_posterior_floor_m, bearing_posterior_floor_rad`.
  - `__post_init__` rejects `range_scale < 1.0` or `bearing_scale < 1.0` with a
    `ValueError` whose message contains the literal substring `at least 1`.
  - `inflate(measurement_covariance)` copies the input (`np.array(..., copy=True)`),
    multiplies `[0,0]` by `range_scale` and `[1,1]` by `bearing_scale`, returns the
    new matrix. Original argument is never mutated.
  - `floor_polar_standard_deviation(range_std_m, bearing_std_rad)` adds the floors
    in quadrature: `sqrt(std^2 + floor^2)` per axis, returned as a tuple. Strictly
    increasing since floors are non-negative inputs (guaranteed by construction, not
    separately validated to be non-negative — see concerns).
- `VarianceComponents(between_group_variance, within_group_variance, group_count, sample_count)`.
- `estimate_variance_components(residuals_by_group)` — the one-level unbalanced
  ANOVA moment estimator, implemented exactly as specified:
  - `MS_within = ΣᵢΣⱼ(x_ij − x̄ᵢ)² / (N − k)`
  - `MS_between = Σᵢnᵢ(x̄ᵢ − x̄)² / (k − 1)`
  - `n_effective = (N − Σᵢnᵢ²/N) / (k − 1)`
  - `τ̂² = max(0, (MS_between − MS_within) / n_effective)`, clamped at zero.
- `NestedVarianceComponents(seed_variance, episode_variance, within_variance, seed_count, episode_count, sample_count)`.
- `estimate_nested_variance_components(residuals_by_seed_episode)` — applies the
  one-level primitive twice: (1) pools each seed's samples across episodes and
  fits with seeds as groups → `seed_variance`; (2) subtracts each seed's own mean
  from every sample belonging to that seed (seed-centred residuals) and re-fits
  with episodes as groups → `episode_variance`, and reports `within_variance` from
  this second, episode-level pass.
- `posterior_floor_from_components(components)` —
  `sqrt(τ_seed² + τ_episode² + τ_seed²/n_seeds + τ_episode²/n_episodes)`, dividing
  the seed component by `seed_count` (not `episode_count`).

Module docstring states the R-inflation-vs-posterior-floor division of labor, and
explicitly labels the nested estimator as an *approximate nested variance-component
estimator* (one-level ANOVA moment estimator applied twice) — not a REML
mixed-effects fit, with no standard errors on components and possible bias under
strong imbalance, justified by the floor-only accuracy requirement and the
NumPy-only constraint. It also states the required sentence about the floor
representing observation bias that the frozen EKF state cannot see.

`tests/test_f9c_covariance_calibration.py` (new file): the 9 test functions from
the brief's Step 1, verbatim.

## TDD evidence

**RED** —
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_covariance_calibration.py -q'
```
Output (abridged):
```
ImportError while importing test module '.../tests/test_f9c_covariance_calibration.py'.
E   ModuleNotFoundError: No module named 'duckie_pomdp.belief.covariance_calibration'
1 error in 0.31s
```
Expected and correct: the module did not exist yet.

**GREEN** — same command with `-v`:
```
tests/test_f9c_covariance_calibration.py::test_inflation_scales_range_and_bearing_variance_independently PASSED
tests/test_f9c_covariance_calibration.py::test_inflated_covariance_stays_positive_semidefinite PASSED
tests/test_f9c_covariance_calibration.py::test_scales_below_one_are_rejected PASSED
tests/test_f9c_covariance_calibration.py::test_posterior_floor_adds_in_quadrature_and_never_shrinks_uncertainty PASSED
tests/test_f9c_covariance_calibration.py::test_variance_components_recover_a_known_between_group_offset PASSED
tests/test_f9c_covariance_calibration.py::test_variance_components_do_not_report_negative_between_group_variance PASSED
tests/test_f9c_covariance_calibration.py::test_posterior_floor_includes_both_levels_and_the_bias_estimation_error PASSED
tests/test_f9c_covariance_calibration.py::test_the_bias_estimation_error_uses_the_seed_count_not_the_episode_count PASSED
tests/test_f9c_covariance_calibration.py::test_nested_components_recover_a_known_two_level_structure PASSED
9 passed in 0.15s
```

## Full-suite result

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```
Result: **153 passed**, 260 warnings, 11.79s. (Prior suite was 144; this task added
9 tests, all new — 144 + 9 = 153, matching the observed count. The brief's stated
checkpoint of 152 is stale, as flagged in the task instructions.)

## Synthetic-recovery observed values vs. tolerances

`test_variance_components_recover_a_known_between_group_offset` (seed 20260808,
40 groups × 40 samples, true between-SD 0.014, true within-SD 0.0073):
- observed `between_group_variance**0.5` = **0.012548**, target 0.014, tolerance
  ±0.004 → within tolerance (margin ≈ 0.0015 to the boundary of 0.0006 slack, i.e.
  well inside).
- observed `within_group_variance**0.5` = **0.007317**, target 0.0073, tolerance
  ±0.002 → comfortably within tolerance.

`test_nested_components_recover_a_known_two_level_structure` (seed 20260808,
8 seeds × 10 episodes × 40 samples, true seed-SD 0.0155, episode-SD 0.0048,
within-SD 0.0074):
- observed `seed_variance**0.5` = **0.017265**, target 0.0155, tolerance ±0.006 →
  within tolerance (deviation ≈ 0.0018, well inside the ±0.006 band).
- observed `episode_variance**0.5` = **0.004535**, target 0.0048, tolerance
  ±0.003 → within tolerance.
- observed `within_variance**0.5` = **0.007375**, target 0.0074, tolerance
  ±0.001 → within tolerance.

No tolerance was widened. All four synthetic-recovery assertions passed with
comfortable margin at the seeds specified in the brief; no concern to escalate
here.

## Files changed

- Created: `src/duckie_pomdp/belief/covariance_calibration.py`
- Created: `tests/test_f9c_covariance_calibration.py`
- No other files touched (no edits to `belief/__init__.py` or anything else). Not
  a git repository; nothing committed.

## Self-review

- **Formulas**: `MS_within`, `MS_between`, `n_effective`, and the `τ̂²` clamp in
  `estimate_variance_components` transcribe the brief's formulas term-for-term.
  Verified by hand for a 2-group toy case during design; confirmed empirically by
  the synthetic-recovery tests above.
- **Nested estimator**: pools each seed's samples across all its episodes for the
  first pass (not one sample per seed, and not a per-episode mean) — this matches
  "with seeds as groups" using the raw residuals. The second pass subtracts each
  seed's *own* mean (not the grand mean) from every sample under that seed before
  grouping by episode, matching "on the seed-centred residuals ... with episodes
  as groups." `within_variance` is taken from the second pass only, as specified.
- **Posterior floor**: seed term divided by `seed_count`, episode term divided by
  `episode_count` — the seed-count-not-episode-count distinction is the one the
  brief calls out as easy to get wrong; `test_the_bias_estimation_error_uses_the_seed_count_not_the_episode_count`
  passes, and passes non-vacuously (the wrong-if-flat value is strictly smaller,
  confirmed by the assertion, not just constructed to be smaller by inspection).
- **No vacuous tests**: checked each of the 9 tests exercises a real code path
  with a non-trivial assertion (equality with computed expected values, strict
  inequalities, `pytest.raises` with a message match). None assert `True` or a
  tautology.
- **Docstring honesty**: module docstring explicitly says "approximate nested
  variance-component estimator," "the one-level ANOVA moment estimator applied
  twice," "not a REML mixed-effects fit," "gives no standard errors on the
  components," "can be biased under strong imbalance," and states the NumPy-only
  constraint as the reason this approximation is acceptable. It also states the
  required sentence that the posterior floor represents observation bias the
  frozen EKF state cannot see, and that v1 does not augment the EKF state with a
  bias term.
- **Non-mutation and PSD**: `inflate` uses `np.array(..., copy=True)` so the
  caller's matrix is untouched; confirmed by
  `test_inflated_covariance_stays_positive_semidefinite` passing on a diagonal
  input (scaling positive diagonal entries by scales ≥ 1 keeps them positive).
- **One divergence in wording, no functional effect**: the task's context section
  (top-level instructions) asks for the docstring to say "...calibrated from
  measured between-group variance," while the brief's Step 3 quotes it as
  "...calibrated from measured between-episode variance." I used "between-group"
  (matching the context section, and more accurate since both seed- and
  episode-level between-group variance feed the floor). No test checks this
  exact wording; flagging the discrepancy in the two source documents in case it
  matters for artifact review.

## Concerns

- None that block. The one item above (between-group vs. between-episode wording
  in the docstring) is cosmetic and untested; happy to change to the brief's
  exact wording if preferred.
- `floor_polar_standard_deviation` does not itself validate that the floor
  fields are non-negative (only `range_scale`/`bearing_scale` are validated in
  `__post_init__`). A caller passing a negative floor could shrink the reported
  std. This is out of scope for the brief's test list (no test constructs a
  negative floor), so I did not add validation beyond what was specified —
  flagging in case a future task expects it.
