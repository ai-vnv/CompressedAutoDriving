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

