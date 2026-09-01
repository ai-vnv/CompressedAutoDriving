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


def test_negative_range_posterior_floor_is_rejected():
    """Task 10 carry-forward from Task 6's review: this task writes floor
    values into config by hand, and a negative floor would shrink reported
    uncertainty (via floor_polar_standard_deviation's quadrature sum)
    instead of inflating it, silently defeating the whole calibration."""

    with pytest.raises(ValueError, match="range_posterior_floor_m must be non-negative"):
        CovarianceCalibration(1.0, 1.0, -0.001, 0.0)


def test_negative_bearing_posterior_floor_is_rejected():
    with pytest.raises(ValueError, match="bearing_posterior_floor_rad must be non-negative"):
        CovarianceCalibration(1.0, 1.0, 0.0, -0.001)


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


# ---------------------------------------------------------------------------
# Task 9: fit_bias / select_bias_model / leave_one_seed_out_range_rmse /
# fit_covariance_scales -- see evaluation/f9c_calibration.py.
# ---------------------------------------------------------------------------

from dataclasses import fields

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.robust_updater import (
    RobustObservationSwitches as BeliefRobustObservationSwitches,
)
from duckie_pomdp.evaluation.f9c_calibration import (
    fit_covariance_scales,
    leave_one_seed_out_range_rmse,
    select_bias_model,
)
from duckie_pomdp.evaluation.f9c_protocol import (
    RobustObservationSwitches as ProtocolRobustObservationSwitches,
)


def _bias_row(
    *,
    seed: str,
    scenario: str,
    distance_bin: str,
    range_error_m: float,
    bearing_error_rad: float = 0.0,
) -> dict:
    return {
        "seed": seed,
        "scenario": scenario,
        "distance_bin": distance_bin,
        "selected_correct_iou50": True,
        "range_error_m": range_error_m,
        "bearing_error_rad": bearing_error_rad,
    }


def test_bias_model_selection_prefers_global_when_per_bin_does_not_generalize():
    """Per-bin differences here are pure seed artefacts: bin membership is a
    re-labelling of seed identity, not a genuine function of range. LOSO
    (holding out a whole seed) cannot generalize a per-seed offset any
    better through a per-bin re-grouping than through the global mean, so
    select_bias_model must return "global_additive"."""

    rng = np.random.default_rng(20260809)
    bins = ("near", "medium", "far")
    scenarios = ("s0", "s1", "s2", "s3")
    rows = []
    for seed_index in range(8):
        seed_offset = float(rng.normal(0.0, 0.02))
        bin_name = bins[seed_index % 3]
        for scenario in scenarios:
            for _ in range(40):
                noise = float(rng.normal(0.0, 0.006))
                rows.append(
                    _bias_row(
                        seed=f"seed{seed_index}",
                        scenario=scenario,
                        distance_bin=bin_name,
                        range_error_m=seed_offset + noise,
                    )
                )

    chosen, fit = select_bias_model(rows)
    assert chosen == "global_additive"
    assert fit["model"] == "global_additive"
    assert fit["selection_evidence"]["per_bin_support"]["sufficient"] is True


def test_bias_model_selection_picks_per_bin_only_on_a_genuine_range_dependence():
    """A true, seed-invariant linear bias(r): near/medium/far means are
    widely separated and shared by every seed, with only small per-frame
    noise. >=100 samples/bin from >=3 scenarios; LOSO per-bin must
    generalize far better than one global mean, so select_bias_model must
    return "per_range_bin"."""

    rng = np.random.default_rng(20260810)
    bin_bias = {"near": 0.005, "medium": 0.03, "far": 0.07}
    scenarios = ("s0", "s1", "s2", "s3")
    rows = []
    for seed_index in range(8):
        for bin_name, bias_m in bin_bias.items():
            for scenario in scenarios:
                for _ in range(15):
                    noise = float(rng.normal(0.0, 0.003))
                    rows.append(
                        _bias_row(
                            seed=f"seed{seed_index}",
                            scenario=scenario,
                            distance_bin=bin_name,
                            range_error_m=bias_m + noise,
                        )
                    )

    chosen, fit = select_bias_model(rows)
    assert chosen == "per_range_bin"
    assert fit["model"] == "per_range_bin"
    evidence = fit["selection_evidence"]
    assert evidence["per_bin_support"]["sufficient"] is True
    assert evidence["per_range_bin_relative_improvement"] >= 0.10
    assert fit["range_bin_bias_m"]["near"] == pytest.approx(0.005, abs=0.01)
    assert fit["range_bin_bias_m"]["far"] == pytest.approx(0.07, abs=0.01)


def test_leave_one_seed_out_holds_out_whole_seeds_not_frames():
    """A rule that shuffles frames would report a far lower RMSE; assert the
    LOSO RMSE exceeds the in-sample RMSE on seed-correlated synthetic data."""

    rng = np.random.default_rng(20260811)
    seed_offsets = {"seedA": 0.05, "seedB": 0.0, "seedC": -0.05}
    rows = []
    for seed, offset in seed_offsets.items():
        for _ in range(200):
            noise = float(rng.normal(0.0, 0.002))
            rows.append(
                _bias_row(
                    seed=seed,
                    scenario="s0",
                    distance_bin="near",
                    range_error_m=offset + noise,
                )
            )

    loso_rmse = leave_one_seed_out_range_rmse(rows, model="global_additive")

    from duckie_pomdp.evaluation.f9c_calibration import fit_bias

    in_sample_fit = fit_bias(rows, model="global_additive")
    in_sample_residuals = np.array(
        [row["range_error_m"] - in_sample_fit["range_bias_m"] for row in rows]
    )
    in_sample_rmse = float(np.sqrt(np.mean(in_sample_residuals**2)))

    assert loso_rmse > in_sample_rmse


def _lambda_row(
    *,
    innovation_r: float,
    innovation_b: float,
    base_range_variance_m2: float,
    base_bearing_variance_rad2: float,
    selected_correct_iou50: bool = True,
    raw_range_m: float = 0.65,
) -> dict:
    return {
        "eligible_visible": True,
        "valid_projected_measurement": True,
        "correct_class": True,
        "selected_correct_iou50": selected_correct_iou50,
        "lambda_fit_eligible": True,
        "raw_range_m": raw_range_m,
        "raw_bearing_rad": 0.0,
        "distance_bin": "medium",
        "hph_rr": 0.0,
        "hph_rb": 0.0,
        "hph_bb": 0.0,
        "base_range_variance_m2": base_range_variance_m2,
        "base_bearing_variance_rad2": base_bearing_variance_rad2,
        "range_innovation_m": innovation_r,
        "bearing_innovation_rad": innovation_b,
    }


def test_lambda_is_fitted_to_the_nis_median_not_to_coverage():
    """"lambda" here is the R-inflation scale, not the gate covariance knob --
    that knob no longer exists. See "Why there is no downweight mode".
    With residuals whose true variance is 4x the modelled variance,
    fit_covariance_scales must return range_scale ~ 4 (rel 0.25)."""

    rng = np.random.default_rng(20260812)
    sigma_r0 = 0.0125
    sigma_b0 = 0.0126
    n = 2000
    range_innovations = rng.normal(0.0, 2.0 * sigma_r0, size=n)  # true var = 4*sigma_r0^2
    bearing_innovations = rng.normal(0.0, sigma_b0, size=n)  # correctly modelled

    rows = [
        _lambda_row(
            innovation_r=float(range_innovations[i]),
            innovation_b=float(bearing_innovations[i]),
            base_range_variance_m2=sigma_r0**2,
            base_bearing_variance_rad2=sigma_b0**2,
        )
        for i in range(n)
    ]

    lambda_r, lambda_beta = fit_covariance_scales(
        rows, bias=FrozenBiasCorrection.identity()
    )
    assert lambda_r == pytest.approx(4.0, rel=0.25)
    assert lambda_beta == pytest.approx(1.0, rel=0.4)


def test_lambda_fitting_set_is_selected_by_ground_truth_not_by_the_gate():
    """Invariant I6. Build calibration rows where one sample has a large
    innovation but selected_correct_iou50=True, and another has a small
    innovation but selected_correct_iou50=False. fit_covariance_scales must
    include the first and exclude the second -- proving selection is by GT
    match, not by any NIS threshold. Then assert the returned lambda is
    unchanged when the rows are re-fed in a different order, and that the
    underlying per-axis bisection converges to the same root regardless of
    where its search bracket starts: a gate-conditioned set would not be
    seed-independent, and (unlike an under-determined joint 2-parameter
    search) this per-axis fit has no seed to begin with."""

    sigma_r0 = 0.0125
    sigma_b0 = 0.0126
    # Eight distinct baseline NIS-at-lambda=1 magnitudes (range innovation
    # multiples of sigma_r0), each also carrying a small, consistent bearing
    # innovation so both axes are exercised.
    multiples = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4)
    baseline = [
        _lambda_row(
            innovation_r=multiple * sigma_r0,
            innovation_b=0.5 * sigma_b0,
            base_range_variance_m2=sigma_r0**2,
            base_bearing_variance_rad2=sigma_b0**2,
        )
        for multiple in multiples
    ]
    # Large innovation, GT-correct: must be INCLUDED.
    probe_included = _lambda_row(
        innovation_r=50.0 * sigma_r0,
        innovation_b=0.5 * sigma_b0,
        base_range_variance_m2=sigma_r0**2,
        base_bearing_variance_rad2=sigma_b0**2,
        selected_correct_iou50=True,
    )
    # Small innovation, GT-incorrect: must be EXCLUDED, however small its NIS.
    probe_excluded = _lambda_row(
        innovation_r=0.0,
        innovation_b=0.0,
        base_range_variance_m2=sigma_r0**2,
        base_bearing_variance_rad2=sigma_b0**2,
        selected_correct_iou50=False,
    )

    bias = FrozenBiasCorrection.identity()
    without_probe = fit_covariance_scales(baseline, bias=bias)
    with_included_probe = fit_covariance_scales(baseline + [probe_included], bias=bias)
    with_both_probes = fit_covariance_scales(
        baseline + [probe_included, probe_excluded], bias=bias
    )

    # The GT-correct large-innovation probe changes the fit ...
    assert with_included_probe != pytest.approx(without_probe, rel=1e-9)
    # ... but the GT-incorrect probe, however small its own innovation,
    # changes nothing further: it was never in the fitting set.
    assert with_both_probes == pytest.approx(with_included_probe, rel=1e-12)

    # Order independence.
    shuffled = [probe_excluded, probe_included] + list(reversed(baseline))
    reordered = fit_covariance_scales(shuffled, bias=bias)
    assert reordered == pytest.approx(with_both_probes, rel=1e-9)

    # The per-axis bisection itself (the numerical core of the fit) is a
    # single monotonic root-find over a fixed bracket: seeding its search at
    # 1.0 vs 10.0 must converge to the identical root, because there is only
    # one root and no starting-point-dependent coordinate descent.
    from duckie_pomdp.evaluation.f9c_calibration import _bisect_lambda

    def probe_curve(lambda_r: float) -> float:
        return 10.0 / lambda_r  # strictly decreasing, crosses CHI2_1DOF_MEDIAN once

    from duckie_pomdp.evaluation.f9c_calibration import CHI2_1DOF_MEDIAN

    seeded_low = _bisect_lambda(probe_curve, target=CHI2_1DOF_MEDIAN, lower=1.0)
    seeded_high = _bisect_lambda(probe_curve, target=CHI2_1DOF_MEDIAN, lower=10.0)
    assert seeded_low == pytest.approx(seeded_high, rel=1e-6)


def test_robust_observation_switches_field_names_match_between_layers():
    """belief.robust_updater.RobustObservationSwitches and
    evaluation.f9c_protocol.RobustObservationSwitches are deliberately
    duplicated (importing evaluation from belief would invert the
    layering), but nothing guarded against drift before this test."""

    belief_fields = {field.name for field in fields(BeliefRobustObservationSwitches)}
    protocol_fields = {field.name for field in fields(ProtocolRobustObservationSwitches)}
    assert belief_fields == protocol_fields


# ---------------------------------------------------------------------------
# Fix round 1: invariant I8's miss-likelihood floor -- L_mean, global P_D/P_FA.
# ---------------------------------------------------------------------------


def _miss_row(episode: str, frame: int, *, eligible: bool, detected: bool) -> dict:
    return {
        "episode": episode,
        "frame": frame,
        "eligible_visible": eligible,
        "detector_detected": detected,
    }


def test_mean_genuine_miss_run_length_finds_consecutive_eligible_misses():
    from duckie_pomdp.evaluation.f9c_calibration import mean_genuine_miss_run_length

    # Episode "e1": detected pattern F,F,T,F,T,T,F,F,F -> miss runs 2,1,3.
    e1_detected = [False, False, True, False, True, True, False, False, False]
    rows = [
        _miss_row("e1", frame, eligible=True, detected=detected)
        for frame, detected in enumerate(e1_detected)
    ]
    # Episode "e2": one run of 5 consecutive genuine misses.
    rows += [
        _miss_row("e2", frame, eligible=True, detected=False) for frame in range(5)
    ]

    report = mean_genuine_miss_run_length(rows)
    assert report["run_count"] == 4
    assert report["mean_run_length"] == pytest.approx(2.75)
    assert report["max_run_length"] == 5


def test_mean_genuine_miss_run_length_excludes_ineligible_frames_from_the_run():
    """A frame where the pedestrian was not GT-visible is not a "genuine"
    miss and must split, not extend, a run."""

    from duckie_pomdp.evaluation.f9c_calibration import mean_genuine_miss_run_length

    rows = [
        _miss_row("e1", 0, eligible=True, detected=False),
        _miss_row("e1", 1, eligible=True, detected=False),
        _miss_row("e1", 2, eligible=False, detected=False),  # not a genuine miss
        _miss_row("e1", 3, eligible=True, detected=False),
    ]
    report = mean_genuine_miss_run_length(rows)
    assert report["run_count"] == 2
    assert sorted(report["run_lengths"]) == [1, 2]


def test_mean_genuine_miss_run_length_with_no_misses_reports_none():
    from duckie_pomdp.evaluation.f9c_calibration import mean_genuine_miss_run_length

    rows = [_miss_row("e1", frame, eligible=True, detected=True) for frame in range(5)]
    report = mean_genuine_miss_run_length(rows)
    assert report["run_count"] == 0
    assert report["mean_run_length"] is None


def test_fit_global_effective_detection_and_false_alarm_probability_are_beta_means():
    from duckie_pomdp.evaluation.f9c_calibration import (
        fit_false_alarm_probability,
        fit_global_effective_detection,
    )

    rows = (
        [_miss_row("e1", i, eligible=True, detected=True) for i in range(8)]
        + [_miss_row("e1", i, eligible=True, detected=False) for i in range(8, 10)]
        + [_miss_row("e1", i, eligible=False, detected=True) for i in range(10, 11)]
        + [_miss_row("e1", i, eligible=False, detected=False) for i in range(11, 15)]
    )
    detection = fit_global_effective_detection(rows)
    assert detection["trials_global"] == 10
    assert detection["successes_global"] == 8
    assert detection["detection_probability_global"] == pytest.approx((8 + 0.5) / 11.0)

    false_alarm = fit_false_alarm_probability(rows)
    assert false_alarm["trials"] == 5
    assert false_alarm["successes"] == 1
    assert false_alarm["false_positive_probability"] == pytest.approx((1 + 0.5) / 6.0)


def test_fit_miss_likelihood_floor_l_mean_misses_reproduce_lr_nominal():
    """floor ** l_mean must equal lr_nominal exactly (the construction
    invariant I8's docstring and ExistenceFilter tests both rely on)."""

    from duckie_pomdp.evaluation.f9c_calibration import fit_miss_likelihood_floor

    rows = (
        [_miss_row("e1", i, eligible=True, detected=True) for i in range(90)]
        + [_miss_row("e1", i, eligible=True, detected=False) for i in range(90, 100)]
    )
    # Five genuine-miss runs of length 2 each -> L_mean = 2.
    detected = [True] * 90
    rows = [_miss_row("e1", i, eligible=True, detected=d) for i, d in enumerate(detected)]
    for start in (10, 30, 50, 70, 88):
        rows[start]["detector_detected"] = False
        if start + 1 < len(rows):
            rows[start + 1]["detector_detected"] = False

    result = fit_miss_likelihood_floor(rows)
    assert result["l_mean"] == pytest.approx(2.0)
    assert result["lr_floor"] ** result["l_mean"] == pytest.approx(
        result["lr_nominal"], rel=1e-9
    )


def test_adjust_miss_likelihood_floor_substitutes_a_different_false_positive_rate():
    from duckie_pomdp.evaluation.f9c_calibration import (
        adjust_miss_likelihood_floor_for_false_positive_rate,
    )

    fit = {"detection_probability_global": 0.98, "l_mean": 4.0}
    adjusted = adjust_miss_likelihood_floor_for_false_positive_rate(
        fit, false_positive_probability=0.0008
    )
    expected_lr_nominal = (1.0 - 0.98) / (1.0 - 0.0008)
    assert adjusted["lr_nominal"] == pytest.approx(expected_lr_nominal)
    assert adjusted["lr_floor"] == pytest.approx(expected_lr_nominal ** (1.0 / 4.0))
    assert adjusted["false_positive_probability_used"] == pytest.approx(0.0008)


def test_load_miss_likelihood_floor_reads_the_frozen_fitted_value():
    """Task 10 froze [conditional_detection].miss_likelihood_floor to the
    final calibration re-run's LR_floor (previously 0.0, a strict no-op,
    per this function's own docstring: "until Task 10 freezes a fitted
    value"). Updated in lockstep with that freeze."""

    from pathlib import Path

    from duckie_pomdp.evaluation.f9c_calibration import load_miss_likelihood_floor

    root = Path(__file__).resolve().parents[1]
    floor = load_miss_likelihood_floor(root / "configs" / "f9c_robust_belief_v1.toml")
    assert floor == pytest.approx(0.37362469458201386)


def test_detection_by_range_and_fov_cross_tabulates_eligible_frames():
    from duckie_pomdp.evaluation.f9c_calibration import detection_by_range_and_fov

    def row(distance_bin, fov, detected):
        return {
            "eligible_visible": True,
            "distance_bin": distance_bin,
            "fov_region": fov,
            "detector_detected": detected,
        }

    rows = (
        [row("near", "center", True) for _ in range(9)]
        + [row("near", "center", False) for _ in range(1)]
        + [row("far", "edge_fov", True) for _ in range(5)]
    )
    table = detection_by_range_and_fov(rows)
    assert table["center"]["near"]["trials"] == 10
    assert table["center"]["near"]["successes"] == 9
    assert table["edge_fov"]["far"]["trials"] == 5
    assert table["edge_fov"]["far"]["successes"] == 5
