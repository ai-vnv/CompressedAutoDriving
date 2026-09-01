"""Task 8: Robust belief coordinator -- integration tests.

Reuses the ``measurement()`` helper pattern from ``tests/test_pedestrian_ekf.py``
(that file's line 43) rather than inventing a new fixture shape.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pytest

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.innovation_gate import InnovationGateConfig
from duckie_pomdp.belief.measurement_association import (
    AssociationConfig,
    CandidateMeasurement,
)
from duckie_pomdp.belief.observability import (
    EffectiveDetectionModel,
    ObservabilityClass,
    PredictedObservabilityModel,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    MeasurementProfile,
    load_pedestrian_ekf_config,
)
from duckie_pomdp.belief.robust_updater import (
    RobustObservationConfig,
    RobustObservationSwitches,
    RobustPedestrianBeliefUpdater,
)
from duckie_pomdp.belief.updater import (
    PedestrianBeliefUpdater,
    initial_belief,
    load_existence_filter_config,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.measurement import ObjectMeasurement, PerceptionObservation
from duckie_pomdp.domain.observation import EgoObservation
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.evaluation.f9c_calibration import (
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
from duckie_pomdp.evaluation.f9c_runtime_cache import (
    RuntimeCacheFrame,
    TruthFrame,
    read_runtime_cache,
    write_evaluation_truth,
    write_runtime_cache,
)
from duckie_pomdp.perception.camera_geometry import (
    CalibratedGroundProjector,
    CameraCalibration,
)
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise

ROOT = Path(__file__).resolve().parents[1]
F9C_CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"

sys.path.insert(0, str(ROOT / "experiments"))
import evaluate_f9c_robust_belief as evaluate_module  # noqa: E402

GATE_THRESHOLD = 9.21034037197618  # innovation gate: 2 DOF, 99%
ASSOCIATION_GATE_THRESHOLD = 13.815510557964274  # association: 2 DOF, 99.9%
STATIONARY = EgoMotion(0.0, 0.0)
DT = 1.0 / 30.0
NO_ACTION = PolicyAction(0.0, 0.0)

FROZEN_RANGE_BIAS_M = -0.045904804710162034
FROZEN_BEARING_BIAS_RAD = 0.00414567890700929


def measurement(range_m: float, bearing_rad: float, confidence: float = 1.0) -> ObjectMeasurement:
    """Mirrors ``tests/test_pedestrian_ekf.py:43``."""

    return ObjectMeasurement(
        object_class=ObjectClass.DUCKIE,
        detected=True,
        confidence=confidence,
        x_left_m=range_m * sin(bearing_rad),
        y_forward_m=range_m * cos(bearing_rad),
        range_m=range_m,
        bearing_rad=bearing_rad,
    )


def candidate(
    range_m: float,
    bearing_rad: float,
    confidence: float = 0.9,
    bbox: tuple[int, int, int, int] = (0, 0, 10, 10),
) -> CandidateMeasurement:
    return CandidateMeasurement(
        measurement=measurement(range_m, bearing_rad, confidence),
        confidence=confidence,
        bbox_key=bbox,
    )


def ekf_config():
    return load_pedestrian_ekf_config(F9C_CONFIG)


def base_noise():
    return load_polar_measurement_noise(F9C_CONFIG)


def existence_config(**overrides):
    base = load_existence_filter_config(F9C_CONFIG)
    return replace(base, **overrides) if overrides else base


def bias_frozen() -> FrozenBiasCorrection:
    return FrozenBiasCorrection(
        model="global_additive",
        range_bias_m=FROZEN_RANGE_BIAS_M,
        bearing_bias_rad=FROZEN_BEARING_BIAS_RAD,
        range_bin_bias_m=None,
        near_max_m=0.55,
        medium_max_m=0.80,
    )


def bias_fitted() -> FrozenBiasCorrection:
    # F9c calibration (Task 9) has not run yet; a deliberately distinct
    # placeholder keeps bias_refit=True observably different from the frozen
    # constants in any test that flips it. None of the required 11 tests
    # below exercise bias_refit=True, but the wiring must still accept it.
    return FrozenBiasCorrection(
        model="global_additive",
        range_bias_m=0.01,
        bearing_bias_rad=-0.001,
        range_bin_bias_m=None,
        near_max_m=0.55,
        medium_max_m=0.80,
    )


def observability_model():
    projector = CalibratedGroundProjector(
        CameraCalibration(
            image_width_px=640,
            image_height_px=480,
            vertical_fov_deg=75.0,
            camera_height_m=0.108,
            camera_pitch_deg=19.15,
            camera_forward_offset_m=0.066,
        )
    )
    return PredictedObservabilityModel(projector, image_width_px=640)


def effective_detection_model(
    *,
    center: float = 0.99,
    mid_fov: float = 0.95,
    edge_fov: float = 0.60,
    outside_domain: float = 0.05,
) -> EffectiveDetectionModel:
    return EffectiveDetectionModel(
        {
            ObservabilityClass.CENTER: center,
            ObservabilityClass.MID_FOV: mid_fov,
            ObservabilityClass.EDGE_FOV: edge_fov,
            ObservabilityClass.OUTSIDE_DOMAIN: outside_domain,
        },
        outside_domain_miss_policy="prediction_only",
    )


def robust_config(
    *,
    switches: RobustObservationSwitches | None = None,
    gate: InnovationGateConfig | None = None,
    association: AssociationConfig | None = None,
    covariance: CovarianceCalibration | None = None,
    effective_detection: EffectiveDetectionModel | None = None,
    active_threshold: float = 0.50,
    delete_threshold: float = 0.05,
    initialization_threshold: float = 0.50,
) -> RobustObservationConfig:
    return RobustObservationConfig(
        switches=switches
        or RobustObservationSwitches(
            bias_refit=False,
            innovation_gate=True,
            temporal_association=True,
            covariance_calibration=True,
            conditional_detection=True,
        ),
        gate=gate or InnovationGateConfig(GATE_THRESHOLD),
        association=association
        or AssociationConfig(
            chi_square_gate=ASSOCIATION_GATE_THRESHOLD,
            initialization_rule="highest_confidence_then_bbox_lexicographic",
        ),
        covariance=covariance or CovarianceCalibration(1.0, 1.0, 0.0, 0.0),
        effective_detection=effective_detection or effective_detection_model(),
        active_threshold=active_threshold,
        delete_threshold=delete_threshold,
        initialization_threshold=initialization_threshold,
    )


def robust_updater(**config_overrides) -> RobustPedestrianBeliefUpdater:
    return RobustPedestrianBeliefUpdater(
        ekf_config=ekf_config(),
        measurement_noise=base_noise(),
        existence_config=existence_config(),
        bias_frozen=bias_frozen(),
        bias_fitted=bias_fitted(),
        observability_model=observability_model(),
        config=robust_config(**config_overrides),
    )


def start_belief():
    return initial_belief(EgoObservation(0.0, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# 1. Robustness to a gross outlier vs. the unguarded baseline.
# ---------------------------------------------------------------------------


def test_a_gross_outlier_does_not_move_the_belief_more_than_the_baseline_does():
    """Feed 10 consistent measurements then one 0.30 m outlier.

    Robust belief range error after the outlier must be < 0.5 x baseline error.
    """

    true_range, true_bearing = 0.85, 0.0
    outlier_range = true_range + 0.30

    robust = robust_updater()
    baseline = PedestrianBeliefUpdater(
        ekf_config=ekf_config(),
        existence_config=existence_config(),
        measurement_noise=base_noise(),
        measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
    )

    robust_belief = start_belief()
    baseline_belief = start_belief()
    for _ in range(10):
        robust_belief, _ = robust.update(
            robust_belief,
            NO_ACTION,
            STATIONARY,
            [candidate(true_range, true_bearing)],
            DT,
        )
        corrected = bias_frozen().correct(measurement(true_range, true_bearing))
        baseline_belief = baseline.update(
            baseline_belief,
            NO_ACTION,
            STATIONARY,
            PerceptionObservation(
                ego=baseline_belief.ego,
                road=None,
                stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
                pedestrian=corrected,
            ),
            DT,
        )

    robust_belief, _ = robust.update(
        robust_belief, NO_ACTION, STATIONARY, [candidate(outlier_range, true_bearing)], DT
    )
    outlier_corrected = bias_frozen().correct(measurement(outlier_range, true_bearing))
    baseline_belief = baseline.update(
        baseline_belief,
        NO_ACTION,
        STATIONARY,
        PerceptionObservation(
            ego=baseline_belief.ego,
            road=None,
            stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
            pedestrian=outlier_corrected,
        ),
        DT,
    )

    robust_error = abs(robust_belief.pedestrian.range_mean_m - true_range)
    baseline_error = abs(baseline_belief.pedestrian.range_mean_m - true_range)
    assert baseline_error > 0.0
    assert robust_error < 0.5 * baseline_error


# ---------------------------------------------------------------------------
# 2. Posterior floor.
# ---------------------------------------------------------------------------


def test_reported_range_std_is_never_below_the_posterior_floor():
    """After 60 consistent updates the EKF std collapses; the reported std
    must still be >= range_posterior_floor_m."""

    floor_m = 0.02
    updater = robust_updater(covariance=CovarianceCalibration(1.0, 1.0, floor_m, 0.01))
    belief = start_belief()
    record = None
    for _ in range(60):
        belief, record = updater.update(
            belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
        )

    assert record is not None
    raw_std = updater.ekf.polar_moments(STATIONARY).standard_deviation[0]
    assert raw_std < floor_m, "the EKF must genuinely have collapsed below the floor"
    assert record.reported_range_std_m >= floor_m
    assert belief.pedestrian.range_std_m >= floor_m


# ---------------------------------------------------------------------------
# 3. Duplicate-frame candidate selection.
# ---------------------------------------------------------------------------


def test_a_duplicate_frame_selects_the_temporally_consistent_candidate():
    """Two candidates, the higher-confidence one 0.25 m off track.

    record.association.differed_from_highest_confidence is True and the
    selected measurement is the consistent one.
    """

    updater = robust_updater()
    belief = start_belief()
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0, confidence=0.9)], DT
    )

    off_track = candidate(1.10, 0.0, confidence=0.95, bbox=(0, 0, 10, 10))
    consistent = candidate(0.85, 0.0, confidence=0.40, bbox=(20, 0, 30, 10))
    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [off_track, consistent], DT
    )

    assert record.association.highest_confidence_index == 0
    assert record.association.differed_from_highest_confidence is True
    expected_range_m = bias_frozen().correct(consistent.measurement).range_m
    assert record.association.selected.measurement.range_m == pytest.approx(
        expected_range_m
    )


# ---------------------------------------------------------------------------
# 4. Invariant I2: a rejected localization is not an existence miss.
# ---------------------------------------------------------------------------


def test_a_rejected_localization_is_not_an_existence_miss():
    """Establish a track, then feed 6 frames whose single candidate is a
    0.30 m gross outlier. Every frame must record detector_detected=True,
    kinematic_measurement_accepted=False, and track_active=True; existence
    must stay above the active threshold. Contrast case: the same 6 frames
    with NO candidate at all must drive existence below the active
    threshold. The two sequences must differ.
    """

    true_range, true_bearing = 0.85, 0.0
    outlier_range = true_range + 0.30

    rejected = robust_updater()
    belief = start_belief()
    belief, _ = rejected.update(
        belief, NO_ACTION, STATIONARY, [candidate(true_range, true_bearing)], DT
    )
    rejected_existence = []
    for _ in range(6):
        belief, record = rejected.update(
            belief, NO_ACTION, STATIONARY, [candidate(outlier_range, true_bearing)], DT
        )
        assert record.detector_detected is True
        assert record.kinematic_measurement_accepted is False
        assert record.track_active is True
        rejected_existence.append(record.existence_probability)

    silent = robust_updater()
    belief = start_belief()
    belief, _ = silent.update(
        belief, NO_ACTION, STATIONARY, [candidate(true_range, true_bearing)], DT
    )
    silent_existence = []
    for _ in range(6):
        belief, record = silent.update(belief, NO_ACTION, STATIONARY, [], DT)
        silent_existence.append(record.existence_probability)

    active_threshold = rejected.config.active_threshold
    assert all(value > active_threshold for value in rejected_existence)
    assert silent_existence[-1] < active_threshold
    assert rejected_existence[-1] != pytest.approx(silent_existence[-1])


# ---------------------------------------------------------------------------
# 5. Invariant I1: one shared innovation covariance.
# ---------------------------------------------------------------------------


def test_association_gate_and_correction_share_one_innovation_covariance(monkeypatch):
    """Monkeypatch the updater's _innovation_covariance to record every
    matrix it returns within one update() call, and assert that the matrix
    used by the associator, the matrix passed to the gate, and the matrix
    implied by the EKF correction's R are identical to 1e-15. Additionally
    assert the returned R equals calibration.inflate(base_R), not base_R.
    """

    calibration = CovarianceCalibration(4.0, 2.0, 0.0, 0.0)
    updater = robust_updater(covariance=calibration)
    belief = start_belief()
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )

    recorded: list[np.ndarray] = []
    original = updater._innovation_covariance

    def spy(range_m: float) -> np.ndarray:
        result = original(range_m)
        recorded.append(result)
        return result

    monkeypatch.setattr(updater, "_innovation_covariance", spy)

    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.851, 0.001)], DT
    )

    assert record.kinematic_measurement_accepted is True
    assert len(recorded) >= 2, "association and the explicit gate step must both call it"
    for matrix in recorded[1:]:
        assert matrix == pytest.approx(recorded[0], abs=1e-15)

    assert updater.last_ekf_diagnostics is not None
    ekf_S = np.array(updater.last_ekf_diagnostics.innovation_covariance, dtype=float)
    assert ekf_S == pytest.approx(recorded[0], abs=1e-15)

    range_m = record.association.selected.measurement.range_m
    base_R = base_noise().covariance(range_m)
    inflated_R = calibration.inflate(base_R)
    adapter_R = updater._inflated_noise.covariance(range_m)
    assert adapter_R == pytest.approx(inflated_R, abs=1e-15)
    assert not np.allclose(adapter_R, base_R)


# ---------------------------------------------------------------------------
# 5b. Invariant I7: association must be strictly looser than the gate.
# ---------------------------------------------------------------------------


def test_association_gate_is_looser_than_the_innovation_gate():
    """Invariant I7. Association filters candidates before the gate sees them,
    so equal thresholds make the innovation gate unreachable and its ablation
    switch meaningless. The coordinator must refuse that configuration."""

    equal_threshold = GATE_THRESHOLD
    with pytest.raises(ValueError, match="unreachable"):
        RobustPedestrianBeliefUpdater(
            ekf_config=ekf_config(),
            measurement_noise=base_noise(),
            existence_config=existence_config(),
            bias_frozen=bias_frozen(),
            bias_fitted=bias_fitted(),
            observability_model=observability_model(),
            config=robust_config(
                gate=InnovationGateConfig(equal_threshold),
                association=AssociationConfig(
                    chi_square_gate=equal_threshold,
                    initialization_rule="highest_confidence_then_bbox_lexicographic",
                ),
            ),
        )


def test_a_candidate_between_the_two_thresholds_is_associated_then_gated():
    """Invariant I7, positive case. A candidate whose NIS falls strictly
    between the innovation-gate threshold (9.21) and the association gate
    (13.82) must be selected by association (mode == "temporal", selected is
    not None) but rejected by the innovation gate itself
    (`InnovationGate.evaluate`, not association's own internal filter):
    kinematic_measurement_accepted is False, record.gate is not None, and per
    invariant I2 detector_detected is still True. This is what proves the
    innovation_gate switch controls something once temporal_association is on.
    """

    updater = robust_updater()
    belief = start_belief()
    # Frame 1 initializes the track; frame 2 settles it (temporal scoring
    # only applies once a track already exists).
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    # +0.085 m range offset -> NIS ~= 11.48 against the settled track's S,
    # strictly between 9.21034037197618 and 13.815510557964274 (confirmed
    # empirically against this exact fixture sequence).
    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.935, 0.0)], DT
    )

    assert record.nis is not None
    assert GATE_THRESHOLD < record.nis < ASSOCIATION_GATE_THRESHOLD

    assert record.association.mode == "temporal"
    assert record.association.selected is not None

    # This is the assertion that proves `InnovationGate.evaluate()` itself
    # ran, rather than the candidate merely having been filtered out earlier
    # by association's own internal `chi_square_gate`: association already
    # selected it (asserted above), so a non-None, rejecting `record.gate`
    # can only come from the explicit gate step calling
    # `self.gate.evaluate(...)` on the associated candidate.
    assert record.gate is not None
    assert record.gate.accepted is False
    assert record.gate.threshold == pytest.approx(GATE_THRESHOLD)

    assert record.kinematic_measurement_accepted is False
    assert record.detector_detected is True


# ---------------------------------------------------------------------------
# 6. Belief survives misses at the edge of the field of view.
# ---------------------------------------------------------------------------


def test_belief_survives_five_consecutive_misses_when_predicted_edge_fov():
    """record.track_active stays True through 5 misses at EDGE_FOV."""

    updater = robust_updater(effective_detection=effective_detection_model())
    belief = start_belief()
    # x_left=0.45, y_forward=0.60 -> EDGE_FOV, mirrors test_f9c_observability.py.
    edge_range = (0.45**2 + 0.60**2) ** 0.5
    edge_bearing = np.arctan2(0.45, 0.60)
    # Frame 1 initializes the track; with no prior track, observability
    # classification falls back to the (zero) initial public belief mean and
    # is not meaningful yet. Frame 2 re-observes the same candidate so the
    # classification below reflects the real predicted state x̂⁻.
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(edge_range, edge_bearing)], DT
    )
    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(edge_range, edge_bearing)], DT
    )
    assert record.observability_class is ObservabilityClass.EDGE_FOV

    for _ in range(5):
        belief, record = updater.update(belief, NO_ACTION, STATIONARY, [], DT)
        assert record.observability_class is ObservabilityClass.EDGE_FOV
        assert record.track_active is True


# ---------------------------------------------------------------------------
# 7. Invariant I3 at the coordinator level.
# ---------------------------------------------------------------------------


def test_an_outside_domain_absence_preserves_belief_far_longer_than_an_in_domain_one():
    """30 missing frames with the belief predicted OUTSIDE_DOMAIN must leave
    P(e) > 0.80 and track_deleted False; the same 30 frames predicted CENTER
    must delete the track.
    """

    outside = robust_updater()
    belief = start_belief()
    # bearing=pi -> y_forward = -range < 0 -> OUTSIDE_DOMAIN (behind camera).
    # Frame 1 initializes; frame 2 re-observes so classification uses the
    # real predicted state rather than the (zero) initial belief fallback.
    belief, _ = outside.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, pi)], DT
    )
    belief, record = outside.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, pi)], DT
    )
    assert record.observability_class is ObservabilityClass.OUTSIDE_DOMAIN
    for _ in range(30):
        belief, record = outside.update(belief, NO_ACTION, STATIONARY, [], DT)
    assert record.existence_probability > 0.80
    assert record.track_deleted is False

    center = robust_updater()
    belief = start_belief()
    belief, _ = center.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    belief, record = center.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    assert record.observability_class is ObservabilityClass.CENTER
    deletions = 0
    for _ in range(30):
        belief, record = center.update(belief, NO_ACTION, STATIONARY, [], DT)
        if record.track_deleted:
            deletions += 1
    assert deletions >= 1
    assert record.track_active is False


# ---------------------------------------------------------------------------
# 8. Belief dies after a long absence.
# ---------------------------------------------------------------------------


def test_belief_still_dies_after_a_long_absence():
    """After 40 consecutive misses at CENTER, P(e) < delete_threshold and
    record.track_deleted becomes True exactly once."""

    updater = robust_updater()
    belief = start_belief()
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    assert record.observability_class is ObservabilityClass.CENTER

    deletions = 0
    for _ in range(40):
        belief, record = updater.update(belief, NO_ACTION, STATIONARY, [], DT)
        if record.track_deleted:
            deletions += 1

    assert record.existence_probability < updater.config.delete_threshold
    assert deletions == 1


# ---------------------------------------------------------------------------
# 9. Reinitialization after deletion.
# ---------------------------------------------------------------------------


def test_a_deleted_track_reinitializes_from_the_next_valid_candidate():
    """After deletion, one good candidate produces frame_mode == 'initialization'
    and an initialized EKF."""

    updater = robust_updater()
    belief = start_belief()
    belief, _ = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    for _ in range(4):
        belief, record = updater.update(belief, NO_ACTION, STATIONARY, [], DT)
    assert updater.ekf.initialized is False

    belief, record = updater.update(
        belief, NO_ACTION, STATIONARY, [candidate(0.85, 0.0)], DT
    )
    assert record.frame_mode == "initialization"
    assert updater.ekf.initialized is True


# ---------------------------------------------------------------------------
# 10. All five switches off reproduces the F9b path exactly.
# ---------------------------------------------------------------------------


def test_switching_every_robust_component_off_reproduces_the_f9b_path():
    """With all five switches False, the belief sequence must equal
    PedestrianBeliefUpdater's output to within 1e-12 on the same inputs.
    This test is the Baseline-A regression guard.
    """

    all_off = RobustObservationSwitches(
        bias_refit=False,
        innovation_gate=False,
        temporal_association=False,
        covariance_calibration=False,
        conditional_detection=False,
    )
    robust = robust_updater(
        switches=all_off, covariance=CovarianceCalibration(1.0, 1.0, 0.0, 0.0)
    )
    baseline = PedestrianBeliefUpdater(
        ekf_config=ekf_config(),
        existence_config=existence_config(),
        measurement_noise=base_noise(),
        measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
    )

    frames = [
        (0.90, 0.05, STATIONARY),
        (0.86, 0.04, EgoMotion(0.15, 0.0)),
        (0.80, 0.03, STATIONARY),
        (None, None, EgoMotion(0.0, 0.3)),  # miss
        (0.70, -0.02, STATIONARY),
        (None, None, STATIONARY),  # miss
        (0.65, -0.05, EgoMotion(0.1, -0.2)),
        (0.60, -0.06, STATIONARY),
    ]

    robust_belief = start_belief()
    baseline_belief = start_belief()
    for range_m, bearing_rad, motion in frames:
        if range_m is None:
            robust_candidates: list[CandidateMeasurement] = []
            raw = ObjectMeasurement.missing(ObjectClass.DUCKIE)
        else:
            robust_candidates = [candidate(range_m, bearing_rad, confidence=1.0)]
            raw = measurement(range_m, bearing_rad)

        robust_belief, _ = robust.update(
            robust_belief, NO_ACTION, motion, robust_candidates, DT
        )
        corrected = bias_frozen().correct(raw)
        baseline_belief = baseline.update(
            baseline_belief,
            NO_ACTION,
            motion,
            PerceptionObservation(
                ego=baseline_belief.ego,
                road=None,
                stop_sign=ObjectMeasurement.missing(ObjectClass.STOP_SIGN),
                pedestrian=corrected,
            ),
            DT,
        )

    r = robust_belief.pedestrian
    b = baseline_belief.pedestrian
    assert r.existence_probability == pytest.approx(b.existence_probability, abs=1e-12)
    assert r.range_mean_m == pytest.approx(b.range_mean_m, abs=1e-12)
    assert r.range_std_m == pytest.approx(b.range_std_m, abs=1e-12)
    assert r.bearing_mean_rad == pytest.approx(b.bearing_mean_rad, abs=1e-12)
    assert r.bearing_std_rad == pytest.approx(b.bearing_std_rad, abs=1e-12)
    assert r.radial_velocity_mean_mps == pytest.approx(
        b.radial_velocity_mean_mps, abs=1e-12
    )
    assert r.radial_velocity_std_mps == pytest.approx(
        b.radial_velocity_std_mps, abs=1e-12
    )
    assert r.bearing_rate_mean_rad_s == pytest.approx(
        b.bearing_rate_mean_rad_s, abs=1e-12
    )
    assert r.bearing_rate_std_rad_s == pytest.approx(
        b.bearing_rate_std_rad_s, abs=1e-12
    )


# ---------------------------------------------------------------------------
# 11. No privileged state in the update() signature.
# ---------------------------------------------------------------------------


def test_updater_never_receives_privileged_state():
    import inspect

    parameters = set(
        inspect.signature(RobustPedestrianBeliefUpdater.update).parameters
    )
    assert parameters == {
        "self",
        "previous_belief",
        "previous_action",
        "ego_motion",
        "candidates",
        "dt_s",
    }


# ---------------------------------------------------------------------------
# Task 12: ablation from the runtime cache.
# ---------------------------------------------------------------------------


F9C_REAL_PROTOCOL = load_f9c_protocol(F9C_CONFIG, require_frozen=True)

_F9C_FITTED_RANGE_BIAS_M = -0.02986607430110723
_F9C_FITTED_BEARING_BIAS_RAD = 0.0012336629252072933


def _deep_approx_equal(
    actual: object, expected: object, *, abs_tol: float = 1e-12, path: str = ""
) -> list[tuple[str, object, object]]:
    """Recursive dict/scalar comparison, returning every mismatch found (empty
    list means an exact match to within ``abs_tol`` on every float leaf)."""

    mismatches: list[tuple[str, object, object]] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            mismatches.append((path, sorted(actual), sorted(expected)))
            return mismatches
        for key in actual:
            mismatches.extend(
                _deep_approx_equal(actual[key], expected[key], abs_tol=abs_tol, path=f"{path}.{key}")
            )
        return mismatches
    if isinstance(actual, float) and isinstance(expected, float):
        if actual != actual and expected != expected:  # both NaN
            return mismatches
        if abs(actual - expected) > abs_tol:
            mismatches.append((path, actual, expected))
        return mismatches
    if actual != expected:
        mismatches.append((path, actual, expected))
    return mismatches


#: Every sub-report ``summarize_f9c`` computes per-variant (``baseline_a``/
#: ``robust_b``) -- i.e. every section the endpoint-equality claim could
#: plausibly be checked against. Fix round 1: the committed test previously
#: checked only ``"ekf"``; ``track_continuity``/``nis``/``miss_sequence``
#: were verified ad hoc, not in the committed test, which is a regression
#: hole on the one test the whole ablation's validity rests on. All four are
#: now asserted.
_PER_VARIANT_METRIC_SECTIONS: tuple[str, ...] = ("ekf", "track_continuity", "nis", "miss_sequence")


def test_ablation_endpoints_match_the_two_headline_systems():
    """ablation['baseline'] must equal the Baseline-A metrics and
    ablation['all_combined'] must equal the Robust-B metrics, field by
    field, to within 1e-12 -- checked across every per-variant section
    ``summarize_f9c`` produces (``ekf``, ``track_continuity``, ``nis``,
    ``miss_sequence``), not just ``ekf``.

    Reference values come from freshly (re-)computed row sets, never from a
    hand-typed expectation: 'Baseline-A metrics' is ablation['baseline']'s
    OWN baseline_a sub-report (computed by the SAME baseline_updater every
    ablation row shares, unaffected by any ablation switch), and
    'Robust-B metrics' is a completely independent canonical replay via
    ``replay_from_cache`` (Task 11's own reconstruction path, using the
    frozen production config unchanged).
    """

    ablation = evaluate_module.run_ablation(F9C_REAL_PROTOCOL)

    for section in _PER_VARIANT_METRIC_SECTIONS:
        baseline_robust_b = ablation["baseline"]["metrics"][section]["robust_b"]
        baseline_baseline_a = ablation["baseline"]["metrics"][section]["baseline_a"]
        mismatches = _deep_approx_equal(baseline_robust_b, baseline_baseline_a)
        assert not mismatches, f"'baseline'.{section} does not equal Baseline A: {mismatches[:10]}"

    reference_rows, _, _ = evaluate_module.replay_from_cache(F9C_REAL_PROTOCOL)
    from duckie_pomdp.evaluation.f9c_belief import summarize_f9c

    reference_metrics, _ = summarize_f9c(reference_rows, protocol=F9C_REAL_PROTOCOL)

    for section in _PER_VARIANT_METRIC_SECTIONS:
        all_combined_robust_b = ablation["all_combined"]["metrics"][section]["robust_b"]
        mismatches = _deep_approx_equal(all_combined_robust_b, reference_metrics[section]["robust_b"])
        assert not mismatches, f"'all_combined'.{section} does not equal Robust B: {mismatches[:10]}"

    # Also confirm the seven-row shape and that every row replayed the
    # SAME 3328 frames.
    row_names = {name for name, _ in evaluate_module.ABLATION_ROWS}
    assert row_names == {
        "baseline",
        "bias_refit_only",
        "innovation_gate_only",
        "temporal_association_only",
        "covariance_calibration_only",
        "conditional_detection_only",
        "all_combined",
    }
    for name in row_names:
        assert ablation[name]["row_count"] == len(reference_rows)


def _small_cache_and_truth(tmp_path: Path) -> tuple[Path, Path, str, str]:
    """A hand-written, two-frame, one-episode cache/truth pair -- no
    simulator, no detector -- used to prove invariant I4."""

    scenario_name = F9C_REAL_PROTOCOL.scenarios[0].name
    episode = f"synthetic_{scenario_name}"
    frames = (
        RuntimeCacheFrame(
            episode=episode,
            seed=9999,
            scenario=scenario_name,
            frame=0,
            dt_s=1.0 / 30.0,
            raw_candidate_range_m=(0.70,),
            raw_candidate_bearing_rad=(0.02,),
            raw_candidate_confidence=(0.9,),
            raw_candidate_bbox=((250.0, 200.0, 350.0, 300.0),),
            raw_candidate_projection_failed=(False,),
            ego_linear_velocity_mps=0.0,
            ego_yaw_rate_rad_s=0.0,
        ),
        RuntimeCacheFrame(
            episode=episode,
            seed=9999,
            scenario=scenario_name,
            frame=1,
            dt_s=1.0 / 30.0,
            raw_candidate_range_m=(0.68,),
            raw_candidate_bearing_rad=(0.01,),
            raw_candidate_confidence=(0.9,),
            raw_candidate_bbox=((252.0, 202.0, 352.0, 302.0),),
            raw_candidate_projection_failed=(False,),
            ego_linear_velocity_mps=0.0,
            ego_yaw_rate_rad_s=0.0,
        ),
    )
    truths = tuple(
        TruthFrame(
            episode=episode,
            frame=frame,
            gt_exists=True,
            gt_range_m=0.70 - 0.01 * frame,
            gt_bearing_rad=0.02 - 0.01 * frame,
            gt_range_rate_mps=0.0,
            gt_bearing_rate_rad_s=0.0,
            eligible_visible=True,
            visible_pixel_count=500,
            gt_bbox=(250.0, 200.0, 350.0, 300.0),
            distance_bin="near",
            fov_region="center",
        )
        for frame in (0, 1)
    )

    cache_path = tmp_path / "cache.npz"
    truth_path = tmp_path / "truth.npz"
    runtime_cache_sha256 = write_runtime_cache(cache_path, frames)
    evaluation_truth_sha256 = write_evaluation_truth(truth_path, truths)
    return cache_path, truth_path, runtime_cache_sha256, evaluation_truth_sha256


def test_ablation_performs_no_inference_and_no_render(monkeypatch, tmp_path: Path):
    """Invariant I4. Monkeypatch YoloObjectDetector.__init__ and
    create_gym_duckietown to raise, then run the ablation entry point on a
    small hand-written runtime cache. It must complete and produce seven
    result sets."""

    def _raise(*args, **kwargs):
        raise AssertionError("ablation must not run inference or render")

    monkeypatch.setattr(evaluate_module.YoloObjectDetector, "__init__", _raise)
    monkeypatch.setattr(evaluate_module, "create_gym_duckietown", _raise)

    cache_path, truth_path, runtime_cache_sha256, evaluation_truth_sha256 = _small_cache_and_truth(
        tmp_path
    )
    fake_protocol = replace(
        F9C_REAL_PROTOCOL,
        artifacts={
            **F9C_REAL_PROTOCOL.artifacts,
            "runtime_cache": cache_path,
            "evaluation_truth": truth_path,
        },
    )

    ablation = evaluate_module.run_ablation(
        fake_protocol,
        expected_runtime_cache_sha256=runtime_cache_sha256,
        expected_evaluation_truth_sha256=evaluation_truth_sha256,
    )

    row_names = {name for name, _ in evaluate_module.ABLATION_ROWS}
    assert row_names <= set(ablation)
    assert len(row_names) == 7
    for name in row_names:
        assert ablation[name]["row_count"] == 2
        assert "metrics" in ablation[name]


def test_ablation_refuses_a_runtime_cache_whose_hash_does_not_match(tmp_path: Path):
    """read_runtime_cache must raise when the recorded SHA256 disagrees
    with the file, so a silently regenerated cache cannot be replayed as
    if it were the final run's."""

    cache_path, truth_path, _runtime_cache_sha256, evaluation_truth_sha256 = _small_cache_and_truth(
        tmp_path
    )
    fake_protocol = replace(
        F9C_REAL_PROTOCOL,
        artifacts={
            **F9C_REAL_PROTOCOL.artifacts,
            "runtime_cache": cache_path,
            "evaluation_truth": truth_path,
        },
    )

    with pytest.raises(ValueError):
        evaluate_module.run_ablation(
            fake_protocol,
            expected_runtime_cache_sha256="0" * 64,
            expected_evaluation_truth_sha256=evaluation_truth_sha256,
        )

    # And read_runtime_cache itself, the lower-level primitive run_ablation
    # relies on, refuses the same way given a wrong hash directly.
    with pytest.raises(ValueError):
        read_runtime_cache(cache_path, expected_sha256="0" * 64)


def _real_bias_frozen() -> FrozenBiasCorrection:
    return load_frozen_bias_correction(F9C_CONFIG, section="baseline_measurement_model")


def _real_bias_fitted() -> FrozenBiasCorrection:
    return load_frozen_bias_correction(F9C_CONFIG, section="measurement_model")


def _build_row_updater(switches: RobustObservationSwitches) -> RobustPedestrianBeliefUpdater:
    """Build a RobustPedestrianBeliefUpdater exactly the way
    ``evaluate_f9c_robust_belief._replay_rows`` builds one for a given
    ablation row's switches -- using the SAME module-private helper
    functions the ablation itself uses, not a re-implementation."""

    settings = evaluate_module._ExperimentSettings(F9C_CONFIG)
    components = evaluate_module._replay_components(F9C_REAL_PROTOCOL, settings)
    fitted_robust_config = load_robust_observation_config(F9C_CONFIG)
    fitted_miss_likelihood_floor = load_miss_likelihood_floor(F9C_CONFIG)

    existence_config = evaluate_module._existence_config_for_switches(
        components["baseline_existence_config"],
        fitted_miss_likelihood_floor=fitted_miss_likelihood_floor,
        switches=switches,
    )
    robust_config = evaluate_module._robust_config_for_switches(fitted_robust_config, switches)

    return RobustPedestrianBeliefUpdater(
        ekf_config=components["ekf_config"],
        measurement_noise=components["measurement_noise"],
        existence_config=existence_config,
        bias_frozen=components["robust_bias_frozen"],
        bias_fitted=components["robust_bias_fitted"],
        observability_model=components["observability_model"],
        config=robust_config,
    )


def test_bias_ablation_uses_f9b_bias_when_switch_off():
    """Every ablation row with bias_refit=False must carry a bias stage
    whose range_bias_m equals -0.045904804710162034 and bearing_bias_rad
    equals 0.00414567890700929 -- the F9b frozen constants, not identity
    and not the F9c fitted values. Asserted on the constructed updater's
    own ``_bias_frozen`` attribute (the object ``update()`` actually
    dispatches to when ``switches.bias_refit`` is False), so the check
    cannot be satisfied by a coincidentally matching metric."""

    off_row_names = (
        "baseline",
        "innovation_gate_only",
        "temporal_association_only",
        "covariance_calibration_only",
        "conditional_detection_only",
    )
    for name, switches in evaluate_module.ABLATION_ROWS:
        if name not in off_row_names:
            continue
        assert switches.bias_refit is False
        updater = _build_row_updater(switches)
        assert updater._bias_frozen.model != "identity"
        assert updater._bias_frozen.range_bias_m == pytest.approx(
            FROZEN_RANGE_BIAS_M, abs=1e-15
        )
        assert updater._bias_frozen.bearing_bias_rad == pytest.approx(
            FROZEN_BEARING_BIAS_RAD, abs=1e-15
        )


def test_bias_refit_switch_applies_f9c_frozen_bias_before_association():
    """With bias_refit=True, feed one candidate and capture the range the
    associator receives. It must equal raw_range - f9c_range_bias_m,
    proving the correction happens upstream of association rather than
    after the gate or not at all. Run the same frame with bias_refit=False
    and assert the associator receives raw_range - f9b_range_bias_m
    instead."""

    raw_range_m = 0.70
    raw_bearing_rad = 0.0
    one_candidate = [candidate(raw_range_m, raw_bearing_rad, confidence=1.0)]

    fitted_switches = RobustObservationSwitches(
        bias_refit=True,
        innovation_gate=False,
        temporal_association=False,
        covariance_calibration=False,
        conditional_detection=False,
    )
    frozen_switches = replace(fitted_switches, bias_refit=False)

    for switches, expected_bias_m in (
        (fitted_switches, _F9C_FITTED_RANGE_BIAS_M),
        (frozen_switches, FROZEN_RANGE_BIAS_M),
    ):
        updater = _build_row_updater(switches)
        belief, record = updater.update(start_belief(), NO_ACTION, STATIONARY, one_candidate, DT)

        assert record.association.selected is not None
        received_range_m = record.association.selected.measurement.range_m
        assert received_range_m == pytest.approx(
            max(0.0, raw_range_m - expected_bias_m), abs=1e-12
        )


def test_runtime_cache_contains_pre_bias_raw_candidates(tmp_path: Path):
    """Invariant I5. Write a cache from a known pipeline observation, read
    it back, and assert raw_candidate_range_m equals the projector's raw
    output -- not that value minus any bias constant. Then replay the same
    candidate twice, once with bias_refit=False and once with True, and
    assert the two runs see candidate ranges differing by exactly
    (f9c_range_bias_m - f9b_range_bias_m). A cache written post-correction
    would make that difference zero."""

    raw_range_m = 0.75
    raw_bearing_rad = -0.03
    frame = RuntimeCacheFrame(
        episode="e",
        seed=1,
        scenario="s",
        frame=0,
        dt_s=1.0 / 30.0,
        raw_candidate_range_m=(raw_range_m,),
        raw_candidate_bearing_rad=(raw_bearing_rad,),
        raw_candidate_confidence=(0.95,),
        raw_candidate_bbox=((100.0, 100.0, 150.0, 150.0),),
        raw_candidate_projection_failed=(False,),
        ego_linear_velocity_mps=0.0,
        ego_yaw_rate_rad_s=0.0,
    )
    cache_path = tmp_path / "cache.npz"
    write_runtime_cache(cache_path, (frame,))
    (restored,) = read_runtime_cache(cache_path)

    # I5: the cache stores the RAW candidate, not raw - bias.
    assert restored.raw_candidate_range_m[0] == pytest.approx(raw_range_m, abs=1e-12)
    assert restored.raw_candidate_bearing_rad[0] == pytest.approx(raw_bearing_rad, abs=1e-12)

    one_candidate = [candidate(raw_range_m, raw_bearing_rad, confidence=1.0)]
    base_switches = RobustObservationSwitches(
        bias_refit=False,
        innovation_gate=False,
        temporal_association=False,
        covariance_calibration=False,
        conditional_detection=False,
    )

    ranges: dict[bool, float] = {}
    for bias_refit in (False, True):
        switches = replace(base_switches, bias_refit=bias_refit)
        updater = _build_row_updater(switches)
        _belief, record = updater.update(start_belief(), NO_ACTION, STATIONARY, one_candidate, DT)
        assert record.association.selected is not None
        ranges[bias_refit] = record.association.selected.measurement.range_m

    expected_difference = _F9C_FITTED_RANGE_BIAS_M - FROZEN_RANGE_BIAS_M
    # bias_refit=True subtracts the (more negative) F9c bias; bias_refit=False
    # subtracts the F9b bias -- corrected_range = raw - bias, so the True
    # run's range minus the False run's range equals -(f9c_bias - f9b_bias).
    actual_difference = ranges[False] - ranges[True]
    assert actual_difference == pytest.approx(expected_difference, abs=1e-12)
