"""Robust belief coordinator: wires every F9c observation-model component
onto the frozen ``PedestrianEKF`` / ``ExistenceFilter`` kinematics.

This module is pure wiring. It never reimplements filter mathematics --
``PedestrianEKF.predict``/``correct`` are called, never re-derived -- and it
never edits ``pedestrian_ekf.py``. The one piece of injection it performs is
installing a lambda-inflated measurement-noise adapter on the frozen EKF's
public ``measurement_noise`` attribute (see ``_LambdaInflatedNoise`` below);
this is what "frozen" is allowed to mean in practice.

Two invariants govern every line here:

- **I1** (single innovation covariance): association, the innovation gate,
  and the EKF correction must all threshold/correct against the *same*
  ``S = H(x̂⁻) P⁻ H(x̂⁻)ᵀ + λR(range_m)`` for the same candidate on the same
  frame. ``_innovation_covariance`` is the one function that builds it, and
  it is the one function whose ``range_m`` argument must always be the
  candidate's MEASURED (bias-corrected) range, matching exactly what
  ``PedestrianEKF.correct`` does internally (pedestrian_ekf.py:182-192): H
  from the predicted state, R looked up at the measured range.
- **I2** (detection evidence is not measurement acceptance): existence is
  driven by ``detector_detected`` -- a raw candidate existed in this frame --
  never by ``kinematic_measurement_accepted``. A frame where the detector
  found the pedestrian but the gate rejected the bbox is scored as a
  detection for existence and as prediction-only for the EKF.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from duckie_pomdp.belief.bias_correction import FrozenBiasCorrection
from duckie_pomdp.belief.covariance_calibration import CovarianceCalibration
from duckie_pomdp.belief.existence_filter import ExistenceFilter, ExistenceFilterConfig
from duckie_pomdp.belief.innovation_gate import (
    GateDecision,
    InnovationGate,
    InnovationGateConfig,
    normalized_innovation_squared,
)
from duckie_pomdp.belief.measurement_association import (
    AssociationConfig,
    AssociationResult,
    CandidateMeasurement,
    MeasurementAssociator,
)
from duckie_pomdp.belief.observability import (
    EffectiveDetectionModel,
    ObservabilityClass,
    PredictedObservabilityModel,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    EKFStepDiagnostics,
    MeasurementProfile,
    PedestrianEKF,
    PedestrianEKFConfig,
    measurement_function,
    measurement_jacobian,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import BeliefState, PedestrianBelief
from duckie_pomdp.domain.state import EgoMotion
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import PolarMeasurementNoiseModel


@dataclass(frozen=True)
class RobustObservationSwitches:
    """Mirrors ``evaluation.f9c_protocol.RobustObservationSwitches`` field
    for field. Declared independently here rather than imported from
    ``evaluation`` so the belief layer does not depend on the experiment
    harness layer above it; the two are kept in lock-step by construction
    (Task 9/12 build one from the other with ``RobustObservationSwitches(
    **dataclasses.asdict(protocol.robust))``).
    """

    bias_refit: bool
    innovation_gate: bool
    temporal_association: bool
    covariance_calibration: bool
    conditional_detection: bool


@dataclass(frozen=True)
class RobustObservationConfig:
    switches: RobustObservationSwitches
    gate: InnovationGateConfig
    association: AssociationConfig
    covariance: CovarianceCalibration
    effective_detection: EffectiveDetectionModel
    active_threshold: float
    delete_threshold: float
    initialization_threshold: float

    def __post_init__(self) -> None:
        for name, value in (
            ("active_threshold", self.active_threshold),
            ("delete_threshold", self.delete_threshold),
            ("initialization_threshold", self.initialization_threshold),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not self.delete_threshold < self.initialization_threshold:
            raise ValueError(
                "delete_threshold must be strictly below initialization_threshold"
            )


@dataclass(frozen=True)
class RobustStepRecord:
    frame_mode: str
    detector_detected: bool
    kinematic_measurement_accepted: bool
    association: AssociationResult
    gate: GateDecision | None
    effective_detection_probability: float
    observation_informative: bool
    observability_class: ObservabilityClass
    existence_probability: float
    track_active: bool
    track_deleted: bool
    nis: float | None
    reported_range_std_m: float
    reported_bearing_std_rad: float


class _LambdaInflatedNoise:
    """Duck-typed ``PolarMeasurementNoiseModel`` adapter installed on the
    frozen EKF's ``measurement_noise`` attribute so ``correct()`` -- and
    ``initialize()`` -- use exactly ``calibration.inflate(base.covariance(r))``
    as their R, the same R invariant I1's ``_innovation_covariance`` uses.

    Bias passes straight through the base model unchanged. Every F9c/F9
    measurement-noise config (``f9c_robust_belief_v1.toml``,
    ``f9_yolo_ekf_v1.toml``) carries zero residual bias in ``[range_noise.*]``
    / ``[bearing]`` -- bias correction lives entirely in the separate,
    externally-applied ``FrozenBiasCorrection`` stage (step 4b in
    ``RobustPedestrianBeliefUpdater.update``), never inside the EKF's own
    bias lookup, or a detected candidate would be bias-corrected twice.
    """

    def __init__(
        self, base: PolarMeasurementNoiseModel, calibration: CovarianceCalibration
    ) -> None:
        self._base = base
        self._calibration = calibration

    def covariance(self, range_m: float) -> NDArray[np.float64]:
        return self._calibration.inflate(self._base.covariance(range_m))

    def bias(self, range_m: float) -> NDArray[np.float64]:
        return self._base.bias(range_m)


class RobustPedestrianBeliefUpdater:
    """Coordinates every F9c robust-observation component around the frozen
    ``PedestrianEKF`` / ``ExistenceFilter`` kinematics. See the module
    docstring for invariants I1/I2 and the class docstring of each injected
    component for its own contract.
    """

    def __init__(
        self,
        *,
        ekf_config: PedestrianEKFConfig,
        measurement_noise: PolarMeasurementNoiseModel,
        existence_config: ExistenceFilterConfig,
        bias_frozen: FrozenBiasCorrection,
        bias_fitted: FrozenBiasCorrection,
        observability_model: PredictedObservabilityModel,
        config: RobustObservationConfig,
    ) -> None:
        self.config = config
        self._bias_frozen = bias_frozen
        self._bias_fitted = bias_fitted
        self._observability_model = observability_model
        self._minimum_range_m = ekf_config.minimum_range_m

        # Invariant I7: association must filter more loosely than the
        # innovation gate rejects. Association runs first (steps 5/6) and
        # discards every candidate whose NIS exceeds `association.chi_square_gate`
        # before the gate (step 7) ever sees a candidate; if the two
        # thresholds are equal (or association is tighter), no candidate
        # that reaches the gate could ever fail it, and `innovation_gate`
        # becomes a switch that controls nothing. Fail loudly here rather
        # than silently producing a meaningless ablation cell in Task 12.
        if config.association.chi_square_gate <= config.gate.chi_square_threshold:
            raise ValueError(
                "invariant I7 violated: association.chi_square_gate "
                f"({config.association.chi_square_gate}) must be strictly "
                "greater than gate.chi_square_threshold "
                f"({config.gate.chi_square_threshold}); an association gate "
                "at or below the innovation-gate threshold makes the "
                "innovation gate unreachable -- association would already "
                "have discarded every candidate the gate could reject."
            )

        # Invariant I1: construct the lambda-inflated noise adapter ONCE and
        # install it on the EKF at construction time (via the constructor
        # argument, i.e. injection through the frozen class's own public
        # ``measurement_noise`` attribute) so `correct()`/`initialize()` use
        # exactly the same inflated R that association, the gate, and
        # `_innovation_covariance` below all use. `pedestrian_ekf.py` is
        # never edited.
        self._inflated_noise = _LambdaInflatedNoise(measurement_noise, config.covariance)
        self.ekf = PedestrianEKF(
            config=ekf_config,
            measurement_noise=self._inflated_noise,
            measurement_profile=MeasurementProfile.CALIBRATED_RESIDUAL,
        )
        self.existence = ExistenceFilter(existence_config)
        self.associator = MeasurementAssociator(config.association)
        self.gate = InnovationGate(config.gate)
        self.last_record: RobustStepRecord | None = None
        self.last_ekf_diagnostics: EKFStepDiagnostics | None = None

    def reset(self) -> None:
        self.ekf.reset()
        self.existence.reset()
        self.last_record = None
        self.last_ekf_diagnostics = None

    def _innovation_covariance(self, range_m: float) -> NDArray[np.float64]:
        """The single Invariant-I1 provider: ``H(x̂⁻) P⁻ H(x̂⁻)ᵀ + λR(range_m)``.

        ``range_m`` must always be the candidate's MEASURED (bias-corrected)
        range -- never the predicted range. ``PedestrianEKF.correct`` builds
        H from the predicted state but looks R up at the *measured* range
        (pedestrian_ekf.py:182-192); reproducing anything else here would
        threshold association and the gate against a different S than the
        one `correct()` actually applies, breaking invariant I1 silently.

        Requires ``self.ekf.initialized`` -- called only from the "temporal"
        branches below, which only exist once a predicted state does.
        """

        jacobian = measurement_jacobian(
            self.ekf.state, minimum_range_m=self._minimum_range_m
        )
        predicted_covariance = self.ekf.covariance
        return (
            jacobian @ predicted_covariance @ jacobian.T
            + self._inflated_noise.covariance(range_m)
        )

    def update(
        self,
        previous_belief: BeliefState,
        previous_action: PolicyAction,
        ego_motion: EgoMotion,
        candidates: Sequence[CandidateMeasurement],
        dt_s: float,
    ) -> tuple[BeliefState, RobustStepRecord]:
        switches = self.config.switches
        track_was_active = self.ekf.initialized

        # 1. EKF predict with actual ego motion (frozen; no-op until a track
        # is initialized).
        self.ekf.predict(ego_motion, dt_s)

        # 2. Classify predicted observability from x̂⁻. Before any track
        # exists there is no x̂⁻ to classify; fall back to the previously
        # *reported* (public, never privileged) belief mean so a
        # classification is always available for the existence update below.
        if self.ekf.initialized:
            predicted_state_for_observability = self.ekf.state
        else:
            previous_pedestrian = previous_belief.pedestrian
            predicted_state_for_observability = np.array(
                [
                    previous_pedestrian.range_mean_m
                    * sin(previous_pedestrian.bearing_mean_rad),
                    previous_pedestrian.range_mean_m
                    * cos(previous_pedestrian.bearing_mean_rad),
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        observability = self._observability_model.classify(
            predicted_state_for_observability
        )

        # 3. Resolve P_D^eff and miss-informativeness for this frame.
        if switches.conditional_detection:
            detection_probability = self.config.effective_detection.probability(
                observability
            )
            miss_informative = self.config.effective_detection.miss_is_informative(
                observability
            )
        else:
            detection_probability = self.existence.config.detection_probability
            miss_informative = True

        # 4. detector_detected is read from the RAW candidate list, before
        # bias correction -- whether the detector saw a Duckie cannot depend
        # on a metric correction applied downstream.
        detector_detected = len(candidates) > 0

        # 4b. Bias correction runs BEFORE association: association scores
        # candidates by innovation against h(x̂⁻), so an uncorrected
        # candidate would inject the full bias into every NIS.
        bias = self._bias_fitted if switches.bias_refit else self._bias_frozen
        corrected_candidates = tuple(
            CandidateMeasurement(
                measurement=bias.correct(candidate.measurement),
                confidence=candidate.confidence,
                bbox_key=candidate.bbox_key,
            )
            for candidate in candidates
        )

        # 5/6. Associate corrected candidates against h(x̂⁻) using the single
        # Invariant-I1 provider. Forcing predicted_measurement=None when
        # temporal_association is off reuses MeasurementAssociator's own
        # "initialization" branch -- pick highest confidence in
        # (-confidence, bbox_key) order, no NIS scoring -- which is exactly
        # `select_single_duckie`'s ordering, so switching this component off
        # degenerates association to the unconditional F9b-style selection.
        if self.ekf.initialized:
            predicted_measurement = measurement_function(
                self.ekf.state, minimum_range_m=self._minimum_range_m
            )
        else:
            predicted_measurement = None
        association_predicted_measurement = (
            predicted_measurement if switches.temporal_association else None
        )
        association = self.associator.associate(
            corrected_candidates,
            predicted_measurement=association_predicted_measurement,
            innovation_covariance_for=self._innovation_covariance,
        )

        # 7/8. Gate the associated candidate on NIS using the same provider.
        # Initialization is the deliberate exception: with no active track
        # there is no innovation to test, so a candidate that survives
        # association's own selection is accepted outright.
        gate_decision: GateDecision | None = None
        kinematic_measurement_accepted = False
        if association.selected is not None:
            if association.mode == "initialization":
                kinematic_measurement_accepted = True
            else:
                measurement = association.selected.measurement
                innovation = np.array(
                    [
                        measurement.range_m - predicted_measurement[0],
                        wrap_angle(
                            measurement.bearing_rad - predicted_measurement[1]
                        ),
                    ],
                    dtype=float,
                )
                innovation_covariance = self._innovation_covariance(
                    measurement.range_m
                )
                nis = normalized_innovation_squared(innovation, innovation_covariance)
                if switches.innovation_gate:
                    gate_decision = self.gate.evaluate(innovation, innovation_covariance)
                    kinematic_measurement_accepted = gate_decision.accepted
                else:
                    gate_decision = GateDecision(
                        accepted=True,
                        nis=nis,
                        threshold=self.config.gate.chi_square_threshold,
                    )
                    kinematic_measurement_accepted = True

        # 9. Correct if accepted; the state otherwise stays exactly as
        # advanced by step 1's predict.
        self.last_ekf_diagnostics = None
        if kinematic_measurement_accepted:
            self.last_ekf_diagnostics = self.ekf.correct(
                association.selected.measurement
            )

        # 10. Existence update -- Invariant I2. Existence is driven by
        # `detector_detected`, never by `kinematic_measurement_accepted`: a
        # gated-out bbox suppresses the EKF correction only, and is still
        # scored as a detection here.
        observation_informative = miss_informative or detector_detected
        existence_probability = self.existence.update(
            detector_detected,
            detection_probability=detection_probability,
            observation_informative=observation_informative,
        )

        # Track lifecycle. `frame_mode` labels the frame for callers/tests;
        # it never feeds back into the numerics above.
        if track_was_active:
            frame_mode = "temporal"
        elif (
            kinematic_measurement_accepted
            and existence_probability >= self.config.initialization_threshold
        ):
            frame_mode = "initialization"
        else:
            frame_mode = "no_track"
            if kinematic_measurement_accepted:
                # An accepted candidate initialized the EKF above (step 9),
                # but this frame's existence posterior does not clear the
                # initialization threshold. A track is created only when
                # both conditions hold, so discard the just-created
                # kinematic state.
                self.ekf.reset()

        # 11. Delete a track whose existence has decayed too far.
        track_deleted = False
        if self.ekf.initialized and existence_probability < self.config.delete_threshold:
            self.ekf.reset()
            track_deleted = True
            frame_mode = "deleted"

        # 12. Report belief with the posterior floor applied. The floor is
        # added ONLY to the value reported below -- it is never written back
        # into `self.ekf._covariance`. Writing it back would corrupt the next
        # `predict()`, which would then propagate the inflated (floored)
        # variance forward as if it were genuine filter uncertainty, instead
        # of the flooring being a report-time-only presentation of the
        # otherwise-unmodeled per-episode bias term.
        if self.ekf.initialized:
            moments = self.ekf.polar_moments(ego_motion)
            raw_std = moments.standard_deviation
            range_std_m, bearing_std_rad = (
                self.config.covariance.floor_polar_standard_deviation(
                    float(raw_std[0]), float(raw_std[1])
                )
            )
            pedestrian = PedestrianBelief(
                existence_probability=existence_probability,
                range_mean_m=max(0.0, float(moments.mean[0])),
                range_std_m=range_std_m,
                bearing_mean_rad=float(moments.mean[1]),
                bearing_std_rad=bearing_std_rad,
                radial_velocity_mean_mps=float(moments.mean[2]),
                radial_velocity_std_mps=float(raw_std[2]),
                bearing_rate_mean_rad_s=float(moments.mean[3]),
                bearing_rate_std_rad_s=float(raw_std[3]),
            )
        else:
            previous = previous_belief.pedestrian
            range_std_m, bearing_std_rad = previous.range_std_m, previous.bearing_std_rad
            pedestrian = PedestrianBelief(
                existence_probability=existence_probability,
                range_mean_m=previous.range_mean_m,
                range_std_m=previous.range_std_m,
                bearing_mean_rad=previous.bearing_mean_rad,
                bearing_std_rad=previous.bearing_std_rad,
                radial_velocity_mean_mps=previous.radial_velocity_mean_mps,
                radial_velocity_std_mps=previous.radial_velocity_std_mps,
                bearing_rate_mean_rad_s=previous.bearing_rate_mean_rad_s,
                bearing_rate_std_rad_s=previous.bearing_rate_std_rad_s,
            )

        belief = BeliefState(
            ego=previous_belief.ego,
            road=previous_belief.road,
            stop_sign=previous_belief.stop_sign,
            pedestrian=pedestrian,
        )
        record = RobustStepRecord(
            frame_mode=frame_mode,
            detector_detected=detector_detected,
            kinematic_measurement_accepted=kinematic_measurement_accepted,
            association=association,
            gate=gate_decision,
            effective_detection_probability=detection_probability,
            observation_informative=observation_informative,
            observability_class=observability.observability_class,
            existence_probability=existence_probability,
            track_active=self.ekf.initialized,
            track_deleted=track_deleted,
            nis=None if gate_decision is None else gate_decision.nis,
            reported_range_std_m=range_std_m,
            reported_bearing_std_rad=bearing_std_rad,
        )
        self.last_record = record
        return belief, record
