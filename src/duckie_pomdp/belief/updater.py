"""BeliefUpdater implementation composed from pedestrian EKF and existence filter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from duckie_pomdp.belief.existence_filter import (
    ExistenceFilter,
    ExistenceFilterConfig,
)
from duckie_pomdp.belief.pedestrian_ekf import (
    EKFStepDiagnostics,
    MeasurementProfile,
    PedestrianEKF,
    PedestrianEKFConfig,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.belief import (
    BeliefState,
    PedestrianBelief,
    RoadBelief,
    StopSignBelief,
)
from duckie_pomdp.domain.measurement import PerceptionObservation
from duckie_pomdp.domain.observation import EgoObservation, RoadMeasurement
from duckie_pomdp.domain.state import EgoMotion, StopMode
from duckie_pomdp.perception.measurement_noise import PolarMeasurementNoiseModel


@dataclass(frozen=True)
class BeliefUpdateDiagnostics:
    previous_action: PolicyAction
    actual_ego_motion: EgoMotion
    ekf: EKFStepDiagnostics


class PedestrianBeliefUpdater:
    """Stateful implementation of the existing framework-independent port."""

    def __init__(
        self,
        *,
        ekf_config: PedestrianEKFConfig,
        existence_config: ExistenceFilterConfig,
        measurement_noise: PolarMeasurementNoiseModel,
        measurement_profile: MeasurementProfile,
    ) -> None:
        self.measurement_profile = MeasurementProfile(measurement_profile)
        self.ekf = PedestrianEKF(
            config=ekf_config,
            measurement_noise=measurement_noise,
            measurement_profile=self.measurement_profile,
        )
        self.existence = ExistenceFilter(existence_config)
        self.last_diagnostics: BeliefUpdateDiagnostics | None = None

    def reset(self) -> None:
        self.ekf.reset()
        self.existence.reset()
        self.last_diagnostics = None

    def update(
        self,
        previous_belief: BeliefState,
        previous_action: PolicyAction,
        ego_motion: EgoMotion,
        perception: PerceptionObservation,
        dt_s: float,
    ) -> BeliefState:
        """Predict with actual ego motion, then correct from agent measurement.

        ``previous_action`` is retained for the formal POMDP transition and
        diagnostics. It is never substituted for actual ego motion.
        """

        existence_probability = self.existence.update(
            perception.pedestrian.detected
        )
        diagnostics = self.ekf.step(perception.pedestrian, ego_motion, dt_s)
        if self.ekf.initialized:
            moments = self.ekf.polar_moments(ego_motion)
            standard_deviation = moments.standard_deviation
            pedestrian = PedestrianBelief(
                existence_probability=existence_probability,
                range_mean_m=max(0.0, float(moments.mean[0])),
                range_std_m=float(standard_deviation[0]),
                bearing_mean_rad=float(moments.mean[1]),
                bearing_std_rad=float(standard_deviation[1]),
                radial_velocity_mean_mps=float(moments.mean[2]),
                radial_velocity_std_mps=float(standard_deviation[2]),
                bearing_rate_mean_rad_s=float(moments.mean[3]),
                bearing_rate_std_rad_s=float(standard_deviation[3]),
            )
        else:
            previous = previous_belief.pedestrian
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

        road = _road_belief(perception.road, previous_belief.road)
        stop_sign = _stop_sign_belief(
            perception,
            previous_belief.stop_sign,
            self.ekf,
        )
        self.last_diagnostics = BeliefUpdateDiagnostics(
            previous_action=previous_action,
            actual_ego_motion=ego_motion,
            ekf=diagnostics,
        )
        return BeliefState(
            ego=perception.ego,
            road=road,
            stop_sign=stop_sign,
            pedestrian=pedestrian,
        )


def initial_belief(
    ego: EgoObservation,
    road: RoadMeasurement | None = None,
    *,
    existence_prior: float = 0.5,
) -> BeliefState:
    road_belief = (
        RoadBelief(0.0, 0.0, StopMode.NONE)
        if road is None
        else RoadBelief(
            road.curvature_inv_m,
            road.stop_line_distance_m,
            StopMode.NONE,
        )
    )
    return BeliefState(
        ego=ego,
        road=road_belief,
        stop_sign=StopSignBelief(0.5, 0.0, 2.0, 0.0, 3.141592653589793),
        pedestrian=PedestrianBelief(
            existence_probability=existence_prior,
            range_mean_m=0.0,
            range_std_m=2.0,
            bearing_mean_rad=0.0,
            bearing_std_rad=3.141592653589793,
            radial_velocity_mean_mps=0.0,
            radial_velocity_std_mps=1.0,
            bearing_rate_mean_rad_s=0.0,
            bearing_rate_std_rad_s=3.141592653589793,
        ),
    )


def load_existence_filter_config(path: str | Path) -> ExistenceFilterConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)["existence"]
    return ExistenceFilterConfig(
        prior_probability=float(data["prior_probability"]),
        detection_probability=float(data["detection_probability"]),
        false_positive_probability=float(data["false_positive_probability"]),
        survival_probability=float(data["survival_probability"]),
        birth_probability=float(data["birth_probability"]),
    )


def _road_belief(
    measurement: RoadMeasurement | None,
    previous: RoadBelief,
) -> RoadBelief:
    if measurement is None:
        return previous
    return RoadBelief(
        curvature_inv_m=measurement.curvature_inv_m,
        stop_line_distance_m=measurement.stop_line_distance_m,
        stop_mode=previous.stop_mode,
    )


def _stop_sign_belief(
    perception: PerceptionObservation,
    previous: StopSignBelief,
    ekf: PedestrianEKF,
) -> StopSignBelief:
    measurement = perception.stop_sign
    if not measurement.detected:
        return previous
    if measurement.range_m is None or measurement.bearing_rad is None:
        raise RuntimeError("detected stop sign has incomplete polar geometry")
    covariance = ekf.measurement_covariance(measurement.range_m)
    return StopSignBelief(
        existence_probability=1.0,
        range_mean_m=measurement.range_m,
        range_std_m=float(covariance[0, 0] ** 0.5),
        bearing_mean_rad=measurement.bearing_rad,
        bearing_std_rad=float(covariance[1, 1] ** 0.5),
    )

