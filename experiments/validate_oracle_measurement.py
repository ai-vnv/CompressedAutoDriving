"""Validate F6 synthetic measurements from real Gym-Duckietown truth states."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from duckie_pomdp.adapters.gym_duckietown import (
    GymDuckietownConfig,
    create_gym_duckietown,
)
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.domain.privileged import PrivilegedSimulatorState
from duckie_pomdp.perception.measurement_calibration import wrap_angle
from duckie_pomdp.perception.measurement_noise import (
    PolarMeasurementNoiseModel,
    load_polar_measurement_noise,
)
from duckie_pomdp.perception.oracle_measurement import (
    OracleMode,
    OracleObservationModel,
    load_oracle_detection_config,
)
from duckie_pomdp.scenario import PedestrianMode, load_scenario


@dataclass(frozen=True)
class OracleValidationRow:
    episode: str
    frame: int
    seed: int
    mode: str
    gt_range: float
    gt_bearing: float
    measured_range: float | None
    measured_bearing: float | None
    range_error: float | None
    bearing_error: float | None
    range_bin: str
    detected: bool


def collect_real_truth_states(
    scenario_path: Path,
) -> dict[str, tuple[PrivilegedSimulatorState, ...]]:
    base = load_scenario(scenario_path).with_pedestrian_mode(PedestrianMode.STATIONARY)
    starts = {
        "far": base.ego_start_pose_m,
        "medium": (0.300, 0.0, 0.400),
        "near": (0.550, 0.0, 0.400),
    }
    result: dict[str, tuple[PrivilegedSimulatorState, ...]] = {}
    for name, start_pose in starts.items():
        scenario = replace(base, ego_start_pose_m=start_pose)
        integration = create_gym_duckietown(
            GymDuckietownConfig(
                scenario=scenario,
                camera_width=80,
                camera_height=60,
            )
        )
        states: list[PrivilegedSimulatorState] = []
        try:
            integration.agent.reset(seed=scenario.seed)
            states.append(integration.privileged.read())
            for _ in range(7):
                transition = integration.agent.step(PolicyAction(0.0, 0.0))
                if transition.terminated or transition.truncated:
                    raise RuntimeError("F6 truth-source trajectory ended unexpectedly")
                states.append(integration.privileged.read())
        finally:
            integration.close()
        result[name] = tuple(states)
    return result


def run_validation(
    *,
    scenario_path: Path,
    measurement_config_path: Path,
    oracle_config_path: Path,
    output_path: Path,
    seed: int,
    samples_per_bin: int,
    dropout_samples: int,
) -> dict[str, object]:
    truth_by_bin = collect_real_truth_states(scenario_path)
    noise = load_polar_measurement_noise(measurement_config_path)
    detection = load_oracle_detection_config(oracle_config_path)
    rows: list[OracleValidationRow] = []

    clean = OracleObservationModel(
        mode=OracleMode.CLEAN,
        measurement_noise=noise,
        detection=detection,
        seed=seed,
    )
    frame = 0
    for name, states in truth_by_bin.items():
        for privileged in states:
            rows.append(_row(name, frame, seed, clean, privileged, noise))
            frame += 1

    noisy = OracleObservationModel(
        mode=OracleMode.NOISY,
        measurement_noise=noise,
        detection=detection,
        seed=seed,
    )
    for name, states in truth_by_bin.items():
        for sample_index in range(samples_per_bin):
            privileged = states[sample_index % len(states)]
            rows.append(_row(name, frame, seed, noisy, privileged, noise))
            frame += 1

    dropout = OracleObservationModel(
        mode=OracleMode.DROPOUT,
        measurement_noise=noise,
        detection=detection,
        seed=seed + 1,
    )
    dropout_states = truth_by_bin["medium"]
    for sample_index in range(dropout_samples):
        privileged = dropout_states[sample_index % len(dropout_states)]
        rows.append(_row("dropout_medium", frame, seed + 1, dropout, privileged, noise))
        frame += 1

    _write_csv(rows, output_path)
    summary = _summarize(rows, noise, detection.miss_probability)
    _assert_monte_carlo_match(summary, noise, detection.miss_probability)
    return summary


def _row(
    episode: str,
    frame: int,
    seed: int,
    model: OracleObservationModel,
    privileged: PrivilegedSimulatorState,
    noise: PolarMeasurementNoiseModel,
) -> OracleValidationRow:
    truth = privileged.true_pomdp_state.pedestrian
    if not truth.exists or truth.range_m is None or truth.bearing_rad is None:
        raise RuntimeError("F6 real trajectory has no pedestrian truth")
    measurement = model.observe(privileged, ObjectClass.DUCKIE)
    measured_range = measurement.range_m
    measured_bearing = measurement.bearing_rad
    return OracleValidationRow(
        episode=episode,
        frame=frame,
        seed=seed,
        mode=model.mode.value,
        gt_range=truth.range_m,
        gt_bearing=truth.bearing_rad,
        measured_range=measured_range,
        measured_bearing=measured_bearing,
        range_error=(
            None if measured_range is None else measured_range - truth.range_m
        ),
        bearing_error=(
            None
            if measured_bearing is None
            else wrap_angle(measured_bearing - truth.bearing_rad)
        ),
        range_bin=noise.range_bin(truth.range_m).name,
        detected=measurement.detected,
    )


def _write_csv(rows: list[OracleValidationRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(OracleValidationRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _summarize(
    rows: list[OracleValidationRow],
    noise: PolarMeasurementNoiseModel,
    expected_miss_probability: float,
) -> dict[str, object]:
    noisy_rows = [row for row in rows if row.mode == OracleMode.NOISY.value]
    by_range: dict[str, dict[str, float | int]] = {}
    for noise_bin in noise.range_bins:
        selected = [row for row in noisy_rows if row.range_bin == noise_bin.name]
        errors = np.array([row.range_error for row in selected], dtype=float)
        by_range[noise_bin.name] = {
            "count": len(selected),
            "configured_bias_m": noise_bin.residual_bias_m,
            "empirical_bias_m": float(np.mean(errors)),
            "configured_sigma_m": noise_bin.sigma_m,
            "empirical_sigma_m": float(np.std(errors, ddof=1)),
        }
    bearing_errors = np.array(
        [row.bearing_error for row in noisy_rows],
        dtype=float,
    )
    clean_rows = [row for row in rows if row.mode == OracleMode.CLEAN.value]
    dropout_rows = [row for row in rows if row.mode == OracleMode.DROPOUT.value]
    empirical_dropout = 1.0 - np.mean([row.detected for row in dropout_rows])
    return {
        "rows": len(rows),
        "real_truth_frames": len(clean_rows),
        "range_by_bin": by_range,
        "bearing": {
            "count": len(bearing_errors),
            "configured_bias_rad": noise.bearing_bias_rad,
            "empirical_bias_rad": float(np.mean(bearing_errors)),
            "configured_sigma_rad": noise.bearing_sigma_rad,
            "empirical_sigma_rad": float(np.std(bearing_errors, ddof=1)),
            "gaussian_approximation": "provisional",
        },
        "dropout": {
            "count": len(dropout_rows),
            "configured_miss_probability": expected_miss_probability,
            "empirical_miss_probability": float(empirical_dropout),
            "performance_source": "synthetic_stress_test",
        },
        "clean_max_abs_range_error_m": max(
            abs(float(row.range_error)) for row in clean_rows
        ),
        "clean_max_abs_bearing_error_rad": max(
            abs(float(row.bearing_error)) for row in clean_rows
        ),
    }


def _assert_monte_carlo_match(
    summary: dict[str, object],
    noise: PolarMeasurementNoiseModel,
    expected_miss_probability: float,
) -> None:
    if summary["clean_max_abs_range_error_m"] > 1.0e-12:
        raise RuntimeError("oracle_clean changed GT range")
    if summary["clean_max_abs_bearing_error_rad"] > 1.0e-12:
        raise RuntimeError("oracle_clean changed GT bearing")
    range_summary = summary["range_by_bin"]
    for noise_bin in noise.range_bins:
        item = range_summary[noise_bin.name]
        if abs(item["empirical_bias_m"] - noise_bin.residual_bias_m) > 2.5e-4:
            raise RuntimeError(f"{noise_bin.name} range bias misses configured model")
        relative_sigma_error = abs(
            item["empirical_sigma_m"] / noise_bin.sigma_m - 1.0
        )
        if relative_sigma_error > 0.05:
            raise RuntimeError(f"{noise_bin.name} range sigma misses configured model")
    bearing = summary["bearing"]
    if abs(bearing["empirical_bias_rad"] - noise.bearing_bias_rad) > 4.0e-4:
        raise RuntimeError("bearing bias misses configured model")
    if abs(bearing["empirical_sigma_rad"] / noise.bearing_sigma_rad - 1.0) > 0.04:
        raise RuntimeError("bearing sigma misses configured model")
    dropout = summary["dropout"]
    if abs(dropout["empirical_miss_probability"] - expected_miss_probability) > 0.015:
        raise RuntimeError("dropout rate misses configured model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/scenario_pomdp_v1.toml"),
    )
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=Path("configs/measurement_model_v1.toml"),
    )
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=Path("configs/oracle_ekf_v1.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/oracle_measurement_validation.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--samples-per-bin", type=int, default=6000)
    parser.add_argument("--dropout-samples", type=int, default=12000)
    args = parser.parse_args()
    summary = run_validation(
        scenario_path=args.scenario,
        measurement_config_path=args.measurement_config,
        oracle_config_path=args.oracle_config,
        output_path=args.output,
        seed=args.seed,
        samples_per_bin=args.samples_per_bin,
        dropout_samples=args.dropout_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

