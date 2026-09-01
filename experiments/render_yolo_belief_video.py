"""Render real Gym-Duckietown proof of RGB -> YOLO11n -> F9c belief.

The runtime boundary is explicit: privileged truth is read only after YOLO,
metric projection, association, and EKF/existence update have completed. It
is used solely for the magenta ``GT EVAL ONLY`` video overlay and manifest
error statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from duckie_pomdp.adapters.gym_duckietown import GymDuckietownConfig, create_gym_duckietown
from duckie_pomdp.belief.measurement_association import CandidateMeasurement
from duckie_pomdp.belief.observability import PredictedObservabilityModel
from duckie_pomdp.belief.pedestrian_ekf import load_pedestrian_ekf_config
from duckie_pomdp.belief.robust_updater import RobustPedestrianBeliefUpdater
from duckie_pomdp.belief.updater import initial_belief, load_existence_filter_config
from duckie_pomdp.domain.action import PolicyAction
from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.f9c_calibration import (
    load_frozen_bias_correction,
    load_miss_likelihood_floor,
    load_robust_observation_config,
)
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
from duckie_pomdp.perception.camera_geometry import CalibratedGroundProjector
from duckie_pomdp.perception.f9_pipeline import YoloPedestrianMeasurementPipeline
from duckie_pomdp.perception.measurement_calibration import LinearRangeCalibration, MeasurementCalibrator
from duckie_pomdp.perception.measurement_noise import load_polar_measurement_noise
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.perception.yolo_measurement import YoloMeasurementProjector
from duckie_pomdp.scenario import load_scenario
from duckie_pomdp.visualization.belief_video import (
    BeliefVideoOverlay,
    DetectionOverlay,
    EvaluationTruthOverlay,
    render_belief_overlay,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "yolo_belief_demo.mp4"
DEFAULT_MANIFEST = ROOT / "artifacts" / "yolo_belief_demo.json"
DEFAULT_PREVIEW = ROOT / "artifacts" / "yolo_belief_demo_preview.png"


def _bbox_key(box) -> tuple[int, int, int, int]:
    return tuple(
        int(round(value))
        for value in (box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px)
    )


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _detection_overlays(result, record) -> tuple[DetectionOverlay, ...]:
    selected_key = (
        None if record.association.selected is None else record.association.selected.bbox_key
    )
    overlays: list[DetectionOverlay] = []
    for candidate in result.duckie_candidates:
        box = candidate.detection.bounding_box
        associated = _bbox_key(box) == selected_key
        overlays.append(
            DetectionOverlay(
                object_class="duckie",
                confidence=candidate.detection.confidence,
                bbox_xyxy=(box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
                associated=associated,
                accepted=associated and record.kinematic_measurement_accepted,
            )
        )
    for detection in result.stop_sign_detections:
        box = detection.bounding_box
        overlays.append(
            DetectionOverlay(
                object_class="stop_sign",
                confidence=detection.confidence,
                bbox_xyxy=(box.x_min_px, box.y_min_px, box.x_max_px, box.y_max_px),
            )
        )
    return tuple(overlays)


def _candidates(result) -> tuple[CandidateMeasurement, ...]:
    return tuple(
        CandidateMeasurement(
            measurement=candidate.measurement,
            confidence=candidate.detection.confidence,
            bbox_key=_bbox_key(candidate.detection.bounding_box),
        )
        for candidate in result.duckie_candidates
        if candidate.projection_error is None
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_demo(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config.resolve()
    protocol = load_f9c_protocol(config_path, require_frozen=True)
    if sha256(protocol.checkpoint_path) != protocol.checkpoint_sha256:
        raise RuntimeError("frozen YOLO checkpoint hash mismatch")

    reserved_seeds = set(protocol.calibration_seeds + protocol.final_evaluation_seeds + protocol.forbidden_seeds)
    if args.seed in reserved_seeds:
        raise ValueError("demo seed must not reuse any calibration/evaluation seed")

    raw_config = _read_config(config_path)
    detector_config = raw_config["detector"]
    simulator_config = raw_config["simulator"]
    spec = next((item for item in protocol.scenarios if item.name == args.scenario), None)
    if spec is None:
        raise ValueError(f"unknown F9c scenario: {args.scenario}")

    scenario_path = (config_path.parent / raw_config["provenance"]["scenario"]).resolve()
    base = load_scenario(scenario_path).with_pedestrian_mode(spec.pedestrian_mode)
    pose = base.ego_start_pose_m
    scenario = replace(
        base,
        seed=args.seed,
        ego_start_pose_m=(pose[0] + spec.ego_start_x_offset_m, pose[1], pose[2]),
        ego_heading_rad=base.ego_heading_rad + spec.ego_heading_offset_rad,
    )

    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    preview_path = args.preview.resolve()
    for destination in (output, manifest_path, preview_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    detector = YoloObjectDetector(
        protocol.checkpoint_path,
        confidence_threshold=float(detector_config["confidence_threshold"]),
        iou_threshold=float(detector_config["nms_iou_threshold"]),
        image_size=int(detector_config["image_size"]),
        device=detector_config["device"],
        max_detections=int(detector_config["max_detections"]),
    )
    ekf_config = load_pedestrian_ekf_config(config_path)
    measurement_noise = load_polar_measurement_noise(config_path)
    existence_config = replace(
        load_existence_filter_config(config_path),
        miss_likelihood_floor=load_miss_likelihood_floor(config_path),
    )

    integration = create_gym_duckietown(
        GymDuckietownConfig(
            scenario=scenario,
            domain_randomization=False,
            dynamics_randomization=False,
            maximum_steps=args.steps + 2,
            camera_width=int(simulator_config["image_width_px"]),
            camera_height=int(simulator_config["image_height_px"]),
        )
    )

    writer: cv2.VideoWriter | None = None
    encoded_frames = 0
    detected_frames = 0
    active_track_frames = 0
    accepted_updates = 0
    rejected_updates = 0
    duplicate_frames = 0
    range_errors: list[float] = []
    bearing_errors: list[float] = []
    preview_rgb: np.ndarray | None = None
    termination = "completed"

    try:
        observation = integration.agent.reset(seed=args.seed)
        camera_projector = CalibratedGroundProjector(integration.camera_calibration.read())
        runtime = YoloPedestrianMeasurementPipeline(
            detector,
            YoloMeasurementProjector(
                camera_projector,
                MeasurementCalibrator(LinearRangeCalibration(1.0, 0.0)),
            ),
        )
        updater = RobustPedestrianBeliefUpdater(
            ekf_config=ekf_config,
            measurement_noise=measurement_noise,
            existence_config=existence_config,
            bias_frozen=load_frozen_bias_correction(
                config_path, section="baseline_measurement_model"
            ),
            bias_fitted=load_frozen_bias_correction(
                config_path, section="measurement_model"
            ),
            observability_model=PredictedObservabilityModel(
                camera_projector,
                image_width_px=int(simulator_config["image_width_px"]),
            ),
            config=load_robust_observation_config(config_path),
        )
        belief = initial_belief(
            observation.ego,
            existence_prior=existence_config.prior_probability,
        )
        timestamp_s = 0.0
        dt_s = 1.0 / 30.0
        previous_action = PolicyAction(0.0, 0.0)

        for frame_index in range(args.steps + 1):
            # Runtime chain: RGB -> YOLO -> projection -> frozen robust EKF.
            result = runtime.observe(observation.front_rgb)
            belief, record = updater.update(
                previous_belief=belief,
                previous_action=previous_action,
                ego_motion=observation.ego.motion,
                candidates=_candidates(result),
                dt_s=dt_s,
            )

            # Evaluation-only boundary. These values never feed the runtime chain.
            truth_state = integration.privileged.read().true_pomdp_state.pedestrian
            truth = (
                None
                if not truth_state.exists
                or truth_state.range_m is None
                or truth_state.bearing_rad is None
                else EvaluationTruthOverlay(
                    range_m=truth_state.range_m,
                    bearing_rad=truth_state.bearing_rad,
                )
            )

            associated = record.association.selected
            measurement = None if associated is None else associated.measurement
            pedestrian = belief.pedestrian
            overlay = BeliefVideoOverlay(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                detections=_detection_overlays(result, record),
                duckie_detection_count=result.duckie_detection_count,
                measurement_range_m=None if measurement is None else measurement.range_m,
                measurement_bearing_rad=None if measurement is None else measurement.bearing_rad,
                belief_range_m=pedestrian.range_mean_m,
                belief_range_std_m=pedestrian.range_std_m,
                belief_bearing_rad=pedestrian.bearing_mean_rad,
                belief_bearing_std_rad=pedestrian.bearing_std_rad,
                radial_velocity_mps=pedestrian.radial_velocity_mean_mps,
                bearing_rate_rad_s=pedestrian.bearing_rate_mean_rad_s,
                existence_probability=pedestrian.existence_probability,
                track_active=updater.ekf.initialized,
                frame_mode=record.frame_mode,
                observability_class=record.observability_class.value,
                measurement_accepted=record.kinematic_measurement_accepted,
                nis=record.nis,
                truth=truth,
            )

            if result.duckie_detection_count:
                detected_frames += 1
            if updater.ekf.initialized:
                active_track_frames += 1
            if record.kinematic_measurement_accepted:
                accepted_updates += 1
            elif result.duckie_detection_count:
                rejected_updates += 1
            if result.duplicate_selection:
                duplicate_frames += 1
            if truth is not None and updater.ekf.initialized:
                range_errors.append(pedestrian.range_mean_m - truth.range_m)
                bearing_errors.append(pedestrian.bearing_mean_rad - truth.bearing_rad)

            if frame_index % args.capture_every == 0:
                rendered = render_belief_overlay(observation.front_rgb, overlay)
                if writer is None:
                    height, width, _ = rendered.shape
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open the MP4 writer")
                writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                encoded_frames += 1
                if preview_rgb is None and record.kinematic_measurement_accepted:
                    preview_rgb = rendered.copy()

            if frame_index == args.steps:
                break
            transition = integration.agent.step(spec.action)
            diagnostics = integration.diagnostics.read()
            if transition.terminated or transition.truncated:
                termination = f"{diagnostics.done_code}@frame_{frame_index}"
                break
            observation = transition.observation
            dt_s = diagnostics.timestamp_s - timestamp_s
            timestamp_s = diagnostics.timestamp_s
            previous_action = spec.action
    finally:
        if writer is not None:
            writer.release()
        integration.close()

    if encoded_frames == 0 or not output.is_file():
        raise RuntimeError("video render produced no encoded frames")
    if preview_rgb is None:
        raise RuntimeError("no accepted YOLO-to-EKF update was available for preview")
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

    capture = cv2.VideoCapture(str(output))
    try:
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if video_frames != encoded_frames or video_width <= 0 or video_height <= 0:
        raise RuntimeError("encoded MP4 failed post-write validation")

    manifest: dict[str, object] = {
        "artifact": str(output),
        "artifact_sha256": _sha256(output),
        "checkpoint": str(protocol.checkpoint_path),
        "checkpoint_sha256": protocol.checkpoint_sha256,
        "f9c_config": str(config_path),
        "f9c_config_sha256": protocol.config_sha256,
        "seed": args.seed,
        "seed_role": "demo_only_disjoint_from_calibration_and_evaluation",
        "scenario": spec.name,
        "simulator_steps_requested": args.steps,
        "capture_every": args.capture_every,
        "encoded_frames": encoded_frames,
        "video_fps": video_fps,
        "duration_s": encoded_frames / video_fps,
        "resolution": [video_width, video_height],
        "termination": termination,
        "detected_frames": detected_frames,
        "active_track_frames": active_track_frames,
        "accepted_updates": accepted_updates,
        "rejected_detection_updates": rejected_updates,
        "duplicate_detection_frames": duplicate_frames,
        "belief_range_rmse_m": (
            None if not range_errors else float(np.sqrt(np.mean(np.square(range_errors))))
        ),
        "belief_bearing_rmse_rad": (
            None if not bearing_errors else float(np.sqrt(np.mean(np.square(bearing_errors))))
        ),
        "runtime_path": "front_rgb -> YOLO11n -> metric projection -> F9c robust EKF -> belief",
        "privileged_boundary": "truth read only after runtime belief update; overlay/evaluation only",
        "preview": str(preview_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--scenario", default="cross_near_left_to_right")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.capture_every <= 0 or args.fps <= 0.0:
        parser.error("steps, capture-every, and fps must be positive")
    return args


def main() -> None:
    print(json.dumps(render_demo(parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
