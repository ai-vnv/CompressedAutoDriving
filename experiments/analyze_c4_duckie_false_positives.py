"""Audit C4 Duckie bbox plausibility; privileged truth labels rows only after inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.belief_runtime import F10BeliefRuntimeFactory
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.domain.detection import ObjectClass
from duckie_pomdp.evaluation.f10_ppo_policy import BeliefAwareSimpleController
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector


class RecordingDetector:
    def __init__(self, detector) -> None:
        self._detector = detector
        self.last = ()

    def detect(self, rgb):
        self.last = tuple(self._detector.detect(rgb))
        return self.last


def run(config: Path, seed: int, output: Path, summary_path: Path) -> dict:
    protocol = load_ppo_curriculum_protocol(config)
    detector = RecordingDetector(
        YoloObjectDetector(
            protocol.detector_checkpoint_path,
            confidence_threshold=0.10,
            iou_threshold=0.45,
            image_size=640,
            device="cuda",
            max_detections=20,
        )
    )
    factory = F10BeliefRuntimeFactory(protocol, detector=detector)
    environment = PPOCurriculumEnvironment(
        config,
        stage="c4",
        split="training",
        seeds=(seed,),
        belief_runtime_factory=factory,
    )
    controller = BeliefAwareSimpleController(protocol)
    rows = []
    try:
        observation, _ = environment.reset(seed=seed)
        controller.reset(seed)
        for step in range(1, protocol.stage("c4").episode_horizon_steps + 1):
            action = controller.act(observation)
            observation, _, terminated, truncated, info = environment.step(action)
            # Evaluation truth is read only after detector -> belief -> policy vector.
            pedestrian_present = (
                environment._integration.privileged.read().pedestrian_world_position
                is not None
            )
            for detection in detector.last:
                if detection.object_class is not ObjectClass.DUCKIE or detection.confidence < 0.40:
                    continue
                box = detection.bounding_box
                width = box.x_max_px - box.x_min_px
                height = box.y_max_px - box.y_min_px
                rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "pedestrian_present_after_inference": pedestrian_present,
                        "confidence": detection.confidence,
                        "x1": box.x_min_px,
                        "y1": box.y_min_px,
                        "x2": box.x_max_px,
                        "y2": box.y_max_px,
                        "width_px": width,
                        "height_px": height,
                        "height_over_width": height / max(width, 1.0e-9),
                    }
                )
            if terminated or truncated:
                break
    finally:
        environment.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]) if rows else ("seed", "step"))
        writer.writeheader()
        writer.writerows(rows)
    true_rows = [row for row in rows if row["pedestrian_present_after_inference"]]
    false_rows = [row for row in rows if not row["pedestrian_present_after_inference"]]

    def stats(group):
        return {
            "count": len(group),
            "confidence_min": min((row["confidence"] for row in group), default=None),
            "height_px_min": min((row["height_px"] for row in group), default=None),
            "height_over_width_min": min((row["height_over_width"] for row in group), default=None),
            "height_over_width_max": max((row["height_over_width"] for row in group), default=None),
        }

    result = {
        "schema_version": 1,
        "config": str(config.resolve()),
        "seed": seed,
        "confidence_floor": 0.40,
        "uses_privileged_truth_for_runtime_filter": False,
        "truth_role": "post-inference audit label only",
        "visible_object_rows": stats(true_rows),
        "absent_object_false_positive_rows": stats(false_rows),
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.seed, args.output.resolve(), args.summary.resolve()), indent=2))


if __name__ == "__main__":
    main()
