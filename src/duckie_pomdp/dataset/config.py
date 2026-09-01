"""Configuration contract for the auditable simulator detection dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.dataset.annotations import SilhouetteRules
from duckie_pomdp.scenario import PedestrianMode


@dataclass(frozen=True)
class DetectionDatasetConfig:
    config_path: Path
    version: str
    output_root: Path
    scenario_path: Path
    artifact_manifest_path: Path
    artifact_stats_path: Path
    qa_output_root: Path
    image_width_px: int
    image_height_px: int
    domain_randomization: bool
    capture_every_steps: int
    maximum_steps: int
    minimum_capture_translation_m: float
    stop_sign_rules: SilhouetteRules
    duckie_rules: SilhouetteRules
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    pedestrian_modes: tuple[PedestrianMode, ...]
    start_x_range_m: tuple[float, float]
    lateral_offset_range_m: tuple[float, float]
    heading_range_rad: tuple[float, float]
    velocity_range_mps: tuple[float, float]
    visual_qa_samples: int

    def __post_init__(self) -> None:
        if self.version != "duckietown_detection_v1":
            raise ValueError("dataset version must be duckietown_detection_v1")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if self.capture_every_steps <= 0 or self.maximum_steps <= 0:
            raise ValueError("capture cadence and maximum steps must be positive")
        if self.minimum_capture_translation_m < 0.0:
            raise ValueError("minimum capture translation cannot be negative")
        if self.visual_qa_samples <= 0:
            raise ValueError("visual_qa_samples must be positive")
        seed_groups = (self.train_seeds, self.validation_seeds, self.test_seeds)
        if any(not group for group in seed_groups):
            raise ValueError("train, validation, and test seeds must be nonempty")
        if len(set().union(*map(set, seed_groups))) != sum(map(len, seed_groups)):
            raise ValueError("dataset seeds may not cross split boundaries")
        if set(self.pedestrian_modes) != set(PedestrianMode):
            raise ValueError("all three Version-1 pedestrian modes are required")
        for bounds in (
            self.start_x_range_m,
            self.lateral_offset_range_m,
            self.heading_range_rad,
            self.velocity_range_mps,
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("trajectory ranges must contain ordered bounds")

    @property
    def split_seeds(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": self.train_seeds,
            "val": self.validation_seeds,
            "test": self.test_seeds,
        }

    def annotation_rules(self, object_class: str) -> SilhouetteRules:
        if object_class == "stop_sign":
            return self.stop_sign_rules
        if object_class == "duckie":
            return self.duckie_rules
        raise ValueError(f"unsupported annotation class: {object_class}")


def load_dataset_config(path: str | Path) -> DetectionDatasetConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)
    dataset = data["dataset"]
    annotation = data["annotation"]
    split = data["split"]
    trajectory = data["trajectory"]
    qa = data["qa"]

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    return DetectionDatasetConfig(
        config_path=config_path,
        version=str(dataset["version"]),
        output_root=relative(str(dataset["output_root"])),
        scenario_path=relative(str(dataset["scenario"])),
        artifact_manifest_path=relative(str(dataset["artifact_manifest"])),
        artifact_stats_path=relative(str(dataset["artifact_stats"])),
        qa_output_root=relative(str(dataset["qa_output_root"])),
        image_width_px=int(dataset["image_width_px"]),
        image_height_px=int(dataset["image_height_px"]),
        domain_randomization=bool(dataset["domain_randomization"]),
        capture_every_steps=int(dataset["capture_every_steps"]),
        maximum_steps=int(dataset["maximum_steps"]),
        minimum_capture_translation_m=float(
            dataset["minimum_capture_translation_m"]
        ),
        stop_sign_rules=_annotation_rules(annotation, "stop_sign"),
        duckie_rules=_annotation_rules(annotation, "duckie"),
        train_seeds=tuple(int(value) for value in split["train_seeds"]),
        validation_seeds=tuple(int(value) for value in split["validation_seeds"]),
        test_seeds=tuple(int(value) for value in split["test_seeds"]),
        pedestrian_modes=tuple(
            PedestrianMode(str(value)) for value in trajectory["pedestrian_modes"]
        ),
        start_x_range_m=tuple(float(value) for value in trajectory["start_x_range_m"]),
        lateral_offset_range_m=tuple(
            float(value) for value in trajectory["lateral_offset_range_m"]
        ),
        heading_range_rad=tuple(
            float(value) for value in trajectory["heading_range_rad"]
        ),
        velocity_range_mps=tuple(
            float(value) for value in trajectory["velocity_range_mps"]
        ),
        visual_qa_samples=int(qa["visual_samples"]),
    )


def _annotation_rules(annotation: dict[str, Any], class_name: str) -> SilhouetteRules:
    class_rules = annotation[class_name]
    return SilhouetteRules(
        minimum_visible_pixels=int(annotation["minimum_visible_pixels"]),
        minimum_width_px=float(annotation["minimum_bbox_width_px"]),
        minimum_height_px=float(annotation["minimum_bbox_height_px"]),
        maximum_border_touches=int(class_rules["maximum_border_touches"]),
        minimum_truncated_height_px=float(
            class_rules["minimum_truncated_height_px"]
        ),
    )
