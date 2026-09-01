"""Offline, privileged simulator tooling for the Version-1 detection dataset."""

from duckie_pomdp.dataset.annotations import (
    AnnotationDecision,
    SilhouetteRules,
    YoloBox,
    assess_silhouette,
)
from duckie_pomdp.dataset.config import DetectionDatasetConfig, load_dataset_config
from duckie_pomdp.dataset.generation import generate_detection_dataset
from duckie_pomdp.dataset.qa import validate_detection_dataset

__all__ = [
    "AnnotationDecision",
    "DetectionDatasetConfig",
    "SilhouetteRules",
    "YoloBox",
    "assess_silhouette",
    "generate_detection_dataset",
    "load_dataset_config",
    "validate_detection_dataset",
]
