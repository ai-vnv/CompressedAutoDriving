"""Camera-only learned lane-pose measurement for tight Duckietown curves."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from duckie_pomdp.domain.measurement import LaneMeasurement

from .lane_measurement import LaneBoundaryDiagnostics


OUTPUT_ORDER = (
    "lateral_error_m",
    "heading_error_rad",
    "curvature_inv_m",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LanePoseMobileNet(nn.Module):
    """MobileNetV3-small regressor with three normalized lane outputs."""

    architecture = "mobilenet_v3_small_lane_pose_v1"

    def __init__(self) -> None:
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        input_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(input_features, len(OUTPUT_ORDER))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


@dataclass(frozen=True)
class LaneRGBModelConfig:
    checkpoint: Path
    checkpoint_sha256: str
    device: str
    image_size_px: int
    target_scales: tuple[float, float, float]
    residual_sigmas: tuple[float, float, float]
    crop_top_fraction: float = 0.0

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("lane RGB checkpoint SHA256 must contain 64 hex digits")
        if self.image_size_px <= 0:
            raise ValueError("lane RGB image size must be positive")
        if not all(isfinite(value) and value > 0.0 for value in self.target_scales):
            raise ValueError("lane RGB target scales must be finite and positive")
        if not all(isfinite(value) and value > 0.0 for value in self.residual_sigmas):
            raise ValueError("lane RGB residual sigmas must be finite and positive")
        if not isfinite(self.crop_top_fraction) or not 0.0 <= self.crop_top_fraction < 0.8:
            raise ValueError("lane RGB top crop must be within [0, 0.8)")


class LaneRGBMeasurementEstimator:
    """Infer d/phi/kappa from front RGB without a privileged input channel."""

    def __init__(self, config: LaneRGBModelConfig) -> None:
        actual = hashlib.sha256(config.checkpoint.read_bytes()).hexdigest()
        if actual != config.checkpoint_sha256:
            raise RuntimeError(
                f"lane RGB checkpoint hash mismatch: expected {config.checkpoint_sha256}, "
                f"got {actual}"
            )
        self.config = config
        self.device = _resolve_device(config.device)
        payload = torch.load(config.checkpoint, map_location=self.device, weights_only=False)
        if payload.get("architecture") != LanePoseMobileNet.architecture:
            raise RuntimeError("unsupported lane RGB checkpoint architecture")
        if tuple(payload.get("output_order", ())) != OUTPUT_ORDER:
            raise RuntimeError("lane RGB checkpoint output order mismatch")
        payload_scales = tuple(float(value) for value in payload.get("target_scales", ()))
        if payload_scales != config.target_scales:
            raise RuntimeError("lane RGB checkpoint target scales mismatch")
        payload_preprocessing = dict(payload.get("preprocessing", {}))
        payload_crop = float(payload_preprocessing.get("crop_top_fraction", 0.0))
        if payload_crop != config.crop_top_fraction:
            raise RuntimeError("lane RGB checkpoint top-crop mismatch")
        self.model = LanePoseMobileNet().to(self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._scales = torch.tensor(config.target_scales, device=self.device).view(1, 3)

    def estimate(self, front_rgb: NDArray[np.uint8]) -> LaneMeasurement:
        return self.estimate_with_diagnostics(front_rgb)[0]

    def estimate_with_diagnostics(
        self, front_rgb: NDArray[np.uint8]
    ) -> tuple[LaneMeasurement, LaneBoundaryDiagnostics]:
        image = np.asarray(front_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("lane RGB estimator expects HxWx3 uint8 RGB")
        tensor = torch.from_numpy(
            np.require(image, dtype=np.uint8, requirements=("C", "W"))
        ).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        crop_rows = int(round(tensor.shape[-2] * self.config.crop_top_fraction))
        if crop_rows:
            tensor = tensor[:, :, crop_rows:, :]
        tensor = F.interpolate(
            tensor,
            size=(self.config.image_size_px, self.config.image_size_px),
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self._mean) / self._std
        with torch.inference_mode():
            values = (self.model(tensor) * self._scales).squeeze(0).cpu().numpy()
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return LaneMeasurement.missing(), _diagnostics("mobilenet_nonfinite")
        lateral, heading, curvature = (float(value) for value in values)
        sigma_d, sigma_phi, sigma_kappa = self.config.residual_sigmas
        return (
            LaneMeasurement(
                detected=True,
                lateral_error_m=lateral,
                lateral_error_std_m=sigma_d,
                heading_error_rad=float(np.arctan2(np.sin(heading), np.cos(heading))),
                heading_error_std_rad=sigma_phi,
                curvature_inv_m=curvature,
                curvature_std_inv_m=sigma_kappa,
                visible_point_count=1,
                fit_residual_m=0.0,
            ),
            _diagnostics("mobilenet_v3_small_rgb"),
        )


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("lane RGB model requires CUDA but it is unavailable")
    return device


def _diagnostics(source: str) -> LaneBoundaryDiagnostics:
    return LaneBoundaryDiagnostics(
        source=source,
        strict_yellow_pixel_count=0,
        strict_white_pixel_count=0,
        adaptive_unknown_pixel_count=0,
        yellow_center_point_count=0,
        white_center_point_count=0,
        boundary_disagreement_m=None,
    )
