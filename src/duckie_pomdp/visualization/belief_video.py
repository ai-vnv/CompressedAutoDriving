"""Pure RGB renderer for YOLO-to-belief evidence videos.

This module is deliberately presentation-only. It receives immutable values
that the runtime pipeline has already produced and cannot call a detector,
projector, simulator, or belief updater itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


_DUCKIE = (67, 211, 130)
_STOP_SIGN = (255, 184, 77)
_REJECTED = (255, 99, 71)
_MEASUREMENT = (255, 190, 70)
_BELIEF = (60, 210, 255)
_TRUTH = (255, 90, 210)
_MUTED = (150, 160, 175)
_WHITE = (240, 244, 250)


@dataclass(frozen=True)
class DetectionOverlay:
    object_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    associated: bool = False
    accepted: bool = False

    def __post_init__(self) -> None:
        if self.object_class not in {"duckie", "stop_sign"}:
            raise ValueError("unsupported overlay class")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(isfinite(value) for value in self.bbox_xyxy) or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be finite with positive area")


@dataclass(frozen=True)
class EvaluationTruthOverlay:
    """Privileged values allowed only after the runtime update, for display."""

    range_m: float
    bearing_rad: float


@dataclass(frozen=True)
class BeliefVideoOverlay:
    frame_index: int
    timestamp_s: float
    detections: tuple[DetectionOverlay, ...]
    duckie_detection_count: int
    measurement_range_m: float | None
    measurement_bearing_rad: float | None
    belief_range_m: float
    belief_range_std_m: float
    belief_bearing_rad: float
    belief_bearing_std_rad: float
    radial_velocity_mps: float
    bearing_rate_rad_s: float
    existence_probability: float
    track_active: bool
    frame_mode: str
    observability_class: str
    measurement_accepted: bool
    nis: float | None
    truth: EvaluationTruthOverlay | None = None


def render_belief_overlay(
    front_rgb: NDArray[np.uint8],
    overlay: BeliefVideoOverlay,
    *,
    panel_width_px: int = 360,
    max_plot_range_m: float = 2.0,
) -> NDArray[np.uint8]:
    """Render one camera frame plus an auditable metric-belief side panel."""

    rgb = np.asarray(front_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("front_rgb must be an HxWx3 uint8 RGB image")
    if panel_width_px < 280:
        raise ValueError("panel_width_px must be at least 280")
    if not isfinite(max_plot_range_m) or max_plot_range_m <= 0.0:
        raise ValueError("max_plot_range_m must be positive and finite")

    height, width, _ = rgb.shape
    canvas = Image.new("RGB", (width + panel_width_px, height), (15, 19, 28))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)

    _draw_camera_detections(draw, overlay)
    _draw_header(draw, width, panel_width_px, overlay)
    _draw_metrics(draw, width, overlay)
    _draw_top_down(draw, width, panel_width_px, height, max_plot_range_m, overlay)
    return np.asarray(canvas, dtype=np.uint8)


def _draw_camera_detections(draw: ImageDraw.ImageDraw, overlay: BeliefVideoOverlay) -> None:
    for detection in overlay.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        if detection.object_class == "stop_sign":
            color = _STOP_SIGN
        elif detection.associated and not detection.accepted:
            color = _REJECTED
        elif detection.associated:
            color = _DUCKIE
        else:
            color = _MUTED
        width = 4 if detection.associated else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        bottom_u = 0.5 * (x1 + x2)
        draw.ellipse(
            (bottom_u - 4, y2 - 4, bottom_u + 4, y2 + 4),
            fill=color,
            outline=(0, 0, 0),
        )
        suffix = " assoc" if detection.associated else ""
        draw.rectangle((x1, max(0, y1 - 17), x1 + 146, y1), fill=(0, 0, 0))
        draw.text(
            (x1 + 3, max(0, y1 - 15)),
            f"{detection.object_class} {detection.confidence:.2f}{suffix}",
            fill=color,
        )


def _draw_header(
    draw: ImageDraw.ImageDraw,
    camera_width: int,
    panel_width: int,
    overlay: BeliefVideoOverlay,
) -> None:
    left = camera_width
    draw.rectangle((left, 0, left + panel_width, 55), fill=(22, 29, 42))
    draw.text((left + 12, 8), "YOLO11n -> metric projection -> F9c EKF", fill=_WHITE)
    draw.text(
        (left + 12, 29),
        f"frame {overlay.frame_index:04d}   sim t={overlay.timestamp_s:5.2f}s",
        fill=(185, 198, 218),
    )


def _draw_metrics(
    draw: ImageDraw.ImageDraw,
    camera_width: int,
    overlay: BeliefVideoOverlay,
) -> None:
    left = camera_width + 12
    y = 66
    status_color = _DUCKIE if overlay.track_active else _REJECTED
    detector_text = f"YOLO Duckie: {overlay.duckie_detection_count} detection(s)"
    draw.text((left, y), detector_text, fill=_WHITE)
    y += 20
    draw.text(
        (left, y),
        f"Track: {'ACTIVE' if overlay.track_active else 'INACTIVE'}  "
        f"P(exists)={overlay.existence_probability:.3f}",
        fill=status_color,
    )
    y += 20
    gate = "accepted" if overlay.measurement_accepted else "prediction only"
    nis = "N/A" if overlay.nis is None else f"{overlay.nis:.2f}"
    draw.text((left, y), f"Update: {gate}   NIS={nis}", fill=_WHITE)
    y += 20
    draw.text(
        (left, y),
        f"Mode: {overlay.frame_mode} / {overlay.observability_class}",
        fill=(185, 198, 218),
    )
    y += 25

    if overlay.measurement_range_m is None:
        draw.text((left, y), "Associated measurement: NONE", fill=_MEASUREMENT)
    else:
        draw.text(
            (left, y),
            f"YOLO z: r={overlay.measurement_range_m:.3f} m  "
            f"beta={overlay.measurement_bearing_rad:+.3f} rad",
            fill=_MEASUREMENT,
        )
    y += 22
    draw.text(
        (left, y),
        f"Belief: r={overlay.belief_range_m:.3f} +/- {overlay.belief_range_std_m:.3f} m",
        fill=_BELIEF,
    )
    y += 20
    draw.text(
        (left, y),
        f"        beta={overlay.belief_bearing_rad:+.3f} +/- "
        f"{overlay.belief_bearing_std_rad:.3f} rad",
        fill=_BELIEF,
    )
    y += 20
    draw.text(
        (left, y),
        f"Rates: rdot={overlay.radial_velocity_mps:+.3f} m/s  "
        f"betadot={overlay.bearing_rate_rad_s:+.3f}",
        fill=_BELIEF,
    )
    y += 22
    if overlay.truth is not None:
        range_error = overlay.belief_range_m - overlay.truth.range_m
        bearing_error = overlay.belief_bearing_rad - overlay.truth.bearing_rad
        draw.text(
            (left, y),
            f"GT EVAL ONLY: r={overlay.truth.range_m:.3f}  "
            f"beta={overlay.truth.bearing_rad:+.3f}",
            fill=_TRUTH,
        )
        y += 20
        draw.text(
            (left, y),
            f"Belief error: dr={range_error:+.3f} m  dbeta={bearing_error:+.3f}",
            fill=_TRUTH,
        )


def _draw_top_down(
    draw: ImageDraw.ImageDraw,
    camera_width: int,
    panel_width: int,
    height: int,
    max_range_m: float,
    overlay: BeliefVideoOverlay,
) -> None:
    plot_top = max(300, height - 170)
    plot_bottom = height - 15
    center_x = camera_width + panel_width // 2
    scale = (plot_bottom - plot_top - 10) / max_range_m

    draw.line((camera_width + 15, plot_bottom, camera_width + panel_width - 15, plot_bottom), fill=(70, 80, 98))
    draw.line((center_x, plot_bottom, center_x, plot_top), fill=(70, 80, 98))
    draw.text((center_x - 14, plot_bottom - 13), "EGO", fill=_WHITE)
    draw.text((camera_width + 18, plot_top), "+LEFT", fill=(135, 148, 170))
    draw.text((camera_width + panel_width - 54, plot_top), "RIGHT", fill=(135, 148, 170))
    draw.text((center_x + 5, plot_top), "FORWARD", fill=(135, 148, 170))

    _draw_metric_point(
        draw,
        center_x,
        plot_bottom,
        scale,
        overlay.measurement_range_m,
        overlay.measurement_bearing_rad,
        _MEASUREMENT,
        "z",
    )
    _draw_metric_point(
        draw,
        center_x,
        plot_bottom,
        scale,
        overlay.belief_range_m if overlay.track_active else None,
        overlay.belief_bearing_rad if overlay.track_active else None,
        _BELIEF,
        "belief",
        radius=max(4, min(18, int(overlay.belief_range_std_m * scale * 2.0))),
    )
    if overlay.truth is not None:
        _draw_metric_point(
            draw,
            center_x,
            plot_bottom,
            scale,
            overlay.truth.range_m,
            overlay.truth.bearing_rad,
            _TRUTH,
            "GT",
        )


def _draw_metric_point(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    origin_y: int,
    scale: float,
    range_m: float | None,
    bearing_rad: float | None,
    color: tuple[int, int, int],
    label: str,
    *,
    radius: int = 4,
) -> None:
    if range_m is None or bearing_rad is None:
        return
    x_left = range_m * sin(bearing_rad)
    y_forward = range_m * cos(bearing_rad)
    x_px = center_x - x_left * scale
    y_px = origin_y - y_forward * scale
    draw.line((center_x, origin_y, x_px, y_px), fill=color, width=1)
    draw.ellipse((x_px - radius, y_px - radius, x_px + radius, y_px + radius), outline=color, width=2)
    draw.text((x_px + radius + 2, y_px - 7), label, fill=color)
