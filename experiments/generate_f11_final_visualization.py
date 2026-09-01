"""Generate the final, frozen F11 Belief-PPO explanation package.

Quantitative plots read the completed R004 artifacts without recomputing IG.
The qualitative episode is the previously registered explanation-only seed
176001.  Its already-stored public trace and evaluation-only pose trace are
joined after inference; a deterministic replay is used only to recover the
five selected RGB frames.  R004 locked seeds are never reset or rerendered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 environment
    import tomli as tomllib

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.explain.development_protocol import PhaseThresholds, public_phase
from duckie_pomdp.explain.final_visualization import (
    GROUP_COLOURS,
    GROUP_ORDER,
    PHASE_LABELS,
    PHASE_ORDER,
    TARGET_ORDER,
    pedestrian_belief_world,
    phase_runs,
    select_representative_frames,
    summary_matrix,
    validate_group_summary_rows,
)
from duckie_pomdp.perception.yolo_detector import YoloObjectDetector
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]
TARGET_LABELS = {
    "v_cmd_mps": r"$v_{cmd}$",
    "omega_cmd_rad_s": r"$\omega_{cmd}$",
}
TARGET_UNITS = {"v_cmd_mps": "m/s", "omega_cmd_rad_s": "rad/s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_final_visualization_v1.toml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-rgb",
        action="store_true",
        help="reuse already-generated selected RGB PNGs after validating hashes",
    )
    return parser.parse_args()


def main() -> None:
    result = generate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


def generate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("method_quantitative") != (
        "phase-conditioned distributional multi-reference Integrated Gradients"
    ):
        raise ValueError("unexpected quantitative explanation method")
    if bool(config["r004"].get("rerender_locked_seeds", True)):
        raise ValueError("final visualization must never rerender R004 locked seeds")
    if bool(config["qualitative_source"].get("pose_enters_policy_or_attribution", True)):
        raise ValueError("evaluation-only pose cannot enter policy or attribution")

    paths = _resolve_and_verify_sources(config_path, config)
    output = _resolve(config_path, config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    destinations = _destinations(output)
    source_frames = output / "source_frames"
    source_frames.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [str(path) for path in destinations.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite final visualization: {existing}")

    rows = _read_group_summary(paths["group_summary"])
    validate_group_summary_rows(rows)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    _validate_r004(metrics, config)

    protocol = load_ppo_curriculum_protocol(paths["policy_config"])
    qualitative = dict(np.load(paths["qualitative_trace"], allow_pickle=False))
    pose_all = dict(np.load(paths["qualitative_pose"], allow_pickle=False))
    seed = int(config["qualitative_source"]["seed"])
    episode = _qualitative_episode(qualitative, pose_all, seed)
    thresholds = PhaseThresholds(
        pedestrian_existence=float(config["phases"]["pedestrian_existence_threshold"]),
        pedestrian_max_range_m=float(
            config["phases"]["pedestrian_relevant_max_range_m"]
        ),
        lane_curve_min_abs_curvature_inv_m=float(
            config["phases"]["lane_curve_min_abs_curvature_inv_m"]
        ),
        stop_satisfied_vicinity_m=float(
            config["phases"]["stop_satisfied_vicinity_m"]
        ),
    )
    phases = np.asarray(
        [
            public_phase(row, protocol.observation_order, thresholds)
            for row in episode["physical_observation"]
        ],
        dtype="U40",
    )
    if set(phases) != set(PHASE_ORDER):
        raise RuntimeError(
            f"qualitative episode phase support mismatch: {sorted(set(phases))}"
        )
    selected = select_representative_frames(
        phases=phases,
        steps=episode["step"],
        physical_observation=episode["physical_observation"],
        feature_names=episode["feature_names"],
        pedestrian_min_existence=float(config["selection"]["pedestrian_min_existence"]),
    )

    frame_paths = {
        phase: source_frames / f"{phase}_rgb_yolo.png" for phase in PHASE_ORDER
    }
    raw_paths = {phase: source_frames / f"{phase}_rgb.png" for phase in PHASE_ORDER}
    if args.reuse_rgb:
        for path in (*frame_paths.values(), *raw_paths.values()):
            if not path.exists():
                raise FileNotFoundError(f"cannot reuse missing RGB frame: {path}")
        previous_manifest_path = destinations["representative_manifest"]
        if not previous_manifest_path.exists():
            raise FileNotFoundError(
                "RGB reuse requires the prior representative-frame manifest"
            )
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
        replay = dict(previous_manifest["replay"])
        replay["rgb_source_reused_without_replay"] = True
    else:
        frames, replay = _replay_selected_rgb(
            policy_config=paths["policy_config"],
            checkpoint=paths["checkpoint"],
            seed=seed,
            trace=episode,
            selected=selected,
            device=args.device,
            observation_tolerance=float(
                config["qualitative_source"]["maximum_observation_replay_error"]
            ),
            action_tolerance=float(
                config["qualitative_source"]["maximum_action_replay_error"]
            ),
        )
        detector = _detector(protocol)
        for phase, rgb in frames.items():
            _write_rgb(raw_paths[phase], rgb)
            _write_rgb(frame_paths[phase], _draw_detections(rgb, detector.detect(rgb)))

    rgb = {
        phase: cv2.cvtColor(cv2.imread(str(frame_paths[phase])), cv2.COLOR_BGR2RGB)
        for phase in PHASE_ORDER
    }
    if any(frame is None for frame in rgb.values()):
        raise RuntimeError("failed to decode selected RGB frames")

    scenario = load_scenario(protocol.stage("c4").scenario_config_path)
    native = protocol.raw["native_start"]
    tiles = load_small_loop_tiles(
        map_name=str(scenario.map_path),
        anchor_tile=tuple(int(value) for value in native["start_tile"]),
        anchor_heading_rad=float(native["base_heading_rad"]),
    )
    _style(config)
    _plot_overall(rows, destinations["overall_png"], destinations["overall_pdf"], config)
    _plot_heatmap(
        rows,
        target="v_cmd_mps",
        destination_png=destinations["heatmap_v_png"],
        destination_pdf=destinations["heatmap_v_pdf"],
        config=config,
    )
    _plot_heatmap(
        rows,
        target="omega_cmd_rad_s",
        destination_png=destinations["heatmap_omega_png"],
        destination_pdf=destinations["heatmap_omega_pdf"],
        config=config,
    )
    _plot_decision_trace(
        episode=episode,
        phases=phases,
        selected=selected,
        rows=rows,
        tiles=tiles,
        scenario=scenario,
        destination_png=destinations["trace_png"],
        destination_pdf=destinations["trace_pdf"],
        config=config,
    )
    _plot_representative_panels(
        episode=episode,
        selected=selected,
        rows=rows,
        rgb=rgb,
        tiles=tiles,
        scenario=scenario,
        destination_png=destinations["panels_png"],
        destination_pdf=destinations["panels_pdf"],
        config=config,
    )

    representative_manifest = _representative_manifest(
        config_path=config_path,
        config=config,
        episode=episode,
        selected=selected,
        phases=phases,
        frame_paths=frame_paths,
        raw_paths=raw_paths,
        replay=replay,
    )
    destinations["representative_manifest"].write_text(
        json.dumps(representative_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_manifest = _figure_manifest(
        config_path=config_path,
        config=config,
        paths=paths,
        rows=rows,
        metrics=metrics,
        destinations=destinations,
        representative_manifest=representative_manifest,
    )
    destinations["figure_manifest"].write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "classification": "COMPLETE",
        "output_directory": str(output),
        "quantitative_source": "frozen R004",
        "qualitative_seed": seed,
        "r004_locked_seeds_rerendered": False,
        "figures": [str(path) for key, path in destinations.items() if "manifest" not in key],
        "replay": replay,
    }


def _resolve_and_verify_sources(config_path: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    r004_dir = _resolve(config_path, config["r004"]["directory"])
    values = {
        "metrics": r004_dir / config["r004"]["metrics"],
        "group_rows": r004_dir / config["r004"]["group_rows"],
        "group_summary": r004_dir / config["r004"]["group_summary"],
        "mean_attribution": r004_dir / config["r004"]["mean_attribution"],
        "draw_attribution": r004_dir / config["r004"]["draw_attribution"],
        "r004_trace": r004_dir / config["r004"]["public_trace"],
        "r004_trace_manifest": r004_dir / config["r004"]["trace_manifest"],
        "policy_config": _resolve(config_path, config["frozen_policy"]["config"]),
        "checkpoint": _resolve(config_path, config["frozen_policy"]["checkpoint"]),
        "qualitative_trace": _resolve(
            config_path, config["qualitative_source"]["public_trace"]
        ),
        "qualitative_pose": _resolve(
            config_path, config["qualitative_source"]["evaluation_pose"]
        ),
        "r006_report": _resolve(config_path, config["r006_audit"]["report"]),
        "scenario_config": _resolve(config_path, config["route_geometry"]["scenario"]),
        "scenario_map": _resolve(config_path, config["route_geometry"]["map"]),
    }
    expected = {
        "metrics": config["r004"]["metrics_sha256"],
        "group_rows": config["r004"]["group_rows_sha256"],
        "group_summary": config["r004"]["group_summary_sha256"],
        "mean_attribution": config["r004"]["mean_attribution_sha256"],
        "draw_attribution": config["r004"]["draw_attribution_sha256"],
        "r004_trace": config["r004"]["public_trace_sha256"],
        "r004_trace_manifest": config["r004"]["trace_manifest_sha256"],
        "policy_config": config["frozen_policy"]["config_sha256"],
        "checkpoint": config["frozen_policy"]["checkpoint_sha256"],
        "qualitative_trace": config["qualitative_source"]["public_trace_sha256"],
        "qualitative_pose": config["qualitative_source"]["evaluation_pose_sha256"],
        "r006_report": config["r006_audit"]["report_sha256"],
        "scenario_config": config["route_geometry"]["scenario_sha256"],
        "scenario_map": config["route_geometry"]["map_sha256"],
    }
    for name, path in values.items():
        actual = _sha256(path)
        if actual != expected[name]:
            raise RuntimeError(f"frozen source hash mismatch for {name}: {actual}")
    return values


def _validate_r004(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if metrics.get("classification") != "PASS" or metrics.get("run_id") != "R004":
        raise RuntimeError("frozen R004 metrics are not PASS")
    if int(metrics["reference_protocol"]["effective_reference_count"]) != int(
        config["r004"]["effective_reference_count"]
    ):
        raise RuntimeError("R004 effective reference count mismatch")
    if tuple(int(seed) for seed in metrics["seeds"]) != tuple(
        int(seed) for seed in config["r004"]["locked_seeds"]
    ):
        raise RuntimeError("R004 locked seed set mismatch")


def _qualitative_episode(
    trace: Mapping[str, np.ndarray], pose: Mapping[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    mask = np.asarray(trace["seed"]) == seed
    pose_mask = np.asarray(pose["seed"]) == seed
    if not np.any(mask) or not np.any(pose_mask):
        raise ValueError("qualitative seed missing from public trace or pose trace")
    episode = {
        key: np.asarray(value[mask]).copy()
        for key, value in trace.items()
        if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == mask.size
    }
    episode["feature_names"] = np.asarray(trace["feature_names"]).copy()
    pose_steps = np.asarray(pose["step"])[pose_mask]
    if not np.array_equal(episode["step"], pose_steps):
        raise RuntimeError("qualitative public trace and pose trace steps differ")
    for key in ("world_x_m", "world_z_m", "heading_rad"):
        episode[key] = np.asarray(pose[key])[pose_mask].copy()
    if not np.array_equal(episode["step"], np.arange(episode["step"].size)):
        raise RuntimeError("qualitative episode steps are not contiguous from zero")
    return episode


def _replay_selected_rgb(
    *,
    policy_config: Path,
    checkpoint: Path,
    seed: int,
    trace: Mapping[str, np.ndarray],
    selected: Mapping[str, Any],
    device: str,
    observation_tolerance: float,
    action_tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_by_index = {value.local_index: phase for phase, value in selected.items()}
    if seed in tuple(range(177101, 177109)):
        raise RuntimeError("refusing to rerender an R004 locked seed")
    checkpoint_before = _sha256(checkpoint)
    agent, _ = PPOAgent.load(checkpoint, device=device)
    environment = PPOCurriculumEnvironment(
        policy_config, stage="c4", split="explanation", seeds=(seed,)
    )
    frames: dict[str, np.ndarray] = {}
    max_observation_error = 0.0
    max_action_error = 0.0
    terminal_match = False
    try:
        observation, _ = environment.reset(seed=seed)
        for local_index in range(len(trace["step"])):
            expected_observation = np.asarray(trace["observation"][local_index])
            observation_error = float(
                np.max(np.abs(np.asarray(observation) - expected_observation))
            )
            max_observation_error = max(max_observation_error, observation_error)
            if observation_error > observation_tolerance:
                raise RuntimeError(
                    f"qualitative replay observation drift at {local_index}: "
                    f"{observation_error}"
                )
            action = agent.act(np.asarray(observation), deterministic=True).environment_action
            expected_action = np.asarray(trace["environment_action"][local_index])
            action_error = float(np.max(np.abs(np.asarray(action) - expected_action)))
            max_action_error = max(max_action_error, action_error)
            if action_error > action_tolerance:
                raise RuntimeError(
                    f"qualitative replay action drift at {local_index}: {action_error}"
                )
            if local_index in selected_by_index:
                frames[selected_by_index[local_index]] = environment.latest_rgb().copy()
            observation, _, terminated, truncated, _ = environment.step(action)
            expected_terminated = bool(trace["terminated"][local_index])
            expected_truncated = bool(trace["truncated"][local_index])
            if bool(terminated) != expected_terminated or bool(truncated) != expected_truncated:
                raise RuntimeError(f"qualitative replay terminal drift at {local_index}")
            if terminated or truncated:
                terminal_match = local_index == len(trace["step"]) - 1
                break
    finally:
        environment.close()
    if set(frames) != set(PHASE_ORDER):
        raise RuntimeError(f"missing selected RGB frames: {sorted(set(PHASE_ORDER)-set(frames))}")
    if _sha256(checkpoint) != checkpoint_before:
        raise RuntimeError("frozen checkpoint changed during qualitative rendering")
    return frames, {
        "mode": "deterministic_rgb_only_replay_of_existing_qualitative_seed",
        "maximum_absolute_observation_error": max_observation_error,
        "maximum_absolute_action_error": max_action_error,
        "terminal_match": terminal_match,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": _sha256(checkpoint),
    }


def _detector(protocol: Any) -> YoloObjectDetector:
    with protocol.belief_config_path.open("rb") as stream:
        detector = tomllib.load(stream)["detector"]
    return YoloObjectDetector(
        protocol.detector_checkpoint_path,
        confidence_threshold=float(detector["confidence_threshold"]),
        iou_threshold=float(detector["nms_iou_threshold"]),
        image_size=int(detector["image_size"]),
        device=str(detector["device"]),
        max_detections=int(detector["max_detections"]),
    )


def _draw_detections(rgb: np.ndarray, detections: Sequence[Any]) -> np.ndarray:
    canvas = np.ascontiguousarray(rgb.copy())
    colours = {"duckie": (230, 159, 0), "stop_sign": (213, 94, 0)}
    for detection in detections:
        box = detection.bounding_box
        name = detection.object_class.value
        colour = colours.get(name, (0, 158, 115))
        p1 = (int(round(box.x_min_px)), int(round(box.y_min_px)))
        p2 = (int(round(box.x_max_px)), int(round(box.y_max_px)))
        cv2.rectangle(canvas, p1, p2, colour, 2)
        cv2.putText(
            canvas,
            f"{name} {detection.confidence:.2f}",
            (p1[0], max(15, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _plot_overall(
    rows: Sequence[Mapping[str, object]], png: Path, pdf: Path, config: Mapping[str, Any]
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=tuple(config["plot"]["overall_figure_size"]), sharex=True
    )
    for axis, target in zip(axes, TARGET_ORDER):
        subset = {
            str(row["group"]): row
            for row in rows
            if row["target"] == target and row["public_phase"] == "all"
        }
        values = np.asarray(
            [float(subset[group]["mean_absolute_group_share"]) for group in GROUP_ORDER]
        )
        low = np.asarray([float(subset[group]["share_ci_low"]) for group in GROUP_ORDER])
        high = np.asarray([float(subset[group]["share_ci_high"]) for group in GROUP_ORDER])
        positions = np.arange(len(GROUP_ORDER))[::-1]
        axis.barh(
            positions,
            values,
            xerr=np.vstack((values - low, high - values)),
            color=[GROUP_COLOURS[group] for group in GROUP_ORDER],
            edgecolor="#222222",
            linewidth=0.6,
            error_kw={"elinewidth": 0.9, "capsize": 2.2, "capthick": 0.9},
        )
        axis.set_yticks(positions, GROUP_ORDER)
        axis.set_xlim(0.0, 0.6)
        axis.set_xlabel("Mean absolute attribution share")
        axis.set_title(f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]})")
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.6)
        axis.set_axisbelow(True)
        for y, value in zip(positions, values):
            axis.text(value + 0.012, y, f"{100*value:.1f}%", va="center", fontsize=8)
    figure.suptitle("Overall R004 Distributional IG by semantic group", fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        "Error bars: seed-bootstrap 95% CI. Shares are relative to the frozen "
        "phase-conditioned reference distribution.",
        ha="center",
        fontsize=8.5,
    )
    _save(figure, png, pdf, int(config["plot"]["png_dpi"]))


def _plot_heatmap(
    rows: Sequence[Mapping[str, object]], *, target: str, destination_png: Path,
    destination_pdf: Path, config: Mapping[str, Any]
) -> None:
    values = summary_matrix(rows, target=target)
    figure, axis = plt.subplots(figsize=tuple(config["plot"]["heatmap_figure_size"]))
    image = axis.imshow(values, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(GROUP_ORDER)), GROUP_ORDER, rotation=25, ha="right")
    axis.set_yticks(
        np.arange(len(PHASE_ORDER)), [PHASE_LABELS[phase] for phase in PHASE_ORDER]
    )
    axis.set_title(
        f"Phase-specific R004 Distributional IG — {TARGET_LABELS[target]} "
        f"({TARGET_UNITS[target]})",
        fontweight="bold",
    )
    threshold = 0.48
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
                fontweight="bold" if value >= 0.35 else "normal",
            )
    colourbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colourbar.set_label("Mean absolute group share")
    axis.set_xlabel("Semantic policy-input group")
    axis.set_ylabel("Public driving phase")
    _save(figure, destination_png, destination_pdf, int(config["plot"]["png_dpi"]))


def _plot_decision_trace(
    *, episode: Mapping[str, np.ndarray], phases: np.ndarray, selected: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]], tiles: Sequence[Any], scenario: Any,
    destination_png: Path, destination_pdf: Path, config: Mapping[str, Any]
) -> None:
    figure = plt.figure(figsize=tuple(config["plot"]["trace_figure_size"]))
    grid = figure.add_gridspec(4, 2, width_ratios=(1.05, 1.75), hspace=0.18, wspace=0.25)
    bev = figure.add_subplot(grid[:, 0])
    action_axis = figure.add_subplot(grid[0, 1])
    belief_axis = figure.add_subplot(grid[1, 1], sharex=action_axis)
    v_axis = figure.add_subplot(grid[2, 1], sharex=action_axis)
    omega_axis = figure.add_subplot(grid[3, 1], sharex=action_axis)

    _draw_route(bev, tiles, scenario)
    x = np.asarray(episode["world_x_m"], dtype=float)
    z = np.asarray(episode["world_z_m"], dtype=float)
    bev.plot(x, z, color="#555555", linewidth=1.5, alpha=0.75, label="Ego trajectory")
    marker_order = (
        "pedestrian_relevant",
        "stop_required",
        "stop_satisfied",
        "lane_curve",
        "nominal",
    )
    marker_shapes = ("o", "s", "D", "^", "P")
    for phase, marker in zip(marker_order, marker_shapes):
        index = selected[phase].local_index
        bev.scatter(
            x[index], z[index], marker=marker, s=55, color=_phase_colour(phase),
            edgecolor="#111111", linewidth=0.6, zorder=8,
            label=PHASE_LABELS[phase],
        )
    bev.set_title("BEV trajectory and selected public phases", fontweight="bold")
    bev.set_xlabel("world x (m)")
    bev.set_ylabel("world z (m)")
    bev.set_aspect("equal", adjustable="box")
    bev.invert_yaxis()
    bev.legend(fontsize=7, loc="best")

    steps = np.asarray(episode["step"], dtype=float)
    actions = np.asarray(episode["physical_action"], dtype=float)
    action_axis.plot(steps, actions[:, 0], color="#0072B2", label=r"$v_{cmd}$ (m/s)")
    action_twin = action_axis.twinx()
    action_twin.plot(
        steps, actions[:, 1], color="#D55E00", alpha=0.85,
        label=r"$\omega_{cmd}$ (rad/s)",
    )
    action_axis.set_ylabel(r"$v_{cmd}$ (m/s)")
    action_twin.set_ylabel(r"$\omega_{cmd}$ (rad/s)")
    action_axis.set_title("Physical deterministic PPO action", fontweight="bold")
    handles1, labels1 = action_axis.get_legend_handles_labels()
    handles2, labels2 = action_twin.get_legend_handles_labels()
    action_axis.legend(handles1 + handles2, labels1 + labels2, ncol=2, loc="upper right")

    physical = np.asarray(episode["physical_observation"], dtype=float)
    names = tuple(str(name) for name in episode["feature_names"])
    belief_axis.plot(
        steps, physical[:, names.index("pedestrian_existence_probability")],
        color=GROUP_COLOURS["Pedestrian"], label="P(pedestrian exists)",
    )
    belief_axis.plot(
        steps, physical[:, names.index("stop_sign_existence_probability")],
        color=GROUP_COLOURS["Stop"], label="P(stop sign exists)",
    )
    belief_axis.plot(
        steps, physical[:, names.index("stop_mode_required")],
        color="#333333", linestyle="--", label="stop REQUIRED",
    )
    belief_axis.plot(
        steps, physical[:, names.index("stop_mode_satisfied")],
        color="#888888", linestyle=":", label="stop SATISFIED",
    )
    belief_axis.set_ylim(-0.04, 1.07)
    belief_axis.set_ylabel("Public belief/state")
    belief_axis.legend(ncol=2, fontsize=7, loc="upper right")

    v_shares = summary_matrix(rows, target="v_cmd_mps")
    omega_shares = summary_matrix(rows, target="omega_cmd_rad_s")
    phase_index = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    timeline_v = np.vstack([v_shares[phase_index[str(phase)]] for phase in phases]).T
    timeline_omega = np.vstack(
        [omega_shares[phase_index[str(phase)]] for phase in phases]
    ).T
    colours = [GROUP_COLOURS[group] for group in GROUP_ORDER]
    v_axis.stackplot(steps, timeline_v, labels=GROUP_ORDER, colors=colours, alpha=0.92)
    omega_axis.stackplot(steps, timeline_omega, colors=colours, alpha=0.92)
    v_axis.set_ylabel("R004 share\nfor $v_{cmd}$")
    omega_axis.set_ylabel("R004 share\nfor $\omega_{cmd}$")
    omega_axis.set_xlabel("Simulator step")
    v_axis.set_ylim(0.0, 1.0)
    omega_axis.set_ylim(0.0, 1.0)
    v_axis.legend(ncol=3, fontsize=6.5, loc="upper right")

    for axis in (action_axis, belief_axis, v_axis, omega_axis):
        _shade_phases(axis, phases)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.5)
    for phase, frame in selected.items():
        for axis in (action_axis, belief_axis, v_axis, omega_axis):
            axis.axvline(frame.simulator_step, color=_phase_colour(phase), linewidth=0.8)

    figure.suptitle(
        "BEV Belief–Action Decision Trace — qualitative C4 seed 176001",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.008,
        "BEV pose/route are post-hoc evaluation-only and were unavailable to PPO/IG. "
        "Attribution ribbons are frozen R004 phase means, not new frame-level estimates.",
        ha="center",
        fontsize=8,
    )
    _save(figure, destination_png, destination_pdf, int(config["plot"]["png_dpi"]))


def _plot_representative_panels(
    *, episode: Mapping[str, np.ndarray], selected: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]], rgb: Mapping[str, np.ndarray],
    tiles: Sequence[Any], scenario: Any, destination_png: Path, destination_pdf: Path,
    config: Mapping[str, Any]
) -> None:
    display_phases = (
        "lane_curve",
        "pedestrian_relevant",
        "stop_required",
        "stop_satisfied",
    )
    figure, axes = plt.subplots(
        len(display_phases),
        4,
        figsize=tuple(config["plot"]["panel_figure_size"]),
        gridspec_kw={"width_ratios": (1.35, 1.1, 1.05, 1.45)},
    )
    v_matrix = summary_matrix(rows, target="v_cmd_mps")
    omega_matrix = summary_matrix(rows, target="omega_cmd_rad_s")
    phase_to_row = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    physical = np.asarray(episode["physical_observation"], dtype=float)
    actions = np.asarray(episode["physical_action"], dtype=float)
    names = tuple(str(name) for name in episode["feature_names"])

    for row_index, phase in enumerate(display_phases):
        frame = selected[phase]
        index = frame.local_index
        rgb_axis, bev_axis, belief_axis, ig_axis = axes[row_index]
        rgb_axis.imshow(rgb[phase])
        rgb_axis.set_title(
            ("Perception provenance\n" if row_index == 0 else "")
            + f"{PHASE_LABELS[phase]} · step {frame.simulator_step}",
            fontsize=9,
            fontweight="bold",
        )
        rgb_axis.set_axis_off()
        rgb_axis.text(
            0.01,
            0.02,
            "RGB + YOLO audit boxes\nMobileNet lane belief shown at right",
            transform=rgb_axis.transAxes,
            fontsize=6.5,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.62, "pad": 2, "edgecolor": "none"},
        )

        _draw_bev_snapshot(
            bev_axis,
            episode=episode,
            local_index=index,
            physical=physical[index],
            feature_names=names,
            tiles=tiles,
            scenario=scenario,
        )
        if row_index == 0:
            bev_axis.set_title("BEV (evaluation only)", fontsize=9, fontweight="bold")

        belief_axis.set_axis_off()
        stop_phase = phase in ("stop_required", "stop_satisfied")
        belief_axis.text(
            0.0,
            0.98,
            _belief_text(phase, physical[index], names),
            va="top",
            ha="left",
            fontsize=6.1 if stop_phase else 7.3,
            family="monospace",
            linespacing=1.12 if stop_phase else 1.38,
        )
        belief_axis.text(
            0.0,
            0.015,
            f"Action\n  v = {actions[index,0]:.3f} m/s\n  ω = {actions[index,1]:+.3f} rad/s",
            va="bottom",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )
        if row_index == 0:
            belief_axis.set_title("Public belief + action", fontsize=9, fontweight="bold")

        phase_row = phase_to_row[phase]
        y = np.arange(len(GROUP_ORDER))
        height = 0.36
        ig_axis.barh(
            y + height / 2,
            v_matrix[phase_row],
            height=height,
            color=[GROUP_COLOURS[group] for group in GROUP_ORDER],
            edgecolor="#222222",
            linewidth=0.35,
            hatch="//",
            label=r"$v_{cmd}$",
        )
        ig_axis.barh(
            y - height / 2,
            omega_matrix[phase_row],
            height=height,
            color=[GROUP_COLOURS[group] for group in GROUP_ORDER],
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.65,
            label=r"$\omega_{cmd}$",
        )
        ig_axis.set_yticks(y, GROUP_ORDER, fontsize=6.8)
        ig_axis.invert_yaxis()
        ig_axis.set_xlim(0.0, 1.0)
        ig_axis.grid(axis="x", color="#E0E0E0", linewidth=0.5)
        ig_axis.set_axisbelow(True)
        if row_index == 0:
            ig_axis.set_title("R004 phase-mean Distributional IG", fontsize=9, fontweight="bold")
            ig_axis.legend(fontsize=7, loc="lower right")
        if row_index == len(display_phases) - 1:
            ig_axis.set_xlabel("Mean absolute group share")

    figure.suptitle(
        "Representative Belief-PPO decisions: perception → belief → attribution → action",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.006,
        "RGB/BEV panels are a qualitative replay of pre-existing seed 176001. "
        "BEV geometry is post-hoc only; bars are frozen R004 phase aggregates.",
        ha="center",
        fontsize=8,
    )
    _save(figure, destination_png, destination_pdf, int(config["plot"]["png_dpi"]))


def _draw_route(axis: Any, tiles: Sequence[Any], scenario: Any) -> None:
    for tile in tiles:
        i, j = tile.coords
        size = float(tile.tile_size_m)
        axis.add_patch(
            Rectangle(
                (i * size, j * size), size, size, facecolor="#ECECEC",
                edgecolor="#B5B5B5", linewidth=0.6, zorder=0,
            )
        )
        control = np.asarray(tile.curve_world, dtype=float)
        degree = control.shape[0] - 1
        t = np.linspace(0.0, 1.0, 80)
        curve = sum(
            comb(degree, k)
            * ((1.0 - t) ** (degree - k) * t**k)[:, None]
            * control[k]
            for k in range(degree + 1)
        )
        axis.plot(curve[:, 0], curve[:, 2], "--", color="#555555", linewidth=0.8)
    pedestrian = scenario.pedestrian
    axis.plot(
        [pedestrian.path_start_world_x_m, pedestrian.path_end_world_x_m],
        [pedestrian.path_start_world_z_m, pedestrian.path_end_world_z_m],
        color=GROUP_COLOURS["Pedestrian"], linestyle=":", linewidth=2.0,
    )
    stop = scenario.stop_line
    half = 0.20
    across_x = np.sin(stop.route_heading_rad)
    across_z = np.cos(stop.route_heading_rad)
    axis.plot(
        [stop.world_x_m - half * across_x, stop.world_x_m + half * across_x],
        [stop.world_z_m - half * across_z, stop.world_z_m + half * across_z],
        color=GROUP_COLOURS["Stop"], linewidth=2.4,
    )


def _draw_bev_snapshot(
    axis: Any, *, episode: Mapping[str, np.ndarray], local_index: int,
    physical: np.ndarray, feature_names: Sequence[str], tiles: Sequence[Any], scenario: Any
) -> None:
    _draw_route(axis, tiles, scenario)
    x = np.asarray(episode["world_x_m"], dtype=float)
    z = np.asarray(episode["world_z_m"], dtype=float)
    axis.plot(x, z, color="#BBBBBB", linewidth=0.7, alpha=0.6)
    axis.plot(x[: local_index + 1], z[: local_index + 1], color="#333333", linewidth=1.4)
    heading = float(episode["heading_rad"][local_index])
    center = np.asarray([x[local_index], z[local_index]])
    forward = np.asarray([np.cos(heading), -np.sin(heading)])
    left = np.asarray([np.sin(heading), np.cos(heading)])
    triangle = np.vstack(
        (
            center + 0.10 * forward,
            center - 0.065 * forward + 0.055 * left,
            center - 0.065 * forward - 0.055 * left,
        )
    )
    axis.add_patch(Polygon(triangle, closed=True, color="#111111", zorder=8))

    values = {name: float(physical[index]) for index, name in enumerate(feature_names)}
    if values["pedestrian_existence_probability"] >= 0.4 and values[
        "pedestrian_range_mean_m"
    ] > 0.0:
        mean, covariance = pedestrian_belief_world(
            ego_x_m=center[0],
            ego_z_m=center[1],
            ego_heading_rad=heading,
            range_mean_m=values["pedestrian_range_mean_m"],
            range_std_m=values["pedestrian_range_std_m"],
            bearing_mean_rad=values["pedestrian_bearing_mean_rad"],
            bearing_std_rad=values["pedestrian_bearing_std_rad"],
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        vector = eigenvectors[:, order[0]]
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        ellipse = Ellipse(
            xy=mean,
            width=2.0 * np.sqrt(eigenvalues[0]),
            height=2.0 * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor=GROUP_COLOURS["Pedestrian"],
            edgecolor="#8A5A00",
            alpha=0.28,
            linewidth=1.0,
            zorder=7,
        )
        axis.add_patch(ellipse)
        axis.scatter(*mean, marker="x", color="#8A5A00", s=24, zorder=9)
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])


def _belief_text(phase: str, row: np.ndarray, names: Sequence[str]) -> str:
    value = {name: float(row[index]) for index, name in enumerate(names)}
    lane = (
        "LANE BELIEF\n"
        f"valid {value['lane_validity_probability']:.2f}\n"
        f"d   {value['lane_lateral_error_mean_m']:+.3f} ± "
        f"{value['lane_lateral_error_std_m']:.3f} m\n"
        f"phi {value['lane_heading_error_mean_rad']:+.3f} ± "
        f"{value['lane_heading_error_std_rad']:.3f} rad\n"
        f"k   {value['lane_curvature_mean_inv_m']:+.2f} ± "
        f"{value['lane_curvature_std_inv_m']:.2f} 1/m"
    )
    if phase == "pedestrian_relevant":
        return (
            "PEDESTRIAN BELIEF\n"
            f"P(e) {value['pedestrian_existence_probability']:.3f}\n"
            f"r    {value['pedestrian_range_mean_m']:.3f} ± "
            f"{value['pedestrian_range_std_m']:.3f} m\n"
            f"beta {value['pedestrian_bearing_mean_rad']:+.3f} ± "
            f"{value['pedestrian_bearing_std_rad']:.3f} rad\n"
            f"rdot {value['pedestrian_radial_velocity_mean_mps']:+.3f} ± "
            f"{value['pedestrian_radial_velocity_std_mps']:.3f} m/s"
        )
    if phase in ("stop_required", "stop_satisfied"):
        mode = "REQUIRED" if value["stop_mode_required"] > 0.5 else "SATISFIED"
        return (
            "STOP BELIEF / ROUTE\n"
            f"P(sign) {value['stop_sign_existence_probability']:.3f}\n"
            f"sign r  {value['stop_sign_range_mean_m']:.3f} ± "
            f"{value['stop_sign_range_std_m']:.3f} m\n"
            f"line d  {value['stop_line_distance_m']:+.3f} m\n"
            f"mode    {mode}\n\n" + lane
        )
    return lane


def _shade_phases(axis: Any, phases: np.ndarray) -> None:
    for start, end, phase in phase_runs(phases):
        axis.axvspan(start, end, color=_phase_colour(phase), alpha=0.055, linewidth=0)


def _phase_colour(phase: str) -> str:
    return {
        "nominal": "#999999",
        "lane_curve": GROUP_COLOURS["Lane"],
        "pedestrian_relevant": GROUP_COLOURS["Pedestrian"],
        "stop_required": GROUP_COLOURS["Stop"],
        "stop_satisfied": GROUP_COLOURS["StopLine"],
    }[phase]


def _representative_manifest(
    *, config_path: Path, config: Mapping[str, Any], episode: Mapping[str, np.ndarray],
    selected: Mapping[str, Any], phases: np.ndarray, frame_paths: Mapping[str, Path],
    raw_paths: Mapping[str, Path], replay: Mapping[str, Any]
) -> dict[str, Any]:
    names = tuple(str(name) for name in episode["feature_names"])
    rows: dict[str, Any] = {}
    for phase, frame in selected.items():
        index = frame.local_index
        public = {
            name: float(episode["physical_observation"][index, feature_index])
            for feature_index, name in enumerate(names)
        }
        rows[phase] = {
            "simulator_step": frame.simulator_step,
            "selection_rule": frame.rule,
            "public_phase_segment": [frame.segment_start, frame.segment_end],
            "physical_action": {
                "v_cmd_mps": float(episode["physical_action"][index, 0]),
                "omega_cmd_rad_s": float(episode["physical_action"][index, 1]),
            },
            "public_29d": public,
            "rgb_path": str(frame_paths[phase].relative_to(ROOT)),
            "rgb_sha256": _sha256(frame_paths[phase]),
            "raw_rgb_path": str(raw_paths[phase].relative_to(ROOT)),
            "raw_rgb_sha256": _sha256(raw_paths[phase]),
        }
    return {
        "schema_version": 1,
        "classification": "qualitative_example_only",
        "seed": int(config["qualitative_source"]["seed"]),
        "r004_locked_seed": False,
        "r004_locked_seeds_rerendered": False,
        "selection_uses_public_29d_only": True,
        "selection_uses_attribution": False,
        "selection_uses_rgb_content": False,
        "selection_uses_world_pose": False,
        "pose_usage": "post-hoc BEV visualization only",
        "pose_enters_policy_or_attribution": False,
        "phase_counts": {phase: int(np.sum(phases == phase)) for phase in PHASE_ORDER},
        "replay": dict(replay),
        "frames": rows,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
    }


def _figure_manifest(
    *, config_path: Path, config: Mapping[str, Any], paths: Mapping[str, Path],
    rows: Sequence[Mapping[str, object]], metrics: Mapping[str, Any],
    destinations: Mapping[str, Path], representative_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    provenance_sources = tuple(paths.values()) + (
        config_path,
        Path(__file__).resolve(),
        ROOT / "src" / "duckie_pomdp" / "explain" / "final_visualization.py",
    )
    sources = {
        str(path.relative_to(ROOT)): _sha256(path) for path in provenance_sources
    }
    figures = {
        str(path.relative_to(ROOT)): _sha256(path)
        for key, path in destinations.items()
        if key not in ("figure_manifest",) and path.exists()
    }
    exact = {
        target: {
            phase: {
                group: next(
                    float(row["mean_absolute_group_share"])
                    for row in rows
                    if row["target"] == target
                    and row["public_phase"] == phase
                    and row["group"] == group
                )
                for group in GROUP_ORDER
            }
            for phase in ("all",) + PHASE_ORDER
        }
        for target in TARGET_ORDER
    }
    return {
        "schema_version": 1,
        "classification": "COMPLETE",
        "method_quantitative": config["method_quantitative"],
        "method_qualitative": config["method_qualitative"],
        "quantitative_source": "frozen R004 artifacts only",
        "effective_references_per_factual_state": int(
            metrics["reference_protocol"]["effective_reference_count"]
        ),
        "semantic_group_order": list(GROUP_ORDER),
        "phase_order": list(PHASE_ORDER),
        "exact_group_shares": exact,
        "r002_status": "LIMITED_fixed_reference_baseline_sensitive",
        "r002b_status": "PASS_distributional_reference_robustness",
        "r004_status": "PASS_once_only_locked_holdout",
        "r006_status": config["r006_audit"]["status"],
        "r007_status": "BLOCKED_not_executed",
        "r006_note": (
            "The preregistered confirmatory holdout intervention run stopped at a "
            "numerical replay-integrity gate before any intervention was evaluated."
        ),
        "qualitative_example": {
            "seed": representative_manifest["seed"],
            "evaluation_only": True,
            "not_used_for_quantitative_claims": True,
            "attribution_display": "frozen R004 phase-mean group shares",
        },
        "source_sha256": sources,
        "artifact_sha256": figures,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "software": {
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
            "opencv": cv2.__version__,
        },
    }


def _destinations(output: Path) -> dict[str, Path]:
    return {
        "overall_png": output / "quantitative_overall_attribution.png",
        "overall_pdf": output / "quantitative_overall_attribution.pdf",
        "heatmap_v_png": output / "quantitative_phase_heatmap_v.png",
        "heatmap_v_pdf": output / "quantitative_phase_heatmap_v.pdf",
        "heatmap_omega_png": output / "quantitative_phase_heatmap_omega.png",
        "heatmap_omega_pdf": output / "quantitative_phase_heatmap_omega.pdf",
        "trace_png": output / "qualitative_bev_decision_trace.png",
        "trace_pdf": output / "qualitative_bev_decision_trace.pdf",
        "panels_png": output / "qualitative_representative_panels.png",
        "panels_pdf": output / "qualitative_representative_panels.pdf",
        "representative_manifest": output / "representative_frame_manifest.json",
        "figure_manifest": output / "figure_data_manifest.json",
    }


def _read_group_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _style(config: Mapping[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": config["plot"]["font_family"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": int(config["plot"]["png_dpi"]),
            "savefig.dpi": int(config["plot"]["png_dpi"]),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: Any, png: Path, pdf: Path, dpi: int) -> None:
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
