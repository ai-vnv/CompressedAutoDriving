"""Replay one frozen C4 explanation seed and map PPO IG onto a BEV route.

World pose and object landmarks are read only after the frozen actor has chosen
its action. They are saved in a separate evaluation-only artifact and never
enter PPO inference or Integrated Gradients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - frozen Python 3.10 environment
    import tomli as tomllib

from duckie_pomdp.control import PPOAgent, PPOCurriculumEnvironment
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol
from duckie_pomdp.control.start_sampler import load_small_loop_tiles
from duckie_pomdp.explain.ig_bev import (
    POSE_TRACE_KEYS,
    aggregate_groups,
    align_pose_to_samples,
    resolve_feature_groups,
    signed_total,
    validate_pose_trace,
)
from duckie_pomdp.scenario import load_scenario


ROOT = Path(__file__).resolve().parents[1]
GROUP_COLOURS = {
    "lane_belief": "#0072B2",
    "ego_motion": "#56B4E9",
    "road": "#009E73",
    "pedestrian_belief": "#E69F00",
    "stop_belief": "#D55E00",
    "previous_action": "#CC79A7",
}
GROUP_LABELS = {
    "lane_belief": "Lane belief",
    "ego_motion": "Ego motion",
    "road": "Road",
    "pedestrian_belief": "Pedestrian belief",
    "stop_belief": "Stop belief",
    "previous_action": "Previous action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f11_ppo_ig_bev_v1.toml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2, sort_keys=True))


def render(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_config(config_path)
    frozen = config["frozen_explanation"]
    f11_config_path = resolve_path(config_path, frozen["config"])
    trajectory_path = resolve_path(config_path, frozen["trajectory"])
    ig_path = resolve_path(config_path, frozen["integrated_gradients"])
    verify_hash(f11_config_path, frozen["config_sha256"])
    verify_hash(trajectory_path, frozen["trajectory_sha256"])
    verify_hash(ig_path, frozen["integrated_gradients_sha256"])

    with f11_config_path.open("rb") as handle:
        f11 = tomllib.load(handle)
    policy_config = resolve_path(f11_config_path, f11["frozen_policy"]["config"])
    checkpoint = resolve_path(f11_config_path, f11["frozen_policy"]["checkpoint"])
    verify_hash(policy_config, f11["frozen_policy"]["config_sha256"])
    checkpoint_hash_before = verify_hash(
        checkpoint, f11["frozen_policy"]["checkpoint_sha256"]
    )
    protocol = load_ppo_curriculum_protocol(policy_config)
    stage_key = str(config["replay"]["stage"])
    stage = protocol.stage(stage_key)
    seed = int(config["replay"]["seed"])
    if seed != int(f11["data"]["qualitative_seed"]):
        raise ValueError("BEV seed must equal the pre-registered qualitative seed")
    if seed not in tuple(int(value) for value in f11["data"]["seeds"]):
        raise ValueError("BEV seed is outside the frozen explanation seed set")

    output = resolve_path(config_path, config["artifacts"]["directory"])
    destinations = {
        "pose": output / "evaluation_only_pose_trace.npz",
        "samples": output / "ig_bev_samples.csv",
        "png": output / "ig_bev_action_map.png",
        "pdf": output / "ig_bev_action_map.pdf",
        "manifest": output / "ig_bev_manifest.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in destinations.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite BEV artifacts: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    trajectory = dict(np.load(trajectory_path, allow_pickle=False))
    ig = dict(np.load(ig_path, allow_pickle=False))
    _validate_ig_alignment(trajectory, ig)
    pose_trace, replay = replay_evaluation_pose(
        policy_config=policy_config,
        checkpoint=checkpoint,
        stage=stage_key,
        seed=seed,
        trajectory=trajectory,
        device=args.device,
        observation_tolerance=float(config["replay"]["maximum_observation_error"]),
        action_tolerance=float(config["replay"]["maximum_action_error"]),
    )
    validate_pose_trace(pose_trace)
    np.savez_compressed(destinations["pose"], **pose_trace)

    sample_mask = np.asarray(ig["seed"]) == seed
    if not np.any(sample_mask):
        raise ValueError("frozen IG artifact has no samples for the qualitative seed")
    sample_seed = np.asarray(ig["seed"])[sample_mask]
    sample_step = np.asarray(ig["step"])[sample_mask]
    aligned_pose = align_pose_to_samples(
        pose_trace, sample_seed=sample_seed, sample_step=sample_step
    )
    target_names = tuple(str(value) for value in ig["target_names"])
    requested_targets = tuple(str(value) for value in config["plot"]["targets"])
    if requested_targets != ("v_cmd_mps", "omega_cmd_rad_s"):
        raise ValueError("BEV v1 requires v_cmd_mps and omega_cmd_rad_s panels")
    target_indices = {name: target_names.index(name) for name in requested_targets}
    feature_names = tuple(str(value) for value in ig["feature_names"])
    groups = resolve_feature_groups(feature_names, f11["feature_groups"])
    target_data: dict[str, dict[str, Any]] = {}
    for target in requested_targets:
        values = np.asarray(ig["attributions"])[target_indices[target], sample_mask]
        target_data[target] = {
            "attributions": values,
            "signed_total": signed_total(values),
            "groups": aggregate_groups(values, groups),
        }

    scenario = load_scenario(stage.scenario_config_path)
    native = protocol.raw["native_start"]
    tiles = load_small_loop_tiles(
        map_name=str(scenario.map_path),
        anchor_tile=tuple(int(value) for value in native["start_tile"]),
        anchor_heading_rad=float(native["base_heading_rad"]),
    )
    plot_bev(
        destination_png=destinations["png"],
        destination_pdf=destinations["pdf"],
        pose=aligned_pose,
        target_data=target_data,
        tiles=tiles,
        scenario=scenario,
        stop_sign_world=replay["stop_sign_world"],
        signed_colour_quantile=float(config["plot"]["signed_colour_quantile"]),
        png_dpi=int(config["plot"]["png_dpi"]),
    )
    write_sample_csv(
        destinations["samples"],
        seed=sample_seed,
        step=sample_step,
        pose=aligned_pose,
        target_data=target_data,
    )
    checkpoint_hash_after = sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("frozen checkpoint changed during BEV replay")
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "seed": seed,
        "seed_role": config["replay"]["seed_role"],
        "stage": stage_key,
        "samples": int(sample_seed.size),
        "trajectory_frames_replayed": int(pose_trace["step"].size),
        "runtime_policy_path": (
            "front RGB -> MobileNet lane belief + YOLO/F9c object beliefs "
            "-> 29D PPO -> action"
        ),
        "pose_source": config["replay"]["pose_source"],
        "pose_usage": "evaluation-only BEV alignment after action selection",
        "pose_enters_policy_or_ig": False,
        "replay_max_abs_observation_error": replay[
            "maximum_absolute_observation_error"
        ],
        "replay_max_abs_action_error": replay["maximum_absolute_action_error"],
        "replay_terminal_match": replay["terminal_match"],
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "source_sha256": {
            str(config_path.relative_to(ROOT)): sha256(config_path),
            str(f11_config_path.relative_to(ROOT)): sha256(f11_config_path),
            str(trajectory_path.relative_to(ROOT)): sha256(trajectory_path),
            str(ig_path.relative_to(ROOT)): sha256(ig_path),
            str(policy_config.relative_to(ROOT)): sha256(policy_config),
            str(checkpoint.relative_to(ROOT)): checkpoint_hash_after,
            str(stage.scenario_config_path.relative_to(ROOT)): sha256(
                stage.scenario_config_path
            ),
            str(scenario.map_path.relative_to(ROOT)): sha256(scenario.map_path),
        },
        "artifact_sha256": {
            path.name: sha256(path)
            for name, path in destinations.items()
            if name != "manifest"
        },
        "semantic_groups": list(groups),
        "figure_panels": [
            "signed total IG for v_cmd",
            "dominant semantic group for v_cmd",
            "signed total IG for omega_cmd",
            "dominant semantic group for omega_cmd",
        ],
    }
    destinations["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("method") != "integrated_gradients_bev":
        raise ValueError("BEV config must select integrated_gradients_bev")
    if bool(config["replay"].get("pose_enters_policy_or_ig", True)):
        raise ValueError("evaluation pose must never enter policy inference or IG")
    return config


def replay_evaluation_pose(
    *,
    policy_config: Path,
    checkpoint: Path,
    stage: str,
    seed: int,
    trajectory: Mapping[str, np.ndarray],
    device: str,
    observation_tolerance: float,
    action_tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    indices = np.flatnonzero(np.asarray(trajectory["seed"]) == seed)
    if indices.size == 0:
        raise ValueError(f"trajectory does not contain seed {seed}")
    agent, _ = PPOAgent.load(checkpoint, device=device)
    environment = PPOCurriculumEnvironment(
        policy_config, stage=stage, split="explanation", seeds=(seed,)
    )
    rows = {key: [] for key in POSE_TRACE_KEYS}
    maximum_observation_error = 0.0
    maximum_action_error = 0.0
    stop_sign_world: tuple[float, float] | None = None
    terminal_match = False
    try:
        observation, _ = environment.reset(seed=seed)
        for local_index, trajectory_index in enumerate(indices):
            expected_step = int(trajectory["step"][trajectory_index])
            if expected_step != local_index:
                raise RuntimeError("frozen explanation trajectory steps are not contiguous")
            action = agent.act(
                np.asarray(observation, dtype=np.float32), deterministic=True
            ).environment_action
            observation_error = float(
                np.max(
                    np.abs(
                        np.asarray(observation, dtype=np.float32)
                        - trajectory["observation"][trajectory_index]
                    )
                )
            )
            action_error = float(
                np.max(
                    np.abs(
                        np.asarray(action, dtype=np.float32)
                        - trajectory["environment_action"][trajectory_index]
                    )
                )
            )
            maximum_observation_error = max(maximum_observation_error, observation_error)
            maximum_action_error = max(maximum_action_error, action_error)
            if observation_error > observation_tolerance:
                raise RuntimeError(
                    f"deterministic replay observation drift at step {expected_step}: "
                    f"{observation_error} > {observation_tolerance}"
                )
            if action_error > action_tolerance:
                raise RuntimeError(
                    f"deterministic replay action drift at step {expected_step}: "
                    f"{action_error} > {action_tolerance}"
                )

            # Read only after the actor has selected its action. This value is
            # written to an evaluation-only trace and never returned to PPO/IG.
            privileged = environment._integration.privileged.read()
            pose = privileged.ego_world_pose
            rows["seed"].append(seed)
            rows["step"].append(expected_step)
            rows["world_x_m"].append(float(pose.x_m))
            rows["world_z_m"].append(float(pose.z_m))
            rows["heading_rad"].append(float(pose.heading_rad))
            if stop_sign_world is None and privileged.stop_sign_world_position is not None:
                point = privileged.stop_sign_world_position
                stop_sign_world = (float(point.x_m), float(point.z_m))

            observation, _, terminated, truncated, _ = environment.step(action)
            expected_terminal = bool(trajectory["terminated"][trajectory_index])
            expected_truncated = bool(trajectory["truncated"][trajectory_index])
            if bool(terminated) != expected_terminal or bool(truncated) != expected_truncated:
                raise RuntimeError(f"deterministic replay terminal drift at step {expected_step}")
            if terminated or truncated:
                terminal_match = local_index == indices.size - 1
                break
    finally:
        environment.close()
    if len(rows["step"]) != indices.size:
        raise RuntimeError("deterministic replay row count differs from frozen trajectory")
    arrays = {
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "step": np.asarray(rows["step"], dtype=np.int32),
        "world_x_m": np.asarray(rows["world_x_m"], dtype=np.float64),
        "world_z_m": np.asarray(rows["world_z_m"], dtype=np.float64),
        "heading_rad": np.asarray(rows["heading_rad"], dtype=np.float64),
    }
    return arrays, {
        "maximum_absolute_observation_error": maximum_observation_error,
        "maximum_absolute_action_error": maximum_action_error,
        "terminal_match": terminal_match,
        "stop_sign_world": stop_sign_world,
    }


def plot_bev(
    *,
    destination_png: Path,
    destination_pdf: Path,
    pose: Mapping[str, np.ndarray],
    target_data: Mapping[str, Mapping[str, Any]],
    tiles: Sequence[Any],
    scenario: Any,
    stop_sign_world: tuple[float, float] | None,
    signed_colour_quantile: float,
    png_dpi: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": png_dpi,
            "savefig.dpi": png_dpi,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
    panels = (
        (axes[0, 0], "v_cmd_mps", "signed", "(a) Signed IG: linear velocity"),
        (axes[0, 1], "v_cmd_mps", "group", "(b) Dominant belief group: linear velocity"),
        (axes[1, 0], "omega_cmd_rad_s", "signed", "(c) Signed IG: yaw rate"),
        (axes[1, 1], "omega_cmd_rad_s", "group", "(d) Dominant belief group: yaw rate"),
    )
    x = np.asarray(pose["world_x_m"], dtype=float)
    z = np.asarray(pose["world_z_m"], dtype=float)
    group_handles: list[Any] = []
    for axis, target, kind, title in panels:
        _draw_map(axis, tiles, Rectangle)
        _draw_landmarks(axis, scenario, stop_sign_world)
        axis.plot(x, z, color="#777777", linewidth=1.0, alpha=0.42, zorder=3)
        if kind == "signed":
            values = np.asarray(target_data[target]["signed_total"], dtype=float)
            limit = float(np.quantile(np.abs(values), signed_colour_quantile))
            limit = max(limit, np.finfo(float).eps)
            points = np.column_stack((x, z))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            segment_values = 0.5 * (values[:-1] + values[1:])
            collection = LineCollection(
                segments,
                cmap="RdBu_r",
                norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                linewidth=3.4,
                zorder=5,
            )
            collection.set_array(segment_values)
            axis.add_collection(collection)
            unit = "m/s" if target == "v_cmd_mps" else "rad/s"
            colourbar = figure.colorbar(collection, ax=axis, fraction=0.046, pad=0.02)
            colourbar.set_label(f"signed IG sum ({unit})", fontsize=8)
        else:
            group_result = target_data[target]["groups"]
            for group_index, group in enumerate(group_result.names):
                mask = group_result.dominant_index == group_index
                if not np.any(mask):
                    continue
                axis.scatter(
                    x[mask],
                    z[mask],
                    s=11,
                    color=GROUP_COLOURS[group],
                    edgecolors="none",
                    alpha=0.92,
                    zorder=5,
                )
            if not group_handles:
                group_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=GROUP_COLOURS[group],
                        markeredgecolor="none",
                        markersize=5,
                        label=GROUP_LABELS[group],
                    )
                    for group in group_result.names
                ]
        axis.scatter(x[0], z[0], marker="*", s=90, color="#111111", zorder=8)
        direction_index = min(45, len(x) - 1)
        axis.annotate(
            "CCW",
            xy=(x[direction_index], z[direction_index]),
            xytext=(x[0] - 0.22, z[0] + 0.18),
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.0},
            fontsize=8,
            zorder=9,
        )
        axis.set_title(title)
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world z (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.grid(False)
    figure.legend(
        handles=group_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
    )
    figure.suptitle(
        "Frozen C4 belief-PPO Integrated Gradients mapped onto the BEV route\n"
        "World pose and landmarks are evaluation-only; PPO/IG consume only the public 29D belief",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.savefig(destination_png, dpi=png_dpi, bbox_inches="tight")
    figure.savefig(destination_pdf, bbox_inches="tight")
    plt.close(figure)


def _draw_map(axis: Any, tiles: Sequence[Any], rectangle_type: Any) -> None:
    from math import comb

    for tile in tiles:
        i, j = tile.coords
        size = float(tile.tile_size_m)
        axis.add_patch(
            rectangle_type(
                (i * size, j * size),
                size,
                size,
                facecolor="#E6E6E6",
                edgecolor="#B0B0B0",
                linewidth=0.7,
                zorder=0,
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
        axis.plot(
            curve[:, 0],
            curve[:, 2],
            color="#555555",
            linestyle="--",
            linewidth=0.85,
            alpha=0.8,
            zorder=2,
        )


def _draw_landmarks(
    axis: Any, scenario: Any, stop_sign_world: tuple[float, float] | None
) -> None:
    pedestrian = scenario.pedestrian
    axis.plot(
        [pedestrian.path_start_world_x_m, pedestrian.path_end_world_x_m],
        [pedestrian.path_start_world_z_m, pedestrian.path_end_world_z_m],
        color="#E69F00",
        linestyle=":",
        linewidth=2.2,
        zorder=4,
    )
    crossing_mid_z = 0.5 * (
        pedestrian.path_start_world_z_m + pedestrian.path_end_world_z_m
    )
    axis.annotate(
        "Duckie crossing",
        xy=(pedestrian.path_start_world_x_m, crossing_mid_z),
        xytext=(pedestrian.path_start_world_x_m - 0.62, crossing_mid_z + 0.28),
        color="#9A6700",
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": "#E69F00", "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        zorder=10,
    )
    stop = scenario.stop_line
    half_width = 0.20
    across_x = np.sin(stop.route_heading_rad)
    across_z = np.cos(stop.route_heading_rad)
    axis.plot(
        [stop.world_x_m - half_width * across_x, stop.world_x_m + half_width * across_x],
        [stop.world_z_m - half_width * across_z, stop.world_z_m + half_width * across_z],
        color="#D55E00",
        linewidth=2.5,
        zorder=6,
    )
    axis.annotate(
        "stop line",
        xy=(stop.world_x_m, stop.world_z_m),
        xytext=(stop.world_x_m - 0.55, stop.world_z_m - 0.16),
        color="#9A3D00",
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": "#D55E00", "lw": 0.8},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        zorder=10,
    )
    if stop_sign_world is not None:
        axis.scatter(
            [stop_sign_world[0]],
            [stop_sign_world[1]],
            marker="8",
            s=70,
            color="#D55E00",
            edgecolor="#7A2700",
            linewidth=0.7,
            zorder=7,
        )


def write_sample_csv(
    path: Path,
    *,
    seed: np.ndarray,
    step: np.ndarray,
    pose: Mapping[str, np.ndarray],
    target_data: Mapping[str, Mapping[str, Any]],
) -> None:
    fieldnames = [
        "seed",
        "step",
        "evaluation_world_x_m",
        "evaluation_world_z_m",
        "evaluation_heading_rad",
        "v_cmd_signed_ig_mps",
        "v_cmd_dominant_group",
        "v_cmd_dominant_group_share",
        "omega_cmd_signed_ig_rad_s",
        "omega_cmd_dominant_group",
        "omega_cmd_dominant_group_share",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(seed.size):
            v_groups = target_data["v_cmd_mps"]["groups"]
            omega_groups = target_data["omega_cmd_rad_s"]["groups"]
            v_index = int(v_groups.dominant_index[row])
            omega_index = int(omega_groups.dominant_index[row])
            writer.writerow(
                {
                    "seed": int(seed[row]),
                    "step": int(step[row]),
                    "evaluation_world_x_m": float(pose["world_x_m"][row]),
                    "evaluation_world_z_m": float(pose["world_z_m"][row]),
                    "evaluation_heading_rad": float(pose["heading_rad"][row]),
                    "v_cmd_signed_ig_mps": float(
                        target_data["v_cmd_mps"]["signed_total"][row]
                    ),
                    "v_cmd_dominant_group": v_groups.names[v_index],
                    "v_cmd_dominant_group_share": float(v_groups.share[row, v_index]),
                    "omega_cmd_signed_ig_rad_s": float(
                        target_data["omega_cmd_rad_s"]["signed_total"][row]
                    ),
                    "omega_cmd_dominant_group": omega_groups.names[omega_index],
                    "omega_cmd_dominant_group_share": float(
                        omega_groups.share[row, omega_index]
                    ),
                }
            )


def _validate_ig_alignment(
    trajectory: Mapping[str, np.ndarray], ig: Mapping[str, np.ndarray]
) -> None:
    indices = np.asarray(ig["sample_index"], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= trajectory["seed"].shape[0]):
        raise ValueError("IG sample index is outside the frozen trajectory")
    if not np.array_equal(np.asarray(ig["seed"]), trajectory["seed"][indices]):
        raise ValueError("IG seed rows do not match the frozen trajectory")
    if not np.array_equal(np.asarray(ig["step"]), trajectory["step"][indices]):
        raise ValueError("IG step rows do not match the frozen trajectory")
    if not np.array_equal(np.asarray(ig["feature_names"]), trajectory["feature_names"]):
        raise ValueError("IG feature order does not match the frozen trajectory")


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
