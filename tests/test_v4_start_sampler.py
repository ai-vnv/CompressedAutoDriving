"""Tests for F10-PPO v4 Task 2: loop-wide start randomisation.

v3 always spawned the ego on tile ``[1, 0]`` (``[native_start]``), jittered by
only a few centimetres. :mod:`duckie_pomdp.control.start_sampler` instead
samples the start *tile* across all of ``small_loop``'s drivable perimeter,
keeping v3's within-tile jitter numerically unchanged, gated behind
``[v4_changes].start_randomisation`` (default off -- a config without that
table, like every v3 config, must keep reproducing v3's single-tile
distribution bit-for-bit).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from duckie_pomdp.control.ppo_environment import PPOCurriculumEnvironment
from duckie_pomdp.control.start_sampler import (
    HEADING_JITTER_RAD,
    LATERAL_JITTER_M,
    LONGITUDINAL_JITTER_M,
    DrivableTile,
    LoopStartSampler,
    StartPose,
    _jittered_pose,
    lane_position,
    load_small_loop_tiles,
    resolve_start_randomisation_enabled,
)

ROOT = Path(__file__).resolve().parents[1]
V3_PROTOCOL_CONFIG = ROOT / "configs" / "f10_ppo_visual_v3.toml"

ANCHOR_TILE = (1, 0)
ANCHOR_HEADING_RAD = math.pi
ACCEPT_HEADING_RAD = math.radians(60.0)


@pytest.fixture(scope="module")
def small_loop_tiles() -> tuple[DrivableTile, ...]:
    return load_small_loop_tiles(
        map_name="small_loop",
        anchor_tile=ANCHOR_TILE,
        anchor_heading_rad=ANCHOR_HEADING_RAD,
    )


@pytest.fixture(scope="module")
def real_simulator():
    import os

    os.environ.setdefault("DUCKIETOWN_HEADLESS", "1")
    from gym_duckietown.envs import DuckietownEnv

    sim = DuckietownEnv(
        map_name="small_loop",
        max_steps=2,
        domain_rand=False,
        dynamics_rand=False,
        frame_rate=30,
        frame_skip=1,
        camera_width=64,
        camera_height=48,
        seed=0,
    )
    try:
        yield sim
    finally:
        sim.close()


# --- the brief's four required tests ----------------------------------------


def test_start_poses_cover_the_whole_loop_not_one_tile(small_loop_tiles):
    """Sample 500 starts; assert at least 80% of drivable tiles are represented.

    v3 used exactly one tile -- that is what this test exists to prevent.
    """

    sampler = LoopStartSampler(small_loop_tiles, rng_seed=47001)
    tiles_seen = {sampler.sample(i).tile for i in range(500)}
    coverage = len(tiles_seen) / len(small_loop_tiles)
    assert coverage >= 0.8, (
        f"only {len(tiles_seen)}/{len(small_loop_tiles)} tiles reached: "
        f"{sorted(tiles_seen)}"
    )


def test_each_sampling_cycle_visits_every_tile_once(small_loop_tiles):
    """Coverage is guaranteed per cycle, not left to random chance."""

    sampler = LoopStartSampler(small_loop_tiles, rng_seed=62000)
    expected = {tile.coords for tile in small_loop_tiles}
    for cycle in range(3):
        sampled = [
            sampler.sample(cycle * len(small_loop_tiles) + index).tile
            for index in range(len(small_loop_tiles))
        ]
        assert len(sampled) == len(set(sampled))
        assert set(sampled) == expected


def test_sampling_is_deterministic_from_seed_and_episode_index(small_loop_tiles):
    """Two samplers with the same seed produce identical pose sequences."""

    first = LoopStartSampler(small_loop_tiles, rng_seed=47002)
    second = LoopStartSampler(small_loop_tiles, rng_seed=47002)
    sequence_first = [first.sample(i) for i in range(50)]
    sequence_second = [second.sample(i) for i in range(50)]
    assert sequence_first == sequence_second

    # A different rng_seed must (almost certainly) diverge -- otherwise the
    # constructor argument would be a no-op.
    third = LoopStartSampler(small_loop_tiles, rng_seed=47003)
    sequence_third = [third.sample(i) for i in range(50)]
    assert sequence_third != sequence_first


def test_every_sampled_pose_is_on_the_drivable_surface_and_faces_the_loop(
    small_loop_tiles,
):
    """No start may begin off-road or facing backwards -- that would inject
    failures the policy cannot be blamed for."""

    sampler = LoopStartSampler(small_loop_tiles, rng_seed=47004)
    tiles_by_coords = {tile.coords: tile for tile in small_loop_tiles}
    for i in range(300):
        pose = sampler.sample(i)
        tile = tiles_by_coords[pose.tile]
        world_x = pose.tile[0] * tile.tile_size_m + pose.local_x_m
        world_z = pose.tile[1] * tile.tile_size_m + pose.local_z_m
        dist_m, angle_rad = lane_position(
            tile.curve_world, world_x, world_z, pose.heading_rad
        )
        assert abs(dist_m) <= 0.10, f"pose {pose} left the lane: dist={dist_m}"
        assert abs(angle_rad) <= ACCEPT_HEADING_RAD, (
            f"pose {pose} does not face the loop direction: "
            f"angle={math.degrees(angle_rad):.1f} deg"
        )


def test_within_tile_jitter_ranges_match_v3():
    """The jitter is calibrated and unchanged; only the tile choice is new."""

    native_start = tomllib.loads(V3_PROTOCOL_CONFIG.read_text())["native_start"]
    assert LONGITUDINAL_JITTER_M == tuple(native_start["start_longitudinal_jitter_m"])
    assert LATERAL_JITTER_M == tuple(native_start["start_lateral_jitter_m"])
    assert HEADING_JITTER_RAD == tuple(native_start["start_heading_jitter_rad"])


# --- map enumeration and direction resolution -------------------------------


def test_load_small_loop_tiles_enumerates_every_drivable_perimeter_tile(
    small_loop_tiles,
):
    """Cross-check against small_loop.yaml's documented 3x3 perimeter layout
    (asserting against the layout is explicitly fine; deriving it from a
    hardcoded list inside the production code is not -- see start_sampler.py)."""

    assert len(small_loop_tiles) == 8
    coords = {tile.coords for tile in small_loop_tiles}
    assert coords == {
        (0, 0), (1, 0), (2, 0),
        (0, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
    }
    kinds = sorted(tile.kind for tile in small_loop_tiles)
    assert kinds == ["curve_left"] * 4 + ["straight"] * 4
    for tile in small_loop_tiles:
        assert tile.tile_size_m == pytest.approx(0.585)


def test_forward_direction_makes_one_smooth_net_loop_rotation(small_loop_tiles):
    """Every tile's resolved heading must chain into a single closed loop
    with a net rotation of exactly one full turn -- proof the anchor-based
    direction resolution did not pick a mix of the two lane directions."""

    order = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    by_coords = {tile.coords: tile for tile in small_loop_tiles}
    total_delta = 0.0
    for a, b in zip(order, order[1:] + order[:1]):
        heading_a = by_coords[a].base_heading_rad
        heading_b = by_coords[b].base_heading_rad
        delta = math.atan2(math.sin(heading_b - heading_a), math.cos(heading_b - heading_a))
        assert abs(delta) <= math.radians(50.0), "no single step should turn more than a corner"
        total_delta += delta
    assert abs(abs(total_delta) - 2 * math.pi) < 1e-6


def test_anchor_tile_reproduces_v3s_lane_center_and_heading(small_loop_tiles):
    """The tile v3 always used must resolve to on-lane, forward-facing geometry."""

    anchor = next(tile for tile in small_loop_tiles if tile.coords == ANCHOR_TILE)
    assert anchor.base_heading_rad == pytest.approx(ANCHOR_HEADING_RAD, abs=1e-9)
    dist_m, angle_rad = lane_position(
        anchor.curve_world,
        ANCHOR_TILE[0] * anchor.tile_size_m + anchor.base_local_x_m,
        ANCHOR_TILE[1] * anchor.tile_size_m + anchor.base_local_z_m,
        anchor.base_heading_rad,
    )
    assert dist_m == pytest.approx(0.0, abs=1e-9)
    assert angle_rad == pytest.approx(0.0, abs=1e-9)


def test_lane_position_matches_the_real_simulators_get_lane_pos2(
    small_loop_tiles, real_simulator
):
    """The reimplemented validator must not drift from gym-duckietown's own
    get_lane_pos2 -- both are exercised against the same sampled poses."""

    sampler = LoopStartSampler(small_loop_tiles, rng_seed=47005)
    tiles_by_coords = {tile.coords: tile for tile in small_loop_tiles}
    for i in range(15):
        pose = sampler.sample(i)
        tile = tiles_by_coords[pose.tile]
        world_x = pose.tile[0] * tile.tile_size_m + pose.local_x_m
        world_z = pose.tile[1] * tile.tile_size_m + pose.local_z_m
        ours_dist, ours_angle = lane_position(
            tile.curve_world, world_x, world_z, pose.heading_rad
        )
        real = real_simulator.get_lane_pos2(
            [world_x, 0.0, world_z], pose.heading_rad
        )
        assert ours_dist == pytest.approx(real.dist, abs=1e-9)
        assert ours_angle == pytest.approx(real.angle_rad, abs=1e-9)


# --- rejection / validity ----------------------------------------------------


def test_pose_is_valid_rejects_a_pose_facing_backwards(small_loop_tiles):
    sampler = LoopStartSampler(small_loop_tiles, rng_seed=1)
    tile = next(t for t in small_loop_tiles if t.coords == ANCHOR_TILE)
    backwards = StartPose(
        tile=tile.coords,
        local_x_m=tile.base_local_x_m,
        local_z_m=tile.base_local_z_m,
        heading_rad=tile.base_heading_rad + math.pi,
    )
    assert sampler.pose_is_valid(tile, backwards) is False


def test_pose_is_valid_rejects_a_pose_far_off_the_lane_center(small_loop_tiles):
    sampler = LoopStartSampler(small_loop_tiles, rng_seed=1)
    tile = next(t for t in small_loop_tiles if t.coords == ANCHOR_TILE)
    # 0.15 m of pure lateral offset stays within the tile footprint but
    # exceeds max_lateral_offset_m (0.10 m default) -- exercises the
    # lane-distance branch specifically, not the tile-footprint branch.
    off_lane = _jittered_pose(tile, 0.0, 0.15, 0.0)
    assert sampler.pose_is_valid(tile, off_lane) is False


def test_pose_is_valid_accepts_the_unjittered_base_pose(small_loop_tiles):
    sampler = LoopStartSampler(small_loop_tiles, rng_seed=1)
    for tile in small_loop_tiles:
        base = StartPose(
            tile=tile.coords,
            local_x_m=tile.base_local_x_m,
            local_z_m=tile.base_local_z_m,
            heading_rad=tile.base_heading_rad,
        )
        assert sampler.pose_is_valid(tile, base) is True


def test_rejection_rate_is_low_over_many_samples(small_loop_tiles):
    """Reject-and-resample must not fire often enough to distort the tile
    distribution v3's calibrated jitter ranges are designed to stay inside."""

    sampler = LoopStartSampler(small_loop_tiles, rng_seed=47006)
    for i in range(500):
        sampler.sample(i)
    assert sampler.rejection_rate < 0.05, sampler.rejection_rate


# --- flag wiring via PPOCurriculumEnvironment -------------------------------


def test_resolve_start_randomisation_enabled_defaults_off_and_needs_explicit_true():
    assert resolve_start_randomisation_enabled({}) is False
    assert resolve_start_randomisation_enabled({"v4_changes": {}}) is False
    assert (
        resolve_start_randomisation_enabled(
            {"v4_changes": {"start_randomisation": False}}
        )
        is False
    )
    assert (
        resolve_start_randomisation_enabled(
            {"v4_changes": {"belief_uncertainty_refit": True}}  # a different switch
        )
        is False
    )
    assert (
        resolve_start_randomisation_enabled(
            {"v4_changes": {"start_randomisation": True}}
        )
        is True
    )


def test_v3_config_flag_off_reproduces_the_single_anchor_tile_across_episodes():
    env = PPOCurriculumEnvironment(V3_PROTOCOL_CONFIG, stage="c0", split="development")
    try:
        assert env._start_randomisation_enabled is False
        seen_tiles = set()
        for episode_index in range(20):
            seed = env._seeds[episode_index % len(env._seeds)]
            config, scenario = env._integration_config(seed, episode_index % len(env._seeds), episode_index)
            assert scenario is None
            seen_tiles.add(config.start_tile)
        assert seen_tiles == {(1, 0)}
    finally:
        env.close()


def _write_v4_shaped_protocol(name: str) -> Path:
    # Provenance paths in the TOML (e.g. "../maps/pomdp_v1.yaml") are
    # resolved relative to the config file's own directory, so the scratch
    # file has to live next to the real configs, not under tmp_path --
    # mirrors the pattern already established in
    # tests/test_v4_lane_belief_uncertainty.py's provenance probe.
    text = V3_PROTOCOL_CONFIG.read_text()
    text += "\n[v4_changes]\nstart_randomisation = true\n"
    scratch = ROOT / "configs" / name
    scratch.write_text(text)
    return scratch


def test_flag_on_randomises_the_small_loop_start_tile_across_episodes():
    scratch = _write_v4_shaped_protocol("_scratch_v4_start_sampler_probe_a.toml")
    env = PPOCurriculumEnvironment(scratch, stage="c0", split="development")
    try:
        assert env._start_randomisation_enabled is True
        seen_tiles = set()
        for episode_index in range(40):
            seed = env._seeds[episode_index % len(env._seeds)]
            config, scenario = env._integration_config(seed, episode_index % len(env._seeds), episode_index)
            assert scenario is None
            seen_tiles.add(config.start_tile)
        assert len(seen_tiles) > 1, "flag-on must not stay pinned to one tile"
    finally:
        env.close()
        scratch.unlink(missing_ok=True)


def test_flag_on_is_deterministic_from_seed_and_episode_index_end_to_end():
    scratch = _write_v4_shaped_protocol("_scratch_v4_start_sampler_probe_b.toml")
    env_a = PPOCurriculumEnvironment(scratch, stage="c0", split="development")
    env_b = PPOCurriculumEnvironment(scratch, stage="c0", split="development")
    try:
        for episode_index in range(10):
            seed = env_a._seeds[episode_index % len(env_a._seeds)]
            scenario_index = episode_index % len(env_a._seeds)
            config_a, _ = env_a._integration_config(seed, scenario_index, episode_index)
            config_b, _ = env_b._integration_config(seed, scenario_index, episode_index)
            assert config_a.start_tile == config_b.start_tile
            assert config_a.start_pose == config_b.start_pose
    finally:
        env_a.close()
        env_b.close()
        scratch.unlink(missing_ok=True)


def test_experiment_loop_stage_is_loop_wide_and_covers_right_curves():
    """C1 must expose the policy to every tile, including both right curves."""

    scratch = _write_v4_shaped_protocol("_scratch_v4_start_sampler_probe_c.toml")
    env = PPOCurriculumEnvironment(scratch, stage="c1", split="development")
    try:
        assert env._start_randomisation_enabled is True
        assert env.stage.map_name == "experiment_loop"
        starts = []
        for episode_index in range(24):
            seed = env._seeds[episode_index % len(env._seeds)]
            config, _ = env._integration_config(
                seed, episode_index % len(env._seeds), episode_index
            )
            starts.append(config.start_tile)
        tiles = load_small_loop_tiles(
            map_name="experiment_loop",
            anchor_tile=ANCHOR_TILE,
            anchor_heading_rad=ANCHOR_HEADING_RAD,
        )
        by_coords = {tile.coords: tile for tile in tiles}
        assert set(starts) == set(by_coords)
        assert any(by_coords[coords].kind == "curve_right" for coords in starts)
    finally:
        env.close()
        scratch.unlink(missing_ok=True)
