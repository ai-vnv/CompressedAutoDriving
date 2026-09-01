"""Loop-wide start-pose sampling for native gym-duckietown loop maps (F10-PPO v4 Task 2).

v3's ``[native_start]`` always spawns the ego on the *same* tile
(``start_tile = [1, 0]``), jittered by only a few centimetres. Over an entire
training run the policy therefore only ever experiences ~22% of the
``small_loop`` perimeter -- it cannot learn to hold a curve it never reaches.

This module enumerates every drivable tile of a loop map (read from the
installed ``gym-duckietown``/``duckietown-world`` package, never hardcoded)
and samples the *start tile* itself, in addition to v3's existing
within-tile jitter, which is kept numerically unchanged.

Two pieces:

* :func:`load_small_loop_tiles` -- resolves every drivable tile of a loop
  map to a canonical, lane-centred base pose (position + heading, facing the
  loop direction). Direction is not guessed: exactly one known-good anchor
  (v3's own ``[native_start]`` tile + heading, already verified in
  production) is used to pick the correct one of each tile's two lane
  curves, and that choice is then propagated to every other drivable tile by
  chaining bezier curve endpoints tile-to-tile around the loop -- a purely
  geometric operation that requires no per-tile knowledge of the map layout.
* :class:`LoopStartSampler` -- given the resolved tiles and a seed, samples a
  tile uniformly and jitters within it using v3's exact calibrated ranges
  (``±0.030`` m longitudinal, ``±0.020`` m lateral, ``±0.040`` rad heading),
  rotated into that tile's own lane-tangent frame. A pose that would leave
  the drivable lane surface or point against the loop direction is rejected
  and resampled.

**Runtime wiring.** :func:`resolve_start_randomisation_enabled` is the single
switch point: it reads ``[v4_changes].start_randomisation`` from a loaded
protocol's raw TOML and returns ``False`` (off) unless the flag is
explicitly ``true`` -- a missing ``[v4_changes]`` table, a missing key, or an
explicit ``false`` are all "off". This is what lets
``configs/f10_ppo_visual_v3.toml`` (which has no ``[v4_changes]`` table at
all) keep reproducing its single-tile start distribution bit-for-bit through
the exact same runtime code path used by v4.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: v3's calibrated within-tile jitter ranges (``configs/f10_ppo_visual_v3.toml``
#: ``[native_start]``). Unchanged by this task -- only the tile choice is new.
LONGITUDINAL_JITTER_M: tuple[float, float] = (-0.030, 0.030)
LATERAL_JITTER_M: tuple[float, float] = (-0.020, 0.020)
HEADING_JITTER_RAD: tuple[float, float] = (-0.040, 0.040)

_TILE_CACHE: dict[tuple[str, tuple[int, int], float], tuple["DrivableTile", ...]] = {}


@dataclass(frozen=True)
class DrivableTile:
    """One drivable tile of a loop map, resolved to its forward (loop-direction) lane.

    ``base_local_x_m``/``base_local_z_m`` are offsets from the tile's own
    ``(i, j) * tile_size_m`` origin -- the same convention gym-duckietown's
    ``Simulator.reset`` uses for ``start_pose`` -- and sit exactly on the
    lane centreline. ``base_heading_rad`` is the lane tangent's heading at
    that point, facing the loop's direction of travel.
    """

    coords: tuple[int, int]
    kind: str
    tile_size_m: float
    base_local_x_m: float
    base_local_z_m: float
    base_heading_rad: float
    curve_world: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class StartPose:
    """One sampled ego start pose, ready to feed ``GymDuckietownConfig``."""

    tile: tuple[int, int]
    local_x_m: float
    local_z_m: float
    heading_rad: float


def resolve_start_randomisation_enabled(protocol_raw: dict[str, Any]) -> bool:
    """``[v4_changes].start_randomisation`` reader -- ``False`` unless explicitly ``true``."""

    v4_changes = protocol_raw.get("v4_changes", {})
    return bool(v4_changes.get("start_randomisation", False))


def load_small_loop_tiles(
    *,
    map_name: str,
    anchor_tile: tuple[int, int],
    anchor_heading_rad: float,
) -> tuple[DrivableTile, ...]:
    """Enumerate ``map_name``'s drivable tiles and resolve each to its forward lane.

    Nothing about the tile layout is hardcoded: the map is loaded through
    gym-duckietown exactly as the real simulator would, and
    ``sim.drivable_tiles`` (with each tile's own bezier ``curves``) is the
    only source of tile geometry. The *direction* of travel is resolved from
    a single anchor pose (``anchor_tile``, ``anchor_heading_rad`` -- v3's own
    ``[native_start]`` tile and heading, already verified correct in
    production) and propagated to every other tile by chaining curve
    endpoints: each tile's forward-lane curve ends almost exactly where the
    next tile's forward-lane curve begins (floating point coincident), so
    following that chain around the loop assigns a consistent forward
    direction to all tiles without ever naming one.
    """

    anchor_tile = (int(anchor_tile[0]), int(anchor_tile[1]))
    cache_key = (map_name, anchor_tile, float(anchor_heading_rad))
    cached = _TILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    os.environ.setdefault("DUCKIETOWN_HEADLESS", "1")
    from gym_duckietown.envs import DuckietownEnv
    from gym_duckietown.graphics import bezier_point, bezier_tangent
    from duckie_pomdp.adapters.gym_duckietown import external_map_environment_type

    environment_type = (
        external_map_environment_type(DuckietownEnv)
        if Path(map_name).is_file()
        else DuckietownEnv
    )
    simulator = environment_type(
        map_name=map_name,
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
        tile_size_m = float(simulator.road_tile_size)
        raw_tiles = {tuple(int(v) for v in t["coords"]): t for t in simulator.drivable_tiles}
        if anchor_tile not in raw_tiles:
            raise ValueError(
                f"anchor tile {anchor_tile} is not a drivable tile of map {map_name!r}"
            )

        forward_curve = _resolve_forward_curves(
            raw_tiles,
            anchor_tile=anchor_tile,
            anchor_heading_rad=float(anchor_heading_rad),
            tile_size_m=tile_size_m,
            bezier_point=bezier_point,
            bezier_tangent=bezier_tangent,
        )

        tiles: list[DrivableTile] = []
        for coords, tile in raw_tiles.items():
            curve_index = forward_curve[coords]
            control_points = np.asarray(tile["curves"][curve_index], dtype=float)
            point = bezier_point(control_points, 0.5)
            tangent = bezier_tangent(control_points, 0.5)
            heading = math.atan2(-float(tangent[2]), float(tangent[0]))
            i, j = coords
            tiles.append(
                DrivableTile(
                    coords=coords,
                    kind=str(tile["kind"]),
                    tile_size_m=tile_size_m,
                    base_local_x_m=float(point[0]) - i * tile_size_m,
                    base_local_z_m=float(point[2]) - j * tile_size_m,
                    base_heading_rad=heading,
                    curve_world=tuple(
                        (float(row[0]), float(row[1]), float(row[2]))
                        for row in control_points
                    ),
                )
            )
    finally:
        simulator.close()

    result = tuple(sorted(tiles, key=lambda tile: tile.coords))
    _TILE_CACHE[cache_key] = result
    return result


def _resolve_forward_curves(
    raw_tiles: dict[tuple[int, int], dict[str, Any]],
    *,
    anchor_tile: tuple[int, int],
    anchor_heading_rad: float,
    tile_size_m: float,
    bezier_point: Any,
    bezier_tangent: Any,
) -> dict[tuple[int, int], int]:
    """Chain curve endpoints around the loop, anchored at one known-good pose."""

    anchor_curves = np.asarray(raw_tiles[anchor_tile]["curves"], dtype=float)
    anchor_dir = np.array(
        [math.cos(anchor_heading_rad), 0.0, -math.sin(anchor_heading_rad)]
    )
    anchor_alignment = [
        float(np.dot(bezier_tangent(anchor_curves[k], 0.5), anchor_dir))
        for k in range(anchor_curves.shape[0])
    ]
    anchor_curve_index = int(np.argmax(anchor_alignment))
    if anchor_alignment[anchor_curve_index] <= 0.0:
        raise ValueError(
            f"no curve at anchor tile {anchor_tile} points along "
            f"anchor_heading_rad={anchor_heading_rad!r}"
        )

    forward_curve = {anchor_tile: anchor_curve_index}
    tolerance_m = max(1e-4, 1e-3 * tile_size_m)
    current_coords, current_curve = anchor_tile, anchor_curve_index
    for _ in range(len(raw_tiles) + 1):
        control_points = np.asarray(
            raw_tiles[current_coords]["curves"][current_curve], dtype=float
        )
        end_point = bezier_point(control_points, 1.0)
        best_coords: tuple[int, int] | None = None
        best_curve = -1
        best_distance = math.inf
        for coords, tile in raw_tiles.items():
            curves = np.asarray(tile["curves"], dtype=float)
            for k in range(curves.shape[0]):
                start_point = bezier_point(curves[k], 0.0)
                distance = float(np.linalg.norm(start_point - end_point))
                if distance < best_distance:
                    best_distance, best_coords, best_curve = distance, coords, k
        if best_coords is None or best_distance > tolerance_m:
            raise RuntimeError(
                f"loop chaining broke leaving tile {current_coords}: nearest curve "
                f"start is {best_distance:.6f} m away (tolerance {tolerance_m:.6f} m)"
            )
        if best_coords == anchor_tile:
            break
        forward_curve[best_coords] = best_curve
        current_coords, current_curve = best_coords, best_curve
    else:
        raise RuntimeError("loop chaining did not close back onto the anchor tile")

    missing = set(raw_tiles) - set(forward_curve)
    if missing:
        raise RuntimeError(f"loop chaining never reached tiles: {sorted(missing)}")
    return forward_curve


class LoopStartSampler:
    """Deterministic, loop-wide start-pose sampler for one native loop map.

    Samples every drivable tile exactly once per shuffled cycle and jitters within it using
    v3's exact calibrated ranges (:data:`LONGITUDINAL_JITTER_M`,
    :data:`LATERAL_JITTER_M`, :data:`HEADING_JITTER_RAD`), rotated into that
    tile's own lane-tangent frame so "longitudinal"/"lateral" keep their
    physical meaning on curved tiles too. Any candidate pose that would leave
    the drivable lane surface, or face more than ``max_heading_offset_rad``
    away from the loop direction, is rejected and resampled.
    """

    def __init__(
        self,
        tiles: Sequence[DrivableTile],
        rng_seed: int,
        *,
        max_lateral_offset_m: float = 0.10,
        max_heading_offset_rad: float = math.radians(60.0),
        max_resample_attempts: int = 64,
    ) -> None:
        if not tiles:
            raise ValueError("LoopStartSampler requires at least one drivable tile")
        if max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")
        self._tiles = tuple(tiles)
        self._rng_seed = int(rng_seed)
        self._max_lateral_offset_m = float(max_lateral_offset_m)
        self._max_heading_offset_rad = float(max_heading_offset_rad)
        self._max_resample_attempts = int(max_resample_attempts)
        self.attempts = 0
        self.rejections = 0

    @property
    def tiles(self) -> tuple[DrivableTile, ...]:
        return self._tiles

    @property
    def rng_seed(self) -> int:
        return self._rng_seed

    def sample(self, episode_index: int) -> StartPose:
        episode_index = int(episode_index)
        cycle, within_cycle = divmod(episode_index, len(self._tiles))
        order_rng = np.random.default_rng((self._rng_seed, cycle, 0xC0DEC0DE))
        tile_index = int(order_rng.permutation(len(self._tiles))[within_cycle])
        tile = self._tiles[tile_index]
        rng = np.random.default_rng((self._rng_seed, episode_index, 0x51A7))
        for _ in range(self._max_resample_attempts):
            self.attempts += 1
            longitudinal_m = float(rng.uniform(*LONGITUDINAL_JITTER_M))
            lateral_m = float(rng.uniform(*LATERAL_JITTER_M))
            heading_jitter_rad = float(rng.uniform(*HEADING_JITTER_RAD))
            pose = _jittered_pose(tile, longitudinal_m, lateral_m, heading_jitter_rad)
            if self.pose_is_valid(tile, pose):
                return pose
            self.rejections += 1
        raise RuntimeError(
            f"LoopStartSampler found no valid pose for episode_index={episode_index} "
            f"after {self._max_resample_attempts} attempts "
            f"(rejection rate so far: {self.rejection_rate:.3f})"
        )

    @property
    def rejection_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.rejections / self.attempts

    def pose_is_valid(self, tile: DrivableTile, pose: StartPose) -> bool:
        """Reject a pose off this tile's footprint, off the lane, or facing backwards."""

        if tile.coords != pose.tile:
            return False
        if not (0.0 <= pose.local_x_m <= tile.tile_size_m):
            return False
        if not (0.0 <= pose.local_z_m <= tile.tile_size_m):
            return False
        world_x = tile.coords[0] * tile.tile_size_m + pose.local_x_m
        world_z = tile.coords[1] * tile.tile_size_m + pose.local_z_m
        dist_m, angle_rad = lane_position(tile.curve_world, world_x, world_z, pose.heading_rad)
        if abs(dist_m) > self._max_lateral_offset_m:
            return False
        if abs(angle_rad) > self._max_heading_offset_rad:
            return False
        return True


def _jittered_pose(
    tile: DrivableTile,
    longitudinal_m: float,
    lateral_m: float,
    heading_jitter_rad: float,
) -> StartPose:
    """Apply v3-style jitter in the tile's own tangent/right frame."""

    dir_x = math.cos(tile.base_heading_rad)
    dir_z = -math.sin(tile.base_heading_rad)
    right_x = -dir_z
    right_z = dir_x
    local_x_m = tile.base_local_x_m + longitudinal_m * dir_x + lateral_m * right_x
    local_z_m = tile.base_local_z_m + longitudinal_m * dir_z + lateral_m * right_z
    heading_rad = _wrap_angle(tile.base_heading_rad + heading_jitter_rad)
    return StartPose(
        tile=tile.coords,
        local_x_m=local_x_m,
        local_z_m=local_z_m,
        heading_rad=heading_rad,
    )


def lane_position(
    curve_world: Sequence[Sequence[float]],
    world_x: float,
    world_z: float,
    heading_rad: float,
) -> tuple[float, float]:
    """Signed lateral distance and heading offset from ``curve_world``'s tangent.

    Mirrors ``gym_duckietown.simulator.Simulator.get_lane_pos2`` exactly (same
    formulas, applied to one already-known curve instead of the closest of a
    tile-grid lookup), using gym-duckietown's own ``bezier_closest`` /
    ``bezier_point`` / ``bezier_tangent`` so the curve math itself is never
    duplicated.
    """

    from gym_duckietown.graphics import bezier_closest, bezier_point, bezier_tangent

    control_points = np.asarray(curve_world, dtype=float)
    position = np.array([world_x, 0.0, world_z])
    t = bezier_closest(control_points, position)
    point = bezier_point(control_points, t)
    tangent = bezier_tangent(control_points, t)

    direction = np.array([math.cos(heading_rad), 0.0, -math.sin(heading_rad)])
    dot_direction = float(np.clip(np.dot(direction, tangent), -1.0, 1.0))

    position_vector = position - point
    up = np.array([0.0, 1.0, 0.0])
    right_vector = np.cross(tangent, up)
    signed_distance = float(np.dot(position_vector, right_vector))

    angle_rad = math.acos(dot_direction)
    if np.dot(direction, right_vector) < 0:
        angle_rad *= -1
    return signed_distance, angle_rad


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
