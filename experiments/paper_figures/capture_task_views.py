"""Scene renders for fig:task — BEV for C0/C1, third-person views for C2-C4.

Builds a raw Gym-Duckietown simulator per curriculum map (no perception stack),
places a static duckiebot at the ego pose so the robot itself is visible, moves
the scenario duckie mid-crossing where the curriculum calls for it, and renders
either the built-in top-down camera or a free perspective camera. Output PNGs
go to paper/figure_sources/scene_views/ for gen_fig_task.py.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DUCKIETOWN_HEADLESS", "1")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figure_sources" / "scene_views"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from gym_duckietown.objects import WorldObj
from gym_duckietown.objmesh import get_mesh
from gym_duckietown.simulator import Simulator

from duckie_pomdp.adapters.gym_duckietown import external_map_environment_type

ExternalMapSimulator = external_map_environment_type(Simulator)


def make_sim(map_name, w=960, h=720):
    return ExternalMapSimulator(
        map_name=map_name,
        max_steps=10_000,
        domain_rand=False,
        camera_width=w,
        camera_height=h,
        seed=180201,
    )


def add_static(sim, kind, mesh_name, pos, angle_rad, height):
    mesh = get_mesh(mesh_name)
    scale = height / mesh.max_coords[1]
    obj = {
        "kind": kind,
        "mesh": mesh,
        "pos": np.asarray(pos, dtype=float),
        "scale": scale,
        "optional": False,
        "static": True,
        "angle": angle_rad,
    }
    world_obj = WorldObj(obj, domain_rand=False, safety_radius_mult=1.0)
    sim.objects.append(world_obj)
    return world_obj


def render_top_down(sim):
    return sim._render_img(
        sim.camera_width, sim.camera_height,
        sim.multi_fbo, sim.final_fbo, sim.img_array, top_down=True,
    )


def render_from(sim, cam_pos, cam_yaw_rad, pitch_deg, cam_height):
    sim.cur_pos = np.asarray(cam_pos, dtype=float)
    sim.cur_angle = float(cam_yaw_rad)
    sim.cam_height = float(cam_height)
    sim.cam_angle = [float(pitch_deg), 0.0, 0.0]
    sim.cam_offset = np.array([0.0, 0.0, 0.0])
    return sim._render_img(
        sim.camera_width, sim.camera_height,
        sim.multi_fbo, sim.final_fbo, sim.img_array, top_down=False,
    )


def find_object(sim, kind):
    for item in sim.objects:
        if getattr(item, "kind", None) == kind:
            return item
    return None


def move_object(sim, obj, pos, angle_rad=None):
    obj.pos = np.asarray(pos, dtype=float)
    if angle_rad is not None:
        obj.angle = float(angle_rad)
        obj.y_rot = float(np.rad2deg(angle_rad))


def save(frame, name):
    Image.fromarray(frame).save(OUT / f"{name}.png")
    print("saved", OUT / f"{name}.png")


# Third-person scene definitions. World frame: x right, z down in the BEV,
# tile 0.585 m. The crossing duckie walks along +z at x=1.4625 (bottom
# straight, ego lane center z~2.19); the stop line sits at (2.1645, 1.4625)
# on the northbound right straight (ego heading -z), sign just off-lane.
SCENES = {
    "c2_crossing": dict(
        map_file="experiment_loop_objects_v1.yaml",
        ego=((0.85, 0.0, 2.19), 0.0),
        duckie_pos=(1.4625, 0.0, 2.165),
        remove_duckie=False,
        cam=dict(pos=(0.30, 0.0, 2.50), yaw=0.15, pitch=19, height=0.26),
    ),
    "c3_stop": dict(
        map_file="experiment_loop_stop_v2.yaml",
        ego=((2.19, 0.0, 1.72), 1.5707963267948966),
        duckie_pos=None,
        remove_duckie=True,
        cam=dict(pos=(2.10, 0.0, 2.20), yaw=1.40, pitch=19, height=0.30),
    ),
    "c4_combined": dict(
        map_file="experiment_loop_stop_v2.yaml",
        ego=((1.05, 0.0, 2.19), 0.0),
        duckie_pos=(1.4625, 0.0, 2.165),
        remove_duckie=False,
        cam=dict(pos=(2.55, 0.0, 2.80), yaw=2.16, pitch=17, height=0.34),
    ),
}

DUCKIE_CROSS_ANGLE = -1.5707963267948966  # facing +z, the crossing direction


def render_scene(name, spec):
    sim = make_sim(str(ROOT / "maps" / spec["map_file"]), 1440, 1080)
    sim.reset()
    duck = find_object(sim, "duckie")
    if spec["remove_duckie"] and duck is not None:
        duck.visible = False
        sim.objects.remove(duck)
    elif spec["duckie_pos"] is not None and duck is not None:
        move_object(sim, duck, spec["duckie_pos"], DUCKIE_CROSS_ANGLE)
    (ego_pos, ego_yaw) = spec["ego"]
    add_static(sim, "duckiebot", "duckiebot", ego_pos, ego_yaw, 0.12)
    cam = spec["cam"]
    frame = render_from(sim, cam["pos"], cam["yaw"], cam["pitch"], cam["height"])
    save(frame, name)
    sim.close()


# fig:phenotypes reconstruction — the A6 freeze on C3, seed 180201. Ego pose
# is derived from the recorded per-step telemetry (stop_line_distance_m), not
# staged: z = stop_line_z + d along the northbound approach, lane center x.
FREEZE_STEPS = [25, 125, 275, 2699]
STOP_LINE = (2.1645, 1.4625)  # from configs/scenario_experiment_loop_stop_v2.toml
FREEZE_CAM = dict(pos=(2.02, 0.0, 2.42), yaw=1.38, pitch=17, height=0.32)


def render_freeze():
    import json

    tel = (ROOT / "artifacts/f17_optimization_method_order_v1/telemetry"
           / "A6/c3/seed_180201/trace.npz")
    with np.load(tel, allow_pickle=False) as z:
        names = [str(v) for v in z["feature_names"]]
        assert names[9] == "stop_line_distance_m"
        v_cmd = np.asarray(z["physical_action"], dtype=np.float32)[:, 0]
        dist = np.asarray(z["public_physical_29d"], dtype=np.float32)[:, 9]

    records = []
    for step in FREEZE_STEPS:
        v, d = float(v_cmd[step]), float(dist[step])
        sim = make_sim(str(ROOT / "maps" / "experiment_loop_stop_v2.yaml"),
                       1440, 1080)
        sim.reset()
        duck = find_object(sim, "duckie")
        if duck is not None:
            duck.visible = False
            sim.objects.remove(duck)
        add_static(sim, "duckiebot", "duckiebot",
                   (STOP_LINE[0], 0.0, STOP_LINE[1] + d),
                   1.5707963267948966, 0.12)
        cam = FREEZE_CAM
        frame = render_from(sim, cam["pos"], cam["yaw"], cam["pitch"],
                            cam["height"])
        name = f"a6_freeze_step{step:04d}"
        save(frame, name)
        sim.close()
        records.append({"step": step, "v_cmd_mps": round(v, 4),
                        "stop_line_distance_m": round(d, 4),
                        "file": f"{name}.png"})
    meta = OUT / "a6_freeze_frames.json"
    meta.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print("saved", meta)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "debug"

    if which == "freeze":
        render_freeze()
        return

    if which == "calib":
        sim = make_sim(str(ROOT / "maps" / "experiment_loop_objects_v1.yaml"), 900, 900)
        sim.reset()
        add_static(sim, "duckiebot", "duckiebot", (1.17, 0.0, 2.19), 0.0, 0.12)
        sim.cur_pos = np.array([-5.0, 0.0, -5.0])  # park the real agent off-scene
        save(render_top_down(sim), "dbg_calib_angle0")
        import math
        save(render_from(sim, (1.80, 0.0, 2.19), math.pi, 12, 0.18),
             "dbg_calib_view_from_east")
        save(render_from(sim, (1.17, 0.0, 2.20), math.pi / 2, 88, 0.55),
             "dbg_calib_overhead")
        sim.close()
        return

    for name, spec in SCENES.items():
        if which in ("all", "scenes", name.split("_")[0]):
            render_scene(name, spec)

    if which in ("debug", "all", "c0", "c1"):
        # C0/C1: BEV of the full map, robot drawn on the lane
        for name, map_name in [("c0_smallloop", "small_loop"),
                               ("c1_experimentloop", "experiment_loop")]:
            if which in ("debug", "all", name[:2]):
                sim = make_sim(map_name, 900, 900)
                sim.reset()
                save(render_top_down(sim), f"{name}_bev")
                sim.close()

    if which in ("debug",):
        # Debug BEV of the object maps to read scene geometry
        for name, map_file in [
            ("dbg_c2map", str(ROOT / "maps" / "experiment_loop_objects_v1.yaml")),
            ("dbg_c34map", str(ROOT / "maps" / "experiment_loop_stop_v2.yaml")),
        ]:
            sim = make_sim(map_file, 900, 900)
            sim.reset()
            duck = find_object(sim, "duckie")
            if duck is not None:
                move_object(sim, duck, [1.4625, 0.0, 2.165])
            print(name, "objects:",
                  [(o.kind, np.round(np.asarray(o.pos, dtype=float), 3).tolist(),
                    round(float(o.y_rot), 1)) for o in sim.objects])
            print(name, "grid:", sim.grid_width, "x", sim.grid_height,
                  "tile", sim.road_tile_size)
            save(render_top_down(sim), name)
            sim.close()


if __name__ == "__main__":
    main()
