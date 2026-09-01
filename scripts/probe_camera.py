"""Read one simulator frame and report only camera-contract facts."""

from __future__ import annotations

import argparse
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="small_loop")
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from gym_duckietown.envs import DuckietownEnv

    environment = DuckietownEnv(
        map_name=args.map,
        domain_rand=False,
        seed=args.seed,
    )
    try:
        result = environment.reset()
        frame = result[0] if isinstance(result, tuple) else result
        rgb = np.asarray(frame)
        report = {
            "map": args.map,
            "seed": args.seed,
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
            "minimum": int(rgb.min()),
            "maximum": int(rgb.max()),
            "nonzero_fraction": float(np.mean(rgb != 0)),
            "valid_rgb": bool(
                rgb.ndim == 3
                and rgb.shape[2] == 3
                and rgb.dtype == np.uint8
            ),
        }
        print(json.dumps(report, indent=2))
        if not report["valid_rgb"]:
            raise SystemExit(1)
    finally:
        environment.close()


if __name__ == "__main__":
    main()

