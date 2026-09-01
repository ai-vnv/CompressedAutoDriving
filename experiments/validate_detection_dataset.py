"""Run deterministic metadata, label, split, and visual dataset QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.dataset import load_dataset_config, validate_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection_dataset_v1.toml"),
    )
    args = parser.parse_args()
    stats = validate_detection_dataset(load_dataset_config(args.config))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
