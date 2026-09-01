"""Generate the Version-1 simulator detection dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.dataset import generate_detection_dataset, load_dataset_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection_dataset_v1.toml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = generate_detection_dataset(
        load_dataset_config(args.config),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
