"""Emit the reproducible environment identity and seeded CUDA witness."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "f10_ppo_env_v1.json"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    torch.manual_seed(10_300)
    device = torch.device("cuda")
    left = torch.randn((8, 8), device=device)
    right = torch.randn((8, 8), device=device)
    witness = float((left @ right).sum().item())
    result = {
        "schema_version": 1,
        "environment_spec": str(SPEC.relative_to(ROOT)),
        "environment_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8],
        "canonical_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "raw_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "python_packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": package_version("ultralytics"),
            "wandb": package_version("wandb"),
            "stable_baselines3": package_version("stable-baselines3"),
        },
        "cuda_witness": {
            "seed": 10_300,
            "shape": [8, 8],
            "sum": witness,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
