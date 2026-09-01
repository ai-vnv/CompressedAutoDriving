"""Verify the frozen F10-PPO W&B destination without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import load_ppo_curriculum_protocol


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "f10_ppo_v1.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = load_ppo_curriculum_protocol(config_path)
    settings = protocol.raw["wandb"]
    run = wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="preflight",
        name="f10-ppo-online-preflight",
        mode="online",
        config={
            "config_sha256": file_sha256(config_path),
            "purpose": "online_auth_destination_preflight",
        },
    )
    run.log({"preflight/ok": 1}, step=0)
    result = {
        "schema_version": 1,
        "run_id": run.id,
        "url": run.url,
        "entity": run.entity,
        "project": run.project,
        "group": str(settings["group"]),
        "config_sha256": file_sha256(config_path),
        "state": "finished",
    }
    run.finish()
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
