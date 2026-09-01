"""Create one authenticated W&B preflight run without exposing credentials."""

from __future__ import annotations

import json
from pathlib import Path

import wandb

from duckie_pomdp.control.f10_protocol import file_sha256

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f10_sac_v1.toml"
OUTPUT = ROOT / "artifacts" / "f10" / "wandb_online_preflight.json"


def main() -> None:
    with CONFIG.open("rb") as stream:
        settings = tomllib.load(stream)["wandb"]
    run = wandb.init(
        entity=str(settings["entity"]),
        project=str(settings["project"]),
        group=str(settings["group"]),
        job_type="preflight",
        name="f10-online-auth-preflight",
        mode="online",
        reinit="finish_previous",
    )
    run.log({"preflight/connected": 1}, step=0)
    result = {
        "schema_version": 1,
        "verified": True,
        "f10_config_sha256": file_sha256(CONFIG),
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_url": run.url,
        "credential_storage": "user_netrc_not_project",
    }
    run.finish()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
