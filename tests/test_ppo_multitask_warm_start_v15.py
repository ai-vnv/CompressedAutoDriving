import json
from pathlib import Path

import numpy as np

from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    pretraining_source_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/f10_ppo_visual_objects_v15.toml"


def test_multitask_dataset_balances_retention_and_hazard_sources() -> None:
    protocol = load_ppo_curriculum_protocol(CONFIG)
    settings = protocol.raw["behavior_warm_start"]["c2"]
    path = (CONFIG.parent / settings["dataset"]).resolve()
    with np.load(path) as data:
        assert set(data.files) == {"observations", "actions", "weights"}
        assert data["observations"].shape == (51_200, 29)
        assert data["actions"].shape == (51_200, 2)
        assert np.all(np.isfinite(data["observations"]))
        assert np.all((data["actions"] >= -1.0) & (data["actions"] <= 1.0))


def test_multitask_manifest_is_policy_only_and_checkpoint_bound() -> None:
    manifest = json.loads((
        ROOT / "artifacts/f10_ppo_visual_objects_v15/c2/behavior_warm_start_manifest.json"
    ).read_text())
    assert manifest["uses_evaluation_gt"] is False
    assert manifest["anchor_rows"] == manifest["hazard_rows"] == 25_600
    assert manifest["source_checkpoint_stage"] == "c1"
    assert manifest["source_checkpoint_sha256"] == (
        "0e26ac28d8806140ff9544ecb094c20e850f66f83972544eb0dd8ac9b4d131b2"
    )
    assert abs(manifest["anchor_weight_mass"] - manifest["hazard_weight_mass"]) <= 0.1
    source = Path(manifest["source_csv"])
    assert file_sha256(source) == manifest["source_csv_sha256"]
    assert "evaluation_gt" not in source.read_text(encoding="utf-8").splitlines()[0]


def test_multitask_generators_are_frozen_sources() -> None:
    sources = set(pretraining_source_paths(load_ppo_curriculum_protocol(CONFIG)))
    assert "experiments/freeze_ppo_multitask_behavior_source.py" in sources
    assert "experiments/build_ppo_multitask_behavior_warm_start.py" in sources
