from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_competence_dataset_is_camera_only_and_seed_disjoint() -> None:
    dataset = ROOT / "datasets" / "lane_rgb_competence_v9"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_input"] == "front_rgb_only"
    assert manifest["direction"] == "counter-clockwise"
    assert all(not values for values in manifest["seed_overlaps"].values())
    assert min(manifest["longitudinal_offsets_m"]) < 0.0
    assert max(manifest["longitudinal_offsets_m"]) > 0.0
    lateral = [abs(float(row["lateral_m"])) for row in manifest["poses"]]
    heading = [abs(float(row["heading_rad"])) for row in manifest["poses"]]
    assert max(lateral) >= 0.09
    assert max(heading) >= 0.35


def test_combined_training_is_turn_balanced_without_pixel_flips() -> None:
    dataset = ROOT / "datasets" / "lane_rgb_combined_v9"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["horizontal_flip_forbidden"] is True
    assert manifest["dynamic_final_excluded"] is True
    assert len(set(manifest["logical_training_counts_by_turn"].values())) == 1
    assert all(not values for values in manifest["split_overlaps"].values())
    rows = list(csv.DictReader((dataset / "metadata.csv").open(encoding="utf-8")))
    assert rows
    assert all((dataset / row["image_path"]).is_file() for row in rows)


def test_v9_training_preprocessing_preserves_lane_boundary_semantics() -> None:
    path = ROOT / "configs" / "lane_rgb_train_v3_competence.toml"
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    assert config["model"]["crop_top_fraction"] == 0.25
    assert config["training"]["horizontal_flip_probability"] == 0.0
    assert config["selection"]["strata"]["maximum_right_turn_heading_rmse_rad"] <= 0.14
    assert config["selection"]["strata"]["maximum_edge_lateral_rmse_m"] <= 0.03
