"""Once-only held-out validation of the selected camera lane checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from duckie_pomdp.perception.lane_rgb_model import LanePoseMobileNet

from train_lane_rgb_v7 import LaneRGBDataset, evaluate, gate_result


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lane_rgb_train_v1.toml"
DATASET = ROOT / "datasets" / "lane_rgb_v1"
MODEL_DIR = ROOT / "artifacts" / "f10_ppo_visual_v7" / "lane_rgb_model"
OUTPUT = ROOT / "artifacts" / "f10_ppo_visual_v7" / "lane_rgb_final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", choices=("development", "final"), default="final")
    args = parser.parse_args()
    config_path = args.config.resolve()
    dataset = args.dataset.resolve()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    split = str(args.split)
    metrics_path = output / f"{split}_metrics.json"
    predictions_path = output / f"{split}_predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError(f"refusing to rerun lane RGB {split} split")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    checkpoint = ROOT / manifest["best_checkpoint"]
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != manifest["best_checkpoint_sha256"]:
        raise RuntimeError("selected lane RGB checkpoint hash drifted")
    rows = [
        row
        for row in csv.DictReader((dataset / "metadata.csv").open(encoding="utf-8"))
        if row["split"] == split
    ]
    device = torch.device(str(config["training"]["device"]))
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LanePoseMobileNet().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    loader = DataLoader(
        LaneRGBDataset(dataset, rows, config, train=False),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    scales = np.asarray(config["model"]["target_scales"], dtype=float)
    overall = evaluate(model, loader, device, scales)

    edge_minimum_m = float(
        config["selection"].get("strata", {}).get(
            "edge_minimum_absolute_lateral_m", 0.075
        )
    )
    predictions = infer_rows(
        model, loader, rows, device, scales, edge_minimum_m=edge_minimum_m
    )
    groups = {
        "by_map": grouped(predictions, "map"),
        "by_turn": grouped(predictions, "turn_family"),
        "by_lateral_region": grouped(predictions, "lateral_region"),
    }
    residual_sigmas = tuple(
        float(manifest["development_metrics"][name]["residual_sd"])
        for name in ("lateral", "heading", "curvature")
    )
    coverage = coverage_metrics(predictions, residual_sigmas)
    gate = extended_gate_result(overall, groups, config["selection"])
    result = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "seed_role": (
            "development/model-selection diagnostic"
            if split == "development"
            else "once-only held-out lane RGB final"
        ),
        "samples": len(rows),
        "checkpoint_sha256": actual_hash,
        "runtime_input": "front_rgb_only",
        "privileged_use": "held-out evaluation target after inference",
        "development_residual_sigmas": {
            "lateral_m": residual_sigmas[0],
            "heading_rad": residual_sigmas[1],
            "curvature_inv_m": residual_sigmas[2],
        },
        "overall": overall,
        **groups,
        "coverage_with_development_sigmas": coverage,
        "pre_registered_gate": gate["criteria"],
        "gate_pass": gate["passed"],
    }
    output.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    result["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def infer_rows(model, loader, metadata, device, scales, *, edge_minimum_m=0.075):
    result = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for images, normalized_truth in loader:
            predicted = model(images.to(device)).cpu().numpy() * scales
            truth = normalized_truth.numpy() * scales
            for index in range(predicted.shape[0]):
                source = metadata[offset + index]
                errors = predicted[index] - truth[index]
                errors[1] = np.arctan2(np.sin(errors[1]), np.cos(errors[1]))
                result.append(
                    {
                        "image_id": source["image_id"],
                        "seed": source["seed"],
                        "map": source["map"],
                        "turn_family": source.get(
                            "turn_family", _turn_family(float(truth[index, 2]))
                        ),
                        "pose_name": source.get("pose_name", "dynamic_trajectory"),
                        "lateral_region": _lateral_region(
                            float(truth[index, 0]),
                            float(edge_minimum_m),
                        ),
                        "gt_d": truth[index, 0],
                        "gt_phi": truth[index, 1],
                        "gt_kappa": truth[index, 2],
                        "pred_d": predicted[index, 0],
                        "pred_phi": predicted[index, 1],
                        "pred_kappa": predicted[index, 2],
                        "error_d": errors[0],
                        "error_phi": errors[1],
                        "error_kappa": errors[2],
                    }
                )
            offset += predicted.shape[0]
    return result


def grouped(rows, key):
    return {
        value: error_metrics([row for row in rows if row[key] == value])
        for value in sorted({row[key] for row in rows})
    }


def error_metrics(rows):
    errors = np.asarray(
        [[float(row[name]) for name in ("error_d", "error_phi", "error_kappa")] for row in rows]
    )
    result = {"n": len(rows)}
    for index, name in enumerate(("lateral", "heading", "curvature")):
        values = errors[:, index]
        result[name] = {
            "bias": float(np.mean(values)),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "residual_sd": float(np.std(values, ddof=1)),
        }
    excited = [row for row in rows if abs(float(row["gt_phi"])) >= 0.10]
    result["heading_sign_accuracy"] = float(
        np.mean(
            [
                np.sign(float(row["pred_phi"])) == np.sign(float(row["gt_phi"]))
                for row in excited
            ]
        )
    )
    return result


def coverage_metrics(rows, sigmas):
    errors = np.asarray(
        [[float(row[name]) for name in ("error_d", "error_phi", "error_kappa")] for row in rows]
    )
    result = {}
    for index, name in enumerate(("lateral", "heading", "curvature")):
        absolute_z = np.abs(errors[:, index]) / sigmas[index]
        result[name] = {
            "coverage_68": float(np.mean(absolute_z <= 1.0)),
            "coverage_95": float(np.mean(absolute_z <= 1.959963984540054)),
        }
    return result


def extended_gate_result(overall, groups, selection):
    base = gate_result(overall, selection)
    strata = selection.get("strata")
    if strata is None:
        return base
    right = groups["by_turn"].get("right")
    edge = groups["by_lateral_region"].get("edge")
    if right is None or edge is None:
        return {
            "criteria": {**base["criteria"], **dict(strata)},
            "checks": {"overall": base["passed"], "right_present": False, "edge_present": False},
            "passed": False,
        }
    checks = {
        "overall": bool(base["passed"]),
        "right_turn_heading": float(right["heading"]["rmse"])
        <= float(strata["maximum_right_turn_heading_rmse_rad"]),
        "edge_lateral": float(edge["lateral"]["rmse"])
        <= float(strata["maximum_edge_lateral_rmse_m"]),
    }
    return {
        "criteria": {**base["criteria"], **dict(strata)},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _lateral_region(lateral_m: float, edge_minimum_m: float = 0.075) -> str:
    return "edge" if abs(lateral_m) >= edge_minimum_m else "non_edge"


def _turn_family(curvature: float) -> str:
    if curvature < -0.75:
        return "right"
    if curvature > 0.75:
        return "left"
    return "straight"


if __name__ == "__main__":
    main()
