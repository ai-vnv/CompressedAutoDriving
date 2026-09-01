"""Train the frozen camera-only lane-pose regressor on calibration RGB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from duckie_pomdp.perception.lane_rgb_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LanePoseMobileNet,
    OUTPUT_ORDER,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lane_rgb_train_v1.toml"


class LaneRGBDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], config: dict, *, train: bool) -> None:
        self.root = root
        self.rows = rows
        self.image_size = int(config["model"]["image_size_px"])
        self.crop_top_fraction = float(config["model"].get("crop_top_fraction", 0.0))
        self.scales = torch.tensor(config["model"]["target_scales"], dtype=torch.float32)
        training = config["training"]
        self.train = train
        self.flip_probability = float(training["horizontal_flip_probability"])
        self.jitter = ColorJitter(
            brightness=float(training["brightness_jitter"]),
            contrast=float(training["contrast_jitter"]),
            saturation=float(training["saturation_jitter"]),
            hue=0.0,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        target = torch.tensor(
            (
                float(row["gt_lateral_error_m"]),
                float(row["gt_heading_error_rad"]),
                float(row["gt_curvature_inv_m"]),
            ),
            dtype=torch.float32,
        )
        if self.train:
            image = self.jitter(image)
            if random.random() < self.flip_probability:
                image = TF.hflip(image)
                target = -target
        crop_rows = int(round(image.height * self.crop_top_fraction))
        if crop_rows:
            image = TF.crop(
                image,
                top=crop_rows,
                left=0,
                height=image.height - crop_rows,
                width=image.width,
            )
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, target / self.scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    dataset_root = (config_path.parent / str(config["dataset"])).resolve()
    output = (config_path.parent / str(config["output"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite lane model run at {output}")
    output.mkdir(parents=True)

    rows = list(csv.DictReader((dataset_root / "metadata.csv").open(encoding="utf-8")))
    train_rows = [row for row in rows if row["split"] == "train"]
    development_rows = [row for row in rows if row["split"] == "development"]
    if not train_rows or not development_rows:
        raise RuntimeError("lane RGB train/development split is empty")

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(str(config["training"]["device"]))

    model = LanePoseMobileNet().to(device)
    initial_checkpoint = config["model"].get("initial_checkpoint")
    if initial_checkpoint is not None:
        pretrained_path = Path(str(initial_checkpoint)).resolve()
        expected_initial_hash = str(config["model"]["initial_checkpoint_sha256"])
        if sha256(pretrained_path) != expected_initial_hash:
            raise RuntimeError("lane model initial checkpoint hash mismatch")
        initial_payload = torch.load(
            pretrained_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial_payload["model_state_dict"], strict=True)
        initialization = "lane_checkpoint"
    else:
        pretrained_path = Path(str(config["model"]["pretrained_weights"])).resolve()
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        current = model.backbone.state_dict()
        compatible = {
            name: value
            for name, value in pretrained.items()
            if name in current and current[name].shape == value.shape
        }
        model.backbone.load_state_dict(compatible, strict=False)
        initialization = "imagenet_mobilenet_v3_small"

    train_loader = DataLoader(
        LaneRGBDataset(dataset_root, train_rows, config, train=True),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    development_loader = DataLoader(
        LaneRGBDataset(dataset_root, development_rows, config, train=False),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    beta = float(config["training"]["smooth_l1_beta"])
    scales = np.asarray(config["model"]["target_scales"], dtype=float)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_metrics: dict[str, object] | None = None
    best_epoch = -1
    best_path = output / "best.pt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = F.smooth_l1_loss(predictions, targets, beta=beta)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, development_loader, device, scales)
        score = selection_score(metrics, config["selection"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            **flatten_metrics(metrics),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                {
                    "schema_version": 1,
                    "architecture": LanePoseMobileNet.architecture,
                    "output_order": OUTPUT_ORDER,
                    "target_scales": tuple(float(value) for value in scales),
                    "preprocessing": {
                        "crop_top_fraction": float(
                            config["model"].get("crop_top_fraction", 0.0)
                        )
                    },
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )

    with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    assert best_metrics is not None
    gate = gate_result(best_metrics, config["selection"])
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(dataset_root / "manifest.json"),
        "pretrained_weights": str(pretrained_path),
        "pretrained_weights_sha256": sha256(pretrained_path),
        "initialization": initialization,
        "torch_version": torch.__version__,
        "torchvision_architecture": LanePoseMobileNet.architecture,
        "device": str(device),
        "train_samples": len(train_rows),
        "development_samples": len(development_rows),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best_path),
        "development_metrics": best_metrics,
        "pre_registered_gate": gate["criteria"],
        "gate_pass": gate["passed"],
        "final_split_consumed": False,
        "runtime_input": "front_rgb_only",
        "preprocessing": {
            "crop_top_fraction": float(
                config["model"].get("crop_top_fraction", 0.0)
            ),
            "horizontal_flip_probability": float(
                config["training"]["horizontal_flip_probability"]
            ),
        },
    }
    (output / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def evaluate(model, loader, device, scales: np.ndarray) -> dict[str, object]:
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for images, normalized_targets in loader:
            predictions.append(model(images.to(device)).cpu().numpy() * scales)
            targets.append(normalized_targets.numpy() * scales)
    predicted = np.concatenate(predictions)
    truth = np.concatenate(targets)
    errors = predicted - truth
    errors[:, 1] = np.arctan2(np.sin(errors[:, 1]), np.cos(errors[:, 1]))
    names = ("lateral", "heading", "curvature")
    result: dict[str, object] = {"n": int(errors.shape[0])}
    for index, name in enumerate(names):
        values = errors[:, index]
        result[name] = {
            "bias": float(np.mean(values)),
            "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "residual_sd": float(np.std(values, ddof=1)),
        }
    excited = np.abs(truth[:, 1]) >= 0.10
    result["heading_sign_accuracy"] = float(
        np.mean(np.sign(predicted[excited, 1]) == np.sign(truth[excited, 1]))
    )
    result["heading_correlation"] = float(np.corrcoef(predicted[:, 1], truth[:, 1])[0, 1])
    return result


def selection_score(metrics: dict[str, object], selection: dict) -> float:
    return float(
        metrics["lateral"]["rmse"] / float(selection["score_lateral_scale_m"])
        + metrics["heading"]["rmse"] / float(selection["score_heading_scale_rad"])
        + float(selection["score_curvature_weight"])
        * metrics["curvature"]["rmse"]
        / float(selection["score_curvature_scale_inv_m"])
    )


def gate_result(metrics: dict[str, object], selection: dict) -> dict[str, object]:
    criteria = {
        "maximum_lateral_rmse_m": float(selection["maximum_lateral_rmse_m"]),
        "maximum_heading_rmse_rad": float(selection["maximum_heading_rmse_rad"]),
        "minimum_heading_sign_accuracy": float(selection["minimum_heading_sign_accuracy"]),
        "maximum_curvature_rmse_inv_m": float(selection["maximum_curvature_rmse_inv_m"]),
    }
    passed = bool(
        metrics["lateral"]["rmse"] <= criteria["maximum_lateral_rmse_m"]
        and metrics["heading"]["rmse"] <= criteria["maximum_heading_rmse_rad"]
        and metrics["heading_sign_accuracy"] >= criteria["minimum_heading_sign_accuracy"]
        and metrics["curvature"]["rmse"] <= criteria["maximum_curvature_rmse_inv_m"]
    )
    return {"criteria": criteria, "passed": passed}


def flatten_metrics(metrics: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "development_n": metrics["n"],
        "heading_sign_accuracy": metrics["heading_sign_accuracy"],
        "heading_correlation": metrics["heading_correlation"],
    }
    for channel in ("lateral", "heading", "curvature"):
        for name, value in metrics[channel].items():
            result[f"{channel}_{name}"] = value
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
