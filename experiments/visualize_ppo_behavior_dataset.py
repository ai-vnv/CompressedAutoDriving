"""Create a deterministic visual audit of a PPO behavior NPZ and source CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_COLORS = ("#0072B2", "#009E73", "#E69F00", "#D55E00")


def _role_colors(count: int) -> tuple[object, ...]:
    """Return a deterministic palette without limiting the number of roles."""
    if count <= len(BASE_COLORS):
        return BASE_COLORS[:count]
    color_map = plt.get_cmap("tab20")
    return tuple(color_map(index / max(1, count - 1)) for index in range(count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.dataset) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 29 or actions.shape != (len(observations), 2):
        raise RuntimeError("unexpected behavior dataset shape")
    if not all(np.all(np.isfinite(value)) for value in (observations, actions, weights)):
        raise RuntimeError("non-finite behavior dataset")

    with args.source_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        feature_names = [name.removeprefix("policy_normalized.") for name in reader.fieldnames or () if name.startswith("policy_normalized.")]
        roles = np.asarray([row["source_role"] for row in reader], dtype=object)
    if len(roles) != len(observations) or len(feature_names) != 29:
        raise RuntimeError("source CSV does not align with NPZ")
    role_names = tuple(dict.fromkeys(roles.tolist()))
    role_colors = _role_colors(len(role_names))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    counts = [int(np.sum(roles == role)) for role in role_names]
    masses = [float(weights[roles == role].sum()) for role in role_names]
    x = np.arange(len(role_names))
    width = 0.38
    ax = axes[0, 0]
    ax.bar(x - width / 2, counts, width, label="rows", color="#56B4E9")
    ax.bar(x + width / 2, masses, width, label="weight mass", color="#E69F00")
    ax.set_xticks(x, [role.replace("_", "\n") for role in role_names], rotation=0)
    ax.set_title("Dataset composition")
    ax.legend()

    ax = axes[0, 1]
    rng = np.random.default_rng(23001)
    for index, role in enumerate(role_names):
        indices = np.flatnonzero(roles == role)
        if len(indices) > 1800:
            indices = rng.choice(indices, 1800, replace=False)
        ax.scatter(
            actions[indices, 0],
            actions[indices, 1],
            s=5,
            alpha=0.30,
            color=role_colors[index],
            label=role,
        )
    ax.set_xlabel("normalized linear action")
    ax.set_ylabel("normalized angular action")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_title("Teacher/rehearsal action support")
    ax.legend(loc="upper left", markerscale=2)

    means = np.stack([np.mean(np.abs(observations[roles == role]), axis=0) for role in role_names])
    ax = axes[1, 0]
    image = ax.imshow(means, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1.0, float(np.max(means))))
    ax.set_yticks(np.arange(len(role_names)), [role.replace("_", " ") for role in role_names])
    ax.set_xticks(np.arange(29), [str(i + 1) for i in range(29)], fontsize=7)
    ax.set_xlabel("policy feature index (see right panel)")
    ax.set_title("Mean absolute normalized feature value")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    ax = axes[1, 1]
    ax.axis("off")
    feature_text = "\n".join(
        f"{i + 1:>2}. {name}" for i, name in enumerate(feature_names)
    )
    ax.text(0.0, 1.0, feature_text, va="top", ha="left", family="monospace", fontsize=7.2)
    ax.set_title("Fixed 29D public-belief ordering", loc="left")

    fig.suptitle(
        f"PPO behavior dataset audit: {args.dataset.stem}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    summary = {
        "rows": len(observations),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "roles": {
            role: {"rows": count, "weight_mass": mass}
            for role, count, mass in zip(role_names, counts, masses, strict=True)
        },
        "feature_names": feature_names,
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
