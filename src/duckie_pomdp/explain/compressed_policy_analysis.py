"""F13 utilities for comparing frozen Original and compressed INT8 actors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray


ACTION_RANGES = np.asarray((0.4, 8.0), dtype=np.float64)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_to_physical(action: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(action, dtype=np.float32)
    result = np.empty_like(values)
    result[..., 0] = (np.clip(values[..., 0], -1.0, 1.0) + 1.0) * 0.2
    result[..., 1] = np.clip(values[..., 1], -1.0, 1.0) * 4.0
    return result


def actor_physical(actor: torch.nn.Module, observations: NDArray[np.floating]) -> NDArray[np.float32]:
    matrix = np.asarray(observations, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 29 or not np.isfinite(matrix).all():
        raise ValueError("actor observations must be finite [N,29]")
    predictions: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(matrix), 2048):
            tensor = torch.from_numpy(matrix[start : start + 2048])
            predictions.append(actor(tensor).detach().cpu().numpy().astype(np.float32))
    return normalized_to_physical(np.concatenate(predictions, axis=0))


def scalar_metrics(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("metric vectors must be non-empty and aligned")
    error = right - left
    absolute = np.abs(error)
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error": float(np.max(absolute)),
        "pearson": _pearson(left, right),
        "spearman": _spearman(left, right),
    }


def paired_effect_metrics(
    original: Sequence[float],
    compressed: Sequence[float],
    *,
    action_range: float,
    direction_deadband: float,
) -> dict[str, Any]:
    original_values = np.asarray(original, dtype=np.float64)
    compressed_values = np.asarray(compressed, dtype=np.float64)
    drift = compressed_values - original_values
    eligible = np.abs(original_values) > direction_deadband
    agreement = (
        float(np.mean(np.sign(original_values[eligible]) == np.sign(compressed_values[eligible])))
        if np.any(eligible)
        else 1.0
    )
    return {
        "count": int(len(original_values)),
        "original_mean": float(np.mean(original_values)),
        "compressed_mean": float(np.mean(compressed_values)),
        "original_mean_absolute": float(np.mean(np.abs(original_values))),
        "compressed_mean_absolute": float(np.mean(np.abs(compressed_values))),
        "mean_drift": float(np.mean(drift)),
        "mean_absolute_drift": float(np.mean(np.abs(drift))),
        "p95_absolute_drift": float(np.quantile(np.abs(drift), 0.95)),
        "maximum_absolute_drift": float(np.max(np.abs(drift))),
        "normalized_mean_effect_drift": float(
            abs(np.mean(compressed_values) - np.mean(original_values)) / action_range
        ),
        "normalized_p95_effect_drift": float(np.quantile(np.abs(drift), 0.95) / action_range),
        "direction_eligible_count": int(np.sum(eligible)),
        "paired_direction_agreement": agreement,
    }


def verify_hash(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def require_quantized_linear_graph(actor: torch.jit.ScriptModule, count: int = 3) -> None:
    graph = str(actor.inlined_graph)
    observed = graph.count("quantized::linear")
    if observed < count:
        raise RuntimeError(f"expected at least {count} quantized Linear ops, found {observed}")


def classification_from_counterfactual(
    primary_checks: Mapping[str, bool], sham_pass: bool
) -> str:
    if not sham_pass:
        return "INVALID"
    passed = sum(bool(value) for value in primary_checks.values())
    if passed == len(primary_checks):
        return "PRESERVED"
    if passed >= max(1, len(primary_checks) - 1):
        return "PARTIALLY PRESERVED"
    return "SHIFTED"


def _pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return _pearson(_rank(left), _rank(right))


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        for index, count in enumerate(counts):
            if count > 1:
                ranks[inverse == index] = np.mean(ranks[inverse == index])
    return ranks

