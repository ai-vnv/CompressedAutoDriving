"""Action-fidelity and deployment-efficiency metrics for F12."""

from __future__ import annotations

import time
import resource
import tracemalloc
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from duckie_pomdp.optimization.actor_compression import ActorSpec, actor_macs, actor_parameter_count, physical_actions


def actor_physical_predictions(actor: nn.Module, observations: NDArray[np.float32]) -> NDArray[np.float32]:
    actor.eval()
    rows = []
    with torch.inference_mode():
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        for start in range(0, len(tensor), 2048):
            rows.append(physical_actions(actor(tensor[start : start + 2048])).cpu().numpy())
    return np.asarray(np.concatenate(rows), dtype=np.float32)


def action_fidelity(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    *,
    omega_deadband: float,
) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("action matrices must align with shape (N,2)")
    result: dict[str, object] = {"rows": len(reference)}
    for index, name in enumerate(("v_cmd_mps", "omega_cmd_rad_s")):
        error = candidate[:, index] - reference[:, index]
        absolute = np.abs(error)
        result[name] = {
            "bias": float(np.mean(error)),
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "median_absolute_error": float(np.median(absolute)),
            "p95_absolute_error": float(np.percentile(absolute, 95)),
            "p99_absolute_error": float(np.percentile(absolute, 99)),
            "maximum_absolute_error": float(np.max(absolute)),
            "pearson": _correlation(reference[:, index], candidate[:, index]),
            "spearman": _correlation(_ranks(reference[:, index]), _ranks(candidate[:, index])),
        }
    reference_saturated = (
        (reference[:, 0] <= 0.0)
        | (reference[:, 0] >= 0.4)
        | (np.abs(reference[:, 1]) >= 4.0)
    )
    candidate_saturated = (
        (candidate[:, 0] <= 0.0)
        | (candidate[:, 0] >= 0.4)
        | (np.abs(candidate[:, 1]) >= 4.0)
    )
    result["action_bound_saturation_frequency"] = {
        "original": float(np.mean(reference_saturated)),
        "candidate": float(np.mean(candidate_saturated)),
        "disagreement": float(np.mean(reference_saturated != candidate_saturated)),
    }
    mask = np.abs(reference[:, 1]) > omega_deadband
    result["omega_sign"] = {
        "deadband_rad_s": omega_deadband,
        "eligible_rows": int(mask.sum()),
        "disagreement_frequency": float(np.mean(np.sign(reference[mask, 1]) != np.sign(candidate[mask, 1]))) if mask.any() else 0.0,
    }
    return result


def phase_fidelity(
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
    phases: Sequence[str],
    *,
    omega_deadband: float,
) -> dict[str, object]:
    phase_array = np.asarray(phases)
    return {
        str(phase): action_fidelity(reference[phase_array == phase], candidate[phase_array == phase], omega_deadband=omega_deadband)
        for phase in np.unique(phase_array)
    }


def benchmark_actor(
    actor: nn.Module,
    spec: ActorSpec,
    model_path: str | Path,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    threads: int,
    int8: bool = False,
) -> dict[str, float | int]:
    torch.set_num_threads(threads)
    actor.cpu().eval()
    sample = torch.zeros((1, 29), dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            actor(sample)
        rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        tracemalloc.start()
        values = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                actor(sample)
            elapsed = time.perf_counter_ns() - start
            values.extend([elapsed / iterations / 1_000.0])
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    latency = np.asarray(values, dtype=np.float64)
    parameter_count = actor_parameter_count(spec)
    h1, h2 = spec.hidden_sizes
    bias_count = h1 + h2 + spec.output_dimension
    weight_count = parameter_count - bias_count
    logical_memory = (
        weight_count + bias_count * 4 + (h1 + h2 + spec.output_dimension) * 8
        if int8
        else parameter_count * 4
    )
    return {
        "dense_parameter_count": parameter_count,
        "active_parameter_count": parameter_count,
        "logical_parameter_memory_bytes": logical_memory,
        "precision": "INT8" if int8 else "FP32",
        "macs": actor_macs(spec),
        "flops_multiply_add_as_two": actor_macs(spec) * 2,
        "actor_checkpoint_size_bytes": Path(model_path).stat().st_size,
        "batch1_latency_us_median": float(np.median(latency)),
        "batch1_latency_us_p95": float(np.percentile(latency, 95)),
        "batch1_latency_us_p99": float(np.percentile(latency, 99)),
        "throughput_actions_per_second": float(1_000_000.0 / np.median(latency)),
        "process_peak_rss_bytes": rss_after,
        "process_peak_rss_delta_bytes": max(rss_after - rss_before, 0),
        "python_tracemalloc_peak_delta_bytes": int(python_peak),
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "repeats": repeats,
        "threads": threads,
    }


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if np.std(first) <= 1.0e-12 or np.std(second) <= 1.0e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks
