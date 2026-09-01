"""Actor-only structured pruning, distillation, and INT8 deployment utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class ActorSpec:
    input_dimension: int = 29
    hidden_sizes: tuple[int, int] = (256, 256)
    output_dimension: int = 2
    activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.input_dimension != 29:
            raise ValueError("F12 may not prune or change the 29D semantic input")
        if len(self.hidden_sizes) != 2 or min(self.hidden_sizes) <= 0:
            raise ValueError("F12 actor requires two positive hidden widths")
        if self.output_dimension != 2 or self.activation != "tanh":
            raise ValueError("actor output/activation must match frozen PPO semantics")


class DenseBeliefActor(nn.Module):
    """Dense actor with the exact frozen PPO Tanh parameterization."""

    def __init__(self, spec: ActorSpec) -> None:
        super().__init__()
        self.spec = spec
        h1, h2 = spec.hidden_sizes
        self.fc1 = nn.Linear(spec.input_dimension, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, spec.output_dimension)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.fc1(observation))
        value = torch.tanh(self.fc2(value))
        return self.out(value)


class QuantizableBeliefActor(nn.Module):
    """Eager static-INT8 actor with float Tanh boundaries.

    Explicit quant/dequant boundaries keep Tanh in float while every Linear is
    converted to a real quantized operator on the x86 backend.
    """

    def __init__(self, spec: ActorSpec) -> None:
        super().__init__()
        self.spec = spec
        h1, h2 = spec.hidden_sizes
        self.quant1 = torch.ao.quantization.QuantStub()
        self.fc1 = nn.Linear(spec.input_dimension, h1)
        self.dequant1 = torch.ao.quantization.DeQuantStub()
        self.quant2 = torch.ao.quantization.QuantStub()
        self.fc2 = nn.Linear(h1, h2)
        self.dequant2 = torch.ao.quantization.DeQuantStub()
        self.quant3 = torch.ao.quantization.QuantStub()
        self.out = nn.Linear(h2, spec.output_dimension)
        self.dequant3 = torch.ao.quantization.DeQuantStub()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = self.dequant1(self.fc1(self.quant1(observation)))
        value = torch.tanh(value)
        value = self.dequant2(self.fc2(self.quant2(value)))
        value = torch.tanh(value)
        return self.dequant3(self.out(self.quant3(value)))


@dataclass(frozen=True)
class PruningResult:
    actor: DenseBeliefActor
    first_layer_survivors: tuple[int, ...]
    second_layer_survivors: tuple[int, ...]
    first_layer_scores: tuple[float, ...]
    second_layer_scores: tuple[float, ...]


def extract_original_actor(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
) -> tuple[DenseBeliefActor, torch.Tensor, dict[str, Any]]:
    """Load only actor/log-std from the immutable PPO checkpoint."""

    path = Path(checkpoint)
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("Original Belief-PPO checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    spec = ActorSpec(
        input_dimension=int(config["observation_dimension"]),
        hidden_sizes=tuple(int(value) for value in config["hidden_sizes"]),
        output_dimension=int(config["action_dimension"]),
        activation="tanh",
    )
    actor = DenseBeliefActor(spec)
    state = payload["model_state"]
    with torch.no_grad():
        actor.fc1.weight.copy_(state["actor.0.weight"])
        actor.fc1.bias.copy_(state["actor.0.bias"])
        actor.fc2.weight.copy_(state["actor.2.weight"])
        actor.fc2.bias.copy_(state["actor.2.bias"])
        actor.out.weight.copy_(state["actor.4.weight"])
        actor.out.bias.copy_(state["actor.4.bias"])
    actor.eval()
    return actor, state["log_std"].detach().clone(), payload


def build_pruned_actor(original: DenseBeliefActor, width: int) -> PruningResult:
    """Direct layer-wise structured pruning from the original actor."""

    if original.spec.hidden_sizes != (256, 256):
        raise ValueError("pruning source must be the original 256x256 actor")
    if width not in {192, 128, 96, 64}:
        raise ValueError("unsupported frozen F12 pruning width")
    with torch.no_grad():
        first = (
            torch.linalg.vector_norm(original.fc1.weight, dim=1)
            + torch.linalg.vector_norm(original.fc2.weight, dim=0)
            + original.fc1.bias.abs()
        )
        second = (
            torch.linalg.vector_norm(original.fc2.weight, dim=1)
            + torch.linalg.vector_norm(original.out.weight, dim=0)
            + original.fc2.bias.abs()
        )
        index1 = _stable_topk(first, width)
        index2 = _stable_topk(second, width)
        student = DenseBeliefActor(ActorSpec(hidden_sizes=(width, width)))
        student.fc1.weight.copy_(original.fc1.weight[index1, :])
        student.fc1.bias.copy_(original.fc1.bias[index1])
        student.fc2.weight.copy_(original.fc2.weight[index2][:, index1])
        student.fc2.bias.copy_(original.fc2.bias[index2])
        student.out.weight.copy_(original.out.weight[:, index2])
        student.out.bias.copy_(original.out.bias)
    student.eval()
    return PruningResult(
        actor=student,
        first_layer_survivors=tuple(int(value) for value in index1.tolist()),
        second_layer_survivors=tuple(int(value) for value in index2.tolist()),
        first_layer_scores=tuple(float(first[value]) for value in index1),
        second_layer_scores=tuple(float(second[value]) for value in index2),
    )


def copy_dense_to_quantizable(actor: DenseBeliefActor) -> QuantizableBeliefActor:
    target = QuantizableBeliefActor(actor.spec)
    with torch.no_grad():
        for destination, source in (
            (target.fc1, actor.fc1), (target.fc2, actor.fc2), (target.out, actor.out)
        ):
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)
    return target


def prepare_ptq(
    actor: DenseBeliefActor,
    calibration_observations: NDArray[np.float32],
    *,
    backend: str = "x86",
) -> nn.Module:
    torch.backends.quantized.engine = backend
    model = copy_dense_to_quantizable(actor).cpu().eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    prepared = torch.ao.quantization.prepare(model, inplace=False)
    calibration = torch.as_tensor(calibration_observations, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(calibration), 512):
            prepared(calibration[start : start + 512])
    converted = torch.ao.quantization.convert(prepared, inplace=False).eval()
    require_real_int8(converted)
    return converted


def prepare_qat(actor: DenseBeliefActor, *, backend: str = "x86") -> nn.Module:
    torch.backends.quantized.engine = backend
    model = copy_dense_to_quantizable(actor).cpu().train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model, inplace=False)


def convert_qat(actor: nn.Module, *, backend: str = "x86") -> nn.Module:
    torch.backends.quantized.engine = backend
    converted = torch.ao.quantization.convert(actor.cpu().eval(), inplace=False)
    require_real_int8(converted)
    return converted


def require_real_int8(actor: nn.Module) -> None:
    quantized = [
        module for module in actor.modules()
        if isinstance(module, torch.ao.nn.quantized.Linear)
    ]
    if len(quantized) != 3:
        raise RuntimeError("INT8 label requires exactly three quantized Linear modules")
    if any(module.weight().dtype != torch.qint8 for module in quantized):
        raise RuntimeError("quantized actor does not contain qint8 weights")


def physical_actions(mean: torch.Tensor) -> torch.Tensor:
    """Frozen normalized-mean to physical-action mapping."""

    clipped = mean.clamp(-1.0, 1.0)
    return torch.stack(((clipped[..., 0] + 1.0) * 0.2, clipped[..., 1] * 4.0), dim=-1)


def distill_dense_actor(
    actor: nn.Module,
    observations: NDArray[np.float32],
    teacher_physical_actions: NDArray[np.float32],
    phases: Sequence[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> list[dict[str, float]]:
    """Phase-balanced physical-action distillation for dense or QAT actors."""

    matrix = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(teacher_physical_actions, dtype=np.float32)
    phase_array = np.asarray(phases)
    if matrix.ndim != 2 or matrix.shape[1] != 29 or targets.shape != (len(matrix), 2):
        raise ValueError("invalid public distillation dataset")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("distillation dataset contains non-finite values")
    unique, counts = np.unique(phase_array, return_counts=True)
    phase_weight = {name: 1.0 / count for name, count in zip(unique, counts, strict=True)}
    probabilities = np.asarray([phase_weight[value] for value in phase_array], dtype=np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    actor.to(device).train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    scale = torch.as_tensor((0.4, 8.0), dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        indexes = rng.choice(len(matrix), size=len(matrix), replace=True, p=probabilities)
        losses = []
        for start in range(0, len(indexes), batch_size):
            batch = torch.as_tensor(indexes[start : start + batch_size], device=device)
            prediction = physical_actions(actor(x[batch]))
            loss = torch.nn.functional.smooth_l1_loss(
                prediction / scale, y[batch] / scale
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    actor.eval()
    return history


def actor_parameter_count(spec: ActorSpec) -> int:
    h1, h2 = spec.hidden_sizes
    return (
        spec.input_dimension * h1 + h1
        + h1 * h2 + h2
        + h2 * spec.output_dimension + spec.output_dimension
    )


def actor_macs(spec: ActorSpec) -> int:
    h1, h2 = spec.hidden_sizes
    return spec.input_dimension * h1 + h1 * h2 + h2 * spec.output_dimension


def save_dense_actor(
    path: str | Path,
    actor: DenseBeliefActor,
    *,
    log_std: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "f12_dense_belief_actor",
            "spec": asdict(actor.spec),
            "state_dict": actor.cpu().state_dict(),
            "log_std": log_std.cpu(),
            "metadata": metadata,
        },
        target,
    )


def load_dense_actor(path: str | Path) -> tuple[DenseBeliefActor, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = dict(payload["spec"])
    raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    actor = DenseBeliefActor(ActorSpec(**raw))
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    return actor, payload


def save_quantized_actor(path: str | Path, actor: nn.Module) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    actor = actor.cpu().eval()
    require_real_int8(actor)
    traced = torch.jit.trace(actor, torch.zeros((1, 29), dtype=torch.float32))
    traced.save(str(target))


def _stable_topk(scores: torch.Tensor, width: int) -> torch.Tensor:
    # Python sort makes the documented lower-index tie break explicit.
    ordered = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:width]
    return torch.as_tensor(sorted(ordered), dtype=torch.long)


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

