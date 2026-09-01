"""Logical parameter memory of the INT8 actor (A6), unpacked from the graph.

Convention (same as docs/F12_COMPRESSION_RESULTS.md): qint8 weights at 1 byte
per element, FP32 biases at 4 bytes, plus the per-channel quantization
parameters the deployed graph must carry (float32 scale + int32 zero point,
8 bytes per output channel). Per-tensor activation scale/zero-point pairs
(2 scalars per layer) are excluded, as in F12. Writes the result next to this
script as int8_parameter_memory.json for gen_fig_precision.py.
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "actor_A6_kd_balanced_ptq_int8.pt"

m = torch.jit.load(str(MODEL), map_location="cpu")
layers, n_w, n_b, n_ch = [], 0, 0, 0
for name, mod in m.named_modules():
    try:
        packed = mod._packed_params._packed_params
    except Exception:
        continue
    w, b = torch.ops.quantized.linear_unpack(packed)
    per_channel = w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
    ch = w.q_per_channel_scales().numel() if per_channel else 1
    layers.append({"layer": name, "weight_shape": list(w.shape),
                   "n_weights": w.numel(), "n_bias": int(b.numel()), "channels": ch})
    n_w += w.numel()
    n_b += int(b.numel())
    n_ch += ch

total = n_w * 1 + n_b * 4 + n_ch * 8
out = {
    "model": str(MODEL.relative_to(ROOT)),
    "convention": "qint8 weights x1B + fp32 bias x4B + per-channel (fp32 scale + int32 zero point) x8B",
    "layers": layers,
    "weights_qint8": n_w,
    "biases_fp32": n_b,
    "quantized_channels": n_ch,
    "logical_parameter_memory_bytes": total,
}
dst = Path(__file__).with_name("int8_parameter_memory.json")
dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"{dst.name}: {total} B ({n_w} w + {n_b} b + {n_ch} ch)")
