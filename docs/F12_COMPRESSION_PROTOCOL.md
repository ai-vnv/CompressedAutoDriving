# F12 Compression Protocol — Frozen Before Experiments

## Scope and immutable reference

F12 compresses only the deterministic actor of the frozen Original Belief-PPO checkpoint
`artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt` (SHA256
`02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`).
The public 29D ordering, physical normalization, MobileNetV3-small lane path,
YOLO11n object path, all belief filters, stop state, route observer, and physical
action mapping remain unchanged. The critic is retained only as frozen training-state
context and is not part of actor deployment cost.

The verified actor is `Linear(29,256) → Tanh → Linear(256,256) → Tanh →
Linear(256,2)`. Its Gaussian `log_std` is state-independent and is copied exactly for
FP32 student checkpoints; deployment fidelity is evaluated on the deterministic mean.
Normalized actor output is clipped to `[-1,1]` and mapped once to
`v_cmd∈[0,0.4] m/s`, `omega_cmd∈[-4,4] rad/s`.

## Data and integrity

New disjoint C4 seeds are fixed in `configs/f12_belief_ppo_compression_v1.toml`:

- development/distillation: 178001–178008;
- model selection: 178021–178028;
- once-only final holdout: 178101–178108;
- two new retention seeds per C0–C3 in the 1782xx block.

All are disjoint from historical and F11 locked seeds 177101–177108. Public states are
collected through unchanged `RGB → MobileNet/YOLO → belief → normalized 29D → Original
Belief-PPO`. Saved optimization rows contain seed, episode, step, public phase,
normalized and physical public 29D, original deterministic raw/normalized/physical
actions, and termination/truncation flags. They contain no evaluation GT, world pose,
bbox/IoU, or privileged state. Simulator truth may be
read after action for closed-loop metrics but never enters distillation.

The public phase taxonomy and thresholds are copied from frozen F11:
`nominal`, `lane_curve`, `pedestrian_relevant`, `stop_required`, `stop_satisfied`.
Mini-batches use inverse-frequency phase weights so nominal driving cannot dominate.

## Structured pruning

Every width is pruned directly from the same immutable 256×256 actor. For neuron `j`,
the frozen layer-wise score is the sum of L2 incoming and outgoing connectivity plus
absolute bias. Ties retain the lower original index. Input dimensions are never ranked
or removed. The exact survivor indices are stored.

Candidates are 192×192 (P25), 128×128 (P50), 96×96 (P62.5), and 64×64 (P75).
Pruning-only and independently distilled counterparts are retained for the pruning
frontier. The selected width is the smallest PD candidate that passes frozen action and
closed-loop development/selection gates; failure does not trigger threshold changes.

## Distillation

The Original Belief-PPO is frozen. The student minimizes Smooth-L1 error between
teacher and student deterministic physical actions after normalizing errors by the full
physical ranges 0.4 m/s and 8 rad/s. No GT labels are used. Adam trains for the fixed
budget in config with phase-balanced sampling. The state-independent `log_std` is copied,
not learned as a separate compression target.

## INT8 protocol

The installed PyTorch 2.12.1 stack provides `torch.ao` eager static INT8 on the x86
backend; TorchAO, ONNX Runtime, and TensorRT are absent and are not silently installed.
All comparable INT8 variants therefore use x86 CPU static quantized `Linear` kernels:
qint8 per-channel symmetric weights and quint8 per-tensor affine activations. Float
`Tanh` is separated by explicit quantize/dequantize boundaries.

PTQ variants calibrate only on development public states. QD/PDQD use actual fake-
quantized QAT preparation during teacher-guided recovery and are converted to the same
deployable INT8 modules. A model is labelled INT8 only if inspection finds quantized
Linear modules. CPU FP32 and INT8 latency are compared on the same one-thread batch-1
path. CUDA FP32 and CPU INT8 are not presented as a direct speedup comparison.

## Mandatory ablations

| ID | Name | Construction |
|---|---|---|
| A0 | B-PPO | original FP32 actor |
| A1 | B-PPO-P | selected structured-pruned actor |
| A2 | B-PPO-PD | A1 architecture plus FP32 KD |
| A3 | B-PPO-Q | A0 plus INT8 PTQ |
| A4 | B-PPO-QD | unpruned fake-quantized KD then INT8 |
| A5 | B-PPO-PQ | A1 plus INT8 PTQ |
| A6 | B-PPO-PDQ | A2 plus INT8 PTQ |
| A7 | B-PPO-PDQD | A2 plus fake-quantized KD then INT8 |

No variant inherits a distilled checkpoint when its label says pruning-only.

## Frozen evaluation and selection

Open-loop action fidelity reports overall and each public phase: MAE, RMSE, median,
P95/P99/max absolute error, Pearson, Spearman, saturation rate, and omega sign
disagreement above the predeclared 0.2 rad/s deadband. Exact numerical gates are in the
config and cannot be weakened after selection/final results.

Closed-loop evaluation uses identical seeds and existing F10 metrics. Eligibility
requires no new collision and bounds regressions in unsafe episodes, stop violations,
lane failures, completion, restart, progress, and minimum clearance according to the
config. Original-policy performance on the same seeds is the reference; a known original
weakness is not charged to compression unless worsened beyond the margin.

For compression-selection only, A0, A1, and A2 are byte-identical to the
actors already evaluated as A0, the selected pruning-only width, and the
selected pruning-plus-distillation width in the pruning-frontier run. Their
same-seed episode rows are reused with checkpoint-SHA verification and an
explicit source-artifact hash record. They are not treated as independent
replicates. A3--A7 receive new rollouts. Final-holdout evaluation never reuses
selection episodes.

Selection is safety/behavior first, then action fidelity, then the Pareto tradeoff among
parameter memory, actor file size, same-device latency, and fidelity. No arbitrary scalar
score is used. If PDQ already passes, it is preferred over PDQD unless QAT improves the
mean normalized two-action MAE by at least 10% without worsening behavior; this freezes
the second-distillation stop rule. Final holdout is claimed and evaluated once after the candidate and all
margins are frozen. All A0--A7 comparisons are completed on the selection split before
that claim. To prevent the holdout from becoming a second model-selection set, the final
split evaluates only immutable Original A0 and the already selected compressed candidate.
Retention compares those same two policies on C0–C4-compatible scenarios.

## Efficiency protocol

Dense/active parameters, MACs, serialized size, parameter memory, process peak-memory
delta, batch-1 latency median/P95/P99, and throughput are reported. Benchmarking uses
one CPU thread, 1,000 warmups, 10,000 timed actions, five repeats. Actor-only results are
separate from end-to-end perception. Because perception is unchanged and expected to
dominate, F12 does not claim equivalent whole-system acceleration without a measured
end-to-end result.

## Final rule

F12 ends after A0–A7, pruning-frontier, frozen selection, once-only final/retention,
artifact verification, and `docs/F12_COMPRESSION_RESULTS.md` plus
`docs/F12_COMPRESSION_ABLATION.md`. It does not run Distributional IG, retrain PPO, or
start Explain Again.

## Documented preflight invocation

Run from repository root before collecting F12 data:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
export DUCKIETOWN_HEADLESS=1
export CUDA_VISIBLE_DEVICES=0
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/run_f12_compression.py verify \
  --config configs/f12_belief_ppo_compression_v1.toml
```

The command must exit zero, report checkpoint SHA256
`02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`,
29D/Tanh/256×256 compatibility through the verifier, and the x86 quantized backend.
