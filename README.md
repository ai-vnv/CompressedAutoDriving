# CompressedAutoDriving

**A Closed-Loop Evaluation of Capability Loss and Recovery in Compressed Driving Policies**

Where does a compressed driving policy stop being able to drive, and what brings
that ability back?

This repository is the code and model artifact for the study of the same name,
prepared for the *IEEE Open Journal of Intelligent Transportation Systems*. A
visuomotor driving policy is trained in Gym-Duckietown as a partially observable
Markov decision process, its actor network is extracted, and that actor is then
pushed through a compression pipeline one stage at a time: structured pruning,
knowledge distillation under two different rehearsal coverages, integer
quantization by two routes, and a reduced floating-point control. Every
intermediate configuration has to drive again, on the same eight seeds per
curriculum, against acceptance criteria fixed before any result existed, on an
evaluation backend verified to reproduce bit for bit.

The framing is a scenario-based assessment turned around. Instead of holding the
vehicle fixed and varying the scenarios, we hold the scenario set and the seeds
fixed and let the system under test change at every compression stage.

![Pipeline overview](assets/pipeline_overview.jpg)

## Key results

Verdicts are decided by the same preregistered acceptance checks on the same
eight seeds per curriculum. `ref` marks the unmodified actor the checks are
anchored to.

| Member | Construction | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| A0 | original actor (reference) | ref | ref | ref | ref | ref |
| A1 | prune (width 256 to 64) | ✗ | ✗ | ✗ | ✗ | ✗ |
| A2 | prune, then KD on C4 data only | ✗ | ✗ | ✗ | ✓ | ✓ |
| A3 | prune, then KD on balanced C0–C4 data | ✓ | ✓ | ✓ | ✓ | ✓ |
| A4 | PTQ INT8 of the unpruned actor (control) | ✓ | ✓ | ✓ | ✓ | ✓ |
| A5 | prune, then PTQ, no KD (control) | ✗ | ✗ | ✗ | ✗ | ✗ |
| A6 | A3, then post-training INT8 | ✓ | ✓ | ✓ | ✗ | ✗ |
| A7 | prune, KD(C4), PTQ, QAT(C4) (control) | ✗ | ✗ | ✗ | ✓ | ✓ |
| A8 | A3 parent, then QAT INT8 | ✓ | ✓ | ✓ | ✗ | ✗ |
| A9 | A3, then FP16 cast (precision control) | ✓ | ✓ | ✓ | ✓ | ✓ |

**The study in five findings:**

1. **Pruning is where driving capability is first lost.** Cutting the hidden
   width from 256 to 64 (73,986 to 6,210 parameters, a 91.6% reduction) fails
   all five curricula at once: invalid poses and lane failures on the early
   tasks, and a stop violation in every episode of the stop task.
2. **How much distillation recovers is decided by its rehearsal data.** Same
   teacher, loss, optimizer, and budget: rehearsing only on the hardest
   curriculum restores C3 and C4 alone, while balanced rehearsal across all
   five curricula (62,176 states) restores the complete tested behavior of the
   FP32 actor.
3. **Integer quantization of the improved actor loses the stop curricula, in
   two opposite ways.** From one byte-identical recovered checkpoint, the
   post-training route yields a policy that stops correctly and then never
   moves again, parked 0.22 m before the line until the horizon, while the
   quantization-aware route yields one that drives through the stop at low
   speed.
4. **The same procedure on the unpruned actor preserves all five curricula**,
   and an FP16 cast of the recovered actor keeps all five while halving
   parameter memory. Reduced precision by itself is not what breaks the policy.
   Under the tested procedures the failure needs both the narrow recovered
   actor and integer quantization.
5. **Action-level similarity does not predict driving, in either direction.**
   The quantization-aware model reproduces the original actions more closely
   than the post-training one and drives worse, while the unpruned INT8 control
   drives every curriculum and fails the numerical similarity thresholds.

## How the policy sees: perception, EKF belief, 29 inputs

The policy never receives simulator state. Each camera frame feeds two networks
in parallel: a fine-tuned **YOLO11n** detector (pedestrians, stop signs) whose
box bottom-centers are ground-projected into range and bearing measurements,
and a **MobileNetV3-Small** regressor producing lane offset, heading error, and
curvature. **Extended Kalman filters** fuse these over time: a lane filter with
ego-motion prediction, per-object filters paired with Bernoulli existence
filters (a missed detection only erodes existence when the camera could
actually have seen the object), and a stop-obligation state machine. The
posteriors are read out into a fixed 29-dimensional belief vector holding
existence probabilities, means, dispersions, stop mode, ego-motion, and the
previous action.

Only the actor that maps this belief to a driving command is compressed. The
perception stack stays fixed, so behavioral changes are attributable to the
compression stage.

## The five driving curricula

| ID | Map | Challenge | Horizon (steps) |
|---|---|---|---|
| C0 | small loop | lane following, domain randomization | 1,900 |
| C1 | larger loop | longer route, domain randomization | 2,700 |
| C2 | larger loop | crossing pedestrian | 2,700 |
| C3 | larger loop | stop sign: stop, hold, resume | 2,700 |
| C4 | larger loop | pedestrian and stop sign combined | 4,200 |

The two stop curricula are where the quantized configurations fail. They are the
ones that require the policy to interrupt its own driving and then re-establish
it.

## Evaluation protocol

- **Matched closed loop**: 10 configurations, 5 curricula, 8 seeds, so 400
  episodes, with identical seeds for every configuration.
- **Acceptance criteria fixed in advance**: completion, progress, collisions,
  stop compliance and restart, lane keeping, pedestrian clearance, as absolute
  bounds plus regression limits relative to the original actor. No threshold was
  changed after results were opened.
- **Bit-for-bit reproducible**: the deterministic backend was verified, not
  assumed. Three repeats of a full episode produce identical action sequences,
  and the INT8 and FP16 execution paths each passed the same check before use.
- **Two independent axes**: task-level driving verdicts and action-level
  fidelity (error and correlation against the original on identical inputs) are
  measured separately, which is what makes the dissociation in finding 5
  visible.
- **Evidence capture**: camera frames are recorded during the evaluation
  rollouts themselves, never re-rendered, and per-step telemetry is stored for
  every episode.

## Actor cost (single-thread CPU, batch 1)

| | File | Parameter memory | Median latency |
|---|---|---|---|
| A3 FP32 | 29.3 KB | 24.8 KB | 19.4 µs |
| A9 FP16 | 15.9 KB | 12.4 KB | 24.5 µs (+26%) |
| A6 INT8 | 34.1 KB | 7.6 KB | 12.9 µs (1.50×) |

Parameter memory counts weights and biases, at 4 bytes each for FP32 and 2 for
FP16. For INT8 it is unpacked from the deployed graph and counts 1-byte
quantized weights, FP32 biases, and the per-channel scale and zero point
(`experiments/paper_figures/compute_int8_memory.py`). FP16 halves memory but is
slower on this x86 backend, which has no native half-precision execution path,
so it is a memory option and not a speedup. The serialized INT8 file is larger
than the FP32 one because the traced quantized graph metadata dominates. The
actor itself is under 0.1% of end-to-end step cost, since perception dominates.

## Repository structure

```
models/         trained weights: actors A0-A9, YOLO11n, MobileNetV3 lane model
                (each SHA256-verified against the study registries; MANIFEST.md)
configs/        experiment configurations: curricula, acceptance criteria,
                seeds, quantization settings, hash-pinned protocol provenance
experiments/    evaluation runners, protocol-freeze scripts, integrity
                verifiers, and the figure-generation code for the paper
src/            perception (YOLO, MobileNetV3), EKF belief, PPO environment,
                compression (pruning / distillation / PTQ / QAT)
docs/           frozen protocols and per-study reports (F12 to F18)
slides/         talk built from this work (Beamer source and PPTX)
tests/          regression suite, 721 tests
maps/, scripts/ simulator maps and utilities
```

Evaluation artifacts, roughly 4 GB of per-episode telemetry, rollout videos, and
result ledgers, are intentionally not tracked. They regenerate from `configs/`
and `experiments/` with the recorded seeds, using the tracked weights in
`models/`.

## The study, stage by stage

| Study | Question | Report |
|---|---|---|
| F12 | historical compression pipeline and selection | `docs/F12_COMPRESSION_RESULTS.md` |
| F15 | where capability is lost, and recovery | `docs/F15_*` |
| F16 | pruning-schedule and training-seed robustness | `docs/F16_FINAL_SCOPE_CORRECTED_REPORT.md` |
| F17 | placement of the compression methods, quantization routes | `docs/F17_FINAL_REPORT.md` |
| F18 | reduced floating-point control | `docs/F18_FP16_CONTROL_REPORT.md` |

Negative and qualifying results are reported alongside the headline findings: no
stable advantage was found between direct and progressive pruning, the
distillation recovery is sensitive to the training seed, and a reserved
evaluation split was never opened because no candidate met the full deployment
criterion.

## Reproducing

The stack expects Gym-Duckietown and the pinned environment (`pyproject.toml`,
`constraints.txt`). All evaluation is deterministic:

```bash
export PYTHONPATH=src:experiments
export DUCKIETOWN_HEADLESS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m pytest tests -q                     # 721 tests, about one minute
python experiments/run_f17_optimization_method_order.py evaluate --pathway A6
```

Two things to know before running:

- **Configuration files record absolute paths from the machine the study ran
  on.** They are deliberately left as they are, because the protocol gates
  verify configuration files by SHA256 and editing a path would break the
  provenance chain the integrity verifiers check. Point them at your own
  checkout and expect the frozen-hash assertions to be the first thing that
  complains.
- **The integrity verifiers** (`artifacts/integrity_phase_c.sh`,
  `artifacts/integrity_final_45.sh`) recompute every reported number from the
  artifact ledgers, so they need the evaluation artifacts to have been
  regenerated first.

## License

The source code, configurations, and documentation in this repository are
released under the MIT license (`LICENSE`).

Model weights are a separate matter. `models/yolo11n_duckietown_best.pt` is
fine-tuned from Ultralytics YOLO11, which is licensed **AGPL-3.0**, and should
be treated under those terms. `THIRD_PARTY_NOTICES.md` records the provenance
and license of every tracked weight and of the main dependencies. Read it
before redistributing the weights.

## Built on

[Gym-Duckietown](https://github.com/duckietown/gym-duckietown),
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics),
MobileNetV3 (torchvision),
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
and PyTorch eager-mode quantization.

Related work from the same line: [DuckieMDP](https://github.com/PannnTastic/DuckieMDP),
certified explanations for the fully observed formulation, and
[Duckietown.jl](https://github.com/ai-vnv/Duckietown.jl), a Julia
reimplementation of the environment validated decision by decision.

## Acknowledgement

This work was supported by the IRC for Smart Mobility and Logistics (IRC-SML),
King Fahd University of Petroleum and Minerals, through research grant INML2654.
