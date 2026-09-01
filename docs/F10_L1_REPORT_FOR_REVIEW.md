# F10-L1 Report for Review — `small_loop` Counter-Clockwise Lane Competence

Classification: **PASS**

F10-L1 is an isolated curriculum stage created after the full F10 baseline
proved unable to drive reliably. It answers only one question: can SAC drive
one counter-clockwise lap of the real Gym-Duckietown `small_loop`, remain in
the right-hand lane, avoid the yellow center line, and stay on the road?

It does not claim full POMDP deployment readiness. YOLO, F9c belief, stop
logic, and pedestrian response are deliberately absent from this lane-only
stage and remain unchanged.

## Frozen protocol

- Config: `configs/f10_l1_lane_v1.toml`
- Config SHA256: `0db9dec6b06280a8f3ad531303ec295eb55aad98dfd15d02dab5567a4997d6c4`
- Map: native Gym-Duckietown `small_loop`, four left turns, counter-clockwise
- Training seeds: `13001-13012`
- Development seeds: `14001-14004`
- Final seeds: `15001-15004`
- Observation: six agent-visible lane/motion values (`d`, `phi`, actual
  velocity/yaw rate, previous commanded velocity/yaw rate)
- Action: normalized SAC action through the existing one-to-one mapper to
  `PolicyAction(v_cmd, omega_cmd)`, then the existing differential-drive
  adapter
- Budget: 60,000 simulator steps; 58,001 gradient updates
- SAC: two 256-unit ReLU layers, learning rate `3e-4`, gamma `0.99`, tau
  `0.005`, replay size 100,000, batch 256, automatic entropy
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

The policy never receives world pose, path length, lap state, yellow-line
clearance, termination flags, or other reward/evaluation truth.

## Pre-training gates

The reward audit passed before SAC training. On two development seeds:

| Policy | Lap success | Mean return | Safety outcome |
|---|---:|---:|---|
| Random | 0% | -8.455 | 50% invalid pose, 50% yellow crossing |
| Always stop | 0% | -3.151 | safe but timed out |
| Simple controller | 100% | 22.256 | no safety event |

The online smoke run completed 128 steps, 97 gradient updates, one full
episode, and an exact checkpoint reload. The pre-training gate independently
verified CUDA, reward provenance, source/config hashes, seed isolation, test
results, and the exact W&B destination.

- W&B smoke: <https://wandb.ai/vnv/DuckiePOMDP/runs/iz0pipsf>
- Evidence: `artifacts/f10_l1/pretraining_gate.json`

## Training

The single declared training run completed all 60,000 steps and produced six
checkpoints. Learning was already visible by step 20,000: the development
policy moved from 0/4 laps with yellow crossings at step 10,000 to 4/4 safe
laps at steps 20,000 through 60,000.

- W&B full run: <https://wandb.ai/vnv/DuckiePOMDP/runs/z39mxtvl>
- Completed episodes: 118
- Checkpoints: every 10,000 steps
- Final training checkpoint: step 60,000, SHA256
  `44733961643dd3ca8435310c2b9ca35d0d019812aef1432bdd64fe1a3896be63`

An earlier launcher-limited process stopped at step 1,000 before learning
started. It is explicitly marked aborted in
`artifacts/f10_l1/aborted_timeout_step1000/` and is not part of the evidence.

## Safety-first checkpoint selection

All checkpoints were evaluated only on development seeds. The predeclared
rule first filtered unsafe checkpoints, then ranked safe checkpoints by lap
success, lower lane error, and return.

Step 50,000 was selected. It was also the best-return checkpoint, while step
60,000 remains the last checkpoint for provenance only.

- Selected checkpoint: `artifacts/f10_l1/sac_lane_baseline.pt`
- Selected step: 50,000
- SHA256: `7d492fbff98fca9200266743151c849dd323a7f3259425e4d13eaa3a0ac32f72`

Only `sac_lane_baseline.pt` is the selected lane-stage checkpoint.
`lane_best_return.pt` and `lane_last.pt` are audit copies, not additional
deployment policies.

## One-shot final evaluation

The frozen checkpoint was evaluated once on the untouched final seeds.

| Metric | Random | Always stop | Simple controller | SAC |
|---|---:|---:|---:|---:|
| Lap success | 0% | 0% | 100% | **100%** |
| Invalid-pose rate | 25% | 0% | 0% | **0%** |
| Yellow-crossing rate | 25% | 0% | 0% | **0%** |
| Lane-departure rate | 50% | 0% | 0% | **0%** |
| Mean return | -7.632 | -3.648 | 22.229 | **30.973** |
| Mean path length | 0.615 m | 0.000 m | 5.440 m | **5.026 m** |
| Mean absolute lateral error | 0.0320 m | 0.0119 m | 0.0585 m | **0.0106 m** |
| Mean episode p95 `|d|` | 0.0970 m | 0.0119 m | 0.1012 m | **0.0250 m** |
| Mean absolute heading error | 0.2168 rad | 0.0262 rad | 0.1010 rad | **0.0916 rad** |
| Minimum yellow clearance | -0.0010 m | 0.0266 m | 0.0258 m | **0.0249 m** |
| Mean actual velocity | 0.1285 m/s | 0.0000 m/s | 0.1327 m/s | **0.1398 m/s** |
| Mean absolute yaw command | 2.0246 rad/s | 0.0000 | 0.3879 rad/s | **1.1881 rad/s** |
| Mean normalized action change | 0.8036 | 0.0000 | 0.0029 | **0.5313** |

All ten pre-registered acceptance checks passed. SAC did not obtain success by
standing still and substantially reduced lane error relative to both random
control and the competent simple controller.

## Video evidence

`artifacts/f10_l1/sac_lane_demo.mp4` is a deterministic real-simulator proof
on development seed `14001`, rendered only after checkpoint selection. It uses
the selected SAC checkpoint and completes one lap in 1,073 steps (`35.77 s`):

- lap completed: yes;
- yellow crossing: no;
- lane departure: no;
- invalid pose: no;
- mean absolute lateral error: `0.01140 m`;
- p95 absolute lateral error: `0.02602 m`;
- minimum yellow clearance: `0.02536 m`.

The video labels yellow clearance and path length as `EVAL ONLY`; neither was
fed to SAC. Reproducibility metadata and hashes are in
`artifacts/f10_l1/sac_lane_demo.json`.

## Reward breakdown for the proof lap

| Component | Sum |
|---|---:|
| Progress | +24.6757 |
| Lane | -0.7391 |
| Yellow barrier | -0.0244 |
| Comfort | -0.8142 |
| Living | -2.1460 |
| Terminal lap bonus | +10.0000 |
| Total return | **30.9520** |

## Known limitations

1. This policy uses simulator-provided lane-relative ego measurements, not an
   RGB lane-perception model. It is a control curriculum checkpoint.
2. It is validated only on `small_loop`, counter-clockwise, without domain or
   dynamics randomization.
3. Four final seeds establish reproducibility for this narrow stage but do
   not establish broad generalization.
4. SAC is less smooth than the simple controller: its mean yaw magnitude and
   action change are higher. That is a later optimization target, not tuned
   post hoc here.
5. Its minimum yellow clearance is safe but approximately 0.9 mm smaller than
   the simple controller's final-set minimum.
6. This checkpoint does not include YOLO/EKF inputs and is not the final
   deployable POMDP checkpoint.

## Decision

**F10-L1 PASS.** The narrow driving prerequisite is satisfied: SAC can drive
the real `small_loop` counter-clockwise, remain in lane, stay clear of the
yellow line, and finish a lap reproducibly. No subsequent curriculum,
explanation, or optimization stage is started by this report.

Final active test suite: **411 passed, 0 failed, 0 skipped** (293 dependency
and simulator warnings).

