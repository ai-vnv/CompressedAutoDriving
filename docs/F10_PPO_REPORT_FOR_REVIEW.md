# F10-PPO C4 Report for Review

Date: 2026-08-14

## Outcome

F10-PPO C4 is **PASS** on real Gym-Duckietown 6.2.0.  The selected
belief-conditioned PPO policy completed the combined pedestrian-and-stop task
on all four once-only stage-final seeds.  No global-final, C5, F11, policy
explanation, or post-final tuning was run.

| Stage | Classification | Evidence lineage |
|---|---|---|
| C0 small_loop | PASS | camera lane belief V9 |
| C1 experiment_loop | PASS | camera lane belief V9 |
| C2 pedestrian | PASS | frozen YOLO + F9c belief, V15 |
| C3 stop | PASS | frozen YOLO + stop belief, V20 predecessor |
| C4 combined | **PASS** | existence-gated public belief, V30 |

The runtime path is:

```text
front RGB
  -> frozen MobileNet lane estimator + lane belief
  -> frozen YOLO11n -> metric projection -> F9c pedestrian belief
  -> frozen YOLO11n -> metric projection -> stop belief
  -> fixed 29D public policy vector
  -> PPO actor and critic
  -> [v_cmd, omega_cmd]
  -> existing differential-drive adapter
```

Privileged simulator truth is used only for offline teacher labels, reward,
and evaluation.  It is not present in the behavior NPZ or the runtime
actor/critic input.

## Frozen protocol

- Configuration: `configs/f10_ppo_visual_objects_v30.toml`
- Configuration SHA256:
  `85e2cbd321e2db53de270e3c0b885a723137d17e94af3640ed5a8a9f917fe829`
- Observation: fixed 29D public lane/ego/road/pedestrian/stop/previous-action
  representation.
- Actor and critic: independent `29 -> 256 -> 256` MLPs.
- Action: normalized continuous two-vector mapped once to
  `v_cmd in [0, 0.4] m/s`, `omega_cmd in [-4, 4] rad/s`.
- C4 training budget: 4,096 simulator steps, checkpoints every 1,024 steps.
- C4 PPO override: learning rate `2e-7`, one epoch, clip `0.01`, entropy
  coefficient `0`, max gradient norm `0.02`, target KL `0.002`.
- Training seeds: `175001-175012`.
- Development seeds: `175101-175104`.
- Once-only stage-final seeds: `175201-175204`.

The V30 correction gives the complete pedestrian kinematic slice neutral
absent semantics when `P(e_ped) < 0.4`.  At and above `0.4`, the public F9c
belief is preserved unchanged.  This prevents stale conditional pedestrian
kinematics from steering the car after existence has collapsed.

## Offline teacher and PPO training

The teacher used privileged state only offline.  Its distilled student sees
only the same 29D public representation used at deployment.  The frozen
student retained the V22 driving/stop actor, used the teacher-guided critic,
and updated only the actor's pedestrian-input columns with neutral-behavior
preservation.  Step zero passed training-only C2/C3/C4 gates but was ineligible
for final selection.

The substantive PPO run then executed 4,096 real-simulator steps.  It produced
four eligible updated checkpoints and ended at 115 cumulative PPO updates.
Checkpoint reload was exact.  The online W&B run is
`vnv/DuckiePOMDP/2kp6inzo`.

At the last PPO update, explained variance was `0.66884`, approximate KL was
`9.29e-8`, and clip fraction was zero.  These values reflect the deliberately
conservative C4 fine-tuning contract.

## Development selection

All four updated candidates passed every frozen safety and C4 skill check on
the same development seeds.  Safety-first selection chose step 1,024.

| Step | Completion | Stop/restart | Collision | Unsafe | Violation | Lane failure | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 100% / 100% | 0% | 0% | 0% | 0% | no |
| 1,024 | 100% | 100% / 100% | 0% | 0% | 0% | 0% | **yes, selected** |
| 2,048 | 100% | 100% / 100% | 0% | 0% | 0% | 0% | yes |
| 3,072 | 100% | 100% / 100% | 0% | 0% | 0% | 0% | yes |
| 4,096 | 100% | 100% / 100% | 0% | 0% | 0% | 0% | yes |

Selected checkpoint:

```text
artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt
SHA256 02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250
```

## Retention matrix

The selected C4 checkpoint was evaluated on every previous development task.
All pre-registered retention checks passed.

| Task | Completion | Collision | Unsafe | Stop completion | Lane failure | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| C0 small_loop | 100% | 0% | 0% | N/A | 0% | 0% |
| C1 experiment_loop | 25% | 0% | 0% | N/A | 75% | 0% |
| C2 pedestrian | 75% | 0% | 0% | N/A | 0% | 25% |
| C3 stop | 100% | 0% | 0% | 100% | 0% | 0% |
| C4 combined | 100% | 0% | 0% | 100% | 0% | 0% |

C1 remains a material limitation: only one of four varied object-free
`experiment_loop` starts completed, with three lane failures.  This is not a
new C4 regression under the frozen retention rule, but it prevents claiming
universal driving competence.

## Once-only C4 stage-final result

The selected checkpoint was evaluated once on seeds `175201-175204` without
checkpoint selection or retraining.

| Metric | Random | Always stop | Simple controller | PPO |
|---|---:|---:|---:|---:|
| Completion | 0% | 0% | 100% | **100%** |
| Mean progress | 0.4107 m | 0.0000 m | 7.2600 m | **7.2097 m** |
| Stop completion | 0% | 0% | 100% | **100%** |
| Restart | 0% | 0% | 100% | **100%** |
| Stop violation | 0% | 0% | 0% | **0%** |
| Collision | 0% | 0% | 0% | **0%** |
| Unsafe episode | 0% | 0% | 0% | **0%** |
| Lane failure | 0% | 0% | 0% | **0%** |
| Invalid pose | 100% | 0% | 0% | **0%** |
| Timeout | 0% | 100% | 0% | **0%** |

Additional PPO metrics:

- minimum pedestrian clearance: `0.49514 m`;
- mean absolute lateral error: `0.01969 m`;
- mean absolute heading error: `0.06045 rad`;
- mean commanded velocity: `0.13042 m/s`;
- mean absolute commanded yaw rate: `0.42270 rad/s`;
- mean action change: `0.00529`;
- stationary fraction: `18.50%`;
- mean return: `38.2107`.

Every C4 skill check, the safety gate, and the retention gate passed;
`progression_permitted=true` is recorded.  The sample count is four episodes,
so these percentages are stage-gate evidence rather than a population-wide
reliability claim.

## Visual evidence and dataset audit

- Front RGB + evaluation-only BEV video (development seed 175101):
  `artifacts/f10_ppo_visual_objects_v30/c4/video/c4_selected_front_bev_seed175101.mp4`
  (`2,367` frames, `30 fps`, `78.9 s`, completed lap, no collision; SHA256
  `de6bd6ed0a72bfd011c75f41cbffb1d46ec000d6221c9c16971c79553f1bc755`).
- Teacher/rehearsal NPZ audit:
  `artifacts/f10_ppo_visual_objects_v30/c4/behavior/privileged_teacher_dataset_audit.png`
  and `.pdf`.  It visualizes 26,822 public 29D observations and two-dimensional
  actions across five auditable source roles.  Privileged labels remain only
  in the offline source CSV, not in the NPZ.

## Evidence hashes

- Development metrics: `86c34cf37a5c363fc4ddec136d800e019b821d88ea7a639633436a17ff74a0e1`
- Retention metrics: `03099e448c49e1fa41bbc748627e450de3663a261427c07244b748f595f5b8cd`
- Stage-final metrics: `d6d5c33807ad5bdd32a5f864c9b54ba92043960129527f9115b6d71370d06fea`
- Stage-final episodes: `d2394b4a712e07d870f38727f6b995494d63bbddb573fd2f7e55d698e4faca17`
- Dataset audit PNG: `08f82c895f80f09e58c44464896603ffe6c7dd3016cd7d6eb9e8a526de116255`
- Dataset audit PDF: `8cb7db5ae8d9c934d61d5ae6c1c05408a72adf13a80ca362dddedde92d586364`

The post-stage full suite passed **621 tests, 0 failed, 0 skipped**.  The only
post-training code change was removal of a four-role color limit in the
dataset-audit renderer; policy, perception, belief, reward, and evaluation
logic were unchanged.  The historical pretraining gate intentionally retains
the exact renderer hash present at launch; the post-stage change is recorded
separately in
`artifacts/f10_ppo_visual_objects_v30/poststage_source_amendment.json`.

## Classification

**F10-PPO C4: PASS.**

This completes the requested C4 stage.  No global-final or later research
stage was started.
