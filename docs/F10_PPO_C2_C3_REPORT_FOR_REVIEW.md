# F10-PPO C2-C3 Report for Review

Date: 2026-08-13

## Outcome

The isolated pedestrian stage C2 and stop-sign stage C3 both satisfy their
frozen stage gates on real Gym-Duckietown. This report stops at C3; C4 was not
started.

| Stage | Classification | Selected checkpoint | Selected step |
|---|---|---|---:|
| C2 pedestrian | PASS | `artifacts/f10_ppo_visual_objects_v15/c2/ppo_selected.pt` | 0 |
| C3 stop sign | PASS | `artifacts/f10_ppo_visual_objects_v18/c3/ppo_selected.pt` | 0 |

The runtime actor and critic consume the same fixed 29-dimensional public
representation. Lane state is estimated from front RGB by the frozen
MobileNet lane estimator and lane EKF. Duckie and stop-sign measurements come
from the frozen YOLO detector, metric projection, and their belief updaters.
Privileged simulator state is read only after the policy vector is built and
is used for reward/evaluation, not actor or critic input.

## C2 pedestrian result

The once-only C2 final seeds were `160201-160204`. The selected checkpoint has
SHA256
`e997f9bd4b1e0f86e622faead6617ad6cf42cb7b6182309544202172f3700042`.

| Metric | C2 PPO |
|---|---:|
| Completion | 50% (2/4) |
| Mean progress | 7.8763 m |
| Collision episodes | 0% |
| Unsafe-proximity episodes | 0% |
| Minimum pedestrian clearance | 0.2325 m |
| Lane-failure episodes | 25% |
| Invalid-pose episodes | 0% |
| Timeout episodes | 25% |
| Mean commanded velocity | 0.1546 m/s |
| Stationary fraction | 6.63% |

All frozen C2 skill checks passed: collision, unsafe interaction, progress,
and non-stationary behavior. The retention gate passed.

## C3 training and checkpoint selection

The V18 C3 run executed 40,960 real-simulator steps and reached 150 total PPO
updates. The final checkpoint reload was exact. The online W&B run is
`vnv/DuckiePOMDP/g9x2c3c2`.

The DAgger warm start used 29,559 public-belief rows from training-only
trajectories. Development selection used the disjoint seeds
`166101-166104` and selected step 0:

| Candidate step | Completion | Stop completion | Stop violation | Invalid pose | Eligible |
|---:|---:|---:|---:|---:|---|
| 0 | 100% | 100% | 0% | 0% | yes |
| 10,240 | 0% | 0% | 100% | 0% | no |
| 20,480 | 0% | 100% | 0% | 100% | no |
| 30,720 | 0% | 100% | 0% | 100% | no |
| 40,960 | 0% | 0% | 100% | 0% | no |

This is an important negative result: the competent C3 policy is the DAgger-
distilled PPO network before the C3 on-policy updates. The full PPO fine-tuning
run was valid and reproducible, but every updated checkpoint failed the frozen
development gate. The result demonstrates a competent policy interface and
stop behavior, not successful reward-only acquisition of stop behavior during
C3 PPO fine-tuning.

## C3 once-only final result

The once-only C3 final seeds were `166201-166204`. The selected checkpoint has
SHA256
`d058caee49a20bbd9f51e41bf2a41fe63c52849e57eb8be72bd2ab4df264f53c`.

| Metric | Random | Always stop | Simple controller | PPO |
|---|---:|---:|---:|---:|
| Completion | 0% | 0% | 100% | 100% |
| Mean progress | 0.5698 m | 0.0000 m | 7.2481 m | 7.2298 m |
| Stop completion | 0% | 0% | 100% | 100% |
| Stop violation | 25% | 0% | 0% | 0% |
| Restart after stop | 0% | 0% | 100% | 100% |
| Collision | 0% | 0% | 0% | 0% |
| Lane failure | 25% | 0% | 0% | 0% |
| Invalid pose | 75% | 0% | 0% | 0% |
| Timeout | 0% | 100% | 0% | 0% |

Additional PPO metrics:

- mean absolute lateral error: 0.02445 m;
- mean absolute heading error: 0.06601 rad;
- mean commanded velocity: 0.15372 m/s;
- mean absolute commanded yaw rate: 0.47926 rad/s;
- stationary fraction: 0.591%;
- mean return: 39.2662.

C3 safety, stop completion, stop violation, restart, and retention checks all
passed. `progression_permitted=true` is recorded, but this task deliberately
does not start C4.

## Retention matrix at C3

All figures below use development seeds. The frozen C3 retention rule passes
because C2 collision rate remains 0%, equal to the imported C2 baseline.

| C3 checkpoint evaluated on | Completion | Collision | Lane failure | Invalid pose |
|---|---:|---:|---:|---:|
| C0 small_loop | 100% | 0% | 0% | 0% |
| C1 experiment_loop | 25% | 0% | 75% | 0% |
| C2 pedestrian | 50% | 0% | 0% | 0% |
| C3 stop | 100% | 0% | 0% | 0% |

The 25% C1 completion and 75% C1 lane-failure rate are a material retention
limitation. The pre-registered C3 retention gate tests C2 safety rather than
requiring full C1 completion, so the formal C3 classification remains PASS;
the policy should not yet be described as universally competent on every
object-free `experiment_loop` start.

## Evidence

- C2 final: `artifacts/f10_ppo_visual_objects_v15/c2/stage_final_metrics.json`
  (SHA256 `c5b8fc97092e75dc8221326c8b6b4a75ade4be3dd20d12a6d1ca9915af1762e1`)
- C3 training manifest:
  `artifacts/f10_ppo_visual_objects_v18/c3/training/training_run_manifest.json`
  (SHA256 `8d922ee2d8a1171e5199e3db5b12d9aa32561699ccbb65797d6d86438cfdd478`)
- C3 development:
  `artifacts/f10_ppo_visual_objects_v18/c3/development_metrics.json`
  (SHA256 `41117495a86b7f1b2bcf6363fb3c5865e6679f2db38f2fced1831acf0025d713`)
- C3 retention:
  `artifacts/f10_ppo_visual_objects_v18/c3/retention_metrics.json`
  (SHA256 `7b67bab1c5be20adb05a469785be698862ad0bc59dcb79d44f44b6e902416a7f`)
- C3 once-only final:
  `artifacts/f10_ppo_visual_objects_v18/c3/stage_final_metrics.json`
  (SHA256 `714e2cd7d03e759f1a2fafdd429bfdf2fbc39bf17e4367890e114a68b20be84e`)
- C3 proof video:
  `artifacts/f10_ppo_visual_objects_v18/c3/video_unobstructed/c3_front_yolo_stop_belief_bev_seed166101.mp4`
  (SHA256 `a786b0537a99b526c9287f567513464310e9bcfaa47ed7b18ec967e337e6dfb7`)

The active pretraining suite passed 576 tests with zero failures and zero
skips. No runtime source changed after that frozen witness; subsequent work
created training/evaluation/video artifacts and this report only.

