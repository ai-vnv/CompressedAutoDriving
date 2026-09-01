# F10 Report for Review — Baseline Visuomotor SAC

## Classification

**LIMITED.** The complete leak-free RGB -> YOLO -> robust belief -> SAC ->
chassis-command pipeline trained, checkpointed, reloaded, and completed the
pre-registered one-shot final evaluation. The selected SAC policy is not a
competent driving baseline: it failed six of seven quantitative acceptance
checks. F11 explanation must not begin from this checkpoint as if F10 had
passed.

No final-evaluation result was used to retrain, retune, or select a different
checkpoint.

## Frozen protocol and provenance

- Config: `configs/f10_sac_v1.toml`
- Config SHA256: `bef9e18d26a1d8b3e0f47db7f5e5c5dab5e41d575154d167ee972516f695a013`
- YOLO SHA256: `3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`
- F9c belief config SHA256: `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`
- Action config SHA256: `80154e4ff22d4d9be6ebc1d6bfcd2f7d29caa3458c18ab7859420d2940c4d94a`
- Training seeds: `10001--10018`
- Development seeds: `11001--11006`
- Final seeds: `12001--12006`
- Training budget: 20,000 environment steps, 19,001 gradient updates, 167
  completed episodes, five periodic checkpoints
- W&B run: `vnv/DuckiePOMDP`, run `kgmucdzw`
- W&B URL: <https://wandb.ai/vnv/DuckiePOMDP/runs/kgmucdzw>

The seed sets are mutually disjoint and exclude the historical detector and
belief-evaluation seeds recorded in the training manifest.

## SAC configuration

F10 used the predeclared canonical PyTorch SAC: 17-dimensional fixed-scale
policy observation, two-dimensional normalized action, `256 x 256` ReLU
networks, learning rate `3e-4`, gamma `0.99`, tau `0.005`, batch size `256`,
replay capacity `100,000`, learning starts at step `1,000`, one update per
environment step, automatic entropy tuning with target entropy `-2`, seed
`10000`, and CUDA on an RTX 4060 Laptop GPU.

The only action mapping remained:

```text
SAC [-1,1]^2 -> PolicyAction(v_cmd in [0,0.4], omega_cmd in [-4,4])
              -> existing DifferentialDriveActionAdapter
              -> Gym-Duckietown
```

## Development checkpoint selection

All five checkpoints were evaluated on the six development seeds. The frozen
safety filter rejected collision rate above `0.20` or invalid-pose rate above
`0.25`, then ranked eligible checkpoints by success, progress, return, and
latest step.

| Step | Success | Progress (m) | Return | Collision | Invalid pose | Eligible |
|---:|---:|---:|---:|---:|---:|---|
| 4,000 | 0.000 | 0.605 | -3.308 | 0.000 | 0.667 | no |
| 8,000 | 0.000 | 0.381 | -8.468 | 0.000 | 0.167 | yes |
| 12,000 | 0.000 | 0.182 | -5.075 | 0.000 | 0.000 | yes |
| 16,000 | 0.000 | 0.424 | -3.038 | 0.000 | 1.000 | no |
| 20,000 | 0.000 | 0.287 | -0.748 | 0.000 | 1.000 | no |

The official `sac_baseline.pt` is step 8,000, SHA256
`79b98acb8f28f042622dc77fd92d3b711a0c81d8406af5d29309c8378124c463`.
`best_return.pt` and `last.pt` both point to step 20,000, SHA256
`34fad09c71484ec42bd8bb9c68c0453011425e84f1562fa68d2307a4923f1152`.
They remain diagnostics and were not substituted into final evaluation.

## One-shot final evaluation

Each policy ran once on each of the six final seeds. The table therefore
contains six episodes per policy.

| Metric | Random | Always stop | Simple controller | SAC baseline |
|---|---:|---:|---:|---:|
| Success rate | 0.000 | 0.000 | 0.000 | 0.000 |
| Mean progress (m) | 0.660 | 0.000 | 0.640 | 0.290 |
| Mean return | -9.412 | -0.407 | 3.227 | -9.296 |
| Collision rate | 0.000 | 0.000 | 0.000 | 0.000 |
| Invalid-pose rate | 0.333 | 0.000 | 0.167 | 0.167 |
| Timeout rate | 0.667 | 1.000 | 0.833 | 1.000 |
| Stop completion rate | 0.000 | 0.000 | 1.000 | 0.333 |
| Stop violation rate | 1.000 | 0.000 | 0.000 | 0.333 |
| Lane-departure episode rate | 0.500 | 0.000 | 0.000 | 0.833 |
| Unsafe-proximity episode rate | 0.333 | 0.000 | 0.000 | 0.000 |
| Minimum pedestrian proxy clearance (m) | 0.105 | 0.783 | 0.278 | 0.282 |
| Mean absolute lateral error (m) | 0.075 | 0.013 | 0.020 | 0.095 |
| Mean absolute heading error (rad) | 0.293 | 0.023 | 0.074 | 0.704 |
| Mean velocity command (m/s) | 0.193 | 0.000 | 0.170 | 0.145 |
| Mean absolute yaw command (rad/s) | 2.011 | 0.000 | 0.083 | 2.276 |
| Mean action change | 0.780 | 0.000 | 0.012 | 0.292 |
| Mean steering oscillations | 77.3 | 0.0 | 0.0 | 25.7 |
| Pedestrian speed response (m/s) | +0.0067 | N/A | -0.1127 | -0.0023 |

The pedestrian clearance is the documented V1 ego-reference-point to
pedestrian-footprint proxy, not full footprint-to-footprint clearance.

### SAC scenario breakdown

| Scenario | N | Progress (m) | Return | Stop complete | Stop violation | Lane departure | Invalid pose |
|---|---:|---:|---:|---:|---:|---:|---:|
| left-to-right crossing | 2 | 0.011 | -8.297 | 0.000 | 0.000 | 0.500 | 0.500 |
| right-to-left crossing | 2 | 0.504 | -11.338 | 0.000 | 1.000 | 1.000 | 0.000 |
| stationary pedestrian | 2 | 0.356 | -8.252 | 1.000 | 0.000 | 1.000 | 0.000 |

No policy reached the `1.25 m` success condition. The deterministic controller
nevertheless provides a strong reward/behavior sanity witness: it completed
the stop in all six episodes, had no lane-departure episode, responded to the
pedestrian, and outscored SAC without receiving privileged information in its
observation.

## SAC reward decomposition

Mean episode components for the selected SAC checkpoint were:

| Component | Mean |
|---|---:|
| Progress | +1.0915 |
| Lane | -7.8311 |
| Stop | -1.9727 |
| Pedestrian | -0.0617 |
| Comfort | -0.3550 |
| Terminal | -0.1667 |

The dominant failure is lane/heading control, followed by incomplete stop
behavior. Collision avoidance alone is not sufficient evidence of a competent
policy because the policy made little progress and timed out in every episode.

## Acceptance decision

| Predeclared check | Result |
|---|---|
| Success rate >= 0.40 | FAIL |
| Mean progress >= 0.75 m | FAIL |
| Progress gain over always-stop >= 0.40 m | FAIL |
| Progress gain over random >= 0.15 m | FAIL |
| Collision rate <= 0.20 | PASS |
| Lane-departure rate <= 60% of random | FAIL |
| Safety-region speed at least 0.03 m/s below clear-region speed | FAIL |

The policy is therefore **LIMITED**: the implementation and experimental
pipeline are valid, but competence is insufficient for normal F11 explanation
experiments or deployment.

## Reproducible artifacts

- `artifacts/f10/training_metrics.csv`: 20,000 transition rows
- `artifacts/f10/episode_metrics.csv`: 167 completed training episodes
- `artifacts/f10/training_run_manifest.json`
- `artifacts/f10/dev_metrics.json`
- `artifacts/f10/development_episodes.csv`: 30 checkpoint episodes
- `artifacts/f10/checkpoint_manifest.json`
- `artifacts/f10/sac_baseline.pt`
- `artifacts/f10/final_metrics.json`
- `artifacts/f10/evaluation_episodes.csv`: 24 final policy episodes
- `artifacts/f10/final_tests.xml`

The exact checkpoint reload was verified after training. All training telemetry
was also synchronized to W&B; local files remain authoritative.

## Tests and limitations

The complete suite passed: **397 passed, 0 failed, 0 skipped** (292 warnings).

Known limitations:

- the deliberately modest 20,000-step budget did not produce a competent SAC
  policy;
- final evidence contains only six episodes per policy and two per pedestrian
  mode;
- development success was zero for all checkpoints, so the safety-first tie
  was resolved using progress and return;
- simulator invalid-pose remains a separately reported outcome, not a proven
  collision;
- the reward's pedestrian clearance is a V1 proxy;
- no final-set retuning, alternate checkpoint substitution, F11 explanation,
  or F12 optimization was performed.

## Stop decision

F10 stops here as required. The next scientific decision is whether to define
a new, separately pre-registered competence-improvement gate or to inspect the
existing failure telemetry. F11 should not start automatically from this
LIMITED checkpoint.
