# F10-PPO Visual-Lane v3 Report for Review

## Classification

```text
C0 small_loop: FAILED at development gate
C1 experiment_loop: NOT STARTED
```

The v3 experiment implemented the requested curve-recovery rule and completed
its full preregistered C0 training budget. It did not produce a checkpoint that
passed the frozen C0 development criteria. Stage-final seeds were not consumed,
and C1 was not launched.

## What changed from v2

Shallow yellow-line contact on a curve became a penalized, nonterminal recovery
event. Deep penetration, straight-line contact without curve context, and
failure to clear contact within 15 frames remained terminal. Recovery required
three consecutive contact-free frames, and lap completion was blocked while a
recovery was pending.

This used privileged curvature only inside reward/evaluation. PPO still saw the
same 29D camera-lane/YOLO/belief vector. The robot still started
counter-clockwise near heading `pi`.

## Pretraining evidence

- Reward audit: PASS. The simple controller completed 2/2 laps without yellow
  contact; always-stop did not complete; random was worse.
- Reset/memory: 36 resets, one integration, one simulator, 2.5 MiB steady
  growth/span.
- PPO smoke: 128 steps, two updates, exact checkpoint reload.
- Active suite: 472 passed, 0 failed, 0 skipped.
- Independent agent-follows-doc audit: PASS.
- Pretraining gate SHA256: `239f28e7790a33dfe62a9362f57da1d95be3af45777ae5ba4ada1aa2ded87a8e`.

## Training

```text
environment steps : 61,440
PPO updates       : 60
episodes          : 336
lap completions   : 0
checkpoints       : 6
checkpoint reload : PASS
```

W&B run: [oog0l05m](https://wandb.ai/vnv/DuckiePOMDP/runs/oog0l05m),
state `finished`.

The optimizer remained numerically stable. The absence of training lap
completion is a behavioral failure, not a crash, OOM, NaN, or broken pipeline.

## Development evaluation

All six checkpoints were evaluated deterministically on the same four new
development seeds `47101-47104`. None was eligible.

The frozen ranking selected step 40,960 only as a diagnostic candidate:

| Metric | Observed | Required |
|---|---:|---:|
| Completion | 25% | >=50% |
| Mean progress | 2.139 m | >=3.5 m |
| Lane-failure rate | 0% | <=25% |
| Invalid-pose rate | 75% | <=25% |
| Mean absolute lateral error | 0.0264 m | <=0.09 m |

Seed `47102` completed one lap, while `47101`, `47103`, and `47104` terminated
with invalid pose. The selected policy was smooth on average
(`|omega_cmd|=0.344 rad/s`) and did not touch the yellow line in these four
episodes, but it was not robust to the held-out start variations.

## Interpretation

The relaxed curve rule fixed the specific scientific mismatch in the terminal
definition, but it was not sufficient to make PPO robust. In the diagnostic
checkpoint, the dominant held-out failure shifted away from yellow-line contact
to invalid pose. Therefore this run does not justify C1 transfer.

No PPO hyperparameter, reward weight, seed, or perception parameter was changed
after inspecting training/development results. A later retry requires a new
reviewed protocol rather than tuning this development set.

## Artifacts

- `artifacts/f10_ppo_visual_v3/c0/training/training_run_manifest.json`
- `artifacts/f10_ppo_visual_v3/c0/development_metrics.json`
- `artifacts/f10_ppo_visual_v3/c0/development_episodes.csv`
- `artifacts/f10_ppo_visual_v3/c0/development_gate_result.json`
- Diagnostic checkpoint: `artifacts/f10_ppo_visual_v3/c0/ppo_selected.pt`

The stage-final and C1 artifact directories are intentionally absent.
