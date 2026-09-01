# F10 Pre-training Gate Report

## Decision

**READY FOR FULL TRAINING.** This report closes only the pre-training gate. No
20,000-step SAC run, development checkpoint selection, or final evaluation has
been executed.

The machine-readable decision is
`artifacts/f10/pretraining_gate.json` (SHA256
`39bcdc7abe7ea2ac3ac7daa9aa9f8abb63463270aaf130c69be8a7230130ecdc`).

## Frozen protocol

- F10 config SHA256:
  `bef9e18d26a1d8b3e0f47db7f5e5c5dab5e41d575154d167ee972516f695a013`
- YOLO checkpoint SHA256:
  `3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`
- F9c belief config SHA256:
  `359dc52020421c248bf6c26e036234191b2b97d24b505a66d85daa85b563704e`
- Action config SHA256:
  `80154e4ff22d4d9be6ebc1d6bfcd2f7d29caa3458c18ab7859420d2940c4d94a`
- Train/dev/final seeds remain disjoint and historical evaluation seeds are
  excluded.

The full training command is guarded by the manifest. If the config or
training implementation changes, `train_f10_sac.py` refuses to start until a
new gate is produced.

## Evidence

| Check | Result |
|---|---|
| Reward audit | PASS; safe controller beats random/always-stop and reckless forward is not best |
| SAC smoke | PASS; 96 transitions, 65 gradient updates, exact deterministic checkpoint reload |
| Evaluator smoke | PASS; real simulator, 180 steps, development seed 11001 only |
| W&B online preflight | PASS; `vnv/DuckiePOMDP`, run `7sorajsj` |
| CUDA | PASS; RTX 4060 Laptop GPU, Torch 2.12.1+cu130 |
| Credential isolation | PASS; no W&B API credential in project files |
| Full tests | 397 passed, 0 failed, 0 skipped |

The untrained smoke checkpoint is not a performance result. Its evaluator run
correctly showed a stop violation and no success; it exists only to prove that
checkpoint loading and the complete RGB-to-evaluation path work before GPU
budget is spent.

## Frozen evaluation behavior

Five periodic checkpoints will be evaluated on development seeds at steps
4,000, 8,000, 12,000, 16,000, and 20,000. The predeclared selection first
rejects collision rate above 0.20 or invalid-pose rate above 0.25, then ranks
eligible candidates by success, progress, return, and finally latest step.

The evaluator will preserve three auditable aliases:

- `last.pt` — final training step;
- `best_return.pt` — highest development return, diagnostic only;
- `sac_baseline.pt` — official safety-first checkpoint.

Only `sac_baseline.pt` is eligible for final evaluation, deployment, and F11.
The final evaluator does not perform model selection and refuses to overwrite
an existing final result.

## W&B monitoring

The main run will log to `vnv/DuckiePOMDP` every ten environment steps and at
episode boundaries. Local CSV/JSON remains authoritative. Once training is
running, health checks will examine finite losses, divergence, entropy
coefficient, replay/update counts, episode progress, safety events, and GPU
activity. W&B authentication remains in the WSL user credential store, not in
the repository.

## Next command

```bash
python experiments/train_f10_sac.py --wandb-mode online
```

That command has not been run as part of this gate.
