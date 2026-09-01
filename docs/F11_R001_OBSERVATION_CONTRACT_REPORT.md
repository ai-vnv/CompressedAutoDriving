# F11 R001 — Observation/Belief Contract Audit

## Classification

**PASS**

R001 verifies the frozen deployment boundary before any final attribution or
counterfactual experiment:

```text
front RGB
→ MobileNetV3-small lane belief
→ YOLO + metric projection + pedestrian/stop belief
→ public 29D policy representation
→ deterministic PPO actor distribution mean
→ physical [v_cmd, omega_cmd]
```

The policy explanation target is the deterministic actor mean before stochastic
sampling. MobileNet and YOLO remain measurement provenance; they are not the
direct target of PPO attribution.

## Frozen system

- PPO checkpoint: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`
- PPO stage/global step: `c4` / `1024`
- Lane checkpoint: `91d471d5ccf9875012d564fa8937838fd0f95e6e3e6aabaefcad654d9b4bb84f`
- YOLO checkpoint: `3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`
- R001 seed/frames: `177000` / `128`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

## Contract result

The exact 29D ordering and physical normalization reconstructed from the public
semantic mapping match the environment observation. The policy vector is built
before privileged simulator state is read in both `reset()` and `step()`.
The stored trace contains no privileged/evaluation truth.

The primary explanation groups form one exact, non-overlapping partition:

```text
Lane | Ego | StopLine | Pedestrian | Stop | PreviousAction
```

Group attribution is the registered primary result. Individual 29-feature
attribution is secondary/appendix evidence.

## Numerical replay evidence

| Check | Maximum absolute error | Frozen tolerance |
|---|---:|---:|
| Public mapping → normalized 29D | `5.9605e-08` | diagnostic |
| Direct actor API → deterministic policy action | `0.0` | `1.0e-06` |
| Normalized action → physical action | `2.3283e-11` | `1.0e-06` |
| Fresh checkpoint replay: actor mean | `2.3842e-07` | `1.0e-06` |
| Fresh checkpoint replay: critic value | `3.7253e-07` | `1.0e-06` |

The model state hash was identical before and after the audit:
`9106ba45ed8a1ea7e0f112e2d607bf3a7b1a4942a5bcc834a6a0e0064a8c6a9b`.

Runtime evidence contains 128 perception, lane, and YOLO frames, with 124
distinct RGB frame hashes.

## Artifacts and reproducibility

- Contract result: `artifacts/f11_ppo_explanation_v2/r001/contract_audit.json`
- Public-only trace: `artifacts/f11_ppo_explanation_v2/r001/public_trace.npz`
- Trace manifest: `artifacts/f11_ppo_explanation_v2/r001/trace_manifest.json`
- Execution log: `artifacts/f11_ppo_explanation_v2_r001.log`
- Full tests: `artifacts/f11_ppo_explanation_v2/r001/full_tests.log`
- Trace SHA256: `f8b9684abcd7e4ea1710126c2f6caf0c0a0bd3c666644237925ae97b834c94b8`
- Full-test log SHA256: `90c4fb5626d02759f15b40851caa7879fefbb5649382f273b482d33aac6e8941`

Read-only verification command:

```bash
export PYTHONHASHSEED=0
export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src
/home/pannntastic/aivnv/duckie/.venv/bin/python \
  experiments/verify_f11_r001_contract.py \
  --config configs/f11_ppo_explanation_v2.toml
```

Full suite: **640 passed, 0 failed, 0 skipped** (426 warnings).

## Disclosed failed audit attempt

The first audit attempt stopped before collecting a simulator step because a
leakage-name guard matched the substring `iou` inside the legitimate word
`previous`. The guard was corrected to token-level IoU matching, covered by a
regression test, and the failed log was retained as
`artifacts/f11_ppo_explanation_v2_r001_attempt1_failed_false_positive.log`.
This was an audit-code false positive, not a runtime-policy failure.

## Gate consequence

R001 is complete. R002 (baseline/phase robustness) and R003 (semantic
intervention validation) remain unexecuted. R004 final attribution remains
blocked until R002 and R003 both pass. No Integrated Gradients or semantic
counterfactual result was generated in R001.
