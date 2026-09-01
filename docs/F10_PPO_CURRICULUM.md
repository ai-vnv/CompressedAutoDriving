# F10-PPO Curriculum Ledger

Full non-smoke C0 training is fail-closed behind
`artifacts/f10_ppo/pretraining_gate.json`. The training entry point re-hashes
the frozen configuration, implementation sources, and gate evidence before it
constructs the simulator.

The mandatory order within each stage is:

```text
train → development checkpoint selection → retention → stage-final once
```

Only a `PASS` stage with `progression_permitted=true` can initialize the next
stage. `LIMITED` and `FAILED` both stop this Version-1 curriculum. After C4
passes, `global-final` writes the untouched five-task result and the objective
forgetting matrix; `report` then creates `docs/F10_PPO_REPORT_FOR_REVIEW.md`.

This ledger is append-only during execution. A stage may start only after its
previous gate is recorded PASS (or protocol-permitted LIMITED).

| Gate | Required evidence | Status |
|---|---|---|
| P0 formulation | Config, fixed 29D camera/belief observation, seeds, criteria frozen | PASS |
| P1 implementation | Observation/reward/env/PPO tests and reward audit | PASS |
| P2 smoke | Real RGB pipeline, PPO gradients, reload, W&B witness | PASS |
| C0 | small_loop dev selection + once-only stage final | PASS (`lane_belief_v9`) |
| C1 | experiment_loop + C0 retention | PASS (`lane_belief_v9`) |
| C2 | pedestrian + C0/C1 retention | PASS (`visual_objects_v15`) |
| C3 | stop + C0/C1/C2 retention | PASS (`visual_objects_v18`) |
| C4 | combined + complete retention matrix | PASS (`visual_objects_v30`) |
| GF | once-only global final | NOT RUN (outside frozen V30 scope) |

No later gate may be populated from mock-only evidence where real simulator or
real YOLO/EKF behavior is required.

The current C2/C3 evidence and scientific limitations are recorded in
`docs/F10_PPO_C2_C3_REPORT_FOR_REVIEW.md`. C3 selected its DAgger-distilled
step-zero PPO network; all C3 on-policy update checkpoints failed development.
C3 therefore passes the frozen behavior gate, but it is not evidence that
reward-only PPO fine-tuning acquired stop behavior. C1 retention at the C4
checkpoint is also limited (25% completion). C4 subsequently passed the
frozen V30 combined-task gate using the existence-gated public belief and an
eligible step-1024 PPO checkpoint. The result and limitations are recorded in
`docs/F10_PPO_REPORT_FOR_REVIEW.md`. The V30 protocol explicitly stopped after
the once-only C4 stage-final, so global-final was not run.
