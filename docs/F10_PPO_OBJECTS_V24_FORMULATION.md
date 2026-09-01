# F10-PPO V24 — One-episode privileged C4 guidance

V23 remains a frozen training-only negative result: its step-zero actor kept
C3 but failed the C2 unsafe gate and one of two C4 stop trials. No V23
development or final seed was used.

V24 adds exactly one guided C4 episode before on-policy PPO. The teacher may
read simulator truth, but only to produce action labels. The student input is
still the independently constructed 29D runtime belief:

```text
front RGB -> visual lane belief + YOLO/F9c pedestrian/stop beliefs -> 29D
                                                                     | student actor
simulator truth -> offline teacher -> action label ------------------+
environment reward -> discounted return label ----------------------| student critic
```

Neither simulator truth nor a truth-corrected belief is stored in the NPZ.
The guided source CSV records teacher truth in explicitly named audit columns;
the runtime loader consumes only `observations`, `actions`, `weights`,
`value_targets`, and `value_weights` from the NPZ.

The actor warm start retains the V23 C2/C3/C4 public-belief rehearsal data,
with C2 and C3 increased to 4x role mass and each C4 role at 1x. The single
privileged guided episode has 1x actor mass. Only its 1,947 public-belief rows
supervise the critic using gamma-0.99 discounted environment returns. Actor
and critic remain 29D feed-forward PPO networks at deployment; the teacher is
absent.

Before substantive PPO, step zero must pass the frozen C2, C3, and C4 gates on
training-only seeds. The PPO phase remains 4,096 steps with checkpoints every
1,024 steps, learning rate 2e-6, one epoch, clip 0.02, entropy coefficient 0,
gradient norm 0.05, and target KL 0.005. Step zero is ineligible for selection.

V24 C4 seeds are training 172001..172012, development 172101..172104, and
stage-final 172201..172204. Development and final seeds are not used by the
teacher or warm-start gate. Stage-final runs once only after an updated
checkpoint passes development and full C0-C4 retention. Stop after C4.
