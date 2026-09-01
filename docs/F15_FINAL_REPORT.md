# F15 Final Report — Cross-Curriculum Compression Recovery

## Classification

**F15 = LIMITED.**

Against the ten success criteria frozen for F15:

| # | Criterion | Result |
|---|---|---|
| 1 | C0 retention gate | **PASS** (FP32 recovered actor) |
| 2 | C1 retention gate | **PASS** (FP32 recovered actor) |
| 3 | C2 retention gate | **PASS** (FP32 recovered actor) |
| 4 | C3 retention gate | **PASS** (FP32 recovered actor) |
| 5 | C4 retention gate | **PASS** (FP32 recovered actor) |
| 6 | Action-fidelity gates | **PASS** (FP32 recovered actor, all five curricula) |
| 7 | No unacceptable safety regression | **PASS** in FP32; **FAIL** for both INT8 paths (C3 stop-violation gate) |
| 8 | Materially compressed vs Original | **PASS** — 91.61% fewer parameters, 90.22% smaller file |
| 9 | Reproducible checkpoint/config/artifact provenance | **PASS** |
| 10 | Once-only final holdout | **NOT SATISFIED** — no eligible INT8 candidate existed, so the holdout was deliberately not opened |

Criteria 1–9 are satisfied by an FP32 actor. Criterion 10 is not satisfied, and the
frozen protocol requires the final candidate to be a deployable INT8 actor. F15 is
therefore **LIMITED**, not PASS: cross-curriculum competence was located, explained by a
controlled ablation, and recovered — but not in a deployable INT8 form.

## Frozen policies and final access

- Original Policy: `artifacts/f10_ppo_visual_objects_v30/c4/ppo_selected.pt`,
  SHA256 `02e898ce…`; actor `29→256→256→2`, Tanh.
- Historical A0–A7 and the P/PD pruning frontier: all nine frontier checkpoints and all
  eight ablation checkpoints SHA256-verified before any F15 simulation.
- Recovered FP32 student: `recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt`,
  SHA256 `64c84cd0…`.
- Recovered INT8 actors: PTQ `7ac05518…`, QAT `c943e34f…`. Neither is eligible.

The final seeds **180301–180308 were never used** — not for dataset collection, not for
recovery training, not for selection, and not for evaluation. No final-holdout claim was
written because no candidate could be frozen. No candidate replacement was attempted.

## Failure localization

On the new paired localization seeds 180001–180008, across 560 episodes:

| Optimization stage | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Original Policy | REFERENCE | REFERENCE | REFERENCE | REFERENCE | REFERENCE |
| Pruning Only | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + Knowledge Distillation | FAIL | FAIL | FAIL | PASS | PASS |
| Post-Training Quantization (PTQ) | PASS | PASS | PASS | FAIL | PASS |
| QAT + Distillation | PASS | FAIL | FAIL | PASS | PASS |
| Pruning + PTQ | FAIL | FAIL | FAIL | FAIL | FAIL |
| Pruning + Distillation + PTQ | FAIL | FAIL | FAIL | PASS | FAIL |
| Final INT8 Policy | FAIL | FAIL | FAIL | PASS | PASS |

**First collapse is the same stage for all five curricula: Original Policy → Pruning
Only.** What differs is what was subsequently restored — historical C4-focused
distillation recovered C3 and C4 only.

The pruning-width diagnosis rules out capacity as the explanation. Pruning-only at
192×192 **preserves C0** (completion 1.000); adding historical distillation at the same
width drops C0 to **0.000**. Every historically distilled width — 192, 128, 96, 64 —
records 0.000 completion in C0–C2, across a 7× parameter range. Narrow-coverage
distillation is associated with the loss, and it did not merely fail to prevent it.

Same-state action fidelity, measured offline on CPU and therefore free of trajectory
feedback, shows the same gradient: Final INT8 Policy omega MAE is 0.05364 rad/s on C3 and
0.03958 on C4, against 0.26946 / 0.34317 / 0.23831 on C0 / C1 / C2 — five to eight times
past the 0.200 gate on exactly the curricula absent from the distillation data. **Action
drift is present before closed-loop collapse.**

## Controlled recovery

Holding the teacher, survivor indices, architecture, loss, optimizer, and 80-epoch budget
fixed, and changing **only** the rehearsal coverage to a curriculum-balanced C0–C4
dataset of 62,176 public states, the 64×64 student passed all five retention gates and
all five fidelity gates.

| | C0 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| Historical A2 completion (C4-only KD, same width) | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| **Recovered completion (balanced C0–C4 KD)** | **1.000** | **0.875** | **0.875** | **1.000** | **1.000** |

Neither PTQ nor multi-curriculum QAT+KD preserved that recovery. Both broke C3 and C4
while keeping C0–C2, and the QAT actor additionally failed the absolute
`maximum_stop_violation_rate` gate on C3.

## Final C0–C4 behavior

No once-only holdout result exists. The best available cross-curriculum standing is on
the recovery-selection seeds 180201–180208, reported in
`figures/08_final_cross_curriculum_performance.*` with that limitation stated on the
figure itself:

| Actor | C0 | C1 | C2 | C3 | C4 | Eligible |
|---|---|---|---|---|---|---|
| Original Policy | REFERENCE | REFERENCE | REFERENCE | REFERENCE | REFERENCE | — |
| Recovered 64×64 + Multi-Curriculum KD (FP32) | PASS | PASS | PASS | PASS | PASS | **yes** |
| Recovered 64×64 + PTQ (INT8) | PASS | PASS | PASS | FAIL | FAIL | no |
| Recovered 64×64 + Multi-Curriculum QAT+KD (INT8) | PASS | PASS | PASS | FAIL | FAIL | no |

These are selection-split numbers. They are **not** holdout numbers and must not be
reported as if a final validation had been passed.

## Final same-state action fidelity

The recovered FP32 actor passes the frozen fidelity gate on every curriculum. On C0 it
records v MAE 0.00081 m/s, omega MAE 0.03172 rad/s, omega sign disagreement 0.000, and
Pearson 0.99962 — against gates of 0.020 m/s, 0.200 rad/s, 0.050, and 0.980.

Both INT8 actors fail on **correlation**, not magnitude: PTQ C0 Pearson 0.97780 and QAT
C0 Spearman 0.97997, against a 0.980 gate, while their omega MAE stays inside the
tolerance. Quantization at this width distorts the ordering of steering commands more
than their size.

## Compression efficiency

Actor-only, one thread, batch 1, frozen benchmark parameters:

| Actor | Params | Bytes | Median latency | Curricula passed |
|---|---:|---:|---:|---:|
| Original Policy | 73,986 | 299,667 | 40.428 µs | 5 |
| **Recovered FP32 (eligible)** | **6,210** | **29,295** | **35.840 µs** | **5** |
| Recovered PTQ INT8 | 6,210 | 34,088 | 15.984 µs | 3 |
| Recovered QAT INT8 | 6,210 | 34,152 | 15.383 µs | 3 |
| Final INT8 Policy (historical A7) | 6,210 | 36,880 | 15.313 µs | 2 |

The recovered FP32 actor is **11.9× smaller in parameters** and **90.22% smaller on
disk**, but only **1.13× faster**. The 2.5–2.6× actor-only speedups belong exclusively to
INT8 actors that lose C3/C4. Actor-only latency is not end-to-end perception-to-action
latency; perception is unchanged and dominates deployment cost.

## Objective visual evidence

All 50 objectively selected failure events and 29 representative successful episodes have
MP4, GIF, PNG contact sheet, CSV, and hash-bound JSON under `failure_telemetry/` and
`success_telemetry/`, generated directly from the immutable primary telemetry
(`visualization_type: direct_frozen_telemetry_timeline`).

**These contain no camera frames.** The primary telemetry does not store RGB, and a
recorded-action simulator replay was attempted and rejected by its own frozen validation:
trajectory reproduction was excellent (2.96e-08 m) but episode termination was not
reproduced, with the first objective failure landing 144 and 171 steps early. Those media
are quarantined under `failure_traces/*/unresolved/` and are not evidence. Consequently
the front-camera side-by-side required by the F15 specification is **not satisfied**; see
`docs/F15_VISUAL_EVIDENCE_OUTCOME.md` and
`docs/F15_VISUAL_REPLAY_IMPLEMENTATION_AMENDMENT.md`.

## Scientific conclusion

1. **Where competence was lost.** Under the frozen gates, direct structured pruning to
   64×64 is the first optimization stage at which every curriculum fails.
2. **What the historical pipeline did and did not restore.** C4-focused distillation
   restored C3 and C4 and left C0–C2 at zero completion at every tested width.
3. **What the controlled ablation supports.** Changing only public rehearsal coverage,
   with capacity, loss, optimizer, and budget fixed, restored C0–C2 retention at the
   smallest historical width. The findings support insufficient rehearsal coverage as a
   recoverable factor in cross-curriculum forgetting under this protocol.
4. **Capacity is not the operative constraint here.** 6,210 parameters suffice for all
   five curricula when rehearsal covers them; 43,202 parameters do not when it does not.
5. **What quantization did.** Under the tested x86 static INT8 procedure with balanced
   calibration, quantization introduced a new C3/C4 retention failure in a recovered actor
   that passed all five gates in FP32, and multi-curriculum QAT+KD improved fidelity
   without restoring behavior.
6. **What remains unachieved.** No deployable INT8 actor preserving C0–C4 was produced,
   so no final candidate was frozen and no holdout was opened.

These conclusions distinguish same-state action fidelity, closed-loop behavior,
curriculum retention, and compression efficiency. Comparative localization supports
associations under this protocol, not neuron-level causal claims.

## Limitations

1. **Closed-loop measurements are not reproducible in this runtime.** The frozen F10
   perception front-end runs nondeterministic CUDA kernels. Across 150 repeated
   `(model, curriculum, seed)` cells, 43 differed numerically and 7 flipped an objective
   outcome label — including Original Policy itself on C2. The practical noise floor is
   about one episode in eight, i.e. 0.125, which equals several frozen relative margins.
   Headline separations of 0/8 versus 8/8 are far outside this band; cells differing by
   one or two episodes are flagged as not conclusive in
   `docs/F15_FAILURE_LOCALIZATION_REPORT.md`.
2. **No once-only holdout.** Every recovery number is a selection-split number. The
   recovered actor has not survived an unopened-seed test.
3. **No camera-based visual evidence**, for the reason given above.
4. **Width search for the INT8 endpoint was not performed.** The frozen rule triggers
   larger widths only when the 64×64 FP32 student fails, and it did not. Whether 96, 128,
   or 192 admits an eligible INT8 actor is **untested**.
5. **Optimization order is unresolved.** Progressive prune–distill was not run, so no
   ordering claim is made.
6. **Inherited baseline weakness.** `docs/F10_PPO_CURRICULUM.md` records that C1
   retention at the C4 checkpoint was already limited to 25% completion in F10, and that
   C3 passed using a DAgger-distilled network rather than reward-only PPO. The Original
   Policy is not uniformly strong across C0–C4, which is why the frozen gate marks a
   curriculum `UNRESOLVED` rather than `FAIL` when Original itself fails its absolute
   gate.
7. **Single seed block per stage, eight seeds per cell.** Conclusions are conditioned on
   this protocol, these curricula, and this simulator.

Historical F10–F14 results and statuses remain unchanged.
