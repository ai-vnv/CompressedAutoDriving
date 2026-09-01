# F16 — Optimization Sequence and INT8 Cross-Curriculum Recovery Protocol

**Status: frozen before any Direct/Progressive scientific result was produced or seen.**

Config: `configs/f16_sequence_int8_recovery_v1.toml`, SHA256
`b08282ec0545180d435459924c353515f839f541dd7da329f2b273d12b2d0fba`.

## Scope and boundary

F15 is complete and immutable. F16 does not reopen, rewrite, reinterpret, repair, or
rerun it, and does not reuse F15 selection results as new F16 evidence. F16 addresses
only what F15 left unresolved:

1. does optimization sequence matter at a matched endpoint width;
2. why the recovered FP32 actor collapses again after INT8 conversion;
3. whether any INT8 actor preserves C0–C4;
4. what the smallest such actor is;
5. and only if one exists, a single once-only holdout evaluation.

F16 runs no attribution, Integrated Gradients, Group Shapley, Grad-CAM, semantic
attribution, counterfactual explanation, or explain-again analysis.

## Verified F15 inheritance

Each of the following was checked against F15 artifacts before this protocol was frozen,
not taken from any prompt:

| Fact | Source | Verified |
|---|---|---|
| First PASS→FAIL transition for C0–C4 is Original → Pruning Only | `localization/matrix_results.json` | A1 = FAIL on all five |
| C4-focused KD recovered C3/C4 only | same | A2 = FAIL, FAIL, FAIL, PASS, PASS |
| Multi-curriculum KD recovered 64×64 without added capacity | `recovery/fp32/w64/selection_result.json` | `eligible: true`, all five PASS |
| PTQ of that actor failed C3/C4 | `recovery/ptq/w64/selection_result.json` | PASS, PASS, PASS, FAIL, FAIL |
| QAT+KD improved fidelity but still failed C3/C4 | `recovery/qat/w64/selection_result.json` | PASS, PASS, PASS, FAIL, FAIL |
| Larger FP32 widths never trained | `recovery/recovery_decision.json` | `larger_width_required: NOT_TESTED` |
| Progressive prune–distill never tested | same | `progressive_pruning_required: NOT_TESTED` |
| No eligible INT8 candidate | same | `eligible_int8_candidates: 0` |
| Holdout 180301–180308 unopened | same | `final_holdout_opened: false` |
| F15 closed-loop runs were not reproducible | `docs/F15_FAILURE_LOCALIZATION_REPORT.md` | 43/150 repeated cells differed numerically, 7 flipped an objective outcome label |
| F15 lacked primary front-camera footage | `docs/F15_VISUAL_EVIDENCE_OUTCOME.md` | RGB was never stored during the original rollouts |

## Survivor hierarchy — verified, not assumed

The F12 pruning criterion scores neurons on the frozen Original actor and takes a stable
top-k. Nesting was therefore checked programmatically rather than presumed:

- 64 ⊂ 96 ⊂ 128 ⊂ 192 holds for **both** hidden layers, with zero elements outside;
- survivor lists recomputed from the Original actor match the lists stored in all four
  F12 pruned checkpoints.

Progressive pruning in F16 therefore follows this **same frozen hierarchy**. A Direct and
a Progressive candidate at the same target width finish with the **same surviving original
neurons**, so the manipulated factor is the distillation trajectory alone.

Recomputing pruning scores from each intermediate distilled actor would change neuron
identity *and* sequence simultaneously. Any resulting difference would be
uninterpretable. That variant is a different question and F16 does not test it; this is
recorded as a limitation, not as a result.

## Integrity Gate 1 — determinism

F16 may not begin its sequence comparison until closed-loop evaluation is reproducible.
The gate is preflight integrity, not a scientific outcome.

**Selected backend: `cuda_strict_deterministic`.**

| Setting | Value |
|---|---|
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| `torch.use_deterministic_algorithms` | `True` |
| `torch.backends.cudnn.deterministic` | `True` |
| `torch.backends.cudnn.benchmark` | `False` |

Preflight: Original Policy, curricula C0 and C3, seeds 181001–181002, three repeats per
cell, eight repeat comparisons.

**Strict deterministic CUDA produced exact repeatability on all preregistered
determinism-gate comparisons: zero normalized-action difference, zero progress
difference, identical episode length, completion outcome, failure labels, and termination
reason.** This claim is scoped to those measured fields on that preflight set. It is not
a claim about the entire simulator state, nor about curricula not yet exercised.

**F16 prospectively eliminates the closed-loop reproducibility limitation observed in F15
under the tested deterministic CUDA configuration.** F15 itself is unchanged and retains
its documented limitation.

CPU fallback was attempted and aborted before any episode because the YOLO detector
requests `device=0` explicitly, so hiding CUDA raises in ultralytics `select_device`.
No determinism conclusion may be drawn from that abort in either direction, and **CUDA is
not claimed to be better than CPU** — CPU was never measured. **CUDA was selected because
it was the first backend in the preregistered fallback chain to satisfy the determinism
gate; the CPU fallback was therefore not required.**

Evidence: `integrity/determinism_gate.json`, `integrity/determinism_test.csv`,
`integrity/determinism_cpu_backend_attempt.json`.

## Seeds

The 181xxx block was verified absent from every config, doc, script, test, and artifact
before allocation; the only apparent hits were file byte counts inside an F15 manifest.

| Purpose | Seeds |
|---|---|
| determinism preflight | 181001–181004 |
| development | 181101–181108 |
| sequence/width selection | 181201–181208 |
| optional confirmation | 181301–181308 |
| **sealed, inherited from F15** | **180301–180308** |

Zero overlap with the 72 seeds recorded across F11–F15, and zero overlap with the sealed
holdout. Seeds 180301–180308 are not touched during training, development, sequence
comparison, width comparison, quantization tuning, QAT, or selection.

## Sequences

| Sequence | Definition |
|---|---|
| **D — Direct** | Original → prune directly to W → multi-curriculum KD → FP32 → PTQ → QAT only if PTQ fails |
| **P — Progressive** | Original → prune 192 → KD → prune 128 → KD → prune 96 → KD → prune 64 → KD → FP32 → PTQ → QAT only if PTQ fails |

Progressive stage lists: W=64 → [192,128,96,64]; W=96 → [192,128,96]; W=128 → [192,128];
W=192 → [192].

At W=192 the two sequences are procedurally identical. **P192 ≡ D192 and must not be
counted as an independent variant or replicate.**

Progressive uses more cumulative gradient steps because it has more KD stages. This
asymmetry is reported explicitly and never hidden. An auxiliary compute-matched
comparison (equal total KD epochs, 320) is reported **separately** from the primary
procedure-matched comparison.

## Held fixed across sequences

Original teacher (`713d26d9…`), public 29D ordering, the frozen F15 KD dataset
(`385e2a3a…`, 62,176 rows, no privileged truth), curriculum and phase balancing,
deterministic teacher action targets, Smooth-L1 on physical actions normalized by
`[0.4, 8.0]`, Adam, learning rate 0.001, weight decay 1e-6, batch size 512, 80 epochs per
recovery stage, action mapping, PTQ observer/calibration method, QAT fake-quant
implementation, x86 static INT8 backend, evaluation environments, evaluation seeds, and
every gate.

No reward, no critic target, no ground-truth action labels, no PPO retraining.

## Gates

Behavior, relative, and fidelity thresholds are copied verbatim from the frozen F15
protocol so F16 results remain directly comparable. They are reproduced in
`configs/f16_sequence_int8_recovery_v1.toml` under `[retention.*]` and `[fidelity]`. No
threshold may change after results open.

## Evaluation order

For every FP32 candidate: same-state action fidelity → deterministic C0–C4 closed-loop →
safety gate. Only a candidate passing all five curricula proceeds to INT8. Then PTQ →
fidelity → closed-loop. **If PTQ is eligible, quantization recovery stops for that
width/sequence.** If not, QAT+KD → convert → fidelity → closed-loop.

## Integrity Gate 2 — primary camera evidence

F15 could not produce camera evidence because RGB was never stored during the scientific
rollout, and a post-hoc replay failed its own validation. F16 captures frames **during**
the primary rollout.

Requirements:

- an in-memory RGB ring buffer of 136 steps is maintained during every evaluated episode;
- on an objective failure, the frames already associated with that same primary episode
  are persisted: 90 steps before the first objective failure, the failure step, and up to
  45 steps after;
- overlays use telemetry from that same primary episode only;
- **actor inference is never rerun to create media**, and no trajectory is reconstructed;
- at least one preregistered representative success per curriculum is recorded;
- media are labelled **Same-Seed Primary Rollouts**, never "causal counterfactual
  trajectory".

This pipeline must pass a smoke test **before** the main workload begins. The smoke test
must demonstrate that frames originate from the primary rollout, that seed/model/
curriculum are recorded, that frame index is synchronised with telemetry step, that
MP4/GIF/contact sheet decode, that SHA256 is stored, that the first-failure marker agrees
with telemetry, and that maintaining the ring buffer does not alter policy execution or
determinism.

## Sequence-effect rule

A sequence effect is **SUPPORTED** only when Direct and Progressive differ at identical
width, precision, curricula, seeds, and gates, beyond the determinism tolerance. Because
the gate measured zero difference, any reproducible difference at a matched endpoint is
attributable to the manipulated factor rather than to run-to-run noise.

Cross-width comparisons never license a sequence claim. Comparing 64 Direct against 96
Progressive is forbidden.

## Selection

An eligible final candidate must be INT8 and pass all five behavior gates, all frozen
same-state fidelity gates, all safety gates, the determinism gate, and provenance
verification. Frozen hierarchy: smallest target width → prefer Direct over Progressive →
prefer PTQ over QAT → latency only as a last tie-break. A larger model is never selected
when a smaller one already passes every gate.

## Final holdout

Only after one eligible INT8 candidate exists may `final_candidate.json` be frozen,
followed by `final_holdout_claim.json`, and only then may 180301–180308 be opened, for
Original versus that candidate alone. No replacement after access. If the candidate
fails, F16 reports FAIL/LIMITED and does not substitute another.

If no INT8 candidate passes, F16 stops at `NO_ELIGIBLE_INT8_CANDIDATE`, the holdout stays
sealed, and the report states which width and sequence came closest, which curricula
failed, which checks failed, and whether sequence or width mattered.

## Planned workload

Under all conditional branches the **maximum planned closed-loop workload** is
approximately 880 episodes: seven distinct FP32 candidates (D64, D96, D128, D192≡P192,
P64, P96, P128) at 40 episodes each, a reusable 40-episode Original baseline, and up to
40 episodes per PTQ and per QAT branch. Measured median cost at the frozen backend is
51.3 s per episode.

This is a ceiling, not a commitment. The branches are conditional: QAT runs only where
PTQ fails, INT8 runs only where FP32 passes all five curricula, and P192 is not rerun as
a separate variant. The eight-seed count is not reduced silently.

## Stop rules

Stop before scientific evaluation if any actor hash, registry hash, config hash, 29D
contract, action map, INT8 invocation, public/privileged separation, or the determinism
gate fails. Stop before the main workload if the primary media pipeline smoke test fails.
Stop before final access if candidate provenance or the once-only claim is missing. Never
change seeds, thresholds, curricula, survivor hierarchy, teacher, dataset, or candidate
after results are seen.

## Artifact namespace

All outputs are new under `artifacts/f16_sequence_int8_recovery_v1/`. F10–F15 paths are
read-only.
