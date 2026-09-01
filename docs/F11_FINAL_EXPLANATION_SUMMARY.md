# F11 Final Belief-PPO Explanation Summary

Status: **COMPLETE**

This package explains the already-frozen C4 Belief-PPO policy. It does not
retrain or modify PPO, MobileNetV3-small, YOLO11n, either EKF, the existence
filter, stop-state logic, or normalization.

## Frozen scientific status

| Run | Status |
|---|---|
| R001 deployment-boundary audit | PASS |
| R002 fixed-reference IG diagnostic | LIMITED |
| R002b distributional-reference robustness | PASS |
| R003 semantic-intervention development validation | PASS |
| R004 once-only final attribution | PASS |
| R006 confirmatory holdout intervention | FAILED at replay-integrity gate |
| R007 closed-loop intervention | BLOCKED / not executed |

The quantitative source is the frozen R004 artifact set. R006 produced no
confirmatory intervention result and did not alter R004.

## Decision pipeline being explained

```text
Front RGB
 ├─ MobileNetV3-small → lane measurement → lane EKF → LaneBelief
 └─ YOLO11n
      ├─ Duckie → projection → F9c pedestrian EKF/existence → PedestrianBelief
      └─ Stop sign → projection → StopSignBelief + StopMode

+ actual ego motion
+ route stop-line observer
+ previous action
        ↓
public semantic 29D representation
        ↓
fixed physical normalization
        ↓
PPO actor
        ↓
[v_cmd, omega_cmd]
```

The PPO does **not** consume RGB, MobileNet embeddings, YOLO boxes, or BEV
coordinates directly. MobileNet/YOLO overlays in the figures document
perception provenance; they are not PPO saliency maps.

## Methods

### Quantitative: phase-conditioned distributional multi-reference Integrated Gradients

The final estimator uses six independent draws with four same-phase,
cross-seed public references per draw: 24 equally weighted reference IG
estimates per factual state. The targets are the deterministic PPO actor means
in physical units, `v_cmd` (m/s) and `omega_cmd` (rad/s). The primary statistic
is mean absolute attribution share over the six complete, non-overlapping
groups:

`Lane | Ego | StopLine | Pedestrian | Stop | PreviousAction`.

R002 showed that three semantically different fixed references answer
different questions and produced baseline-sensitive rankings. R002b and R004
showed that repeated draws from the single frozen phase-conditioned reference
distribution were stable. This is distributional-reference robustness, not a
claim that IG is baseline invariant.

### Qualitative: BEV Belief–Action Decision Trace

The episode visualization joins a pre-existing public 29D trace with a
separately stored evaluation-only pose trace for qualitative seed `176001`.
Five RGB frames were recovered by deterministic replay of this non-R004 seed.
Replay maximum observation error and action error were both `0.0`, the terminal
event matched, and the checkpoint hash was unchanged.

Representative states were selected before rendering with public-only rules:

- nominal, lane curve, stop required, and stop satisfied: midpoint of the
  longest continuous public-phase segment;
- pedestrian relevant: range nearest the median valid range within the longest
  segment, restricted to public `P(e) >= 0.9`.

Selection did not use attribution magnitude, RGB appearance, world pose, or
privileged labels. The qualitative episode is not used for a new quantitative
claim. Its attribution ribbons/bars are the frozen R004 phase means, not newly
computed frame-level IG.

## Quantitative findings

### Overall mean absolute group share

| Group | `v_cmd` share (95% CI) | `omega_cmd` share (95% CI) |
|---|---:|---:|
| Lane | 0.3154 [0.3080, 0.3229] | 0.5330 [0.5310, 0.5351] |
| Ego | 0.0689 [0.0682, 0.0696] | 0.0448 [0.0442, 0.0454] |
| StopLine | 0.0629 [0.0619, 0.0640] | 0.0334 [0.0328, 0.0340] |
| Pedestrian | 0.1913 [0.1876, 0.1948] | 0.1961 [0.1921, 0.2001] |
| Stop | 0.2372 [0.2337, 0.2409] | 0.1474 [0.1443, 0.1505] |
| PreviousAction | 0.1242 [0.1235, 0.1249] | 0.0452 [0.0447, 0.0457] |

The overall plot includes the seed-bootstrap 95% confidence intervals:

- [PNG: overall attribution](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_overall_attribution.png)
- [PDF: overall attribution](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_overall_attribution.pdf)

### Phase-specific attribution

The phase-conditioned results are the main explanation:

- **Lane curve:** Lane contributes `0.429` of `v_cmd` share and `0.768` of
  `omega_cmd` share.
- **Pedestrian relevant:** Pedestrian contributes `0.881` of `v_cmd` share and
  `0.903` of `omega_cmd` share.
- **Stop required:** Stop contributes `0.396` of `v_cmd` share; steering is
  jointly associated with Lane (`0.437`) and Stop (`0.376`).
- **Stop satisfied:** Lane contributes `0.650` of `v_cmd` share and `0.885` of
  `omega_cmd` share.
- Where the pedestrian tuple is neutral, its share is `0` under the frozen
  reference protocol.

The policy attribution is therefore phase-dependent rather than a single
global ranking:

- [PNG: phase heatmap for velocity](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_phase_heatmap_v.png)
- [PDF: phase heatmap for velocity](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_phase_heatmap_v.pdf)
- [PNG: phase heatmap for yaw rate](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_phase_heatmap_omega.png)
- [PDF: phase heatmap for yaw rate](../artifacts/f11_ppo_explanation_v2/final_visualization/quantitative_phase_heatmap_omega.pdf)

## Qualitative findings

The BEV timeline shows the factual trajectory, physical actions, public
belief/state, and frozen R004 phase attribution through one complete C4 lap:

- [PNG: BEV belief–action decision trace](../artifacts/f11_ppo_explanation_v2/final_visualization/qualitative_bev_decision_trace.png)
- [PDF: BEV belief–action decision trace](../artifacts/f11_ppo_explanation_v2/final_visualization/qualitative_bev_decision_trace.pdf)

The representative panels connect front-camera provenance, BEV context,
public belief values, phase-mean Distributional IG, and physical action:

- [PNG: representative decision panels](../artifacts/f11_ppo_explanation_v2/final_visualization/qualitative_representative_panels.png)
- [PDF: representative decision panels](../artifacts/f11_ppo_explanation_v2/final_visualization/qualitative_representative_panels.pdf)

The panels illustrate the following artifact-supported pattern:

1. Lane attribution is concentrated on steering during a curve.
2. Attribution shifts strongly to pedestrian belief during the supported
   pedestrian-relevant state.
3. Stop belief contributes strongly to velocity during the required-stop
   phase, while Lane and Stop jointly contribute to yaw.
4. After the stop obligation is satisfied, attribution shifts back to Lane as
   normal driving resumes.

These statements describe attribution and association, not causal behavior.

## Privileged-data boundary

BEV pose, route geometry, the crossing path, and the stop-line marker are shown
for post-hoc visualization only. They were not available to the PPO policy or
the attribution method. The pedestrian uncertainty marker is calculated from
the public pedestrian range/bearing mean and standard deviation; it is not a
simulator-truth uncertainty graphic.

## R006 audit note

The preregistered confirmatory holdout intervention run stopped at a numerical
replay-integrity gate before any intervention was evaluated. Therefore this
package does not claim a confirmed holdout counterfactual effect. R003 remains
development-only supporting evidence of semantic policy-input dependence, and
no optional counterfactual panel is included.

## Interpretation limits

- Integrated Gradients does not prove causality.
- Attribution shares are relative to the frozen phase-conditioned public
  reference distribution; they are not percentages of behavior explained.
- The BEV is not what PPO observes.
- MobileNet/YOLO overlays do not explain PPO decisions.
- The qualitative seed and locked R004 seeds have distinct roles; no locked
  seed was rerendered for this package.
- R007 was not executed, so behavioral consequences of interventions remain
  unconfirmed.

## Reproducibility and artifacts

- Generator: `experiments/generate_f11_final_visualization.py`
- Frozen visualization config: `configs/f11_final_visualization_v1.toml`
- Figure-data provenance: `artifacts/f11_ppo_explanation_v2/final_visualization/figure_data_manifest.json`
- Frame selection and public values: `artifacts/f11_ppo_explanation_v2/final_visualization/representative_frame_manifest.json`
- Selected raw/provenance RGB: `artifacts/f11_ppo_explanation_v2/final_visualization/source_frames/`
- Frozen PPO SHA256: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`

## Tests

Artifact/hash verification: **PASS** (`11` hashed outputs, `5` selected frames,
all PNGs at least 300 dpi, all PDFs valid).

Focused F11 suite: **24 passed, 0 failed, 0 skipped**.

Full active repository suite (`pytest -q tests`):
**668 passed, 0 failed, 0 skipped**, with 426 warnings.

An initial unscoped `pytest -q` attempt also collected obsolete tests under
`_archive/attempt_01`; that attempt is preserved separately and is not the
project's active-suite result.
