# F10-PPO Visual-Lane v4 — Lane Competence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the PPO policy to drive the `small_loop` lap — staying in lane, not crossing the yellow centre line, and holding the curves — by fixing the two things v3's diagnosis showed were actually blocking it: the policy only ever trained on 22% of the track, and its lane belief reports uncertainty ~3× too small.

**Architecture:** Four changes, each behind its own config switch so the retrain can be attributed afterwards. Two are belief/environment fixes that require no policy change (start randomisation, uncertainty recalibration); two are reward shaping (curvature-aware heading, distance-based yellow warning). The PPO algorithm, network and optimiser are unchanged from v3 — v3's training was healthy and is not the problem.

**Tech Stack:** Python 3.10, PyTorch PPO (existing), gym-duckietown 6.2.0 `small_loop`, frozen YOLO11n checkpoint, NumPy, pytest, TOML config.

---

## Global Constraints

**Environment (identical to every earlier gate):**

```bash
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && export PYTHONHASHSEED=8123 && export CUDA_VISIBLE_DEVICES=0 && /home/pannntastic/aivnv/duckie/.venv/bin/python <command>'
```

Windows UNC paths for edits; `wsl.exe` only for execution. `$?` between `;`-separated
statements is unreliable — use `&&`/`||`. Training and renders exceed the 10-minute Bash cap:
use `run_in_background: true`. Not a git repository — checkpoints are ledger entries plus a
green suite.

**Frozen and not to be touched:**

```
YOLO checkpoint       3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c
PPO algorithm/network/optimiser hyperparameters   (v3's training was healthy: 61,440 steps,
                                                   60 updates, no NaN, W&B state finished,
                                                   checkpoint reload verified)
the lane-belief POINT EXTRACTOR and its heading/lateral MEAN estimates
                       (bias calibration already succeeded: heading bias +0.245 -> -0.007,
                        RMSE 0.075 against a 0.15 gate)
```

v4 changes **what uncertainty the belief reports**, **where episodes start**, and **two reward
terms**. It does not change how the lane pose is estimated.

**Seeds:**

```
v3 development (used)     47101-47104
v4 development            47201-47204
v4 stage-final            30201-30204   (already reserved by the v3 protocol; still untouched)
```

Scenario/reward tuning happens only against development seeds. Stage-final is read once, after
the development gate passes.

**Attribution requirement.** The operator chose to apply all four changes at once. Each must
therefore sit behind an independent switch so a follow-up can isolate it without re-deriving
anything:

```toml
[v4_changes]
start_randomisation      = true
belief_uncertainty_refit = true
curvature_heading_reward = true
early_yellow_warning     = true
```

---

## What v3's diagnosis established

All numbers below come from `artifacts/f10_ppo_visual_v3/c0/` and
`artifacts/visual_lane/lane_belief_final_validation.csv`. They are the reason each task exists.

### The policy never saw the track

Every episode starts at `start_tile = [1, 0]` with ±0.03 m jitter. 22 of 24 development
episodes ended at ~1.1 m of a 4.94 m lap. After 336 training episodes the policy had trained
on **the same 22% of the loop, repeatedly**, and had never experienced the remaining 78%.

Progress did not improve across training — 1.04 → 1.02 → 1.11 → 1.15 → 1.14 → 0.82 m across the
six checkpoints. More steps cannot fix this; the policy cannot learn a curve it never reaches.

### The belief is confidently wrong

Terminal frame of the failure replay (seed 47101, step 309):

```
policy belief : phi  0.059 +/- 0.019 rad     lateral 0.047 +/- 0.008 m
ground truth  : phi  0.243 rad               lateral 0.125 m
```

The heading estimate is off by 0.184 rad (10.5°) while reporting σ = 0.019 — a 9.7σ event, on a
frame the belief marked `valid 1.000`. In the *successful* episode the heading estimate is off
by 0.216 rad with the sign inverted. The policy did not correct its heading after the curve
because, from its own instrument's point of view, there was nothing to correct.

Aggregate calibration over 2,400 validation samples:

| | RMSE | mean σ | σ/RMSE | cov68 | cov95 |
|---|---|---|---|---|---|
| heading | 0.0674 | 0.0231 | 0.34 | **0.197** | **0.472** |
| lateral | 0.0269 | 0.0089 | 0.33 | **0.184** | **0.318** |

Both should be 0.68 and 0.95. The σ floor sits at 0.0197 rad with **38.2% of samples pinned to
it**, and at the floor the mean error (0.0513) is no better than above it (0.0505) — so at the
floor σ carries no information at all.

In the regime that killed the car — |heading error| > 0.15 rad, 3.5% of samples — the belief
reports mean σ 0.042 against a mean actual error of 0.233, a 5.6× understatement, and 21.7% of
those samples still sit at the floor. The instrument is most confident exactly where it is most
wrong.

**This is the same failure F9c spent an entire gate fixing for the pedestrian belief**
(reported σ 0.0058 against RMSE 0.028, cov68 0.152). The machinery already exists in
`src/duckie_pomdp/belief/covariance_calibration.py`.

### The gate never checked calibration

The lane-belief gate is a single line: `maximum_heading_belief_rmse_rad = 0.15`. Accuracy was
gated; uncertainty was not. That is why cov68 = 0.197 passed unnoticed.

### The track is much sharper than the reward assumes

`small_loop` GT curvature: median |κ| **1.089** 1/m (radius 0.92 m), p90 **3.708** (radius
**0.27 m**). Half this track is a sharp curve. Heading error has a heavy tail exactly there —
p95 is 0.183 rad on sharp curves versus 0.085 on straights — while reported σ barely moves
(0.0255 vs 0.0205).

Meanwhile the reward makes heading nearly free: with `lane_heading_weight = -0.006` and
`lane_heading_scale_rad = 0.40`, a **7.6° heading error costs the same as one step of
`living_penalty`**. Lateral costs the same at only 0.017 m.

### The failure mode was already migrating

At the last checkpoint (61,440) three of four episodes ended in `yellow_crossing` with 42–86
steps of yellow contact, at the *best* heading errors of the run (0.079–0.095). The policy had
started cutting curves from the inside. v3's curve-recovery rule only fires *after* yellow
contact — it reacts rather than prevents.

---

## File Structure

```text
configs/f10_ppo_visual_v4.toml          new; v3 copy + [v4_changes] + the four changes
configs/lane_belief_v2.toml             new; recalibrated uncertainty only

src/duckie_pomdp/control/lane_belief_uncertainty.py   floor/scale application + its gate
src/duckie_pomdp/control/start_sampler.py             loop-wide start pose sampling

modified:
  src/duckie_pomdp/control/ppo_environment.py   start sampling hook, reward hooks
  src/duckie_pomdp/control/lane_reward.py       curvature-aware heading, distance yellow warning

experiments/calibrate_lane_belief_uncertainty.py      refit floor/scale, emit the gate
experiments/train_f10_ppo_visual_v4.py                retrain
experiments/evaluate_f10_ppo_v4.py                    development gate

tests/test_v4_lane_belief_uncertainty.py
tests/test_v4_start_sampler.py
tests/test_v4_reward_shaping.py

artifacts/visual_lane_v4/lane_belief_uncertainty_metrics.json
artifacts/f10_ppo_visual_v4/c0/...
```

---

## Task 1: Recalibrate the lane-belief uncertainty, and gate it

**Files:**
- Create: `src/duckie_pomdp/control/lane_belief_uncertainty.py`,
  `experiments/calibrate_lane_belief_uncertainty.py`, `configs/lane_belief_v2.toml`
- Test: `tests/test_v4_lane_belief_uncertainty.py`

**This task changes only the reported uncertainty. The mean estimates stay exactly as they
are** — their bias calibration already succeeded and is not in question.

**Interfaces:**
- `LaneUncertaintyCalibration(heading_floor_rad, heading_scale, lateral_floor_m, lateral_scale)`
  with `apply(heading_std, lateral_std) -> tuple[float, float]`, computing
  `max(std * scale, floor)` per channel.
- `coverage_report(errors, sigmas) -> dict` returning `coverage_68`, `coverage_95`,
  `sigma_over_rmse`, `rmse`, `n`.

### The values, already derived from the validation set — verify, do not re-invent

Re-derive these from `artifacts/visual_lane/lane_belief_final_validation.csv` and confirm they
reproduce:

```
heading:  scale 1.0, floor 0.051 rad
          -> cov68 0.602, cov95 0.925, sigma/rmse 0.76      lands in band
lateral:  scale ~3.5-4.0, floor 0.0 m
          -> at scale 4.0: cov68 0.738, cov95 1.000, sigma/rmse 1.33
```

### Lateral cannot be fully calibrated, and the plan says so rather than pretending

The lateral z-distribution is the wrong *shape*, not merely the wrong scale:

```
z = |error|/sigma    p50 3.01   p95 4.90   p99 5.31
Gaussian             p50 0.67   p95 1.96   p99 2.58
```

The typical error is already 3σ, and the spread is far tighter than Gaussian (p95/p50 = 1.6
against 2.9). No `(scale, floor)` pair satisfies both bands: by the time cov95 reaches 0.90,
cov68 has passed 0.76.

**Chosen resolution:** pick the scale that puts **cov68** in band and accept that cov95
over-covers. Over-coverage at 95% is conservative, not dangerous — the interval is wider than
it needs to be, which is the safe direction for a controller. Record it explicitly as a
documented limitation in the metrics artifact under `lateral_overcoverage_disclosure`, with
the z-quantiles above as the evidence. Do **not** report lateral as "calibrated".

### Pre-registered gate — add what v3 was missing

```toml
[uncertainty_gate]
heading_coverage_68_band = [0.60, 0.76]
heading_coverage_95_band = [0.90, 0.98]
lateral_coverage_68_band = [0.60, 0.76]
maximum_sigma_over_rmse  = 1.5
maximum_heading_rmse_rad = 0.15    # retained from v3 -- accuracy must not regress
```

`lateral_coverage_95` is **reported but not gated**, with the disclosure above stating why.
Silently dropping it would be the same omission that let v3 through.

- [ ] **Step 1: Write the failing tests**

```python
def test_floor_and_scale_only_ever_widen_the_interval():
    """A calibration must never report less uncertainty than the raw belief."""
    cal = LaneUncertaintyCalibration(0.051, 1.0, 0.0, 4.0)
    h, l = cal.apply(0.0197, 0.0089)
    assert h >= 0.0197 and l >= 0.0089


def test_heading_floor_lands_the_measured_validation_set_in_band():
    """Load the real validation CSV, apply the calibration, and assert coverage.
    This is the whole point of the task -- if it does not reproduce, the numbers
    in the plan are wrong and the task must stop, not adjust the band."""
    # cov68 in [0.60, 0.76], cov95 in [0.90, 0.98], sigma/rmse <= 1.5


def test_lateral_overcoverage_is_disclosed_not_hidden():
    """The metrics artifact must carry lateral_overcoverage_disclosure with the
    z-quantiles, and lateral cov95 must NOT be presented as passing a band."""


def test_the_gate_rejects_a_calibration_that_understates_uncertainty():
    """Feed the pre-v4 floor (0.0197) and assert the gate FAILS on cov68.
    A gate that has never been seen to fail is not a gate."""


def test_mean_estimates_are_untouched():
    """Applying the calibration must not alter heading_mean or lateral_mean."""
```

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement, then run the calibration experiment**

```bash
<wsl wrapper> experiments/calibrate_lane_belief_uncertainty.py
```

- [ ] **Step 4: Check the reproduction against the plan's numbers.** If heading cov68 does not
land near 0.60 or the required floor differs materially from 0.051, **stop and report** — do
not widen the band to fit.

- [ ] **Step 5: Write `configs/lane_belief_v2.toml`** with the fitted values and its SHA256.

- [ ] **Step 6: Checkpoint.**

---

## Task 2: Loop-wide start randomisation

**Files:**
- Create: `src/duckie_pomdp/control/start_sampler.py`
- Modify: `src/duckie_pomdp/control/ppo_environment.py`
- Test: `tests/test_v4_start_sampler.py`

**This is the highest-leverage change in the plan and it costs nothing to run.** A policy
cannot learn a curve it never reaches; v3 spent 336 episodes on the same 1.1 m.

**Interfaces:**
- `LoopStartSampler(tiles, rng_seed)` with
  `sample(episode_index) -> StartPose(tile, local_x_m, local_z_m, heading_rad)`.

### Design constraints

- Sample the start **tile** across the drivable loop, not just jitter within one tile. Read the
  `small_loop` map to enumerate drivable tiles rather than hardcoding a list.
- Keep v3's within-tile jitter ranges unchanged (`±0.030 m` longitudinal, `±0.020 m` lateral,
  `±0.040 rad` heading) — those are calibrated and not the problem.
- The sample must be **deterministic from the seed and episode index**, so a training run is
  reproducible and a failure can be replayed exactly.
- Every start pose must be valid: on the drivable surface, inside the lane, pointing along the
  loop direction (counter-clockwise). Reject and resample otherwise; assert the rejection rate
  is low enough that it does not distort the distribution.

- [ ] **Step 1: Write the failing tests**

```python
def test_start_poses_cover_the_whole_loop_not_one_tile():
    """Sample 500 starts; assert at least 80% of drivable tiles are represented.
    v3 used exactly one tile -- that is what this test exists to prevent."""


def test_sampling_is_deterministic_from_seed_and_episode_index():
    """Two samplers with the same seed produce identical pose sequences."""


def test_every_sampled_pose_is_on_the_drivable_surface_and_faces_the_loop():
    """No start may begin off-road or facing backwards -- that would inject
    failures the policy cannot be blamed for."""


def test_within_tile_jitter_ranges_match_v3():
    """The jitter is calibrated and unchanged; only the tile choice is new."""
```

- [ ] **Steps 2-4: RED, implement, GREEN.**

- [ ] **Step 5: Sanity-render 20 sampled starts** to `artifacts/f10_ppo_visual_v4/start_samples/`
and confirm visually that they are spread around the loop and all on-road.

- [ ] **Step 6: Checkpoint.**

---

## Task 3: Curvature-aware heading reward and early yellow warning

**Files:**
- Modify: `src/duckie_pomdp/control/lane_reward.py`, `configs/f10_ppo_visual_v4.toml`
- Test: `tests/test_v4_reward_shaping.py`

### 3a — heading pressure that rises with curvature

The current weighting makes a 7.6° heading error cost one `living_penalty` step, while 0.017 m
of lateral error costs the same. The policy learned the rational response: hold position,
ignore angle. That works on straights and fails on the exit of a curve, which is exactly what
the failure video shows.

**Do not simply raise the constant weight.** With honest σ now reporting that the estimate is
poor on sharp curves, aggressively correcting toward a bad number is its own hazard. Scale the
heading penalty with |κ| so the pressure to straighten appears where curves are, and stays mild
on straights:

```
heading_penalty = lane_heading_weight * (|phi| / lane_heading_scale_rad)
                                      * (1 + curvature_gain * |kappa| / kappa_reference)
```

Pre-register `curvature_gain` and `kappa_reference` in the config. Set `kappa_reference` to the
measured median |κ| of the track (**1.089** 1/m), so the multiplier is ~1 on a typical curve
and grows toward the p90 (3.708).

Choose `curvature_gain` so that at p90 curvature a 7.6° heading error costs at least as much as
0.017 m of lateral error does — i.e. the two channels become comparable exactly where v3 failed.
State the resulting number in the config comment.

### 3b — yellow warning by distance, not contact

`yellow_warning_weight = -0.040` currently engages on contact. At checkpoint 61,440 three of
four episodes ended in `yellow_crossing` after 42–86 contact steps, so the signal arrives after
the policy is already committed. Trigger it on **distance to the yellow line**, with a
pre-registered activation distance, so the gradient exists before contact.

- [ ] **Step 1: Write the failing tests**

```python
def test_heading_penalty_grows_with_curvature():
    """Same heading error, higher |kappa| -> strictly larger penalty."""


def test_heading_and_lateral_penalties_are_comparable_at_p90_curvature():
    """At |kappa| = 3.708, a 7.6 deg heading error must cost at least as much as
    0.017 m of lateral error. This is the specific imbalance that produced v3's
    'hold position, ignore angle' behaviour."""


def test_straight_road_heading_penalty_is_close_to_v3():
    """At |kappa| ~ 0 the shaping must not silently become a different reward --
    otherwise v4 is not attributable to the curvature term."""


def test_yellow_warning_engages_before_contact():
    """A pose approaching the yellow line but not touching it must already carry
    a penalty; v3's fired only on contact."""


def test_each_change_is_independently_switchable():
    """With [v4_changes] flags off, the reward must reproduce v3's values exactly
    -- this is what makes the retrain attributable."""
```

- [ ] **Steps 2-4: RED, implement, GREEN.**

- [ ] **Step 5: Audit the reward on a replayed v3 episode** with all switches off and confirm
the per-step reward matches v3's logged values to 1e-9. If it does not, the shaping is not
attributable and the task must stop.

- [ ] **Step 6: Checkpoint.**

---

## Task 4: Freeze the v4 configuration

**Files:**
- Modify: `configs/f10_ppo_visual_v4.toml`
- Create: `artifacts/f10_ppo_visual_v4/v4_frozen_config.json`

- [ ] **Step 1:** Fill every fitted value, set `parameters_frozen = true`, record the
`lane_belief_v2.toml` SHA256 and the YOLO checkpoint SHA256.

- [ ] **Step 2:** Record the pre-registered development gate, **including the new checks**:

```toml
[development_gate]
minimum_completion_rate            = 0.50
minimum_mean_progress_m            = 3.50
maximum_lane_failure_rate          = 0.25
maximum_invalid_pose_rate          = 0.25
maximum_mean_abs_lateral_error_m   = 0.09
maximum_p95_abs_lateral_error_m    = 0.15   # NEW
maximum_yellow_contact_steps       = 20     # NEW
```

**Why `p95` is new and necessary.** v3 recorded `mean_abs_lateral_error_m = 0.026` as a **PASS**
on episodes where the car drove off the track. The mean is dominated by the ~300 good steps
before the failure, so it looks healthy precisely on the runs that failed. A tail statistic is
required or the metric certifies nothing. `maximum_yellow_contact_steps` exists because v3's
final checkpoint accumulated 42–86 contact steps while every gated metric still looked fine.

- [ ] **Step 3:** Record the config SHA256 and
`"development_seeds_not_yet_rendered": true`.

- [ ] **Step 4: Checkpoint.** From here the config is read-only until the development gate reports.

---

## Task 5: Retrain

**Files:** Create `experiments/train_f10_ppo_visual_v4.py`

- [ ] **Step 1:** Build on v3's trainer. **Change no PPO hyperparameter** — v3's optimisation
was healthy (60 updates, no NaN, W&B `finished`, checkpoint reload verified). Any hyperparameter
change here would confound the four v4 changes with a fifth.

- [ ] **Step 2: Smoke-test on a handful of episodes** and confirm: starts are spread around the
loop, the belief reports the new wider σ, and per-step rewards carry the curvature term.

- [ ] **Step 3: Run the full training in background.** Match v3's budget (61,440 env steps) so
the comparison is like-for-like. If the policy is still stuck at ~1.1 m at the halfway
checkpoint, **stop and report** rather than spending the remaining budget.

- [ ] **Step 4:** Record training curves, W&B run id, and checkpoint manifest.

- [ ] **Step 5: Checkpoint.**

---

## Task 6: Development gate and diagnosis

**Files:** Create `experiments/evaluate_f10_ppo_v4.py`

- [ ] **Step 1:** Evaluate every checkpoint on development seeds **47201–47204** (not v3's
47101–47104 — those informed this plan's design and are no longer held out).

- [ ] **Step 2: Read the progress distribution FIRST**, before any other metric. The single
question this whole plan exists to answer is whether episodes still die at ~1.1 m. If they do,
the diagnosis was wrong and no other number matters yet.

- [ ] **Step 3:** Then the full gate, including the two new checks.

- [ ] **Step 4: Attribute the outcome.** With all four switches on, re-run the evaluation on the
saved runtime cache with each switch individually off, and report the per-switch deltas. The
operator chose to change everything at once; this is what makes that recoverable.

- [ ] **Step 5: Render a failure and a success replay** with the same belief-vs-GT overlay v3
used. That overlay is what made this diagnosis possible; keep it.

- [ ] **Step 6:** Classify `PASSED` / `FAILED` against the pre-registered gate, write the
report, and **STOP**. Do not proceed to stage-final seeds `30201–30204` without the development
gate passing.

---

## Self-Review

**Coverage of the diagnosis.** Track exposure → Task 2. Overconfident belief → Task 1. Weak and
curvature-blind heading reward → Task 3a. Late yellow signal → Task 3b. Misleading mean-only
lateral metric → Task 4's `p95` and `yellow_contact_steps`. Attribution under an all-at-once
change → the `[v4_changes]` switches and Task 6 Step 4.

**Deliberate departures, flagged.** (i) Lateral uncertainty is *not* claimed to be calibrated —
its z-distribution is the wrong shape and no scale/floor fixes both bands; the plan takes
cov68-in-band with disclosed cov95 over-coverage, and records the evidence. (ii) The heading
reward is made curvature-dependent rather than uniformly stronger, because honest σ now says
the estimate is least reliable exactly where the pressure would be highest. (iii) PPO
hyperparameters are frozen deliberately, so a v4 result is attributable to the four changes and
not to optimiser drift.

**What would falsify the diagnosis.** If, after Task 2 alone, episodes still cluster at ~1.1 m,
then start exposure was not the constraint. If after Task 1 the belief's σ is honest and the
policy still fails to correct heading out of curves, the problem is the policy's capacity or the
reward, not the instrument. Task 6 Step 4's per-switch attribution is what distinguishes these.

**Type consistency.** `LaneUncertaintyCalibration.apply` (Task 1) is called by the belief
runtime feeding `lane_heading_error_std_rad` / `lane_lateral_error_std_m` into the 29-D
observation. `LoopStartSampler.sample` (Task 2) is called by `ppo_environment.reset`.
`[v4_changes]` flag names (Global Constraints) are the same strings Task 3's switch test and
Task 6's attribution step read.
