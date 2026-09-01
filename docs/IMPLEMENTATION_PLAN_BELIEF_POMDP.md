# Codex Implementation Plan — Belief-Aware POMDP Visuomotor Policy in Gym-Duckietown

Revision note: the action sections below use the later, reviewed decision
`PolicyAction(v_cmd, omega_cmd)`. Wheel duty is an actuator output, not a policy
action.

## 1. Project Goal

Implement a simple but methodologically valid **POMDP-based visuomotor driving system in Gym-Duckietown**.

The main agent must use:

- Front RGB camera as the primary visual sensor.
- Object detection for:
  - Stop sign.
  - One dynamic Duckie pedestrian.
- Existing or simulator-supported lane perception for lane-related variables.
- Ground-plane / BEV coordinate projection to transform visual detections into metric relative positions.
- A Bayesian belief updater for pedestrian state estimation.
- A continuous control policy such as SAC or TD3.
- Commanded chassis linear velocity and yaw rate as policy actions.

The primary pipeline is:

RGB Front Camera  
→ YOLO Object Detection  
→ Ground-Plane Projection  
→ Cartesian Tracking  
→ Bayesian Belief Update  
→ Polar Belief Representation  
→ SAC/TD3 Policy  
→ `(v_cmd, omega_cmd)`  
→ Differential-Drive Action Adapter  
→ Left/Right Wheel Duty Commands

Do NOT give simulator ground-truth object states directly to the policy.

Simulator ground truth may only be used for:

- Training supervision.
- Calibration.
- Evaluation.
- Oracle baseline.
- Debugging.

---

# 2. Final POMDP Formulation

Use:

\[
\mathcal P =
\langle
\mathcal S,
\mathcal A,
T,
R,
\Omega,
O,
\gamma
\rangle
\]

## 2.1 State Space

Use the following conceptual true state:

\[
s_t =
[
s_t^{ego},
s_t^{road},
s_t^{sign},
s_t^{ped}
]
\]

### Ego state

\[
s_t^{ego}
=
[
d_t,
\phi_t,
v_t,
\omega_t
]
\]

where:

- \(d_t\): lateral deviation from lane center.
- \(\phi_t\): heading error relative to lane.
- \(v_t\): linear ego velocity.
- \(\omega_t\): ego yaw rate.

Yaw rate MUST be retained.

### Road state

\[
s_t^{road}
=
[
\kappa_t,
\rho_t^{stop},
m_t^{stop}
]
\]

where:

- \(\kappa_t\): road curvature.
- \(\rho_t^{stop}\): relative distance to stop line/stopping point.
- \(m_t^{stop}\): stop-state machine.

Use:

```text
NONE
REQUIRED
SATISFIED
```

Transition:

```text
NONE
  ↓ stop sign recognized
REQUIRED
  ↓ reaches stop line and sufficiently stops
SATISFIED
  ↓ intersection / stopping region passed
NONE
```

### Stop-sign state

\[
s_t^{sign}
=
[
e_t^{sign},
r_t^{sign},
\beta_t^{sign}
]
\]

where:

- \(e_t^{sign}\): sign exists.
- \(r_t^{sign}\): relative distance to stop sign.
- \(\beta_t^{sign}\): relative bearing of stop sign.

The stop sign determines the **stop obligation**.

The stop sign distance MUST NOT be used as the stopping location.

Stopping location is determined by:

\[
\rho_t^{stop}
\]

### Pedestrian state

For one relevant Duckie pedestrian:

\[
s_t^{ped}
=
[
e_t^{ped},
r_t^{ped},
\beta_t^{ped},
\dot r_t^{ped},
\dot\beta_t^{ped}
]
\]

where:

- \(e_t^{ped}\): pedestrian exists.
- \(r_t^{ped}\): pedestrian relative distance.
- \(\beta_t^{ped}\): pedestrian relative bearing.
- \(\dot r_t^{ped}\): radial relative velocity.
- \(\dot\beta_t^{ped}\): bearing rate.

For the first implementation, support only:

```text
N_pedestrian = 1
```

Select the nearest or most safety-relevant pedestrian.

Do NOT implement multi-pedestrian association until the single-pedestrian pipeline works.

---

# 3. Internal Tracking Representation

Although the POMDP pedestrian state is represented in polar ego-relative coordinates:

\[
[
r,\beta,\dot r,\dot\beta
]
\]

the internal Bayesian tracker SHOULD use Cartesian ground-plane coordinates:

\[
X_t=
[
x_t,
y_t,
v_{x,t},
v_{y,t}
]^T
\]

Reason:

- Easier constant-velocity transition.
- Easier Kalman filtering.
- Easier ground-plane projection.
- Easier ego-motion compensation.
- Easier future trajectory prediction.

After filtering, convert:

\[
[x,y,v_x,v_y]
\rightarrow
[r,\beta,\dot r,\dot\beta]
\]

using:

\[
r=\sqrt{x^2+y^2}
\]

\[
\beta=\operatorname{atan2}(x,y)
\]

\[
\dot r
=
\frac{xv_x+yv_y}{r}
\]

\[
\dot\beta
=
\frac{yv_x-xv_y}{r^2}
\]

Handle numerical stability when:

\[
r\approx0
\]

---

# 4. Action Space

Primary continuous action:

\[
a_t=
[
v_t^{cmd},
\omega_t^{cmd}
]
\]

where:

- \(v_t^{cmd}\): commanded forward linear velocity in m/s.
- \(\omega_t^{cmd}\): commanded yaw rate in rad/s; positive is
  counter-clockwise.

For Version 1, use the Gate-A0 bounds:

\[
v_t^{cmd}\in[0,0.4]\ \text{m/s}
\]

\[
\omega_t^{cmd}\in[-4.0,4.0]\ \text{rad/s}
\]

The differential-drive action adapter converts this command to normalized
left/right wheel duty. The policy does not perform that conversion.

Primary algorithms:

1. SAC.
2. TD3.

Optional later baselines:

- PPO.
- Discrete Q-learning.
- SARSA.

Do not change the action representation during initial implementation.

---

# 5. Observation Space

The policy MUST NOT directly receive true simulator object states.

Visual observation begins with:

\[
I_t^{front}
\]

The perception pipeline produces:

### Ego/lane observation

\[
[
\hat d_t,
\hat\phi_t,
\hat v_t,
\hat\omega_t,
\hat\kappa_t,
\hat\rho_t^{stop}
]
\]

Initially, lane-related variables may come from the existing Duckietown lane-localization pipeline or a clearly separated simulator/estimator interface.

Do not make lane perception the primary research problem in Version 1.

### Stop-sign measurement

YOLO provides:

```text
class
confidence
bounding box
```

Convert to:

\[
[
D_t^{sign},
c_t^{sign},
\hat r_t^{sign},
\hat\beta_t^{sign}
]
\]

where:

- \(D\): detection flag.
- \(c\): detector confidence.

### Pedestrian measurement

Convert YOLO result to:

\[
[
D_t^{ped},
c_t^{ped},
\hat r_t^{ped},
\hat\beta_t^{ped}
]
\]

Velocity is NOT a direct one-frame observation.

\[
\dot r,\dot\beta
\]

must be estimated temporally by the belief updater.

---

# 6. Ground-Plane / BEV Projection

BEV is NOT the policy input.

BEV is used as a geometric intermediate representation.

Pipeline:

```text
RGB
→ YOLO bbox
→ bottom-center object pixel
→ camera calibration
→ ground-plane coordinate (x,y)
→ relative distance / bearing
```

For an object bounding box:

```text
(x1, y1, x2, y2)
```

use the bottom-center point:

\[
u_f=\frac{x_1+x_2}{2}
\]

\[
v_f=y_2
\]

Then apply calibrated ground-plane projection:

\[
(u_f,v_f)
\rightarrow
(x,y)
\]

Then:

\[
r=\sqrt{x^2+y^2}
\]

\[
\beta=\operatorname{atan2}(x,y)
\]

Do NOT use bounding-box height as the primary distance estimator unless used as a baseline.

---

# 7. Ground-Truth Interface

Create a strict separation between:

```text
Agent Observation
```

and:

```text
Privileged Simulator State
```

Example architecture:

```python
observation = {
    "rgb": ...,
    "ego": ...,
}

privileged = {
    "pedestrian_true_position": ...,
    "pedestrian_true_velocity": ...,
    "stop_sign_true_position": ...,
    "true_lane_state": ...,
}
```

Policy code MUST NOT access `privileged`.

Use privileged state only for:

- YOLO dataset generation.
- Projection validation.
- Kalman validation.
- Belief calibration.
- Oracle MDP baseline.
- Evaluation metrics.

Add an automated test ensuring the policy observation dictionary contains no ground-truth pedestrian pose or velocity.

---

# 8. Phase 0 — Repository Audit

Before modifying code:

1. Inspect the existing repository.
2. Identify:
   - Gym-Duckietown environment wrapper.
   - Existing RL implementations.
   - Existing Q-learning/SARSA/SAC/TD3 code.
   - Current state representation.
   - Reward implementation.
   - Episode logger.
   - Map configuration.
   - Camera observation handling.
3. Do NOT rewrite working algorithms unnecessarily.
4. Create a dependency diagram of current modules.
5. Write a short `IMPLEMENTATION_NOTES.md`.

Output:

```text
Current architecture
Files to modify
Files to add
Potential compatibility issues
```

Do not begin major refactoring before this audit is complete.

---

# 9. Phase 1 — Environment Instrumentation

Create a dedicated POMDP environment wrapper.

Suggested module:

```text
envs/
    pomdp_duckietown_env.py
```

Responsibilities:

- Return agent observation.
- Store privileged ground-truth state separately.
- Compute ego variables.
- Identify stop line.
- Track stop-state machine.
- Obtain ground-truth pedestrian position for evaluation.
- Expose deterministic seeds.

Add logging for every timestep:

```text
episode_id
frame_id
timestamp

ego_d
ego_phi
ego_v
ego_yaw_rate
road_curvature

stop_line_distance
stop_mode

true_sign_x
true_sign_y

true_ped_x
true_ped_y
true_ped_vx
true_ped_vy

action_linear_velocity_cmd
action_angular_velocity_cmd
wheel_duty_left
wheel_duty_right

reward
collision
done
```

Acceptance gate:

- Run at least several seeded episodes.
- Logged state must match visually inspected simulator behavior.
- Yaw rate must have the correct sign convention.
- Stop-line distance must monotonically decrease when approaching a fixed stop line.

---

# 10. Phase 2 — Stop-Sign and Pedestrian Scenario

Create a minimal controlled map/scenario with:

- Lane-following path.
- One stop sign.
- One stop line.
- One Duckie pedestrian.
- No additional Duckiebots initially.

### Stop-sign randomization

Randomize sign placement inside a semantically valid region.

For every episode:

\[
(x^{sign},y^{sign})
\sim
p_{valid}(x,y)
\]

Constraints:

- Sign must remain visible from the reasonable approach region.
- Sign must correspond to a valid stop line/intersection.
- Sign cannot appear randomly in unrelated road locations.
- Sign cannot block the lane.
- Sign-to-stop-line relationship must remain realistic.

Add configuration such as:

```yaml
stop_sign_randomization:
    enabled: true
    lateral_range: ...
    longitudinal_range: ...
    yaw_range: ...
```

Keep the stop line separately defined.

Acceptance gate:

- Generate at least 100 randomized episodes.
- Automatically verify every sign remains inside the valid placement region.
- No invalid sign/road combinations.

---

# 11. Phase 3 — Camera Calibration and Projection

Implement:

```text
perception/
    camera_geometry.py
```

Functions should include:

```python
pixel_to_ground(u, v)
ground_to_polar(x, y)
bbox_bottom_center(bbox)
```

Validate projection against simulator GT.

Metrics:

\[
MAE_x
\]

\[
MAE_y
\]

\[
MAE_r
\]

\[
MAE_\beta
\]

Also report error as a function of:

- Distance.
- Bearing.
- Image position.

Example evaluation bins:

```text
distance:
0.0–0.5 m
0.5–1.0 m
1.0–1.5 m
>1.5 m

bearing:
center
mid-FOV
edge-FOV
```

Save results to CSV.

Acceptance gate:

Projection error must be quantified before integrating YOLO.

Do NOT proceed using unvalidated ground projection.

---

# 12. Phase 4 — YOLO Integration

Create:

```text
perception/
    detector.py
```

Classes:

```text
STOP_SIGN
DUCKIE
```

Detector output format:

```python
Detection(
    class_id,
    confidence,
    bbox,
    bottom_center,
)
```

Then convert to:

```python
ObjectMeasurement(
    detected,
    confidence,
    x,
    y,
    r,
    beta,
)
```

Important:

YOLO confidence must NOT be treated directly as calibrated existence probability.

First evaluate:

- Precision.
- Recall.
- mAP.
- False-positive rate.
- Detection probability by distance.
- Detection probability by bearing.
- Detection probability under occlusion.

Store:

\[
P_D(r,\beta)
\]

or a simpler binned approximation.

Acceptance gate:

The detector must produce repeatable measurements on recorded simulator episodes.

---

# 13. Phase 5 — Initial Oracle Detector Baseline

Before relying on YOLO, create an oracle measurement mode:

```text
detector_mode = oracle
```

Oracle detector should:

- Use simulator GT.
- Add configurable measurement noise.
- Add configurable missed detections.
- Never expose GT directly to policy.

Example:

\[
\hat x=x+\epsilon_x
\]

\[
\hat y=y+\epsilon_y
\]

with:

\[
\epsilon\sim\mathcal N(0,R)
\]

This mode is extremely important.

Use it to debug:

- Kalman filter.
- Belief updater.
- Policy.
- Reward.

before YOLO errors are introduced.

Required detector modes:

```text
oracle_clean
oracle_noisy
yolo
```

---

# 14. Phase 6 — Cartesian Kalman Belief Filter

Create:

```text
belief/
    pedestrian_filter.py
```

Use state:

\[
X=
[x,y,v_x,v_y]^T
\]

Initial model:

\[
X_{t+1}=FX_t+w
\]

with constant velocity.

Measurement:

\[
z_t=[x_t,y_t]^T
\]

Implement:

```python
predict()
update(measurement)
missed_detection_update()
reset()
```

Track:

\[
\mu_t
\]

and:

\[
\Sigma_t
\]

Do not estimate velocity through raw finite differences as the primary implementation.

Acceptance tests:

1. Stationary pedestrian.
2. Constant lateral motion.
3. Constant longitudinal motion.
4. Noisy measurement.
5. Missing 1 frame.
6. Missing 5 frames.
7. Pedestrian reappears.

Verify uncertainty grows during missing observations.

---

# 15. Phase 7 — Ego-Motion Compensation

This phase is mandatory.

Use:

\[
v_t
\]

and:

\[
\omega_t
\]

to compensate relative pedestrian motion.

At every update estimate:

\[
\Delta\psi=\omega_t\Delta t
\]

and ego translation:

\[
\Delta p_{ego}
\]

Transform the previous object belief into the new ego coordinate frame before applying measurement correction.

Test scenarios:

### Test A
Pedestrian stationary, ego stationary.

Expected:

```text
vx ≈ 0
vy ≈ 0
```

### Test B
Pedestrian stationary, ego moving forward.

Filter must understand that decreasing distance is primarily caused by ego motion.

### Test C
Pedestrian stationary, ego turning.

Bearing change must not automatically be interpreted as pedestrian lateral motion.

Acceptance gate:

Stationary pedestrian estimated world/relative motion must remain within predefined tolerance while ego moves and turns.

---

# 16. Phase 8 — Polar Belief Representation

After Cartesian filtering, expose:

\[
r,\beta,\dot r,\dot\beta
\]

and uncertainty.

Create:

```text
belief/
    belief_features.py
```

Output:

```python
PedestrianBelief(
    existence_probability,
    r_mean,
    r_std,
    beta_mean,
    beta_std,
    rdot_mean,
    rdot_std,
    betadot_mean,
    betadot_std,
)
```

Uncertainty should be propagated from Cartesian covariance.

Prefer Jacobian covariance propagation:

\[
\Sigma_{polar}
=
J\Sigma_{cart}J^T
\]

rather than assigning arbitrary standard deviations.

---

# 17. Phase 9 — Existence Belief

Maintain:

\[
P(e_t^{ped}=1)
\]

separately from kinematic state.

Inputs:

- Previous existence probability.
- Detection/no detection.
- Detector \(P_D\).
- False-positive probability.
- Track age.
- Occlusion if available.

When detection disappears:

Do NOT immediately set:

```text
existence = 0
```

Instead posterior probability should decay.

Example expected behavior:

```text
detected       0.97
missed 1       0.90
missed 2       0.80
missed 3       0.69
redetected     0.96
```

Use actual detector statistics where possible.

---

# 18. Phase 10 — Stop-Sign Belief and State Machine

Create:

```text
belief/
    stop_sign_belief.py

control/
    stop_state_machine.py
```

Stop-sign belief:

\[
[
P(e^{sign}),
r^{sign},
\beta^{sign}
]
\]

Stop-state machine:

```text
NONE
REQUIRED
SATISFIED
```

Trigger `REQUIRED` when stop-sign belief exceeds a threshold and is temporally stable.

Example:

```text
P(stop sign exists) > threshold
for N consecutive frames
```

Do not require the sign to remain visible afterward.

A successful stop requires:

\[
\rho^{stop}<\epsilon_{\rho}
\]

and:

\[
v<v_{stop}
\]

for a minimum duration.

Do NOT count stopping far before the line as successful compliance.

---

# 19. Phase 11 — Final Belief Vector

Build a fixed-size policy input.

Suggested Version 1:

\[
\begin{aligned}
b_t=[
&d,\phi,v,\omega,\\
&\kappa,\rho^{stop},m^{stop},\\
&P(e^{sign}),\mu_r^s,\sigma_r^s,
\mu_\beta^s,\sigma_\beta^s,\\
&P(e^{ped}),\\
&\mu_r^p,\sigma_r^p,\\
&\mu_\beta^p,\sigma_\beta^p,\\
&\mu_{\dot r}^p,\sigma_{\dot r}^p,\\
&\mu_{\dot\beta}^p,\sigma_{\dot\beta}^p
]
\end{aligned}
\]

Encode `m_stop` using:

- One-hot encoding, or
- Small categorical encoding.

Prefer one-hot encoding.

Normalize continuous variables before policy input.

Do NOT normalize angular variables incorrectly across the \(-\pi,\pi\) boundary.

Consider using:

\[
\sin\beta,\cos\beta
\]

instead of raw \(\beta\) if angular discontinuity becomes problematic.

---

# 20. Optional Derived Safety Features

After the baseline works, add:

\[
TTC=-\frac{r}{\dot r}
\]

only when:

\[
\dot r<-\epsilon
\]

Also calculate predicted pedestrian path in Cartesian coordinates:

\[
\hat x_{t+\tau}=x_t+v_x\tau
\]

\[
\hat y_{t+\tau}=y_t+v_y\tau
\]

Then determine whether the pedestrian is predicted to enter an ego collision corridor.

These should initially be treated as derived policy features, not fundamental state variables.

Add them through an ablation:

```text
belief only
vs
belief + TTC
vs
belief + predicted collision corridor
```

---

# 21. Phase 12 — Reward Function

Implement reward as separate components:

\[
R=
R_{progress}
+
R_{lane}
+
R_{stop}
+
R_{pedestrian}
+
R_{comfort}
\]

### Progress

\[
R_{progress}=w_p\Delta progress
\]

### Lane keeping

\[
R_{lane}
=
-w_d|d|
-w_\phi|\phi|
\]

### Collision

Large terminal penalty:

\[
R_{collision}
=
-w_c
\]

### Stop compliance

Reward correct stopping only near the stop line.

Penalize crossing the stop line while:

```text
m_stop = REQUIRED
```

### Pedestrian risk

Do NOT use only:

\[
1/r
\]

Use a combination of:

- Distance.
- Bearing.
- Closing speed.
- Predicted collision corridor.

### Comfort

Optional initial term:

\[
R_{smooth}
=
-w_a\|a_t-a_{t-1}\|^2
\]

Log every reward component independently.

This is required for debugging reward hacking.

---

# 22. Phase 13 — SAC Baseline

Implement or adapt existing SAC first.

Policy input:

```text
belief vector
```

Output:

```text
commanded linear velocity
commanded yaw rate
```

Do not add recurrent networks initially.

Reason:

The Bayesian filter already provides temporal memory.

This lets us test:

> Does explicit probabilistic belief improve control?

without confounding it with an RNN.

---

# 23. Phase 14 — Required Baselines

Implement the following experimental conditions.

## Baseline A — Oracle MDP

Policy gets true simulator state.

Purpose:

Upper bound.

## Baseline B — Current-frame detection

Policy receives:

```text
r
beta
detection flag
```

from the current frame only.

No temporal belief.

## Baseline C — Deterministic tracker

Policy receives filtered:

```text
r
beta
rdot
betadot
```

but no uncertainty.

## Model D — Probabilistic belief

Policy receives:

```text
mean + uncertainty + existence probability
```

This is the primary model.

This comparison is essential.

It answers whether belief uncertainty contributes beyond ordinary object tracking.

---

# 24. Phase 15 — Partial-Observability Stress Tests

Explicitly manipulate observation quality.

Conditions:

```text
Clean observation

Measurement noise

Random missed detections

Short occlusion

Longer occlusion

Edge-of-FOV pedestrian

Low-confidence detection
```

Measure whether belief maintains usable state estimates during temporary observation loss.

A good POMDP system should degrade gradually rather than fail instantly.

---

# 25. Phase 16 — Stop-Sign Randomization Experiment

Compare:

```text
Fixed stop-sign placement
```

versus:

```text
Random valid stop-sign placement
```

Evaluate:

- Stop-sign detection.
- Stop compliance.
- Generalization to unseen placement.
- Policy performance.

Purpose:

Demonstrate that the policy learns semantic stop behavior rather than memorizing image location.

---

# 26. Phase 17 — Side-Camera Ablation

Do this only after front-camera model works.

Conditions:

```text
Front

Front + Left

Front + Right

Front + Left + Right
```

All cameras must project detections into the same ego-relative ground coordinate frame.

Policy architecture and belief representation MUST remain unchanged.

Only observation coverage should change.

Evaluate:

- Detector recall.
- Position RMSE.
- Bearing RMSE.
- Belief uncertainty.
- Missed detections.
- Collision rate.
- Near misses.
- Stop compliance.
- Route completion.

Research question:

\[
FOV\uparrow
\Rightarrow
uncertainty\downarrow?
\]

and:

\[
uncertainty\downarrow
\Rightarrow
safety\uparrow?
\]

---

# 27. Evaluation Metrics

## Perception

Report:

```text
Precision
Recall
mAP
False-positive rate
Detection probability by distance
Detection probability by bearing
```

## Ground projection

\[
MAE_x
\]

\[
MAE_y
\]

\[
MAE_r
\]

\[
MAE_\beta
\]

## Belief estimation

\[
RMSE_r
\]

\[
RMSE_\beta
\]

\[
RMSE_{\dot r}
\]

\[
RMSE_{\dot\beta}
\]

Also report uncertainty quality:

```text
Negative log likelihood
95% interval coverage
Calibration curve
```

A belief model must not only be accurate.

It must be calibrated.

## Driving

Measure:

```text
Episode success rate
Route completion
Collision rate
Pedestrian collision rate
Near-miss rate
Mean lane deviation
Mean heading error
Stop-sign violation rate
Successful stop rate
Mean progress
Control smoothness
```

---

# 28. Experiment Seeds

All experiments must use reproducible seeds.

For every result store:

```text
random_seed
map_seed
pedestrian_seed
stop_sign_seed
policy_seed
```

Do not report only a single RL run.

Support multiple independent training seeds.

---

# 29. Suggested Repository Structure

Use existing structure where appropriate, but aim for clear separation:

```text
project/
│
├── envs/
│   ├── pomdp_duckietown_env.py
│   └── scenario_randomization.py
│
├── perception/
│   ├── detector.py
│   ├── camera_geometry.py
│   ├── ground_projection.py
│   └── line_stop_detector.py
│
├── belief/
│   ├── pedestrian_filter.py
│   ├── existence_filter.py
│   ├── stop_sign_belief.py
│   ├── belief_features.py
│   └── ego_motion.py
│
├── control/
│   ├── stop_state_machine.py
│   └── safety_features.py
│
├── policies/
│   ├── sac_belief.py
│   └── td3_belief.py
│
├── experiments/
│   ├── train_oracle.py
│   ├── train_detection.py
│   ├── train_belief.py
│   ├── eval_occlusion.py
│   ├── eval_random_sign.py
│   └── eval_camera_ablation.py
│
├── evaluation/
│   ├── perception_metrics.py
│   ├── belief_metrics.py
│   └── driving_metrics.py
│
├── configs/
│   ├── pomdp.yaml
│   ├── detector.yaml
│   ├── belief.yaml
│   ├── reward.yaml
│   └── experiments.yaml
│
├── tests/
│   ├── test_projection.py
│   ├── test_filter.py
│   ├── test_ego_motion.py
│   ├── test_stop_state.py
│   └── test_no_privileged_leak.py
│
└── logs/
```

Adapt this structure to the current repository rather than blindly replacing the repository layout.

---

# 30. Configuration-Driven Design

Do not hardcode research parameters.

Expose:

```yaml
camera:
  front: true
  left: false
  right: false

detector:
  mode: oracle_clean
  confidence_threshold: ...

belief:
  process_noise: ...
  measurement_noise: ...
  existence_threshold: ...

scenario:
  pedestrian_enabled: true
  stop_sign_enabled: true
  randomize_stop_sign: true

policy:
  algorithm: SAC

reward:
  progress_weight: ...
  lane_weight: ...
  collision_weight: ...
  stop_weight: ...
  risk_weight: ...
```

This is necessary for clean ablation experiments.

---

# 31. Implementation Order

Do NOT implement all modules at once.

Follow this exact sequence:

```text
1. Audit existing repository

2. Instrument simulator ground truth

3. Create simple stop-sign + one-pedestrian scenario

4. Validate state variables
   d, phi, v, yaw rate, curvature

5. Implement stop-line distance

6. Implement camera ground projection

7. Validate pixel → x,y → r,beta

8. Build oracle noisy detector

9. Implement Cartesian Kalman filter

10. Implement ego-motion compensation

11. Convert posterior → polar belief

12. Implement existence belief

13. Implement stop-state machine

14. Build fixed belief vector

15. Train SAC using oracle noisy observations

16. Evaluate belief and driving

17. Integrate YOLO

18. Re-evaluate perception and belief

19. Train/evaluate with YOLO observations

20. Add partial-observability stress tests

21. Add randomized stop-sign experiment

22. Add TD3 comparison

23. Add side-camera ablation last
```

---

# 32. Critical Acceptance Gates

Do not continue to RL training if any previous gate fails.

### Gate 1 — Geometry

Pixel-to-ground projection validated quantitatively.

### Gate 2 — Tracking

Stationary and moving pedestrian states tracked correctly.

### Gate 3 — Ego motion

Stationary pedestrian is not falsely interpreted as moving because ego moves/turns.

### Gate 4 — Missing observations

Uncertainty increases when detection disappears.

### Gate 5 — Stop behavior

Stop sign triggers `REQUIRED`; stopping point comes from stop line.

### Gate 6 — No privileged leakage

Policy cannot access simulator GT.

### Gate 7 — Belief calibration

Posterior uncertainty is quantitatively evaluated.

### Gate 8 — Oracle noisy control

SAC must work with noisy oracle measurements before YOLO is introduced.

Only after these gates pass should full YOLO-based training begin.

---

# 33. Main Scientific Ablation

The central comparison should be:

```text
Oracle true state
        ↓
Current-frame object observation
        ↓
Deterministic temporal tracker
        ↓
Probabilistic belief
```

This gives a clear scientific progression:

\[
MDP
\rightarrow
partial\ observation
\rightarrow
temporal\ state\ estimation
\rightarrow
probabilistic\ POMDP\ belief
\]

Then separately study:

```text
Front camera
vs
additional side cameras
```

and:

```text
Fixed stop sign
vs
randomized valid stop sign
```

Do not mix every ablation into the first experiment.

---

# 34. Main Research Hypotheses

H1:

A probabilistic temporal belief updater will reduce collision rate compared with current-frame object detections under noisy and missing visual observations.

H2:

Explicit uncertainty will improve safety compared with a deterministic tracker using the same estimated object kinematics.

H3:

Valid stop-sign position randomization will improve generalization and reduce dependence on fixed visual location.

H4:

Increasing camera field of view will reduce object-state uncertainty, especially during pedestrian crossing from lateral directions.

H5:

Lower belief uncertainty should correlate with lower collision/near-miss risk, but additional sensors may show diminishing returns.

---

# 35. Version 1 Scope

Keep Version 1 intentionally limited to:

```text
Front camera
One Duckie pedestrian
One stop sign
One valid stopping location
Single-route/lane-following scenario
Cartesian Kalman filter
Polar POMDP belief
SAC
```

Explicitly exclude from Version 1:

```text
Multiple pedestrians
Duckiebot-to-Duckiebot interaction
Intent prediction
Deep latent RSSM
End-to-end RGB policy
Multi-camera fusion
Full-city navigation
Complex intersections
```

Those are extensions.

---

# 36. Definition of Done for Version 1

Version 1 is complete when:

1. Front RGB frames are generated correctly.
2. Stop sign and one pedestrian can be detected.
3. Detection can be projected to metric ground coordinates.
4. Pedestrian position can be tracked through time.
5. Ego-motion compensation works.
6. Belief provides:

```text
P(existence)
r mean/std
beta mean/std
rdot mean/std
betadot mean/std
```

7. Stop state machine works:

```text
NONE → REQUIRED → SATISFIED → NONE
```

8. Policy receives belief only.
9. SAC drives the Duckiebot through the scenario.
10. Agent can stop at a valid stopping point.
11. Agent can respond safely to a crossing pedestrian.
12. Temporary pedestrian detection loss does not immediately cause unsafe acceleration.
13. Belief RMSE and calibration are reported.
14. Collision, lane, stop-compliance and route metrics are reported.
15. Results are reproducible from configuration + seed.

---

# 37. Codex Working Instructions

While implementing:

- Inspect existing code before creating replacements.
- Preserve working RL code whenever possible.
- Make small commits/changes by subsystem.
- Add tests before integrating the next subsystem.
- Do not silently substitute simulator GT for unavailable perception.
- Do not hide uncertainty by filling missing detections with GT values.
- Keep coordinate-system conventions documented.
- Document angle sign conventions.
- Document units for every state variable.
- Log raw observation, filtered estimate and GT separately.
- Keep experiment configuration reproducible.
- Prefer simple interpretable implementations before introducing neural approximations.
- Report blockers instead of bypassing methodological constraints.

Most importantly:

**The first goal is not high reward. The first goal is to demonstrate that the observation → belief → policy pipeline is methodologically correct.**
