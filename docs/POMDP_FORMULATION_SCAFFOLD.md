# POMDP Problem Formulation Scaffold — Gym-Duckietown

Status: domain-contract scaffold implemented; algorithmic components remain
gated. The action sub-contract is locked by Gate A0.

Implement and document the problem formulation for a simple **belief-aware visuomotor POMDP in Gym-Duckietown**.

The initial scope is intentionally limited to:

- Front RGB camera.
- Lane following.
- Stop-sign recognition and stopping behavior.
- One dynamic Duckie pedestrian.
- Continuous chassis-level control.
- Explicit probabilistic belief over pedestrian state.
- No multi-pedestrian interaction yet.
- No side cameras yet.
- No simulator object ground truth exposed to the policy.

The POMDP is:

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

---

# 1. State Space \(\mathcal S\)

Represent the conceptual true state as:

\[
s_t =
[
s_t^{ego},
s_t^{road},
s_t^{sign},
s_t^{ped}
]
\]

## 1.1 Ego State

\[
\boxed{
s_t^{ego}
=
[
d_t,
\phi_t,
v_t,
\omega_t
]
}
\]

where:

- \(d_t\): lateral deviation from lane center, meters.
- \(\phi_t\): heading error relative to lane tangent, radians.
- \(v_t\): actual ego linear velocity, m/s.
- \(\omega_t\): actual ego yaw rate, rad/s.

Important:

\[
v_t
\neq
v_t^{cmd}
\]

and:

\[
\omega_t
\neq
\omega_t^{cmd}
\]

The state stores actual vehicle motion, not commanded action.

---

## 1.2 Road State

\[
\boxed{
s_t^{road}
=
[
\kappa_t,
\rho_t^{stop},
m_t^{stop}
]
}
\]

where:

- \(\kappa_t\): road curvature.
- \(\rho_t^{stop}\): ego-relative distance to the valid stopping point or stop line, meters.
- \(m_t^{stop}\): stop-compliance mode.

Use:

```text
NONE
REQUIRED
SATISFIED
```

State-machine semantics:

```text
NONE
  ↓ valid stop sign recognized
REQUIRED
  ↓ ego reaches stop region and sufficiently stops
SATISFIED
  ↓ stopping/intersection region has been passed
NONE
```

The stop sign indicates that stopping is required.

The stop line or stopping point determines where stopping must occur.

Do NOT use distance to the physical stop-sign pole as the stopping distance.

---

## 1.3 Stop-Sign State

\[
\boxed{
s_t^{sign}
=
[
e_t^{sign},
r_t^{sign},
\beta_t^{sign}
]
}
\]

where:

- \(e_t^{sign}\): whether a relevant stop sign exists.
- \(r_t^{sign}\): ego-relative distance to the stop-sign model origin, meters.
- \(\beta_t^{sign}\): ego-relative bearing to the stop sign, radians.

The stop sign is treated as a static road object.

Its physical placement may later be randomized within a semantically valid region.

---

## 1.4 Pedestrian State

For Version 1 support only one relevant Duckie pedestrian:

\[
\boxed{
s_t^{ped}
=
[
e_t^{ped},
r_t^{ped},
\beta_t^{ped},
\dot r_t^{ped},
\dot\beta_t^{ped}
]
}
\]

where:

- \(e_t^{ped}\): pedestrian existence.
- \(r_t^{ped}\): ego-relative distance to the pedestrian model origin, meters.
- \(\beta_t^{ped}\): ego-relative pedestrian bearing, radians.
- \(\dot r_t^{ped}\): relative radial velocity, m/s.
- \(\dot\beta_t^{ped}\): bearing rate, rad/s.

Interpretation:

\[
\dot r < 0
\]

means relative distance is decreasing.

\[
\beta \approx 0
\]

means the pedestrian is approximately in front of the ego heading.

A changing \(\beta\) may be caused by pedestrian motion, ego yaw, or both.

Therefore ego-motion compensation is required in the belief estimator.

---

# 2. Action Space \(\mathcal A\)

The policy action is:

\[
\boxed{
a_t =
[
v_t^{cmd},
\omega_t^{cmd}
]
}
\]

where:

- \(v_t^{cmd}\): commanded linear velocity, m/s.
- \(\omega_t^{cmd}\): commanded yaw rate, rad/s.

The policy MUST NOT directly output left/right wheel commands.

Use:

```python
PolicyAction(
    linear_velocity_mps=...,
    angular_velocity_rad_s=...,
)
```

A separate action adapter converts:

\[
[v^{cmd},\omega^{cmd}]
\rightarrow
WheelCommand(left,right)
\]

before calling Gym-Duckietown.

For Version 1:

\[
v^{cmd}\in[0,0.4]\ \text{m/s}
\]

and:

\[
\omega^{cmd}\in[-4.0,+4.0]\ \text{rad/s}
\]

Reverse driving is not required initially.

Positive \(\omega^{cmd}\) means counter-clockwise yaw. The adapter output is a
normalized wheel duty command, not a wheel angular velocity. These numerical
bounds are configuration values selected by the measured Gate A0 envelope;
they are not theoretical motor limits.

---

# 3. Transition Function \(T\)

Define:

\[
\boxed{
T(s_{t+1}\mid s_t,a_t)
}
\]

The simulator is the primary generative transition model for ego motion.

The transition function conceptually includes:

- Ego differential-drive dynamics.
- Lane-relative state evolution.
- Stop-state evolution.
- Pedestrian motion.
- Ego-relative transformation of object states.

For pedestrian tracking, use an internal Cartesian state:

\[
X_t =
[
x_t,
y_t,
v_{x,t},
v_{y,t}
]^T
\]

with an initial constant-velocity model:

\[
x_{t+1}
=
x_t+v_{x,t}\Delta t+w_x
\]

\[
y_{t+1}
=
y_t+v_{y,t}\Delta t+w_y
\]

\[
v_{x,t+1}
=
v_{x,t}+w_{vx}
\]

\[
v_{y,t+1}
=
v_{y,t}+w_{vy}
\]

where:

\[
w\sim\mathcal N(0,Q)
\]

Ego motion must be compensated using actual:

\[
v_t,\omega_t
\]

before interpreting apparent pedestrian motion.

---

# 4. Observation Space \(\Omega\)

The raw sensor observation remains:

```python
@dataclass(frozen=True)
class SensorObservation:
    front_rgb: NDArray[np.uint8]
    ego: EgoObservation
    road: RoadMeasurement
```

The structured perception observation is conceptually:

\[
\boxed{
o_t =
[
\hat d_t,
\hat\phi_t,
\hat v_t,
\hat\omega_t,
\hat\kappa_t,
\hat\rho_t^{stop},
o_t^{sign},
o_t^{ped}
]
}
\]

## 4.1 Stop-Sign Observation

\[
\boxed{
o_t^{sign}
=
[
D_t^{sign},
c_t^{sign},
\hat r_t^{sign},
\hat\beta_t^{sign}
]
}
\]

where:

- \(D_t^{sign}\): detected/not detected.
- \(c_t^{sign}\): detector confidence.
- \(\hat r_t^{sign}\): measured relative distance.
- \(\hat\beta_t^{sign}\): measured relative bearing.

After ground-plane projection, \(\hat r_t^{sign}\) is passed through the fixed
offline F5b calibration so its physical target is the same model-origin range
used by \(r_t^{sign}\). Nearest collision-footprint distance remains a separate
privileged safety/evaluation quantity.

## 4.2 Pedestrian Observation

\[
\boxed{
o_t^{ped}
=
[
D_t^{ped},
c_t^{ped},
\hat r_t^{ped},
\hat\beta_t^{ped}
]
}
\]

Important:

Do NOT put:

\[
\dot r_t^{ped}
\]

or:

\[
\dot\beta_t^{ped}
\]

inside the one-frame observation.

These are hidden temporal quantities estimated by the belief updater.

The same F5b convention applies to pedestrian observations:
\(\hat r_t^{ped}\) targets the pedestrian model origin after fixed calibration;
it is not silently compared with nearest-footprint ground truth.

---

# 5. Perception Pipeline

The structured object observation must be produced through:

```text
Front RGB
    ↓
Object Detector
    ↓
class + confidence + bounding box
    ↓
bottom-center pixel
    ↓
ground-plane projection
    ↓
(x, y)
    ↓
(r, beta)
```

For a bounding box:

\[
(x_1,y_1,x_2,y_2)
\]

use the ground-contact approximation:

\[
u_f=
\frac{x_1+x_2}{2}
\]

\[
v_f=y_2
\]

Then:

\[
(u_f,v_f)
\xrightarrow{H}
(x,y)
\]

Then:

\[
\boxed{
r=\sqrt{x^2+y^2}
}
\]

and:

\[
\boxed{
\beta=\operatorname{atan2}(x,y)
}
\]

BEV is therefore an intermediate ground-coordinate representation.

Do NOT feed a simulator top-down BEV image directly to the main policy.

---

# 6. Observation Function \(O\)

Define:

\[
\boxed{
O(o_t\mid s_t)
=
P(o_t\mid s_t)
}
\]

Factor it initially as:

\[
\boxed{
O
=
O_{ego}
O_{road}
O_{sign}
O_{ped}
}
\]

---

## 6.1 Ego Observation Model

Initially ego variables may be treated as fully observed or nearly deterministic:

\[
\hat d=d
\]

\[
\hat\phi=\phi
\]

\[
\hat v=v
\]

\[
\hat\omega=\omega
\]

Optional later extension:

\[
\hat x=x+\epsilon_x
\]

with Gaussian sensor noise.

The primary source of partial observability in Version 1 is the visual object state.

---

## 6.2 Road Observation Model

\[
O_{road}
=
P(
\hat\kappa,
\hat\rho^{stop}
\mid
\kappa,\rho^{stop}
)
\]

Initially these may be treated as deterministic or low-noise measurements.

---

## 6.3 Stop-Sign Observation Model

\[
\boxed{
O_{sign}
=
P(
D^s,
c^s,
\hat r^s,
\hat\beta^s
\mid
e^s,r^s,\beta^s
)
}
\]

Detection probability:

\[
P(D^s=1\mid e^s=1,r^s,\beta^s)
=
P_D^s(r^s,\beta^s)
\]

False-positive probability:

\[
P(D^s=1\mid e^s=0)
=
P_{FA}^s
\]

If detected:

For Version 1 the calibrated measurement is modeled using empirical residual
bias and distance-binned variance from `configs/measurement_model_v1.toml`:

\[
\hat r^s = r^s + b_r(r^s) + \epsilon_r,
\qquad
\epsilon_r \sim \mathcal N(0,\sigma_r^2(r^s)).
\]

\[
\hat\beta^s = \beta^s + b_\beta + \epsilon_\beta,
\qquad
\epsilon_\beta \sim \mathcal N(0,\sigma_{\beta,s}^2)
\]

The Gaussian bearing form is a provisional F6 approximation; F5b records a
non-Gaussian held-out residual and does not claim normality.

---

## 6.4 Pedestrian Observation Model

\[
\boxed{
O_{ped}
=
P(
D^p,
c^p,
\hat r^p,
\hat\beta^p
\mid
e^p,r^p,\beta^p
)
}
\]

Detection probability:

\[
P(D^p=1\mid e^p=1,r^p,\beta^p)
=
P_D^p(r^p,\beta^p)
\]

False-positive probability:

\[
P(D^p=1\mid e^p=0)
=
P_{FA}^p
\]

If detected:

The pedestrian range uses the same canonical model-origin semantics and fixed
calibration:

\[
\hat r^p = r^p + b_r(r^p) + \epsilon_r,
\qquad
\epsilon_r \sim \mathcal N(0,\sigma_r^2(r^p)).
\]

\[
\hat\beta^p = \beta^p + b_\beta + \epsilon_\beta,
\qquad
\epsilon_\beta \sim \mathcal N(0,\sigma_{\beta,p}^2)
\]

If no detection is available, represent the measurement as missing.

Do NOT substitute:

```text
range = 0
bearing = 0
```

for a missed detection.

---

# 7. Belief State

The policy does not receive true object state.

The pedestrian belief is:

\[
\boxed{
b_t^{ped}
=
P(
e_t^{ped},
r_t^{ped},
\beta_t^{ped},
\dot r_t^{ped},
\dot\beta_t^{ped}
\mid
o_{1:t},
a_{1:t-1}
)
}
\]

Represent the belief approximately as:

\[
\boxed{
b_t^{ped}
=
[
P(e^p),
\mu_r^p,\sigma_r^p,
\mu_\beta^p,\sigma_\beta^p,
\mu_{\dot r}^p,\sigma_{\dot r}^p,
\mu_{\dot\beta}^p,\sigma_{\dot\beta}^p
]
}
\]

For the stop sign:

\[
\boxed{
b_t^{sign}
=
[
P(e^s),
\mu_r^s,\sigma_r^s,
\mu_\beta^s,\sigma_\beta^s
]
}
\]

---

# 8. Belief Updater

The formal update is:

\[
\boxed{
b_{t+1}
=
\tau(
b_t,
a_t,
o_{t+1}
)
}
\]

The updater MUST explicitly receive:

- Previous belief.
- Previous policy action.
- Current ego motion.
- Current structured perception observation.

Suggested interface:

```python
next_belief = belief_updater.update(
    previous_belief=belief_t,
    previous_action=action_t,
    ego_motion=ego_t1.motion,
    perception=perception_t1,
    dt_s=dt_s,
)
```

Use a Cartesian extended Kalman filter internally:

\[
[x,y,v_x,v_y]
\]

Then convert the posterior to:

\[
[r,\beta,\dot r,\dot\beta]
\]

for the POMDP belief representation.

F7 uses an EKF, not a linear Cartesian measurement update, because the
validated observation is polar:

```text
h(X) = [sqrt(x^2+y^2), atan2(x,y)]
```

The internal `vx,vy` semantics are pedestrian physical world velocity
expressed in the current ego-oriented axes. Actual ego translation/yaw is
applied as a known SE(2) frame transform before correction. The previous
policy command remains in the formal updater interface but is never used as a
substitute for actual motion.

The public `rdot,betadot` remain ego-relative. They are computed from physical
pedestrian velocity together with current actual ego velocity/yaw and their
uncertainty is propagated from the Cartesian posterior. Existence probability
is maintained by a separate scalar Bayesian filter; neither oracle confidence
nor future detector confidence is used directly as `P(exists)`.

---

# 9. Polar Conversion

From Cartesian posterior:

\[
r=
\sqrt{x^2+y^2}
\]

\[
\beta=
\operatorname{atan2}(x,y)
\]

\[
\dot r=
\frac{xv_x+y(v_y-v^{actual})}{r}
\]

\[
\dot\beta=
\frac{yv_x-x(v_y-v^{actual})}{r^2}
-
\omega^{actual}
\]

These are the ego-relative rates when internal `v_x,v_y` denote pedestrian
physical velocity in ego-oriented axes. Ego-yaw terms cancel from radial rate
but remain explicitly in bearing rate.

Use numerical guards when:

\[
r\rightarrow0
\]

Propagate covariance using the transformation Jacobian where feasible.

---

# 10. Reward Function \(R\)

Define:

\[
\boxed{
R_t
=
R_{progress}
+
R_{lane}
+
R_{stop}
+
R_{ped}
+
R_{comfort}
}
\]

Initial structure:

\[
R_{progress}
=
w_p\Delta progress_t
\]

\[
R_{lane}
=
-w_d|d_t|
-w_\phi|\phi_t|
\]

\[
R_{collision}
=
-w_c\mathbf1_{collision}
\]

Stop compliance:

- Reward stopping sufficiently close to the valid stopping location.
- Penalize crossing while `stop_mode == REQUIRED`.
- Do not reward stopping very far before the stop line.

Pedestrian risk should depend on belief-derived safety information such as:

\[
r^p,\beta^p,\dot r^p,\dot\beta^p
\]

and later possibly TTC or predicted collision corridor.

Do NOT use only distance \(1/r\) as the complete pedestrian risk function.

Keep each reward component separately logged.

---

# 11. Discount Factor

Use initial:

\[
\boxed{
\gamma=0.99
}
\]

Expose it through configuration.

---

# 12. Policy Input

The initial fixed belief-vector input should contain:

\[
\boxed{
\begin{aligned}
B_t=[
&d,\phi,v,\omega,\\
&\kappa,\rho^{stop},m^{stop},\\
&P(e^s),
\mu_r^s,\sigma_r^s,
\mu_\beta^s,\sigma_\beta^s,\\
&P(e^p),
\mu_r^p,\sigma_r^p,
\mu_\beta^p,\sigma_\beta^p,\\
&\mu_{\dot r}^p,\sigma_{\dot r}^p,
\mu_{\dot\beta}^p,\sigma_{\dot\beta}^p
]
\end{aligned}
}
\]

Encode `m_stop` categorically, preferably one-hot.

Normalize continuous features before policy use.

The policy outputs:

\[
\boxed{
[v^{cmd},\omega^{cmd}]
}
\]

---

# 13. Privileged Simulator State

Create an explicit privileged-state interface.

It may contain:

```text
true pedestrian position
true pedestrian velocity
true stop-sign position
true lane state
true stop-line geometry
```

Privileged information may be used only for:

- Training labels.
- Measurement-model calibration.
- Evaluation.
- Belief calibration.
- Oracle MDP baseline.
- Debugging.

It MUST NOT be concatenated into the POMDP policy input.

Add an automated test preventing privileged-state leakage.

---

# 14. Version-1 Partial Observability Assumption

For Version 1:

- Ego state may be treated as effectively observed.
- Road/lane state may be treated as effectively observed or low-noise.
- Stop sign is visually observed and may be missed.
- Pedestrian is visually observed and may be missed.
- Pedestrian velocity is hidden.
- Pedestrian bearing rate is hidden.
- Pedestrian existence remains uncertain during temporary visual loss.

Therefore the primary POMDP uncertainty is:

\[
\boxed{
P(
e^p,
r^p,
\beta^p,
\dot r^p,
\dot\beta^p
)
}
\]

Do not add unnecessary hidden variables before this baseline works.

---

# 15. Required Domain Types

Create or verify domain types equivalent to:

```python
EgoState
EgoMotion
RoadState
StopSignState
PedestrianState
POMDPState

PolicyAction
WheelCommand

SensorObservation
ObjectMeasurement
RoadMeasurement
PerceptionObservation

StopSignBelief
PedestrianBelief
RoadBelief
BeliefState

PrivilegedSimulatorState
```

Keep state, observation, belief, and action types explicitly separated.

---

# 16. Important Terminology

Use these terms consistently:

```text
v_actual
omega_actual
```

for ego-state quantities.

Use:

```text
v_cmd
omega_cmd
```

for policy actions.

Use:

```text
range / r
bearing / beta
```

for ego-relative object geometry.

Use:

```text
measurement
```

for one-frame perception outputs.

Use:

```text
belief
```

for temporally filtered probabilistic state estimates.

Use:

```text
ground-plane projection
```

or:

```text
BEV coordinate projection
```

for pixel-to-metric conversion.

Do not call a simulator top-down image the normal policy observation.

---

# 17. Scaffold Acceptance Criteria

The formulation scaffold is complete when:

1. State variables and units are explicitly defined.
2. Actual state and commanded action are separated.
3. Stop sign and stop line are separated conceptually.
4. Observation space is distinct from state space.
5. Pedestrian velocity is not directly inserted into one-frame observation.
6. Observation function explicitly represents:
   - detection probability,
   - false positives,
   - range noise,
   - bearing noise.
7. Belief updater explicitly uses previous belief and action.
8. Cartesian internal tracking and polar policy representation are documented.
9. Ego-motion compensation is documented as mandatory.
10. Simulator ground truth is structurally prevented from entering the policy.
11. Reward components are declared but tunable.
12. \(\gamma=0.99\) is configurable.
13. `v_max=0.4 m/s` and `omega_max=4.0 rad/s` come from the passed
    actuator-envelope gate and remain configuration-driven.
14. Version 1 remains limited to one pedestrian and front camera.

Do not implement additional complexity until this formulation is reflected consistently in the domain classes, interfaces, FORMULATION.md, and tests.
