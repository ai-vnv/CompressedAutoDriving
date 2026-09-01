# F10-PPO C0-C1 dynamic camera-lane remediation v8

V7 proved that static pose-grid accuracy was insufficient: its selected RGB
model passed a 300-frame held-out gate but collapsed after longitudinal motion
that the static dataset did not contain.  Closed-loop development failed all
eight C0/C1 episodes, with belief heading remaining near zero while evaluation
truth reached about +0.73 rad.

V8 adds `lane_rgb_dynamic_v2`, generated from counter-clockwise expert recovery
trajectories over every part of both loop maps.  It has 2,880 train, 1,440
development, and a disjoint replacement held-out final set of 1,440 RGB frames.
Horizontal paired augmentation supplies both heading signs.  Runtime input is
front RGB only; simulator lane pose is an offline label/evaluation target.

The selected checkpoint SHA256 is
`637d517fe443134b45195e8ed91d2e900c40502f01fcde7e86847220263dc0a5`.
On the once-only valid replacement final it achieved lateral RMSE 0.00778 m,
heading RMSE 0.05034 rad, curvature RMSE 0.58798 m^-1, and 100% heading-sign
accuracy.  Two earlier final attempts stopped before producing metrics because
of validator schema errors; their logs and consumed seeds are retained, and a
new untouched seed block was used without changing the model.

Before PPO, the model must pass disjoint closed-loop development and final
gates on C0/C1, then the normal reward/reset/W&B/smoke/full-test/audit gates.
C0 starts from random weights because lane-feature semantics changed.  C1 may
inherit only a newly selected passing C0 checkpoint.  PPO/reward/action/29D
ordering remain unchanged, and this remediation stops after C1.
