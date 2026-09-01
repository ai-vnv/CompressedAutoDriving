# F14 Final A0 vs A7 Re-explanation

## Frozen basis

The final comparison reuses the 4,400 public factual states and the exact
6-draw x 4-reference assignments stored by F11 R004. No simulator trajectory,
locked seed, perception model, policy, or historical explanation was rerun or
modified. A0 and deployed static-INT8 A7 receive identical coalition vectors.
Exact-Shapley local-accuracy residual is at most `2.384e-07` for both actors.

## Semantic attribution

Classification: **SHIFTED** (1/10 phase/action cells preserved under the frozen
A0 self-consistency thresholds).

The hierarchy is partly retained but attribution mass is redistributed:

- pedestrian-relevant: Pedestrian remains top for both outputs, but its share
  changes from 0.772 to 0.531 for velocity and 0.535 to 0.333 for yaw;
- lane-curve: Stop remains top for velocity and Lane for yaw, while Lane yaw
  share changes from 0.771 to 0.543;
- stop-required velocity: Stop remains top (0.471 to 0.502);
- stop-required yaw: the top group changes from Lane (0.430) to PreviousAction
  (0.363);
- stop-satisfied: Lane remains top, but its share changes from 0.725 to 0.530
  for velocity and 0.908 to 0.517 for yaw.

Thus A7 preserves several dominant semantic roles while failing the stricter
share/ranking-equivalence criterion. Group Shapley is relative to the frozen
complete-row reference construction and is not a causal-world effect.

## Functional sensitivity

Classification: **SHIFTED** (1/3 preregistered primary phase/action cells pass all frozen
direction and magnitude criteria; sham is exactly zero).

- pedestrian removal preserves velocity direction but reduces mean yaw effect
  from 1.890 to 0.916 rad/s;
- stop removal preserves velocity direction but changes its mean effect from
  0.219 to 0.177 m/s; stop-removal yaw direction agreement is 0.860;
- lane-centering direction agreement is 0.763 for velocity and 0.847 for yaw;
- sham is exactly identity on both actors.

These are semantic policy-input sensitivities, not real-world causal effects.

## Behavior and scope

Frozen F12 C4 behavior remains **PRESERVED**: A0 and A7 both completed 8/8
episodes with no collision, unsafe episode, stop violation, or lane failure and
100% restart. Explanation/functional drift therefore does not constitute a C4
failure by itself. Historical C0--C2 retention remains not preserved.

This model-agnostic F14 analysis does not retroactively resolve F13's blocked
gradient attribution. F13 remains LIMITED and its historical records are
unchanged.
