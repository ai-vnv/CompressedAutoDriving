# F11 Integrated Gradients Report for Review

Classification: **LIMITED**

This report explains the frozen C4 PPO checkpoint only. No policy,
perception, belief, or normalization parameter was updated.

## Provenance

- Checkpoint SHA256 before/after: `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250` / `02e898ce12d71f97016d50ed8a40574807e6d2fd995fc9f0dcd24f357f2c6250`.
- Explanation seeds: `[176001, 176002, 176003, 176004]` (`4` episodes).
- Sampled frames: `2389` at stride `4`.
- IG path intervals: `128`.
- Baseline: `per_episode_reset_public_observation`.
- Privileged truth stored: `false`.

## Quantitative results

- `v_cmd_mps` completeness |delta|: median `3.994e-06`, P99 `1.778e-04`.
- `omega_cmd_rad_s` completeness |delta|: median `1.919e-05`, P99 `8.379e-05`.
- `critic_value` completeness |delta|: median `1.740e-05`, P99 `1.947e-04`.

Faithfulness AUC (larger action change is better for top-ranked deletion):

- `v_cmd_mps`: top `0.231687`, random `0.177026`, bottom `0.056167`, top/random `1.309`.
- `omega_cmd_rad_s`: top `0.059538`, random `0.062501`, bottom `0.034291`, top/random `0.953`.

Top features by mean absolute IG:

- `critic_value`: stop_line_distance_m (15.6%), stop_mode_none (10.0%), stop_sign_bearing_std_rad (8.9%), stop_sign_range_mean_m (8.2%), stop_sign_range_std_m (7.2%).
- `omega_cmd_rad_s`: pedestrian_range_mean_m (19.9%), lane_curvature_mean_inv_m (16.4%), lane_heading_error_mean_rad (10.0%), stop_sign_range_std_m (4.9%), pedestrian_existence_probability (4.8%).
- `v_cmd_mps`: pedestrian_range_mean_m (20.0%), stop_sign_range_mean_m (15.2%), pedestrian_existence_probability (14.1%), stop_line_distance_m (5.4%), previous_linear_velocity_cmd_mps (4.0%).

Top semantic groups by mean absolute IG:

- `critic_value`: stop_belief (45.3%), road (21.9%), lane_belief (14.0%).
- `omega_cmd_rad_s`: pedestrian_belief (31.8%), road (21.2%), lane_belief (21.0%).
- `v_cmd_mps`: pedestrian_belief (45.1%), stop_belief (31.4%), road (7.8%).

Adjacent sampled-frame attribution-rank stability:

- `v_cmd_mps`: median Spearman `0.9987`, mean `0.9919`.
- `omega_cmd_rad_s`: median Spearman `0.9975`, mean `0.9884`.
- `critic_value`: median Spearman `0.9985`, mean `0.9914`.

## Why the result is LIMITED

IG-ranked deletion improves velocity action faithfulness over the
random control, but it does not improve yaw-rate faithfulness over
random on the aggregate AUC. Steering attributions are therefore
useful descriptive sensitivities, not a validated causal ranking.

## Interpretation boundary

Integrated Gradients explains sensitivity of the frozen MLP relative
to each episode's public reset observation. It does not by itself
establish closed-loop causality. The separately registered semantic
counterfactual belief-ablation method remains unexecuted.

## BEV attribution map

Method 1 is additionally mapped onto the real C4 route for qualitative spatial
inspection. Seed `176001` was replayed for 2,375 frames; the replay matched the
frozen trajectory with maximum absolute observation error `0.0`, maximum
absolute action error `0.0`, and the same terminal event. The 594 stride-4 IG
samples were aligned by exact `(seed, step)` keys.

On this qualitative trajectory, pedestrian belief is the largest absolute
group on 65.2% of velocity samples and 75.9% of yaw-rate samples. Stop belief
dominates 18.9% of velocity samples, concentrated on the stop approach, while
lane/road groups become locally dominant for yaw at geometric transitions.
These are spatial descriptions relative to the episode-start baseline, not
causal claims.

The BEV has separate panels for signed `v_cmd`/`omega_cmd` IG and the dominant
semantic belief group for each action. Duckie crossing and stop-line/sign
geometry are shown only as evaluation references. Simulator world pose is kept
in `bev/evaluation_only_pose_trace.npz`; it never enters PPO or IG.

- Raster: `artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_action_map.png`
- Vector: `artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_action_map.pdf`
- Per-sample data: `artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_samples.csv`
- Provenance: `artifacts/f11_ppo_integrated_gradients_v1/bev/ig_bev_manifest.json`

## Tests

- Full active suite after BEV extension: `633 passed, 0 failed, 0 skipped`.
