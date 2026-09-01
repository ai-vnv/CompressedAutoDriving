# Model Manifest

Every file verified by SHA256 against the frozen experiment registries (F17/F18 pathway registries, F10 configs) at copy time.

| File | ID | Role | SHA256 | Source artifact |
|---|---|---|---|---|
| `actor_A0_original_fp32.pt` | A0 | original actor, 29-256-256-2, 73,986 params | `713d26d93488a17f...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/baseline/a0_original_actor.pt` |
| `actor_A1_pruned_fp32.pt` | A1 | structured-pruned to width 64 | `6e4ff154a209f44d...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/pruning/p75/actor_pruned_fp32.pt` |
| `actor_A2_kd_c4_fp32.pt` | A2 | pruned + KD on C4 data only | `fd79dba7c2b4aa63...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/prune_distill/pd75/actor_pruned_distilled_fp32.pt` |
| `actor_A3_kd_balanced_fp32.pt` | A3 | pruned + balanced C0-C4 KD (recovered anchor) | `64c84cd0bad44dda...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f15_cross_curriculum_recovery_v1/recovery/fp32/w64/actor_multicurriculum_kd_fp32.pt` |
| `actor_A4_ptq_unpruned_int8.pt` | A4 | PTQ of unpruned A0 (control) | `b79c2b4c489826cf...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/quant_only/actor_int8.pt` |
| `actor_A5_prune_ptq_int8.pt` | A5 | pruned + PTQ, no KD (control) | `cf0c093c7523ac11...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/prune_quant/actor_int8.pt` |
| `actor_A6_kd_balanced_ptq_int8.pt` | A6 | A3 + PTQ | `7ac05518f79a22b4...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f15_cross_curriculum_recovery_v1/recovery/ptq/w64/actor_int8.pt` |
| `actor_A7_pdqd_int8.pt` | A7 | historical prune+KD(C4)+PTQ+QAT(C4) | `f8e4e3ae5c43028d...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f12_belief_ppo_compression_v1/prune_distill_quant_distill/actor_int8.pt` |
| `actor_A8_kd_balanced_qat_int8.pt` | A8 | A3 parent + balanced QAT+KD | `c943e34f46a99c8c...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f15_cross_curriculum_recovery_v1/recovery/qat/w64/actor_int8.pt` |
| `actor_A9_kd_balanced_fp16.pt` | A9 | A3 cast to FP16 | `7e621e812aadae97...` | `/home/pannntastic/aivnv/duckie-pomdp/artifacts/f18_fp16_control_v1/candidates/actor_fp16.pt` |
| `yolo11n_duckietown_best.pt` | YOLO | fine-tuned YOLO11n detector (pedestrians, stop signs) | `3d4f816d44069049...` | `artifacts/yolo_v1/best.pt` |
| `mobilenetv3_lane_pose_best.pt` | LANE | MobileNetV3-small lane-pose regressor (d, phi, kappa) | `91d471d5ccf98750...` | `artifacts/f10_ppo_visual_v9/lane_rgb_model/best.pt` |

Full hashes:

```
713d26d93488a17fae246b227e1de38f51501dc87a3d20ac6176036a8a8e64c5  actor_A0_original_fp32.pt
6e4ff154a209f44daf5f6ba45415ce47d0d4b60506aed28689e3795113da3904  actor_A1_pruned_fp32.pt
fd79dba7c2b4aa63bdcbe0e28f84847e15ea15baba33a8e927e6b6136b18a69f  actor_A2_kd_c4_fp32.pt
64c84cd0bad44ddaa564a5895c88b82254950752b322030ce67df912a3667276  actor_A3_kd_balanced_fp32.pt
b79c2b4c489826cf7ea4853d7104be1630356deea0611138c9e606d0740b179d  actor_A4_ptq_unpruned_int8.pt
cf0c093c7523ac1188e0b94f7373277477eed8fac85ca3334498c3528bdf4358  actor_A5_prune_ptq_int8.pt
7ac05518f79a22b46d3079e4012eebf15f47605d40345d7e697ad45d100b79ee  actor_A6_kd_balanced_ptq_int8.pt
f8e4e3ae5c43028d7b5d08e64c31d20dcce28153fb102ffac53a3b1c7b7cbc7e  actor_A7_pdqd_int8.pt
c943e34f46a99c8c954a33d1dbd695fcc1fa81f3f7ffc15573dca653a12a1375  actor_A8_kd_balanced_qat_int8.pt
7e621e812aadae978cc12d8daa1b07d28b829106340e5f8683b7d462e8737d7d  actor_A9_kd_balanced_fp16.pt
3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c  yolo11n_duckietown_best.pt
91d471d5ccf9875012d564fa8937838fd0f95e6e3e6aabaefcad654d9b4bb84f  mobilenetv3_lane_pose_best.pt
```
