# F10-PPO object curriculum v13 — policy-visible actor warm start

V13 preserves the frozen 29D RGB→MobileNet lane belief + YOLO→F9c pedestrian
belief runtime, counter-clockwise `experiment_loop`, continuous action adapter,
canonical PPO hyperparameters, object geometry, and all V10 acceptance gates.

V12 was stopped at step 20,480 because full-timing episodes remained 93–97%
collision. Sparse on-policy exploration did not discover the required sequence.
Before canonical PPO updates, V13 therefore gives the inherited C1 actor a
supervised behavior warm start from 25,600 **training-only policy observations**
recorded by the failed V12 attempt. Targets are computed by the existing simple
belief-aware controller. Only the 29 `policy.*` columns are read; evaluation GT,
privileged state, images, and future data are not inputs. Stop-action examples
receive weight 8; 8,392 of 25,600 rows are stop targets.

The actor/critic weights still originate from the passing C1 checkpoint. Adam
state is reset and log-std begins at -1.20 as in V12. The critic is not behavior
cloned. After warm start, learning is ordinary canonical PPO. The zero-step
warm-start checkpoint is a pre-registered development candidate alongside
10,240/20,480/30,720/40,960 PPO checkpoints, so development evaluation can
detect and reject PPO degradation instead of hiding it.

Physical Duckie speed remains 0.20 m/s. Training-only RTL delays are
0.60→1.00→1.55 s; development and stage-final always use the full 1.55 s.
C3 remains locked until C2 development, retention, and stage-final PASS.

Launch uses the same exports and command as V12, replacing config/artifact
names with `f10_ppo_visual_objects_v13`.
