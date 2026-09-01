# F15 Visual Evidence Outcome

## Evidence boundary

The F15 localization traces store public physical/normalized 29D inputs, physical and
normalized actions, progress, phase, objective event flags, and separate evaluation-only
arrays. They do not store front-camera RGB frames.

The frozen visual-replay amendment allowed a recorded-action same-seed simulator replay
only when the replay reproduced the complete recorded action sequence, trajectory, and
objective outcome within its preregistered criteria. The first two attempts terminated
before the frozen failure window and were classified `UNRESOLVED`. Their media remain
quarantined under:

- `artifacts/f15_cross_curriculum_recovery_v1/failure_traces/A1/c0/seed_180002/unresolved/`
- `artifacts/f15_cross_curriculum_recovery_v1/failure_traces/A1/c1/seed_180002/unresolved/`

They are retained for audit and must not be presented as visual evidence of those frozen
episodes.

## Valid visualization used instead

F15 therefore visualizes the immutable primary telemetry directly. No simulator replay,
perception inference, actor inference, or synthetic camera frame is used. Each objectively
selected failure produces:

- a short MP4 telemetry animation;
- a GIF;
- a PNG contact sheet;
- a CSV containing the plotted Original and compressed values;
- a hash-bound JSON record.

The aggregate manifest is:

`artifacts/f15_cross_curriculum_recovery_v1/failure_telemetry/failure_telemetry_manifest.json`

Passing/Reference cells receive a deterministic successful-episode telemetry animation,
selected as the lowest seed with completion and no objective failure flag. Its aggregate
manifest is:

`artifacts/f15_cross_curriculum_recovery_v1/success_telemetry/success_telemetry_manifest.json`

These animations are direct views of recorded evidence, but they are not camera videos.
The Original-versus-compressed comparison is a same-seed recorded telemetry comparison,
not a causal paired trajectory.

## Separate qualitative simulator video

The already existing A7 C4 qualitative simulator video is:

`artifacts/f12_belief_ppo_compression_v1/final/a7_c4_front_bev.mp4`

Its manifest labels it `qualitative_example_only`. It is useful for understanding the
runtime architecture and visual behavior, but it is not an F15 localization replicate or
holdout result.
