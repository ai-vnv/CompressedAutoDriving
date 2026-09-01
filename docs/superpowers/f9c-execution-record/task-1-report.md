# Task 1 Report — F9c protocol, config, and freeze-boundary guards

## What was implemented

1. `configs/f9c_robust_belief_v1.toml` — a copy of `configs/f9_yolo_ekf_v1.toml`
   with `[provenance]`, `[detector]`, `[simulator]`, `[trajectory_perturbation]`,
   all eight `[[scenario_matrix]]` entries, `[calibration_protocol]`, `[range]`,
   `[range_noise]`, `[bearing]`, `[covariance]`, `[ekf]`, and `[existence]`
   kept verbatim, plus the new/replaced sections from the brief: `schema_version`,
   `[split]` (new calibration/final/forbidden seeds), `[robust_observation]`,
   `[innovation_gate]`, `[association]`, `[covariance_calibration]`,
   `[measurement_model]` (reset to unfit defaults), `[baseline_measurement_model]`
   (F9b's frozen bias, copied verbatim), `[conditional_detection]`,
   `[existence_track]`, `[acceptance]`, `[minimum_support]`, `[artifacts]`
   (new F9c artifact paths).

2. `src/duckie_pomdp/evaluation/f9c_protocol.py` — new module, modeled on
   `f9_protocol.py`. Imports `F9ScenarioSpec` and `sha256` from `f9_protocol`
   (does not copy those dataclasses/functions). Defines
   `RobustObservationSwitches`, `AcceptanceBands`, `F9cProtocol` (frozen
   dataclasses) and `load_f9c_protocol(path, *, require_frozen=False)`.
   `_validate` enforces, in order:
   1. checkpoint sha256 matches `checkpoint_sha256`;
   2. `data["ekf"] == frozen_f7["ekf"]`, else `ValueError("F9c changes frozen F7 dynamics")` (contains literal substring `frozen F7`);
   3. `prior_probability` / `survival_probability` / `birth_probability` match the frozen F7 config;
   4. `calibration_seeds` and `final_evaluation_seeds` are nonempty, disjoint, and neither intersects `forbidden_seeds` nor the YOLO detector-manifest split seeds;
   5. when `require_frozen=True`: `measurement_model.parameters_frozen`, `covariance_calibration.parameters_frozen`, and `conditional_detection.parameters_frozen` are all true, `artifacts/f9c_frozen_config.json` exists, and its `config_sha256` matches `protocol.config_sha256`.

   A code comment explicitly states that `existence.detection_probability`
   is deliberately **not** re-validated against F7 — it is the intentionally
   unfrozen parameter for F9c.

3. `tests/test_f9c_protocol.py` — the brief's 6 tests, verbatim in intent and
   assertions. Two mechanical adaptations per the task's documented traps:
   - `import tomllib` wrapped in the `try/except ModuleNotFoundError: import tomli as tomllib` fallback (Python 3.10 has no stdlib `tomllib`).
   - `test_f9c_rejects_a_config_that_edits_process_noise` writes the mutated
     copy to `configs/_tmp_process_noise_probe.toml` (not `tmp_path`) because
     the loader resolves `frozen_f7_config`, the checkpoint, and
     `artifacts/...` relative to the config file's own directory; a copy
     under `tmp_path` would fail path resolution before ever reaching the
     process-noise/`[ekf]` check. Cleanup happens in a `try/finally` via
     `Path.unlink(missing_ok=True)`.

4. Appended a `## F9c` section to `IMPLEMENTATION_NOTES.md`: freeze-boundary
   table, seed allocation, and pre-specified acceptance bands, stating
   explicitly they were written before any 7101-series frame was rendered.

## TDD evidence

**RED** — before `f9c_protocol.py` existed:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_protocol.py -q'
```

```
==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_f9c_protocol.py __________________
ImportError while importing test module '.../tests/test_f9c_protocol.py'.
tests/test_f9c_protocol.py:5: in <module>
    from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
E   ModuleNotFoundError: No module named 'duckie_pomdp.evaluation.f9c_protocol'
1 error in 0.22s
```

This is exactly the expected failure reason (Step 2 of the brief).

**RED-for-the-right-reason check on the process-noise test** — after writing
the module but before trusting the test, I manually reproduced the mutated
config against the real loader (not `tmp_path`) and confirmed the specific
exception raised is the frozen-F7-dynamics check, not a path-resolution
error:

```python
CONFIG = Path('configs/f9c_robust_belief_v1.toml')
text = CONFIG.read_text(...).replace('position_process_std_m_per_sqrt_s = 0.001', '... = 0.002')
broken = Path('configs/_tmp_process_noise_probe.toml'); broken.write_text(text)
load_f9c_protocol(broken)   # -> ValueError
```

Output: `RAISED: F9c changes frozen F7 dynamics`

**GREEN** — focused file:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_protocol.py -q'
```

```
......                                                                   [100%]
6 passed in 0.47s
```

**GREEN** — full suite:

```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

```
114 passed, 260 warnings in 14.18s
```

108 pre-existing + 6 new = 114, matching the brief's expectation exactly. The
260 warnings are unchanged from the pre-task baseline (all from third-party
`zuper_nodes`/`gym` deprecation warnings in unrelated tests) — no new
warnings introduced.

## Files changed

- Created: `configs/f9c_robust_belief_v1.toml`
- Created: `src/duckie_pomdp/evaluation/f9c_protocol.py`
- Created: `tests/test_f9c_protocol.py`
- Modified: `IMPLEMENTATION_NOTES.md` (appended `## F9c` section)
- Not modified: `src/duckie_pomdp/evaluation/f9_protocol.py` (frozen, per instructions)

## Verification performed

- Confirmed `configs/f9c_robust_belief_v1.toml`'s parsed `[ekf]` block and
  `existence.{prior_probability,survival_probability,birth_probability}`
  equal `configs/oracle_ekf_v1.toml`'s (via a direct tomllib comparison, not
  just the unit test).
- Confirmed the kept sections (`provenance`, `detector`, `simulator`,
  `trajectory_perturbation`, `scenario_matrix`, `calibration_protocol`,
  `range`, `range_noise`, `bearing`, `covariance`) parse identically to
  `configs/f9_yolo_ekf_v1.toml`.
- Confirmed the YOLO checkpoint's actual sha256
  (`3d4f816d440690493b856d25403a84a3249e4250599319c32569b97cb8d7482c`) matches
  `provenance.checkpoint_sha256` in the new config.
- Manually exercised `require_frozen=True` against the (deliberately unfrozen)
  Task-1 config and confirmed it raises a clean `ValueError` ("F9c
  measurement/covariance/detection parameters are not frozen") rather than a
  `KeyError` or other crash — this path has no dedicated test in this task
  (none of the brief's 6 tests exercise it) but is exercised by later tasks.
- Confirmed no leftover `configs/_tmp_process_noise_probe.toml` after the
  test suite runs (cleanup verified via `ls configs/ | grep tmp`).

## Self-review

- **Completeness:** all 5 numbered `_validate` requirements from the brief
  are implemented; all 6 brief tests pass; `IMPLEMENTATION_NOTES.md` updated
  per Step 6.
- **Naming/interfaces:** `F9cProtocol` field names, `RobustObservationSwitches`,
  and `AcceptanceBands` match the brief's interface spec exactly (field names
  and order).
- **YAGNI:** `_validate` implements exactly the 5 checks the brief specifies —
  no extra checks (e.g., scenario-name uniqueness, IoU-threshold bounds) were
  added, unlike `f9_protocol.py`'s broader validation, because the brief's
  interface for `F9cProtocol` is deliberately slimmer (no `detector`,
  `matching_iou_threshold`, etc. fields) and those checks aren't in scope.
  `F9DetectorConfig` was not imported since `F9cProtocol` has no field that
  needs it — importing it unused would have been dead weight.
- **Reuse:** `F9ScenarioSpec` and `sha256` imported from `f9_protocol`, not
  copied. `_scenario_spec` in the new module is a small local builder (not a
  copy of `f9_protocol._scenario_spec`, which is private) — it exists because
  `f9_protocol.py` was explicitly frozen and its private helper wasn't listed
  in the "Consumes" interface.
- **Test quality:** the RED check for the process-noise test was done for
  the *actual* reason (frozen-F7-dynamics ValueError), not merely "some
  exception raised," per the brief's explicit warning about this trap.
- **Output pristine:** full suite run shows 114 passed, 0 failed, 0 skipped,
  no new warnings.

## Concerns

- None blocking. One judgment call worth flagging: the brief's `F9cProtocol`
  field list omits a `detector_dataset_manifest_path` field (unlike
  `F9Protocol`), so I read the detector manifest path directly from
  `data["provenance"]["detector_dataset_manifest"]` inside `_validate` rather
  than storing it on the dataclass. This matches the literal interface spec
  but means that path isn't independently accessible off `F9cProtocol` the
  way it is on `F9Protocol`. If a later task needs it, it can be added then.
- The `require_frozen=True` path (item 5) has no test coverage in this task
  because none of the brief's 6 tests exercise it — this is expected per the
  brief (Task 10 in the full plan writes `artifacts/f9c_frozen_config.json`
  and exercises this). I manually smoke-tested it (see Verification) to make
  sure it fails cleanly rather than crashing, but there is no automated
  regression test for it yet.
