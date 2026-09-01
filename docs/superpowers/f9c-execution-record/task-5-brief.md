## Task 5: Perception emits candidates

**Files:**
- Modify: `src/duckie_pomdp/perception/f9_pipeline.py`
- Test: `tests/test_f9_pipeline.py` (add tests; do not weaken existing ones)

**Interfaces:**
- Consumes: existing `YoloPedestrianMeasurementPipeline`, `Detection`, `ObjectClass`.
- Produces: `F9ImageObservation` gains a field `duckie_candidates: tuple[CandidateProjection, ...]` where `CandidateProjection(detection: Detection, measurement: ObjectMeasurement, projection_error: str | None)`. All existing fields keep their meaning and values so Baseline A is bit-identical.

- [ ] **Step 1: Write the failing tests**

```python
def test_pipeline_projects_every_duckie_candidate_not_only_the_selected_one():
    # Build a stub detector returning two Duckie boxes and one stop-sign box.
    observation = pipeline.observe(front_rgb)
    assert len(observation.duckie_candidates) == 2
    assert all(candidate.measurement.detected for candidate in observation.duckie_candidates)
    assert observation.duckie_detection_count == 2


def test_selected_duckie_still_matches_highest_confidence_for_baseline_parity():
    observation = pipeline.observe(front_rgb)
    best = max(observation.duckie_candidates, key=lambda item: item.detection.confidence)
    assert observation.selected_duckie is best.detection
    assert observation.pedestrian.range_m == best.measurement.range_m


def test_a_candidate_whose_projection_fails_is_kept_with_its_error_and_no_measurement():
    observation = pipeline.observe(front_rgb_with_horizon_box)
    failed = [c for c in observation.duckie_candidates if c.projection_error is not None]
    assert failed
    assert all(not c.measurement.detected for c in failed)


def test_no_duckie_detections_yields_no_candidates_and_a_missing_measurement():
    observation = pipeline.observe(empty_rgb)
    assert observation.duckie_candidates == ()
    assert not observation.pedestrian.detected
```

Reuse whatever stub-detector fixture `tests/test_f9_pipeline.py` already defines rather than writing a new one; read that file first and follow its existing pattern.

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9_pipeline.py -q`
Expected: FAIL — `AttributeError: 'F9ImageObservation' object has no attribute 'duckie_candidates'`

- [ ] **Step 3: Implement**

In `observe()`, project every Duckie detection (wrapping each `project_raw` call in its own `try/except ValueError`), build the candidate tuple, then keep the existing selection logic untouched to populate `pedestrian`, `selected_duckie`, `duplicate_selection`, and `projection_error`. Add `duckie_candidates` as the last field of `F9ImageObservation` with a default of `()` so any positional construction elsewhere keeps working.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9_pipeline.py -q`
Expected: PASS — all pre-existing tests plus 4 new ones.

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 143 passed. Confirm in the notes that `selected_duckie` semantics are unchanged, because Baseline A depends on it.

---

