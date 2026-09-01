## Task 2: Final-evaluation near-range scenarios

**Files:**
- Modify: `configs/f9c_robust_belief_v1.toml` (`[[scenario_matrix]]` entries)
- Test: `tests/test_f9c_protocol.py` (add one test)

**Interfaces:**
- Consumes: `F9cProtocol.scenarios` from Task 1.
- Produces: scenario names `approach_near_stationary_ego`, `approach_near_moving_ego`, `cross_near_left_to_right` usable by both calibration and final evaluation.

The existing near scenario uses `ego_start_x_offset_m = 0.50` with `use_for_final_evaluation = false`, which is exactly why F9b had N=0 near. F9c needs near frames on **both** sides of the split.

- [ ] **Step 1: Write the failing test**

```python
def test_f9c_scenario_matrix_supports_near_range_final_evaluation():
    protocol = load_f9c_protocol(CONFIG)
    final = [spec for spec in protocol.scenarios if spec.use_for_final_evaluation]
    calibration = [spec for spec in protocol.scenarios if spec.use_for_calibration]
    near_final = [spec for spec in final if spec.ego_start_x_offset_m >= 0.35]
    near_calibration = [spec for spec in calibration if spec.ego_start_x_offset_m >= 0.35]
    assert len(near_final) >= 2, "final evaluation must contain near-range scenarios"
    assert len(near_calibration) >= 2
    # A moving-ego approach is required so near range is traversed, not only sampled statically.
    assert any(
        spec.action.linear_velocity_mps > 0.0 for spec in near_final
    ), "at least one near-range final scenario must approach the pedestrian"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_f9c_protocol.py::test_f9c_scenario_matrix_supports_near_range_final_evaluation -q`
Expected: FAIL — `assert 0 >= 2`

- [ ] **Step 3: Add the scenarios to `configs/f9c_robust_belief_v1.toml`**

Keep the six original F9b scenarios byte-identical (they are Baseline A's trajectories), flip the two old `calibration_*` scenarios to `use_for_final_evaluation = true` under new names, and add a moving approach:

```toml
[[scenario_matrix]]
name = "approach_near_stationary_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 60
ego_start_x_offset_m = 0.50
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "approach_medium_stationary_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 60
ego_start_x_offset_m = 0.25
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "approach_near_moving_ego"
pedestrian_mode = "stationary"
linear_velocity_mps = 0.20
angular_velocity_rad_s = 0.0
steps = 90
ego_start_x_offset_m = 0.30
use_for_calibration = true
use_for_final_evaluation = true

[[scenario_matrix]]
name = "cross_near_left_to_right"
pedestrian_mode = "cross_left_to_right"
linear_velocity_mps = 0.0
angular_velocity_rad_s = 0.0
steps = 110
ego_start_x_offset_m = 0.40
use_for_calibration = true
use_for_final_evaluation = true
```

`approach_near_moving_ego` sweeps range continuously downward, which is what makes the near bin a traversed regime rather than a single static pose — and it is the regime a future stop policy will actually operate in.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Verify the support minima are actually reachable — dry run on ONE calibration seed**

Run:

```bash
$PY - <<'PY'
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
protocol = load_f9c_protocol(Path("configs/f9c_robust_belief_v1.toml"))
final = [s for s in protocol.scenarios if s.use_for_final_evaluation]
print(sum(s.steps + 1 for s in final) * len(protocol.final_evaluation_seeds), "final frames")
PY
```

Then render seed 6101 only through the Task 11 collector once it exists, and count `distance_bin`. **If near < 100/4 seeds proportionally, increase `steps` or `ego_start_x_offset_m` NOW, before any 7101 frame is rendered.** Record the dry-run counts in `IMPLEMENTATION_NOTES.md`. Adjusting scenario geometry after seeing final-seed results is a gate failure; adjusting it from a calibration-seed dry run is correct practice.

- [ ] **Step 6: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 115 passed. Note the dry-run distance-bin counts in `IMPLEMENTATION_NOTES.md`.

---

