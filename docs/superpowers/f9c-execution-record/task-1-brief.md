## Task 1: F9c protocol, config, and freeze-boundary guards

**Files:**
- Create: `configs/f9c_robust_belief_v1.toml`
- Create: `src/duckie_pomdp/evaluation/f9c_protocol.py`
- Test: `tests/test_f9c_protocol.py`

**Interfaces:**
- Consumes: `duckie_pomdp.evaluation.f9_protocol.sha256`, `F9ScenarioSpec`, `load_scenario`.
- Produces: `F9cProtocol` dataclass with fields `config_path`, `checkpoint_path`, `checkpoint_sha256`, `frozen_f7_config_path`, `calibration_seeds: tuple[int, ...]`, `final_evaluation_seeds: tuple[int, ...]`, `forbidden_seeds: tuple[int, ...]`, `scenarios: tuple[F9ScenarioSpec, ...]`, `robust: RobustObservationSwitches`, `acceptance: AcceptanceBands`, `minimum_support: dict[str, int]`, `artifacts: dict[str, Path]`, plus `config_sha256` property; `RobustObservationSwitches(bias_refit: bool, innovation_gate: bool, temporal_association: bool, covariance_calibration: bool, conditional_detection: bool)`; `AcceptanceBands(coverage_68_low, coverage_68_high, coverage_95_low, coverage_95_high, max_std_over_rmse, max_rmse_ratio_vs_baseline)`; `load_f9c_protocol(path, *, require_frozen: bool = False) -> F9cProtocol`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_f9c_protocol.py
from pathlib import Path

import pytest

from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "f9c_robust_belief_v1.toml"


def test_f9c_seeds_are_disjoint_from_every_earlier_split():
    protocol = load_f9c_protocol(CONFIG)
    calibration = set(protocol.calibration_seeds)
    final = set(protocol.final_evaluation_seeds)
    assert calibration and final
    assert not calibration & final
    forbidden = set(protocol.forbidden_seeds)
    assert {5101, 5102, 5103, 5104} <= forbidden
    assert {4101, 4102, 4103, 4104} <= forbidden
    assert not (calibration | final) & forbidden


def test_f9c_may_not_change_frozen_f7_dynamics(tmp_path):
    protocol = load_f9c_protocol(CONFIG)
    import tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    assert f9c["ekf"] == f7["ekf"]


def test_f9c_keeps_survival_and_birth_frozen():
    protocol = load_f9c_protocol(CONFIG)
    import tomllib

    with protocol.config_path.open("rb") as stream:
        f9c = tomllib.load(stream)
    with protocol.frozen_f7_config_path.open("rb") as stream:
        f7 = tomllib.load(stream)
    for key in ("prior_probability", "survival_probability", "birth_probability"):
        assert f9c["existence"][key] == f7["existence"][key]


def test_f9c_rejects_a_config_that_edits_process_noise(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace(
        "position_process_std_m_per_sqrt_s = 0.001",
        "position_process_std_m_per_sqrt_s = 0.002",
    )
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen F7"):
        load_f9c_protocol(broken)


def test_f9c_acceptance_bands_are_pre_specified():
    protocol = load_f9c_protocol(CONFIG)
    bands = protocol.acceptance
    assert bands.coverage_68_low == 0.60 and bands.coverage_68_high == 0.76
    assert bands.coverage_95_low == 0.90 and bands.coverage_95_high == 0.98
    assert bands.max_std_over_rmse == 1.5
    assert protocol.minimum_support["near"] >= 100


def test_f9c_ablation_switches_default_to_all_enabled():
    protocol = load_f9c_protocol(CONFIG)
    switches = protocol.robust
    assert switches.bias_refit
    assert switches.innovation_gate
    assert switches.temporal_association
    assert switches.covariance_calibration
    assert switches.conditional_detection
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckie_pomdp.evaluation.f9c_protocol'`

- [ ] **Step 3: Write `configs/f9c_robust_belief_v1.toml`**

Copy `configs/f9_yolo_ekf_v1.toml` and change only the sections below. The `[ekf]` block and the three `[existence]` keys `prior_probability`, `survival_probability`, `birth_probability` must be byte-identical to `configs/oracle_ekf_v1.toml`.

```toml
schema_version = 1

[split]
calibration_seeds = [6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108]
final_evaluation_seeds = [7101, 7102, 7103, 7104]
forbidden_seeds = [1101, 1102, 1103, 1104, 1105, 1106, 2101, 2102, 3101, 3102, 4101, 4102, 4103, 4104, 5101, 5102, 5103, 5104]

[robust_observation]
bias_refit = true
innovation_gate = true
temporal_association = true
covariance_calibration = true
conditional_detection = true

[innovation_gate]
# Hard reject only in F9c v1. A soft-downweight variant would make the
# correction use 25*lambda*R while association and the gate used lambda*R,
# breaking invariant I1. See "Why there is no downweight mode".
mode = "hard_reject"
chi_square_threshold = 9.21034037197618   # 2 DOF, 99%
# Invariant I2: a gated-out bbox suppresses the EKF correction only. Existence
# evidence comes from the detector, never from the gate decision.
existence_evidence_source = "detector"

[association]
rule = "minimum_nis"
chi_square_gate = 9.21034037197618
initialization_rule = "highest_confidence_then_bbox_lexicographic"

[covariance_calibration]
# Filled by experiments/calibrate_f9c_robust_belief.py, then frozen.
parameters_frozen = false
range_scale = 1.0
bearing_scale = 1.0
range_posterior_floor_m = 0.0
bearing_posterior_floor_rad = 0.0

[measurement_model]
# F9c-fitted bias, used when robust_observation.bias_refit = true.
parameters_frozen = false
bias_model = "global_additive"   # or "per_range_bin" if the LOSO rule in Task 9 selects it
range_bias_m = 0.0
bearing_bias_rad = 0.0

[baseline_measurement_model]
# F9b frozen bias. This is Baseline A's correction and is also what every
# ablation row with bias_refit = false uses. Never refit; copied verbatim from
# configs/f9_yolo_ekf_v1.toml so "all switches off == Baseline A" holds by
# construction rather than by coincidence.
bias_model = "global_additive"
range_bias_m = -0.045904804710162034
bearing_bias_rad = 0.00414567890700929

[conditional_detection]
# Filled by calibration; keys are predicted-observability classes.
parameters_frozen = false
detection_probability_center = 0.9766775777414075
detection_probability_mid_fov = 0.9766775777414075
detection_probability_edge_fov = 0.9766775777414075
# Invariant I3: this value is a reported diagnostic only. It is never applied to
# a miss, because an outside-domain absence carries no likelihood.
detection_probability_outside_domain = 0.9766775777414075
outside_domain_miss_policy = "prediction_only"
false_positive_probability = 0.00078003120124805

[existence_track]
active_threshold = 0.50
delete_threshold = 0.05
initialization_threshold = 0.50

[acceptance]
coverage_68_low = 0.60
coverage_68_high = 0.76
coverage_95_low = 0.90
coverage_95_high = 0.98
max_std_over_rmse = 1.5
max_rmse_ratio_vs_baseline = 1.15

[minimum_support]
near = 100
medium = 200
far = 200
edge_fov = 50

[artifacts]
calibration_csv = "../artifacts/f9c_calibration.csv"
calibration_metrics_json = "../artifacts/f9c_calibration_metrics.json"
frozen_config_json = "../artifacts/f9c_frozen_config.json"
validation_csv = "../artifacts/f9c_validation.csv"
belief_metrics_json = "../artifacts/f9c_belief_metrics.json"
ablation_metrics_json = "../artifacts/f9c_ablation_metrics.json"
nis_metrics_json = "../artifacts/f9c_nis_metrics.json"
error_case_dir = "../artifacts/f9c_error_cases"
runtime_cache = "../artifacts/f9c_runtime_cache.npz"
evaluation_truth = "../artifacts/f9c_evaluation_truth.npz"
calibration_log = "../artifacts/f9c_calibration.log"
validation_log = "../artifacts/f9c_validation.log"
```

- [ ] **Step 4: Write `src/duckie_pomdp/evaluation/f9c_protocol.py`**

Model it on `f9_protocol.py`. The `_validate` function must:
1. verify `sha256(checkpoint_path) == checkpoint_sha256`;
2. raise `ValueError("F9c changes frozen F7 dynamics")` if `data["ekf"] != frozen_f7["ekf"]` — message must contain the literal substring `frozen F7`;
3. raise if any of `prior_probability`, `survival_probability`, `birth_probability` differ from the frozen F7 config;
4. raise if `calibration_seeds` and `final_evaluation_seeds` are empty or overlapping, or if either intersects `forbidden_seeds`, or if either intersects the detector manifest seeds;
5. when `require_frozen=True`, raise unless `measurement_model.parameters_frozen`, `covariance_calibration.parameters_frozen`, and `conditional_detection.parameters_frozen` are all true and `artifacts/f9c_frozen_config.json` exists with a matching `config_sha256`.

Note that F9c **deliberately does not** re-validate `data["existence"]["detection_probability"]` against F7 — that is the unfrozen parameter. Add a comment saying so.

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_f9c_protocol.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 114 passed. Append to `IMPLEMENTATION_NOTES.md` under `## F9c`: the freeze-boundary table, the seed allocation, and the pre-specified acceptance bands, stating explicitly that they were written before any 7101-series frame was rendered.

---

