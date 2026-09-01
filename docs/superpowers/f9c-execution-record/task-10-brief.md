## Task 10: Freeze the configuration

**Files:**
- Modify: `configs/f9c_robust_belief_v1.toml`
- Create: `artifacts/f9c_frozen_config.json`
- Create: `experiments/verify_f9c_artifacts.py`

- [ ] **Step 1: Write the fitted values into the config and set all three `parameters_frozen = true`**

Copy every number from `artifacts/f9c_calibration_metrics.json` into `[measurement_model]`, `[covariance_calibration]`, and `[conditional_detection]`. Set `parameters_frozen = true` in all three sections.

- [ ] **Step 2: Write `artifacts/f9c_frozen_config.json`**

Contains `config_sha256`, `checkpoint_sha256`, `calibration_artifact_sha256`, `frozen_f7_config_sha256`, `calibration_seeds`, `final_evaluation_seeds`, the full fitted parameter set, the pre-specified acceptance bands, and the pre-specified minimum support — plus an ISO timestamp and the literal statement `"final_evaluation_seeds_not_yet_rendered": true`.

- [ ] **Step 3: Verify `require_frozen=True` now loads**

```bash
$PY -c "
from pathlib import Path
from duckie_pomdp.evaluation.f9c_protocol import load_f9c_protocol
p = load_f9c_protocol(Path('configs/f9c_robust_belief_v1.toml'), require_frozen=True)
print('frozen config sha256:', p.config_sha256)
"
```

Expected: prints a hash and does not raise.

- [ ] **Step 4: Write `experiments/verify_f9c_artifacts.py`**

Model it on `experiments/verify_f9_artifacts.py`. It must re-derive every metric in `artifacts/f9c_belief_metrics.json` from `artifacts/f9c_validation.csv` without running inference, and re-check every hash. It must exit non-zero on any mismatch.

- [ ] **Step 5: Checkpoint**

Run: `$PY -m pytest tests -q` → expect 186 passed. Record the frozen `config_sha256` in `IMPLEMENTATION_NOTES.md` and in `GATES.md` as the pre-final-run witness. **From this point the config is read-only until F9c reports.**

---

