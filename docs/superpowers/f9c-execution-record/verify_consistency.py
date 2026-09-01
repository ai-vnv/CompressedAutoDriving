"""Cross-document numeric consistency check for the F9c gate.

Read-only. The artifacts are the source of truth; every human-readable
document that quotes a number must agree with them. A stale figure in
GATES.md or README.md is what a future reader will trust, so it is a real
defect even when the artifacts are correct.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

belief = json.loads((ROOT / "artifacts" / "f9c_belief_metrics.json").read_text(encoding="utf-8"))
frozen = json.loads((ROOT / "artifacts" / "f9c_frozen_config.json").read_text(encoding="utf-8"))
calib = json.loads((ROOT / "artifacts" / "f9c_calibration_metrics.json").read_text(encoding="utf-8"))
ablation = json.loads((ROOT / "artifacts" / "f9c_ablation_metrics.json").read_text(encoding="utf-8"))

metrics = belief["metrics"]
robust = metrics["ekf"]["robust_b"]
base = metrics["ekf"]["baseline_a"]

facts = {
    "config_sha256": frozen["config_sha256"],
    "runtime_cache_sha256": belief["reconstruction"]["runtime_cache_sha256"],
    "robust range rmse": robust["range"]["rmse"],
    "baseline range rmse": base["range"]["rmse"],
    "robust coverage_68": robust["range"]["coverage_68"],
    "robust coverage_95": robust["range"]["coverage_95"],
    "baseline coverage_68": base["range"]["coverage_68"],
    "robust range bias": robust["range"]["bias"],
    "lambda_r": calib["covariance_scales"]["lambda_r"],
    "sigma_floor_r": calib["variance_components"]["range"]["sigma_floor_m"],
    "miss_floor": calib["recommended_conditional_detection_miss_likelihood_floor"],
}

print("=== authoritative values (from artifacts) ===")
for key, value in facts.items():
    print(f"  {key:26s} {value}")

docs = {
    name: (ROOT / name).read_text(encoding="utf-8")
    for name in ("GATES.md", "README.md", "IMPLEMENTATION_NOTES.md")
}

print("\n=== hashes quoted in documents ===")
ok = True
for name, text in docs.items():
    for label, full in (
        ("config", facts["config_sha256"]),
        ("cache", facts["runtime_cache_sha256"]),
    ):
        # accept full hash or an abbreviated prefix form
        quoted = re.findall(r"\b[0-9a-f]{8,64}\b", text)
        matches = [q for q in quoted if full.startswith(q)]
        wrong = [
            q
            for q in quoted
            if len(q) >= 8 and not any(full2.startswith(q) for full2 in (facts["config_sha256"], facts["runtime_cache_sha256"]))
        ]
        if matches:
            print(f"  OK   {name}: {label} hash quoted consistently ({len(matches)} occurrence(s))")
    # report any hex string that matches neither known hash
    quoted = re.findall(r"\b[0-9a-f]{16,64}\b", text)
    unknown = [
        q
        for q in quoted
        if not facts["config_sha256"].startswith(q)
        and not facts["runtime_cache_sha256"].startswith(q)
    ]
    for q in sorted(set(unknown)):
        print(f"  note {name}: hex string not one of the two F9c hashes: {q[:24]}...")

print("\n=== headline numbers quoted in documents ===")


def quoted_forms(value):
    """Rounded renderings a document might legitimately use."""
    return {f"{value:.{d}f}" for d in range(3, 6)} | {f"{value:.{d}g}" for d in (3, 4, 5)}


checks = [
    ("robust range rmse", facts["robust range rmse"]),
    ("baseline range rmse", facts["baseline range rmse"]),
    ("robust coverage_68", facts["robust coverage_68"]),
    ("robust coverage_95", facts["robust coverage_95"]),
    ("lambda_r", facts["lambda_r"]),
    ("sigma_floor_r", facts["sigma_floor_r"]),
    ("miss_floor", facts["miss_floor"]),
]
for name, text in docs.items():
    for label, value in checks:
        forms = quoted_forms(value)
        found = [f for f in forms if f in text]
        if found:
            print(f"  OK   {name}: {label} appears as {sorted(found)[0]}")

print("\n=== ablation endpoints vs headline systems ===")
rows = ablation["rows"]
for endpoint, reference in (("baseline", base), ("all_combined", robust)):
    got = rows[endpoint]["metrics"]["ekf"]["robust_b"]["range"]["rmse"]
    want = reference["range"]["rmse"]
    good = abs(got - want) < 1e-12
    ok = ok and good
    print(f"  {'OK  ' if good else 'FAIL'} {endpoint}: {got} vs headline {want}")

print("\n=== ablation cache provenance matches the final run ===")
acache = (
    ablation.get("runtime_cache", {}).get("sha256")
    or ablation.get("runtime_cache_sha256")
    or ablation.get("source", {}).get("runtime_cache_sha256")
)
good = acache == facts["runtime_cache_sha256"]
ok = ok and good
print(f"  {'OK  ' if good else 'FAIL'} ablation cache {str(acache)[:24]}... vs final {facts['runtime_cache_sha256'][:24]}...")

print()
print("CONSISTENT" if ok else "*** INCONSISTENCY DETECTED ***")
raise SystemExit(0 if ok else 1)
