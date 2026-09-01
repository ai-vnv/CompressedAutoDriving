"""Controller-side independent verification of the Task 10 freeze.

Read-only. Confirms the frozen config transcribes the FINAL calibration run
bit-for-bit, that the frozen F7 physics were not perturbed, that invariant I7
still holds, and that the recorded SHA256 matches the file on disk.
"""

import hashlib
import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

CONFIG = "configs/f9c_robust_belief_v1.toml"
ARTIFACT = "artifacts/f9c_calibration_metrics.json"
FROZEN = "artifacts/f9c_frozen_config.json"
F7 = "configs/oracle_ekf_v1.toml"

ok = True


def check(label, actual, expected):
    global ok
    good = actual == expected
    ok = ok and good
    mark = "OK  " if good else "FAIL"
    print("  %s %s: config=%r artifact=%r" % (mark, label, actual, expected))


with open(CONFIG, "rb") as stream:
    cfg = tomllib.load(stream)
with open(F7, "rb") as stream:
    f7 = tomllib.load(stream)
with open(ARTIFACT, encoding="utf-8") as stream:
    art = json.load(stream)
with open(FROZEN, encoding="utf-8") as stream:
    frz = json.load(stream)

print("=== config vs FINAL-run artifact (bit-for-bit) ===")
check(
    "range_bias_m",
    cfg["measurement_model"]["range_bias_m"],
    art["bias"]["fit"]["range_bias_m"],
)
check(
    "bearing_bias_rad",
    cfg["measurement_model"]["bearing_bias_rad"],
    art["bias"]["fit"]["bearing_bias_rad"],
)
check(
    "range_scale (lambda_r)",
    cfg["covariance_calibration"]["range_scale"],
    art["covariance_scales"]["lambda_r"],
)
check(
    "bearing_scale (lambda_beta)",
    cfg["covariance_calibration"]["bearing_scale"],
    art["covariance_scales"]["lambda_beta"],
)
check(
    "range_posterior_floor_m",
    cfg["covariance_calibration"]["range_posterior_floor_m"],
    art["variance_components"]["range"]["sigma_floor_m"],
)
check(
    "bearing_posterior_floor_rad",
    cfg["covariance_calibration"]["bearing_posterior_floor_rad"],
    art["variance_components"]["bearing"]["sigma_floor_rad"],
)
check(
    "miss_likelihood_floor",
    cfg["conditional_detection"]["miss_likelihood_floor"],
    art["recommended_conditional_detection_miss_likelihood_floor"],
)

print("=== first-pass contamination probe ===")
raw = open(CONFIG, encoding="utf-8").read()
contaminated = "10.125" in raw
ok = ok and not contaminated
print("  %s config does not contain the first-pass 10.125" % ("FAIL" if contaminated else "OK  "))

print("=== freeze flags ===")
for section in ("measurement_model", "covariance_calibration", "conditional_detection"):
    value = cfg[section].get("parameters_frozen")
    good = value is True
    ok = ok and good
    print("  %s %s.parameters_frozen = %r" % ("OK  " if good else "FAIL", section, value))

print("=== frozen F7 physics untouched ===")
check("[ekf] block", cfg["ekf"], f7["ekf"])
for key in ("prior_probability", "survival_probability", "birth_probability"):
    check("existence.%s" % key, cfg["existence"][key], f7["existence"][key])

print("=== invariant I7 (association strictly looser than gate) ===")
assoc = cfg["association"]["chi_square_gate"]
gate = cfg["innovation_gate"]["chi_square_threshold"]
good = assoc > gate
ok = ok and good
print("  %s association %r > gate %r" % ("OK  " if good else "FAIL", assoc, gate))

print("=== recorded hash vs file on disk ===")
sha = hashlib.sha256(open(CONFIG, "rb").read()).hexdigest()
recorded = frz.get("config_sha256")
good = sha == recorded
ok = ok and good
print("  %s live=%s..." % ("OK  " if good else "FAIL", sha[:24]))
print("       recorded=%s..." % str(recorded)[:24])
print("  final_evaluation_seeds_not_yet_rendered = %r" % frz.get("final_evaluation_seeds_not_yet_rendered"))
print("  calibration_seeds = %r" % frz.get("calibration_seeds"))
print("  final_evaluation_seeds = %r" % frz.get("final_evaluation_seeds"))

print()
print("ALL CHECKS PASS" if ok else "*** MISMATCH DETECTED ***")
raise SystemExit(0 if ok else 1)
