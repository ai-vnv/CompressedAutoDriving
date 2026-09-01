"""Post-hoc gate confusion analysis for F9c. Read-only, diagnostic only.

Answers: of the measurements the innovation gate judged, how many were genuine
localization errors and how many were good measurements it discarded?

DIAGNOSTIC ONLY. The config is frozen and the final seeds were rendered exactly
once; nothing here may be used to retune a threshold.

Also reconciles the two outlier counts that appear in the report:
`localization_outlier_count` and `outlier_impact.outlier_frame_count`.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
rows = list(csv.DictReader((ROOT / "artifacts" / "f9c_validation.csv").open(encoding="utf-8")))

IOU_MATCH = 0.5


def num(row, key):
    value = row.get(key, "")
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def flag(row, key):
    return str(row.get(key, "")).strip().lower() == "true"


print(f"total rows: {len(rows)}")

# --- the gate confusion table -------------------------------------------------
# Only frames where the gate actually ran: association selected a candidate and a
# decision was recorded. The IoU that matters is the one of the box the gate judged.
judged = [
    row
    for row in rows
    if str(row.get("robust_b_gate_decision", "")).strip() not in ("", "None")
    and num(row, "robust_b_associated_iou") is not None
]
print(f"frames where the gate ran with a GT-comparable box: {len(judged)}")

table = {}
for row in judged:
    decision = str(row["robust_b_gate_decision"]).strip().lower()
    accepted = decision in ("accept", "accepted", "true")
    good = num(row, "robust_b_associated_iou") >= IOU_MATCH
    table[(good, accepted)] = table.get((good, accepted), 0) + 1

tp = table.get((False, False), 0)   # bad localization, correctly rejected
fn = table.get((False, True), 0)    # bad localization, wrongly accepted
fp = table.get((True, False), 0)    # good localization, wrongly rejected
tn = table.get((True, True), 0)     # good localization, correctly accepted

print()
print("=== gate confusion (GT localization x gate decision) ===")
print(f"{'':22s}{'gate accept':>13s}{'gate reject':>13s}")
print(f"{'IoU >= 0.5 (good)':22s}{tn:>13d}{fp:>13d}")
print(f"{'IoU <  0.5 (outlier)':22s}{fn:>13d}{tp:>13d}")

bad = tp + fn
good = tn + fp
print()
if bad:
    print(f"outlier rejection sensitivity : {tp}/{bad} = {tp/bad:.4f}")
else:
    print("outlier rejection sensitivity : undefined (no gate-judged outliers)")
if good:
    print(f"good-measurement false reject : {fp}/{good} = {fp/good:.6f}")
if tp + fp:
    print(f"rejection precision           : {tp}/{tp+fp} = {tp/(tp+fp):.4f}")

# --- reconcile the two outlier counts ----------------------------------------
print()
print("=== reconciling the two outlier counts ===")

visible_detected = [
    row for row in rows if flag(row, "eligible_visible") and flag(row, "detector_detected")
]
sel_bad = [
    row
    for row in visible_detected
    if num(row, "selected_iou") is not None and num(row, "selected_iou") < IOU_MATCH
]
assoc_bad = [
    row
    for row in visible_detected
    if num(row, "robust_b_associated_iou") is not None
    and num(row, "robust_b_associated_iou") < IOU_MATCH
]
print(f"frames, eligible+detected, Baseline-A selection IoU < 0.5 : {len(sel_bad)}")
print(f"frames, eligible+detected, Robust-B associated IoU  < 0.5 : {len(assoc_bad)}")

# contiguous runs of Baseline-A localization failure, per episode
by_episode = {}
for row in rows:
    by_episode.setdefault(row["episode"], []).append(row)
runs = 0
for episode, frames in by_episode.items():
    frames.sort(key=lambda r: int(r["frame"]))
    previous = False
    for row in frames:
        iou = num(row, "selected_iou")
        current = (
            flag(row, "eligible_visible")
            and flag(row, "detector_detected")
            and iou is not None
            and iou < IOU_MATCH
        )
        if current and not previous:
            runs += 1
        previous = current
print(f"contiguous localization-failure EVENTS (runs) on the Baseline-A path : {runs}")
print()
print("Interpretation: the two numbers count different things -- discrete events")
print("versus the individual frames those events span. Both are correct; the")
print("report must name which is which.")
