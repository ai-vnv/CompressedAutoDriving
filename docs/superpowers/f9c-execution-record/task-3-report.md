# Task 3 Report: Innovation Gate

## Status
DONE_WITH_CONCERNS - Implementation completed exactly as specified, but one boundary-condition test fails due to floating-point precision.

## Files Created
1. `src/duckie_pomdp/belief/innovation_gate.py` - Implementation (complete, exact transcription from brief)
2. `tests/test_f9c_innovation_gate.py` - Test suite (complete, exact transcription from brief)

## TDD Evidence

### Step 1-2: RED Phase
**Command:**
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_innovation_gate.py -q'
```

**Output:**
```
ERROR tests/test_f9c_innovation_gate.py
ImportError while importing test module '/home/pannntastic/aivnv/duckie-pomdp/tests/test_f9c_innovation_gate.py'.
Traceback:
E   ModuleNotFoundError: No module named 'duckie_pomdp.belief.innovation_gate'
```

**Expected:** Test collection fails with ModuleNotFoundError ✓ (Step 2 passed)

### Step 3-4: GREEN Phase (Partial)
**Command:**
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_innovation_gate.py -q'
```

**Output:**
```
.....F..                                                                 [100%]
=================================== FAILURES ===================================
_______________ test_gate_is_exactly_inclusive_at_the_threshold ________________

    def test_gate_is_exactly_inclusive_at_the_threshold():
        covariance = np.diag([1.0, 1.0])
        innovation = np.array([THRESHOLD**0.5, 0.0])
        decision = _gate().evaluate(innovation, covariance)
        assert decision.nis == pytest.approx(THRESHOLD)
>       assert decision.accepted, "NIS exactly at the threshold must be accepted"
E       AssertionError: NIS exactly at the threshold must be accepted
E       assert False
E        +  where False = GateDecision(accepted=False, nis=9.210340371976182, threshold=9.21034037197618).accepted
```

**Result:** 7 of 8 tests pass; 1 failure on boundary condition

### Step 5: Full Suite
**Command:**
```
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'
```

**Output Summary:**
- Total: 122 passed, 1 failed
- Expected: 123 passed
- The single failure is `test_gate_is_exactly_inclusive_at_the_threshold`

## Concern: Floating-Point Precision Edge Case

### Root Cause
The test `test_gate_is_exactly_inclusive_at_the_threshold` creates an innovation vector computed as:
```python
innovation = np.array([THRESHOLD**0.5, 0.0])
```

With identity covariance, the NIS should be computed as `(THRESHOLD**0.5)**2 = THRESHOLD`. However:

1. Taking the square root of `9.21034037197618` introduces floating-point rounding
2. The quadratic form computation `vector @ np.linalg.solve(symmetric, vector)` produces `9.210340371976182`
3. This is approximately `2e-15` larger than the threshold due to accumulated floating-point error
4. The gate's decision logic uses `nis <= threshold`, which fails when `nis > threshold` by this tiny margin

### Expected vs. Actual
- Threshold: `9.21034037197618`
- Computed NIS: `9.210340371976182` (larger by ~2e-15)
- First assertion (`nis == pytest.approx(THRESHOLD)`) passes because `pytest.approx()` uses relative tolerance
- Second assertion (`decision.accepted`) fails because exact comparison `nis <= threshold` is False

### Verification of Transcription
Both implementation and test code were transcribed exactly character-by-character from the brief:
- THRESHOLD constant: `9.21034037197618` (exact match)
- Implementation logic: bit-for-bit copy of brief lines 144-158
- All method signatures, names, and assertions match precisely

The brief provides no special handling for this boundary case, and the implementation follows the spec exactly.

## Implementation Quality

### Positive Findings
- All 7 non-boundary tests pass perfectly
- Code correctly implements:
  - `InnovationGateConfig` validation (chi_square_threshold finite and positive)
  - `GateDecision` structure with frozen dataclass
  - `normalized_innovation_squared()` with proper error handling:
    - Shape validation (2D polar innovation required)
    - Finiteness check
    - Positive-definite covariance check via eigenvalue test
    - Symmetrization of covariance for numerical stability
  - `InnovationGate.evaluate()` with clean interface
- Invariant I1 enforced: no `GateMode`, `covariance_scale`, or `downweighted` fields
- Signature correctly restricted to only `{self, innovation, innovation_covariance}`
- All validation tests pass (non-positive-definite rejection, signature check, etc.)

### Files Modified
- Created: `src/duckie_pomdp/belief/innovation_gate.py` (59 lines)
- Created: `tests/test_f9c_innovation_gate.py` (62 lines)
- Modified: None (additive only, no __init__.py export as per spec)

## Recommendations

1. **For now:** Accept this as known floating-point precision limitation at the exact threshold boundary. This is a pathological edge case—in practice, NIS values will have natural variation that doesn't hit this exact floating-point cliff.

2. **If this must pass:** The test's THRESHOLD constant would need additional decimal places, or the gate's comparison would need epsilon tolerance (both changes to the brief's spec).

3. **No code changes needed:** The implementation follows the brief exactly and passes 7/8 tests plus validates all invariants.

## Test Summary (Initial Round)
- Innovation gate tests: 7 passing, 1 failing (floating-point boundary edge case)
- Full suite: 122 passing (1 failure from task 3, 121 pre-existing passes expected)
- Checkpoint expectation: 123 passed, actual: 122 passed (off by 1 due to floating-point precision in boundary test)

---

## Fix Round 1/5: Corrected Boundary Test Construction

### Diagnosis
The defect was in the test itself, not the implementation. The test attempted to construct a boundary value by computing `THRESHOLD**0.5` and later squaring it back, but this round-trip introduces ~2e-15 floating-point error. The implementation is correct.

### Change
**File:** `tests/test_f9c_innovation_gate.py`

**What Changed:** Replaced `test_gate_is_exactly_inclusive_at_the_threshold` to build the boundary exactly instead of round-tripping through sqrt.

**Old Test (Lines 56-61):**
```python
def test_gate_is_exactly_inclusive_at_the_threshold():
    covariance = np.diag([1.0, 1.0])
    innovation = np.array([THRESHOLD**0.5, 0.0])
    decision = _gate().evaluate(innovation, covariance)
    assert decision.nis == pytest.approx(THRESHOLD)
    assert decision.accepted, "NIS exactly at the threshold must be accepted"
```

**New Test (Lines 56-70):**
```python
def test_gate_is_exactly_inclusive_at_the_threshold():
    # Build the boundary exactly rather than round-tripping through sqrt:
    # (THRESHOLD**0.5)**2 lands ~2e-15 above THRESHOLD, so an inclusive gate
    # would appear to reject its own boundary for purely floating-point reasons.
    covariance = np.diag([1.0, 1.0])
    innovation = np.array([3.0, 0.0])
    nis = normalized_innovation_squared(innovation, covariance)
    assert nis == 9.0
    gate = InnovationGate(InnovationGateConfig(chi_square_threshold=nis))
    decision = gate.evaluate(innovation, covariance)
    assert decision.nis == nis
    assert decision.accepted, "NIS exactly at the threshold must be accepted"
    just_above = gate.evaluate(np.array([3.0 + 1.0e-6, 0.0]), covariance)
    assert not just_above.accepted, "the boundary must still be a boundary"
```

### Why This Works
- `3.0**2 / 1.0 = 9.0` is exact in binary floating point
- `np.linalg.solve(identity, vector)` is exact  
- The NIS computation returns exactly `9.0`
- The threshold is set to this identical value
- The `nis <= threshold` inclusivity is tested directly with no rounding involved
- Added companion assertion to verify the boundary is still a boundary (rejection just above)

### Test Results

**Innovation Gate Tests Only:**
```
Command:
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests/test_f9c_innovation_gate.py -q'

Output:
........                                                                 [100%]
8 passed in 0.23s
```

**Full Test Suite:**
```
Command:
wsl.exe -d Ubuntu-Baru -- bash -lc 'cd /home/pannntastic/aivnv/duckie-pomdp && export PYTHONPATH=src:/home/pannntastic/aivnv/duckie/src && export DUCKIETOWN_HEADLESS=1 && /home/pannntastic/aivnv/duckie/.venv/bin/python -m pytest tests -q'

Output:
........................................................................ [ 58%]
...................................................                      [100%]
123 passed, 260 warnings in 10.38s
```

### Verification
- No changes to implementation file (`src/duckie_pomdp/belief/innovation_gate.py`) — it was correct all along
- No changes to `belief/__init__.py` as per spec
- No changes to other 7 tests in `test_f9c_innovation_gate.py`
- All 123 tests now pass (checkpoint achieved)
- Invariant I1 still enforced
- Boundary test now proves both:
  1. Values exactly at threshold are accepted (`nis <= threshold` with `nis == threshold`)
  2. Values just above threshold are rejected (boundary is still a boundary)
