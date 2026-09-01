"""Configuration and freeze-boundary guards for the F9d evidence-closure runs.

F9c produced a pedestrian belief estimator and was classified LIMITED because
two claims had no evidence: gross localization-outlier robustness, and
long-absence existence decay. F9d exists only to collect that evidence -- it
adds no estimator capability whatsoever, and it must not be able to tune
F9c's parameters even by accident.

This module enforces that structurally:

* The F9d config carries no estimator-parameter sections at all (no
  ``measurement_model``, ``covariance_calibration``, ``conditional_detection``,
  ``innovation_gate``, ``association``, ``ekf``, or ``existence``). Adding one
  would be an estimator change wearing a stress-test costume.
* F9c's parameters are exposed only through a read-only accessor,
  :func:`f9c_parameters`, that reloads the frozen F9c config from disk on
  every call. There is no cached copy of any F9c parameter anywhere in this
  module, so there is nothing that can silently drift.
* F9d's own config pins the F9c config by SHA256 and refuses to load if the
  F9c config has changed since that hash was recorded.
* F9d's seed bands (development / outlier-final / absence-final) are
  pairwise disjoint and disjoint from every earlier F7/F8/F9/F9b/F9c seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from duckie_pomdp.evaluation.f9_protocol import sha256
from duckie_pomdp.evaluation.f9c_protocol import F9cProtocol, load_f9c_protocol

# Estimator-parameter sections that F9c owns. F9d must never define any of
# these itself -- it may only read them from the frozen F9c config.
_FORBIDDEN_ESTIMATOR_SECTIONS = frozenset(
    {
        "measurement_model",
        "covariance_calibration",
        "conditional_detection",
        "innovation_gate",
        "association",
        "ekf",
        "existence",
    }
)


@dataclass(frozen=True)
class F9dProtocol:
    config_path: Path
    parameters_frozen: bool
    f9c_config_path: Path
    f9c_config_sha256: str
    checkpoint_sha256: str
    development_seeds: tuple[int, ...]
    outlier_final_seeds: tuple[int, ...]
    absence_final_seeds: tuple[int, ...]
    forbidden_seeds: tuple[int, ...]
    minimum_outlier_frames: int
    minimum_outlier_events: int
    minimum_outlier_seeds: int
    insufficient_outlier_frames: int
    minimum_absence_runs_20: int
    minimum_absence_runs_40: int
    criteria: dict[str, float]
    scenarios: tuple[Any, ...]
    artifacts: dict[str, Path]

    @property
    def config_sha256(self) -> str:
        return sha256(self.config_path)


def load_f9d_protocol(
    path: str | Path,
    *,
    require_frozen: bool = False,
) -> F9dProtocol:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        data: dict[str, Any] = tomllib.load(stream)

    def relative(value: str) -> Path:
        return (config_path.parent / value).resolve()

    provenance = data["provenance"]
    split = data["split"]
    minima = data["minima"]
    protocol = F9dProtocol(
        config_path=config_path,
        parameters_frozen=bool(data.get("parameters_frozen", False)),
        f9c_config_path=relative(str(provenance["f9c_config"])),
        f9c_config_sha256=str(provenance["f9c_config_sha256"]),
        checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        development_seeds=tuple(int(value) for value in split["development_seeds"]),
        outlier_final_seeds=tuple(
            int(value) for value in split["outlier_final_seeds"]
        ),
        absence_final_seeds=tuple(
            int(value) for value in split["absence_final_seeds"]
        ),
        forbidden_seeds=tuple(int(value) for value in split["forbidden_seeds"]),
        minimum_outlier_frames=int(minima["minimum_outlier_frames"]),
        minimum_outlier_events=int(minima["minimum_outlier_events"]),
        minimum_outlier_seeds=int(minima["minimum_outlier_seeds"]),
        insufficient_outlier_frames=int(minima["insufficient_outlier_frames"]),
        minimum_absence_runs_20=int(minima["minimum_absence_runs_20"]),
        minimum_absence_runs_40=int(minima["minimum_absence_runs_40"]),
        criteria={name: float(value) for name, value in data["criteria"].items()},
        scenarios=tuple(data.get("scenario_matrix", [])),
        artifacts={
            name: relative(str(value)) for name, value in data["artifacts"].items()
        },
    )
    _validate(protocol, data, require_frozen=require_frozen)
    return protocol


def f9c_parameters(protocol: F9dProtocol, *, require_frozen: bool = True) -> F9cProtocol:
    """Read-only accessor for F9c's parameters.

    Loads the frozen F9c config fresh from disk on every call -- there is no
    cached copy anywhere in F9d, so there is nothing that can drift out of
    sync with the file on disk. F9d must never write to the object this
    returns; ``F9cProtocol`` is itself a frozen dataclass so any attempted
    mutation raises.
    """

    return load_f9c_protocol(protocol.f9c_config_path, require_frozen=require_frozen)


def outlier_support_satisfied(
    protocol: F9dProtocol,
    *,
    frames: int,
    events: int,
    seeds: int,
) -> bool:
    """Whether the gross-localization-outlier evidence is strong enough.

    All three conditions must hold together: enough frames, enough distinct
    outlier events, and enough seeds contributing them. Frames alone can be
    satisfied by two long bursts from a single seed, which is not the same
    evidence as scattered failures across the outlier-final band -- so this
    also fails outright whenever frames falls below the insufficient-evidence
    floor, regardless of how the other two conditions look.
    """

    if frames < protocol.insufficient_outlier_frames:
        return False
    return (
        frames >= protocol.minimum_outlier_frames
        and events >= protocol.minimum_outlier_events
        and seeds >= protocol.minimum_outlier_seeds
    )


def absence_support_satisfied(
    protocol: F9dProtocol,
    *,
    runs_ge_20: float,
    runs_ge_40: float,
) -> bool:
    """Whether one absence kind's evidence clears the pre-registered
    long-absence support minima.

    Both conditions must hold together: enough runs of at least 20
    consecutive absence frames, AND enough runs of at least 40. This is the
    one place ``minimum_absence_runs_20``/``minimum_absence_runs_40`` are
    checked -- callers (Task 4's probe script, and any later task reading
    the same artifact) must call this rather than re-implementing the
    threshold comparison locally, for the same reason
    ``outlier_support_satisfied`` exists as a single function: a local copy
    of a pre-registered minimum can silently drift from it.

    Deliberately called ONCE PER ABSENCE KIND (B1, B2, B3 each
    independently), never on a count that has already pooled kinds
    together -- B1 and B2 decay through mechanistically different paths
    (B1 through the pure ``P_S``/``P_birth`` prediction recurrence with no
    likelihood applied at all; B2 through the I8-floored likelihood
    update), so a single number that has already summed them together would
    hide which kind, if any, was actually carrying the evidence. Nothing
    about this function's signature enforces that -- it takes whatever
    counts it is given -- so the discipline lives in the caller; see
    ``experiments/probe_f9d_absence_yield.py``.
    """

    return (
        runs_ge_20 >= protocol.minimum_absence_runs_20
        and runs_ge_40 >= protocol.minimum_absence_runs_40
    )


def _validate(
    protocol: F9dProtocol,
    data: dict[str, Any],
    *,
    require_frozen: bool,
) -> None:
    if sha256(protocol.f9c_config_path) != protocol.f9c_config_sha256:
        raise ValueError(
            "F9d config's f9c_config_sha256 no longer matches the frozen F9c "
            "config on disk; F9d must refuse to run against a drifted "
            "frozen F9c estimator"
        )

    f9c_protocol = load_f9c_protocol(protocol.f9c_config_path)
    if f9c_protocol.checkpoint_sha256 != protocol.checkpoint_sha256:
        raise ValueError(
            "F9d checkpoint_sha256 does not match the frozen F9c checkpoint hash"
        )

    forbidden_sections = _FORBIDDEN_ESTIMATOR_SECTIONS & set(data)
    if forbidden_sections:
        raise ValueError(
            "F9d config must not define estimator sections owned by the "
            f"frozen F9c config: {sorted(forbidden_sections)}"
        )

    development = set(protocol.development_seeds)
    outlier = set(protocol.outlier_final_seeds)
    absence = set(protocol.absence_final_seeds)
    if not development or not outlier or not absence:
        raise ValueError(
            "F9d development/outlier-final/absence-final seed bands must "
            "all be nonempty"
        )
    if (development & outlier) or (development & absence) or (outlier & absence):
        raise ValueError(
            "F9d development/outlier-final/absence-final seed bands must be "
            "pairwise disjoint"
        )

    forbidden = set(protocol.forbidden_seeds)
    if (development | outlier | absence) & forbidden:
        raise ValueError(
            "F9d seeds overlap a forbidden earlier-split seed"
        )

    if require_frozen:
        if not protocol.parameters_frozen:
            raise ValueError("F9d parameters_frozen is not true")
        scenario_items = [
            *data.get("outlier_scenario_matrix", []),
            *data.get("absence_scenario_matrix", []),
        ]
        if not scenario_items or not all(
            bool(item.get("use_for_final", False)) for item in scenario_items
        ):
            raise ValueError("F9d final scenario matrix is not fully frozen")
        frozen_config_path = protocol.artifacts["frozen_config_json"]
        if not frozen_config_path.exists():
            raise ValueError("frozen F9d config artifact is missing")
        frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        if frozen_config.get("config_sha256") != protocol.config_sha256:
            raise ValueError("frozen F9d config artifact hash mismatch")
        # F9d's entire purpose depends on F9c staying frozen: re-loading with
        # require_frozen re-verifies F9c's own freeze artifact and hash, so a
        # silently un-frozen F9c estimator cannot be used for F9d evidence.
        load_f9c_protocol(protocol.f9c_config_path, require_frozen=True)
