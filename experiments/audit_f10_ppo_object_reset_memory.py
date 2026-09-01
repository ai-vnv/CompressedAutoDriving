"""Audit separated-object scenario resets and bounded resident-memory growth."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256
from duckie_pomdp.control.ppo_protocol import (
    load_ppo_curriculum_protocol,
    require_stage_in_protocol_scope,
)


ROOT = Path(__file__).resolve().parents[1]


def _rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def audit(
    config_path: Path,
    *,
    resets_per_stage: int,
    warmup_resets: int,
) -> dict:
    if resets_per_stage < 12:
        raise ValueError("object reset audit requires at least 12 resets per stage")
    if warmup_resets < 2 or warmup_resets >= resets_per_stage:
        raise ValueError("object reset audit requires a valid warmup")

    rss_samples: list[float] = []
    reused: list[bool] = []
    pedestrian_activation_transitions = 0
    isolation_checks: list[bool] = []
    stage_records: dict[str, dict] = {}
    protocol = load_ppo_curriculum_protocol(config_path)
    stages = ["c2", "c3"]
    try:
        require_stage_in_protocol_scope(protocol, "c4")
    except RuntimeError:
        pass
    else:
        stages.append("c4")
    for stage_key in stages:
        stage_rss_start = len(rss_samples)
        env = PPOCurriculumEnvironment(config_path, stage=stage_key, split="training")
        previous_integration = None
        previous_simulator = None
        previous_variant = None
        modes: list[str | None] = []
        try:
            seeds = env.protocol.stage(stage_key).training_seeds
            for index in range(resets_per_stage):
                _, info = env.reset(seed=seeds[index % len(seeds)])
                integration = env._integration
                simulator = integration.agent._session._simulator
                variant = (
                    bool(info.get("c2_rehearsal_no_pedestrian", False)),
                    stage_key,
                )
                if previous_integration is not None:
                    reused.append(
                        integration is previous_integration
                        and simulator is previous_simulator
                    )
                    if variant != previous_variant:
                        pedestrian_activation_transitions += 1
                previous_integration = integration
                previous_simulator = simulator
                previous_variant = variant

                privileged = integration.privileged.read()
                pedestrian_present = (
                    privileged.true_pomdp_state.pedestrian.exists
                    and privileged.pedestrian_world_position is not None
                )
                stop_present = privileged.stop_sign_world_position is not None
                expected = (
                    (
                        not pedestrian_present and not stop_present
                        if info.get("c2_rehearsal_no_pedestrian", False)
                        else pedestrian_present and not stop_present
                    )
                    if stage_key == "c2"
                    else (
                        stop_present and pedestrian_present
                        if stage_key == "c4"
                        else stop_present and not pedestrian_present
                    )
                )
                isolation_checks.append(expected)
                modes.append(info["pedestrian_mode"])
                rss_samples.append(_rss_mib())
        finally:
            env.close()
        stage_rss = rss_samples[stage_rss_start:]
        stage_steady = stage_rss[warmup_resets:]
        stage_records[stage_key] = {
            "seeds": list(seeds[:resets_per_stage]),
            "pedestrian_modes": modes,
            "isolation_pass": all(
                isolation_checks[-resets_per_stage:]
            ),
            "rss_mib": stage_rss,
            "steady_growth_mib": stage_steady[-1] - stage_steady[0],
            "steady_span_mib": max(stage_steady) - min(stage_steady),
        }

    # Each stage intentionally creates one distinct runtime and may incur a
    # one-time model/map allocation. The leak question is whether repeatedly
    # resetting that reused runtime grows without bound, so compare steady
    # samples within each stage instead of treating stage transitions as leaks.
    steady_growth = max(
        float(record["steady_growth_mib"]) for record in stage_records.values()
    )
    steady_span = max(
        float(record["steady_span_mib"]) for record in stage_records.values()
    )
    integration_reused = bool(reused) and all(reused)
    isolation_preserved = bool(isolation_checks) and all(isolation_checks)
    passed = (
        integration_reused
        and isolation_preserved
        and steady_growth <= 192.0
        and steady_span <= 256.0
    )
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_sha256": file_sha256(Path(__file__)),
        "stages": stages,
        "resets_per_stage": resets_per_stage,
        "warmup_resets": warmup_resets,
        "integration_reused_within_stage": integration_reused,
        "pedestrian_activation_transitions": pedestrian_activation_transitions,
        "stage_isolation_preserved": isolation_preserved,
        "stage_records": stage_records,
        "rss_mib": rss_samples,
        "steady_growth_mib": steady_growth,
        "steady_span_mib": steady_span,
        "maximum_growth_mib": 192.0,
        "maximum_span_mib": 256.0,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f10_ppo_visual_objects_v10.toml",
    )
    parser.add_argument("--resets-per-stage", type=int, default=12)
    parser.add_argument("--warmup-resets", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "f10_ppo_visual_objects_v10"
            / "object_reset_memory_audit.json"
        ),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(
        args.config.resolve(),
        resets_per_stage=args.resets_per_stage,
        warmup_resets=args.warmup_resets,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
