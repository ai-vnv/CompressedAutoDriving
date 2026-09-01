"""Audit native-map PPO reset reuse and bounded resident-memory growth."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from duckie_pomdp.control import PPOCurriculumEnvironment
from duckie_pomdp.control.f10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _rss_mib() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def audit(config_path: Path, *, resets: int, warmup_resets: int) -> dict:
    if resets < 36 or warmup_resets < 2 or warmup_resets >= resets:
        raise ValueError("memory audit requires >=36 resets and a valid warmup")
    env = PPOCurriculumEnvironment(config_path, stage="c0", split="training")
    rss_samples: list[float] = []
    integration_ids: set[int] = set()
    simulator_ids: set[int] = set()
    try:
        seeds = env.protocol.stage("c0").training_seeds
        for index in range(resets):
            env.reset(seed=seeds[index % len(seeds)])
            rss_samples.append(_rss_mib())
            integration_ids.add(id(env._integration))
            simulator_ids.add(id(env._integration.agent._session._simulator))
    finally:
        env.close()
    steady = rss_samples[warmup_resets:]
    steady_growth = steady[-1] - steady[0]
    steady_span = max(steady) - min(steady)
    passed = (
        len(integration_ids) == 1
        and len(simulator_ids) == 1
        and steady_growth <= 64.0
        and steady_span <= 96.0
    )
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_sha256": file_sha256(Path(__file__)),
        "resets": resets,
        "warmup_resets": warmup_resets,
        "unique_integration_count": len(integration_ids),
        "unique_simulator_count": len(simulator_ids),
        "rss_mib": rss_samples,
        "steady_growth_mib": steady_growth,
        "steady_span_mib": steady_span,
        "maximum_growth_mib": 64.0,
        "maximum_span_mib": 96.0,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "f10_ppo_visual_v2.toml",
    )
    parser.add_argument("--resets", type=int, default=36)
    parser.add_argument("--warmup-resets", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "f10_ppo_visual_v2" / "c0" / "reset_memory_audit.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.config.resolve(), resets=args.resets, warmup_resets=args.warmup_resets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
