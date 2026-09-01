"""Leakage checks for episode-level dataset splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


def assert_no_split_leakage(rows: Iterable[Mapping[str, object]]) -> None:
    episode_splits: dict[str, set[str]] = defaultdict(set)
    seed_splits: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown dataset split: {split}")
        episode_splits[str(row["episode_id"])].add(split)
        seed_splits[int(row["seed"])].add(split)
    leaking_episodes = {
        key: sorted(value) for key, value in episode_splits.items() if len(value) > 1
    }
    leaking_seeds = {
        key: sorted(value) for key, value in seed_splits.items() if len(value) > 1
    }
    if leaking_episodes or leaking_seeds:
        raise ValueError(
            f"split leakage detected: episodes={leaking_episodes}, seeds={leaking_seeds}"
        )
