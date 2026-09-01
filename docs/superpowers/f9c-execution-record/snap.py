#!/usr/bin/env python3
"""No-git substitute for the SDD review-package script.

usage:
  snap.py save <label>          snapshot the source tree under snapshots/<label>
  snap.py diff <label> <out>    unified diff snapshots/<label> -> current tree
  snap.py restore <label>       roll the source tree back to snapshots/<label>

Exists because subagent-driven-development's review-package/sdd-workspace both
require git (`git rev-parse`), and this project is not a git repository. The
snapshot pair gives the task reviewer the same artifact (file list + unified
diff in one file) and additionally restores the per-task rollback point that
running without git would otherwise cost.
"""

from __future__ import annotations

import difflib
import os
import shutil
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
ROOT = WS.parents[2]
SNAPSHOTS = WS / "snapshots"
PATHS = (
    "src",
    "tests",
    "configs",
    "experiments",
    "scripts",
    "docs",
    "IMPLEMENTATION_NOTES.md",
    "GATES.md",
    "README.md",
)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {".py", ".toml", ".md", ".json", ".txt", ".cfg", ".ini", ".sh"}


def _copy(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in PATHS:
        source = ROOT / name
        if not source.exists():
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                destination / name,
                ignore=shutil.ignore_patterns(*SKIP_DIRS),
            )
        else:
            shutil.copy2(source, destination / name)


def _files(base: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in base.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            found[str(path.relative_to(base)).replace(os.sep, "/")] = path
    return found


def _read(path: Path) -> list[str] | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (UnicodeDecodeError, OSError):
        return None


def save(label: str) -> None:
    destination = SNAPSHOTS / label
    _copy(destination)
    print(destination)


def restore(label: str) -> None:
    source = SNAPSHOTS / label
    if not source.is_dir():
        raise SystemExit(f"no such snapshot: {label}")
    for name in PATHS:
        staged = source / name
        live = ROOT / name
        if not staged.exists():
            continue
        if live.is_dir():
            shutil.rmtree(live)
        elif live.exists():
            live.unlink()
        if staged.is_dir():
            shutil.copytree(staged, live, ignore=shutil.ignore_patterns(*SKIP_DIRS))
        else:
            shutil.copy2(staged, live)
    print(f"restored {label}")


def diff(label: str, out: str) -> None:
    base = SNAPSHOTS / label
    if not base.is_dir():
        raise SystemExit(f"no such snapshot: {label}")
    current = SNAPSHOTS / ".current"
    _copy(current)
    before, after = _files(base), _files(current)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    common = sorted(set(before) & set(after))
    changed = [
        name
        for name in common
        if before[name].read_bytes() != after[name].read_bytes()
    ]

    lines: list[str] = [f"# Review package: {label} -> current", ""]
    lines.append("## Files changed")
    for name in added:
        lines.append(f"  A  {name}")
    for name in changed:
        lines.append(f"  M  {name}")
    for name in removed:
        lines.append(f"  D  {name}")
    if not (added or changed or removed):
        lines.append("  (no changes)")
    lines += ["", "## Diff", ""]

    for name in added + changed:
        new = _read(after[name])
        old = _read(before[name]) if name in before else []
        if new is None:
            lines.append(f"--- binary or non-text file changed: {name}")
            continue
        lines.extend(
            difflib.unified_diff(
                old or [],
                new,
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
                n=10,
            )
        )
        lines.append("")
    for name in removed:
        lines.append(f"--- deleted: {name}")

    Path(out).write_text(
        "\n".join(line.rstrip("\n") for line in lines) + "\n", encoding="utf-8"
    )
    shutil.rmtree(current)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "restore":
        restore(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
