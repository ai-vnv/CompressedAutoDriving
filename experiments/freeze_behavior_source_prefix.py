"""Freeze an immutable CSV prefix used by a behavior warm-start artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckie_pomdp.control.f10_protocol import file_sha256


def freeze_prefix(source: Path, output: Path, *, data_rows: int) -> dict[str, object]:
    if data_rows <= 0:
        raise ValueError("data_rows must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite behavior source snapshot: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with source.open("rb") as source_stream, output.open("xb") as output_stream:
        header = source_stream.readline()
        if not header:
            raise ValueError("source CSV is empty")
        output_stream.write(header)
        for _ in range(data_rows):
            row = source_stream.readline()
            if not row:
                raise ValueError(
                    f"source CSV contains only {copied} data rows; expected {data_rows}"
                )
            output_stream.write(row)
            copied += 1

    return {
        "schema_version": 1,
        "source_csv": str(source.resolve()),
        "source_csv_current_sha256": file_sha256(source),
        "snapshot_csv": str(output.resolve()),
        "snapshot_csv_sha256": file_sha256(output),
        "data_rows": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-rows", type=int, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    result = freeze_prefix(
        args.source.resolve(), args.output.resolve(), data_rows=args.data_rows
    )
    if (
        args.expected_sha256 is not None
        and result["snapshot_csv_sha256"] != args.expected_sha256
    ):
        raise RuntimeError("frozen behavior source prefix hash mismatch")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
