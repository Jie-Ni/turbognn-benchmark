"""Render the canonical hash-bound revision result asset package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from cbac_revision.release_assets import ASSET_GROUPS, render_release_assets


def _source_bindings(values: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source-binding requires NAME=SHA256")
        name, sha256 = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate source binding {name!r}")
        output[name] = sha256
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the release-asset renderer CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("author-review", "released"), required=True)
    parser.add_argument("--table-root", type=Path)
    parser.add_argument("--source-binding", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render placeholders or a fully validated released asset package."""

    parsed = build_argument_parser().parse_args(argv)
    tables: dict[str, pd.DataFrame] = {}
    if parsed.mode == "released":
        if parsed.table_root is None:
            raise ValueError("--table-root is required in released mode")
        tables = {group: pd.read_csv(parsed.table_root / f"{group}.csv") for group in ASSET_GROUPS}
    manifest = render_release_assets(
        parsed.output,
        mode=parsed.mode,
        tables=tables,
        source_bindings=_source_bindings(parsed.source_binding),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
