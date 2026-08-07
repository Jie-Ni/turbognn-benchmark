"""Validate the detached caller-pinned trust anchor for the main release chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cbac_revision.release_trust import validate_main_release_trust_anchor


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the main release trust-boundary CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--evidence-bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every anchored source and optionally persist the trust registry."""

    parsed = build_argument_parser().parse_args(argv)
    registry = validate_main_release_trust_anchor(
        parsed.anchor,
        expected_anchor_file_sha256=parsed.anchor_sha256,
        evidence_bundle_root=parsed.evidence_bundle_root,
    )
    encoded = json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
