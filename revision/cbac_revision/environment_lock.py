"""Export a complete exact-version lock for the active benchmark interpreter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .runner import write_complete_environment_lock


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    path = write_complete_environment_lock(parsed.output)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
