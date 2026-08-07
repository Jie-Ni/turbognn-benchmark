"""Validate the frozen protocol without running a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import load_protocol


def main() -> None:
    """Run the protocol validator CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path, help="Path to protocol.yaml")
    arguments = parser.parse_args()
    loaded = load_protocol(arguments.protocol)
    print(
        f"Protocol {arguments.protocol} is valid: "
        f"{len(loaded['datasets'])} datasets, schema {loaded['schema_version']}."
    )


if __name__ == "__main__":
    main()
