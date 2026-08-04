#!/usr/bin/env python
"""Compatibility wrapper for the strict seed/fold merger."""

from merge_all_seeds import main, merge_directory, merge_sources, parse_split_file

__all__ = ["main", "merge_directory", "merge_sources", "parse_split_file"]


if __name__ == "__main__":
    main()
