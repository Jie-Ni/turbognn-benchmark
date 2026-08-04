"""Crash-resistant, no-clobber persistence helpers for audit artifacts."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any, *, overwrite: bool = False) -> None:
    """Atomically publish JSON and reject an existing destination by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
