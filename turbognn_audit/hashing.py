"""Canonical hashing helpers used by benchmark audit records."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize_json_nfc(value: Any) -> Any:
    """Recursively NFC-normalize every JSON string, including object keys."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalize_json_nfc(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json_nfc(item) for item in value]
    if isinstance(value, dict):
        normalized = {
            unicodedata.normalize("NFC", str(key)): normalize_json_nfc(item)
            for key, item in value.items()
        }
        if len(normalized) != len(value):
            raise ValueError("JSON object keys collide after NFC normalization")
        return normalized
    return value


def canonical_json_nfc(value: Any) -> str:
    """Serialize recursively NFC-normalized JSON under the TDS-02 byte contract."""
    return json.dumps(
        normalize_json_nfc(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def sha256_json_nfc(value: Any) -> str:
    """Hash the exact recursively NFC-normalized canonical JSON bytes."""
    return hashlib.sha256(canonical_json_nfc(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with a stable byte representation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON-compatible data."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_strings(values: list[str]) -> str:
    """Return an order-sensitive SHA-256 digest for a sequence of strings."""
    return sha256_json(values)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of the exact bytes in a filesystem artifact."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nfc_text(value: str) -> str:
    """Normalize an identifier to Unicode NFC before hashing or statistical ordering."""
    return unicodedata.normalize("NFC", str(value))


def utf8_sort(values: list[str]) -> list[str]:
    """Sort NFC identifiers by their UTF-8 byte representation."""
    normalized = [nfc_text(value) for value in values]
    return sorted(normalized, key=lambda value: value.encode("utf-8"))
