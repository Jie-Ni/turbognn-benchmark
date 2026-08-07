"""Machine-readable frozen-design provenance tables."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

from .artifacts import canonical_sha256
from .errors import RevisionProtocolError


def hyperparameter_provenance_table(protocol: Mapping[str, Any]) -> pd.DataFrame:
    """Return the exact ordered hyperparameter ledger validated by the protocol loader."""

    provenance = protocol.get("hyperparameter_provenance")
    if not isinstance(provenance, dict):
        raise RevisionProtocolError("hyperparameter_provenance is missing")
    columns = provenance.get("required_fields")
    records = provenance.get("records")
    if not isinstance(columns, list) or not isinstance(records, list):
        raise RevisionProtocolError("hyperparameter_provenance schema is invalid")
    frame = pd.DataFrame(records, columns=columns)
    if list(frame.columns) != columns or frame.isna().any().any():
        raise RevisionProtocolError("Hyperparameter provenance table is incomplete")
    return frame


def hyperparameter_provenance_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Hash-bind the fixed-protocol estimand to its complete provenance table."""

    frame = hyperparameter_provenance_table(protocol)
    records = [
        {
            str(key): value.isoformat() if isinstance(value, (date, datetime)) else value
            for key, value in record.items()
        }
        for record in frame.to_dict(orient="records")
    ]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "estimand": "fixed_frozen_protocol_not_per_arm_optimum",
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "records_sha256": canonical_sha256(records),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload
