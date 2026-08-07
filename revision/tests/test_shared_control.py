from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbac_revision.artifacts import canonical_sha256
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.shared_control import (
    attach_shared_control_bootstrap,
    build_shared_control_evidence,
    read_shared_control_evidence,
    write_shared_control_evidence,
)
from cbac_revision.statistics import AnalysisReleaseResult

DATASETS = ("adamson", "norman", "replogle_k562", "replogle_rpe1")


def _evidence(tmp_path: Path, dataset: str, hvg: int = 200) -> tuple[Path, dict]:
    manifest, matrix = build_shared_control_evidence(
        dataset=dataset,
        hvg=hvg,
        gene_names=[f"G{index}" for index in range(hvg)],
        control_row_ids=[f"{dataset}:{index}" for index in range(6)],
        standardized_control_profiles=np.asarray(
            [np.linspace(-0.2, 0.2, hvg) * (index - 2.5) for index in range(6)]
        ),
        dataset_sha256=canonical_sha256({"dataset": dataset}),
        preprocessing_state_sha256=canonical_sha256({"preprocessing": dataset}),
    )
    directory = tmp_path / f"{dataset}_{hvg}"
    path, _ = write_shared_control_evidence(directory, manifest, matrix)
    return path, manifest


def test_shared_control_evidence_round_trip_and_matrix_tamper(tmp_path: Path) -> None:
    path, manifest = _evidence(tmp_path, "adamson")
    observed, matrix = read_shared_control_evidence(path)
    assert observed == manifest
    assert matrix.shape == (6, 200)

    matrix[0, 0] += 1.0
    np.save(path.parent / manifest["profile_matrix_source_id"], matrix, allow_pickle=False)
    with pytest.raises(RevisionProtocolError, match="BINDING_INVALID"):
        read_shared_control_evidence(path)


def test_shared_control_bootstrap_is_measured_and_bound_to_fold_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = {}
    evidence_paths = []
    for dataset in DATASETS:
        path, manifest = _evidence(tmp_path, dataset)
        evidence_paths.append(path)
        manifests[dataset] = manifest
        for hvg in (500, 1000):
            extra_path, _ = _evidence(tmp_path, dataset, hvg)
            evidence_paths.append(extra_path)
    artifact_paths = []
    payloads = {}
    genes = [f"G{index}" for index in range(200)]
    truth = np.linspace(-1.0, 1.0, 200)
    for dataset_index, dataset in enumerate(DATASETS):
        for condition_index in range(50):
            condition = f"C{condition_index:02d}"
            for arm in ("string_go", "dense"):
                for seed in (42, 43, 44):
                    path = tmp_path / f"{dataset}_{condition}_{arm}_{seed}.json.gz"
                    artifact_paths.append(path)
                    offset = 0.03 if arm == "string_go" else 0.08
                    prediction = truth + offset * np.sin(np.arange(200) + dataset_index + seed)
                    payloads[path.resolve()] = {
                        "identity": {
                            "run_id": "run",
                            "dataset": dataset,
                            "hvg": 200,
                            "arm": arm,
                            "condition": condition,
                            "seed": seed,
                        },
                        "config": {"panel": "primary"},
                        "gene_names": genes,
                        "y_true": truth.tolist(),
                        "y_pred": prediction.tolist(),
                        "preprocessing_state_hash": manifests[dataset][
                            "preprocessing_state_sha256"
                        ],
                        "input_hashes": {
                            "dataset": manifests[dataset]["dataset_sha256"],
                            "shared_control_cell_evidence": manifests[dataset]["manifest_sha256"],
                        },
                    }
    monkeypatch.setattr(
        "cbac_revision.shared_control.read_fold_artifact",
        lambda path: payloads[path.resolve()],
    )
    primary = AnalysisReleaseResult(
        registry={"analysis_id": "PRIMARY", "status": "RELEASED"},
        detail_tables={
            "condition_contrasts": pd.DataFrame(
                [{"dataset": dataset, "condition": "C00", "delta": 0.1} for dataset in DATASETS]
            )
        },
        failures=pd.DataFrame(),
    )

    released = attach_shared_control_bootstrap(
        primary,
        artifact_paths=artifact_paths,
        evidence_manifest_paths=evidence_paths,
        n_bootstrap=100,
        random_seed=7,
    )
    audit = released.registry["shared_control_cell_bootstrap"]
    assert audit["status"] == "MEASURED"
    assert (
        audit["uncertainty_interval_95_low"]
        <= audit["estimate"]
        <= audit["uncertainty_interval_95_high"]
    )
    assert set(audit["evidence_manifest_hashes"]) == set(DATASETS)
    assert audit["neural_seed_support"] == [42, 43, 44]

    with pytest.raises(RevisionProtocolError, match="DATASET_HVG_DUPLICATE"):
        attach_shared_control_bootstrap(
            primary,
            artifact_paths=artifact_paths,
            evidence_manifest_paths=[*evidence_paths, evidence_paths[0]],
            n_bootstrap=100,
            random_seed=7,
        )

    forged = json.loads(evidence_paths[0].read_text(encoding="utf-8"))
    forged["selection_rule"] = "attacker_selected_low_variance_controls"
    forged["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "manifest_sha256"}
    )
    evidence_paths[0].write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(RevisionProtocolError):
        attach_shared_control_bootstrap(
            primary,
            artifact_paths=artifact_paths,
            evidence_manifest_paths=evidence_paths,
            n_bootstrap=100,
            random_seed=7,
        )
