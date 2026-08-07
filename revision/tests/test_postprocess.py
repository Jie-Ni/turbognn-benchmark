from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cbac_revision.artifacts import canonical_sha256, file_sha256, write_fold_artifact
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.global_trigger import aggregate_global_trigger
from cbac_revision.postprocess import main, postprocess_and_release
from cbac_revision.protocol import RESULT_LOCK_IDS, load_protocol
from cbac_revision.statistics import AnalysisReleaseResult
from test_artifacts import _artifact, _write_bound_checkpoint
from test_global_trigger import _summaries
from test_release_assets import _released_analysis_inputs


def test_incomplete_synthetic_package_withholds_cleanly_and_self_hashes(
    tmp_path: Path,
) -> None:
    protocol_path = Path(__file__).parents[1] / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    protocol_hash = file_sha256(protocol_path)
    checkpoint = tmp_path / "checkpoint.pt"
    _write_bound_checkpoint(checkpoint, _artifact())
    artifact_path = write_fold_artifact(
        tmp_path / "one_fold.json.gz",
        _artifact("checkpoint.pt", file_sha256(checkpoint)),
    )
    trigger = aggregate_global_trigger(
        _summaries(protocol_hash, triggered_dataset=None), protocol, protocol_hash
    )
    trigger_path = tmp_path / "global_trigger.json"
    trigger_path.write_text(json.dumps(trigger), encoding="utf-8")
    output = tmp_path / "release"

    registry = postprocess_and_release(
        [artifact_path],
        protocol_path,
        trigger_path,
        output,
    )

    assert registry["package_release_status"] == "WITHHELD"
    assert len(registry["result_locks"]) == 13
    assert (output / "decision_release_registry.json").is_file()
    assert registry["release_output_manifest_hash"]
    assert registry["release_package_hash"]
    saved = json.loads((output / "decision_release_registry.json").read_text(encoding="utf-8"))
    release_hash = saved.pop("release_package_hash")
    assert release_hash == canonical_sha256(saved)
    decision_hash = saved.pop("decision_registry_hash")
    assert decision_hash == canonical_sha256(saved)


def test_asset_candidate_then_caller_pinned_released_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _released_analysis_inputs(tmp_path, monkeypatch)
    coverage = AnalysisReleaseResult(
        registry={"analysis_id": "MANDATORY-FIT-COVERAGE", "status": "RELEASED"},
        detail_tables={},
        failures=pd.DataFrame(),
    )
    monkeypatch.setattr("cbac_revision.postprocess.load_protocol", lambda _: inputs["protocol"])
    monkeypatch.setattr(
        "cbac_revision.postprocess.read_global_trigger_manifest",
        lambda *_: inputs["global_trigger_manifest"],
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.metric_frame_from_artifacts", lambda _: inputs["metrics"]
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.mandatory_fit_coverage_release",
        lambda *_args, **_kwargs: coverage,
    )
    for name, key in (
        ("primary_hierarchical_release", "primary"),
        ("scale_interaction_release", "scale"),
        ("topology_hierarchical_release", "topology"),
        ("propagation_control_release", "propagation"),
        ("mixed_support_sensitivity_release", "mixed_support"),
    ):
        monkeypatch.setattr(
            f"cbac_revision.postprocess.{name}",
            lambda *_args, _key=key, **_kwargs: inputs[_key],
        )
    monkeypatch.setattr(
        "cbac_revision.postprocess.conditional_nonoverlap_release",
        lambda *_args, **_kwargs: inputs["conditional"],
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.secondary_metric_release",
        lambda *_args, **_kwargs: inputs["secondary"],
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.condition_ranking_overlap",
        lambda *_args, **_kwargs: inputs["secondary"].detail_tables["condition_ranking_overlap"],
    )
    monkeypatch.setattr("cbac_revision.postprocess.descriptive_paired_tests", lambda *_: {})
    monkeypatch.setattr(
        "cbac_revision.postprocess.attach_shared_control_bootstrap",
        lambda primary, **_: primary,
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.read_external_comparator_family",
        lambda *_args, **_kwargs: inputs["external_registry"],
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.release_external_performance_family",
        lambda *_args, **_kwargs: {"status": "RELEASED", "adapter_results": {}},
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.combine_external_validity_and_performance",
        lambda validity, _performance: validity,
    )
    monkeypatch.setattr(
        "cbac_revision.postprocess.release_measured_compute",
        lambda *_args, **_kwargs: inputs["measured_compute_registry"],
    )
    monkeypatch.setattr("cbac_revision.postprocess.code_tree_sha256", lambda: "9" * 64)

    def fake_locks(**_: object) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "registry_id": "DECISION-RELEASE-REGISTRY",
            "result_lock_ids": list(RESULT_LOCK_IDS),
            "result_locks": {
                lock_id: {"result_lock_id": lock_id, "status": "RELEASED"}
                for lock_id in RESULT_LOCK_IDS
            },
            "result_lock_hashes": {
                lock_id: canonical_sha256({"result_lock_id": lock_id, "status": "RELEASED"})
                for lock_id in RESULT_LOCK_IDS
            },
            "withheld_result_locks": [],
        }

    monkeypatch.setattr("cbac_revision.postprocess.build_thirteen_result_lock_registry", fake_locks)
    dummy_artifact = tmp_path / "runs" / "artifacts" / "fold.json.gz"
    dummy_artifact.parent.mkdir(parents=True)
    dummy_artifact.write_bytes(b"synthetic path; metric loader is fixed above")
    preflight_paths = [path for path, _ in inputs["preflight_sources"]]
    trigger_path = tmp_path / "runs" / "global_trigger.json"
    trigger_path.write_text("{}", encoding="utf-8")
    external_path = tmp_path / "runs" / "external_family.json"
    external_path.write_text("{}", encoding="utf-8")

    candidate_output = tmp_path / "runs" / "candidate"
    candidate_assets = candidate_output / "generated"
    candidate = postprocess_and_release(
        artifact_paths=[dummy_artifact],
        protocol_path=Path(__file__).parents[1] / "protocol.yaml",
        global_trigger_manifest_path=trigger_path,
        output_dir=candidate_output,
        preflight_summary_paths=preflight_paths,
        external_family_manifest_path=external_path,
        external_family_manifest_sha256="8" * 64,
        baseline_artifact_paths=inputs["baseline_artifact_paths"],
        release_mode="asset-candidate",
        release_asset_output_dir=candidate_assets,
    )
    manifest_path = candidate_assets / "release_asset_manifest.json"
    pinned_manifest_hash = file_sha256(manifest_path)
    assert candidate["package_release_status"] == "WITHHELD"
    assert candidate["release_asset_manifest_mode"] == "released"
    assert manifest_path.is_file()

    bundle_root = tmp_path / "runs"
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    anchor_path = trusted_root / "main_release.anchor.json"
    anchor_path.write_text(
        json.dumps({"candidate_manifest": pinned_manifest_hash}), encoding="utf-8"
    )
    pinned_anchor_hash = file_sha256(anchor_path)
    trust_calls: list[tuple[Path, str, Path]] = []

    def fake_trust(
        path: Path, *, expected_anchor_file_sha256: str, evidence_bundle_root: Path
    ) -> dict[str, object]:
        assert file_sha256(path) == expected_anchor_file_sha256
        assert not path.resolve().is_relative_to(evidence_bundle_root.resolve())
        trust_calls.append((path, expected_anchor_file_sha256, evidence_bundle_root))
        registry: dict[str, object] = {
            "schema_version": "1.0",
            "registry_id": "MAIN-RELEASE-TRUST-ANCHOR",
            "status": "RELEASED",
            "anchor_file_sha256": expected_anchor_file_sha256,
        }
        registry["registry_hash"] = canonical_sha256(registry)
        return registry

    monkeypatch.setattr("cbac_revision.postprocess.validate_main_release_trust_anchor", fake_trust)
    released_output = bundle_root / "released"
    released = postprocess_and_release(
        artifact_paths=[dummy_artifact],
        protocol_path=Path(__file__).parents[1] / "protocol.yaml",
        global_trigger_manifest_path=trigger_path,
        output_dir=released_output,
        preflight_summary_paths=preflight_paths,
        external_family_manifest_path=external_path,
        external_family_manifest_sha256="8" * 64,
        baseline_artifact_paths=inputs["baseline_artifact_paths"],
        release_mode="released",
        release_asset_manifest_path=manifest_path,
        release_asset_manifest_sha256=pinned_manifest_hash,
        main_trust_anchor_path=anchor_path,
        main_trust_anchor_sha256=pinned_anchor_hash,
        evidence_bundle_root=bundle_root,
    )
    assert released["package_release_status"] == "RELEASED"
    assert released["release_asset_manifest_file_sha256"] == pinned_manifest_hash
    assert released["main_release_trust_anchor_file_sha256"] == pinned_anchor_hash
    assert len(trust_calls) == 1

    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RevisionProtocolError, match="CALLER_PINNED_MANIFEST_MISMATCH"):
        postprocess_and_release(
            artifact_paths=[dummy_artifact],
            protocol_path=Path(__file__).parents[1] / "protocol.yaml",
            global_trigger_manifest_path=trigger_path,
            output_dir=bundle_root / "tampered",
            preflight_summary_paths=preflight_paths,
            external_family_manifest_path=external_path,
            external_family_manifest_sha256="8" * 64,
            baseline_artifact_paths=inputs["baseline_artifact_paths"],
            release_mode="released",
            release_asset_manifest_path=manifest_path,
            release_asset_manifest_sha256=pinned_manifest_hash,
            main_trust_anchor_path=anchor_path,
            main_trust_anchor_sha256=pinned_anchor_hash,
            evidence_bundle_root=bundle_root,
        )


def test_asset_candidate_cli_returns_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cbac_revision.postprocess.postprocess_and_release",
        lambda **_: {
            "package_release_status": "WITHHELD",
            "release_asset_manifest_mode": "released",
        },
    )
    result = main(
        [
            "--protocol",
            str(tmp_path / "protocol.yaml"),
            "--output",
            str(tmp_path / "output"),
            "--global-trigger-manifest",
            str(tmp_path / "trigger.json"),
            "--preflight-summary",
            str(tmp_path / "preflight.json"),
            "--external-family-manifest",
            str(tmp_path / "external.json"),
            "--external-family-manifest-sha256",
            "a" * 64,
            "--compute-registry",
            str(tmp_path / "compute.json"),
            "--release-mode",
            "asset-candidate",
        ]
    )
    assert result == 0
