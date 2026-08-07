"""Single-command release gate for all mandatory revision analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import canonical_sha256, file_sha256
from .compute_release import release_measured_compute
from .external_validation import read_external_comparator_family
from .external_performance import (
    combine_external_validity_and_performance,
    release_external_performance_family,
)
from .global_trigger import read_global_trigger_manifest
from .protocol import load_protocol
from .release_assets import (
    build_claim_consequence_gate,
    render_release_assets,
    validate_release_asset_manifest,
)
from .release_asset_builder import assemble_release_asset_tables, assert_asset_semantics_match
from .release_trust import validate_main_release_trust_anchor
from .runner import code_tree_sha256
from .shared_control import attach_shared_control_bootstrap
from .statistics import (
    AnalysisReleaseResult,
    build_thirteen_result_lock_registry,
    conditional_nonoverlap_release,
    condition_ranking_overlap,
    descriptive_paired_tests,
    mandatory_fit_coverage_release,
    metric_frame_from_artifacts,
    mixed_support_sensitivity_release,
    primary_hierarchical_release,
    propagation_control_release,
    scale_interaction_release,
    secondary_metric_release,
    topology_hierarchical_release,
)


def postprocess_and_release(
    artifact_paths: Sequence[Path],
    protocol_path: Path,
    global_trigger_manifest_path: Path,
    output_dir: Path,
    preflight_summary_paths: Sequence[Path] = (),
    external_family_manifest_path: Path | None = None,
    external_family_manifest_sha256: str | None = None,
    external_performance_manifest_path: Path | None = None,
    external_performance_manifest_sha256: str | None = None,
    compute_registry_paths: Sequence[Path] = (),
    baseline_artifact_paths: Sequence[Path] = (),
    release_mode: str = "author-review",
    release_asset_output_dir: Path | None = None,
    release_asset_manifest_path: Path | None = None,
    release_asset_manifest_sha256: str | None = None,
    main_trust_anchor_path: Path | None = None,
    main_trust_anchor_sha256: str | None = None,
    evidence_bundle_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute fold metrics and apply all mandatory registry locks."""

    paths = tuple(sorted({Path(path).resolve() for path in artifact_paths}))
    if not paths:
        raise ValueError("At least one fold artifact is required")
    protocol = load_protocol(protocol_path)
    protocol_file_hash = file_sha256(protocol_path)
    expected_code_hash = code_tree_sha256()
    global_trigger_manifest = read_global_trigger_manifest(
        global_trigger_manifest_path, protocol, protocol_file_hash
    )
    preflight_source_paths = sorted({Path(path).resolve() for path in preflight_summary_paths})
    preflight_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in preflight_source_paths
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_frame_from_artifacts(paths)
    metrics.to_csv(output_dir / "fold_metrics_recomputed.csv", index=False)

    coverage = mandatory_fit_coverage_release(
        metrics,
        protocol,
        preflight_summaries,
        protocol_file_hash=protocol_file_hash,
        expected_code_hash=expected_code_hash,
    )
    primary = primary_hierarchical_release(metrics, protocol)
    scale = scale_interaction_release(metrics, protocol)
    topology = topology_hierarchical_release(metrics, protocol)
    propagation = propagation_control_release(metrics, protocol)
    mixed_support = mixed_support_sensitivity_release(metrics, protocol)
    if primary.released and coverage.released:
        shared_control_summary_paths = [
            path
            for path, summary in zip(preflight_source_paths, preflight_summaries, strict=True)
            if summary.get("hvg") == 200 and summary.get("analysis_block") == "topology_primary"
        ]
        primary = attach_shared_control_bootstrap(
            primary,
            artifact_paths=paths,
            evidence_manifest_paths=[
                path.parent / "shared_control_cell_evidence.json"
                for path in shared_control_summary_paths
            ],
            n_bootstrap=int(protocol["statistics"]["bootstrap_replicates"]),
            random_seed=int(protocol["statistics"]["bootstrap_random_seed"]),
        )
    if not coverage.released:
        primary = _cascade_coverage_withhold(primary)
        scale = _cascade_coverage_withhold(scale)
        topology = _cascade_coverage_withhold(topology)
        propagation = _cascade_coverage_withhold(propagation)
        mixed_support = _cascade_coverage_withhold(mixed_support)
    conditional = conditional_nonoverlap_release(
        metrics,
        protocol,
        global_trigger_manifest,
        preflight_summaries,
        protocol_file_hash=protocol_file_hash,
        expected_code_hash=expected_code_hash,
    )
    mandatory = (
        coverage,
        primary,
        scale,
        topology,
        propagation,
        mixed_support,
        conditional,
    )
    for result in mandatory:
        _write_analysis_result(output_dir, result)
    secondary_registry: dict[str, Any] = {}
    if primary.released:
        secondary_registry["paired_descriptive_tests"] = descriptive_paired_tests(
            primary.detail_tables["condition_contrasts"]
        )
        ranking_table = condition_ranking_overlap(metrics, protocol, top_k=10)
        secondary_registry["condition_ranking_overlap"] = ranking_table.to_dict(orient="records")
        secondary = secondary_metric_release(metrics, protocol)
        secondary.registry["verified_condition_ranking_stability"] = {
            "top_k": 10,
            "records": secondary_registry["condition_ranking_overlap"],
            "records_hash": canonical_sha256(secondary_registry["condition_ranking_overlap"]),
            "input_artifact_manifest_hash": canonical_sha256(
                sorted(metrics["artifact_hash"].astype(str))
            ),
        }
        secondary.detail_tables["condition_ranking_overlap"] = ranking_table
        if not coverage.released:
            secondary = _cascade_coverage_withhold(secondary)
    else:
        secondary = AnalysisReleaseResult(
            registry={
                "analysis_id": "SECONDARY-METRICS",
                "status": "WITHHELD",
                "reason_codes": ["PRIMARY_OR_COVERAGE_GATE_WITHHELD"],
            },
            detail_tables={},
            failures=pd.DataFrame(
                [
                    {
                        "analysis_id": "SECONDARY-METRICS",
                        "reason_code": "PRIMARY_OR_COVERAGE_GATE_WITHHELD",
                        "dataset": None,
                        "condition": None,
                        "hvg": None,
                        "arm": None,
                        "detail": "",
                    }
                ]
            ),
        )
    secondary_registry["benefit_oriented_metrics"] = secondary.registry
    _write_analysis_result(output_dir, secondary)
    if external_family_manifest_path is None or external_family_manifest_sha256 is None:
        external_registry = {
            "registry_id": "EXTERNAL-COMPARATOR-VALIDATION",
            "status": "WITHHELD",
            "failures": [
                {"reason_code": "EXTERNAL_FAMILY_MANIFEST_OR_PINNED_HASH_MISSING", "detail": ""}
            ],
        }
        external_registry["registry_hash"] = canonical_sha256(external_registry)
    else:
        external_validity_registry = read_external_comparator_family(
            external_family_manifest_path,
            expected_manifest_sha256=external_family_manifest_sha256,
        )
        if external_validity_registry["status"] == "RELEASED":
            external_performance_registry = release_external_performance_family(
                external_validity_registry,
                manifest_path=external_performance_manifest_path,
                expected_manifest_sha256=external_performance_manifest_sha256,
                bootstrap_replicates=int(protocol["statistics"]["bootstrap_replicates"]),
                bootstrap_random_seed=int(protocol["statistics"]["bootstrap_random_seed"]),
            )
            external_registry = combine_external_validity_and_performance(
                external_validity_registry, external_performance_registry
            )
        else:
            external_registry = external_validity_registry
    measured_compute_registry = release_measured_compute(compute_registry_paths, metrics)
    asset_output = release_asset_output_dir or (output_dir / "generated")
    if release_mode == "author-review":
        release_asset_manifest = render_release_assets(
            asset_output,
            mode="author-review",
        )
        release_asset_manifest_path = asset_output / "release_asset_manifest.json"
        release_asset_manifest_sha256 = file_sha256(release_asset_manifest_path)
        main_trust_registry: dict[str, Any] = {
            "schema_version": "1.0",
            "registry_id": "MAIN-RELEASE-TRUST-ANCHOR",
            "status": "WITHHELD",
            "reason_codes": ["AUTHOR_REVIEW_MODE_HAS_NO_PLACEHOLDER_TRUST_ANCHOR"],
            "cryptographic_boundary": ("caller_pinned_anchor_sha256_required_for_released_mode"),
        }
        main_trust_registry["registry_hash"] = canonical_sha256(main_trust_registry)
    elif release_mode in {"asset-candidate", "released"}:
        derived_tables, derived_bindings = assemble_release_asset_tables(
            protocol=protocol,
            protocol_file_sha256=protocol_file_hash,
            metrics=metrics,
            primary=primary,
            scale=scale,
            topology=topology,
            propagation=propagation,
            conditional=conditional,
            secondary=secondary,
            mixed_support=mixed_support,
            preflight_sources=list(zip(preflight_source_paths, preflight_summaries, strict=True)),
            global_trigger_manifest=global_trigger_manifest,
            baseline_artifact_paths=baseline_artifact_paths,
            measured_compute_registry=measured_compute_registry,
            external_registry=external_registry,
        )
        if release_mode == "asset-candidate":
            release_asset_manifest = render_release_assets(
                asset_output,
                mode="released",
                tables=derived_tables,
                source_bindings=derived_bindings,
            )
            release_asset_manifest_path = asset_output / "release_asset_manifest.json"
            release_asset_manifest_sha256 = file_sha256(release_asset_manifest_path)
            assert_asset_semantics_match(
                derived_tables,
                release_asset_manifest["semantic_records_hashes"],
            )
            main_trust_registry = {
                "schema_version": "1.0",
                "registry_id": "MAIN-RELEASE-TRUST-ANCHOR",
                "status": "WITHHELD",
                "reason_codes": ["ASSET_CANDIDATE_REQUIRES_CALLER_PINNED_DETACHED_MAIN_ANCHOR"],
                "cryptographic_boundary": (
                    "candidate_assets_must_be_added_to_detached_anchor_and_caller_pinned"
                ),
            }
            main_trust_registry["registry_hash"] = canonical_sha256(main_trust_registry)
        else:
            if (
                release_asset_manifest_path is None
                or release_asset_manifest_sha256 is None
                or main_trust_anchor_path is None
                or main_trust_anchor_sha256 is None
                or evidence_bundle_root is None
            ):
                raise ValueError(
                    "Released mode requires pinned release assets, detached main anchor, and "
                    "evidence bundle root"
                )
            release_asset_manifest = validate_release_asset_manifest(
                release_asset_manifest_path,
                expected_manifest_file_sha256=release_asset_manifest_sha256,
                expected_source_bindings=derived_bindings,
            )
            if release_asset_manifest.get("mode") != "released":
                raise ValueError("Released postprocessing requires a RELEASED_ASSETS manifest")
            assert_asset_semantics_match(
                derived_tables,
                release_asset_manifest["semantic_records_hashes"],
            )
            main_trust_registry = validate_main_release_trust_anchor(
                main_trust_anchor_path,
                expected_anchor_file_sha256=main_trust_anchor_sha256,
                evidence_bundle_root=evidence_bundle_root,
            )
    else:
        raise ValueError("release_mode must be author-review, asset-candidate, or released")
    claim_consequence_registry = build_claim_consequence_gate(
        release_asset_manifest_path,
        expected_manifest_file_sha256=str(release_asset_manifest_sha256),
    )
    decision_registry = build_thirteen_result_lock_registry(
        primary=primary,
        propagation=propagation,
        scale=scale,
        topology=topology,
        coverage=coverage,
        conditional=conditional,
        secondary=secondary,
        mixed_support=mixed_support,
        global_trigger_manifest=global_trigger_manifest,
        external_comparator_registry=external_registry,
        measured_compute_registry=measured_compute_registry,
        claim_consequence_registry=claim_consequence_registry,
    )
    all_headline_locks_released = not decision_registry["withheld_result_locks"]
    decision_registry["package_release_status"] = (
        "RELEASED"
        if all_headline_locks_released
        and main_trust_registry["status"] == "RELEASED"
        and release_asset_manifest["mode"] == "released"
        else "WITHHELD"
    )
    failures = (
        pd.concat(
            [result.failures for result in mandatory if len(result.failures)], ignore_index=True
        )
        if any(len(result.failures) for result in mandatory)
        else pd.DataFrame(
            columns=["analysis_id", "reason_code", "dataset", "condition", "hvg", "arm", "detail"]
        )
    )
    failures.to_csv(output_dir / "failure_ledger.csv", index=False)
    decision_registry.pop("decision_registry_hash", None)
    decision_registry["secondary_registry"] = secondary_registry
    decision_registry["artifact_count"] = len(paths)
    decision_registry["artifact_content_manifest_hash"] = canonical_sha256(
        sorted(metrics["artifact_hash"].astype(str))
    )
    decision_registry["protocol_source_id"] = protocol_path.name
    decision_registry["protocol_file_sha256"] = protocol_file_hash
    decision_registry["release_code_tree_sha256"] = expected_code_hash
    decision_registry["global_trigger_manifest_hash"] = global_trigger_manifest["manifest_hash"]
    decision_registry["preflight_summary_manifest_hash"] = canonical_sha256(
        sorted(summary.get("summary_hash") for summary in preflight_summaries)
    )
    decision_registry["release_mode"] = release_mode
    decision_registry["release_asset_manifest_source_id"] = str(release_asset_manifest_path.name)
    decision_registry["release_asset_manifest_file_sha256"] = str(release_asset_manifest_sha256)
    decision_registry["release_asset_manifest_self_hash"] = release_asset_manifest[
        "manifest_sha256"
    ]
    decision_registry["release_asset_manifest_mode"] = release_asset_manifest["mode"]
    decision_registry["main_release_trust_gate"] = main_trust_registry
    decision_registry["main_release_trust_anchor_file_sha256"] = (
        main_trust_registry.get("anchor_file_sha256")
        if main_trust_registry["status"] == "RELEASED"
        else None
    )
    decision_registry["cryptographic_boundary"] = (
        "release_is_valid_only_while_the_caller_pinned_detached_anchor_sha256_is_"
        "immutable_to_the_evidence_bundle_attacker"
    )
    decision_registry["release_output_manifest"] = [
        {
            "source_id": path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "decision_release_registry.json"
    ]
    decision_registry["release_output_manifest_hash"] = canonical_sha256(
        decision_registry["release_output_manifest"]
    )
    decision_registry["decision_registry_hash"] = canonical_sha256(decision_registry)
    decision_registry["release_package_hash"] = canonical_sha256(decision_registry)
    _write_json(output_dir / "decision_release_registry.json", decision_registry)
    return decision_registry


def _write_analysis_result(output_dir: Path, result: AnalysisReleaseResult) -> None:
    stem = result.registry["analysis_id"].casefold().replace("-", "_")
    table_manifest = []
    for name, table in result.detail_tables.items():
        path = output_dir / f"{stem}_{name}.csv"
        temporary = path.with_name(f".{path.name}.tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)
        table_manifest.append(
            {
                "table_id": name,
                "source_id": path.name,
                "sha256": file_sha256(path),
                "canonical_records_hash": canonical_sha256(
                    table.fillna("NOT_APPLICABLE").to_dict(orient="records")
                ),
                "rows": len(table),
                "columns": list(table.columns),
            }
        )
    result.registry["detail_table_manifest"] = table_manifest
    result.registry["detail_table_manifest_hash"] = canonical_sha256(table_manifest)
    result.registry.pop("analysis_registry_hash", None)
    result.registry["analysis_registry_hash"] = canonical_sha256(result.registry)
    _write_json(output_dir / f"{stem}_registry.json", result.registry)


def _cascade_coverage_withhold(result: AnalysisReleaseResult) -> AnalysisReleaseResult:
    """Remove numerical payloads when the exact-fit/preflight coverage gate failed."""

    failure = pd.DataFrame(
        [
            {
                "analysis_id": result.registry["analysis_id"],
                "reason_code": "MANDATORY_FIT_COVERAGE_WITHHELD",
                "dataset": None,
                "condition": None,
                "hvg": None,
                "arm": None,
                "detail": "No numerical registry is releasable before coverage passes",
            }
        ]
    )
    return AnalysisReleaseResult(
        registry={
            "analysis_id": result.registry["analysis_id"],
            "status": "WITHHELD",
            "reason_codes": ["MANDATORY_FIT_COVERAGE_WITHHELD"],
        },
        detail_tables={},
        failures=pd.concat([result.failures, failure], ignore_index=True),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )


def _manifest_artifacts(path: Path) -> list[Path]:
    frame = pd.read_csv(path)
    source_column = "source_id" if "source_id" in frame else "artifact_path"
    if source_column not in frame:
        raise ValueError(f"Artifact manifest {path} requires a source_id column")
    output = []
    for row in frame.itertuples(index=False):
        value = str(getattr(row, source_column))
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if "file_sha256" in frame and file_sha256(candidate) != str(row.file_sha256):
            raise ValueError(f"Artifact manifest file hash mismatch: {candidate}")
        output.append(candidate)
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-trigger-manifest", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, action="append", required=True)
    parser.add_argument("--external-family-manifest", type=Path, required=True)
    parser.add_argument("--external-family-manifest-sha256", required=True)
    parser.add_argument("--external-performance-manifest", type=Path)
    parser.add_argument("--external-performance-manifest-sha256")
    parser.add_argument("--compute-registry", type=Path, action="append", required=True)
    parser.add_argument("--baseline-artifact", type=Path, action="append", default=[])
    parser.add_argument("--baseline-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--release-mode",
        choices=("author-review", "asset-candidate", "released"),
        default="author-review",
    )
    parser.add_argument("--release-asset-output", type=Path)
    parser.add_argument("--release-asset-manifest", type=Path)
    parser.add_argument("--release-asset-manifest-sha256")
    parser.add_argument("--main-trust-anchor", type=Path)
    parser.add_argument("--main-trust-anchor-sha256")
    parser.add_argument("--evidence-bundle-root", type=Path)
    parser.add_argument("--artifact-manifest", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    artifacts: list[Path] = []
    for manifest in parsed.artifact_manifest:
        artifacts.extend(_manifest_artifacts(manifest))
    for root in parsed.artifact_root:
        artifacts.extend(root.rglob("*.json.gz"))
    baseline_artifacts = list(parsed.baseline_artifact)
    for root in parsed.baseline_root:
        baseline_artifacts.extend(root.rglob("*.json.gz"))
    registry = postprocess_and_release(
        artifact_paths=artifacts,
        protocol_path=parsed.protocol,
        global_trigger_manifest_path=parsed.global_trigger_manifest,
        output_dir=parsed.output,
        preflight_summary_paths=parsed.preflight_summary,
        external_family_manifest_path=parsed.external_family_manifest,
        external_family_manifest_sha256=parsed.external_family_manifest_sha256,
        external_performance_manifest_path=parsed.external_performance_manifest,
        external_performance_manifest_sha256=parsed.external_performance_manifest_sha256,
        compute_registry_paths=parsed.compute_registry,
        baseline_artifact_paths=baseline_artifacts,
        release_mode=parsed.release_mode,
        release_asset_output_dir=parsed.release_asset_output,
        release_asset_manifest_path=parsed.release_asset_manifest,
        release_asset_manifest_sha256=parsed.release_asset_manifest_sha256,
        main_trust_anchor_path=parsed.main_trust_anchor,
        main_trust_anchor_sha256=parsed.main_trust_anchor_sha256,
        evidence_bundle_root=parsed.evidence_bundle_root,
    )
    print(json.dumps(registry, indent=2, sort_keys=True))
    if parsed.release_mode == "asset-candidate":
        return 0 if registry.get("release_asset_manifest_mode") == "released" else 2
    return 0 if registry["package_release_status"] == "RELEASED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
