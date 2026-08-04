from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from turbognn_audit.frozen_rules import (
    G2_CONTRAST_ORDER,
    RANKING_GRAPHS,
    FrozenRuleError,
    PairEvidence,
    ResolutionEstimate,
    analyze_ranking_identity,
    audit_output_validity,
    directional_claim_gate,
    evaluate_external_family,
    evaluate_g2_family,
    evaluate_official_sanity,
    pair_specific_resolution,
    ranking_identity_gate,
)


def _ranking_frame() -> tuple[pd.DataFrame, dict[tuple[str, int], list[str]]]:
    conditions = [f"c{index:02d}" for index in range(12)]
    rows = []
    for graph, condition, seed in product(RANKING_GRAPHS, conditions, (42, 43, 44)):
        rows.append(
            {
                "dataset": "toy",
                "scale": 200,
                "condition": condition,
                "graph_type": graph,
                "seed": seed,
                "pearson_r": 1.0 - int(condition[1:]) / 20.0,
            }
        )
    return pd.DataFrame(rows), {("toy", 200): conditions}


def _resolution(value: float = 0.01) -> ResolutionEstimate:
    return ResolutionEstimate(
        point=value,
        upper_95=value,
        bootstrap_distribution=(value, value),
        required_seeds=(42, 43, 44),
        bootstrap_draw_n=2,
    )


def _evidence(contrasts: tuple[str, ...]) -> dict[str, PairEvidence]:
    return {
        contrast: PairEvidence(
            delta_r_star=0.03,
            ci_lower=0.01,
            ci_upper=0.05,
            resolution=_resolution(),
        )
        for contrast in contrasts
    }


def _family_frame(
    contrasts: tuple[str, ...], conditions: tuple[str, ...] = ("a", "b", "c")
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[str, ...]]]:
    rows = [
        {
            "contrast": contrast,
            "dataset": "toy",
            "condition": condition,
            "scale": scale,
            "seed": seed,
            "delta_z": 0.0,
        }
        for contrast, condition, scale, seed in product(
            contrasts, conditions, (200, 500, 1000), (42, 43, 44)
        )
    ]
    universes = {(contrast, "toy"): conditions for contrast in contrasts}
    return pd.DataFrame(rows), universes


def test_ranking_identity_uses_descending_scores_ties_and_frozen_graph_family() -> None:
    frame, universes = _ranking_frame()
    frame.loc[frame["condition"].isin(["c00", "c01"]), "pearson_r"] = 0.9
    result = analyze_ranking_identity(
        frame,
        condition_universes=universes,
        datasets=("toy",),
        scales=(200,),
        minimum_conditions=10,
        top_k=10,
        draws=5,
    )
    aggregated = result.top_conditions.loc[
        (result.top_conditions["graph_type"] == "string_ppi")
        & (result.top_conditions["ranking"] == "seed_aggregated")
    ]
    assert aggregated.iloc[0]["condition"] == "c00"
    assert aggregated.iloc[1]["condition"] == "c01"
    assert len(result.graph_pair_overlaps) == 6
    assert result.gate.passed


def test_ranking_gate_boundaries_are_inclusive() -> None:
    gate = ranking_identity_gate(0.80, [0.70], 0.80, [0.70])
    assert gate.passed
    assert not ranking_identity_gate(0.80, [0.699], 0.80, [0.70]).passed
    assert not ranking_identity_gate(0.80, [0.70], 0.799, [0.70]).passed


def test_ranking_permutation_is_deterministic_and_input_order_invariant() -> None:
    frame, universes = _ranking_frame()
    first = analyze_ranking_identity(
        frame,
        condition_universes=universes,
        datasets=("toy",),
        scales=(200,),
        minimum_conditions=10,
        top_k=10,
        draws=11,
    )
    second = analyze_ranking_identity(
        frame.sample(frac=1.0, random_state=17),
        condition_universes=universes,
        datasets=("toy",),
        scales=(200,),
        minimum_conditions=10,
        top_k=10,
        draws=11,
    )
    assert first.permutation_seed == 20_260_814
    assert first.permutation_benchmark_medians == second.permutation_benchmark_medians
    pd.testing.assert_frame_equal(first.graph_pair_overlaps, second.graph_pair_overlaps)


def test_ranking_permutation_reference_counts_inclusive_ties() -> None:
    frame, universes = _ranking_frame()
    frame["pearson_r"] = 0.2
    result = analyze_ranking_identity(
        frame,
        condition_universes=universes,
        datasets=("toy",),
        scales=(200,),
        minimum_conditions=10,
        top_k=10,
        draws=7,
    )
    assert result.permutation_benchmark_medians == (1.0,) * 7
    assert result.permutation_upper_tail_p == 1.0


def test_ranking_rejects_nfc_duplicates_and_incomplete_family() -> None:
    frame, universes = _ranking_frame()
    duplicated_universe = dict(universes)
    duplicated_universe[("toy", 200)] = [
        "e\N{COMBINING ACUTE ACCENT}",
        "é",
        *universes[("toy", 200)],
    ]
    with pytest.raises(FrozenRuleError, match="NFC"):
        analyze_ranking_identity(
            frame,
            condition_universes=duplicated_universe,
            datasets=("toy",),
            scales=(200,),
            minimum_conditions=10,
            draws=2,
        )
    incomplete = frame.loc[frame["graph_type"] != "gene_ontology"]
    with pytest.raises(FrozenRuleError, match="four-arm"):
        analyze_ranking_identity(
            incomplete,
            condition_universes=universes,
            datasets=("toy",),
            scales=(200,),
            minimum_conditions=10,
            draws=2,
        )


def test_official_sanity_uses_absolute_error_formulas_and_boundaries() -> None:
    correlation = evaluate_official_sanity(-0.45, -0.5, metric_kind="correlation_like")
    assert correlation.tolerance == pytest.approx(0.05)
    assert correlation.passed
    zero = evaluate_official_sanity(0.02, 0.0, metric_kind="correlation_like")
    assert zero.passed
    loss = evaluate_official_sanity(1.1, 1.0, metric_kind="loss")
    assert loss.passed
    assert not evaluate_official_sanity(1.1001, 1.0, metric_kind="loss").passed


def test_official_sanity_fails_closed_without_required_tolerance() -> None:
    nonpositive_loss = evaluate_official_sanity(0.0, 0.0, metric_kind="loss")
    assert not nonpositive_loss.eligible
    assert not nonpositive_loss.passed
    other = evaluate_official_sanity(2.0, 1.0, metric_kind="other")
    assert not other.eligible
    official = evaluate_official_sanity(1.01, 1.0, metric_kind="other", official_tolerance=0.01)
    assert official.passed
    with pytest.raises(FrozenRuleError, match="finite"):
        evaluate_official_sanity(np.nan, 0.0, metric_kind="correlation_like")


def _output_rows() -> tuple[pd.DataFrame, dict[tuple[str, str, int], list[str]]]:
    rows = []
    folds = [f"c{index:02d}" for index in range(20)]
    for fold in folds:
        rows.append(
            {
                "model": "m",
                "dataset": "d",
                "scale": 200,
                "fold_id": fold,
                "mapping_passed": True,
                "status": "succeeded",
                "y_true": [0.0, 1.0],
                "y_pred": [0.0, 1.0],
                "training_side_mean": [0.1, 0.2],
            }
        )
    return pd.DataFrame(rows), {("m", "d", 200): folds}


def test_output_validity_uses_prespecified_denominator_and_95_percent_boundary() -> None:
    frame, expected = _output_rows()
    frame.loc[0, "status"] = "failed"
    frame.at[0, "y_pred"] = None
    audit = audit_output_validity(frame, expected_folds=expected)
    row = audit.cell_summary.iloc[0]
    assert row["denominator_n"] == 20
    assert row["valid_n"] == 19
    assert row["valid_fraction"] == pytest.approx(0.95)
    assert audit.passed


def test_output_validity_excludes_only_mapping_failures_and_flags_nonterminal() -> None:
    frame, expected = _output_rows()
    frame.loc[0, "mapping_passed"] = False
    frame.loc[0, "status"] = None
    frame.loc[1, "status"] = "planned"
    audit = audit_output_validity(frame, expected_folds=expected)
    summary = audit.cell_summary.iloc[0]
    assert summary["denominator_n"] == 19
    assert summary["terminal_n"] == 18
    assert not audit.passed
    reasons = audit.fold_ledger.set_index("fold_id")["reason"]
    assert reasons["c00"] == "mapping_gate_excluded"
    assert reasons["c01"] == "nonterminal_status"


def test_output_validity_requires_std_strictly_above_threshold_and_vectors() -> None:
    frame, expected = _output_rows()
    frame.at[0, "y_pred"] = [-1e-8, 1e-8]
    frame.at[1, "training_side_mean"] = [0.0]
    audit = audit_output_validity(frame, expected_folds=expected)
    reasons = audit.fold_ledger.set_index("fold_id")["reason"]
    assert reasons["c00"] == "degenerate_prediction"
    assert reasons["c01"] == "vector_length_mismatch"
    assert not audit.passed


def test_output_validity_rejects_manifest_omission_and_nfc_collision() -> None:
    frame, expected = _output_rows()
    with pytest.raises(FrozenRuleError, match="prespecified"):
        audit_output_validity(frame.iloc[1:], expected_folds=expected)
    collision = frame.iloc[:2].copy()
    collision.loc[0, "fold_id"] = "e\N{COMBINING ACUTE ACCENT}"
    collision.loc[1, "fold_id"] = "é"
    with pytest.raises(FrozenRuleError, match="duplicated"):
        audit_output_validity(
            collision,
            expected_folds={("m", "d", 200): ["x", "y"]},
        )


def test_pair_specific_resolution_uses_linear_quantiles_and_exact_seed_support() -> None:
    result = pair_specific_resolution(
        {42: 0.1, 43: 0.2, 44: 0.3},
        {
            42: [0.1, 0.1, 0.1],
            43: [0.2, 0.15, 0.1],
            44: [0.3, 0.2, 0.1],
        },
        expected_bootstrap_draws=3,
    )
    pairs = np.array([0.1, 0.2, 0.1]) / np.sqrt(2.0)
    assert result.point == pytest.approx(np.quantile(pairs, 0.95, method="linear"))
    assert result.bootstrap_draw_n == 3
    with pytest.raises(FrozenRuleError, match="required seed"):
        pair_specific_resolution(
            {42: 0.1, 43: 0.2},
            {42: [0.1], 43: [0.2]},
            expected_bootstrap_draws=1,
        )


def test_directional_gate_has_strict_p_and_magnitude_boundaries_and_direction() -> None:
    passing = PairEvidence(0.03, 0.01, 0.04, _resolution(0.01))
    assert directional_claim_gate(passing, 0.049).passed
    equality = PairEvidence(0.02, 0.001, 0.03, _resolution(0.01))
    assert not directional_claim_gate(equality, 0.049).passed
    assert not directional_claim_gate(passing, 0.05).passed
    negative = PairEvidence(-0.03, -0.05, -0.01, _resolution(0.01))
    gate = directional_claim_gate(negative, 0.049)
    assert gate.direction == "negative"
    assert gate.passed


def test_g2_family_uses_single_ordered_pcg64_stream_inclusive_ties_and_holm() -> None:
    frame, universes = _family_frame(G2_CONTRAST_ORDER)
    result = evaluate_g2_family(
        frame,
        condition_universes=universes,
        evidence=_evidence(G2_CONTRAST_ORDER),
        datasets=("toy",),
        minimum_conditions=3,
        draws=7,
        expected_resolution_bootstrap_draws=2,
    )
    assert result.contrast_order == G2_CONTRAST_ORDER
    assert result.seed == 20_260_822
    assert result.contrasts["extreme_draw_n"].tolist() == [7, 7]
    assert result.contrasts["raw_p"].tolist() == [1.0, 1.0]
    assert result.contrasts["holm_p"].tolist() == [1.0, 1.0]


def test_external_family_consumes_one_pcg64_stream_in_canonical_order() -> None:
    models = ("zeta", "alpha")
    frame, universes = _family_frame(models)
    effects = {"alpha": (0.3, -0.1, 0.2), "zeta": (0.1, 0.2, -0.2)}
    for model, values in effects.items():
        for condition, value in zip(("a", "b", "c"), values, strict=True):
            frame.loc[
                (frame["contrast"] == model) & (frame["condition"] == condition), "delta_z"
            ] = value
    draws = 13
    result = evaluate_external_family(
        frame,
        retained_models=models,
        condition_universes=universes,
        evidence=_evidence(models),
        datasets=("toy",),
        minimum_conditions=3,
        draws=draws,
        expected_resolution_bootstrap_draws=2,
    )
    generator = np.random.Generator(np.random.PCG64(20_260_818))
    expected_raw = []
    for model in ("alpha", "zeta"):
        values = np.asarray(effects[model])
        observed = float(values.mean())
        extreme = 0
        for _ in range(draws):
            signs = generator.choice((-1.0, 1.0), size=len(values))
            extreme += int(abs(float(np.mean(values * signs))) >= abs(observed))
        expected_raw.append((extreme + 1) / (draws + 1))
    assert result.contrasts["raw_p"].tolist() == expected_raw


def test_external_family_canonicalizes_model_order_and_is_row_permutation_invariant() -> None:
    models = ("zeta", "alpha")
    frame, universes = _family_frame(models)
    first = evaluate_external_family(
        frame,
        retained_models=models,
        condition_universes=universes,
        evidence=_evidence(models),
        datasets=("toy",),
        minimum_conditions=3,
        draws=9,
        expected_resolution_bootstrap_draws=2,
    )
    second = evaluate_external_family(
        frame.sample(frac=1.0, random_state=5),
        retained_models=reversed(models),
        condition_universes=universes,
        evidence=_evidence(models),
        datasets=("toy",),
        minimum_conditions=3,
        draws=9,
        expected_resolution_bootstrap_draws=2,
    )
    assert first.contrast_order == ("alpha", "zeta")
    pd.testing.assert_frame_equal(first.contrasts, second.contrasts)


def test_external_contrast_direction_labels_negative_as_union_favoring() -> None:
    frame, universes = _family_frame(("external",))
    evidence = {"external": PairEvidence(-0.03, -0.05, -0.01, _resolution())}
    result = evaluate_external_family(
        frame,
        retained_models=("external",),
        condition_universes=universes,
        evidence=evidence,
        datasets=("toy",),
        minimum_conditions=3,
        draws=3,
        expected_resolution_bootstrap_draws=2,
    )
    row = result.contrasts.iloc[0]
    assert row["direction"] == "negative"
    assert row["favored_arm"] == "string_go_union"


def test_directional_families_reject_nfc_duplicates_and_incomplete_membership() -> None:
    frame, universes = _family_frame(("é",))
    with pytest.raises(FrozenRuleError, match="NFC"):
        evaluate_external_family(
            frame,
            retained_models=("é", "e\N{COMBINING ACUTE ACCENT}"),
            condition_universes=universes,
            evidence=_evidence(("é",)),
            datasets=("toy",),
            minimum_conditions=3,
            draws=2,
            expected_resolution_bootstrap_draws=2,
        )


def test_directional_family_rejects_incomplete_seed_scale_matrix_and_resolution_draws() -> None:
    frame, universes = _family_frame(G2_CONTRAST_ORDER)
    with pytest.raises(FrozenRuleError, match="frozen support"):
        evaluate_g2_family(
            frame.iloc[1:],
            condition_universes=universes,
            evidence=_evidence(G2_CONTRAST_ORDER),
            datasets=("toy",),
            minimum_conditions=3,
            draws=2,
            expected_resolution_bootstrap_draws=2,
        )
    with pytest.raises(FrozenRuleError, match="resolution-bootstrap"):
        evaluate_g2_family(
            frame,
            condition_universes=universes,
            evidence=_evidence(G2_CONTRAST_ORDER),
            datasets=("toy",),
            minimum_conditions=3,
            draws=2,
        )
    g2_frame, g2_universes = _family_frame(G2_CONTRAST_ORDER)
    with pytest.raises(FrozenRuleError, match="membership"):
        evaluate_g2_family(
            g2_frame.loc[g2_frame["contrast"] == "string_ppi"],
            condition_universes=g2_universes,
            evidence=_evidence(G2_CONTRAST_ORDER),
            datasets=("toy",),
            minimum_conditions=3,
            draws=2,
            expected_resolution_bootstrap_draws=2,
        )
