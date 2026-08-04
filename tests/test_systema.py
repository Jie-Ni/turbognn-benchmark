from __future__ import annotations

import pandas as pd
import pytest

from turbognn_audit.systema import (
    centroid_accuracies_exact,
    centroid_sign_flip_families,
    verify_systema_source,
)


def test_exact_centroid_accuracy_uses_strict_distance_and_ties_are_incorrect() -> None:
    predictions = pd.DataFrame(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "perfect"),
                ("A", "tie"),
                ("B", "perfect"),
                ("B", "tie"),
                ("C", "perfect"),
                ("C", "tie"),
            ],
            names=["condition", "method"],
        ),
        columns=["g1", "g2"],
    )
    truth = pd.DataFrame(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        index=["A", "B", "C"],
        columns=["g1", "g2"],
    )
    scores = centroid_accuracies_exact(predictions, truth)
    assert scores["perfect"].tolist() == [1.0, 1.0, 1.0]
    assert scores["tie"].tolist() == [0.5, 0.5, 0.5]


def test_pinned_source_rejects_line_ending_or_content_substitution(tmp_path) -> None:
    source = tmp_path / "centroid_accuracy.py"
    source.write_text("def calculate_centroid_accuracies():\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Pinned Systema source mismatch"):
        verify_systema_source(source)


def _contrast(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": condition,
                "centroid_accuracy_difference": value,
            }
            for dataset in ("d1", "d2", "d3", "d4")
            for condition in ("A", "B")
        ]
    )


def test_centroid_families_use_frozen_graph_only_order_and_holm_family() -> None:
    contrasts = {
        "string_go_union": _contrast(0.2),
        "string_ppi": _contrast(0.1),
        "gene_ontology": _contrast(0.05),
    }
    result = centroid_sign_flip_families(
        contrasts,
        datasets=("d1", "d2", "d3", "d4"),
        draws=20,
        seed=20_260_836,
    )
    assert result["contrast"].tolist() == [
        "string_go_union",
        "string_ppi",
        "gene_ontology",
    ]
    assert result.iloc[0]["family"] == "G1_unadjusted_primary_sensitivity"
    assert set(result.iloc[1:3]["family"]) == {"G2_STRING_GO_Holm"}
    assert (result["adjusted_p"] >= result["raw_p"]).all()


def test_centroid_sign_flip_rejects_external_models() -> None:
    contrasts = {
        "string_go_union": _contrast(0.2),
        "string_ppi": _contrast(0.1),
        "gene_ontology": _contrast(0.05),
    }
    with pytest.raises(ValueError, match="not members"):
        centroid_sign_flip_families(
            contrasts,
            datasets=("d1", "d2", "d3", "d4"),
            retained_external_models=("gears",),
            draws=10,
        )


def test_centroid_families_allow_frozen_pair_specific_external_support() -> None:
    contrasts = {
        "string_go_union": _contrast(0.2),
        "string_ppi": _contrast(0.1),
        "gene_ontology": _contrast(0.05).iloc[:-1],
    }
    first = centroid_sign_flip_families(
        contrasts,
        datasets=("d1", "d2", "d3", "d4"),
        draws=10,
        seed=4,
    )
    second = centroid_sign_flip_families(
        contrasts,
        datasets=("d1", "d2", "d3", "d4"),
        draws=10,
        seed=4,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first["contrast"] == "gene_ontology", "pair_support_n"].iloc[0] == 7


def test_centroid_metric_rejects_partial_methods_and_nonfinite_values() -> None:
    truth = pd.DataFrame(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        index=["A", "B", "C"],
        columns=["g1", "g2"],
    )
    partial = pd.DataFrame(
        [[0.0, 0.0], [2.0, 0.0]],
        index=pd.MultiIndex.from_tuples(
            [("A", "model"), ("B", "model")], names=["condition", "method"]
        ),
        columns=["g1", "g2"],
    )
    with pytest.raises(ValueError, match="every frozen truth condition"):
        centroid_accuracies_exact(partial, truth)
    complete = pd.concat(
        [
            partial,
            pd.DataFrame(
                [[float("nan"), 2.0]],
                index=pd.MultiIndex.from_tuples([("C", "model")], names=["condition", "method"]),
                columns=["g1", "g2"],
            ),
        ]
    )
    with pytest.raises(ValueError, match="finite"):
        centroid_accuracies_exact(complete, truth)


def test_centroid_sign_flip_rejects_nonpositive_draw_count() -> None:
    with pytest.raises(ValueError, match="draws must be positive"):
        centroid_sign_flip_families(
            {
                "string_go_union": _contrast(0.2),
                "string_ppi": _contrast(0.1),
                "gene_ontology": _contrast(0.05),
            },
            datasets=("d1", "d2", "d3", "d4"),
            draws=0,
        )
