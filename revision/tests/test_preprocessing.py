from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cbac_revision.controls import ControlDefinition
from cbac_revision.errors import RevisionProtocolError
from cbac_revision.preprocessing import ControlOnlyPreprocessor, PreprocessingConfig


def _fit(expression: np.ndarray) -> ControlOnlyPreprocessor:
    preprocessor = ControlOnlyPreprocessor(
        PreprocessingConfig(n_hvg=2, input_scale="log1p", hvg_method="control_variance")
    )
    preprocessor.fit(
        expression,
        gene_names=["A", "B", "C", "D"],
        condition_labels=["ctrl", "ctrl", "ctrl", "TP53", "MYC"],
        control=ControlDefinition("gene", "control", aliases=("ctrl",)),
    )
    return preprocessor


def test_perturbation_values_cannot_change_hvgs_or_scaler() -> None:
    controls = np.asarray(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 5.0, 2.0],
            [1.0, 5.0, 10.0, 1.0],
        ]
    )
    first = np.vstack([controls, np.zeros((2, 4))])
    second = np.vstack([controls, np.full((2, 4), 1_000_000.0)])

    first_state = _fit(first).state
    second_state = _fit(second).state

    assert first_state is not None and second_state is not None
    assert first_state.selected_indices == second_state.selected_indices
    assert first_state.selected_genes == second_state.selected_genes
    assert first_state.control_means == second_state.control_means
    assert first_state.control_scales == second_state.control_scales
    assert first_state.sha256() == second_state.sha256()


def test_transformed_controls_use_control_only_center_and_scale() -> None:
    expression = np.asarray(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 5.0, 2.0],
            [1.0, 5.0, 10.0, 1.0],
            [500.0, 500.0, 500.0, 500.0],
            [800.0, 800.0, 800.0, 800.0],
        ]
    )
    preprocessor = _fit(expression)
    transformed = preprocessor.transform(expression)

    np.testing.assert_allclose(transformed[:3].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(transformed[:3].std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_unscaled_selected_control_channel_preserves_gene_baselines() -> None:
    expression = np.asarray(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 5.0, 2.0],
            [1.0, 5.0, 10.0, 1.0],
            [500.0, 500.0, 500.0, 500.0],
            [800.0, 800.0, 800.0, 800.0],
        ]
    )
    preprocessor = _fit(expression)
    control_channel = preprocessor.transform_selected_unscaled(expression[:3]).mean(axis=0)

    assert not np.allclose(control_channel, 0.0)
    assert len(np.unique(control_channel)) > 1


def test_forced_panel_target_is_retained_before_control_variance_fill() -> None:
    expression = np.asarray(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 5.0, 2.0],
            [1.0, 5.0, 10.0, 1.0],
            [2.0, 4.0, 8.0, 2.0],
        ]
    )
    preprocessor = ControlOnlyPreprocessor(PreprocessingConfig(n_hvg=2, input_scale="log1p"))
    state = preprocessor.fit(
        expression,
        ["LOW_VARIANCE_TARGET", "B", "C", "D"],
        ["ctrl", "ctrl", "ctrl", "TP53"],
        ControlDefinition("gene", "control", aliases=("ctrl",)),
        forced_gene_names=("LOW_VARIANCE_TARGET",),
    )

    assert "LOW_VARIANCE_TARGET" in state.selected_genes
    assert state.forced_genes == ("LOW_VARIANCE_TARGET",)


def test_forced_target_overflow_is_reason_coded() -> None:
    expression = np.arange(20, dtype=float).reshape(5, 4)
    preprocessor = ControlOnlyPreprocessor(PreprocessingConfig(n_hvg=1, input_scale="log1p"))

    with pytest.raises(RevisionProtocolError, match="HVG_FORCED_RETENTION_OVERFLOW"):
        preprocessor.fit(
            expression,
            ["A", "B", "C", "D"],
            ["ctrl", "ctrl", "ctrl", "X", "Y"],
            ControlDefinition("gene", "control", aliases=("ctrl",)),
            forced_gene_names=("A", "B"),
        )


def test_sparse_counts_profile_aggregation_matches_dense_without_full_densification() -> None:
    expression = np.asarray(
        [
            [1, 2, 0, 4],
            [2, 1, 3, 0],
            [3, 2, 1, 1],
            [4, 0, 2, 2],
            [0, 5, 1, 2],
        ],
        dtype=float,
    )
    labels = ["ctrl", "ctrl", "ctrl", "A", "B"]
    control = ControlDefinition("gene", "control", aliases=("ctrl",))
    dense = ControlOnlyPreprocessor(PreprocessingConfig(n_hvg=3, input_scale="counts"))
    sparse_fit = ControlOnlyPreprocessor(PreprocessingConfig(n_hvg=3, input_scale="counts"))
    dense.fit(expression, ["A", "B", "C", "D"], labels, control)
    sparse_fit.fit(sparse.csr_matrix(expression), ["A", "B", "C", "D"], labels, control)

    dense_profiles = dense.mean_profiles_by_label(expression, labels)
    sparse_profiles = sparse_fit.mean_profiles_by_label(sparse.csr_matrix(expression), labels)
    assert dense.state is not None and sparse_fit.state is not None
    assert dense.state.selected_genes == sparse_fit.state.selected_genes
    for label in dense_profiles:
        np.testing.assert_allclose(dense_profiles[label], sparse_profiles[label], atol=1e-12)
