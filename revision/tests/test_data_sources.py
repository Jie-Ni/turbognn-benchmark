from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cbac_revision.data import ExpressionSource, _select_h5ad_expression
from cbac_revision.errors import RevisionProtocolError


def _fake_h5ad() -> SimpleNamespace:
    return SimpleNamespace(
        X=np.log1p(np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)),
        var_names=np.asarray(["G0", "G1", "G2"]),
        layers={"counts": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)},
        raw=SimpleNamespace(
            X=np.asarray([[7, 8, 9], [10, 11, 12]], dtype=float),
            var_names=np.asarray(["R0", "R1", "R2"]),
        ),
    )


def test_declared_counts_in_log_transformed_x_hard_fails() -> None:
    with pytest.raises(RevisionProtocolError, match="COUNT_MATRIX_NOT_INTEGER_LIKE"):
        _select_h5ad_expression(
            _fake_h5ad(), ExpressionSource(container="X", declared_scale="counts")
        )


def test_explicit_counts_layer_passes_without_x_fallback() -> None:
    matrix, genes = _select_h5ad_expression(
        _fake_h5ad(),
        ExpressionSource(container="layer", layer_key="counts", declared_scale="counts"),
    )

    np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])
    assert genes == ("G0", "G1", "G2")


def test_explicit_raw_counts_passes_and_uses_raw_gene_order() -> None:
    matrix, genes = _select_h5ad_expression(
        _fake_h5ad(), ExpressionSource(container="raw", declared_scale="counts")
    )

    np.testing.assert_array_equal(matrix, [[7, 8, 9], [10, 11, 12]])
    assert genes == ("R0", "R1", "R2")
