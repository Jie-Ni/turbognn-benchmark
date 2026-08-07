"""Control-only, leakage-safe feature selection and scaling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .controls import ControlDefinition, resolve_control_labels
from .errors import LeakageRiskError, RevisionProtocolError


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fully explicit preprocessing choices; no data-scale auto-detection is allowed."""

    n_hvg: int
    input_scale: str = "counts"
    target_library_size: float = 10_000.0
    hvg_method: str = "control_variance"
    scale_epsilon: float = 1e-8

    def validate(self) -> None:
        """Validate supported, predeclared preprocessing choices."""

        if self.n_hvg <= 0:
            raise RevisionProtocolError("n_hvg must be positive")
        if self.input_scale not in {"counts", "log1p"}:
            raise RevisionProtocolError("input_scale must be exactly 'counts' or 'log1p'")
        if self.target_library_size <= 0:
            raise RevisionProtocolError("target_library_size must be positive")
        if self.hvg_method != "control_variance":
            raise RevisionProtocolError("Only the predeclared control_variance method is supported")
        if self.scale_epsilon <= 0:
            raise RevisionProtocolError("scale_epsilon must be positive")


@dataclass(frozen=True)
class PreprocessingState:
    """Fitted HVG and scaler state derived exclusively from control cells."""

    config: PreprocessingConfig
    selected_indices: tuple[int, ...]
    selected_genes: tuple[str, ...]
    forced_indices: tuple[int, ...]
    forced_genes: tuple[str, ...]
    hvg_scores: tuple[float, ...]
    control_means: tuple[float, ...]
    control_scales: tuple[float, ...]
    n_control_cells: int
    matched_control_labels: tuple[str, ...]
    fit_scope: str = "control_only"
    selection_policy: str = "forced_panel_targets_then_control_variance_fill"

    def sha256(self) -> str:
        """Hash the fitted state using canonical JSON serialization."""

        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ControlOnlyPreprocessor:
    """Fit HVGs, means, and scales on controls; transform all cells with that state."""

    def __init__(self, config: PreprocessingConfig) -> None:
        config.validate()
        self.config = config
        self.state: PreprocessingState | None = None

    def fit(
        self,
        expression: np.ndarray | sparse.spmatrix,
        gene_names: Sequence[str],
        condition_labels: Sequence[object],
        control: ControlDefinition,
        *,
        forced_gene_names: Sequence[str] = (),
    ) -> PreprocessingState:
        """Fit HVGs and scaling statistics using the resolved control rows only."""

        matrix = _validate_expression(expression, gene_names, condition_labels)
        if self.config.n_hvg > matrix.shape[1]:
            raise RevisionProtocolError(
                f"Requested {self.config.n_hvg} HVGs but only {matrix.shape[1]} genes are present"
            )
        resolution = resolve_control_labels(condition_labels, control)
        transformed_controls = _per_cell_transform(matrix[resolution.mask], self.config)
        if transformed_controls.shape[0] < 2:
            raise LeakageRiskError("At least two explicitly resolved control cells are required")

        scores = _sample_variance(transformed_controls)
        genes = np.asarray([str(gene) for gene in gene_names], dtype=object)
        normalized_gene_index: dict[str, int] = {}
        for index, gene in enumerate(genes):
            key = str(gene).casefold()
            if key in normalized_gene_index:
                raise RevisionProtocolError(
                    f"Gene identifiers collide case-insensitively: {gene!r}"
                )
            normalized_gene_index[key] = index
        missing_forced = sorted(
            {
                str(gene)
                for gene in forced_gene_names
                if str(gene).casefold() not in normalized_gene_index
            }
        )
        if missing_forced:
            raise RevisionProtocolError(
                "[FORCED_TARGET_NOT_IN_DATASET_GENE_SPACE] "
                f"Missing forced genes: {missing_forced}"
            )
        forced = sorted({normalized_gene_index[str(gene).casefold()] for gene in forced_gene_names})
        if len(forced) > self.config.n_hvg:
            raise RevisionProtocolError(
                "[HVG_FORCED_RETENTION_OVERFLOW] "
                f"{len(forced)} panel target genes exceed n_hvg={self.config.n_hvg}"
            )
        ranking = np.lexsort((genes, -scores))
        fill = [index for index in ranking if int(index) not in set(forced)]
        selected = np.asarray(
            sorted([*forced, *fill[: self.config.n_hvg - len(forced)]]), dtype=int
        )
        selected_controls = transformed_controls[:, selected]
        means = _column_mean(selected_controls)
        scales = np.sqrt(np.maximum(_sample_variance(selected_controls), 0.0))
        scales = np.where(scales > self.config.scale_epsilon, scales, 1.0)

        state = PreprocessingState(
            config=self.config,
            selected_indices=tuple(int(index) for index in selected),
            selected_genes=tuple(str(genes[index]) for index in selected),
            forced_indices=tuple(int(index) for index in forced),
            forced_genes=tuple(str(genes[index]) for index in forced),
            hvg_scores=tuple(float(scores[index]) for index in selected),
            control_means=tuple(float(value) for value in means),
            control_scales=tuple(float(value) for value in scales),
            n_control_cells=resolution.n_control_cells,
            matched_control_labels=resolution.matched_raw_labels,
        )
        self.state = state
        return state

    def transform(self, expression: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Apply the fitted control-only transform to arbitrary cells."""

        if self.state is None:
            raise LeakageRiskError("fit must be called before transform")
        matrix = _as_float_matrix(expression)
        if matrix.ndim != 2:
            raise RevisionProtocolError("expression must be a two-dimensional cell-by-gene matrix")
        max_index = max(self.state.selected_indices)
        if matrix.shape[1] <= max_index:
            raise RevisionProtocolError("expression does not contain the fitted gene columns")
        transformed = _per_cell_transform(matrix, self.config)
        selected = transformed[:, np.asarray(self.state.selected_indices, dtype=int)]
        if sparse.issparse(selected):
            selected = selected.toarray()
        means = np.asarray(self.state.control_means)
        scales = np.asarray(self.state.control_scales)
        return (selected - means) / scales

    def transform_selected_unscaled(
        self, expression: np.ndarray | sparse.spmatrix
    ) -> np.ndarray | sparse.spmatrix:
        """Return selected log-expression before gene-wise centering and scaling.

        This is the auditable control-expression input channel. The fitted means and
        scales remain control-only, but subtracting the control mean from the model's
        control input itself would erase gene-specific baseline expression.
        """

        if self.state is None:
            raise LeakageRiskError("fit must be called before transform_selected_unscaled")
        matrix = _as_float_matrix(expression)
        if matrix.ndim != 2:
            raise RevisionProtocolError("expression must be a two-dimensional cell-by-gene matrix")
        max_index = max(self.state.selected_indices)
        if matrix.shape[1] <= max_index:
            raise RevisionProtocolError("expression does not contain the fitted gene columns")
        transformed = _per_cell_transform(matrix, self.config)
        return transformed[:, np.asarray(self.state.selected_indices, dtype=int)]

    def mean_profiles_by_label(
        self, expression: np.ndarray | sparse.spmatrix, labels: Sequence[object]
    ) -> dict[str, np.ndarray]:
        """Aggregate selected standardized profiles without loading full cell-by-HVG blocks."""

        if self.state is None:
            raise LeakageRiskError("fit must be called before mean_profiles_by_label")
        matrix = _as_float_matrix(expression)
        if matrix.shape[0] != len(labels):
            raise RevisionProtocolError("labels length must equal expression rows")
        transformed = _per_cell_transform(matrix, self.config)
        selected = transformed[:, np.asarray(self.state.selected_indices, dtype=int)]
        means = np.asarray(self.state.control_means)
        scales = np.asarray(self.state.control_scales)
        label_array = np.asarray(labels, dtype=object)
        output: dict[str, np.ndarray] = {}
        for label in sorted(set(str(value) for value in label_array)):
            rows = selected[label_array == label]
            profile = _column_mean(rows)
            output[label] = (profile - means) / scales
        return output

    def fit_transform(
        self,
        expression: np.ndarray | sparse.spmatrix,
        gene_names: Sequence[str],
        condition_labels: Sequence[object],
        control: ControlDefinition,
        *,
        forced_gene_names: Sequence[str] = (),
    ) -> np.ndarray:
        """Fit on controls and transform every row without using perturbation statistics."""

        self.fit(
            expression,
            gene_names,
            condition_labels,
            control,
            forced_gene_names=forced_gene_names,
        )
        return self.transform(expression)


def fit_nested_preprocessors(
    expression: np.ndarray | sparse.spmatrix,
    gene_names: Sequence[str],
    condition_labels: Sequence[object],
    control: ControlDefinition,
    *,
    scales: Sequence[int] = (200, 500, 1000),
    input_scale: str = "counts",
    target_library_size: float = 10_000.0,
    hvg_method: str = "control_variance",
    scale_epsilon: float = 1e-8,
    forced_gene_names: Sequence[str] = (),
) -> dict[int, ControlOnlyPreprocessor]:
    """Fit one deterministic nested HVG family from the same control-only ranking."""

    ordered_scales = tuple(int(scale) for scale in scales)
    if not ordered_scales or ordered_scales != tuple(sorted(set(ordered_scales))):
        raise RevisionProtocolError("Nested HVG scales must be a strictly increasing sequence")
    if ordered_scales[-1] > len(gene_names):
        raise RevisionProtocolError(
            f"Requested nested scale {ordered_scales[-1]} with only {len(gene_names)} genes"
        )
    if len({str(gene).casefold() for gene in forced_gene_names}) > ordered_scales[0]:
        raise RevisionProtocolError(
            "[HVG_FORCED_RETENTION_OVERFLOW] Forced targets exceed the smallest nested panel"
        )
    fitted: dict[int, ControlOnlyPreprocessor] = {}
    previous: set[str] = set()
    for scale in ordered_scales:
        preprocessor = ControlOnlyPreprocessor(
            PreprocessingConfig(
                n_hvg=scale,
                input_scale=input_scale,
                target_library_size=target_library_size,
                hvg_method=hvg_method,
                scale_epsilon=scale_epsilon,
            )
        )
        state = preprocessor.fit(
            expression,
            gene_names,
            condition_labels,
            control,
            forced_gene_names=forced_gene_names,
        )
        current = set(state.selected_genes)
        if not previous <= current:
            raise RevisionProtocolError(
                f"Nested HVG invariant failed between {len(previous)} and {scale} genes"
            )
        previous = current
        fitted[scale] = preprocessor
    return fitted


def nested_gene_panel_manifest(
    preprocessors: Mapping[int, ControlOnlyPreprocessor],
) -> dict[str, Any]:
    """Create a self-hashed manifest binding every native panel to the common smallest panel."""

    scales = tuple(sorted(int(scale) for scale in preprocessors))
    if not scales:
        raise RevisionProtocolError("Nested preprocessor mapping cannot be empty")
    panels: list[dict[str, Any]] = []
    previous: set[str] = set()
    for scale in scales:
        state = preprocessors[scale].state
        if state is None:
            raise RevisionProtocolError(f"Nested preprocessor {scale} is not fitted")
        genes = tuple(state.selected_genes)
        current = set(genes)
        if len(genes) != scale or not previous <= current:
            raise RevisionProtocolError("Nested gene-panel manifest invariant failed")
        panels.append(
            {
                "hvg": scale,
                "gene_order": list(genes),
                "gene_order_sha256": _canonical_hash(genes),
                "preprocessing_state_sha256": state.sha256(),
                "contains_previous_panel": True,
            }
        )
        previous = current
    common = tuple(preprocessors[scales[0]].state.selected_genes)  # type: ignore[union-attr]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "PASS",
        "nesting_rule": "200_subset_500_subset_1000" if scales == (200, 500, 1000) else "strict",
        "scales": list(scales),
        "panels": panels,
        "common_evaluation_hvg": scales[0],
        "common_gene_order": list(common),
        "common_gene_order_sha256": _canonical_hash(common),
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_expression(
    expression: np.ndarray | sparse.spmatrix,
    gene_names: Sequence[str],
    condition_labels: Sequence[object],
) -> np.ndarray | sparse.spmatrix:
    matrix = _as_float_matrix(expression)
    if matrix.ndim != 2:
        raise RevisionProtocolError("expression must be a two-dimensional cell-by-gene matrix")
    if matrix.shape[0] != len(condition_labels):
        raise RevisionProtocolError("condition_labels length must equal the number of cells")
    if matrix.shape[1] != len(gene_names):
        raise RevisionProtocolError("gene_names length must equal the number of genes")
    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(values).all():
        raise RevisionProtocolError("expression contains non-finite values")
    if len(set(str(gene) for gene in gene_names)) != len(gene_names):
        raise RevisionProtocolError("gene_names must be unique")
    return matrix


def _per_cell_transform(
    matrix: np.ndarray | sparse.spmatrix, config: PreprocessingConfig
) -> np.ndarray | sparse.spmatrix:
    values = matrix.data if sparse.issparse(matrix) else matrix
    if config.input_scale == "log1p":
        if (values < 0).any():
            raise RevisionProtocolError("log1p expression cannot contain negative values")
        return matrix.astype(np.float64, copy=True)
    if (values < 0).any():
        raise RevisionProtocolError("count input cannot contain negative values")
    library_sizes = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if (library_sizes <= 0).any():
        raise RevisionProtocolError("count input contains an empty cell")
    factors = config.target_library_size / library_sizes
    if sparse.issparse(matrix):
        normalized = sparse.diags(factors) @ matrix.tocsr()
        normalized.data = np.log1p(normalized.data)
        return normalized
    normalized = matrix * factors[:, None]
    return np.log1p(normalized)


def _as_float_matrix(expression: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.spmatrix:
    if sparse.issparse(expression):
        return expression.astype(np.float64).tocsr()
    return np.asarray(expression, dtype=np.float64)


def _column_mean(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).reshape(-1)


def _sample_variance(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    n_rows = matrix.shape[0]
    if n_rows < 2:
        raise LeakageRiskError("At least two rows are required for sample variance")
    means = _column_mean(matrix)
    if sparse.issparse(matrix):
        squared_means = np.asarray(matrix.power(2).mean(axis=0)).reshape(-1)
    else:
        squared_means = np.mean(np.square(matrix), axis=0)
    return np.maximum((squared_means - np.square(means)) * n_rows / (n_rows - 1), 0.0)
