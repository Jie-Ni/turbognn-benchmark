"""Auditable components for the CBAC major-revision benchmark."""

from .artifacts import FoldArtifact, FoldIdentity, RuntimeRecord, write_fold_artifact
from .controls import ControlDefinition, ControlResolution, resolve_control_labels
from .graph_supports import GraphMode, GraphSupport, build_graph_support
from .preprocessing import ControlOnlyPreprocessor, PreprocessingConfig, PreprocessingState
from .statistics import (
    benjamini_hochberg,
    bootstrap_settings_from_protocol,
    paired_condition_contrasts,
    seed_average_by_condition,
    summarize_stratified_paired_contrast,
)

__all__ = [
    "ControlDefinition",
    "ControlOnlyPreprocessor",
    "ControlResolution",
    "FoldArtifact",
    "FoldIdentity",
    "GraphMode",
    "GraphSupport",
    "PreprocessingConfig",
    "PreprocessingState",
    "RuntimeRecord",
    "build_graph_support",
    "benjamini_hochberg",
    "bootstrap_settings_from_protocol",
    "paired_condition_contrasts",
    "resolve_control_labels",
    "seed_average_by_condition",
    "summarize_stratified_paired_contrast",
    "write_fold_artifact",
]
