"""Explicit perturbation-target encoding with reason-coded failures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .errors import AmbiguousAliasError


@dataclass(frozen=True)
class TargetEncoding:
    """Success or reason-coded failure for one perturbation condition."""

    condition: str
    success: bool
    canonical_targets: tuple[str, ...]
    target_indices: tuple[int, ...]
    reason_code: str | None
    detail: str | None

    def mask(self, n_genes: int) -> np.ndarray:
        """Return a boolean perturbation mask or fail if the target was unencodable."""

        if not self.success:
            raise UnencodableTargetError(self)
        mask = np.zeros(n_genes, dtype=bool)
        mask[list(self.target_indices)] = True
        return mask


class UnencodableTargetError(ValueError):
    """Raised instead of silently using a zero-signal perturbation mask."""

    def __init__(self, encoding: TargetEncoding) -> None:
        self.encoding = encoding
        super().__init__(
            f"[{encoding.reason_code}] condition={encoding.condition!r}: {encoding.detail}"
        )


class TargetEncoder:
    """Encode declared single- or multi-gene targets in the selected gene space."""

    def __init__(
        self,
        gene_names: Sequence[str],
        *,
        gene_aliases: Mapping[str, str] | None = None,
        condition_targets: Mapping[str, Sequence[str] | str] | None = None,
        case_sensitive: bool = False,
        separators_pattern: str = r"[+,;|]",
        gene_space_name: str = "selected_gene_space",
        condition_regex_pattern: str | None = None,
        condition_regex_group: str = "target",
        ignored_target_tokens: Sequence[str] = (),
    ) -> None:
        self.gene_names = tuple(str(gene) for gene in gene_names)
        self.case_sensitive = case_sensitive
        self.separators_pattern = separators_pattern
        self.gene_space_name = str(gene_space_name).strip()
        self.condition_regex_pattern = condition_regex_pattern
        self.condition_regex_group = condition_regex_group
        self._ignored_target_tokens = {self._normalize(token) for token in ignored_target_tokens}
        if not self.gene_space_name:
            raise ValueError("gene_space_name must be non-empty")
        self._gene_index: dict[str, tuple[str, int]] = {}
        for index, gene in enumerate(self.gene_names):
            key = self._normalize(gene)
            if key in self._gene_index:
                raise AmbiguousAliasError(f"Gene names collide after normalization: {gene!r}")
            self._gene_index[key] = (gene, index)
        self._aliases = {
            self._normalize(raw): str(canonical) for raw, canonical in (gene_aliases or {}).items()
        }
        self._condition_targets = dict(condition_targets or {})

    def encode(self, condition: str) -> TargetEncoding:
        """Encode one condition or return a stable reason code explaining failure."""

        raw_condition = str(condition).strip()
        if not raw_condition:
            return self._failure(condition, "EMPTY_TARGET", "Condition label is empty")
        inferred_condition = raw_condition
        if self.condition_regex_pattern:
            match = re.fullmatch(self.condition_regex_pattern, raw_condition)
            if match is not None:
                try:
                    inferred_condition = match.group(self.condition_regex_group)
                except (IndexError, KeyError) as error:
                    raise ValueError(
                        "condition_regex_group is absent from condition_regex_pattern"
                    ) from error
        declared = self._condition_targets.get(raw_condition, inferred_condition)
        if isinstance(declared, str):
            raw_targets = [part.strip() for part in re.split(self.separators_pattern, declared)]
        else:
            raw_targets = [str(part).strip() for part in declared]
        raw_targets = [
            target
            for target in raw_targets
            if self._normalize(target) not in self._ignored_target_tokens
        ]
        if not raw_targets or any(not target for target in raw_targets):
            return self._failure(
                condition,
                "MALFORMED_MULTI_TARGET",
                f"Could not parse an explicit target list from {declared!r}",
            )

        canonical_targets: list[str] = []
        indices: list[int] = []
        missing: list[str] = []
        for raw_target in raw_targets:
            alias_target = self._aliases.get(self._normalize(raw_target), raw_target)
            resolved = self._gene_index.get(self._normalize(alias_target))
            if resolved is None:
                missing.append(str(alias_target))
                continue
            canonical, index = resolved
            canonical_targets.append(canonical)
            indices.append(index)
        if missing:
            reason_code = (
                "TARGET_NOT_IN_DATASET_GENE_UNIVERSE"
                if self.gene_space_name == "dataset_gene_universe"
                else "TARGET_NOT_IN_SELECTED_GENE_SPACE"
            )
            return self._failure(
                condition,
                reason_code,
                f"Unresolved target genes in {self.gene_space_name}: {sorted(set(missing))}",
            )
        unique_pairs = sorted(
            set(zip(canonical_targets, indices, strict=True)), key=lambda item: item[1]
        )
        if not unique_pairs:
            return self._failure(
                condition, "NO_TARGET_AFTER_ALIAS_MAPPING", "No target remained after alias mapping"
            )
        return TargetEncoding(
            condition=str(condition),
            success=True,
            canonical_targets=tuple(pair[0] for pair in unique_pairs),
            target_indices=tuple(pair[1] for pair in unique_pairs),
            reason_code=None,
            detail=None,
        )

    def _normalize(self, value: str) -> str:
        normalized = str(value).strip()
        return normalized if self.case_sensitive else normalized.casefold()

    @staticmethod
    def _failure(condition: str, reason_code: str, detail: str) -> TargetEncoding:
        return TargetEncoding(
            condition=str(condition),
            success=False,
            canonical_targets=(),
            target_indices=(),
            reason_code=reason_code,
            detail=detail,
        )
