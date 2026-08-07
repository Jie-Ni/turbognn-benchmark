# Dataset control-label audit

## Frozen production contract

The staged Adamson and Norman files use `obs.perturbation`; neither uses `obs.gene`. Norman has an
explicit `control` category (11,855 cells). The Adamson file has no
literal `control` category. A source-rebuild notebook at Pfizer's public repository commit
`41f70980b481b9c854772cd8b8c5a4753c1c3eac` strips the vector suffix and maps both `62(mod)` and `63(mod)` to `control`.
The exact staged labels are `62(mod)_pBA581` (2 cells) and
`63(mod)_pBA580` (6,010 cells); they are pooled only as the
verified control reference.

The Adamson labels `Gal4-4(mod)_pBA582`, `*`, and missing perturbation annotations are not admitted
as prediction endpoints. `Gal4-4(mod)_pBA582` is control-like/non-human but is not silently pooled
without an exact source mapping. This exclusion prevents the archived error in which a modal-label
fallback selected `63(mod)_pBA580` while another control-like label entered the 50-condition panel.

## Release gate

The production runner must match both the condition column and every control/exclusion label above,
record counts in the dataset passport, and stop on any mismatch. It may not choose the modal condition
or infer a control from string frequency. Different Adamson guide labels remain distinct held-out
conditions; a shared canonical target is used for target masking and related-target exclusion, not for
automatic guide pooling.

The exact Replogle K562 and RPE1 production matrices are not included in this author-review package.
Their 310,385/247,914 row and 2,057/2,393 condition counts are therefore retained only as
source-stage descriptors, not locally revalidated execution invariants. Both datasets remain
fail-closed until a production passport binds the exact H5AD checksum, matrix shape, condition
column, observed labels, control contract, and condition-to-target mappings. The status and missing
fields are explicit in `dataset_matrix_count_provenance.csv`.

