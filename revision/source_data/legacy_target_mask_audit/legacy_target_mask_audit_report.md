# Legacy perturbation-target mask audit

## Code-level finding

The archived runner splits perturbation labels only on `+`, requires an exact match to the post-HVG gene list, and returns `None` when no token matches. The archived graph model documents `None` as a wild-type prediction and applies target masking only for a non-None mask. This is a fail-open target-conditioning path.

## Condition-label audit

| Dataset | Archived conditions | With `+` | With `_` | Guide suffix | Labels verified in source values | Guaranteed no match in full source gene universe |
|---|---:|---:|---:|---:|---:|---:|
| adamson | 50 | 0 | 50 | 50 | 50 | 50 |
| norman | 50 | 0 | 27 | 0 | 50 | 27 |
| replogle_k562 | 50 | 0 | 0 | 0 | not_assessed | not_assessed |
| replogle_rpe1 | 50 | 0 | 0 | 0 | not_assessed | not_assessed |

A full-gene-universe non-match is a sufficient condition for a `None` mask after any HVG subset. A full-universe match is not sufficient for successful legacy masking because the target may still have been excluded by full-data HVG selection.

Against the exact public files named by the archived downloader, all 50 archived Adamson labels and all 50 archived Norman labels are verified source perturbation values. All 50 Adamson guide-suffixed labels and the 27 Norman underscore-pair labels have no legacy-token match even in the full source gene universe. They therefore necessarily yield `None` after every 200-, 500-, or 1,000-HVG subset. The remaining 23 Norman single-target labels may still fail after HVG selection, but the missing fold-specific gene lists prevent resolving that additional fraction.

## Interpretation boundary

The archived artifacts do not retain the per-dataset/per-scale gene order or a mask-success ledger. Therefore the affected fraction beyond the guaranteed Adamson 50/50 and Norman 27/50 failures cannot be reconstructed from scalar outputs alone. Legacy graph contrasts, condition rankings, and comparisons that use the submitted scale-dependent union arm (saved label `Combined`: STRING--GO--control-co-expression at 200 HVGs and STRING--GO at 500/1,000 HVGs) are source-output descriptors; they are not reliable estimates of target-conditioned perturbation prediction performance.

The revision-stage runner addresses this defect by canonicalising dataset-declared separators, retaining every selected perturbation target in the gene panel, recording the resolved targets, and stopping a fold when the target indicator is empty or inconsistent.
