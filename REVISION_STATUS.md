# CBAC-D-26-03802 major-revision status

The `cbac-major-revision-audit-20260806` branch contains the auditable revision layer for
the manuscript. The repository-root pipeline is retained as the submitted legacy
baseline; its graph-free arm changes architecture and must not be described as a
fixed-backbone adjacency ablation.

The revision implementation, tests, frozen protocol, forensic scripts, source tables,
and figures are under [`revision/`](revision/). No new graph-model or external-model
training result is included in this branch. Compute-dependent result fields remain
locked until the prespecified run ledger is complete.

Verified local evidence in this branch includes:

- 7,660 unique seed-explicit graph-arm records reconstructed from a nominal 9,600-record
  design;
- 40 incomplete expected configuration-seed cells;
- condition-keyed legacy contrasts with seeds averaged inside perturbations;
- 5,400 scheduled external-adaptation evaluations, of which 4,896 contain parse-valid
  scalar source outputs and 504 are reason-coded exclusions; these counts do not by
  themselves establish target-conditioned or implementation validity;
- an architecture-matched GAT implementation, control-only preprocessing, strict control
  resolution, degree-preserving rewires, lossless fold artifacts, and condition-level
  statistics with invariant tests.

Start with [`revision/README.md`](revision/README.md). Protocol validation and unit tests
do not train a model or submit a compute job.
