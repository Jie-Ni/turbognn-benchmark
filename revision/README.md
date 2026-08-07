# CBAC major-revision analysis package

This directory is the repository-ready implementation of the matched-control experiments,
audits, statistics, and release gates required by the major revision. It contains no new
scientific model result. The only model execution performed while preparing this package
was a synthetic two-epoch CPU integration test; no real H5AD benchmark or HPC job was run.
The submitted legacy pipeline at repository root was not modified.

## Scientific correction implemented here

The submitted comparison changed adjacency and architecture together: the submitted sparse
graph arms used a three-layer GAT, whereas `no_graph` used a two-layer Transformer. The
revision uses one `MatchedGAT` in every arm and changes only the explicit edge support:

| Arm class | Edge support | Architecture |
|---|---|---|
| `dense` | all directed gene pairs plus explicit self-loops | `MatchedGAT` |
| `self_loop` | self-edge-only support; no graph-edge message passing | `MatchedGAT` |
| `string_go` (primary sparse support) | STRING combined-score functional associations plus GO biological-process ontology support and explicit self-loops | `MatchedGAT` |
| topology null | within-component double-edge swaps plus explicit self-loops | `MatchedGAT` |
| `combined` (mixed sensitivity only) | STRING--GO plus control-only coexpression support | `MatchedGAT` |

Every layer fixes `GATConv(add_self_loops=False)`, so the support builder—not a library
default—controls loops. Each topology null must preserve the exact degree sequence, isolate
count, connected-component count, and component membership; all ten null hashes must be
unique and each must replace at least 80% of the source edges. A mean-degree-matched
Barabasi-Albert null is prohibited.

Each node receives two biological channels: control-only log-normalized mean expression
and a binary perturbation-target indicator. Every arm uses an identically parameterised
gene-identity embedding copied from the same paired initial state and bound to the
selected-gene order and preprocessing hash; each separately trained arm learns its own
weights and no learned weights are shared. Hidden
states use per-node `LayerNorm`; `BatchNorm1d` is prohibited. The self-edge-only test changes
all other node inputs while holding the target node and fixed model state constant and
requires the target output to remain unchanged, demonstrating no graph-edge message
passing. For a fixed dataset/HVG/condition/seed, paired arms are identically initialized,
as verified by initial-state hashes, and then trained separately under identical training,
validation, optimizer, scheduler, stopping, and target-exclusion rules.

Preprocessing is control-only. Per-cell normalization is independent, and the gene
variance rank, center, scale, and coexpression graph are fitted exclusively on explicitly
resolved controls. The held-out perturbation never enters those fits. Panel construction
uses identifiers, cell counts, perturbation order, and declared metadata—not perturbation
effect magnitude. Only `condition_cell_count_quartile` and `perturbation_order` are
selection strata; `perturbation_class` and `pathway_class` are descriptive audits.

The primary estimand is explicitly guide-condition weighted. A separate secondary
sensitivity first averages guide conditions within each canonical target set and then
resamples target sets. Neither analysis may silently reinterpret guide labels as unique
genes.

## Fail-closed contracts

- Production H5AD inputs require a hash-verified dataset passport. The passport binds the
  accession, source checksum, matrix schema and shape, expression container/scale, exact
  perturbation column, raw-to-canonical condition map, target map, raw and canonical cell
  counts, control evidence and counts, attrition ledger, legacy first-50 panel, and source
  file hash. Production preflight independently compares those declarations with the
  loaded H5AD and canonicalisation ledger.
- Adamson uses `perturbation` and accepts the two archived controls
  `62(mod)_pBA581` and `63(mod)_pBA580`; the Gal4 construct, unresolved archive labels, and
  missing condition values are reason-coded exclusions. Norman also uses `perturbation`.
  Replogle condition columns must be resolved by their real dataset passports.
- No majority-class, modal-label, first-match, or control-alias fallback is allowed.
- Each compared condition must contain exactly seeds 42, 43, and 44 in both arms. Seeds are
  averaged within guide condition before inference and are not bootstrap units.
- Compared arms must have identical `(dataset, hvg, panel, condition)` coverage. Inner-join
  loss, missing epochs, incomplete arms, or duplicate fit keys withhold release.
- A target absent from the selected gene space is a reason-coded failure. An all-zero target
  indicator is rejected.
- Every fold artifact stores raw `y_true`/`y_pred`, all loss and learning-rate histories,
  the full model/training/split/space config, input and code hashes, runtime/hardware fields,
  and a best-state checkpoint. Metrics are recomputed from vectors, never trusted from
  training-time scalar output.
- The checkpoint is safely decoded on read and cross-bound to the artifact identity, full
  config hash, input hashes, architecture, graph, preprocessing, gene order, initialization,
  code, Git state, vector space, and validation-selected best epoch.
- Early stopping is replayed epoch by epoch from the bound validation history using the
  frozen strict `minimum_delta` and patience state machine. The selected epoch, observed
  stopping epoch, reason, and complete decision trace must agree across training, artifact,
  and checkpoint. PyTorch deterministic algorithms, cuDNN flags, and the cuBLAS workspace
  setting are enforced and recorded.
- The exact dependency lock, environment manifest, base Git commit, path-redacted dirty
  state hash, and code-tree hash are bound into every released artifact.

## Repository layout

```text
revision/
├── protocol.yaml                         frozen experiment and inference contract
├── run_matched_benchmark.py              dry-run-first benchmark CLI
├── aggregate_conditional_trigger.py      four-dataset global-OR trigger wrapper
├── postprocess_revision_results.py       complete 13-lock release wrapper
├── render_source_values.py               self-hashed 13-lock registry-to-TeX renderer
├── run_gears_positive_control.py         pinned official-GEARS evidence adapter
├── cbac_revision/
│   ├── controls.py                       strict control resolution
│   ├── data.py                           NPZ fixture and H5AD/passport loaders
│   ├── preprocessing.py                  control-only HVG/scaling state
│   ├── panels.py                         deterministic 50 + 50 panels
│   ├── targets.py                        guide/target mapping with reason codes
│   ├── graph_builders.py                 executable STRING and GO builders
│   ├── graph_supports.py                 matched supports and topology nulls
│   ├── model.py                          fixed dynamic-support GAT
│   ├── runner.py                         preflight, execution, attempts, manifests
│   ├── artifacts.py                      lossless records and checkpoint validation
│   ├── global_trigger.py                 cross-dataset conditional gate
│   ├── statistics.py                     hierarchical and sensitivity inference
│   ├── gears_reproduction.py             official checkout execution/package gate
│   ├── gears_validation.py               three-branch GEARS eligibility decision
│   ├── compute_release.py                measured-compute release validation
│   ├── compute_reconcile.py              scheduler-grounded interruption reconciliation
│   ├── environment_lock.py               complete active-environment lock exporter
│   ├── postprocess.py                    deterministic 13-lock release package
│   └── source_values.py                  strict TeX formatting and round-trip verification
├── manifests/                            field-complete, deliberately non-releasable examples
└── tests/                                contract, tamper, interruption, and toy execution tests
```

The forensic utilities for saved legacy outputs do not substitute for the matched-control
experiment. Their outputs are source-output descriptors only.

## Install and validate without model execution

From `revision/`:

```bash
python -m pip install -e ".[test,model]"
python -m cbac_revision protocol.yaml
python -m pytest
```

Install `.[h5ad,model]` only in the approved compute environment. No package command
downloads benchmark data or submits a scheduler job.

## Build STRING functional-association and GO support reproducibly

The STRING builder retains human v12.0 functional-association edges only when
`combined_score > 400`, requires an
explicit identifier map unless the raw identifiers are already gene symbols, collapses
undirected duplicates deterministically, and writes a self-auditing derivation manifest:

```bash
python -m cbac_revision.graph_builders string \
  --raw /inputs/9606.protein.links.v12.0.txt \
  --identifier-map /inputs/string_to_gene_symbol.tsv \
  --map-id-column string_id --map-symbol-column gene_symbol --map-delimiter tab \
  --source-column protein1 --target-column protein2 \
  --score-column combined_score --delimiter whitespace \
  --protein-taxon-prefix 9606. --taxon 9606 \
  --source-url https://string-db.org/ --release STRING_v12.0 \
  --output /derived/string_edges.csv \
  --manifest /derived/string_derivation.json
```

The GO builder reads a real GAF and OBO, restricts annotations to biological process,
applies an explicit evidence-code, qualifier, ancestor, and minimum-depth policy, and
writes both the gene-gene edge list and derivation manifest:

```bash
python -m cbac_revision.graph_builders go \
  --gaf /inputs/goa_human.gaf \
  --obo /inputs/go-basic.obo \
  --release-metadata /inputs/go_release.json \
  --evidence-code EXP --evidence-code IDA --evidence-code IMP \
  --ancestor-policy is_a_transitive --minimum-depth 3 \
  --output /derived/go_edges.csv \
  --manifest /derived/go_derivation.json
```

The example manifests are deliberately blocked by `REPLACE_WITH_*` tokens. A production
run must replace them with observed sources and hashes; there is no synthetic fallback.
For backwards-compatible CLI wiring only, the internal key `string_ppi` means STRING
combined-score functional associations, not a physical protein--protein interaction claim;
the internal key `curated` means generic supplied sparse support, not expert curation. The
primary manuscript arm is always `string_go`.

## Exact production environment lock

Production preflight requires a plain-text `cbac-complete-pip-freeze-v2` lock containing
every distribution installed in the interpreter—not a curated package subset. Generate it
inside the exact approved runtime:

```bash
python -m cbac_revision.environment_lock --output /inputs/cbac_environment.lock
```

The exporter derives direct runtime dependencies from `pyproject.toml` and fails if the
H5AD/model runtime is incomplete. In particular, a production H5AD lock must contain
`h5py`, `anndata`, PyTorch, and PyTorch Geometric. The lock also binds Python, CUDA, cuDNN,
NVIDIA driver, and the PyG compiled-extension version set:

```text
# lock-format: cbac-complete-pip-freeze-v2
# python-version: <platform.python_version()>
# cuda-runtime-version: <torch.version.cuda or NOT_AVAILABLE>
# cudnn-version: <torch.backends.cudnn.version() or NOT_AVAILABLE>
# nvidia-driver-version: <nvidia-smi driver or NOT_AVAILABLE>
# pytorch-version: <installed torch distribution or NOT_INSTALLED>
# pytorch-geometric-version: <installed torch-geometric distribution or NOT_INSTALLED>
# complete-distribution-set-sha256: <sha256>
# direct-runtime-dependencies-sha256: <sha256>
# pyg-extension-versions-sha256: <sha256>
<every normalized installed distribution>==<exact-version>
```

Any active distribution omitted from the lock, any extra distribution added to it, a
changed compiled dependency, ranges, editable references, duplicate normalized names,
mismatched, missing, duplicate, or extra headers, and placeholders are hard failures. The
same runtime-derived header contract is recomputed in the selected GEARS interpreter and
cross-bound to both phase audits; copied header text is never treated as runtime evidence.

## Production preflight, then explicit execution

The default command performs all data, control, panel, target, graph, topology-null,
environment, Git, and fit-count checks and writes a complete preflight package. It does not
initialize a model:

```bash
python run_matched_benchmark.py \
  --protocol protocol.yaml \
  --dataset adamson \
  --data /inputs/adamson.h5ad \
  --dataset-passport /inputs/adamson.dataset_passport.json \
  --environment-lock /inputs/cbac_environment.lock \
  --graph string_ppi=/derived/string_edges.csv \
  --graph string_raw=/inputs/9606.protein.links.v12.0.txt \
  --graph string_identifier_map=/inputs/string_to_gene_symbol.tsv \
  --graph string_builder=cbac_revision/graph_builders.py \
  --graph string_derivation_manifest=/derived/string_derivation.json \
  --graph gene_ontology=/derived/go_edges.csv \
  --graph go_gaf=/inputs/goa_human.gaf \
  --graph go_obo=/inputs/go-basic.obo \
  --graph go_builder=cbac_revision/graph_builders.py \
  --graph go_derivation_manifest=/derived/go_derivation.json \
  --graph go_release_metadata=/inputs/go_release.json \
  --output /runs/adamson_hvg200_primary \
  --hvg 200 --analysis-block topology_primary --panel primary --dry-run
```

Training begins only when the same verified command receives `--execute`. Real compute
still requires the separate compute approval specified by the project workflow. Any
preflight blocker prevents model initialization.

## Frozen fit matrix and conditional gate

Every arm/dataset cell contains 50 guide conditions × 3 seeds = 150 fits.

| Analysis block | Arm × scale cells per dataset | Four-dataset fits |
|---|---:|---:|
| 200-HVG topology: dense + self-loop + `string_go` + 10 `string_go` rewires | 13 | 7,800 |
| 200-HVG mixed-support sensitivity: `combined` (the dense comparator is reused) | 1 | 600 |
| Scale extension: `string_go` + dense at 500 and 1,000 (both 200-HVG fits are reused) | 4 | 2,400 |
| **Mandatory total** | **18** | **10,800** |

The 200-HVG `string_go` and dense results are reused in the scale analysis. The independent
non-overlap panel is always constructed, but its 200-HVG `string_go`+dense replication runs
only if the frozen pre-outcome trigger is met: combined selection-stratum total-variation
distance is strictly above 0.10, or the absolute standardized log1p cell-count difference
is strictly above 0.25. Optional pathway/class metadata are descriptive only and cannot
trigger the analysis. The hash-bound long-form audit reports every categorical level,
numeric summary, optional-field `NOT_AVAILABLE` state, target-mapping coverage, raw mapped
and excluded label counts, and exclusion reason. One local trigger activates all four
datasets, adding 1,200 fits; the maximum is 12,000. Exact threshold equality does not
trigger. Fewer than 50 eligible non-overlapping primary conditions blocks preflight;
triggered conditional support below 50 conditions is withheld.

Each primary 200-HVG topology preflight also retains every hash-bound standardized control
cell profile. Postprocessing selects exactly one 200-HVG topology evidence manifest per
dataset, rejects duplicate or missing dataset×HVG evidence, resamples control cells and
guide conditions within dataset, reconstructs the shared pseudobulk reference, and
recomputes prediction/truth correlations on every bootstrap draw. A feasibility flag is
not a result: the primary uncertainty lock is withheld unless this sensitivity is
`MEASURED` with a finite interval.

After the four production 200-HVG primary preflights, aggregate the global OR:

```bash
python -m cbac_revision.global_trigger \
  --protocol protocol.yaml \
  --preflight-summary /runs/adamson_hvg200_primary/preflight/preflight_summary.json \
  --preflight-summary /runs/norman_hvg200_primary/preflight/preflight_summary.json \
  --preflight-summary /runs/replogle_k562_hvg200_primary/preflight/preflight_summary.json \
  --preflight-summary /runs/replogle_rpe1_hvg200_primary/preflight/preflight_summary.json \
  --output /runs/global_conditional_trigger.json
```

The aggregator revalidates every summary, 50-condition panel, protocol/data/passport/target,
environment, Git, and code binding. Conditional runs require this manifest and
`--analysis-block conditional_nonoverlap --panel sensitivity`; a non-triggered manifest
blocks those fits as not required.

## Interruption-safe compute accounting

Before each fit starts, the runner atomically publishes a content-addressed attempt ledger
containing `STARTED_UNFINALIZED`. Success or a reason-coded failure replaces that exact row
and publishes a new immutable ledger plus an atomic registry pointer. Resume is blocked if
an unresolved started row remains or if a newer orphan ledger exists without its registry
pointer. A hard interruption is reconciled only from an exact scheduler record whose
content hash, job ID, final state, timestamps, allocation seconds, device count, and exit
reason all match the command. The reconciliation replaces—not deletes—the unresolved row,
publishes a new content-addressed ledger, and permits the next attempt:

```bash
python -m cbac_revision.compute_reconcile \
  --registry /runs/adamson_hvg200_primary/measured_compute_registry.json \
  --run-id <run-id> --dataset adamson --hvg 200 --panel primary \
  --arm combined --condition <condition> --seed 42 --attempt-index 1 \
  --scheduler-job-id SLURM_JOB_ID:<job-id> --scheduler-state PREEMPTED \
  --started-at-utc <UTC> --finished-at-utc <UTC> \
  --elapsed-allocation-seconds <seconds> --device-count 1 \
  --exit-or-preemption-reason <reason-code> \
  --source-record /scheduler/job_record.json \
  --source-record-sha256 <sha256>
```

Completed fits are skipped only after the artifact and safely decoded checkpoint still
exist and match the ledger. A missing or changed retained file invalidates the earlier
success, records a reason-coded failed attempt, and forces a new attempt. Prior failures are
retained, indices must be consecutive, and the final attempt for every fit key must be the
unique success.

Accounting separates in-process fit scope, end-to-end scheduler allocation, failed or
preempted allocation, retry allocation overhead, checkpoint/serialization/registry/other
non-fit work, and allocation that cannot be phase-attributed after a hard interruption.
The registry also reports phase median/IQR, device and accelerator hours, peak-memory
status/distribution, and failure-reason denominators.

## GEARS positive-control reproduction

GEARS is pinned to the official repository
`https://github.com/snap-stanford/GEARS` at commit
`f374e43e197b295016d80395d7a54ddb81cc6769`. The official metric implementation is
`gears/inference.py::compute_metrics`, which returns `mse`, `mse_de`, `pearson`, and
`pearson_de`. The adapter calls the official `PertData` and `GEARS` APIs, captures the
official epoch-level train/validation MSE log, reevaluates the best model, exports raw
prediction/truth and DE vectors, computes a control-mean baseline, and audits gene order
and target coverage. Each phase also retains safe numeric NPZ sidecars for initialization,
best checkpoint, and graph state; split membership, row IDs, condition-target mappings,
gene order, loss histories, baseline vectors, and sidecar array manifests are semantic
evidence rather than trusted booleans.

Copy `manifests/gears_execution_plan.example.json` next to
`run_gears_positive_control.py`, replace every placeholder with real hash-bound values,
predeclare expected metrics and tolerances, and compute `plan_hash`. Then run:

```bash
python -m cbac_revision.gears_reproduction \
  --plan /gears_plan/gears_execution_plan.json \
  --repository-root /checkouts/GEARS \
  --python /gears_env/bin/python \
  --output /runs/gears_positive_control \
  --trust-anchor-output /trusted/gears_positive_control.anchor.json \
  --execute
```

The checkout must have the exact origin/commit and no tracked modifications; the
interpreter must exactly match the declared GEARS package lock. Both commands are argv
arrays—shell strings and `python -c` are rejected. The executor recomputes all metrics by
loading the hash-bound official source; reported values must match exactly.
The two phases have explicit, non-interchangeable roles: `official_reference` is a frozen
repeatability run through the pinned official API, not an independently implemented model
or metric, while `candidate` is the `heldout_adaptation` assessment. Each phase requires a
different hash-bound expected-value provenance record frozen before execution. That record
separately binds role, expected metrics, repository URL and commit, dataset, split,
entrypoint, and normalized argv. Reusing the identical entrypoint and argv is rejected
unless both phase specifications explicitly declare the pinned-official-workflow reuse
policy and still carry distinct provenance records and source identifiers. This exception
documents an identical official-workflow rerun; it must never be described as an
independent implementation.
The final validator independently reopens the raw evidence and NPZ sidecars, recomputes
model and retained-baseline metrics, requires a positive direction-aware improvement over
the bound control-mean baseline for every declared metric, derives strictly positive
first-to-final training and validation loss improvements, verifies condition- and
between-condition non-degeneracy, and cross-checks the copied plan, argv template,
entrypoint, split, graph, initialization, checkpoint, retained gene order, retained
perturbable-gene order, every condition target index and binary indicator, and loss length.
Refreshing outer hashes cannot turn malformed semantic evidence into a valid run.

Release also requires an authentication boundary outside the evidence bundle. The executor
copies every bound source and generated evidence file into the sibling source store of the
detached anchor, independently recomputes the environment, plan, dataset, expected-value,
gene-order, target-mapping, and graph commitments from those copies, and writes the anchor
outside `--output`. Record the exact SHA-256 of that anchor in an access-controlled release
configuration before accepting or transferring the evidence bundle. Final validation is:

```bash
python -m cbac_revision.gears_validation \
  --manifest /runs/gears_positive_control/gears_positive_control_manifest.json \
  --trust-anchor /trusted/gears_positive_control.anchor.json \
  --expected-trust-anchor-sha256 CALLER_PINNED_64_HEX_SHA256 \
  --output /runs/gears_positive_control/gears_validation_registry.json
```

The validator rejects a missing anchor, an anchor inside the evidence bundle, any mismatch
against the caller-pinned hash, any changed external trusted source, or any package file that
differs from its authenticated external copy. Bundle-local self-hashes remain useful for
structure and accidental-corruption checks, but they are not an authentication root. The
cryptographic boundary is the caller-provided expected anchor SHA-256: if an attacker can
rewrite both the anchor and that expected value, offline validation has no security meaning.
The pinned value therefore must not be stored in, derived from, or controlled by the evidence
bundle.

Eligibility has three non-conflated branches:

- valid evidence within tolerance: `ELIGIBLE_POSITIVE_CONTROL_PASSED`;
- valid executed evidence outside tolerance: `EXCLUDED_FAILED_POSITIVE_CONTROL` and not
  ranking-eligible, but evidentiary release remains valid;
- missing, placeholder, tampered, or incomplete evidence:
  `WITHHELD_INVALID_OR_INCOMPLETE_EVIDENCE`.

scGPT and Geneformer use the same three-state family but require their own official
repository/commit, independent expected-value provenance, dataset/split/condition/target/
gene-order bindings, raw truth/prediction/baseline vectors, loss histories, and independently
recomputed Pearson/MSE/MAE gates. A valid reason-coded claim-deletion manifest resolves a
member as `EXCLUDED`; an unverifiable member is `WITHHELD`. Only `PASS` members enter the
external performance family. At each of 200, 500, and 1,000 HVGs, that family recomputes the
condition-paired Pearson difference against the revision-stage `string_go` reference on
identical dataset, primary panel, condition order, gene order, split, and target support;
the three contrasts receive BH adjustment within adapter. Legacy Combined, dense, or an
anonymous reference cannot substitute for revision `string_go`.

The analytic baselines are zero/control delta, training-condition mean, and deterministic
ridge (`alpha=1.0`) fitted from training-only target indicators. Ridge is fitted once per
dataset/HVG/held-out condition, outside the neural-seed loop: 4 datasets × 3 HVGs × 50
conditions = 600 unique ridge refits. Evaluating three analytic methods gives 1,800 method
evaluations; neither number is included in the 10,800 neural fits. Pearson for the constant
zero predictor remains explicitly undefined; its absolute-skill comparison uses MSE/MAE.

## Deterministic postprocessing and the 13 locks

After all approved fits are complete, postprocessing uses two explicit passes. The
asset-candidate pass reads every artifact, safely validates its checkpoint, recomputes
metrics, enforces mandatory coverage, runs guide-condition and target-level analyses,
validates measured compute and the three-member external-comparator family, and derives all
reader assets from those same analysis objects:

```bash
python -m cbac_revision.postprocess \
  --protocol protocol.yaml \
  --global-trigger-manifest /runs/global_conditional_trigger.json \
  --preflight-summary /runs/adamson_hvg200_primary/preflight/preflight_summary.json \
  --preflight-summary /runs/adamson_hvg500_scale/preflight/preflight_summary.json \
  --preflight-summary /runs/adamson_hvg1000_scale/preflight/preflight_summary.json \
  --artifact-root /runs/adamson_hvg200_primary/artifacts \
  --artifact-root /runs/adamson_hvg500_scale/artifacts \
  --artifact-root /runs/adamson_hvg1000_scale/artifacts \
  --external-family-manifest /runs/external_family_manifest.json \
  --external-family-manifest-sha256 CALLER_PINNED_64_HEX_SHA256 \
  --external-performance-manifest /runs/external_performance_manifest.json \
  --external-performance-manifest-sha256 CALLER_PINNED_64_HEX_SHA256 \
  --compute-registry /runs/adamson_hvg200_primary/measured_compute_registry.json \
  --compute-registry /runs/adamson_hvg500_scale/measured_compute_registry.json \
  --compute-registry /runs/adamson_hvg1000_scale/measured_compute_registry.json \
  --baseline-root /runs/baselines \
  --release-mode asset-candidate \
  --release-asset-output /runs/release_candidate/generated \
  --output /runs/release_candidate
```

Record the resulting `release_asset_manifest.json` file SHA-256, add that exact manifest
and every required raw/source commitment to the detached main-release anchor, and freeze
the anchor file SHA-256 in caller-controlled configuration. Then rerun the same complete
analysis inputs in verification mode, now supplying both caller-pinned boundaries:

```bash
python -m cbac_revision.postprocess \
  --protocol protocol.yaml \
  --global-trigger-manifest /runs/global_conditional_trigger.json \
  --preflight-summary /runs/adamson_hvg200_primary/preflight/preflight_summary.json \
  --preflight-summary /runs/adamson_hvg500_scale/preflight/preflight_summary.json \
  --preflight-summary /runs/adamson_hvg1000_scale/preflight/preflight_summary.json \
  --artifact-root /runs/adamson_hvg200_primary/artifacts \
  --artifact-root /runs/adamson_hvg500_scale/artifacts \
  --artifact-root /runs/adamson_hvg1000_scale/artifacts \
  --external-family-manifest /runs/external_family_manifest.json \
  --external-family-manifest-sha256 CALLER_PINNED_64_HEX_SHA256 \
  --external-performance-manifest /runs/external_performance_manifest.json \
  --external-performance-manifest-sha256 CALLER_PINNED_64_HEX_SHA256 \
  --compute-registry /runs/adamson_hvg200_primary/measured_compute_registry.json \
  --compute-registry /runs/adamson_hvg500_scale/measured_compute_registry.json \
  --compute-registry /runs/adamson_hvg1000_scale/measured_compute_registry.json \
  --baseline-root /runs/baselines \
  --release-mode released \
  --release-asset-manifest /runs/release_candidate/generated/release_asset_manifest.json \
  --release-asset-manifest-sha256 CALLER_PINNED_RELEASE_ASSET_MANIFEST_SHA256 \
  --main-trust-anchor /trusted/main_release.anchor.json \
  --main-trust-anchor-sha256 CALLER_PINNED_MAIN_ANCHOR_SHA256 \
  --evidence-bundle-root /runs \
  --output /runs/release_package
```

Repeat `--preflight-summary`, `--artifact-root`, and `--compute-registry` for every executed
dataset/HVG/block. The abbreviated command above is illustrative and cannot release the
four-dataset package by itself. `--baseline-root` must contain exactly the 600 unique
primary-panel artifacts keyed by 4 datasets × 3 HVGs × 50 held-out conditions; do not point
it at a conditional-panel or mixed artifact tree. An explicit list of 600
`--baseline-artifact` arguments is equivalent and avoids an over-broad recursive root.
`asset-candidate` exits successfully after deterministic asset construction but keeps the
package release status withheld; only the second pass can validate the detached anchor and
promote the package.

The decision registry contains exactly these 13 locks:

`PRIMARY-DELTA-R`, `PRIMARY-UNCERTAINTY-INTERVAL`, `PRIMARY-DECISION`, `PROPAGATION-CONTROL`,
`SCALE-INTERACTION`, `TOPOLOGY-NULL`, `REPRESENTATIVENESS-TRIGGER`,
`CONDITIONAL-PANEL`, `FRACTION-IMPROVED`, `EMPIRICAL-SEED-RESOLUTION`,
`SECONDARY-METRICS`, `EXTERNAL-COMPARATOR-VALIDATION`, and `MEASURED-COMPUTE`.

The registry-to-TeX renderer accepts JSON numbers only for numeric fields: booleans,
numeric strings, NaN, infinities, out-of-range probabilities, and inconsistent integer or
IQR summaries are rejected. It also verifies that the primary estimate lies inside the
reported 95% conditional bootstrap uncertainty interval and independently recomputes the
directional part of `PRIMARY-DECISION` from the unrounded interval relative to zero:
strictly above zero is directional positive, strictly below zero is directional negative,
and an interval containing zero is inconclusive. The archived optimisation-repeatability
reference `0.010` has no formal success or claim authority; `0.020` appears only as the
pre-outcome q90 simulated interval-width precision target. Sparse-support wording further
requires eligible absolute-skill and condition-ranking consequences. Biological-edge
wording additionally requires directionally supportive STRING--GO-vs-rewire topology
evidence and passed graph diagnostics. TeX display rounding uses decimal round-half-up and
never changes any decision.

`PROPAGATION-CONTROL` contains one two-contrast family on complete matched 200-HVG support:
`string_go_minus_self_loop` and `dense_minus_self_loop`, both subtracting the identical
dataset-condition-seed self-loop observation and both receiving the same two-member BH
adjustment.

Released mode derives all 12 rich reader tables from the same validated analysis registries
and retained artifacts; caller-supplied unrelated tables cannot substitute. The canonical
builder emits CSV, compact group-specific TeX, SVG, and PDF, then binds all 49 files and
their semantic table hashes in `release_asset_manifest.json`. The generated
`released_results_include.tex` reads `\CBACReleaseConsumer`: `main` receives headline
primary/topology/claim material, `supplement` receives all 12 groups, and `response`
receives the compact closure subset. Author-review mode shows a human-readable 12-row
withheld-status block and still inputs all 12 generated placeholder TeX files, proving the
complete dependency path without fabricating numbers.

Any mandatory coverage failure cascades to the numerical locks. A correctly non-triggered
conditional panel is a valid resolved state; missing conditional evidence when triggered
is not. No manuscript result should be promoted from planned/withheld to observed/released
until this final registry and its source-table hashes are present.

The manuscript lock macros are never hand-transcribed. Author review retains canonical
placeholders; a fully released registry can populate the same 13 macros with fixed display
precision and validated decision tokens. Both modes emit a self-hashed round-trip manifest:

```bash
python render_source_values.py \
  --registry /runs/release_package/decision_release_registry.json \
  --template /paper/source_values.tex \
  --output /paper/generated/source_values.tex \
  --manifest-output /paper/generated/source_values_render_manifest.json \
  --mode released

python render_source_values.py \
  --registry /runs/release_package/decision_release_registry.json \
  --template /paper/generated/source_values.tex \
  --output /paper/generated/source_values.tex \
  --manifest-output /paper/generated/source_values_render_manifest.json \
  --mode released --verify
```

Released mode rejects any withheld lock, missing or extra lock, broken inner or outer hash,
unknown decision token, malformed interval, invalid probability, or manual macro edit.
