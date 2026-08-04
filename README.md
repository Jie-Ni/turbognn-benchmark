# TurboGNN biological-prior benchmark revision scaffold

This repository is the auditable execution scaffold prepared for the
CBAC-D-26-03802 major revision. It contains stricter configuration, provenance,
LOPO-result, graph, schedule, and partial statistical-analysis code. It does **not**
contain newly executed manuscript experiments or new numerical results.

## Frozen planning scope

The planned benchmark covers Norman, Adamson, Replogle K562, and Replogle RPE1 at
200, 500, and 1,000 genes with training seeds 42, 43, and 44. The same-architecture
primary contrast is the fixed STRING-GO-union GAT versus the self-loop-only GAT.
STRING and GO are separate evidence arms. Co-expression and Transformer are
sensitivity arms; degree-preserving rewired and density-matched Barabási-Albert
graphs are topology controls.

The schedule generator freezes these matrices:

| Matrix | Strict configurations | Planned LOPO folds | Other records |
|---|---:|---:|---:|
| E0 | 16 | 160 | 3 official-reference records |
| Core | 144 | 7,200 | 0 |
| Sensitivity | 72 | 3,600 | 0 |
| Topology | 360 | 18,000 | 0 |

E0 includes two external GEARS configuration records for which no executable
adapter is currently supplied. Counts are plans, not completed runs or hardware
usage.

## What is implemented and unit-tested

- Strictly validated dataset-passport, control, target, preprocessing, graph, model,
  manifest, and fold-result loaders; JSON Schema is used where a corresponding schema
  is present, while the control map uses exact-key validation.
- NFC-safe canonical condition/alias pooling, deterministic condition ranking,
  target-preserving gene panels, split hashes, and explicit cell/alias leakage
  checks.
- Frozen graph construction and graph hashes for STRING, GO, fixed union,
  co-expression, self-loop, rewired, and Barabási-Albert arms.
- A 68-column feature contract, lossless true/predicted/control/training-mean
  vectors, explicit invalid-metric states, and exactly-20-gene delta Jaccard.
- Run-key/terminal-manifest lineage checks across runner and merge paths.
- Schedule rows bound to the exact input hashes and code commit. The canonical
  launcher checks the task-file byte hash, ordered-row hash, row count, complete
  SLURM array geometry, bounds, and non-empty process launch.
- Partial condition/seed aggregation and primary-estimand functions with toy tests.

The tests use synthetic or toy data and do not train the manuscript benchmark.

## Inputs are deliberately not bundled

Files under `config/*.example.json` are non-executable templates and fail when
placeholders remain. Before any run, a curator must provide all of the following
outside the checkout and preserve their exact bytes:

- source-verified dataset passports and the matching AnnData files;
- control labels, complete raw-to-canonical target mappings, design evidence, and
  the physical raw-to-stable mapping table;
- preprocessing/expression-state evidence;
- projected STRING and GO edge files with releases, licences, checksums, and
  projection evidence;
- a verified GPU-capable environment lock; and
- pinned external-reference/Systema sources and adapters where applicable.

The revision scaffold must be committed and the checkout clean before the runner's
code-commit gate can pass. Publishing this non-main branch does not by itself create
an experiment release or satisfy the evidence gates below.

## Canonical flow after those blockers are resolved

Generate one immutable schedule and its exact task file:

> Use only `generate_revision_tasks.py` for revision schedules. The older
> `generate_tasks.py`, `generate_split_tasks.py`, `generate_replogle_tasks.py`,
> `generate_missing_seeds.py`, and `generate_missing_v2.py` are historical,
> noncanonical utilities; some write task files immediately instead of implementing
> a safe `--help` path.

```text
python -m slurm.generate_revision_tasks --stage core \
  --output-manifest <schedule.json> --output-tasks <tasks.txt> \
  --code-commit <40-character-lowercase-git-sha> \
  --dataset-passport <verified-passport.json> \
  --control-map <verified-controls.json> --target-map <verified-targets.json> \
  --preprocessing-config <verified-preprocessing.json> \
  --model-config config/model_tds42.json \
  --environment-lock <verified-environment-lock> \
  --graph-source-config <verified-graph-sources.json> \
  --graph-ensemble-config config/graph_ensembles.json
```

The data synchronizer requires an executable passport:

```text
python -m slurm.download_data --manifest <verified-passport.json> --data-root <data-root>
```

`slurm/job_revision.sh` is the canonical four-GPU array launcher. Submit exactly
`ceil(runner_task_row_count / 4)` array elements, numbered from zero with step one.
It requires the `TURBOGNN_*` paths documented in the script and an output root
outside the checkout. HPC submission, paid computation, and real experiments are
not performed here and require explicit authorization.

The equivalent single-row runner interface is:

```text
python run_benchmark.py --stage full --datasets <dataset> --graph-types <arm> \
  --results-dir <fresh-directory-outside-checkout> \
  --dataset-passport <passport.json> --data-root <data-root> \
  --control-map <controls.json> --target-map <targets.json> \
  --preprocessing-config <preprocessing.json> --model-config config/model_tds42.json \
  --code-commit <sha> --environment-lock <verified-lock> \
  --schedule-manifest <schedule.json> --schedule-config-hash <configuration-hash> \
  --graph-source-config <graph-sources.json> \
  --graph-ensemble-config config/graph_ensembles.json \
  --panel-size 50 --num-hvg <200-or-500-or-1000> --seed <42-or-43-or-44> \
  --fold-start 0 --fold-end 50
```

Use a frozen `--graph-instance-seed` for rewired or Barabási-Albert rows. Merge only
terminal-manifest-backed task chunks with `merge_all_seeds.py`; do not use
`--include-aggregates` for a canonical revision analysis. `statistical_analysis.py`
is not yet the complete submission analysis described below.

## Environment evidence

`requirements-test-environment-captured.txt` records the exact direct package
versions observed for the local unit-test run. It is CPU-only, lacks Scanpy/AnnData
and compiled PyG extension packages, has no wheel/artifact hashes, and has not been
recreated from a fresh clone. It is therefore **not** an experiment/HPC lock and
must not be supplied as evidence for TDS-20/TDS-42 reproducibility.

PyG 2.7.0 exposes `GATConv.node_dim = 0`; the model asserts that runtime value.
Because `node_dim` is upstream-hardcoded rather than passed as a public constructor
argument, the final experiment environment must pin and verify the exact PyTorch,
PyG, CUDA, compiled-extension, driver, and transitive dependency artifacts.

## PM-TBR: required before a submittable execution package

The following are explicit planning blockers, not completed features:

- **Manifest/result conservation (TDS-01):** create the global fold-key manifest
  before data preparation; quarantine a result if terminal-manifest finalization
  fails; add a final aggregate-wide ordered succeeded-run-key hash/count; and make
  the canonical path reject pre-existing aggregates rather than relying on operator
  discipline.
- **Target evidence (TDS-02):** emit malformed/unsupported mappings as explicit
  ineligible-ledger rows instead of aborting; cross-check source observation design
  fields against each raw alias; bind the physical mapping-table path, bytes, size,
  and non-placeholder digest into the schedule and runtime provenance.
- **Model parity (TDS-05):** persist and validate canonical non-edge parameter and
  buffer manifests, names, shapes, trainable/total counts, exclusion policy, and
  exact paired-arm parity in addition to the current initialization hash.
- **Formal analyses (TDS-14/15/16/17/18/22):** wire the frozen-rule helpers into the
  production CLI, including the T1 effect analysis, exact G2/δ-res estimands,
  bootstrap/sign-flip families, branch decisions, and machine-verifiable outputs.
- **Telemetry (TDS-19):** add UTC start/end, reserved and peak CPU/RAM/GPU resources,
  driver/GPU count, SLURM cluster/job/step identity, accelerator-seconds, and `sacct`
  reconciliation.
- **Reproducible runtime (TDS-20/TDS-42):** produce an exact transitive lock with
  artifact hashes, recreate it from a fresh clone on the target GPU stack, and run
  deterministic parity tests. The present test capture is insufficient.
- **Systema/external evidence (TDS-36 and adapters):** pin and verify the external
  source, implement the formal two-layer CA/CI/δ-CA/F2 workflow, and implement GEARS
  official-reference and strict-LOPO adapters.
- **Analysis-build coherence (TDS-40):** make the production analysis consume and
  enforce schedule manifest/content/configuration-universe hashes and a frozen
  configuration allowlist across every paired primary arm.
- **Production feature proof (TDS-41):** rerun the feature-artifact contract in the
  final GPU environment and include it in end-to-end analysis provenance.
- Add the prespecified exact R/lme4/emmeans mixed-model environment and production
  invocation; currently only Python-side planning/scaffolding exists.

Until these items and the verified external inputs exist, this repository must be
described as a tested revision scaffold, not as a complete executable conversion,
completed benchmark, or submission-ready result package.

## Local non-experimental verification

```text
python -m black --check .
python -m ruff check .
python -m pytest -q
git diff --check
```

## Licence

MIT; see `LICENSE`.
