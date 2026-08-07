# Archived graph-output accounting repair

This folder reconstructs the accepted seed-explicit output lineage, removes unsafe positional
pairing, and reports saved scalar differences on exact condition-key support. It does **not** restore
perturbation conditioning or convert the archive into a valid graph-prior experiment.

Interpret every table here together with `../legacy_target_mask_audit/`: all 50 Adamson labels and
at least 27 of 50 Norman labels necessarily produced an empty legacy target mask. The additional
affected Norman and Replogle fractions cannot be recovered because fold-specific post-HVG gene lists
and mask-success records were not retained. The submitted dense comparator also changed architecture,
and preprocessing was transductive. Accordingly, effects, intervals, BA contrasts, and top-condition
overlaps in this folder are **source-output descriptors**, not target-conditioned prediction results.

- `accepted_file_manifest.csv` and `canonical_fold_level.csv` preserve logical source identifiers and
  SHA-256-backed lineage without workstation paths.
- `configuration_manifest.csv` records complete, partial, and absent configuration-seed cells.
- `paired_condition_contrasts.csv` and `paired_condition_details.csv` implement key-based accounting
  repair; their inferential-looking columns are retained for numerical reconciliation only.
- `top_condition_identity.csv` measures archived output-key overlap and cannot be interpreted as
  perturbation prioritisation where target-mask success is absent or unknown.
- `existing_results_integrity_report.md` states the hard interpretation boundary.

Regenerate the tables with `SOURCE/scripts/reanalyze_existing_results.py` and explicit input/output
directories. No model training is performed.
