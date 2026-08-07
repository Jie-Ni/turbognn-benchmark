# Legacy benchmark snapshot — forensic use only

> **DO NOT RUN THIS ROOT WORKFLOW FOR THE REVISION.** The files described below are
> the submitted-study implementation retained only to reconstruct and audit the
> original analysis. They contain design defects documented in the major revision,
> including non-matched architectures, transductive preprocessing, and target-mask
> fail-open behaviour. The executable, fail-closed revision protocol is documented
> in [`revision/README.md`](revision/README.md).

This directory preserves the historical `turbognn-benchmark` snapshot. Statements
below describe the submitted workflow and must not be read as claims about the
revision-stage design or as verified current results.

## Scope

Six historical model/graph conditions × four datasets × three gene-space scales ×
three random seeds, evaluated via leave-one-perturbation-out (LOPO) cross-validation.
Five graph-supported conditions used a three-layer GAT, whereas the graph-free arm
used a two-layer Transformer; the six conditions therefore did not share one
architecture.

- Datasets: Norman 2019, Adamson 2016, Replogle 2022 K562, Replogle 2022 RPE1 (all
  from scPerturb Zenodo record 10044268)
- Graphs/models: STRING combined-score functional associations (submitted label:
  PPI), Gene Ontology co-annotation, Pearson co-expression, combined union,
  degree-matched Barabási–Albert random, and no-graph (Transformer baseline)
- HVG scales: 200 / 500 / 1000
- Seeds: 42, 43, 44

## Files

| File | Purpose |
|---|---|
| `run_benchmark.py` | Main pipeline: dataset loading, graph construction, GAT training, LOPO evaluation. Writes per-fold JSON. |
| `turbognn_v2_models.py` | Model definitions (graph attention backbone + Transformer baseline). |
| `merge_results.py` | Merge per-task JSONs into per-(dataset, graph, HVG) aggregates. |
| `merge_all_seeds.py` | Merge all seed-split JSONs into fold-level tables. |
| `statistical_analysis.py` | Paired t-test + Bonferroni + Wilcoxon + Cohen's d + bootstrap CI + ANOVA + Shapiro–Wilk. |
| `slurm/` | HPC launch scripts for the MUSICA clusters (INN + LNZ sites). |

## Historical usage record (do not use for revision results)

1. Install: `pip install torch torch-geometric scanpy anndata scipy numpy pandas`
2. Download Perturb-seq data: `python slurm/download_data.py`
3. Run a single config: `python run_benchmark.py --dataset adamson --graph combined --hvg 200 --seed 42`
4. Full benchmark on SLURM: `bash slurm/submit_all.sh`
5. Merge results: `python merge_all_seeds.py <results_dir>`
6. Statistical analysis: `python statistical_analysis.py <merged_dir>`

## Historical hardware statement (not independently verified)

The submitted materials stated that the benchmark used NVIDIA H100 (96 GB) nodes
on the MUSICA clusters at INN (Innsbruck) and LNZ (Linz), part of the Austrian
Scientific Computing infrastructure, and estimated approximately 1,600 H100-hours
across 720 SLURM tasks. No scheduler/accounting export supplied with the manuscript
independently verifies that estimate; the revision therefore treats it as a
historical author statement rather than measured compute evidence.

## Licence

MIT.

## Contact

Jie Ni (`njie@seu.edu.cn`)
