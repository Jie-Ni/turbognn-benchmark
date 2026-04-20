# turbognn-benchmark

Minimal code release for the controlled benchmark of biological knowledge graph
priors in perturbation prediction across four Perturb-seq datasets.

## Scope

Six graph conditions × four datasets × three gene-space scales × three random seeds,
evaluated via leave-one-perturbation-out (LOPO) cross-validation on a shared graph
attention network backbone.

- Datasets: Norman 2019, Adamson 2016, Replogle 2022 K562, Replogle 2022 RPE1 (all
  from scPerturb Zenodo record 10044268)
- Graphs: STRING PPI, Gene Ontology co-annotation, Pearson co-expression, combined
  union, degree-matched Barabási–Albert random, no-graph (Transformer baseline)
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

## Usage

1. Install: `pip install torch torch-geometric scanpy anndata scipy numpy pandas`
2. Download Perturb-seq data: `python slurm/download_data.py`
3. Run a single config: `python run_benchmark.py --dataset adamson --graph combined --hvg 200 --seed 42`
4. Full benchmark on SLURM: `bash slurm/submit_all.sh`
5. Merge results: `python merge_all_seeds.py <results_dir>`
6. Statistical analysis: `python statistical_analysis.py <merged_dir>`

## Hardware

Benchmark was executed on NVIDIA H100 (96 GB) nodes on the MUSICA clusters at INN
(Innsbruck) and LNZ (Linz), part of the Austrian Scientific Computing infrastructure.
Total compute: approximately 1,600 H100-hours across 720 SLURM tasks.

## Licence

MIT.

## Contact

Jie Ni (`njie@seu.edu.cn`)
