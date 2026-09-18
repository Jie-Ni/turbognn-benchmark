# Benchmarking Biological Knowledge-Graph Priors Against Single-Cell Foundation Models for Perturb-seq Prediction

[![Journal](https://img.shields.io/badge/Journal-Computational%20Biology%20and%20Chemistry-blue)](https://www.sciencedirect.com/journal/computational-biology-and-chemistry)
[![Status](https://img.shields.io/badge/Status-Minor%20Revision%20(CBAC--D--26--03802R1)-green)](https://www.editorialmanager.com/cbac/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Benchmark components and evaluation utilities accompanying the manuscript:
> **Benchmarking Biological Knowledge-Graph Priors Against Single-Cell Foundation Models for Perturb-seq Prediction**  
> *Computational Biology and Chemistry* (CBAC-D-26-03802R1).

---

## 1. Study Scope & Design Hierarchy

This study investigates whether curated biological knowledge graphs (STRING functional associations and Gene Ontology co-annotation) improve Perturb-seq transcriptional phenotype predictions over structural and uninformative graph supports under a strictly controlled, matched evaluation protocol.

- **Primary Analysis:** Fixed-backbone Graph Attention Network (GAT) evaluating STRING--GO support against dense and self-edge-only graph supports across 4 public Perturb-seq datasets (`Adamson_2016`, `Norman_2019`, `Replogle_K562`, `Replogle_RPE1`), evaluated on identical held-out guide conditions at 200 Highly Variable Genes (HVG) with 3 independent optimization seeds (`42`, `43`, `44`).
- **Topology Controls:** Ten degree-preserving, component-preserving random rewires (`REWIRE_01`–`REWIRE_10`).
- **Gene-Space Scaling:** Native panel evaluations across 200, 500, and 1,000 HVGs.
- **Matched External Baselines:** GEARS, scGPT, and Geneformer evaluated on the identical held-out guide conditions, output-gene panels, and Pearson-$r$ metric.

---

## 2. Environment Setup

Requirements:
- Python 3.10+
- PyTorch >= 2.0.0
- PyTorch Geometric (PyG) >= 2.3.0
- scanpy >= 1.9.0, anndata >= 0.9.0

Install dependencies via Conda or pip:

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate cbac-benchmark

# Option B: pip
pip install -r requirements.txt
pip install -e .
```

---

## 3. Repository Structure

```
├── cbac_revision/              # Core benchmark package
│   ├── graph_builders.py       # Biological & rewired graph constructors
│   ├── model.py                # Fixed-architecture GAT implementation
│   ├── runner.py               # Preflight & execution harness
│   ├── targets.py              # Canonical perturbation target mapping
│   ├── statistics.py           # Condition-level bootstrap & paired intervals
│   └── ...
├── gears_model_architecture.py # Baseline GEARS model & GNN message passing
├── run_matched_benchmark.py    # Main training & execution entry point
├── reproduce_benchmark_evaluation.py # Standalone evaluation & diagnostic verification
├── protocol.yaml               # Protocol configuration and dataset manifests
├── pyproject.toml              # Build & package specifications
├── environment.yml             # Conda environment specification
└── README.md                   # Repository documentation
```

---

## 4. Usage

### A. Verify Diagnostic Metrics & Results from Retained Predictions
Calculate GEARS vector diagnostics and summarize the existing external-model scores. Supply the prediction directory and summary table explicitly:

```bash
python reproduce_benchmark_evaluation.py --predictions raw_predictions/gears --results-csv 05_External_Baseline_Per_Fold_Results.csv
```

### B. Run Matched GAT Benchmark (Preflight Dry-Run)
To verify environment configuration, input datasets, and graph support integrity:

```bash
python run_matched_benchmark.py \
    --protocol protocol.yaml \
    --dataset Adamson_2016 \
    --data data/Adamson_2016.h5ad \
    --output results/Adamson_2016_hvg200 \
    --hvg 200 \
    --dry-run
```

### C. Full Model Training
To execute full GAT model training across seeds and support arms:

```bash
python run_matched_benchmark.py \
    --protocol protocol.yaml \
    --dataset Adamson_2016 \
    --data data/Adamson_2016.h5ad \
    --output results/Adamson_2016_hvg200 \
    --hvg 200 \
    --execute
```

---

## 5. Public Data Availability

All primary single-cell count matrices analyzed in this study are hosted in public repositories:
- **scPerturb Benchmark Archive:** Zenodo DOI [10.5281/zenodo.10044268](https://doi.org/10.5281/zenodo.10044268)
- **NCBI Gene Expression Omnibus (GEO):**
  - Adamson et al. (2016): [GSE84705](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84705)
  - Norman et al. (2019): [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344)
  - Replogle et al. (2022): [GSE198420](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198420)

---

## 6. Statistical interpretation

The primary graph-support comparison was formulated during peer-review revision.
Its positive mean predictive difference is distinct from practical biological utility.
The external-model band of +/-0.030 is a descriptive reference retained from the
preceding revision. For scale context, it is 1.23 times the pooled condition-level
SD (0.0244) and 1.91 times the median within-cell SD (0.0157). It is not an
equivalence margin or a practical-utility threshold.

GEARS variance, dimension-normalized RMS and correlation with the training-mean
effect vector describe complementary properties of standardized-delta predictions.
The evaluation utility computes these diagnostics and summarizes the supplied
predictive scores; it does not reconstruct all study results from primary data.
