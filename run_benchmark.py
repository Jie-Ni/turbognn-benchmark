"""
Benchmark Pipeline: Graph Topology Ablation for TurboGNN

Tests 6 graph variants across 4 datasets with LOPO cross-validation:
  1. STRING PPI graph (score > 400)
  2. Gene Ontology graph (shared GO biological process terms)
  3. Co-expression graph (|Pearson r| > 0.3)
  4. Combined graph (PPI + GO + co-expression union)
  5. Random graph (Barabasi-Albert, matched average degree)
  6. No graph (SimpleTransformer baseline)

Quick mode: 1 seed x 20 folds.  Full mode: 5 seeds x 50 folds.

Usage:
    python run_benchmark.py                  # quick mode
    python run_benchmark.py --full           # full mode
    python run_benchmark.py --datasets norman replogle_k562
"""

import sys
import os
import argparse
import json
import logging
import signal
import time
import gzip
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Import models from turbognn_v2_models.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turbognn_v2_models import TurboGNN, SimpleTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = Path(os.path.expanduser("~"))
_DATA_BASE = Path(os.environ.get("TURBOGNN_DATA_DIR", str(HOME / "TurboGNN" / "data")))
DATA_DIR_PROCESSED = _DATA_BASE / "processed"
DATA_DIR_SCPERTURB = _DATA_BASE / "scperturb"
RESULTS_DIR = Path(__file__).resolve().parent / "results_benchmark"

NUM_HVG_DEFAULT = 200
HIDDEN_DIM = 128
NUM_HEADS = 8
LR = 5e-4
WEIGHT_DECAY = 1e-4
LOPO_EPOCHS = 50
DROPOUT = 0.1

COEXPR_THRESHOLD = 0.3
STRING_SCORE_THRESHOLD = 400
GO_ANNOTATION_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"

STRING_API_URL = "https://string-db.org/api"

DATASET_PATHS = {
    "norman": DATA_DIR_PROCESSED / "norman_mapped.h5ad",
    "replogle_k562": DATA_DIR_SCPERTURB / "replogle_k562.h5ad",
    "replogle_rpe1": DATA_DIR_SCPERTURB / "replogle_rpe1.h5ad",
    "adamson": DATA_DIR_SCPERTURB / "adamson.h5ad",
}

GRAPH_TYPES = [
    "string_ppi",
    "gene_ontology",
    "coexpression",
    "combined",
    "random",
    "no_graph",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_partial_results: Dict[str, dict] = {}
_results_dir: Path = RESULTS_DIR


def _save_partial(tag: str = "partial") -> None:
    """Persist whatever results we have so far."""
    if not _partial_results:
        return
    _results_dir.mkdir(parents=True, exist_ok=True)
    out = _results_dir / f"results_{tag}.json"
    with open(str(out), "w") as fh:
        json.dump(_partial_results, fh, indent=2)
    logger.info("Saved %d result entries to %s", len(_partial_results), out)


def _sigterm_handler(signum, frame):
    logger.warning("SIGTERM received — saving partial results before exit.")
    _save_partial("sigterm")
    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm_handler)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, num_hvg: int = NUM_HVG_DEFAULT) -> sc.AnnData:
    """Load and preprocess one dataset: normalize, log1p, select HVGs."""
    path = DATASET_PATHS[name]
    logger.info("Loading dataset '%s' from %s (num_hvg=%d)", name, path, num_hvg)
    adata = sc.read_h5ad(str(path))

    # Normalise if raw counts
    x_sample = adata.X[:10].toarray() if hasattr(adata.X, "toarray") else adata.X[:10]
    if x_sample.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def extract_perturbations(adata: sc.AnnData) -> Tuple[str, List[str], str]:
    """Identify perturbation column, control label, and condition list."""
    candidates = ["gene", "condition", "perturbation", "guide_ids", "perturbations"]
    pert_col = next((c for c in candidates if c in adata.obs.columns), None)
    if pert_col is None:
        raise ValueError(f"No perturbation column found. Available: {list(adata.obs.columns)}")

    unique = adata.obs[pert_col].unique().tolist()
    ctrl_candidates = ["ctrl", "control", "non-targeting", "NT", ""]
    ctrl_label = next((c for c in ctrl_candidates if c in unique), None)
    if ctrl_label is None:
        ctrl_label = adata.obs[pert_col].value_counts().idxmax()

    conditions = [c for c in unique if c != ctrl_label]
    return pert_col, conditions, ctrl_label


def prepare_dataset(name: str, num_hvg: int = NUM_HVG_DEFAULT) -> dict:
    """Full preprocessing: returns everything needed for LOPO evaluation."""
    adata = load_dataset(name, num_hvg=num_hvg)
    gene_list = np.array(adata.var_names)
    num_genes = len(gene_list)

    X_np = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float32)
    gene_mean = X_np.mean(axis=0, keepdims=True)
    gene_std = X_np.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_np - gene_mean) / gene_std

    pert_col, conditions, ctrl_label = extract_perturbations(adata)

    ctrl_mask = adata.obs[pert_col] == ctrl_label
    ctrl_np = X_norm[ctrl_mask.values].mean(axis=0)
    ctrl_tensor = torch.tensor(ctrl_np, dtype=torch.float32).to(DEVICE)

    cond_profiles: Dict[str, np.ndarray] = {}
    for cond in conditions:
        mask = adata.obs[pert_col] == cond
        if mask.sum() >= 2:
            cond_profiles[cond] = X_norm[mask.values].mean(axis=0)

    valid_conds = list(cond_profiles.keys())
    logger.info("Dataset '%s': %d genes, %d valid conditions", name, num_genes, len(valid_conds))

    return {
        "adata": adata,
        "gene_list": gene_list,
        "num_genes": num_genes,
        "X_norm": X_norm,
        "ctrl_np": ctrl_np,
        "ctrl_tensor": ctrl_tensor,
        "cond_profiles": cond_profiles,
        "valid_conds": valid_conds,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _gene_name_to_string_ids(gene_names: List[str], species: int = 9606) -> Dict[str, str]:
    """Map gene symbols to STRING identifiers via the API (batched)."""
    mapping: Dict[str, str] = {}
    batch_size = 200
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i : i + batch_size]
        identifiers = "%0d".join(batch)
        url = (
            f"{STRING_API_URL}/json/get_string_ids"
            f"?identifiers={identifiers}&species={species}&limit=1"
        )
        try:
            req = Request(url, headers={"User-Agent": "TurboGNN-benchmark/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            for entry in data:
                query = entry.get("queryItem", "")
                string_id = entry.get("stringId", "")
                if query and string_id:
                    mapping[query] = string_id
        except (URLError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("STRING ID mapping batch %d failed: %s", i, exc)
    logger.info("STRING ID mapping: %d / %d genes resolved", len(mapping), len(gene_names))
    return mapping


def build_string_ppi_graph(
    gene_list: np.ndarray, species: int = 9606
) -> Optional[torch.Tensor]:
    """Fetch STRING PPI edges (combined_score > 400) for gene_list."""
    num_genes = len(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Map gene names -> STRING IDs
    id_map = _gene_name_to_string_ids(list(gene_list), species)
    if len(id_map) < 10:
        logger.warning("Too few STRING IDs resolved (%d). Skipping PPI graph.", len(id_map))
        return None

    string_to_gene: Dict[str, str] = {v: k for k, v in id_map.items()}
    string_ids = list(id_map.values())

    # Fetch interactions in batches
    rows, cols = [], []
    batch_size = 200
    for i in range(0, len(string_ids), batch_size):
        batch = string_ids[i : i + batch_size]
        identifiers = "%0d".join(batch)
        url = (
            f"{STRING_API_URL}/json/network"
            f"?identifiers={identifiers}&species={species}"
            f"&required_score={STRING_SCORE_THRESHOLD}"
        )
        try:
            req = Request(url, headers={"User-Agent": "TurboGNN-benchmark/1.0"})
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            for edge in data:
                g1 = string_to_gene.get(edge.get("stringId_A", ""))
                g2 = string_to_gene.get(edge.get("stringId_B", ""))
                if g1 and g2 and g1 in gene_to_idx and g2 in gene_to_idx:
                    i1, i2 = gene_to_idx[g1], gene_to_idx[g2]
                    rows.extend([i1, i2])
                    cols.extend([i2, i1])
        except (URLError, json.JSONDecodeError) as exc:
            logger.warning("STRING network batch %d failed: %s", i, exc)

    if not rows:
        logger.warning("No STRING PPI edges obtained.")
        return None

    # Add self-loops
    self_loops = list(range(num_genes))
    rows.extend(self_loops)
    cols.extend(self_loops)

    edge_index = torch.tensor([rows, cols], dtype=torch.long).to(DEVICE)
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    logger.info("STRING PPI graph: %d edges (including self-loops)", edge_index.shape[1])
    return edge_index


def _fetch_go_annotations(gene_set: Set[str]) -> Dict[str, Set[str]]:
    """Download GO biological-process annotations and return gene -> {GO term} map."""
    gene_to_go: Dict[str, Set[str]] = defaultdict(set)
    cache_path = Path(__file__).resolve().parent / "goa_human.gaf.gz"

    # Download if not cached
    if not cache_path.exists():
        logger.info("Downloading GO annotations from %s ...", GO_ANNOTATION_URL)
        try:
            req = Request(GO_ANNOTATION_URL, headers={"User-Agent": "TurboGNN-benchmark/1.0"})
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(str(cache_path), "wb") as fh:
                fh.write(data)
            logger.info("GO annotations cached to %s", cache_path)
        except URLError as exc:
            logger.error("Failed to download GO annotations: %s", exc)
            return gene_to_go

    # Parse GAF (Gene Association File)
    with gzip.open(str(cache_path), "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 15:
                continue
            gene_symbol = fields[2]  # DB Object Symbol
            go_id = fields[4]        # GO ID
            aspect = fields[8]       # P = biological process, F = molecular function, C = cellular component
            if aspect == "P" and gene_symbol in gene_set:
                gene_to_go[gene_symbol].add(go_id)

    logger.info("GO annotations: %d genes with BP terms (of %d queried)", len(gene_to_go), len(gene_set))
    return gene_to_go


def build_go_graph(gene_list: np.ndarray) -> Optional[torch.Tensor]:
    """Connect genes that share at least 1 GO biological process term."""
    num_genes = len(gene_list)
    gene_set = set(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    gene_to_go = _fetch_go_annotations(gene_set)
    if len(gene_to_go) < 10:
        logger.warning("Too few genes with GO annotations (%d). Skipping GO graph.", len(gene_to_go))
        return None

    # Invert: GO term -> list of gene indices
    go_to_genes: Dict[str, List[int]] = defaultdict(list)
    for gene, terms in gene_to_go.items():
        if gene in gene_to_idx:
            idx = gene_to_idx[gene]
            for term in terms:
                go_to_genes[term].append(idx)

    # Build edges: genes sharing any GO BP term
    rows, cols = [], []
    for term, indices in go_to_genes.items():
        if len(indices) > 100:
            # Skip overly broad terms to avoid dense cliques
            continue
        for a in indices:
            for b in indices:
                if a != b:
                    rows.append(a)
                    cols.append(b)

    if not rows:
        logger.warning("No GO graph edges obtained.")
        return None

    # Self-loops
    self_loops = list(range(num_genes))
    rows.extend(self_loops)
    cols.extend(self_loops)

    edge_index = torch.tensor([rows, cols], dtype=torch.long).to(DEVICE)
    edge_index = torch.unique(edge_index, dim=1)
    logger.info("GO graph: %d edges (including self-loops)", edge_index.shape[1])
    return edge_index


def build_coexpression_graph(
    X_norm: np.ndarray, gene_list: np.ndarray, threshold: float = COEXPR_THRESHOLD
) -> Optional[torch.Tensor]:
    """Connect genes with |Pearson correlation| > threshold across cells."""
    num_genes = len(gene_list)
    logger.info("Computing co-expression graph (threshold=%.2f) ...", threshold)

    corr = np.corrcoef(X_norm.T)  # [num_genes, num_genes]
    np.fill_diagonal(corr, 0.0)
    r, c = np.where(np.abs(corr) > threshold)

    if len(r) == 0:
        logger.warning("No co-expression edges at threshold %.2f.", threshold)
        return None

    # Self-loops
    self_loops = np.arange(num_genes)
    r = np.concatenate([r, self_loops])
    c = np.concatenate([c, self_loops])

    edge_index = torch.tensor(
        np.stack([r, c]), dtype=torch.long
    ).to(DEVICE)
    edge_index = torch.unique(edge_index, dim=1)
    logger.info("Co-expression graph: %d edges (including self-loops)", edge_index.shape[1])
    return edge_index


def build_combined_graph(
    ppi_ei: Optional[torch.Tensor],
    go_ei: Optional[torch.Tensor],
    coexpr_ei: Optional[torch.Tensor],
    num_genes: int,
) -> Optional[torch.Tensor]:
    """Union of PPI + GO + co-expression edges."""
    parts = [ei for ei in [ppi_ei, go_ei, coexpr_ei] if ei is not None]
    if not parts:
        logger.warning("No component graphs available for combined graph.")
        return None

    combined = torch.cat(parts, dim=1)
    combined = torch.unique(combined, dim=1)
    logger.info("Combined graph: %d edges (union of %d components)", combined.shape[1], len(parts))
    return combined


def build_random_graph(
    num_genes: int, target_avg_degree: float, seed: int = 42
) -> torch.Tensor:
    """Barabasi-Albert random graph matching target average degree."""
    m = max(1, int(target_avg_degree / 2))
    G = nx.barabasi_albert_graph(num_genes, m, seed=seed)
    adj = sp.coo_matrix(nx.to_scipy_sparse_array(G, format="coo"))

    rows = np.concatenate([adj.row, np.arange(num_genes)])
    cols = np.concatenate([adj.col, np.arange(num_genes)])

    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long).to(DEVICE)
    edge_index = torch.unique(edge_index, dim=1)
    logger.info("Random (BA) graph: %d edges (m=%d)", edge_index.shape[1], m)
    return edge_index


def build_all_graphs(
    gene_list: np.ndarray, X_norm: np.ndarray, num_genes: int
) -> Dict[str, Optional[torch.Tensor]]:
    """Construct all 6 graph variants for a dataset."""
    graphs: Dict[str, Optional[torch.Tensor]] = {}

    # 1. STRING PPI
    logger.info("--- Building STRING PPI graph ---")
    graphs["string_ppi"] = build_string_ppi_graph(gene_list)

    # 2. Gene Ontology
    logger.info("--- Building Gene Ontology graph ---")
    graphs["gene_ontology"] = build_go_graph(gene_list)

    # 3. Co-expression
    logger.info("--- Building Co-expression graph ---")
    graphs["coexpression"] = build_coexpression_graph(X_norm, gene_list)

    # 4. Combined
    logger.info("--- Building Combined graph ---")
    graphs["combined"] = build_combined_graph(
        graphs["string_ppi"], graphs["gene_ontology"], graphs["coexpression"], num_genes
    )

    # 5. Random (match combined average degree)
    logger.info("--- Building Random graph ---")
    ref = graphs["combined"]
    if ref is not None:
        avg_deg = ref.shape[1] / num_genes
    else:
        avg_deg = 10.0
    graphs["random"] = build_random_graph(num_genes, avg_deg)

    # 6. No graph
    graphs["no_graph"] = None

    return graphs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def make_mask(gene_list: np.ndarray, pert_str: str) -> Optional[torch.Tensor]:
    """Create boolean perturbation mask. True = knocked-out gene."""
    mask = torch.zeros(len(gene_list), dtype=torch.bool)
    found = False
    for g in pert_str.split("+"):
        idx = np.where(gene_list == g.strip())[0]
        if len(idx) > 0:
            mask[idx[0]] = True
            found = True
    return mask if found else None


def jaccard_top_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    """Jaccard similarity of top-k differentially expressed genes."""
    top_true = set(np.argsort(np.abs(y_true))[-k:])
    top_pred = set(np.argsort(np.abs(y_pred))[-k:])
    union = top_true | top_pred
    if not union:
        return 0.0
    return len(top_true & top_pred) / len(union)


def compute_fold_metrics(
    delta_true: np.ndarray, delta_pred: np.ndarray, condition: str
) -> dict:
    """Compute all metrics for one LOPO fold."""
    pr = sr = 0.0
    if np.std(delta_true) > 1e-8 and np.std(delta_pred) > 1e-8:
        pr, _ = pearsonr(delta_true, delta_pred)
        sr, _ = spearmanr(delta_true, delta_pred)
    mse = float(np.mean((delta_true - delta_pred) ** 2))
    jac = jaccard_top_k(delta_true, delta_pred)
    return {
        "condition": condition,
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "mse": mse,
        "jaccard": jac,
    }


# ---------------------------------------------------------------------------
# LOPO evaluation
# ---------------------------------------------------------------------------

def run_lopo(
    model_cls,
    model_kwargs: dict,
    ctrl_tensor: torch.Tensor,
    ctrl_np: np.ndarray,
    gene_list: np.ndarray,
    cond_profiles: Dict[str, np.ndarray],
    valid_conds: List[str],
    is_gnn: bool,
    seed: int,
    max_folds: int,
    fold_start: int = 0,
    fold_end: Optional[int] = None,
) -> List[dict]:
    """Run LOPO cross-validation for one (model, graph) configuration.

    fold_start/fold_end: run only folds[fold_start:fold_end] for parallel splitting.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    criterion = nn.MSELoss()
    n_folds = min(max_folds, len(valid_conds))
    if fold_end is None:
        fold_end = n_folds
    fold_end = min(fold_end, n_folds)
    fold_start = min(fold_start, fold_end)
    results: List[dict] = []

    for fi, held_out in enumerate(valid_conds[fold_start:fold_end], start=fold_start):
        train_conds = [c for c in valid_conds if c != held_out]
        ho_target = cond_profiles[held_out]
        ho_mask = make_mask(gene_list, held_out)

        model = model_cls(**model_kwargs).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LOPO_EPOCHS)

        # --- Training ---
        for ep in range(LOPO_EPOCHS):
            model.train()
            np.random.shuffle(train_conds)
            for tc in train_conds:
                tm = make_mask(gene_list, tc)
                tgt = torch.tensor(cond_profiles[tc], dtype=torch.float32).to(DEVICE)
                opt.zero_grad()
                if is_gnn:
                    pred = model(
                        ctrl_tensor,
                        perturbation_mask=tm.to(DEVICE) if tm is not None else None,
                    )
                else:
                    pred = model(
                        ctrl_tensor,
                        mask=tm.to(DEVICE) if tm is not None else None,
                    )
                loss = criterion(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

        # --- Inference ---
        model.eval()
        with torch.no_grad():
            if is_gnn:
                pred = model(
                    ctrl_tensor,
                    perturbation_mask=ho_mask.to(DEVICE) if ho_mask is not None else None,
                ).cpu().numpy()
            else:
                pred = model(
                    ctrl_tensor,
                    mask=ho_mask.to(DEVICE) if ho_mask is not None else None,
                ).cpu().numpy()

        delta_true = ho_target - ctrl_np
        delta_pred = pred - ctrl_np
        metrics = compute_fold_metrics(delta_true, delta_pred, held_out)
        results.append(metrics)

        if (fi + 1) % 5 == 0:
            recent = results[-5:]
            avg_p = np.mean([r["pearson_r"] for r in recent])
            logger.info(
                "  Fold %d/%d done (recent-5 Pearson=%.4f)", fi + 1, n_folds, avg_p
            )

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_combination_result(
    dataset: str,
    graph_type: str,
    seed_results: Dict[int, List[dict]],
    results_dir: Path,
    fold_tag: str = "",
) -> None:
    """Save fold-level results for one (dataset, graph_type) combination."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset}__{graph_type}{fold_tag}.json"
    out_path = results_dir / fname

    # Aggregate across seeds
    all_pearson, all_spearman, all_jaccard, all_mse = [], [], [], []
    seed_summaries = {}
    for seed, folds in seed_results.items():
        ps = [r["pearson_r"] for r in folds]
        ss = [r["spearman_rho"] for r in folds]
        js = [r["jaccard"] for r in folds]
        ms = [r["mse"] for r in folds]
        all_pearson.extend(ps)
        all_spearman.extend(ss)
        all_jaccard.extend(js)
        all_mse.extend(ms)
        seed_summaries[f"seed_{seed}"] = {
            "pearson_mean": float(np.mean(ps)),
            "pearson_std": float(np.std(ps)),
            "spearman_mean": float(np.mean(ss)),
            "jaccard_mean": float(np.mean(js)),
            "mse_mean": float(np.mean(ms)),
            "n_folds": len(folds),
            "folds": folds,
        }

    payload = {
        "dataset": dataset,
        "graph_type": graph_type,
        "overall": {
            "pearson_mean": float(np.mean(all_pearson)),
            "pearson_std": float(np.std(all_pearson)),
            "spearman_mean": float(np.mean(all_spearman)),
            "spearman_std": float(np.std(all_spearman)),
            "jaccard_mean": float(np.mean(all_jaccard)),
            "jaccard_std": float(np.std(all_jaccard)),
            "mse_mean": float(np.mean(all_mse)),
            "mse_std": float(np.std(all_mse)),
        },
        "seeds": seed_summaries,
    }

    with open(str(out_path), "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved results: %s", out_path)

    # Update global tracker for SIGTERM handler
    key = f"{dataset}__{graph_type}"
    _partial_results[key] = payload["overall"]


def build_summary_csv(results_dir: Path) -> pd.DataFrame:
    """Aggregate all per-combination JSONs into a summary CSV."""
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name.startswith("results_"):
            continue
        with open(str(jf)) as fh:
            data = json.load(fh)
        ov = data["overall"]
        rows.append({
            "dataset": data["dataset"],
            "graph_type": data["graph_type"],
            "pearson_mean": round(ov["pearson_mean"], 4),
            "pearson_std": round(ov["pearson_std"], 4),
            "spearman_mean": round(ov["spearman_mean"], 4),
            "spearman_std": round(ov["spearman_std"], 4),
            "jaccard_mean": round(ov["jaccard_mean"], 4),
            "jaccard_std": round(ov["jaccard_std"], 4),
            "mse_mean": round(ov["mse_mean"], 6),
            "mse_std": round(ov["mse_std"], 6),
        })

    df = pd.DataFrame(rows)
    csv_path = results_dir / "summary.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Summary table saved to %s", csv_path)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboGNN Graph Topology Benchmark")
    parser.add_argument(
        "--full", action="store_true",
        help="Full mode: 5 seeds x 50 folds (default: 1 seed x 20 folds)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
        choices=list(DATASET_PATHS.keys()),
        help="Datasets to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--graph-types", nargs="+", default=GRAPH_TYPES,
        choices=GRAPH_TYPES,
        help="Graph types to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--num-hvg", type=int, default=NUM_HVG_DEFAULT,
        help=f"Number of highly variable genes (default: {NUM_HVG_DEFAULT})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run a single specific seed (overrides --full seed loop)",
    )
    parser.add_argument(
        "--fold-start", type=int, default=0,
        help="Start fold index (for parallel fold splitting)",
    )
    parser.add_argument(
        "--fold-end", type=int, default=None,
        help="End fold index exclusive (for parallel fold splitting)",
    )
    args = parser.parse_args()

    global _results_dir
    num_hvg = args.num_hvg

    if args.results_dir:
        _results_dir = Path(args.results_dir)
    else:
        _results_dir = RESULTS_DIR / f"hvg{num_hvg}"
    _results_dir.mkdir(parents=True, exist_ok=True)

    # Seed configuration
    max_folds = 50 if args.full else 20
    if args.seed is not None:
        seeds = [args.seed]
        mode_str = f"SINGLE-SEED (seed={args.seed}, {max_folds} folds, hvg={num_hvg})"
    elif args.full:
        seeds = [42, 43, 44, 45, 46]
        mode_str = f"FULL (5 seeds x {max_folds} folds, hvg={num_hvg})"
    else:
        seeds = [42]
        mode_str = f"QUICK (1 seed x {max_folds} folds, hvg={num_hvg})"

    logger.info("=" * 70)
    logger.info("TurboGNN Graph Topology Benchmark — %s", mode_str)
    logger.info("Device: %s", DEVICE)
    logger.info("Datasets: %s", args.datasets)
    logger.info("Graph types: %s", args.graph_types)
    logger.info("Seeds: %s", seeds)
    logger.info("HVG: %d", num_hvg)
    logger.info("Results dir: %s", _results_dir)
    logger.info("=" * 70)

    t0 = time.time()

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("DATASET: %s", ds_name)
        logger.info("=" * 60)

        # Check dataset exists
        if not DATASET_PATHS[ds_name].exists():
            logger.error("Dataset file not found: %s — skipping.", DATASET_PATHS[ds_name])
            continue

        # Prepare data
        ds = prepare_dataset(ds_name, num_hvg=num_hvg)

        # Build all graphs for this dataset
        graphs = build_all_graphs(ds["gene_list"], ds["X_norm"], ds["num_genes"])

        for gt in args.graph_types:
            logger.info("-" * 50)
            logger.info("Graph type: %s", gt)
            logger.info("-" * 50)

            edge_index = graphs.get(gt)
            is_gnn = gt != "no_graph"

            # Skip if graph construction failed (but not for no_graph)
            if is_gnn and edge_index is None:
                logger.warning(
                    "Graph '%s' unavailable for dataset '%s'. Skipping.", gt, ds_name
                )
                continue

            # Configure model
            if is_gnn:
                model_cls = TurboGNN
                model_kwargs = {
                    "num_genes": ds["num_genes"],
                    "edge_index": edge_index,
                    "hidden_dim": HIDDEN_DIM,
                    "num_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                }
            else:
                model_cls = SimpleTransformer
                model_kwargs = {
                    "num_genes": ds["num_genes"],
                    "d_model": HIDDEN_DIM,
                    "nhead": 4,
                    "num_layers": 2,
                }

            # Run across seeds
            seed_results: Dict[int, List[dict]] = {}
            for seed_idx, seed in enumerate(seeds):
                logger.info("  Seed %d/%d (seed=%d)", seed_idx + 1, len(seeds), seed)

                fold_results = run_lopo(
                    model_cls=model_cls,
                    model_kwargs=model_kwargs,
                    ctrl_tensor=ds["ctrl_tensor"],
                    ctrl_np=ds["ctrl_np"],
                    gene_list=ds["gene_list"],
                    cond_profiles=ds["cond_profiles"],
                    valid_conds=ds["valid_conds"],
                    is_gnn=is_gnn,
                    seed=seed,
                    max_folds=max_folds,
                    fold_start=args.fold_start,
                    fold_end=args.fold_end,
                )
                seed_results[seed] = fold_results

                avg_p = np.mean([r["pearson_r"] for r in fold_results])
                avg_j = np.mean([r["jaccard"] for r in fold_results])
                logger.info(
                    "  Seed %d done: Pearson=%.4f, Jaccard=%.4f", seed, avg_p, avg_j
                )

            # Incremental save after each (dataset, graph_type)
            fold_tag = ""
            if args.fold_start > 0 or args.fold_end is not None:
                fe = args.fold_end if args.fold_end is not None else max_folds
                seed_str = f"s{seeds[0]}" if len(seeds) == 1 else ""
                fold_tag = f"__{seed_str}f{args.fold_start}-{fe}"
            save_combination_result(ds_name, gt, seed_results, _results_dir, fold_tag)

    # Final summary
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("All experiments complete. Total time: %.1f min", elapsed / 60)
    logger.info("=" * 70)

    df = build_summary_csv(_results_dir)
    if not df.empty:
        logger.info("\n%s", df.to_string(index=False))

    _save_partial("final")
    logger.info("Done. Results in %s", _results_dir)


if __name__ == "__main__":
    main()
