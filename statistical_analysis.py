#!/usr/bin/env python
"""
Statistical Analysis for TurboGNN Benchmark Paper
- Two-way ANOVA: graph_type × hvg_scale interaction
- Pairwise Cohen's d with bootstrap 95% CI
- Paired t-tests with Bonferroni correction
- Wilcoxon signed-rank tests (non-parametric)
- Shapiro-Wilk normality tests
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results_final"
OUTPUT_DIR = Path(__file__).parent / "stats_output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = ["adamson", "norman", "replogle_k562", "replogle_rpe1"]
ALL_GRAPHS = ["string_ppi", "gene_ontology", "coexpression", "combined", "random", "no_graph"]
KEY_GRAPHS = ["string_ppi", "gene_ontology", "combined", "random", "no_graph"]
HVG_SCALES = [200, 500, 1000]
METRIC = "pearson_r"


def load_fold_data() -> pd.DataFrame:
    """Load all fold-level data into a single DataFrame."""
    rows = []
    for hvg in HVG_SCALES:
        hvg_dir = RESULTS_DIR / f"hvg{hvg}"
        for jf in sorted(hvg_dir.glob("*.json")):
            if "__f" in jf.name or jf.name.startswith("results_") or jf.name == "summary.csv":
                continue
            with open(jf) as f:
                data = json.load(f)
            ds = data["dataset"]
            gt = data["graph_type"]
            for seed_key, seed_data in data.get("seeds", {}).items():
                seed_val = int(seed_key.replace("seed_", ""))
                for fold in seed_data.get("folds", []):
                    rows.append({
                        "dataset": ds,
                        "graph_type": gt,
                        "hvg": hvg,
                        "seed": seed_val,
                        "condition": fold.get("condition", ""),
                        "pearson_r": fold["pearson_r"],
                        "spearman_rho": fold["spearman_rho"],
                        "mse": fold["mse"],
                        "jaccard": fold["jaccard"],
                    })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} fold-level observations")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Graphs: {sorted(df['graph_type'].unique())}")
    print(f"  HVG scales: {sorted(df['hvg'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    return df


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / (np.std(diff, ddof=0) + 1e-10)


def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for Cohen's d."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    rng = np.random.RandomState(42)
    boot_ds = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_diff = diff[idx]
        d = np.mean(boot_diff) / (np.std(boot_diff, ddof=0) + 1e-10)
        boot_ds.append(d)
    lo = np.percentile(boot_ds, 100 * alpha / 2)
    hi = np.percentile(boot_ds, 100 * (1 - alpha / 2))
    return lo, hi


def run_analysis(df: pd.DataFrame):
    """Run all statistical analyses."""
    results = []

    # ================================================================
    # 1. SHAPIRO-WILK NORMALITY TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print("1. SHAPIRO-WILK NORMALITY TESTS")
    print("=" * 70)
    normality_rows = []
    for ds in DATASETS:
        for hvg in HVG_SCALES:
            graphs_available = sorted(df[(df.dataset == ds) & (df.hvg == hvg)].graph_type.unique())
            for gt in graphs_available:
                vals = df[(df.dataset == ds) & (df.hvg == hvg) & (df.graph_type == gt)][METRIC].values
                if len(vals) >= 8:
                    stat, p = stats.shapiro(vals)
                    normality_rows.append({
                        "dataset": ds, "hvg": hvg, "graph_type": gt,
                        "W": round(stat, 4), "p": round(p, 6),
                        "normal": "Yes" if p > 0.05 else "No",
                        "n": len(vals),
                    })
    norm_df = pd.DataFrame(normality_rows)
    n_normal = (norm_df["normal"] == "Yes").sum()
    n_total = len(norm_df)
    print(f"  {n_normal}/{n_total} distributions pass normality (p>0.05)")
    norm_df.to_csv(OUTPUT_DIR / "shapiro_wilk.csv", index=False)

    # ================================================================
    # 2. TWO-WAY ANOVA: graph_type × hvg_scale INTERACTION
    # ================================================================
    print("\n" + "=" * 70)
    print("2. TWO-WAY ANOVA: graph_type × hvg_scale INTERACTION")
    print("=" * 70)
    anova_rows = []
    for ds in DATASETS:
        sub = df[df.dataset == ds].copy()
        # Use only key graphs present in all HVG scales
        graphs_in_all = None
        for hvg in HVG_SCALES:
            g_set = set(sub[sub.hvg == hvg].graph_type.unique())
            graphs_in_all = g_set if graphs_in_all is None else graphs_in_all & g_set
        if not graphs_in_all:
            continue
        sub = sub[sub.graph_type.isin(graphs_in_all)]

        # One-way ANOVA per HVG scale: does graph_type matter?
        for hvg in HVG_SCALES:
            hvg_sub = sub[sub.hvg == hvg]
            groups = [hvg_sub[hvg_sub.graph_type == g][METRIC].values for g in sorted(graphs_in_all)]
            if all(len(g) > 0 for g in groups):
                F, p = stats.f_oneway(*groups)
                anova_rows.append({
                    "dataset": ds, "hvg": hvg, "test": "one-way ANOVA (graph_type)",
                    "F": round(F, 4), "p": round(p, 6),
                    "significant": "Yes" if p < 0.05 else "No",
                    "n_groups": len(groups),
                })
                print(f"  {ds} HVG={hvg}: F={F:.4f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

        # Kruskal-Wallis (non-parametric alternative)
        for hvg in HVG_SCALES:
            hvg_sub = sub[sub.hvg == hvg]
            groups = [hvg_sub[hvg_sub.graph_type == g][METRIC].values for g in sorted(graphs_in_all)]
            if all(len(g) > 0 for g in groups):
                H, p = stats.kruskal(*groups)
                anova_rows.append({
                    "dataset": ds, "hvg": hvg, "test": "Kruskal-Wallis",
                    "F": round(H, 4), "p": round(p, 6),
                    "significant": "Yes" if p < 0.05 else "No",
                    "n_groups": len(groups),
                })

    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(OUTPUT_DIR / "anova_results.csv", index=False)

    # ================================================================
    # 3. PAIRWISE COMPARISONS: KG vs no_graph, KG vs random
    # ================================================================
    print("\n" + "=" * 70)
    print("3. PAIRWISE COHEN'S d + PAIRED t-TEST + WILCOXON")
    print("=" * 70)
    pairwise_rows = []
    comparisons = [
        ("string_ppi", "no_graph"),
        ("gene_ontology", "no_graph"),
        ("combined", "no_graph"),
        ("random", "no_graph"),
        ("string_ppi", "random"),
        ("combined", "random"),
    ]

    for ds in DATASETS:
        for hvg in HVG_SCALES:
            sub = df[(df.dataset == ds) & (df.hvg == hvg)]
            for g1, g2 in comparisons:
                v1 = sub[sub.graph_type == g1][METRIC].values
                v2 = sub[sub.graph_type == g2][METRIC].values
                if len(v1) == 0 or len(v2) == 0:
                    continue
                # Use min length for pairing
                n = min(len(v1), len(v2))
                v1, v2 = v1[:n], v2[:n]

                d = cohens_d(v1, v2)
                ci_lo, ci_hi = bootstrap_ci(v1, v2)

                # Paired t-test
                t_stat, t_p = stats.ttest_rel(v1, v2)

                # Wilcoxon signed-rank
                try:
                    w_stat, w_p = stats.wilcoxon(v1 - v2)
                except ValueError:
                    w_stat, w_p = np.nan, np.nan

                mean_diff = np.mean(v1 - v2)

                pairwise_rows.append({
                    "dataset": ds, "hvg": hvg,
                    "comparison": f"{g1} vs {g2}",
                    "mean_diff": round(mean_diff, 6),
                    "cohens_d": round(d, 4),
                    "ci_lo": round(ci_lo, 4),
                    "ci_hi": round(ci_hi, 4),
                    "t_stat": round(t_stat, 4),
                    "t_p": round(t_p, 8),
                    "wilcoxon_p": round(w_p, 8) if not np.isnan(w_p) else np.nan,
                    "n_pairs": n,
                })

    pw_df = pd.DataFrame(pairwise_rows)

    # Bonferroni correction
    n_tests = len(pw_df)
    pw_df["t_p_bonf"] = np.minimum(pw_df["t_p"] * n_tests, 1.0).round(8)
    # BH FDR
    from scipy.stats import rankdata
    ranks = rankdata(pw_df["t_p"])
    pw_df["t_p_fdr"] = np.minimum(pw_df["t_p"] * n_tests / ranks, 1.0).round(8)

    pw_df.to_csv(OUTPUT_DIR / "pairwise_comparisons.csv", index=False)

    # Print key comparisons
    print("\nKey comparisons (KG vs no_graph):")
    for _, row in pw_df[pw_df.comparison.str.contains("vs no_graph")].iterrows():
        sig = "***" if row.t_p_bonf < 0.001 else "**" if row.t_p_bonf < 0.01 else "*" if row.t_p_bonf < 0.05 else "ns"
        print(f"  {row.dataset:15s} HVG={row.hvg:4d} {row.comparison:25s}  d={row.cohens_d:+.4f} [{row.ci_lo:+.4f},{row.ci_hi:+.4f}]  Δ={row.mean_diff:+.6f}  {sig}")

    print("\nKey comparisons (KG vs random):")
    for _, row in pw_df[pw_df.comparison.str.contains("vs random")].iterrows():
        sig = "***" if row.t_p_bonf < 0.001 else "**" if row.t_p_bonf < 0.01 else "*" if row.t_p_bonf < 0.05 else "ns"
        print(f"  {row.dataset:15s} HVG={row.hvg:4d} {row.comparison:25s}  d={row.cohens_d:+.4f} [{row.ci_lo:+.4f},{row.ci_hi:+.4f}]  Δ={row.mean_diff:+.6f}  {sig}")

    # ================================================================
    # 4. HVG SCALE EFFECT: Does KG advantage grow with HVG?
    # ================================================================
    print("\n" + "=" * 70)
    print("4. HVG SCALE EFFECT: KG advantage across scales")
    print("=" * 70)
    scale_rows = []
    for ds in DATASETS:
        for g in ["string_ppi", "combined"]:
            deltas = {}
            for hvg in HVG_SCALES:
                sub = df[(df.dataset == ds) & (df.hvg == hvg)]
                kg = sub[sub.graph_type == g][METRIC].values
                ng = sub[sub.graph_type == "no_graph"][METRIC].values
                if len(kg) > 0 and len(ng) > 0:
                    n = min(len(kg), len(ng))
                    delta = np.mean(kg[:n]) - np.mean(ng[:n])
                    d = cohens_d(kg[:n], ng[:n])
                    deltas[hvg] = (delta, d)
                    scale_rows.append({
                        "dataset": ds, "graph": g, "hvg": hvg,
                        "delta_pearson": round(delta, 6),
                        "cohens_d": round(d, 4),
                    })
            if deltas:
                print(f"  {ds:15s} {g:15s}: ", end="")
                for hvg in HVG_SCALES:
                    if hvg in deltas:
                        delta, d = deltas[hvg]
                        print(f"HVG{hvg}={delta:+.4f}(d={d:+.3f}) ", end="")
                print()

    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(OUTPUT_DIR / "hvg_scale_effect.csv", index=False)

    # ================================================================
    # 5. RANDOM GRAPH ANOMALY ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("5. RANDOM GRAPH ANOMALY (MSE explosion)")
    print("=" * 70)
    for ds in DATASETS:
        for hvg in HVG_SCALES:
            sub = df[(df.dataset == ds) & (df.hvg == hvg)]
            for gt in sorted(sub.graph_type.unique()):
                mse_vals = sub[sub.graph_type == gt]["mse"].values
                if len(mse_vals) > 0:
                    mean_mse = np.mean(mse_vals)
                    if mean_mse > 0.5:
                        print(f"  WARNING: {ds} HVG={hvg} {gt}: MSE={mean_mse:.4f} (median={np.median(mse_vals):.4f}, max={np.max(mse_vals):.2f})")

    # ================================================================
    # 6. SUMMARY TABLE FOR PAPER
    # ================================================================
    print("\n" + "=" * 70)
    print("6. SUMMARY TABLE FOR PAPER (Pearson r, mean ± std)")
    print("=" * 70)
    summary_rows = []
    for ds in DATASETS:
        for hvg in HVG_SCALES:
            row = {"dataset": ds, "hvg": hvg}
            sub = df[(df.dataset == ds) & (df.hvg == hvg)]
            for gt in ALL_GRAPHS:
                vals = sub[sub.graph_type == gt][METRIC].values
                if len(vals) > 0:
                    row[gt] = f"{np.mean(vals):.3f}±{np.std(vals, ddof=0):.3f}"
                else:
                    row[gt] = "—"
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "paper_summary_table.csv", index=False)

    print(f"\n=== All results saved to {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    df = load_fold_data()
    run_analysis(df)
