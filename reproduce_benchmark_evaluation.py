#!/usr/bin/env python
"""
Check GEARS output diagnostics and external-model summary statistics.

Manuscript: CBAC-D-26-03802R1, Computational Biology and Chemistry.

GEARS y_pred is a final standardized-delta array; its saved reference is the mean
training-condition effect vector. The diagnostics use gene-axis sample variance
(ddof=1), dimension-normalized RMS and Pearson correlation with that reference.
The script compares scale-level and overall diagnostic means with the reported
values. It also calculates 36 paired external-model comparisons from the supplied
600-row summary table and checks their descriptive band status (32 within and
4 outside |delta_r| <= 0.030).

This utility calculates diagnostics from arrays and summaries from existing scores.
It does not train models or reconstruct the predictive scores from primary data.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

# Expected scalar diagnostics from manuscript (Section 2.6, Section 5.6, Table S10, Table S12)
EXPECTED_GEARS = {
    200: {"var": 0.842147, "rms": 1.250509, "r_train": 0.123992},
    500: {"var": 0.841455, "rms": 1.249770, "r_train": 0.125460},
    1000: {"var": 0.842191, "rms": 1.250557, "r_train": 0.124760},
    "overall": {"var": 0.841931, "rms": 1.250279, "r_train": 0.124737},
}

# Expected outside-band cells (Section 2.6, Table 3, Table S11)
EXPECTED_OUTSIDE_CELLS = {
    ("GEARS", "Adamson_2016", 1000): 0.0443,
    ("scGPT", "Adamson_2016", 500): 0.0324,
    ("Geneformer", "Adamson_2016", 1000): 0.0340,
    ("Geneformer", "Norman_2019", 1000): 0.0342,
}


def verify_gears_diagnostics(pred_dir: str) -> bool:
    print("=" * 70)
    print(" 1. GEARS Scalar Diagnostic Verification (Table 3, Section 2.6, SI S1.4)")
    print("=" * 70)
    if not os.path.exists(pred_dir):
        print(f"Directory not found: {pred_dir}")
        return False

    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    files = [f for f in files if "master" not in os.path.basename(f)]
    if not files:
        print(f"No non-master GEARS prediction files found in: {pred_dir}")
        return False

    print(f"Found {len(files)} GEARS prediction files. Evaluating arrays...")
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        y_pred = d["y_pred"]
        ref_mean = d["ref_mean"] if "ref_mean" in d else d["reference_mean"]
        hvg = int(d["hvg"]) if "hvg" in d else int(d["hvg_n"])
        if y_pred.shape != (50, hvg) or ref_mean.shape != (hvg,):
            raise ValueError(f"Unexpected prediction/reference shape in {f}")
        if not np.isfinite(y_pred).all() or not np.isfinite(ref_mean).all():
            raise ValueError(f"Non-finite prediction/reference values in {f}")

        # 1. Sample variance across genes with Bessel's correction (ddof=1)
        v = np.var(y_pred, axis=1, ddof=1)
        # 2. Dimension-normalized RMS norm
        rms = np.sqrt(np.mean(y_pred**2, axis=1))
        # The experimenter identifies this reference as the training-condition mean effect.
        r_train = [np.corrcoef(y_pred[i], ref_mean)[0, 1] for i in range(len(y_pred))]

        rows.append(
            {
                "file": os.path.basename(f),
                "dataset": str(d["dataset"]),
                "hvg": hvg,
                "seed": int(d["seed"]),
                "var": float(np.mean(v)),
                "rms": float(np.mean(rms)),
                "r_train": float(np.mean(r_train)),
            }
        )

    df = pd.DataFrame(rows)
    keys = ["dataset", "hvg", "seed"]
    if len(df) != 36 or df.duplicated(keys).any() or df["dataset"].nunique() != 4:
        raise ValueError("Expected 36 unique dataset/HVG/seed records for four datasets")
    expected_pairs = {(h, seed) for h in (200, 500, 1000) for seed in (42, 43, 44)}
    for dataset, group in df.groupby("dataset"):
        if set(zip(group["hvg"], group["seed"])) != expected_pairs:
            raise ValueError(f"Incomplete HVG/seed coverage: {dataset}")
    print(f"Computed diagnostics across {len(df)} runs (4 datasets x 3 scales x 3 seeds):\n")

    # Verify and assert each HVG scale
    for hvg in [200, 500, 1000]:
        g = df[df["hvg"] == hvg]
        obs_v = float(g["var"].mean())
        obs_rms = float(g["rms"].mean())
        obs_r = float(g["r_train"].mean())
        tgt = EXPECTED_GEARS[hvg]

        print(f"  HVG {hvg:4d} (12 runs):")
        print(
            f"    Variance (ddof=1) : {obs_v:.6f}  (target: {tgt['var']:.6f}, diff: {abs(obs_v - tgt['var']):.2e})"
        )
        print(
            f"    RMS Norm         : {obs_rms:.6f}  (target: {tgt['rms']:.6f}, diff: {abs(obs_rms - tgt['rms']):.2e})"
        )
        print(
            f"    r vs Training mean   : {obs_r:.6f}  (target: {tgt['r_train']:.6f}, diff: {abs(obs_r - tgt['r_train']):.2e})"
        )

        assert (
            abs(obs_v - tgt["var"]) < 1e-4
        ), f"HVG {hvg} variance mismatch: {obs_v} vs {tgt['var']}"
        assert (
            abs(obs_rms - tgt["rms"]) < 1e-4
        ), f"HVG {hvg} RMS norm mismatch: {obs_rms} vs {tgt['rms']}"
        assert (
            abs(obs_r - tgt["r_train"]) < 1e-4
        ), f"HVG {hvg} r vs Training mean mismatch: {obs_r} vs {tgt['r_train']}"

    # Overall mean
    ov_v = float(df["var"].mean())
    ov_rms = float(df["rms"].mean())
    ov_r = float(df["r_train"].mean())
    tgt_ov = EXPECTED_GEARS["overall"]
    print(f"\n  Overall Mean (36 runs):")
    print(
        f"    Variance (ddof=1) : {ov_v:.6f}  (target: {tgt_ov['var']:.6f}, diff: {abs(ov_v - tgt_ov['var']):.2e})"
    )
    print(
        f"    RMS Norm         : {ov_rms:.6f}  (target: {tgt_ov['rms']:.6f}, diff: {abs(ov_rms - tgt_ov['rms']):.2e})"
    )
    print(
        f"    r vs Training mean   : {ov_r:.6f}  (target: {tgt_ov['r_train']:.6f}, diff: {abs(ov_r - tgt_ov['r_train']):.2e})"
    )

    assert abs(ov_v - tgt_ov["var"]) < 1e-4, f"Overall variance mismatch: {ov_v} vs {tgt_ov['var']}"
    assert (
        abs(ov_rms - tgt_ov["rms"]) < 1e-4
    ), f"Overall RMS norm mismatch: {ov_rms} vs {tgt_ov['rms']}"
    assert (
        abs(ov_r - tgt_ov["r_train"]) < 1e-4
    ), f"Overall r vs Training mean mismatch: {ov_r} vs {tgt_ov['r_train']}"

    print(
        "\n=> VERIFIED: Retained-array diagnostic summaries agree within the stated 1e-4 tolerance; provenance and scale are not verified.\n"
    )
    return True


def verify_external_model_envelope(csv_path: str) -> bool:
    print("=" * 70)
    print(" 2. External Model Comparison Descriptive Reference Envelope (|dr| <= 0.030)")
    print("=" * 70)
    if not os.path.exists(csv_path):
        print(f"Per-fold results CSV not found at: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    groups = df.groupby(["dataset_id", "hvg_n"]).size()
    if len(df) != 600 or len(groups) != 12 or not (groups == 50).all():
        raise ValueError("Expected 600 rows across twelve 50-condition dataset/HVG cells")
    print(f"Loaded {len(df)} condition rows from: {os.path.basename(csv_path)}")

    models = [
        ("GEARS", "gears_minus_string_go"),
        ("scGPT", "scgpt_minus_string_go"),
        ("Geneformer", "geneformer_minus_string_go"),
    ]

    within_cells = []
    outside_cells = []

    print("\nEvaluating all 36 cells (mean paired difference over 50 conditions per cell):")
    print("-" * 70)
    print(
        f"{'Adapter':12s} | {'Dataset':15s} | {'HVGs':5s} | {'Mean delta_r':12s} | {'Band Status':14s}"
    )
    print("-" * 70)

    for m_name, col in models:
        assert col in df.columns, f"Expected column '{col}' not found in CSV."
        grouped = df.groupby(["dataset_id", "hvg_n"])[col].mean()
        for (ds, hvg), val in grouped.items():
            within = abs(val) <= 0.030
            status = "Within band" if within else "Outside band"
            print(f"{m_name:12s} | {ds:15s} | {hvg:5d} | {val:+12.4f} | {status:14s}")
            if within:
                within_cells.append((m_name, ds, hvg, val))
            else:
                outside_cells.append((m_name, ds, hvg, val))

    print("-" * 70)
    print(f"Total cells evaluated: {len(within_cells) + len(outside_cells)}")
    print(f"Within reference envelope (|delta_r| <= 0.030): {len(within_cells)}/36")
    print(f"Outside reference envelope (|delta_r| > 0.030) : {len(outside_cells)}/36")

    # Assert counts
    assert len(within_cells) == 32, f"Expected 32 cells within band, got {len(within_cells)}"
    assert len(outside_cells) == 4, f"Expected 4 cells outside band, got {len(outside_cells)}"

    # -------------------------------------------------------------------------
    # Statistical Basis Calculation: Condition-Level Differences
    # -------------------------------------------------------------------------
    print("\nRetrospective Descriptive Context for the +-0.030 Band:")
    print("-" * 70)
    all_diffs = np.concatenate(
        [
            df["gears_minus_string_go"].values,
            df["scgpt_minus_string_go"].values,
            df["geneformer_minus_string_go"].values,
        ]
    )
    pooled_sd = float(np.std(all_diffs, ddof=1))
    pooled_mean = float(np.mean(all_diffs))
    print(
        f"  Total condition paired differences (N)     : {len(all_diffs)} (600 conditions x 3 external models)"
    )
    print(f"  Pooled mean paired difference (ext - prim) : {pooled_mean:+.6f}")
    print(
        f"  Pooled condition paired difference SD      : {pooled_sd:.6f}  (sample SD ddof=1, target: 0.0244)"
    )

    cell_sds = []
    for m_name, col in models:
        for (ds, hvg), sub in df.groupby(["dataset_id", "hvg_n"]):
            cell_sds.append(sub[col].std(ddof=1))

    median_cell_sd = float(np.median(cell_sds))
    mean_cell_sd = float(np.mean(cell_sds))
    min_cell_sd = float(np.min(cell_sds))
    max_cell_sd = float(np.max(cell_sds))

    print(
        f"  Within-cell condition difference SD (36 cells): median = {median_cell_sd:.6f} (target: 0.0157)"
    )
    print(
        f"                                                  mean   = {mean_cell_sd:.6f}, range: [{min_cell_sd:.4f}, {max_cell_sd:.4f}]"
    )

    # Statistical assertions
    assert abs(pooled_sd - 0.024410) < 1e-4, f"Pooled SD mismatch: {pooled_sd:.6f} vs 0.024410"
    assert (
        abs(median_cell_sd - 0.015718) < 1e-4
    ), f"Median cell SD mismatch: {median_cell_sd:.6f} vs 0.015718"

    ratio_pooled = 0.030 / pooled_sd
    ratio_median = 0.030 / median_cell_sd
    print(
        f"  Envelope ratio vs pooled condition SD       : 0.030 / {pooled_sd:.4f} = {ratio_pooled:.2f}x (~1.23x)"
    )
    print(
        f"  Envelope ratio vs median cell condition SD  : 0.030 / {median_cell_sd:.4f} = {ratio_median:.2f}x (~1.91x)"
    )
    print(
        f"  Seed variability is not estimated by this check; condition dispersion is not optimization noise."
    )
    print(
        "=> CHECKED: Descriptive SDs and ratios match; these do not justify the historical choice or validate an equivalence margin.\n"
    )

    # Assert identity of the 4 outside cells
    print("\nVerifying 4 outside-band cells against manuscript targets:")
    outside_dict = {(m, ds, hvg): val for m, ds, hvg, val in outside_cells}
    for (m, ds, hvg), exp_val in EXPECTED_OUTSIDE_CELLS.items():
        assert (
            m,
            ds,
            hvg,
        ) in outside_dict, f"Expected outside cell {(m, ds, hvg)} not found in outside cells."
        actual_val = outside_dict[(m, ds, hvg)]
        print(
            f"  {m:12s} {ds:15s} HVG {hvg:4d}: delta_r = {actual_val:+.4f} (expected: +{exp_val:.3f}, favoring external adapter)"
        )
        assert (
            abs(actual_val - exp_val) < 0.001
        ), f"Value mismatch for {(m, ds, hvg)}: {actual_val} vs {exp_val}"
        assert actual_val > 0.030, f"Expected positive difference exceeding +0.030: {actual_val}"

    print(
        "\n=> VERIFIED: Exactly 32/36 within band and 4/36 outside band (all favoring external models)."
    )
    print("=> Status: Verified against manuscript Table 3 and Table S11.\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", help="Directory of retained GEARS NPZ files")
    parser.add_argument("--results-csv", help="External-model summary CSV")
    args = parser.parse_args()
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # 1. Look for GEARS predictions in standard locations
    gears_candidates = [
        os.path.normpath(os.path.join(base_dir, "..", "..", "CBAC_raw_model_predictions", "gears")),
        os.path.normpath(os.path.join(base_dir, "raw_predictions", "gears")),
    ]
    gears_path = args.predictions
    for cand in ([] if args.predictions else gears_candidates):
        if os.path.exists(cand) and glob.glob(os.path.join(cand, "*.npz")):
            gears_path = cand
            break

    if gears_path:
        if not verify_gears_diagnostics(gears_path):
            sys.exit(1)
    else:
        raise SystemExit(
            "Required GEARS prediction arrays were not found; verification incomplete."
        )

    # 2. Look for external baseline results CSV
    csv_candidates = [
        os.path.join(base_dir, "05_External_Baseline_Per_Fold_Results.csv"),
        os.path.normpath(
            os.path.join(
                base_dir, "..", "journal_upload", "05_External_Baseline_Per_Fold_Results.csv"
            )
        ),
    ]
    csv_path = args.results_csv
    for cand in ([] if args.results_csv else csv_candidates):
        if os.path.exists(cand):
            csv_path = cand
            break

    if csv_path:
        if not verify_external_model_envelope(csv_path):
            sys.exit(1)
    else:
        raise SystemExit("Required external-model CSV was not found; verification incomplete.")
