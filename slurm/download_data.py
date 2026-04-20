#!/usr/bin/env python
"""
Download all 4 Perturb-seq datasets for TurboGNN benchmark.
Uses scPerturb datasets from Zenodo record 10044268.
"""
import os
import sys
import subprocess
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
DATA_PROCESSED = HOME / "TurboGNN" / "data" / "processed"
DATA_SCPERTURB = HOME / "TurboGNN" / "data" / "scperturb"

ZENODO_BASE = "https://zenodo.org/records/10044268/files"

# Mapping: (local_filename, target_dir, zenodo_filename)
DATASETS = [
    ("norman_mapped.h5ad", DATA_PROCESSED,
     f"{ZENODO_BASE}/NormanWeissman2019_filtered.h5ad"),
    ("adamson.h5ad", DATA_SCPERTURB,
     f"{ZENODO_BASE}/AdamsonWeissman2016_GSM2406681_10X010.h5ad"),
    ("replogle_k562.h5ad", DATA_SCPERTURB,
     f"{ZENODO_BASE}/ReplogleWeissman2022_K562_essential.h5ad"),
    ("replogle_rpe1.h5ad", DATA_SCPERTURB,
     f"{ZENODO_BASE}/ReplogleWeissman2022_rpe1.h5ad"),
]


def download_file(url: str, dest: Path) -> bool:
    """Download a file using wget with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Remove zero-byte files from previous failed attempts
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()

    print(f"\nDownloading: {url}")
    print(f"       -> {dest}")
    try:
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "--timeout=60",
             "--tries=3", "-O", str(dest), url],
            check=True,
        )
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  OK: {size_mb:.0f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("=== Downloading Perturb-seq datasets from scPerturb (Zenodo) ===")

    for local_name, dest_dir, url in DATASETS:
        dest = dest_dir / local_name
        if dest.exists() and dest.stat().st_size > 10_000_000:  # >10MB
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"\nSKIP: {local_name} already exists ({size_mb:.0f} MB)")
            continue

        if not download_file(url, dest):
            print(f"\nERROR: Could not download {local_name}")
            sys.exit(1)

    print("\n\n=== Verifying datasets ===")
    all_ok = True
    for local_name, dest_dir, _ in DATASETS:
        dest = dest_dir / local_name
        if dest.exists() and dest.stat().st_size > 10_000_000:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  OK: {local_name} ({size_mb:.0f} MB)")
        else:
            print(f"  MISSING/TOO SMALL: {local_name}")
            all_ok = False

    if not all_ok:
        print("\nSome datasets missing!")
        sys.exit(1)

    # Quick validation with scanpy
    print("\n=== Quick validation ===")
    try:
        import scanpy as sc
        for local_name, dest_dir, _ in DATASETS:
            dest = dest_dir / local_name
            adata = sc.read_h5ad(str(dest), backed="r")
            print(f"  {local_name}: {adata.n_obs} cells x {adata.n_vars} genes")
            cols = list(adata.obs.columns)
            pert_candidates = ["gene", "condition", "perturbation",
                               "guide_ids", "perturbations"]
            found = [c for c in pert_candidates if c in cols]
            if not found:
                # Check all columns for potential perturbation info
                print(f"    WARNING: No standard pert column. Columns: {cols[:10]}")
            else:
                print(f"    Perturbation columns: {found}")
            adata.file.close()
    except Exception as e:
        print(f"  Validation error: {e}")

    print("\n=== All datasets ready! ===")


if __name__ == "__main__":
    main()
