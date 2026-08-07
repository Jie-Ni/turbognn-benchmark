from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

RNG_SEED = 20260806
N_BOOT = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def completeness_table(foundation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, dataset, hvg), group in foundation.groupby(["model", "dataset", "hvg"], sort=True):
        valid = group[~group["skipped"]]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "hvg": int(hvg),
                "scheduled_seed_condition_rows": int(len(group)),
                "valid_seed_condition_rows": int(len(valid)),
                "skipped_seed_condition_rows": int(group["skipped"].sum()),
                "seeds_scheduled": int(group["seed"].nunique()),
                "conditions_scheduled": int(group["condition"].nunique()),
                "conditions_valid_in_all_seeds": int(
                    valid.groupby("condition")["seed"].nunique().eq(3).sum()
                ),
                "valid_row_fraction": float(len(valid) / len(group)),
                "skip_reasons": "|".join(
                    f"{key}:{value}"
                    for key, value in group.loc[group["skipped"], "skip_reason"]
                    .fillna("")
                    .value_counts()
                    .items()
                ),
            }
        )
    return pd.DataFrame(rows)


def condition_level_pairs(
    foundation: pd.DataFrame, turbo: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foundation_valid = foundation[~foundation["skipped"]].copy()
    turbo_valid = turbo[~turbo["skipped"]].copy()

    f_condition = foundation_valid.groupby(
        ["model", "dataset", "hvg", "condition"], as_index=False
    ).agg(
        foundation_pearson_seed_mean=("pearson_r", "mean"),
        foundation_seed_count=("seed", "nunique"),
    )
    t_condition = turbo_valid.groupby(["dataset", "hvg", "condition"], as_index=False).agg(
        turbognn_pearson_seed_mean=("pearson_r", "mean"),
        turbognn_seed_count=("seed", "nunique"),
    )
    paired = f_condition.merge(
        t_condition,
        on=["dataset", "hvg", "condition"],
        how="inner",
        validate="many_to_one",
    )
    paired["pearson_delta_foundation_minus_turbognn"] = (
        paired["foundation_pearson_seed_mean"] - paired["turbognn_pearson_seed_mean"]
    )

    rows: list[dict] = []
    for (model, dataset, hvg), group in paired.groupby(["model", "dataset", "hvg"], sort=True):
        delta = group["pearson_delta_foundation_minus_turbognn"].to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(delta)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "hvg": int(hvg),
                "n_distinct_held_out_conditions": int(len(group)),
                "all_conditions_have_3_foundation_seeds": bool(
                    group["foundation_seed_count"].eq(3).all()
                ),
                "all_conditions_have_3_turbognn_seeds": bool(
                    group["turbognn_seed_count"].eq(3).all()
                ),
                "foundation_pearson_mean": float(group["foundation_pearson_seed_mean"].mean()),
                "turbognn_pearson_mean": float(group["turbognn_pearson_seed_mean"].mean()),
                "delta_foundation_minus_turbognn_mean": float(delta.mean()),
                "delta_median": float(np.median(delta)),
                "condition_win_rate_foundation": float((delta > 0).mean()),
                "condition_bootstrap_ci95_low": ci_low,
                "condition_bootstrap_ci95_high": ci_high,
                "direction": (
                    "foundation_higher"
                    if delta.mean() > 0
                    else "turbognn_higher" if delta.mean() < 0 else "tie"
                ),
            }
        )
    return paired, pd.DataFrame(rows)


def raw_json_inventory(raw_root: Path, foundation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows: list[dict] = []
    metadata_rows: list[dict] = []
    for model_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        model = model_dir.name
        for path in sorted(model_dir.rglob("*.json")):
            partial = path.name.endswith(".partial.json")
            file_rows.append(
                {
                    "model": model,
                    "partial": partial,
                    "path": (
                        Path("04_foundation_models") / path.relative_to(foundation_root)
                    ).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            if partial:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            folds = [
                fold
                for seed_payload in payload.get("seeds", {}).values()
                for fold in seed_payload.get("folds", [])
            ]
            all_keys = set(payload) | set(metadata)
            for fold in folds:
                all_keys.update(fold)
            metadata_rows.append(
                {
                    "model": model,
                    "dataset": payload.get("dataset"),
                    "hvg": payload.get("hvg"),
                    "seed": payload.get("seed"),
                    "fold_start": payload.get("fold_start"),
                    "fold_end": payload.get("fold_end"),
                    "training_mode": metadata.get("training_mode"),
                    "epochs": metadata.get("epochs"),
                    "n_valid_conditions": metadata.get("n_valid_conditions"),
                    "hvg_vocab_coverage_fraction": metadata.get("hvg_vocab_coverage_fraction"),
                    "has_prediction_vector": any(
                        "pred" in key.lower() and key.lower() not in {"pretrained_model"}
                        for key in all_keys
                    ),
                    "has_training_loss_history": any(
                        "history" in key.lower() or "train_loss" in key.lower() for key in all_keys
                    ),
                    "has_validation_loss_scalar": any(
                        key in {"best_val_mse", "val_mse"} for key in all_keys
                    ),
                    "source_json": (
                        Path("04_foundation_models") / path.relative_to(foundation_root)
                    ).as_posix(),
                }
            )
    return pd.DataFrame(file_rows), pd.DataFrame(metadata_rows)


def partial_pair_table(raw_inventory: pd.DataFrame) -> pd.DataFrame:
    hashes = dict(zip(raw_inventory["path"], raw_inventory["sha256"]))
    rows = []
    for path_text, formal_hash in hashes.items():
        path = PurePosixPath(path_text)
        if path.name.endswith(".partial.json"):
            continue
        partial = path.with_name(f"{path.stem}.partial.json")
        partial_hash = hashes.get(partial.as_posix())
        rows.append(
            {
                "formal_json": path.as_posix(),
                "partial_json": partial.as_posix(),
                "partial_exists": partial_hash is not None,
                "byte_identical": partial_hash == formal_hash,
            }
        )
    return pd.DataFrame(rows)


def reproducibility_table(merged_dir: Path, recomputed_dir: Path) -> pd.DataFrame:
    names = [
        "foundation_summary_by_dataset_hvg_model.csv",
        "paired_stats_by_dataset_hvg_model.csv",
        "paired_global_stats_by_model.csv",
        "merge_audit.json",
    ]
    rows = []
    for name in names:
        original = merged_dir / name
        recomputed = recomputed_dir / name
        rows.append(
            {
                "file": name,
                "original_sha256": sha256(original),
                "recomputed_sha256": sha256(recomputed),
                "byte_identical": sha256(original) == sha256(recomputed),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--recomputed-dir", type=Path)
    args = parser.parse_args()

    foundation_root = args.project_root / "04_foundation_models"
    merged_dir = foundation_root / "results" / "merged_analysis"
    raw_root = foundation_root / "results" / "raw_lnz"
    output_dir = args.output_dir
    recomputed_dir = args.recomputed_dir or output_dir / "recomputed_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    foundation = pd.read_csv(merged_dir / "foundation_fold_level.csv")
    turbo = pd.read_csv(merged_dir / "turbognn_combined_fold_level.csv")
    foundation["skipped"] = foundation["skipped"].astype(bool)
    turbo["skipped"] = turbo["skipped"].astype(bool)

    key = ["model", "dataset", "hvg", "seed", "fold_index", "condition"]
    turbo_key = ["dataset", "hvg", "seed", "fold_index", "condition"]
    if foundation.duplicated(key).any():
        raise RuntimeError("Duplicate foundation merge keys detected")
    if turbo.duplicated(turbo_key).any():
        raise RuntimeError("Duplicate TurboGNN merge keys detected")

    completeness = completeness_table(foundation)
    paired_conditions, condition_stats = condition_level_pairs(foundation, turbo)
    raw_inventory, raw_metadata = raw_json_inventory(raw_root, foundation_root)
    partial_pairs = partial_pair_table(raw_inventory)
    reproducibility = reproducibility_table(merged_dir, recomputed_dir)

    completeness.to_csv(output_dir / "foundation_completeness_by_cell.csv", index=False)
    paired_conditions.to_csv(
        output_dir / "foundation_condition_level_seed_averaged_pairs.csv", index=False
    )
    condition_stats.to_csv(output_dir / "foundation_condition_level_paired_stats.csv", index=False)
    raw_inventory.to_csv(output_dir / "foundation_raw_json_sha256_manifest.csv", index=False)
    raw_metadata.to_csv(output_dir / "foundation_raw_json_metadata_audit.csv", index=False)
    partial_pairs.to_csv(output_dir / "foundation_partial_pair_status.csv", index=False)
    reproducibility.to_csv(output_dir / "foundation_merge_reproducibility.csv", index=False)

    print(
        json.dumps(
            {
                "scheduled_foundation_rows": int(len(foundation)),
                "valid_foundation_rows": int((~foundation["skipped"]).sum()),
                "skipped_foundation_rows": int(foundation["skipped"].sum()),
                "condition_level_pairs": int(len(paired_conditions)),
                "raw_formal_json": int((~raw_inventory["partial"]).sum()),
                "raw_partial_json": int(raw_inventory["partial"].sum()),
                "formal_partial_byte_identical": int(partial_pairs["byte_identical"].sum()),
                "formal_partial_different": int((~partial_pairs["byte_identical"]).sum()),
                "all_recomputed_summary_files_byte_identical": bool(
                    reproducibility["byte_identical"].all()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
