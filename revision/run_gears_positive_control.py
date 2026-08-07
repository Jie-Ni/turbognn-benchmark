"""Run the pinned official GEARS API and export auditable raw positive-control evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EPOCH_RE = re.compile(
    r"^Epoch (?P<epoch>[0-9]+): Train Overall MSE: (?P<train>[0-9.eE+-]+) "
    r"Validation Overall MSE: (?P<validation>[0-9.eE+-]+)\.\s*$"
)


def _numeric_mapping(value: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _json_array(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {}
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            arrays[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
    return {"sha256": _file_sha256(path), "arrays": arrays}


def _numeric_array(value: Any) -> np.ndarray | None:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        return None
    array = np.ascontiguousarray(array)
    if not np.isfinite(array).all():
        raise RuntimeError("GEARS state contains non-finite values")
    return array


def _save_state_dict(path: Path, module: Any) -> dict[str, Any]:
    arrays = {}
    for key, value in module.state_dict().items():
        array = _numeric_array(value)
        if array is None:
            raise RuntimeError(f"GEARS state entry is not a safe numeric array: {key}")
        arrays[str(key).replace("/", "__")] = array
    if not arrays:
        raise RuntimeError("GEARS model state is empty")
    np.savez_compressed(path, **arrays)
    return _array_manifest(path)


def _save_graph_state(
    path: Path,
    config: dict[str, Any],
    gene_names: Sequence[str],
    perturbable_gene_order: Sequence[str],
) -> dict[str, Any]:
    arrays = {}
    for key in sorted(config):
        lowered = str(key).casefold()
        if "graph" not in lowered and not lowered.startswith("g_"):
            continue
        array = _numeric_array(config[key])
        if array is not None:
            arrays[str(key).replace("/", "__")] = array
    if not arrays:
        raise RuntimeError("GEARS graph configuration did not expose numeric graph state")
    encoded_gene_order = "\0".join(str(name) for name in gene_names).encode("utf-8")
    encoded_perturbable_order = "\0".join(str(name) for name in perturbable_gene_order).encode(
        "utf-8"
    )
    arrays["__cbac_gene_order_utf8"] = np.frombuffer(encoded_gene_order, dtype=np.uint8)
    arrays["__cbac_perturbable_gene_order_utf8"] = np.frombuffer(
        encoded_perturbable_order, dtype=np.uint8
    )
    np.savez_compressed(path, **arrays)
    return _array_manifest(path)


def _baseline_results(
    test_results: dict[str, np.ndarray], loader: Any, control_expression: np.ndarray
) -> dict[str, np.ndarray]:
    baseline_de: list[np.ndarray] = []
    for batch in loader:
        for de_index in batch.de_idx:
            if hasattr(de_index, "detach"):
                de_index = de_index.detach().cpu().numpy()
            indices = np.asarray(de_index, dtype=int)
            baseline_de.append(control_expression[indices])
    if len(baseline_de) != len(test_results["pert_cat"]):
        raise RuntimeError("GEARS baseline DE rows do not align with official test evaluation")
    return {
        "pert_cat": np.asarray(test_results["pert_cat"]),
        "pred": np.repeat(control_expression.reshape(1, -1), len(test_results["pert_cat"]), axis=0),
        "truth": np.asarray(test_results["truth"]),
        "pred_de": np.stack(baseline_de),
        "truth_de": np.asarray(test_results["truth_de"]),
    }


def _parse_epoch_history(messages: Sequence[str], epochs: int) -> tuple[list[float], list[float]]:
    parsed: dict[int, tuple[float, float]] = {}
    for message in messages:
        match = EPOCH_RE.fullmatch(message.strip())
        if match is not None:
            parsed[int(match.group("epoch"))] = (
                float(match.group("train")),
                float(match.group("validation")),
            )
    if sorted(parsed) != list(range(1, epochs + 1)):
        raise RuntimeError("Official GEARS epoch-level train/validation history is incomplete")
    return (
        [parsed[index][0] for index in range(1, epochs + 1)],
        [parsed[index][1] for index in range(1, epochs + 1)],
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    repository_root = arguments.repository_root.resolve()
    if not (repository_root / "gears" / "inference.py").is_file():
        raise RuntimeError("Pinned GEARS repository root is invalid")
    sys.path.insert(0, str(repository_root))

    import scanpy as sc
    import torch
    from gears import GEARS, PertData
    from gears.inference import compute_metrics, evaluate
    import gears
    import gears.gears as gears_module

    imported_root = Path(gears.__file__).resolve().parent.parent
    if imported_root != repository_root:
        raise RuntimeError("GEARS import did not resolve to the hash-verified repository checkout")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(arguments.dataset)
    pert_data = PertData(str(arguments.work_dir))
    pert_data.new_data_process(dataset_name=arguments.dataset_name, adata=adata)
    pert_data.prepare_split(split=arguments.split, seed=arguments.seed)
    pert_data.get_dataloader(
        batch_size=arguments.batch_size,
        test_batch_size=arguments.test_batch_size,
    )

    model = GEARS(pert_data, device=arguments.device)
    model.model_initialize(
        hidden_size=arguments.hidden_size,
        num_go_gnn_layers=arguments.num_go_gnn_layers,
        num_gene_gnn_layers=arguments.num_gene_gnn_layers,
        decoder_hidden_size=arguments.decoder_hidden_size,
        num_similar_genes_go_graph=arguments.num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=(arguments.num_similar_genes_co_express_graph),
        coexpress_threshold=arguments.coexpress_threshold,
        uncertainty=False,
        direction_lambda=arguments.direction_lambda,
    )
    initialization_path = Path(f"{arguments.output}.initialization.npz")
    checkpoint_path = Path(f"{arguments.output}.checkpoint.npz")
    graph_path = Path(f"{arguments.output}.graph.npz")
    gene_names = [str(value) for value in model.gene_list]
    perturbable_gene_order = [str(value) for value in model.pert_list]
    if (
        not gene_names
        or len(set(gene_names)) != len(gene_names)
        or not perturbable_gene_order
        or len(set(perturbable_gene_order)) != len(perturbable_gene_order)
        or not set(perturbable_gene_order).issubset(set(gene_names))
    ):
        raise RuntimeError("GEARS gene and perturbable-gene orders are invalid")
    initialization_state = _save_state_dict(initialization_path, model.model)
    graph_state = _save_graph_state(graph_path, model.config, gene_names, perturbable_gene_order)
    messages: list[str] = []
    original_print = gears_module.print_sys

    def capture(message: Any) -> None:
        messages.append(str(message))
        original_print(message)

    gears_module.print_sys = capture
    try:
        model.train(
            epochs=arguments.epochs,
            lr=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
        )
    finally:
        gears_module.print_sys = original_print
    training_loss, validation_loss = _parse_epoch_history(messages, arguments.epochs)
    checkpoint_state = _save_state_dict(checkpoint_path, model.best_model)

    test_loader = pert_data.dataloader.get("test_loader")
    if test_loader is None:
        raise RuntimeError("The declared GEARS split did not produce a test loader")
    test_results = evaluate(test_loader, model.best_model, False, arguments.device)
    reported_metrics, _ = compute_metrics(test_results)
    control_expression = np.asarray(model.ctrl_expression.detach().cpu(), dtype=float)
    baseline = _baseline_results(test_results, test_loader, control_expression)
    baseline_metrics, _ = compute_metrics(baseline)

    perturbations = sorted(set(str(value) for value in test_results["pert_cat"]))
    perturbable = set(perturbable_gene_order)
    perturbable_index = {gene: index for index, gene in enumerate(perturbable_gene_order)}
    unmapped = sorted(
        condition
        for condition in perturbations
        if any(target not in perturbable for target in condition.split("+") if target != "ctrl")
    )
    total_conditions = len(perturbations)
    mapped_conditions = total_conditions - len(unmapped)
    if total_conditions <= 0:
        raise RuntimeError("GEARS test split contains no conditions")

    results = {
        "pert_cat": _json_array(test_results["pert_cat"]),
        "pred": _json_array(test_results["pred"]),
        "truth": _json_array(test_results["truth"]),
        "pred_de": _json_array(test_results["pred_de"]),
        "truth_de": _json_array(test_results["truth_de"]),
    }
    baseline_results = {
        "pert_cat": _json_array(baseline["pert_cat"]),
        "pred": _json_array(baseline["pred"]),
        "truth": _json_array(baseline["truth"]),
        "pred_de": _json_array(baseline["pred_de"]),
        "truth_de": _json_array(baseline["truth_de"]),
    }
    eligible_test_conditions = sorted(
        str(value) for value in pert_data.set2conditions.get("test", perturbations)
    )
    condition_target_mapping = []
    for condition in perturbations:
        targets = [target for target in condition.split("+") if target != "ctrl"]
        target_indices = [
            perturbable_index[target] for target in targets if target in perturbable_index
        ]
        target_indicator = [0] * len(perturbable_gene_order)
        for index in target_indices:
            target_indicator[index] = 1
        condition_target_mapping.append(
            {
                "condition": condition,
                "targets": targets,
                "target_indices": target_indices,
                "target_indicator": target_indicator,
                "mapped": condition not in unmapped,
            }
        )
    return {
        "schema_version": "2.0",
        "results": results,
        "reported_metrics": _numeric_mapping(reported_metrics),
        "baselines": {
            "control_mean_zero_change": {
                "results": baseline_results,
                "reported_metrics": _numeric_mapping(baseline_metrics),
            }
        },
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "gene_names": gene_names,
        "perturbable_gene_order": perturbable_gene_order,
        "split_membership": {
            "name": arguments.split,
            "seed": arguments.seed,
            "row_ids": [
                f"test_evaluation_row:{index}" for index in range(len(test_results["pert_cat"]))
            ],
            "condition_ids": _json_array(test_results["pert_cat"]),
            "eligible_test_conditions": eligible_test_conditions,
        },
        "condition_target_mapping": condition_target_mapping,
        "identifier_coverage": float(mapped_conditions / total_conditions),
        "target_audit": {
            "total_conditions": total_conditions,
            "mapped_conditions": mapped_conditions,
            "unmapped_conditions": unmapped,
            "perturbable_gene_count": len(perturbable_gene_order),
        },
        "model_identity": {
            "initialization_state": initialization_state,
            "checkpoint_state": checkpoint_state,
        },
        "graph_provenance": {
            "source": "GEARS.model.config numeric graph tensors after model_initialize",
            **graph_state,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="simulation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-go-gnn-layers", type=int, default=1)
    parser.add_argument("--num-gene-gnn-layers", type=int, default=1)
    parser.add_argument("--decoder-hidden-size", type=int, default=16)
    parser.add_argument("--num-similar-genes-go-graph", type=int, default=20)
    parser.add_argument("--num-similar-genes-co-express-graph", type=int, default=20)
    parser.add_argument("--coexpress-threshold", type=float, default=0.4)
    parser.add_argument("--direction-lambda", type=float, default=0.1)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_argument_parser().parse_args(arguments)
    payload = run(parsed)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
