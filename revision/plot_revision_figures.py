# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Generate publication figures for the evidence-based CBAC revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


def format_signed_three(value: float) -> str:
    rounded = round(float(value), 3)
    return "0.000" if rounded == 0 else f"{rounded:+.3f}"


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, frame: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    if frame:
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)


def plot_design_audit(config: pd.DataFrame, output_dir: Path) -> None:
    expected = config[config["structurally_expected"]].copy()
    cell = expected.groupby(["dataset", "hvg", "graph"], as_index=False).agg(
        observed=("n_conditions", "sum")
    )
    cell["completion"] = cell["observed"] / 150
    cell["label"] = cell["dataset"].replace(
        {
            "adamson": "Adamson",
            "norman": "Norman",
            "replogle_k562": "Replogle K562",
            "replogle_rpe1": "Replogle RPE1",
        }
    )
    cell["column"] = (
        cell["graph"].replace(
            {
                "string_ppi": "STRING",
                "gene_ontology": "GO",
                "coexpression": "Coexpr.",
                "combined": "Union",
                "random": "BA random",
                "no_graph": "Dense",
            }
        )
        + "\n"
        + cell["hvg"].astype(str)
    )
    row_order = ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"]
    column_order = []
    for hvg in (200, 500, 1000):
        graphs = ["STRING", "GO", "Union", "BA random", "Dense"]
        if hvg == 200:
            graphs.insert(2, "Coexpr.")
        column_order.extend([f"{graph}\n{hvg}" for graph in graphs])
    matrix = cell.pivot(index="label", columns="column", values="completion").reindex(
        index=row_order, columns=column_order
    )

    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.02, 1.8], height_ratios=[1, 1])
    gs.update(wspace=0.10, hspace=0.30, left=0.05, right=0.99, top=0.91, bottom=0.09)
    ax_a = plt.subplot(gs[:, 0])
    ax_b = plt.subplot(gs[0, 1])
    ax_c = plt.subplot(gs[1, 1])

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.patch.set_visible(True)
    style_axis(ax_a)
    ax_a.set_title(
        r"$\bf{(a)}$" + "  What the submitted comparison changed", loc="left", fontsize=12
    )
    boxes = [
        (
            0.08,
            0.76,
            0.84,
            0.12,
            "Submitted sparse graph arms",
            "3-layer GAT • masked cross-gene attention",
            "#4575b4",
        ),
        (
            0.08,
            0.56,
            0.84,
            0.12,
            "Submitted dense arm",
            "2-layer Transformer • positional encoding",
            "#d73027",
        ),
        (
            0.08,
            0.36,
            0.84,
            0.12,
            "Fail-open target path",
            "plus-only parser → mask=None\nAdamson 50/50; Norman ≥27/50",
            "#b2182b",
        ),
        (
            0.08,
            0.16,
            0.84,
            0.12,
            "Revision-stage matched design",
            "same LayerNorm GAT • exact controls\nnonempty target indicator",
            "#444444",
        ),
    ]
    for x, y, width, height, title, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015",
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
        )
        ax_a.add_patch(patch)
        ax_a.text(x + 0.03, y + height - 0.038, title, fontsize=11, weight="semibold", color=color)
        ax_a.text(x + 0.03, y + 0.025, body, fontsize=10.5, color="dimgrey", va="bottom")
    for y_start, y_end in ((0.75, 0.69), (0.55, 0.49), (0.35, 0.29)):
        ax_a.add_patch(
            FancyArrowPatch(
                (0.5, y_start), (0.5, y_end), arrowstyle="-|>", mutation_scale=13, color="grey"
            )
        )
    ax_a.text(
        0.5,
        0.02,
        "Saved values are source-output descriptors; they do not isolate\ntarget-conditioned adjacency effects.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="dimgrey",
        style="italic",
    )

    sns.heatmap(
        matrix,
        ax=ax_b,
        cmap=sns.light_palette("#4575b4", as_cmap=True),
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Saved seed-condition completion"},
        annot_kws={"size": 10.5, "weight": "medium"},
    )
    ax_b.set_title(r"$\bf{(b)}$" + "  Seed-explicit result completeness", loc="left", fontsize=12)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=45, ha="right", fontsize=10.5)
    ax_b.set_yticklabels(ax_b.get_yticklabels(), rotation=0, fontsize=10.5)
    ax_b.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    ax_c.axis("off")
    ax_c.patch.set_visible(True)
    style_axis(ax_c)
    ax_c.set_title(r"$\bf{(c)}$" + "  Revision evidence hierarchy", loc="left", fontsize=12)
    levels = [
        (
            0.04,
            0.69,
            0.92,
            0.20,
            "REVISION-STAGE PRIMARY",
            "STRING–GO versus dense in one GAT; fail-closed targets; outcome pending",
            "#4575b4",
        ),
        (
            0.11,
            0.39,
            0.78,
            0.18,
            "SENSITIVITY",
            "Scale, topology, dependence, simple baselines and secondary metrics",
            "#72a5b4",
        ),
        (
            0.18,
            0.11,
            0.64,
            0.16,
            "FORENSIC",
            "7,660 records; graph-source rows, fail-open targets, transduction",
            "#bdbdbd",
        ),
    ]
    for x, y, width, height, title, body, color in levels:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015",
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.95,
        )
        ax_c.add_patch(patch)
        text_color = "white" if color != "#bdbdbd" else "#333333"
        ax_c.text(x + 0.03, y + height - 0.055, title, fontsize=11, weight="bold", color=text_color)
        ax_c.text(x + 0.03, y + 0.045, body, fontsize=10.5, color=text_color)

    fig.suptitle("Audit-led redesign of the benchmark", fontsize=15, color="dimgrey", y=0.98)
    fig.text(
        0.99,
        0.015,
        f"Only {int(expected['n_conditions'].sum()):,} of 9,600 planned seed-condition records are traceable ({expected['n_conditions'].sum()/9600:.1%}).",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="dimgrey",
        style="italic",
    )
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig1_audit_led_redesign")


def plot_legacy_effects(contrasts: pd.DataFrame, output_dir: Path) -> None:
    data = contrasts[contrasts["contrast"] == "combined - no_graph"].copy()
    data["dataset_label"] = data["dataset"].replace(
        {
            "adamson": "Adamson",
            "norman": "Norman",
            "replogle_k562": "Replogle K562",
            "replogle_rpe1": "Replogle RPE1",
        }
    )
    data["label"] = data["dataset_label"] + " · " + data["hvg"].astype(int).astype(str) + " HVGs"
    order = [
        f"{dataset} · {hvg} HVGs"
        for dataset in ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"]
        for hvg in (200, 500, 1000)
    ]
    data["label"] = pd.Categorical(data["label"], categories=order[::-1], ordered=True)
    data = data.sort_values("label")

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=300)
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    dataset_colors = {
        name: pal[index]
        for index, name in enumerate(
            ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"], start=1
        )
    }
    y = np.arange(len(data))
    for index, (_, row) in enumerate(data.iterrows()):
        color = dataset_colors[row["dataset_label"]]
        ax.hlines(index, row["ci95_low"], row["ci95_high"], color=color, linewidth=2.3, alpha=0.85)
        ax.scatter(
            row["mean_delta_r"],
            index,
            color=color,
            marker="o",
            s=70,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            row["ci95_high"] + 0.0012,
            index,
            format_signed_three(row["mean_delta_r"]),
            va="center",
            fontsize=8.5,
            color="dimgrey",
        )
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.axvspan(-0.01, 0.01, color="#fee090", alpha=0.35, zorder=0)
    ax.set_yticks(y, data["label"])
    ax.set_xlabel(
        "Mean paired source-output difference\n(scale-dependent union GAT − submitted dense Transformer)",
        fontsize=10.5,
        color="dimgrey",
    )
    ax.set_ylabel("")
    ax.set_title(
        "Accounting-repaired legacy source outputs",
        fontsize=14,
        loc="left",
        color="dimgrey",
        pad=30,
    )
    ax.text(
        0.01,
        1.005,
        "Intervals resample archived condition keys; target masks failed for Adamson 50/50 and Norman ≥27/50.",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgrey",
        va="bottom",
    )
    ax.grid(False)
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    fig.text(
        0.98,
        0.01,
        "Not target-conditioned: fail-open masks, architecture mismatch, and transductive preprocessing remain.",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    save_figure(fig, output_dir, "fig2_corrected_legacy_effects")


def plot_topology_scale(contrasts: pd.DataFrame, output_dir: Path) -> None:
    data = contrasts[contrasts["contrast"] == "combined - random"].copy()
    labels = {
        "adamson": "Adamson",
        "norman": "Norman",
        "replogle_k562": "Replogle K562",
        "replogle_rpe1": "Replogle RPE1",
    }
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
    fig = plt.figure(figsize=(11, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.08, hspace=0.30, left=0.08, right=0.99, top=0.91, bottom=0.09)
    axes = [plt.subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ymax = max(0.15, float(data["ci95_high"].max()) * 1.10)
    ymin = min(-0.04, float(data["ci95_low"].min()) * 1.10)
    labels_counter = ["a", "b", "c", "d"]
    for index, (dataset, ax) in enumerate(zip(labels, axes, strict=True)):
        subset = data[data["dataset"] == dataset].sort_values("hvg")
        x = subset["hvg"].to_numpy()
        y = subset["mean_delta_r"].to_numpy()
        low = y - subset["ci95_low"].to_numpy()
        high = subset["ci95_high"].to_numpy() - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([low, high]),
            color=pal[5],
            marker="o",
            markersize=7,
            linewidth=2.0,
            capsize=3,
            markerfacecolor=pal[2],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.axhline(0, color="lightgrey", linewidth=0.9)
        ax.set_xticks([200, 500, 1000])
        ax.set_ylim(ymin, ymax)
        ax.set_title(
            r"$\bf{(" + labels_counter[index] + r")}$" + f"  {labels[dataset]}",
            loc="left",
            fontsize=11,
        )
        if index % 2 == 0:
            ax.set_ylabel(
                "Paired source-output difference\n(scale-dependent union GAT − BA GAT)",
                fontsize=9.5,
                color="dimgrey",
            )
        else:
            ax.tick_params(labelleft=False)
        if index // 2 == 1:
            ax.set_xlabel("Number of HVGs", fontsize=9.5, color="dimgrey")
        style_axis(ax)
    fig.suptitle(
        "Saved source-output dependence on scale: single BA comparison",
        fontsize=14,
        color="dimgrey",
        y=0.98,
    )
    strongest = data.loc[data["mean_delta_r"].idxmax()]
    fig.text(
        0.99,
        0.015,
        f"Largest saved output difference: {labels[strongest['dataset']]} at {int(strongest['hvg'])} HVGs ({format_signed_three(strongest['mean_delta_r'])}); target-mask success is unknown and one BA graph cannot identify biology.",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig3_single_random_scale_effect")


def plot_k562_distributions(condition_means: pd.DataFrame, output_dir: Path) -> None:
    data = condition_means[
        (condition_means["dataset"] == "replogle_k562")
        & condition_means["graph"].isin(["combined", "random", "no_graph"])
    ].copy()
    data["graph_label"] = data["graph"].replace(
        {"combined": "Union", "random": "BA random", "no_graph": "Dense comparator"}
    )
    order = ["Union", "BA random", "Dense comparator"]
    palette = {"Union": "#4575b4", "BA random": "#d73027", "Dense comparator": "#bdbdbd"}
    fig = plt.figure(figsize=(7.4, 4.2), dpi=300)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.08, left=0.09, right=0.995, top=0.74, bottom=0.18)
    axes = [plt.subplot(gs[0, index]) for index in range(3)]
    for index, (hvg, ax) in enumerate(zip((200, 500, 1000), axes, strict=True)):
        subset = data[data["hvg"] == hvg].dropna(subset=["pearson_r"])
        sns.violinplot(
            data=subset,
            x="graph_label",
            y="pearson_r",
            hue="graph_label",
            order=order,
            hue_order=order,
            palette=palette,
            inner=None,
            linewidth=0.8,
            cut=0,
            ax=ax,
            legend=False,
        )
        sns.stripplot(
            data=subset,
            x="graph_label",
            y="pearson_r",
            order=order,
            color="dimgrey",
            alpha=0.35,
            size=2.6,
            jitter=0.16,
            ax=ax,
        )
        medians = subset.groupby("graph_label")["pearson_r"].median()
        for x_index, label in enumerate(order):
            ax.scatter(
                x_index,
                medians[label],
                marker="D",
                s=35,
                color="#222222",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        ax.axhline(0, color="lightgrey", linewidth=0.8)
        ax.set_ylim(-0.35, 0.95)
        union_source = (
            "Union: STRING–GO–\ncontrol co-expression" if hvg == 200 else "Union: STRING–GO"
        )
        ax.set_title(
            r"$\bf{(" + chr(ord("a") + index) + r")}$" + f"  {hvg} HVGs\n{union_source}",
            loc="left",
            fontsize=7.8,
        )
        ax.set_xlabel("")
        ax.set_xticks(
            [0, 1, 2],
            ["Union\nGAT", "Mean-degree BA\nGAT", "All-to-all\nTransformer"],
            fontsize=7.3,
        )
        if index == 0:
            ax.set_ylabel("Seed-averaged archived Pearson r", fontsize=8.5, color="dimgrey")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        style_axis(ax)
    fig.suptitle(
        "Replogle K562 condition-labelled source-output distributions",
        fontsize=11,
        color="dimgrey",
        y=0.97,
    )
    fig.text(
        0.99,
        0.02,
        "Union composition changes after 200 HVGs; BA is mean-degree-matched. Target-mask success is unrecoverable.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="dimgrey",
        style="italic",
    )
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig4_k562_condition_distributions")


def plot_identity(identity: pd.DataFrame, output_dir: Path) -> None:
    data = identity.copy()
    data["dataset_label"] = data["dataset"].replace(
        {
            "adamson": "Adamson",
            "norman": "Norman",
            "replogle_k562": "Replogle K562",
            "replogle_rpe1": "Replogle RPE1",
        }
    )
    data["row"] = data["dataset_label"] + " · " + data["hvg"].astype(int).astype(str)
    data["column"] = (
        data["comparison"]
        .str.replace("combined vs ", "", regex=False)
        .replace(
            {
                "string_ppi": "STRING",
                "gene_ontology": "GO",
                "coexpression": "Coexpr.",
                "random": "BA random",
                "no_graph": "Dense\nTransformer",
            }
        )
    )
    rows = [
        f"{dataset} · {hvg}"
        for dataset in ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"]
        for hvg in (200, 500, 1000)
    ]
    matrix = data.pivot(index="row", columns="column", values="top10_overlap").reindex(
        index=rows, columns=["STRING", "GO", "Coexpr.", "BA random", "Dense\nTransformer"]
    )
    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="rocket",
        vmin=0,
        vmax=10,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Shared condition keys among top 10 outputs"},
        annot_kws={"size": 9, "weight": "medium"},
        ax=ax,
    )
    ax.set_title(
        "Archived top-condition-key overlap relative to scale-dependent union-GAT output",
        fontsize=14,
        loc="left",
        color="dimgrey",
        pad=8,
    )
    ax.set_xlabel("Comparator", fontsize=10.5, color="dimgrey")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    adamson_min = int(matrix.loc[[row for row in rows if row.startswith("Adamson")]].min().min())
    fig.text(
        0.98,
        0.01,
        f"Output-key stability only: Adamson has {adamson_min}/10 overlap, but all 50 archived target masks failed open.",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig5_top_condition_identity")


def plot_external_audit(
    foundation: pd.DataFrame, completeness: pd.DataFrame, output_dir: Path
) -> None:
    data = foundation.copy()
    data["dataset_label"] = data["dataset"].replace(
        {
            "adamson": "Adamson",
            "norman": "Norman",
            "replogle_k562": "Replogle K562",
            "replogle_rpe1": "Replogle RPE1",
        }
    )
    row_order = [
        f"{dataset} · {hvg}"
        for dataset in ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"]
        for hvg in (200, 500, 1000)
    ][::-1]
    data["row"] = data["dataset_label"] + " · " + data["hvg"].astype(int).astype(str)
    model_order = ["gears", "scgpt", "geneformer"]
    model_titles = {
        "gears": "GEARS condition-mean adaptation",
        "scgpt": "scGPT condition-mean adaptation",
        "geneformer": "Geneformer-V2 embedding + ridge",
    }
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[5.0, 1.45])
    gs.update(wspace=0.06, hspace=0.34, left=0.13, right=0.99, top=0.90, bottom=0.08)
    axes = [plt.subplot(gs[0, index]) for index in range(3)]
    x_min = min(-0.55, float(data["condition_bootstrap_ci95_low"].min()) - 0.02)
    x_max = max(0.08, float(data["condition_bootstrap_ci95_high"].max()) + 0.02)
    for panel_index, (model, ax) in enumerate(zip(model_order, axes, strict=True)):
        subset = data[data["model"] == model].set_index("row").reindex(row_order).reset_index()
        y = np.arange(len(subset))
        for index, row in subset.iterrows():
            value = row["delta_foundation_minus_turbognn_mean"]
            low = row["condition_bootstrap_ci95_low"]
            high = row["condition_bootstrap_ci95_high"]
            color = "#d73027" if value > 0 else "#4575b4"
            marker = {200: "o", 500: "s", 1000: "^"}[int(row["hvg"])]
            ax.hlines(index, low, high, color=color, linewidth=1.8, alpha=0.85)
            ax.scatter(
                value,
                index,
                color=color,
                marker=marker,
                s=58,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        ax.axvline(0, color="#444444", linewidth=1.0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.6, len(subset) - 0.4)
        ax.set_yticks(y)
        if panel_index == 0:
            ax.set_yticklabels(subset["row"], fontsize=13.5)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Adapter − union-GAT source output", fontsize=13, color="dimgrey")
        ax.set_title(
            r"$\bf{(" + chr(ord("a") + panel_index) + r")}$" + f"  {model_titles[model]}",
            loc="left",
            fontsize=13.5,
        )
        ax.tick_params(axis="x", labelsize=12.5)
        style_axis(ax)

    coverage = (
        completeness.groupby("model", as_index=False)[
            [
                "scheduled_seed_condition_rows",
                "valid_seed_condition_rows",
                "skipped_seed_condition_rows",
            ]
        ]
        .sum()
        .set_index("model")
        .reindex(model_order)
    )
    coverage["valid_percent"] = (
        100 * coverage["valid_seed_condition_rows"] / coverage["scheduled_seed_condition_rows"]
    )
    coverage["skipped_percent"] = (
        100 * coverage["skipped_seed_condition_rows"] / coverage["scheduled_seed_condition_rows"]
    )
    ax_d = plt.subplot(gs[1, :])
    y = np.arange(len(model_order))[::-1]
    ax_d.barh(y, coverage["valid_percent"], color="#4575b4", height=0.52, label="Valid")
    ax_d.barh(
        y,
        coverage["skipped_percent"],
        left=coverage["valid_percent"],
        color="#bdbdbd",
        height=0.52,
        label="Excluded",
    )
    status = {
        "gears": "descriptive only · ranking eligibility gated",
        "scgpt": "descriptive only · custom protocol",
        "geneformer": "descriptive only · custom embedding/decoder",
    }
    for position, model in zip(y, model_order, strict=True):
        row = coverage.loc[model]
        valid = int(row["valid_seed_condition_rows"])
        scheduled = int(row["scheduled_seed_condition_rows"])
        skipped = int(row["skipped_seed_condition_rows"])
        ax_d.text(
            min(float(row["valid_percent"]) / 2, 46),
            position,
            f"{valid:,}/{scheduled:,} valid",
            ha="center",
            va="center",
            fontsize=12.5,
            color="white",
            weight="semibold",
        )
        if skipped:
            ax_d.text(
                float(row["valid_percent"]) + float(row["skipped_percent"]) / 2,
                position,
                f"{skipped:,}",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#333333",
                weight="semibold",
            )
        ax_d.text(103, position, status[model], va="center", fontsize=13, color="dimgrey")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(["GEARS", "scGPT", "Geneformer"], fontsize=12.5)
    ax_d.set_xlim(0, 150)
    ax_d.set_xlabel("Coverage (% of scheduled seed–condition evaluations)", fontsize=12.5)
    ax_d.set_title(
        r"$\bf{(d)}$" + "  Evaluation coverage and interpretation gate",
        loc="left",
        fontsize=13.5,
    )
    ax_d.legend(
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1, 1.02),
        fontsize=12,
    )
    style_axis(ax_d)
    fig.suptitle(
        "Archived external-adaptation output contrasts",
        fontsize=16,
        color="dimgrey",
        y=0.97,
    )
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig6_external_model_audit")


def plot_gears_scalar_diagnostic(pairs: pd.DataFrame, output_dir: Path) -> None:
    data = pairs[pairs["model"] == "gears"].copy()
    data["dataset_label"] = data["dataset"].replace(
        {
            "adamson": "Adamson",
            "norman": "Norman",
            "replogle_k562": "Replogle K562",
            "replogle_rpe1": "Replogle RPE1",
        }
    )
    data["hvg_label"] = data["hvg"].astype(int).astype(str)
    datasets = ["Adamson", "Norman", "Replogle K562", "Replogle RPE1"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300, sharex=True, sharey=True)
    y_min = min(-0.11, float(data["foundation_pearson_seed_mean"].min()) - 0.01)
    y_max = max(0.14, float(data["foundation_pearson_seed_mean"].max()) + 0.01)
    np.random.seed(20260806)
    for panel_index, (dataset, ax) in enumerate(zip(datasets, axes.flat, strict=True)):
        subset = data[data["dataset_label"] == dataset]
        sns.boxplot(
            data=subset,
            x="hvg_label",
            y="foundation_pearson_seed_mean",
            order=["200", "500", "1000"],
            ax=ax,
            color="#dce8f4",
            width=0.55,
            linewidth=1.2,
            fliersize=0,
        )
        sns.stripplot(
            data=subset,
            x="hvg_label",
            y="foundation_pearson_seed_mean",
            order=["200", "500", "1000"],
            ax=ax,
            color="#4575b4",
            size=3.2,
            jitter=0.18,
            alpha=0.7,
        )
        ax.axhline(0, color="#444444", linewidth=1.0)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            r"$\bf{(" + chr(ord("a") + panel_index) + r")}$" + f"  {dataset}",
            loc="left",
            fontsize=11,
        )
        ax.set_xlabel("Number of HVGs", fontsize=10)
        ax.set_ylabel(
            (
                "Saved GEARS Pearson r\n(seed-averaged within condition)"
                if panel_index % 2 == 0
                else ""
            ),
            fontsize=10,
        )
        style_axis(ax)
    fig.suptitle(
        "Saved GEARS scalar distribution: a partial diagnostic only",
        fontsize=14,
        color="dimgrey",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Each point is one held-out condition averaged over three optimisation seeds (n = 50 per dataset–scale cell). "
        "Prediction vectors were not retained, so output–training-mean correlation, gene order, and non-degeneracy cannot be tested.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgrey",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    sns.despine(left=True, bottom=True)
    save_figure(fig, output_dir, "fig7_gears_scalar_diagnostic")


def main() -> None:
    """Load audited tables and save seven figures as editable SVG, PDF, and 300-dpi PNG."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foundation-stats", type=Path)
    parser.add_argument("--foundation-completeness", type=Path)
    parser.add_argument("--foundation-pairs", type=Path)
    args = parser.parse_args()

    sns.set_theme(font_scale=1.0, style="whitegrid", font="Arial")
    config = pd.read_csv(args.data_dir / "configuration_manifest.csv").dropna()
    contrasts = pd.read_csv(args.data_dir / "paired_condition_contrasts.csv").dropna(
        subset=["mean_delta_r", "ci95_low", "ci95_high"]
    )
    condition_means = pd.read_csv(args.data_dir / "condition_seed_averages.csv").dropna(
        subset=["pearson_r"]
    )
    identity = pd.read_csv(args.data_dir / "top_condition_identity.csv").dropna(
        subset=["top10_overlap"]
    )

    plot_design_audit(config, args.output_dir)
    plot_legacy_effects(contrasts, args.output_dir)
    plot_topology_scale(contrasts, args.output_dir)
    plot_k562_distributions(condition_means, args.output_dir)
    plot_identity(identity, args.output_dir)
    if args.foundation_stats:
        foundation = pd.read_csv(args.foundation_stats).dropna(
            subset=[
                "delta_foundation_minus_turbognn_mean",
                "condition_bootstrap_ci95_low",
                "condition_bootstrap_ci95_high",
            ]
        )
        completeness_path = args.foundation_completeness or (
            args.foundation_stats.parent / "foundation_completeness_by_cell.csv"
        )
        completeness = pd.read_csv(completeness_path)
        plot_external_audit(foundation, completeness, args.output_dir)
        pairs_path = args.foundation_pairs or (
            args.foundation_stats.parent / "foundation_condition_level_seed_averaged_pairs.csv"
        )
        pairs = pd.read_csv(pairs_path).dropna(subset=["foundation_pearson_seed_mean"])
        plot_gears_scalar_diagnostic(pairs, args.output_dir)


if __name__ == "__main__":
    main()
