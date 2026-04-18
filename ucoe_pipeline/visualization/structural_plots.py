"""
Structural analysis visualizations.

Generates publication-quality plots comparing DNA structural properties
across UCOE candidates, known UCOEs, and random CpG island controls.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from ucoe_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "UCOE_candidates": "#3274A1",
    "Known_UCOEs": "#E1812C",
    "CpG_island_controls": "#78B7B2",
}

GROUP_LABELS = {
    "UCOE_candidates": "UCOE Candidates (n=599)",
    "Known_UCOEs": "Known UCOEs (n=3)",
    "CpG_island_controls": "CpG Island Controls (n=200)",
}

METRIC_LABELS = {
    "flexibility_mean": "Flexibility Index\n(Brukner, higher = more flexible)",
    "stiffness_mean": "Stiffness Index\n(higher = more rigid)",
    "bendability_mean": "Bendability\n(Brukner trinuc., higher = more bendable)",
    "gc_content": "GC Content",
    "cpg_obs_exp": "CpG Obs/Exp Ratio",
    "cpg_density": "CpG Density (per kb)",
    "poly_at_fraction": "Poly(dA:dT) Fraction",
    "nuc_score_mean": "Nucleosome Formation Score\n(higher = more nucleosome-prone)",
    "nuc_depleted_fraction": "Nucleosome-Depleted Fraction\n(score < 0)",
    "nuc_enriched_fraction": "Nucleosome-Enriched Fraction\n(score > 0.1)",
    "nfr_score": "NFR Score\n(nucleosome-free region, higher = more open)",
    "poly_at_coverage": "Poly(dA:dT) Coverage",
}


def plot_structural_boxplots(
    combined_df: pd.DataFrame,
    output_dir: Path,
    metrics: list[str] | None = None,
):
    """Create boxplots comparing structural metrics across the three groups."""
    if metrics is None:
        metrics = [
            "flexibility_mean", "stiffness_mean", "bendability_mean",
            "gc_content", "cpg_obs_exp", "cpg_density",
            "nuc_score_mean", "nuc_depleted_fraction", "nfr_score",
        ]

    ensure_dir(output_dir)
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = combined_df[["group", metric]].dropna()

        groups_present = [g for g in ["UCOE_candidates", "Known_UCOEs", "CpG_island_controls"]
                          if g in data["group"].values]
        palette = [COLORS[g] for g in groups_present]

        sns.boxplot(
            data=data, x="group", y=metric, ax=ax,
            order=groups_present, palette=palette,
            showfliers=False, width=0.6,
        )
        sns.stripplot(
            data=data[data["group"] == "Known_UCOEs"],
            x="group", y=metric, ax=ax,
            order=groups_present, color=COLORS["Known_UCOEs"],
            size=10, marker="D", edgecolor="black", linewidth=1, zorder=5,
        )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([GROUP_LABELS.get(g, g).split("(")[0].strip()
                            for g in groups_present], fontsize=8, rotation=15, ha="right")

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "DNA Structural Properties: UCOE Candidates vs. Controls",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    outpath = output_dir / "structural_boxplots.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved structural boxplots: %s", outpath)


def plot_stiffness_vs_nucleosome(
    combined_df: pd.DataFrame,
    output_dir: Path,
):
    """Scatter plot: DNA stiffness vs nucleosome formation score."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 7))

    for group in ["CpG_island_controls", "UCOE_candidates", "Known_UCOEs"]:
        subset = combined_df[combined_df["group"] == group]
        if subset.empty:
            continue
        marker = "D" if group == "Known_UCOEs" else "o"
        size = 120 if group == "Known_UCOEs" else 20
        alpha = 1.0 if group == "Known_UCOEs" else 0.5
        zorder = 10 if group == "Known_UCOEs" else 2
        edgecolor = "black" if group == "Known_UCOEs" else "none"

        ax.scatter(
            subset["stiffness_mean"], subset["nuc_score_mean"],
            c=COLORS[group], label=GROUP_LABELS.get(group, group),
            marker=marker, s=size, alpha=alpha, zorder=zorder,
            edgecolors=edgecolor, linewidths=1,
        )

        # Label known UCOEs
        if group == "Known_UCOEs":
            for _, row in subset.iterrows():
                name = row["header"].replace("_", " ").replace("UCOE ", "")
                if "A2UCOE" in row["header"]:
                    name = "A2UCOE"
                elif "TBP" in row["header"]:
                    name = "TBP/PSMB1"
                elif "SRF" in row["header"]:
                    name = "SRF-UCOE"
                ax.annotate(
                    name, (row["stiffness_mean"], row["nuc_score_mean"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold",
                )

    ax.set_xlabel("DNA Stiffness Index (higher = more rigid)", fontsize=12)
    ax.set_ylabel("Nucleosome Formation Score (higher = more nucleosome-prone)", fontsize=12)
    ax.set_title("DNA Stiffness vs. Nucleosome Formation Potential", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    outpath = output_dir / "stiffness_vs_nucleosome.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stiffness vs nucleosome scatter: %s", outpath)


def plot_flexibility_distribution(
    combined_df: pd.DataFrame,
    output_dir: Path,
):
    """Overlapping histograms of flexibility index by group."""
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))

    for group in ["CpG_island_controls", "UCOE_candidates"]:
        subset = combined_df[combined_df["group"] == group]["flexibility_mean"].dropna()
        ax.hist(
            subset, bins=40, alpha=0.5, color=COLORS[group],
            label=GROUP_LABELS.get(group, group), density=True, edgecolor="white",
        )

    # Mark known UCOEs as vertical lines
    known = combined_df[combined_df["group"] == "Known_UCOEs"]["flexibility_mean"].dropna()
    for val in known:
        ax.axvline(val, color=COLORS["Known_UCOEs"], linewidth=2, linestyle="--")
    # Add legend entry for known UCOEs line
    ax.axvline(np.nan, color=COLORS["Known_UCOEs"], linewidth=2, linestyle="--",
               label="Known UCOEs")

    ax.set_xlabel("Flexibility Index (Brukner)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("DNA Flexibility Distribution: Candidates vs. Controls", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    outpath = output_dir / "flexibility_distribution.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved flexibility distribution: %s", outpath)


def plot_structural_summary_table(
    comparison_df: pd.DataFrame,
    output_dir: Path,
):
    """Create a summary table figure with statistical comparison results."""
    ensure_dir(output_dir)

    # Select key metrics for the table
    key_metrics = [
        "flexibility_mean", "stiffness_mean", "bendability_mean",
        "gc_content", "cpg_obs_exp", "nuc_score_mean",
        "nuc_depleted_fraction", "nfr_score",
    ]
    table_data = comparison_df[comparison_df["metric"].isin(key_metrics)].copy()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    col_labels = ["Metric", "Candidates\n(median)", "Known UCOEs\n(median)",
                  "Controls\n(median)", "p-value", "Significance"]

    rows = []
    for _, row in table_data.iterrows():
        p = row["p_value"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        label = METRIC_LABELS.get(row["metric"], row["metric"]).split("\n")[0]
        rows.append([
            label,
            f"{row['candidates_median']:.4f}",
            f"{row['known_ucoe_median']:.4f}",
            f"{row['controls_median']:.4f}",
            f"{p:.2e}",
            sig,
        ])

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        colColours=["#E8E8E8"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color significant rows
    for i, row_data in enumerate(rows):
        if row_data[-1] in ("***", "**"):
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#E8F4FD")

    ax.set_title(
        "Statistical Comparison: UCOE Candidates vs. Random CpG Islands\n"
        "(Mann-Whitney U test, * p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=12, fontweight="bold", pad=20,
    )

    outpath = output_dir / "structural_comparison_table.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved structural comparison table: %s", outpath)


def generate_structural_plots(
    combined_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: Path | None = None,
):
    """Generate all structural analysis plots."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / "figures" / "structural"

    ensure_dir(output_dir)

    plot_structural_boxplots(combined_df, output_dir)
    plot_stiffness_vs_nucleosome(combined_df, output_dir)
    plot_flexibility_distribution(combined_df, output_dir)
    plot_structural_summary_table(comparison_df, output_dir)

    logger.info("All structural plots saved to %s", output_dir)
