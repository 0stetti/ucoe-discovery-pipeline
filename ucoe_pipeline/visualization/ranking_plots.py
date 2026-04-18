"""
Ranking visualizations: score distributions and ranked bar charts.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucoe_pipeline.config import KNOWN_UCOES
from ucoe_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


def plot_score_distribution(
    scored_candidates: pd.DataFrame,
    output_path: Path,
):
    """Histogram of composite scores with known UCOEs marked."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = scored_candidates["composite_score"].values
    ax.hist(scores, bins=50, color="#78909C", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Composite UCOE Score", fontsize=12)
    ax.set_ylabel("Number of Candidates", fontsize=12)
    ax.set_title("Distribution of Composite UCOE Scores", fontsize=14, fontweight="bold")

    # Mark known UCOEs
    colors = ["#F44336", "#4CAF50", "#2196F3"]
    for i, (name, info) in enumerate(KNOWN_UCOES.items()):
        gene_a, gene_b = info["genes"]
        matches = scored_candidates[
            (scored_candidates.get("gene1", pd.Series(dtype=str)).str.upper() == gene_a.upper())
            | (scored_candidates.get("gene2", pd.Series(dtype=str)).str.upper() == gene_b.upper())
        ]
        if not matches.empty:
            score = matches["composite_score"].max()
            ax.axvline(score, color=colors[i % len(colors)], linewidth=2,
                       linestyle="--", label=f"{name} ({score:.3f})")

    ax.legend(fontsize=9)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved score distribution: %s", output_path)


def plot_top_ranked_bar(
    scored_candidates: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
):
    """Horizontal bar chart of top-N candidates by composite score."""
    top = scored_candidates.head(top_n).copy()

    # Build labels
    labels = []
    for _, row in top.iterrows():
        gene_info = ""
        if "gene1" in row and "gene2" in row:
            gene_info = f"{row['gene1']}/{row['gene2']}"
        else:
            gene_info = f"{row['chrom']}:{row['start']}-{row['end']}"
        labels.append(gene_info)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y = np.arange(len(top))

    # Color known UCOEs differently
    known_genes = set()
    for info in KNOWN_UCOES.values():
        known_genes.update(g.upper() for g in info["genes"])

    colors = []
    for _, row in top.iterrows():
        g1 = str(row.get("gene1", "")).upper()
        g2 = str(row.get("gene2", "")).upper()
        if g1 in known_genes or g2 in known_genes:
            colors.append("#FF5722")
        else:
            colors.append("#1976D2")

    ax.barh(y, top["composite_score"].values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Composite UCOE Score", fontsize=12)
    ax.set_title(f"Top {top_n} UCOE Candidates", fontsize=14, fontweight="bold")

    # Add score labels
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["composite_score"] + 0.005, i,
                f"{row['composite_score']:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF5722", label="Known UCOE"),
        Patch(facecolor="#1976D2", label="Novel candidate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved top-ranked bar chart: %s", output_path)


def plot_metric_comparison(
    scored_candidates: pd.DataFrame,
    output_path: Path,
):
    """Scatter plot: Mahalanobis score vs Cosine score, colored by percentile score."""
    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(
        scored_candidates["mahalanobis_score"],
        scored_candidates["cosine_score"],
        c=scored_candidates["percentile_score"],
        cmap="RdYlGn",
        s=30,
        alpha=0.7,
        edgecolors="grey",
        linewidths=0.3,
    )
    plt.colorbar(sc, label="Percentile Score")

    # Mark known UCOEs
    known_genes = {}
    for name, info in KNOWN_UCOES.items():
        for g in info["genes"]:
            known_genes[g.upper()] = name

    for _, row in scored_candidates.iterrows():
        g1 = str(row.get("gene1", "")).upper()
        g2 = str(row.get("gene2", "")).upper()
        matched_name = known_genes.get(g1) or known_genes.get(g2)
        if matched_name:
            ax.annotate(
                matched_name,
                (row["mahalanobis_score"], row["cosine_score"]),
                fontsize=8, fontweight="bold", color="red",
                xytext=(5, 5), textcoords="offset points",
            )
            ax.scatter(
                [row["mahalanobis_score"]], [row["cosine_score"]],
                s=100, marker="*", color="red", zorder=5,
            )

    ax.set_xlabel("Mahalanobis Score", fontsize=12)
    ax.set_ylabel("Cosine Score", fontsize=12)
    ax.set_title("Metric Comparison: Mahalanobis vs Cosine", fontsize=14, fontweight="bold")

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric comparison plot: %s", output_path)
