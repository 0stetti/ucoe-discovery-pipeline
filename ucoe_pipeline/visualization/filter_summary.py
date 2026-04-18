"""
Filter summary — funnel chart showing how many candidates survived each filter.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ucoe_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


def plot_filter_funnel(
    filter_counts: dict[str, int],
    output_path: Path,
):
    """Create a funnel/waterfall chart showing candidate counts at each filter stage.

    Parameters
    ----------
    filter_counts : ordered dict mapping filter name → candidate count after that filter.
        Example: {"All gene pairs": 18432, "Filter 1: Divergent HKG": 1247, ...}
    """
    labels = list(filter_counts.keys())
    counts = list(filter_counts.values())
    n = len(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors: gradient from light to dark
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    bars = ax.barh(range(n), counts, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Candidate Regions", fontsize=12)
    ax.set_title("Phase I Filtering Funnel", fontsize=14, fontweight="bold")

    # Add count labels
    max_count = max(counts) if counts else 1
    for i, (count, bar) in enumerate(zip(counts, bars)):
        pct = ""
        if i > 0 and counts[i - 1] > 0:
            pct = f" ({count/counts[i-1]*100:.0f}%)"
        ax.text(
            count + max_count * 0.01, i,
            f"{count:,}{pct}",
            va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, max_count * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved filter funnel: %s", output_path)
