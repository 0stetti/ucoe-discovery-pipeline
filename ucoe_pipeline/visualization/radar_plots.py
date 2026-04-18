"""
Spider/radar plots comparing candidate feature profiles to the known UCOE reference.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ucoe_pipeline.phase2.feature_extraction import FEATURE_NAMES
from ucoe_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

# Shorter labels for radar plot axes
FEATURE_LABELS = {
    "H3K4me3_mean": "H3K4me3",
    "H3K27ac_mean": "H3K27ac",
    "H3K9ac_mean": "H3K9ac",
    "H3K36me3_mean": "H3K36me3",
    "H3K27me3_mean": "H3K27me3\n(inv.)",
    "H3K9me3_mean": "H3K9me3\n(inv.)",
    "meth_mean": "Methylation\n(inv.)",
    "DNase_mean": "DNase",
    "repliseq_mean": "Repli-seq\n(E/L)",
    "CTCF_n_peaks": "CTCF\npeaks",
    "cpg_obs_exp": "CpG\nO/E",
    "cpg_gc_pct": "GC%",
}

# Features used for radar (subset of mean values — skip CVs for clarity)
RADAR_FEATURES = [
    "H3K4me3_mean", "H3K27ac_mean", "H3K9ac_mean", "H3K36me3_mean",
    "DNase_mean", "cpg_obs_exp", "repliseq_mean", "CTCF_n_peaks",
    "H3K27me3_mean", "H3K9me3_mean", "meth_mean",
]

# Features where lower is better (will be inverted for visualization)
INVERT_FEATURES = {"H3K27me3_mean", "H3K9me3_mean", "meth_mean"}


def _normalize_for_radar(
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    all_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize candidate and reference values to [0, 1] for each feature.

    Uses min/max from all candidates for scaling. Inverts features where lower is better.
    """
    n_features = len(RADAR_FEATURES)
    norm_cand = np.zeros(n_features)
    norm_ref = np.zeros(n_features)

    for i, feat in enumerate(RADAR_FEATURES):
        col_all = all_values[:, i]
        vmin = np.nanmin(col_all)
        vmax = np.nanmax(col_all)

        if vmax == vmin:
            norm_cand[i] = 0.5
            norm_ref[i] = 0.5
        else:
            norm_cand[i] = (candidate_values[i] - vmin) / (vmax - vmin)
            norm_ref[i] = (reference_values[i] - vmin) / (vmax - vmin)

        # Invert features where lower is better
        if feat in INVERT_FEATURES:
            norm_cand[i] = 1.0 - norm_cand[i]
            norm_ref[i] = 1.0 - norm_ref[i]

    return np.clip(norm_cand, 0, 1), np.clip(norm_ref, 0, 1)


def plot_radar(
    candidate_row: pd.Series,
    reference_centroid: np.ndarray,
    all_candidates: pd.DataFrame,
    output_path: Path,
    title: str = "",
):
    """Create a single radar plot comparing one candidate to the reference profile."""
    available = [f for f in RADAR_FEATURES if f in candidate_row.index and f in all_candidates.columns]
    if len(available) < 3:
        logger.warning("Too few features (%d) for radar plot", len(available))
        return

    # Get values
    cand_vals = np.array([float(candidate_row.get(f, np.nan)) for f in available])
    ref_vals = reference_centroid[:len(available)]
    all_vals = all_candidates[available].values.astype(float)

    # Normalize
    norm_cand, norm_ref = _normalize_for_radar(cand_vals, ref_vals, all_vals)

    # Radar plot
    labels = [FEATURE_LABELS.get(f, f) for f in available]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    norm_cand = np.append(norm_cand, norm_cand[0])
    norm_ref = np.append(norm_ref, norm_ref[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, norm_ref, "o-", color="#2196F3", linewidth=2, label="Known UCOEs (reference)")
    ax.fill(angles, norm_ref, alpha=0.15, color="#2196F3")

    ax.plot(angles, norm_cand, "o-", color="#FF5722", linewidth=2, label="Candidate")
    ax.fill(angles, norm_cand, alpha=0.15, color="#FF5722")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.05), fontsize=10)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar plot: %s", output_path)


def plot_top_candidates_radar(
    scored_candidates: pd.DataFrame,
    reference_centroid: np.ndarray,
    output_dir: Path,
    top_n: int = 10,
):
    """Generate radar plots for the top N candidates."""
    logger.info("Generating radar plots for top %d candidates", top_n)
    ensure_dir(output_dir)

    # Build reference centroid for radar features
    # reference_centroid is indexed by FEATURE_NAMES, so we need to map
    # each RADAR_FEATURE to its position in FEATURE_NAMES
    available = [f for f in RADAR_FEATURES if f in scored_candidates.columns]
    ref_radar = []
    for f in available:
        if f in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(f)
            ref_radar.append(reference_centroid[idx] if idx < len(reference_centroid) else np.nan)
        else:
            ref_radar.append(np.nan)
    ref_radar = np.array(ref_radar)

    for i, (_, row) in enumerate(scored_candidates.head(top_n).iterrows()):
        gene_info = ""
        if "gene1" in row and "gene2" in row:
            gene_info = f" ({row['gene1']}/{row['gene2']})"
        title = f"Rank #{i+1}{gene_info}\n{row['chrom']}:{row['start']}-{row['end']}"
        fname = f"radar_rank{i+1:02d}_{row['chrom']}_{row['start']}_{row['end']}.png"
        plot_radar(row, ref_radar, scored_candidates, output_dir / fname, title)
