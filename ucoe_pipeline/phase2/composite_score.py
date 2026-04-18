"""
Phase II — Composite UCOE score and sensitivity analysis.
"""

import logging
from itertools import product

import numpy as np
import pandas as pd

from ucoe_pipeline.config import RANKING_WEIGHTS
from ucoe_pipeline.phase2.feature_extraction import FEATURE_NAMES
from ucoe_pipeline.phase2.similarity_metrics import (
    compute_cosine_similarities,
    compute_mahalanobis_distances,
    compute_percentile_ranks,
)

logger = logging.getLogger(__name__)


def normalize_to_01(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Normalize an array to [0, 1] range. If invert=True, lower input → higher output."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values)
    vmin, vmax = finite.min(), finite.max()
    if vmax == vmin:
        return np.full_like(values, 0.5)
    normed = (values - vmin) / (vmax - vmin)
    if invert:
        normed = 1.0 - normed
    return np.clip(normed, 0, 1)


def compute_composite_scores(
    candidates: pd.DataFrame,
    ref_matrix: np.ndarray,
    centroid: np.ndarray,
    weights: dict[str, float] = RANKING_WEIGHTS,
) -> pd.DataFrame:
    """Compute the composite UCOE score for all candidates.

    Parameters
    ----------
    candidates : DataFrame with all feature columns from Phase II extraction.
    ref_matrix : (n_ucoes, n_features) from known UCOEs.
    centroid : (n_features,) mean profile of known UCOEs.
    weights : dict with keys 'mahalanobis', 'cosine', 'percentile'.

    Returns
    -------
    candidates with added columns: mahalanobis_dist, mahalanobis_score,
    cosine_sim, cosine_score, percentile_score, composite_score, composite_rank.
    """
    logger.info("Computing composite UCOE scores for %d candidates", len(candidates))

    # Build feature matrix from candidates
    available_features = [f for f in FEATURE_NAMES if f in candidates.columns]
    cand_matrix = candidates[available_features].values.astype(float).copy()
    ref_copy = ref_matrix.copy()
    centroid_copy = centroid.copy()

    # 1. Mahalanobis distance (lower = more similar)
    maha_dist = compute_mahalanobis_distances(cand_matrix.copy(), ref_copy, centroid_copy)
    maha_score = normalize_to_01(maha_dist, invert=True)  # Invert: lower distance → higher score

    # 2. Cosine similarity (higher = more similar)
    cosine_sim = compute_cosine_similarities(cand_matrix.copy(), centroid_copy)
    cosine_score = normalize_to_01(cosine_sim, invert=False)

    # 3. Percentile rank composite
    percentile_score = compute_percentile_ranks(cand_matrix.copy(), available_features)

    # Composite weighted score
    w_m = weights.get("mahalanobis", 0.4)
    w_c = weights.get("cosine", 0.3)
    w_p = weights.get("percentile", 0.3)

    composite = w_m * maha_score + w_c * cosine_score + w_p * percentile_score

    # Add to DataFrame
    result = candidates.copy()
    result["mahalanobis_dist"] = maha_dist
    result["mahalanobis_score"] = maha_score
    result["cosine_sim"] = cosine_sim
    result["cosine_score"] = cosine_score
    result["percentile_score"] = percentile_score
    result["composite_score"] = composite
    result["composite_rank"] = result["composite_score"].rank(ascending=False, method="min").astype(int)

    # Sort by composite score descending
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)

    logger.info("Top 10 candidates by composite score:")
    for i, row in result.head(10).iterrows():
        gene_info = ""
        if "gene1" in row and "gene2" in row:
            gene_info = f" ({row['gene1']}/{row['gene2']})"
        logger.info(
            "  #%d: %s:%d-%d%s — score=%.4f (maha=%.4f, cos=%.4f, pctl=%.4f)",
            i + 1, row["chrom"], row["start"], row["end"], gene_info,
            row["composite_score"], row["mahalanobis_score"],
            row["cosine_score"], row["percentile_score"],
        )

    return result


def sensitivity_analysis(
    candidates: pd.DataFrame,
    ref_matrix: np.ndarray,
    centroid: np.ndarray,
    weight_steps: list[float] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Vary ranking weights and check stability of top-ranked candidates.

    Tests all weight combinations that sum to 1.0 (in steps).
    Returns a DataFrame with candidate stability across weight combinations.
    """
    if weight_steps is None:
        weight_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    logger.info("Running sensitivity analysis with %d weight steps", len(weight_steps))

    # Generate weight triplets that sum to ~1.0
    triplets = []
    for w_m in weight_steps:
        for w_c in weight_steps:
            w_p = round(1.0 - w_m - w_c, 2)
            if 0.05 <= w_p <= 0.7:
                triplets.append({"mahalanobis": w_m, "cosine": w_c, "percentile": w_p})

    logger.info("Testing %d weight combinations", len(triplets))

    # For each combination, get the top_n candidates
    top_appearances = {}  # region_key -> count of appearances in top_n
    for weights in triplets:
        scored = compute_composite_scores(candidates, ref_matrix.copy(), centroid.copy(), weights)
        for _, row in scored.head(top_n).iterrows():
            key = f"{row['chrom']}:{row['start']}-{row['end']}"
            top_appearances[key] = top_appearances.get(key, 0) + 1

    # Build stability report
    stability = pd.DataFrame([
        {"region": k, "top_n_appearances": v, "stability_pct": v / len(triplets) * 100}
        for k, v in top_appearances.items()
    ]).sort_values("stability_pct", ascending=False)

    logger.info("Sensitivity analysis: %d unique regions appeared in top %d across %d weight combos",
                len(stability), top_n, len(triplets))
    logger.info("Most stable candidates (appearing in >80%% of combinations):")
    stable = stability[stability["stability_pct"] > 80]
    for _, row in stable.iterrows():
        logger.info("  %s — %.0f%% of combinations", row["region"], row["stability_pct"])

    return stability
