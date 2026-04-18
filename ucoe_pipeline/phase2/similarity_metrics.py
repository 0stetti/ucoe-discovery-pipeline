"""
Phase II — Similarity metrics.

Three complementary metrics to rank candidates against the known UCOE profile:
1. Mahalanobis distance (regularized covariance via Ledoit-Wolf)
2. Cosine similarity (z-score normalized)
3. Percentile rank composite
"""

import logging

import numpy as np
from scipy.spatial.distance import mahalanobis, cosine
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_mahalanobis_distances(
    candidates_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance from each candidate to the UCOE centroid.

    Uses Ledoit-Wolf shrinkage estimator for covariance estimated on the
    candidate population only (n=599, p=21). This avoids the circularity of
    including the 3 reference UCOEs in the covariance estimation, and provides
    a stable estimate since n >> p.

    Note: This is a population-adjusted Mahalanobis distance that measures
    how far each candidate deviates from the UCOE centroid relative to the
    variability observed among all candidates. It is NOT the classical
    Mahalanobis distance from the UCOE class (which would require covariance
    estimated from UCOEs only, infeasible with n=3). This design choice is
    explicitly acknowledged: the covariance structure defines the metric
    space, and using the candidate population ensures stable estimation
    while the centroid (target) remains external to this distribution.

    Parameters
    ----------
    candidates_matrix : (n_candidates, n_features)
    ref_matrix : (n_ucoes, n_features), used only for NaN imputation context
    centroid : (n_features,), mean of reference UCOEs

    Returns
    -------
    distances : (n_candidates,) Mahalanobis distances
    """
    n_ucoes, n_features = ref_matrix.shape
    n_candidates = candidates_matrix.shape[0]

    # Handle NaN: replace with column means from candidate data only
    col_means = np.nanmean(candidates_matrix, axis=0)
    for j in range(n_features):
        mask_ref = np.isnan(ref_matrix[:, j])
        ref_matrix[mask_ref, j] = col_means[j] if not np.isnan(col_means[j]) else 0
        mask_cand = np.isnan(candidates_matrix[:, j])
        candidates_matrix[mask_cand, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    # Replace NaN in centroid
    centroid = np.where(np.isnan(centroid), col_means, centroid)
    centroid = np.where(np.isnan(centroid), 0, centroid)

    # Estimate covariance using Ledoit-Wolf shrinkage on candidates ONLY.
    # This avoids circularity: the metric space is defined by the candidate
    # population's variability, while the centroid (target) is external.
    # With n=599 and p=21, the Ledoit-Wolf estimator produces a well-
    # conditioned covariance matrix without needing the reference samples.
    try:
        lw = LedoitWolf()
        lw.fit(candidates_matrix)
        cov = lw.covariance_
        cov_inv = np.linalg.pinv(cov)  # Use pseudo-inverse for stability
    except Exception as e:
        logger.warning("Ledoit-Wolf failed (%s); falling back to diagonal covariance", e)
        variances = np.var(candidates_matrix, axis=0)
        variances = np.where(variances > 0, variances, 1.0)
        cov_inv = np.diag(1.0 / variances)

    # Compute Mahalanobis distance for each candidate
    distances = np.zeros(n_candidates)
    for i in range(n_candidates):
        try:
            diff = candidates_matrix[i] - centroid
            distances[i] = np.sqrt(np.abs(diff @ cov_inv @ diff))
        except Exception:
            distances[i] = np.inf

    logger.info(
        "Mahalanobis distances: min=%.3f, median=%.3f, max=%.3f",
        np.min(distances[np.isfinite(distances)]),
        np.median(distances[np.isfinite(distances)]),
        np.max(distances[np.isfinite(distances)]),
    )
    return distances


def compute_cosine_similarities(
    candidates_matrix: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between each candidate and the UCOE centroid
    in z-score normalized feature space.

    Returns similarities in [0, 1] where 1 = most similar.
    """
    # Handle NaN
    col_means = np.nanmean(candidates_matrix, axis=0)
    for j in range(candidates_matrix.shape[1]):
        mask = np.isnan(candidates_matrix[:, j])
        candidates_matrix[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    centroid = np.where(np.isnan(centroid), col_means, centroid)
    centroid = np.where(np.isnan(centroid), 0, centroid)

    # Z-score normalize
    scaler = StandardScaler()
    combined = np.vstack([centroid.reshape(1, -1), candidates_matrix])
    scaled = scaler.fit_transform(combined)
    centroid_scaled = scaled[0]
    candidates_scaled = scaled[1:]

    # Compute cosine similarity (scipy.cosine returns *distance*, so 1 - distance = similarity)
    similarities = np.zeros(candidates_scaled.shape[0])
    for i in range(len(similarities)):
        try:
            similarities[i] = 1.0 - cosine(candidates_scaled[i], centroid_scaled)
        except Exception:
            similarities[i] = 0.0

    # Clamp to [0, 1]
    similarities = np.clip(similarities, 0, 1)

    logger.info(
        "Cosine similarities: min=%.3f, median=%.3f, max=%.3f",
        np.min(similarities), np.median(similarities), np.max(similarities),
    )
    return similarities


def compute_percentile_ranks(
    candidates_matrix: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Compute a composite percentile rank score for each candidate.

    For each feature, compute where each candidate falls relative to all others.
    Features where higher is better (active marks, DNase, CpG) use ascending percentile.
    Features where lower is better (repressive marks, methylation, CV) use descending.

    Note on implicit feature weighting: Of the 21 features, 12 are mean/CV pairs
    for 6 histone marks, giving each mark an implicit weight of 2/21 ≈ 9.5%, while
    unpaired features (CpG obs/exp, GC%, Repli-seq, CTCF, inter-TSS distance) have
    weight 1/21 ≈ 4.8%. This structure reflects the importance of cross-cell-line
    consistency (CV) alongside magnitude (mean) for each mark.

    Returns composite percentile score in [0, 1] for each candidate.
    """
    n_candidates, n_features = candidates_matrix.shape

    # Define which features are "higher is better" vs "lower is better"
    lower_is_better = {
        "H3K27me3_mean", "H3K9me3_mean",
        "meth_mean",
        # CVs: lower CV = more consistent across cell lines = better
        "H3K4me3_cv", "H3K27ac_cv", "H3K9ac_cv", "H3K36me3_cv",
        "H3K27me3_cv", "H3K9me3_cv", "meth_cv", "DNase_cv",
    }

    percentile_matrix = np.zeros_like(candidates_matrix)

    for j in range(n_features):
        col = candidates_matrix[:, j].copy()
        # Replace NaN with worst rank
        nan_mask = np.isnan(col)

        if feature_names[j] in lower_is_better:
            # Lower is better → rank ascending (lowest gets highest percentile)
            col[nan_mask] = np.inf
            ranks = col.argsort().argsort()
            percentile_matrix[:, j] = 1.0 - (ranks / (n_candidates - 1)) if n_candidates > 1 else 0.5
        else:
            # Higher is better → rank descending
            col[nan_mask] = -np.inf
            ranks = (-col).argsort().argsort()
            percentile_matrix[:, j] = 1.0 - (ranks / (n_candidates - 1)) if n_candidates > 1 else 0.5

    # Composite: mean percentile across all features
    composite = np.nanmean(percentile_matrix, axis=1)

    logger.info(
        "Percentile composite: min=%.3f, median=%.3f, max=%.3f",
        np.min(composite), np.median(composite), np.max(composite),
    )
    return composite
