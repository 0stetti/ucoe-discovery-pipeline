"""
Phase II — Validation.

1. Sanity check: all 3 known human UCOEs must rank high.
2. Leave-one-out: remove each known UCOE from reference, rank, check recovery.
"""

import logging

import numpy as np
import pandas as pd

from ucoe_pipeline.config import KNOWN_UCOES, CELL_LINES, RANKING_WEIGHTS
from ucoe_pipeline.phase2.feature_extraction import FEATURE_NAMES, extract_all_features
from ucoe_pipeline.phase2.reference_profile import build_known_ucoe_dataframe
from ucoe_pipeline.phase2.composite_score import compute_composite_scores

logger = logging.getLogger(__name__)


def sanity_check(
    scored_candidates: pd.DataFrame,
    top_n: int = 50,
) -> dict[str, dict]:
    """Check that known UCOEs appear among top-ranked candidates.

    Returns dict mapping UCOE name → {rank, score, recovered}.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION — Sanity Check")
    logger.info("=" * 60)

    results = {}
    for name, info in KNOWN_UCOES.items():
        gene_a, gene_b = info["genes"]

        # Try to find by gene names
        matches = scored_candidates[
            ((scored_candidates.get("gene1", pd.Series(dtype=str)).str.upper() == gene_a.upper())
             | (scored_candidates.get("gene2", pd.Series(dtype=str)).str.upper() == gene_a.upper())
             | (scored_candidates.get("gene1", pd.Series(dtype=str)).str.upper() == gene_b.upper())
             | (scored_candidates.get("gene2", pd.Series(dtype=str)).str.upper() == gene_b.upper()))
        ]

        # Also try coordinate overlap
        if matches.empty:
            matches = scored_candidates[
                (scored_candidates["chrom"] == info["chrom"])
                & (scored_candidates["start"] <= info["end"])
                & (scored_candidates["end"] >= info["start"])
            ]

        if not matches.empty:
            best = matches.sort_values("composite_score", ascending=False).iloc[0]
            rank = int(best.get("composite_rank", -1))
            score = float(best.get("composite_score", 0))
            recovered = rank <= top_n
            results[name] = {
                "rank": rank,
                "score": score,
                "recovered": recovered,
                "chrom": best["chrom"],
                "start": int(best["start"]),
                "end": int(best["end"]),
            }
            status = "PASS" if recovered else "WARN"
            logger.info(
                "[%s] %s — rank #%d, score=%.4f (%s:%d-%d)",
                status, name, rank, score,
                best["chrom"], best["start"], best["end"],
            )
        else:
            results[name] = {"rank": -1, "score": 0, "recovered": False}
            logger.warning("[FAIL] %s — NOT FOUND in scored candidates", name)

    n_recovered = sum(1 for v in results.values() if v["recovered"])
    logger.info(
        "Sanity check: %d / %d known UCOEs in top %d",
        n_recovered, len(results), top_n,
    )
    return results


def leave_one_out_validation(
    scored_candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
) -> pd.DataFrame:
    """Leave-one-out cross-validation on known UCOEs.

    For each known UCOE:
    1. Remove it from the reference profile
    2. Rebuild reference from remaining UCOEs
    3. Re-score all candidates
    4. Record where the excluded UCOE lands in the ranking

    Returns DataFrame with columns: excluded_ucoe, rank, score, top_10, top_50.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION — Leave-One-Out")
    logger.info("=" * 60)

    ucoe_names = list(KNOWN_UCOES.keys())
    results = []

    for excluded in ucoe_names:
        logger.info("LOO: excluding %s", excluded)

        # Build reference from remaining UCOEs
        remaining = {k: v for k, v in KNOWN_UCOES.items() if k != excluded}
        if len(remaining) < 2:
            logger.warning("Only %d UCOEs remaining — LOO may be unreliable", len(remaining))

        # Build reference DataFrame
        ref_records = []
        for name, info in remaining.items():
            ref_records.append({
                "chrom": info["chrom"],
                "start": info["start"],
                "end": info["end"],
                "gene1": info["genes"][0],
                "gene2": info["genes"][1],
                "ucoe_name": name,
                "inter_tss_distance": info["end"] - info["start"],
            })
        ref_df = pd.DataFrame(ref_records)

        # Enrich with CpG and methylation features from Phase I output
        from ucoe_pipeline.phase2.reference_profile import _enrich_from_phase1
        ref_df = _enrich_from_phase1(ref_df)

        # Add placeholder columns for any still missing
        for col in ["cpg_obs_exp", "cpg_gc_pct", "meth_mean", "meth_cv"]:
            if col not in ref_df.columns:
                ref_df[col] = np.nan

        ref_features = extract_all_features(ref_df, cell_lines)
        available_features = [f for f in FEATURE_NAMES if f in ref_features.columns]
        ref_matrix = ref_features[available_features].values.astype(float)
        centroid = np.nanmean(ref_matrix, axis=0)

        # Re-score candidates
        rescored = compute_composite_scores(
            scored_candidates.drop(columns=[
                "mahalanobis_dist", "mahalanobis_score",
                "cosine_sim", "cosine_score",
                "percentile_score", "composite_score", "composite_rank",
            ], errors="ignore"),
            ref_matrix, centroid, RANKING_WEIGHTS,
        )

        # Find the excluded UCOE in the ranking
        excl_info = KNOWN_UCOES[excluded]
        gene_a, gene_b = excl_info["genes"]

        # Check all 4 gene-column combinations
        matches = rescored[
            (rescored.get("gene1", pd.Series(dtype=str)).str.upper() == gene_a.upper())
            | (rescored.get("gene2", pd.Series(dtype=str)).str.upper() == gene_a.upper())
            | (rescored.get("gene1", pd.Series(dtype=str)).str.upper() == gene_b.upper())
            | (rescored.get("gene2", pd.Series(dtype=str)).str.upper() == gene_b.upper())
        ]

        # Fallback: coordinate overlap
        if matches.empty:
            matches = rescored[
                (rescored["chrom"] == excl_info["chrom"])
                & (rescored["start"] <= excl_info["end"])
                & (rescored["end"] >= excl_info["start"])
            ]

        if not matches.empty:
            best = matches.sort_values("composite_score", ascending=False).iloc[0]
            rank = int(best.get("composite_rank", -1))
            score = float(best.get("composite_score", 0))
        else:
            rank = -1
            score = 0.0

        results.append({
            "excluded_ucoe": excluded,
            "rank": rank,
            "score": score,
            "top_10": rank <= 10 and rank > 0,
            "top_50": rank <= 50 and rank > 0,
        })
        logger.info("  LOO result: %s → rank #%d (score=%.4f)", excluded, rank, score)

    loo_df = pd.DataFrame(results)
    logger.info("\nLeave-one-out summary:")
    for _, row in loo_df.iterrows():
        logger.info(
            "  %s: rank #%d, score=%.4f, top-10=%s, top-50=%s",
            row["excluded_ucoe"], row["rank"], row["score"],
            row["top_10"], row["top_50"],
        )

    return loo_df
