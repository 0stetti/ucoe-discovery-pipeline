"""
Phase II — Build reference profile from known UCOEs.

Extract the same features for the 3 known human UCOEs and compute
the reference centroid and covariance for similarity metrics.

When a Phase I output is available, CpG and methylation features
are pulled from there (since those are extracted during Phase I).
Otherwise, CpG and methylation features are extracted fresh.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ucoe_pipeline.config import KNOWN_UCOES, CELL_LINES, OUTPUT_DIR
from ucoe_pipeline.phase2.feature_extraction import (
    FEATURE_NAMES,
    extract_all_features,
)

logger = logging.getLogger(__name__)


def build_known_ucoe_dataframe() -> pd.DataFrame:
    """Create a DataFrame of known UCOE regions for feature extraction."""
    records = []
    for name, info in KNOWN_UCOES.items():
        records.append({
            "chrom": info["chrom"],
            "start": info["start"],
            "end": info["end"],
            "gene1": info["genes"][0],
            "gene2": info["genes"][1],
            "ucoe_name": name,
            "inter_tss_distance": info["end"] - info["start"],
        })
    return pd.DataFrame(records)


def _enrich_from_phase1(ucoe_df: pd.DataFrame) -> pd.DataFrame:
    """Try to pull CpG and methylation features from Phase I output.

    Matches known UCOEs to Phase I candidates by gene names and copies
    over columns that are present in Phase I but not yet in ucoe_df.
    """
    phase1_path = OUTPUT_DIR / "phase1" / "phase1_final.tsv"
    if not phase1_path.exists():
        # Try alternate path
        phase1_path = OUTPUT_DIR / "phase1" / "after_filter5.tsv"
    if not phase1_path.exists():
        logger.info("No Phase I output found for enrichment — using fresh extraction")
        return ucoe_df

    phase1 = pd.read_csv(phase1_path, sep="\t")
    result = ucoe_df.copy()

    cols_to_copy = [
        "cpg_overlap_fraction", "cpg_obs_exp", "cpg_gc_pct",
        "meth_mean", "meth_cv", "meth_n_hypo", "meth_frac_hypo",
    ]

    for idx, row in result.iterrows():
        gene1 = row["gene1"]
        gene2 = row["gene2"]

        # Find matching row in Phase I output
        match = phase1[
            ((phase1["gene1"] == gene1) | (phase1["gene1"] == gene2)) &
            ((phase1["gene2"] == gene1) | (phase1["gene2"] == gene2))
        ]

        if match.empty:
            # Try coordinate overlap
            match = phase1[
                (phase1["chrom"] == row["chrom"]) &
                (phase1["start"] <= row["end"]) &
                (phase1["end"] >= row["start"])
            ]

        if not match.empty:
            best = match.iloc[0]
            for col in cols_to_copy:
                if col in best.index:
                    result.at[idx, col] = best[col]
            logger.info(
                "Enriched %s from Phase I (cpg_obs_exp=%.2f, meth_mean=%.1f%%)",
                row.get("ucoe_name", f"{gene1}/{gene2}"),
                best.get("cpg_obs_exp", float("nan")),
                best.get("meth_mean", float("nan")),
            )

    return result


def build_reference_profile(
    cell_lines: list[str] = CELL_LINES,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build the reference profile from known human UCOEs.

    Returns
    -------
    ref_df : DataFrame with features for each known UCOE
    centroid : mean feature vector (1D array)
    ref_matrix : feature matrix for known UCOEs (n_ucoes × n_features)
    """
    logger.info("Building reference profile from %d known UCOEs", len(KNOWN_UCOES))

    ucoe_df = build_known_ucoe_dataframe()

    # Enrich with CpG and methylation data from Phase I output
    ucoe_df = _enrich_from_phase1(ucoe_df)

    # Add placeholder columns for any still missing
    for col in ["cpg_obs_exp", "cpg_gc_pct", "meth_mean", "meth_cv"]:
        if col not in ucoe_df.columns:
            ucoe_df[col] = np.nan

    ref_df = extract_all_features(ucoe_df, cell_lines)

    # Build feature matrix using only the defined feature names
    available_features = [f for f in FEATURE_NAMES if f in ref_df.columns]
    ref_matrix = ref_df[available_features].values.astype(float)
    centroid = np.nanmean(ref_matrix, axis=0)

    logger.info("Reference centroid computed from %d UCOEs, %d features",
                ref_matrix.shape[0], ref_matrix.shape[1])

    # Log the reference values
    for i, feat in enumerate(available_features):
        vals = ref_matrix[:, i]
        logger.info("  %s: mean=%.3f (values: %s)", feat, centroid[i],
                     ", ".join(f"{v:.3f}" for v in vals if not np.isnan(v)))

    return ref_df, centroid, ref_matrix
