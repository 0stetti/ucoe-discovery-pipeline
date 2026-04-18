"""
Phase II — Feature extraction.

Extract the full feature vector for each candidate region and for known UCOEs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ucoe_pipeline.config import (
    ALL_MARKS_FOR_RANKING,
    CELL_LINES,
    CHIPSEQ_DIR,
    DNASE_DIR,
    METHYLATION_DIR,
    REPLISEQ_DIR,
)
from ucoe_pipeline.utils.bigwig_utils import extract_signal_batch, find_bigwig_files

logger = logging.getLogger(__name__)

# Feature names used in the ranking (order matters for consistency)
FEATURE_NAMES = [
    "H3K4me3_mean", "H3K4me3_cv",
    "H3K27ac_mean", "H3K27ac_cv",
    "H3K9ac_mean", "H3K9ac_cv",
    "H3K36me3_mean", "H3K36me3_cv",
    "H3K27me3_mean", "H3K27me3_cv",
    "H3K9me3_mean", "H3K9me3_cv",
    "meth_mean", "meth_cv",
    "cpg_obs_exp", "cpg_gc_pct",
    "DNase_mean", "DNase_cv",
    "repliseq_mean",
    "CTCF_n_peaks",
    "inter_tss_distance",
]


def extract_histone_features(
    regions: list[tuple[str, int, int]],
    cell_lines: list[str] = CELL_LINES,
    chipseq_dir: Path = CHIPSEQ_DIR,
) -> pd.DataFrame:
    """Extract mean and CV for all histone marks across cell lines."""
    features = {}

    for mark in ALL_MARKS_FOR_RANKING:
        mark_dir = chipseq_dir / mark
        bw_files = find_bigwig_files(mark_dir, cell_lines, mark)

        if not bw_files:
            logger.warning("No bigWig files for %s", mark)
            features[f"{mark}_mean"] = [np.nan] * len(regions)
            features[f"{mark}_cv"] = [np.nan] * len(regions)
            continue

        # Extract signal for each cell line
        cl_signals = []
        for cl, bw_path in bw_files.items():
            signals = extract_signal_batch(bw_path, regions)
            cl_signals.append(signals)

        # Convert to array: (n_regions, n_cell_lines)
        arr = np.array(cl_signals, dtype=float).T  # shape: (n_regions, n_cls)
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=1)
            stds = np.nanstd(arr, axis=1)
            cvs = np.where(means > 0, stds / means, np.nan)

        features[f"{mark}_mean"] = means
        features[f"{mark}_cv"] = cvs

    return pd.DataFrame(features)


def extract_dnase_features(
    regions: list[tuple[str, int, int]],
    cell_lines: list[str] = CELL_LINES,
    dnase_dir: Path = DNASE_DIR,
) -> pd.DataFrame:
    """Extract DNase-seq mean and CV across cell lines."""
    signal_dir = dnase_dir / "signal"
    bw_files = find_bigwig_files(signal_dir, cell_lines, "DNase")
    if not bw_files:
        bw_files = find_bigwig_files(dnase_dir, cell_lines, "DNase")

    if not bw_files:
        return pd.DataFrame({
            "DNase_mean": [np.nan] * len(regions),
            "DNase_cv": [np.nan] * len(regions),
        })

    cl_signals = []
    for cl, bw_path in bw_files.items():
        signals = extract_signal_batch(bw_path, regions)
        cl_signals.append(signals)

    arr = np.array(cl_signals, dtype=float).T
    with np.errstate(all="ignore"):
        means = np.nanmean(arr, axis=1)
        stds = np.nanstd(arr, axis=1)
        cvs = np.where(means > 0, stds / means, np.nan)

    return pd.DataFrame({"DNase_mean": means, "DNase_cv": cvs})


def extract_repliseq_features(
    regions: list[tuple[str, int, int]],
    cell_lines: list[str] = CELL_LINES,
    repliseq_dir: Path = REPLISEQ_DIR,
) -> pd.DataFrame:
    """Extract mean Repli-seq E/L ratio across cell lines."""
    bw_files = find_bigwig_files(repliseq_dir, cell_lines, "Repli")
    # Also try without target filter (files might just be named by cell line)
    if not bw_files and repliseq_dir.exists():
        for bw in list(repliseq_dir.glob("*.bigWig")) + list(repliseq_dir.glob("*.bigwig")):
            for cl in cell_lines:
                if cl in bw.stem:
                    bw_files[cl] = bw
                    break

    if not bw_files:
        return pd.DataFrame({"repliseq_mean": [np.nan] * len(regions)})

    cl_signals = []
    for cl, bw_path in bw_files.items():
        signals = extract_signal_batch(bw_path, regions)
        cl_signals.append(signals)

    arr = np.array(cl_signals, dtype=float).T
    means = np.nanmean(arr, axis=1)

    return pd.DataFrame({"repliseq_mean": means})


def extract_ctcf_peaks(
    regions: list[tuple[str, int, int]],
    chipseq_dir: Path = CHIPSEQ_DIR,
    flank: int = 2000,
) -> pd.DataFrame:
    """Count CTCF binding peaks in/around each candidate region.

    Extends each region by `flank` bp on each side for the count.
    """
    import pybedtools

    peaks_dir = chipseq_dir / "CTCF" / "peaks"
    if not peaks_dir.exists():
        return pd.DataFrame({"CTCF_n_peaks": [0] * len(regions)})

    peak_files = list(peaks_dir.glob("*.narrowPeak*")) + list(peaks_dir.glob("*.bed*"))
    if not peak_files:
        return pd.DataFrame({"CTCF_n_peaks": [0] * len(regions)})

    # Merge all CTCF peak files
    all_peaks = None
    for pf in peak_files:
        bt = pybedtools.BedTool(str(pf))
        all_peaks = bt if all_peaks is None else all_peaks.cat(bt)

    # Create flanked regions
    cand_lines = []
    for i, (chrom, start, end) in enumerate(regions):
        flanked_start = max(0, start - flank)
        flanked_end = end + flank
        cand_lines.append(f"{chrom}\t{flanked_start}\t{flanked_end}\t{i}")
    cand_bt = pybedtools.BedTool("\n".join(cand_lines), from_string=True)

    # Count overlapping peaks
    counted = cand_bt.intersect(all_peaks, c=True)
    counts = [0] * len(regions)
    for interval in counted:
        idx = int(interval.fields[3])
        count = int(interval.fields[4])
        counts[idx] = count

    return pd.DataFrame({"CTCF_n_peaks": counts})


def extract_all_features(
    candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
) -> pd.DataFrame:
    """Extract the complete feature vector for all candidate regions.

    Expects candidates to already have columns from Phase I filters
    (cpg_obs_exp, cpg_gc_pct, meth_mean, meth_cv, inter_tss_distance, etc.).
    This function adds/overwrites the histone, DNase, Repli-seq, and CTCF features.
    """
    logger.info("=" * 60)
    logger.info("PHASE II — Feature Extraction")
    logger.info("=" * 60)

    regions = list(zip(candidates["chrom"], candidates["start"], candidates["end"]))
    n = len(regions)
    logger.info("Extracting features for %d candidate regions", n)

    # Extract each feature group
    histone_df = extract_histone_features(regions, cell_lines)
    dnase_df = extract_dnase_features(regions, cell_lines)
    repliseq_df = extract_repliseq_features(regions, cell_lines)
    ctcf_df = extract_ctcf_peaks(regions)

    # Merge all features into candidates
    result = candidates.reset_index(drop=True).copy()
    for df in [histone_df, dnase_df, repliseq_df, ctcf_df]:
        df = df.reset_index(drop=True)
        for col in df.columns:
            result[col] = df[col].values

    # Ensure inter_tss_distance is present
    if "inter_tss_distance" not in result.columns:
        result["inter_tss_distance"] = 0

    logger.info("Feature extraction complete. Shape: %s", result.shape)

    # Report how many features have data
    for feat in FEATURE_NAMES:
        if feat in result.columns:
            n_valid = result[feat].notna().sum()
            logger.info("  %s: %d / %d regions with data", feat, n_valid, n)

    return result
