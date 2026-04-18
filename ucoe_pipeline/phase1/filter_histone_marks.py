"""
Filters 3 & 4 — Histone mark filtering.

Filter 3: Ubiquitous active histone marks (H3K4me3, H3K27ac) present in ≥80% of cell lines.
Filter 4: Absence of repressive marks (H3K27me3, H3K9me3) in ≥80% of cell lines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ucoe_pipeline.config import (
    CELL_LINES,
    CHIPSEQ_DIR,
    ACTIVE_MARKS,
    REPRESSIVE_MARKS,
    SIGNAL_PRESENT_THRESHOLD,
    SIGNAL_ABSENT_THRESHOLD,
    UBIQUITY_FRACTION,
)
from ucoe_pipeline.utils.bigwig_utils import extract_signal_batch, find_bigwig_files

logger = logging.getLogger(__name__)


def _extract_mark_signals(
    candidates: pd.DataFrame,
    mark: str,
    cell_lines: list[str],
    chipseq_dir: Path,
) -> pd.DataFrame:
    """Extract bigWig signal for one histone mark across all cell lines.

    Returns DataFrame indexed like candidates with columns:
    {mark}_{cell_line} for each cell line, plus {mark}_mean, {mark}_cv,
    {mark}_n_present, {mark}_frac_present.
    """
    mark_dir = chipseq_dir / mark
    bw_files = find_bigwig_files(mark_dir, cell_lines, mark)

    regions = list(zip(candidates["chrom"], candidates["start"], candidates["end"]))

    signal_cols = {}
    for cl in cell_lines:
        if cl not in bw_files:
            continue
        signals = extract_signal_batch(bw_files[cl], regions)
        signal_cols[f"{mark}_{cl}"] = signals

    if not signal_cols:
        logger.warning("No bigWig files found for %s — skipping", mark)
        return pd.DataFrame(index=candidates.index)

    sig_df = pd.DataFrame(signal_cols, index=candidates.index)

    # Replace None with NaN
    sig_df = sig_df.where(sig_df.notna(), np.nan)

    # Compute summary stats across cell lines (ignore NaN)
    values = sig_df.values.astype(float)
    with np.errstate(all="ignore"):
        mean_vals = np.nanmean(values, axis=1)
        std_vals = np.nanstd(values, axis=1)
        cv_vals = np.where(mean_vals > 0, std_vals / mean_vals, np.nan)

    sig_df[f"{mark}_mean"] = mean_vals
    sig_df[f"{mark}_cv"] = cv_vals

    return sig_df


def run_filter3(
    candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
    chipseq_dir: Path = CHIPSEQ_DIR,
    threshold: float = SIGNAL_PRESENT_THRESHOLD,
    ubiquity: float = UBIQUITY_FRACTION,
) -> pd.DataFrame:
    """Filter 3: Require ubiquitous active histone marks.

    For each active mark (H3K4me3, H3K27ac), the signal must exceed
    threshold in ≥ ubiquity fraction of cell lines.
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 3: Ubiquitous Active Histone Marks")
    logger.info("=" * 60)

    result = candidates.copy()

    for mark in ACTIVE_MARKS:
        sig_df = _extract_mark_signals(result, mark, cell_lines, chipseq_dir)
        if sig_df.empty:
            logger.warning("Skipping %s — no data available", mark)
            continue

        # Count cell lines where signal > threshold
        cl_cols = [c for c in sig_df.columns if c.startswith(f"{mark}_") and c != f"{mark}_mean" and c != f"{mark}_cv"]
        if not cl_cols:
            continue

        present_matrix = sig_df[cl_cols].values.astype(float) > threshold
        n_available = np.sum(~np.isnan(sig_df[cl_cols].values.astype(float)), axis=1)
        n_present = np.nansum(present_matrix, axis=1)
        frac_present = np.where(n_available > 0, n_present / n_available, 0.0)

        result[f"{mark}_n_present"] = n_present.astype(int)
        result[f"{mark}_frac_present"] = frac_present
        result[f"{mark}_mean"] = sig_df[f"{mark}_mean"].values
        result[f"{mark}_cv"] = sig_df[f"{mark}_cv"].values

        # Filter: require ≥ ubiquity fraction
        before = len(result)
        result = result[result[f"{mark}_frac_present"] >= ubiquity].copy()
        logger.info(
            "Filter 3 (%s): %d → %d candidates (signal > %.1f in ≥ %.0f%% of cell lines)",
            mark, before, len(result), threshold, ubiquity * 100,
        )

    return result


def run_filter4(
    candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
    chipseq_dir: Path = CHIPSEQ_DIR,
    threshold: float = SIGNAL_ABSENT_THRESHOLD,
    ubiquity: float = UBIQUITY_FRACTION,
) -> pd.DataFrame:
    """Filter 4: Require absence of repressive histone marks.

    For each repressive mark (H3K27me3, H3K9me3), the signal must be
    BELOW threshold in ≥ ubiquity fraction of cell lines.
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 4: Absence of Repressive Histone Marks")
    logger.info("=" * 60)

    result = candidates.copy()

    for mark in REPRESSIVE_MARKS:
        sig_df = _extract_mark_signals(result, mark, cell_lines, chipseq_dir)
        if sig_df.empty:
            logger.warning("Skipping %s — no data available", mark)
            continue

        cl_cols = [c for c in sig_df.columns if c.startswith(f"{mark}_") and c != f"{mark}_mean" and c != f"{mark}_cv"]
        if not cl_cols:
            continue

        # For repressive marks: "absent" means signal < threshold
        absent_matrix = sig_df[cl_cols].values.astype(float) < threshold
        n_available = np.sum(~np.isnan(sig_df[cl_cols].values.astype(float)), axis=1)
        n_absent = np.nansum(absent_matrix, axis=1)
        frac_absent = np.where(n_available > 0, n_absent / n_available, 0.0)

        result[f"{mark}_n_absent"] = n_absent.astype(int)
        result[f"{mark}_frac_absent"] = frac_absent
        result[f"{mark}_mean"] = sig_df[f"{mark}_mean"].values
        result[f"{mark}_cv"] = sig_df[f"{mark}_cv"].values

        before = len(result)
        result = result[result[f"{mark}_frac_absent"] >= ubiquity].copy()
        logger.info(
            "Filter 4 (%s): %d → %d candidates (signal < %.1f in ≥ %.0f%% of cell lines)",
            mark, before, len(result), threshold, ubiquity * 100,
        )

    return result
