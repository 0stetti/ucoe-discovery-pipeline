"""
Filter 6 — Ubiquitous DNase I hypersensitivity.

Require DNase-seq signal present in ≥80% of cell lines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ucoe_pipeline.config import (
    CELL_LINES,
    DNASE_DIR,
    DNASE_PRESENT_THRESHOLD,
    UBIQUITY_FRACTION,
)
from ucoe_pipeline.utils.bigwig_utils import extract_signal_batch, find_bigwig_files

logger = logging.getLogger(__name__)


def run_filter6(
    candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
    dnase_dir: Path = DNASE_DIR,
    threshold: float = DNASE_PRESENT_THRESHOLD,
    ubiquity: float = UBIQUITY_FRACTION,
) -> pd.DataFrame:
    """Filter 6: Require ubiquitous DNase accessibility.

    DNase-seq signal must exceed threshold in ≥ ubiquity fraction of cell lines.
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 6: Ubiquitous DNase Accessibility")
    logger.info("=" * 60)

    signal_dir = dnase_dir / "signal"
    bw_files = find_bigwig_files(signal_dir, cell_lines, "DNase")

    # Also try finding files directly in dnase_dir if signal/ subdir is empty
    if not bw_files:
        bw_files = find_bigwig_files(dnase_dir, cell_lines, "DNase")
    if not bw_files:
        logger.warning("No DNase bigWig files found — skipping Filter 6")
        return candidates

    result = candidates.copy()
    regions = list(zip(result["chrom"], result["start"], result["end"]))

    signal_cols = {}
    for cl, bw_path in bw_files.items():
        signals = extract_signal_batch(bw_path, regions)
        signal_cols[f"DNase_{cl}"] = signals

    sig_df = pd.DataFrame(signal_cols, index=result.index)
    sig_df = sig_df.where(sig_df.notna(), np.nan)
    values = sig_df.values.astype(float)

    # Count cell lines where DNase signal > threshold
    present_matrix = values > threshold
    n_available = np.sum(~np.isnan(values), axis=1)
    n_present = np.nansum(present_matrix, axis=1)
    frac_present = np.where(n_available > 0, n_present / n_available, 0.0)

    result["DNase_mean"] = np.nanmean(values, axis=1)
    with np.errstate(all="ignore"):
        std = np.nanstd(values, axis=1)
        mean = np.nanmean(values, axis=1)
        result["DNase_cv"] = np.where(mean > 0, std / mean, np.nan)
    result["DNase_n_present"] = n_present.astype(int)
    result["DNase_frac_present"] = frac_present

    before = len(result)
    result = result[result["DNase_frac_present"] >= ubiquity].copy()
    logger.info(
        "Filter 6: %d → %d candidates (DNase signal > %.1f in ≥ %.0f%% of cell lines)",
        before, len(result), threshold, ubiquity * 100,
    )

    return result
