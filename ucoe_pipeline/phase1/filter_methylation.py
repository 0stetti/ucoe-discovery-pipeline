"""
Filter 5 — Constitutive hypomethylation.

Require mean DNA methylation < 10% in ≥80% of cell lines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pybedtools

from ucoe_pipeline.config import (
    CELL_LINES,
    METHYLATION_DIR,
    HYPOMETHYLATION_THRESHOLD,
    UBIQUITY_FRACTION,
)

logger = logging.getLogger(__name__)


def find_methylation_files(
    meth_dir: Path,
    cell_lines: list[str],
) -> dict[str, Path]:
    """Find WGBS/RRBS BED files per cell line.

    Expects naming like: {CellLine}_wgbs_{accession}.bed.gz or similar.
    Downloads produce: GM12878_wgbs_ENCFF123ABC.bed.gz
    """
    found = {}
    if not meth_dir.exists():
        logger.warning("Methylation directory not found: %s", meth_dir)
        return found

    bed_files = list(meth_dir.iterdir())
    # Accept .bed, .bed.gz, .bedGraph, .bedGraph.gz extensions
    bed_files = [
        f for f in bed_files
        if f.is_file() and any(
            f.name.endswith(ext)
            for ext in (".bed", ".bed.gz", ".bedGraph", ".bedGraph.gz")
        )
    ]
    for bf in bed_files:
        # Strip all extensions to get the base name
        name = bf.name.split(".")[0]  # e.g. "GM12878_wgbs_ENCFF123ABC"
        for cl in cell_lines:
            if name.startswith(cl):
                found[cl] = bf
                break
    if found:
        logger.info("Found %d methylation files in %s", len(found), meth_dir)
    return found


def extract_methylation_for_regions(
    bed_path: Path,
    candidates: pd.DataFrame,
) -> list[float | None]:
    """Extract mean methylation level for each candidate region from a BED file.

    ENCODE WGBS/RRBS BED format typically has the methylation percentage
    in column 11 (0-indexed col 10). We compute the mean percentage of all
    CpGs falling within each candidate region.
    """
    # Create candidate BedTool
    cand_lines = []
    for idx, row in candidates.iterrows():
        cand_lines.append(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{idx}")
    cand_bt = pybedtools.BedTool("\n".join(cand_lines), from_string=True)

    # Load methylation BED
    meth_bt = pybedtools.BedTool(str(bed_path))

    # Intersect: for each CpG site, find which candidate it falls in
    intersected = cand_bt.intersect(meth_bt, wa=True, wb=True)

    # Collect methylation values per candidate index
    meth_by_idx = {}
    for interval in intersected:
        fields = interval.fields
        try:
            cand_idx = int(fields[3])
            # Methylation percentage is typically in the last few columns
            # Try column index 14 (0-based) which is col 11 of the methylation file
            # (4 candidate cols + methylation cols starting at index 4)
            # The percentage column varies; try common positions
            meth_pct = None
            for col_offset in [14, 13, 12, 11, 10]:
                if col_offset < len(fields):
                    try:
                        val = float(fields[col_offset])
                        if 0 <= val <= 100:
                            meth_pct = val
                            break
                    except ValueError:
                        continue
            if meth_pct is not None:
                meth_by_idx.setdefault(cand_idx, []).append(meth_pct)
        except (ValueError, IndexError):
            continue

    # Compute mean methylation per candidate
    results = []
    for idx in candidates.index:
        if idx in meth_by_idx and meth_by_idx[idx]:
            results.append(np.mean(meth_by_idx[idx]))
        else:
            results.append(None)

    return results


def run_filter5(
    candidates: pd.DataFrame,
    cell_lines: list[str] = CELL_LINES,
    meth_dir: Path = METHYLATION_DIR,
    threshold: float = HYPOMETHYLATION_THRESHOLD,
    ubiquity: float = UBIQUITY_FRACTION,
) -> pd.DataFrame:
    """Filter 5: Require constitutive hypomethylation.

    Mean methylation must be < threshold in ≥ ubiquity fraction of cell lines.
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 5: Constitutive Hypomethylation")
    logger.info("=" * 60)

    meth_files = find_methylation_files(meth_dir, cell_lines)
    if not meth_files:
        logger.warning("No methylation files found — skipping Filter 5")
        return candidates

    result = candidates.copy()

    meth_data = {}
    for cl, path in meth_files.items():
        logger.info("Extracting methylation for %s from %s", cl, path.name)
        meth_data[cl] = extract_methylation_for_regions(path, result)

    # Build methylation matrix
    meth_df = pd.DataFrame(
        {f"meth_{cl}": vals for cl, vals in meth_data.items()},
        index=result.index,
    )

    # Replace None with NaN
    meth_df = meth_df.where(meth_df.notna(), np.nan)
    values = meth_df.values.astype(float)

    # Count cell lines where methylation < threshold
    hypo_matrix = values < threshold
    n_available = np.sum(~np.isnan(values), axis=1)
    n_hypo = np.nansum(hypo_matrix, axis=1)
    frac_hypo = np.where(n_available > 0, n_hypo / n_available, 0.0)

    result["meth_mean"] = np.nanmean(values, axis=1)
    with np.errstate(all="ignore"):
        std = np.nanstd(values, axis=1)
        mean = np.nanmean(values, axis=1)
        result["meth_cv"] = np.where(mean > 0, std / mean, np.nan)
    result["meth_n_hypo"] = n_hypo.astype(int)
    result["meth_frac_hypo"] = frac_hypo

    before = len(result)
    result = result[result["meth_frac_hypo"] >= ubiquity].copy()
    logger.info(
        "Filter 5: %d → %d candidates (methylation < %.1f%% in ≥ %.0f%% of cell lines)",
        before, len(result), threshold, ubiquity * 100,
    )

    return result
