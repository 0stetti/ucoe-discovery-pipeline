"""Utilities for extracting signal from bigWig files using pyBigWig."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyBigWig

logger = logging.getLogger(__name__)


def extract_mean_signal(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
) -> Optional[float]:
    """Extract mean signal over a genomic region from a bigWig file.

    Returns None if the chromosome is missing or the query fails.
    """
    try:
        bw = pyBigWig.open(str(bw_path))
        chroms = bw.chroms()
        if chrom not in chroms:
            bw.close()
            return None
        # Clamp to chromosome length
        chrom_len = chroms[chrom]
        end = min(end, chrom_len)
        if start >= end:
            bw.close()
            return None
        val = bw.stats(chrom, start, end, type="mean")[0]
        bw.close()
        return val
    except Exception as e:
        logger.warning("Error reading %s at %s:%d-%d: %s", bw_path, chrom, start, end, e)
        return None


def extract_signal_batch(
    bw_path: str | Path,
    regions: list[tuple[str, int, int]],
) -> list[Optional[float]]:
    """Extract mean signal for a batch of regions from one bigWig file.

    Parameters
    ----------
    regions : list of (chrom, start, end)

    Returns
    -------
    list of float or None, one per region.
    """
    results = []
    try:
        bw = pyBigWig.open(str(bw_path))
        chroms = bw.chroms()
        for chrom, start, end in regions:
            if chrom not in chroms:
                results.append(None)
                continue
            chrom_len = chroms[chrom]
            clamped_end = min(end, chrom_len)
            if start >= clamped_end:
                results.append(None)
                continue
            val = bw.stats(chrom, start, clamped_end, type="mean")[0]
            results.append(val)
        bw.close()
    except Exception as e:
        logger.warning("Error reading %s: %s", bw_path, e)
        # Pad remaining with None
        results.extend([None] * (len(regions) - len(results)))
    return results


def find_bigwig_files(
    directory: Path,
    cell_lines: list[str],
    target: str,
) -> dict[str, Path]:
    """Find bigWig files matching cell lines for a given target (e.g. H3K4me3).

    Expects naming convention: {CellLine}_{Target}_{ENCFFaccession}.bigWig
    Returns dict mapping cell_line -> file path.
    """
    found = {}
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return found

    bw_files = list(directory.glob("*.bigWig")) + list(directory.glob("*.bigwig"))
    for bw in bw_files:
        name = bw.stem  # e.g. GM12878_H3K4me3_ENCFF123ABC
        for cl in cell_lines:
            if name.startswith(cl) and target in name:
                found[cl] = bw
                break
    if found:
        logger.info("Found %d bigWig files for %s in %s", len(found), target, directory)
    return found
