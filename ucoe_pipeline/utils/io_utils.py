"""File I/O and general utilities."""

import gzip
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_gzipped_or_plain(filepath: Path) -> list[str]:
    """Read a file that may or may not be gzipped. Returns list of lines."""
    opener = gzip.open if filepath.suffix == ".gz" else open
    with opener(filepath, "rt") as f:
        return f.readlines()


def save_candidates(
    df: pd.DataFrame,
    output_path: Path,
    step_name: str,
) -> Path:
    """Save candidate regions to TSV and log the count."""
    ensure_dir(output_path.parent)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("%s: saved %d candidates to %s", step_name, len(df), output_path)
    return output_path


def save_bed(
    df: pd.DataFrame,
    output_path: Path,
    name_col: str | None = None,
) -> Path:
    """Save a DataFrame as a BED file (chrom, start, end, name, score, strand)."""
    ensure_dir(output_path.parent)
    bed_df = df[["chrom", "start", "end"]].copy()
    if name_col and name_col in df.columns:
        bed_df["name"] = df[name_col]
    else:
        bed_df["name"] = "."
    bed_df["score"] = 0
    bed_df["strand"] = "."
    bed_df.to_csv(output_path, sep="\t", header=False, index=False)
    logger.info("Saved BED file with %d regions to %s", len(bed_df), output_path)
    return output_path
