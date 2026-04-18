"""Utilities for genomic interval operations using pybedtools."""

import logging
from pathlib import Path

import pandas as pd
import pybedtools

logger = logging.getLogger(__name__)


def regions_to_bedtool(df: pd.DataFrame) -> pybedtools.BedTool:
    """Convert a DataFrame with chrom/start/end columns to a BedTool.

    Expects at minimum columns: chrom, start, end.
    Additional columns are preserved.
    """
    cols = ["chrom", "start", "end"] + [
        c for c in df.columns if c not in ("chrom", "start", "end")
    ]
    bed_str = df[cols].to_csv(sep="\t", header=False, index=False)
    return pybedtools.BedTool(bed_str, from_string=True)


def bedtool_to_df(bt: pybedtools.BedTool, names: list[str] | None = None) -> pd.DataFrame:
    """Convert a BedTool back to a DataFrame."""
    if names is None:
        names = ["chrom", "start", "end"]
    df = bt.to_dataframe(names=names)
    return df


def compute_overlap_fraction(
    regions_a: pybedtools.BedTool,
    regions_b: pybedtools.BedTool,
) -> pd.DataFrame:
    """For each region in A, compute the fraction overlapping with any region in B.

    Returns DataFrame with columns: chrom, start, end, overlap_fraction.
    """
    # Use intersect with -wao to get overlap base pairs
    result = regions_a.intersect(regions_b, wao=True)
    records = []
    for interval in result:
        fields = interval.fields
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        overlap_bp = int(fields[-1])
        region_len = end - start
        frac = overlap_bp / region_len if region_len > 0 else 0.0
        records.append((chrom, start, end, frac))

    df = pd.DataFrame(records, columns=["chrom", "start", "end", "overlap_fraction"])
    # A region may appear multiple times if it overlaps multiple B regions; take max
    df = df.groupby(["chrom", "start", "end"], as_index=False)["overlap_fraction"].sum()
    # Cap at 1.0
    df["overlap_fraction"] = df["overlap_fraction"].clip(upper=1.0)
    return df
