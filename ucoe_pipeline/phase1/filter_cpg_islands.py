"""
Filter 2 — CpG island overlap and region extension.

Step 1: Extend each candidate region (inter-TSS) to encompass overlapping
        or flanking CpG islands within a small window. This creates the
        biologically relevant UCOE candidate region (CpG island at the
        bidirectional promoter).
Step 2: Require that the extended region overlaps at least one CpG island.
        Annotate with CpG metrics (GC%, obs/exp ratio, overlap fraction).
"""

import logging
from pathlib import Path

import pandas as pd
import pybedtools

from ucoe_pipeline.config import CPG_ISLANDS_FILE, CPG_OVERLAP_FRACTION
from ucoe_pipeline.utils.io_utils import read_gzipped_or_plain

logger = logging.getLogger(__name__)

# Maximum distance to search for flanking CpG islands beyond the inter-TSS
# boundaries. This allows capturing CpG islands that extend past each TSS.
CPG_FLANK_WINDOW = 500  # bp


def load_cpg_islands(cpg_path: Path = CPG_ISLANDS_FILE) -> pybedtools.BedTool:
    """Load UCSC CpG islands into a BedTool.

    cpgIslandExt.txt.gz columns:
    bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum,
    perCpg, perGc, obsExp
    """
    logger.info("Loading CpG islands from %s", cpg_path)
    lines = read_gzipped_or_plain(cpg_path)

    bed_lines = []
    for line in lines:
        if line.startswith("#") or line.startswith("bin"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 11:
            continue
        chrom = fields[1]
        start = fields[2]
        end = fields[3]
        name = fields[4]
        obs_exp = fields[10]
        per_gc = fields[9]
        bed_lines.append(f"{chrom}\t{start}\t{end}\t{name}\t{per_gc}\t{obs_exp}")

    bed_str = "\n".join(bed_lines)
    bt = pybedtools.BedTool(bed_str, from_string=True)
    logger.info("Loaded %d CpG islands", len(bed_lines))
    return bt


def load_cpg_islands_df(cpg_path: Path = CPG_ISLANDS_FILE) -> pd.DataFrame:
    """Load CpG islands into a DataFrame for region extension logic."""
    lines = read_gzipped_or_plain(cpg_path)
    records = []
    for line in lines:
        if line.startswith("#") or line.startswith("bin"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 11:
            continue
        records.append({
            "chrom": fields[1],
            "cpg_start": int(fields[2]),
            "cpg_end": int(fields[3]),
            "cpg_name": fields[4],
            "cpg_gc_pct": float(fields[9]),
            "cpg_obs_exp": float(fields[10]),
        })
    return pd.DataFrame(records)


def extend_regions_to_cpg_islands(
    candidates: pd.DataFrame,
    cpg_df: pd.DataFrame,
    flank: int = CPG_FLANK_WINDOW,
) -> pd.DataFrame:
    """Extend candidate inter-TSS regions to encompass overlapping CpG islands.

    For each candidate, find all CpG islands that overlap or are within
    `flank` bp of the inter-TSS region. The candidate region is then
    extended to the union of the inter-TSS region and all such CpG islands.

    This is critical because UCOEs are defined by CpG islands at
    bidirectional promoters, and the inter-TSS gap may be much smaller
    than the actual CpG island(s).

    Stores original inter-TSS coordinates in orig_start / orig_end.
    """
    result = candidates.copy()
    result["orig_start"] = result["start"].copy()
    result["orig_end"] = result["end"].copy()

    new_starts = []
    new_ends = []
    n_extended = 0

    for _, row in result.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]

        # Find CpG islands on same chromosome that overlap or are within flank
        cpg_chrom = cpg_df[cpg_df["chrom"] == chrom]
        # Overlap condition: CpG island overlaps the extended search window
        search_start = start - flank
        search_end = end + flank
        overlapping = cpg_chrom[
            (cpg_chrom["cpg_start"] < search_end) &
            (cpg_chrom["cpg_end"] > search_start)
        ]

        if not overlapping.empty:
            # Extend region to encompass all overlapping CpG islands
            ext_start = min(start, overlapping["cpg_start"].min())
            ext_end = max(end, overlapping["cpg_end"].max())
            if ext_start != start or ext_end != end:
                n_extended += 1
            new_starts.append(ext_start)
            new_ends.append(ext_end)
        else:
            new_starts.append(start)
            new_ends.append(end)

    result["start"] = new_starts
    result["end"] = new_ends

    logger.info(
        "Region extension: %d / %d candidates extended to encompass CpG islands "
        "(flank window = %d bp)",
        n_extended, len(result), flank,
    )

    return result


def compute_cpg_overlap(
    candidates: pd.DataFrame,
    cpg_islands: pybedtools.BedTool,
    min_overlap: float = CPG_OVERLAP_FRACTION,
) -> pd.DataFrame:
    """Filter candidates by CpG island overlap and annotate with CpG metrics.

    After region extension, this checks that at least `min_overlap` fraction
    of the (extended) candidate region overlaps a CpG island.

    Returns filtered DataFrame with added columns:
        cpg_overlap_fraction, cpg_obs_exp, cpg_gc_pct
    """
    # Create BedTool from candidates
    cand_bed_lines = []
    for idx, row in candidates.iterrows():
        cand_bed_lines.append(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{idx}")
    cand_bt = pybedtools.BedTool("\n".join(cand_bed_lines), from_string=True)

    # Intersect: -wao gives overlap info for all, including non-overlapping
    intersected = cand_bt.intersect(cpg_islands, wao=True)

    # Parse results
    overlap_data = {}
    for interval in intersected:
        fields = interval.fields
        idx = int(fields[3])
        cand_start = int(fields[1])
        cand_end = int(fields[2])
        cand_len = cand_end - cand_start
        overlap_bp = int(fields[-1])

        if idx not in overlap_data:
            overlap_data[idx] = {
                "overlap_bp": 0,
                "cand_len": cand_len,
                "obs_exp_values": [],
                "gc_values": [],
            }

        overlap_data[idx]["overlap_bp"] += overlap_bp

        if overlap_bp > 0 and len(fields) >= 10:
            try:
                gc_pct = float(fields[8])
                obs_exp = float(fields[9])
                overlap_data[idx]["obs_exp_values"].append(obs_exp)
                overlap_data[idx]["gc_values"].append(gc_pct)
            except (ValueError, IndexError):
                pass

    # Build annotation columns
    cpg_overlap_frac = []
    cpg_obs_exp = []
    cpg_gc_pct = []

    for idx in candidates.index:
        if idx in overlap_data:
            d = overlap_data[idx]
            frac = d["overlap_bp"] / d["cand_len"] if d["cand_len"] > 0 else 0.0
            cpg_overlap_frac.append(min(frac, 1.0))
            cpg_obs_exp.append(
                sum(d["obs_exp_values"]) / len(d["obs_exp_values"])
                if d["obs_exp_values"] else 0.0
            )
            cpg_gc_pct.append(
                sum(d["gc_values"]) / len(d["gc_values"])
                if d["gc_values"] else 0.0
            )
        else:
            cpg_overlap_frac.append(0.0)
            cpg_obs_exp.append(0.0)
            cpg_gc_pct.append(0.0)

    result = candidates.copy()
    result["cpg_overlap_fraction"] = cpg_overlap_frac
    result["cpg_obs_exp"] = cpg_obs_exp
    result["cpg_gc_pct"] = cpg_gc_pct

    # Filter
    passed = result[result["cpg_overlap_fraction"] >= min_overlap].copy()
    logger.info(
        "Filter 2: %d candidates → %d with CpG island overlap ≥ %.0f%%",
        len(candidates), len(passed), min_overlap * 100,
    )
    return passed


def run_filter2(
    candidates: pd.DataFrame,
    cpg_path: Path = CPG_ISLANDS_FILE,
    min_overlap: float = CPG_OVERLAP_FRACTION,
) -> pd.DataFrame:
    """Execute Filter 2 end-to-end.

    1. Load CpG islands.
    2. Extend candidate regions to encompass overlapping/flanking CpG islands.
    3. Filter by CpG overlap fraction on the extended regions.
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 2: CpG Island Overlap & Region Extension")
    logger.info("=" * 60)

    # Load CpG islands in both formats
    cpg_islands_bt = load_cpg_islands(cpg_path)
    cpg_islands_df = load_cpg_islands_df(cpg_path)

    # Step 1: Extend regions to encompass CpG islands
    extended = extend_regions_to_cpg_islands(candidates, cpg_islands_df)

    # Step 2: Filter by CpG overlap on extended regions
    result = compute_cpg_overlap(extended, cpg_islands_bt, min_overlap)
    return result
