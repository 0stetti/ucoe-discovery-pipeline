"""
Filter 1 — Divergent housekeeping gene promoters.

Identify head-to-head (divergent) pairs of housekeeping genes with
inter-TSS distance ≤ 5 kb. The intergenic region between each pair
is a candidate UCOE region.
"""

import logging
from pathlib import Path

import pandas as pd

from ucoe_pipeline.config import (
    GENCODE_GTF,
    HK_GENES_FILE,
    MAX_INTER_TSS_DISTANCE,
    KNOWN_UCOES,
)
from ucoe_pipeline.utils.io_utils import read_gzipped_or_plain

logger = logging.getLogger(__name__)


def parse_gencode_tss(gtf_path: Path) -> pd.DataFrame:
    """Parse GENCODE GTF to extract gene-level TSSs.

    Returns DataFrame with columns: gene_name, chrom, tss, strand, gene_id.
    TSS is defined as start (+ strand) or end (- strand) of the gene record.
    """
    logger.info("Parsing GENCODE GTF: %s", gtf_path)
    records = []
    lines = read_gzipped_or_plain(gtf_path)

    for line in lines:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9:
            continue
        if fields[2] != "gene":
            continue

        chrom = fields[0]
        start = int(fields[3]) - 1  # GTF is 1-based; convert to 0-based
        end = int(fields[4])
        strand = fields[6]

        # Parse attributes
        attrs = _parse_gtf_attributes(fields[8])
        gene_name = attrs.get("gene_name", "")
        gene_id = attrs.get("gene_id", "")
        gene_type = attrs.get("gene_type", "")

        # Only protein-coding genes
        if gene_type != "protein_coding":
            continue

        # TSS definition
        tss = start if strand == "+" else end

        records.append({
            "gene_name": gene_name,
            "gene_id": gene_id,
            "chrom": chrom,
            "tss": tss,
            "strand": strand,
        })

    df = pd.DataFrame(records)
    # Keep standard chromosomes only
    standard_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    df = df[df["chrom"].isin(standard_chroms)].copy()
    # Deduplicate by gene_name (keep one TSS per gene; prefer longest transcript is
    # already handled by taking the gene-level record)
    df = df.drop_duplicates(subset="gene_name", keep="first")
    logger.info("Parsed %d protein-coding gene TSSs from GENCODE", len(df))
    return df


def load_housekeeping_genes(hk_path: Path) -> set[str]:
    """Load housekeeping gene list from Eisenberg & Levanon (2013).

    Expects a file with one gene name per line (or a tab-separated file
    where the first column is the gene name). Comment lines start with #.
    """
    logger.info("Loading housekeeping gene list: %s", hk_path)
    genes = set()
    lines = read_gzipped_or_plain(hk_path)
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Take first column (gene symbol)
        gene = line.split("\t")[0].split(",")[0].strip()
        if gene:
            genes.add(gene.upper())
    logger.info("Loaded %d housekeeping genes", len(genes))
    return genes


def find_divergent_hkg_pairs(
    tss_df: pd.DataFrame,
    hk_genes: set[str],
    max_distance: int = MAX_INTER_TSS_DISTANCE,
) -> pd.DataFrame:
    """Find bidirectional housekeeping gene pairs with close TSSs.

    A bidirectional (divergent) promoter is defined as two adjacent genes
    transcribed in opposite directions with TSSs ≤ max_distance apart.
    Both arrangements are detected:

    Pattern A — g1(−) ... g2(+):
        ←←gene1(−)  [intergenic]  gene2(+)→→
        Gene bodies extend outward; intergenic region has no gene body.

    Pattern B — g1(+) ... g2(−):
        gene1(+)→→  [shared promoter]  ←←gene2(−)
        Gene bodies overlap through the shared promoter region.
        (e.g., CBX3/HNRNPA2B1 A2UCOE, SURF2/SURF1 SRF-UCOE)

    Both patterns represent bidirectional promoters with transcription
    diverging from the shared regulatory region (Adachi & Lieber, 2002).

    Returns DataFrame with columns:
        chrom, start, end, gene1, gene2, gene1_strand, gene2_strand,
        inter_tss_distance, pair_type
    where start/end define the region between the two TSSs.
    """
    # Filter to housekeeping genes only
    hk_tss = tss_df[tss_df["gene_name"].str.upper().isin(hk_genes)].copy()
    logger.info("Found %d housekeeping genes with TSS annotations", len(hk_tss))

    # Sort by chromosome and TSS position
    hk_tss = hk_tss.sort_values(["chrom", "tss"]).reset_index(drop=True)

    pairs = []
    # For each chromosome, find adjacent gene pairs with opposite strands
    for chrom, group in hk_tss.groupby("chrom"):
        group = group.sort_values("tss").reset_index(drop=True)
        for i in range(len(group) - 1):
            g1 = group.iloc[i]
            g2 = group.iloc[i + 1]

            # Both bidirectional patterns: opposite strands, TSSs close
            if g1["strand"] != g2["strand"]:
                distance = g2["tss"] - g1["tss"]
                if 0 < distance <= max_distance:
                    # Classify the pair type
                    if g1["strand"] == "-" and g2["strand"] == "+":
                        pair_type = "divergent_classic"
                    else:  # g1(+) and g2(-)
                        pair_type = "divergent_overlapping"

                    pairs.append({
                        "chrom": chrom,
                        "start": g1["tss"],
                        "end": g2["tss"],
                        "gene1": g1["gene_name"],
                        "gene2": g2["gene_name"],
                        "gene1_strand": g1["strand"],
                        "gene2_strand": g2["strand"],
                        "inter_tss_distance": distance,
                        "pair_type": pair_type,
                    })

    result = pd.DataFrame(pairs)
    if not result.empty:
        n_classic = (result["pair_type"] == "divergent_classic").sum()
        n_overlap = (result["pair_type"] == "divergent_overlapping").sum()
        logger.info(
            "Filter 1: %d HKG TSSs → %d bidirectional pairs with distance ≤ %d bp "
            "(%d classic divergent, %d overlapping divergent)",
            len(hk_tss), len(result), max_distance, n_classic, n_overlap,
        )
    else:
        logger.info(
            "Filter 1: %d HKG TSSs → 0 bidirectional pairs with distance ≤ %d bp",
            len(hk_tss), max_distance,
        )
    return result


def check_known_ucoes_recovered(candidates: pd.DataFrame) -> dict[str, bool]:
    """Check whether the intergenic region of each known UCOE is captured.

    A known UCOE is 'recovered' if any candidate region overlaps it
    (i.e., at least one of the UCOE's constituent genes appears as
    gene1 or gene2 in the candidate list).
    """
    recovery = {}
    for name, ucoe in KNOWN_UCOES.items():
        gene_a, gene_b = ucoe["genes"]
        # Check if either gene appears in the candidate pairs
        found = candidates[
            (candidates["gene1"].str.upper() == gene_a.upper())
            | (candidates["gene2"].str.upper() == gene_a.upper())
            | (candidates["gene1"].str.upper() == gene_b.upper())
            | (candidates["gene2"].str.upper() == gene_b.upper())
        ]
        recovered = len(found) > 0
        recovery[name] = recovered
        if recovered:
            logger.info("✓ Known UCOE '%s' (%s) RECOVERED in candidates", name, ucoe["genes"])
            for _, row in found.iterrows():
                logger.info(
                    "  → %s:%d-%d (%s / %s, distance=%d bp)",
                    row["chrom"], row["start"], row["end"],
                    row["gene1"], row["gene2"], row["inter_tss_distance"],
                )
        else:
            logger.warning(
                "✗ Known UCOE '%s' (%s) NOT recovered — check gene names or distance threshold",
                name, ucoe["genes"],
            )
    return recovery


def run_filter1(
    gtf_path: Path = GENCODE_GTF,
    hk_path: Path = HK_GENES_FILE,
    max_distance: int = MAX_INTER_TSS_DISTANCE,
) -> pd.DataFrame:
    """Execute Filter 1 end-to-end.

    Returns DataFrame of candidate UCOE regions (divergent HKG pairs).
    """
    logger.info("=" * 60)
    logger.info("PHASE I — FILTER 1: Divergent Housekeeping Gene Pairs")
    logger.info("=" * 60)

    tss_df = parse_gencode_tss(gtf_path)
    hk_genes = load_housekeeping_genes(hk_path)
    candidates = find_divergent_hkg_pairs(tss_df, hk_genes, max_distance)

    # Sanity check
    recovery = check_known_ucoes_recovered(candidates)
    n_recovered = sum(recovery.values())
    logger.info(
        "Sanity check: %d / %d known UCOEs recovered after Filter 1",
        n_recovered, len(recovery),
    )

    return candidates


# ── Private helpers ──────────────────────────────────────────────────────────

def _parse_gtf_attributes(attr_string: str) -> dict[str, str]:
    """Parse GTF attribute string into a dict."""
    attrs = {}
    for entry in attr_string.strip().split(";"):
        entry = entry.strip()
        if not entry:
            continue
        # Format: key "value"
        parts = entry.split('"')
        if len(parts) >= 2:
            key = parts[0].strip()
            value = parts[1].strip()
            attrs[key] = value
    return attrs
