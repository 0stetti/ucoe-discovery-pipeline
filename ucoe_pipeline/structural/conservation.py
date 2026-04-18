"""
Evolutionary conservation analysis for UCOE candidate sequences.

Computes per-region conservation metrics using PhyloP and PhastCons scores
from the 100-vertebrate whole-genome alignment (UCSC hg38).

PhyloP: per-base measure of evolutionary conservation.
    Positive = conserved (slower evolution than neutral).
    Negative = accelerated (faster evolution than neutral).
    Scale: log-odds under phylogenetic model (Pollard et al., 2010).

PhastCons: per-base probability of belonging to a conserved element,
    estimated by a phylo-HMM (Siepel et al., 2005).
    Range: 0 (not conserved) to 1 (strongly conserved).

Literature:
    Pollard et al. (2010) Genome Res 20:110 — phyloP method
    Siepel et al. (2005) Genome Res 15:1034 — phastCons method
    Davydov et al. (2010) PLoS Comput Biol 6:e1001025 — conservation at CpG islands
    Cooper et al. (2005) Genome Res 15:901 — non-coding conservation

Design decisions:
    - Remote BigWig access: queries UCSC servers for specific regions,
      avoiding ~13 GB download. Feasible for ~800 regions of ~2 kb.
    - Both PhyloP and PhastCons: complementary measures. PhyloP detects
      both conservation and acceleration; PhastCons identifies conserved
      elements as contiguous blocks.
    - Positional analysis: conservation at ETS motif positions vs.
      non-motif positions tests whether the identified functional motifs
      are under selective constraint.
"""

import logging
import re
import time
from collections import defaultdict

import numpy as np
import pyBigWig

logger = logging.getLogger(__name__)

# UCSC BigWig URLs for hg38 100-way alignment
PHYLOP_URL = (
    "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
    "phyloP100way/hg38.phyloP100way.bw"
)
PHASTCONS_URL = (
    "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/"
    "phastCons100way/hg38.phastCons100way.bw"
)


def open_bigwig(url: str, max_retries: int = 3) -> pyBigWig.pyBigWig:
    """Open a remote BigWig file with retry logic."""
    for attempt in range(max_retries):
        try:
            bw = pyBigWig.open(url)
            if bw is not None:
                return bw
        except Exception as e:
            logger.warning("Attempt %d to open %s failed: %s", attempt + 1, url, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to open BigWig after {max_retries} attempts: {url}")


def get_scores(bw, chrom: str, start: int, end: int) -> np.ndarray:
    """Get per-base scores from a BigWig file for a region.

    Returns numpy array with NaN for missing values.
    """
    try:
        values = bw.values(chrom, start, end)
        arr = np.array(values, dtype=np.float64)
        # pyBigWig returns None for missing; numpy converts to nan
        return arr
    except Exception as e:
        logger.warning("Failed to get scores for %s:%d-%d: %s", chrom, start, end, e)
        return np.full(end - start, np.nan)


def conservation_metrics(
    chrom: str,
    start: int,
    end: int,
    phylop_bw,
    phastcons_bw,
) -> dict:
    """Compute conservation metrics for a single genomic region.

    Returns dict with:
        phylop_mean, phylop_median: average conservation pressure
        phylop_positive_frac: fraction under purifying selection (phyloP > 0)
        phylop_gt1_frac: fraction strongly conserved (phyloP > 1)
        phylop_gt2_frac: fraction very strongly conserved (phyloP > 2)
        phastcons_mean: mean probability of conserved element
        phastcons_gt05_frac: fraction in conserved elements (phastCons > 0.5)
        phastcons_gt09_frac: fraction highly conserved (phastCons > 0.9)
    """
    phylop = get_scores(phylop_bw, chrom, start, end)
    phastcons = get_scores(phastcons_bw, chrom, start, end)

    phylop_valid = phylop[~np.isnan(phylop)]
    phastcons_valid = phastcons[~np.isnan(phastcons)]

    if len(phylop_valid) == 0 or len(phastcons_valid) == 0:
        return {k: np.nan for k in [
            "phylop_mean", "phylop_median",
            "phylop_positive_frac", "phylop_gt1_frac", "phylop_gt2_frac",
            "phastcons_mean", "phastcons_gt05_frac", "phastcons_gt09_frac",
        ]}

    return {
        "phylop_mean": float(np.mean(phylop_valid)),
        "phylop_median": float(np.median(phylop_valid)),
        "phylop_positive_frac": float(np.mean(phylop_valid > 0)),
        "phylop_gt1_frac": float(np.mean(phylop_valid > 1)),
        "phylop_gt2_frac": float(np.mean(phylop_valid > 2)),
        "phastcons_mean": float(np.mean(phastcons_valid)),
        "phastcons_gt05_frac": float(np.mean(phastcons_valid > 0.5)),
        "phastcons_gt09_frac": float(np.mean(phastcons_valid > 0.9)),
    }


def conservation_at_motifs(
    chrom: str,
    start: int,
    end: int,
    sequence: str,
    phylop_bw,
    motif_pattern: str = "CGGAA[GA]",
) -> dict:
    """Compare conservation at motif positions vs. non-motif positions.

    Parameters
    ----------
    chrom, start, end : genomic coordinates
    sequence : DNA sequence for the region
    phylop_bw : open BigWig file handle
    motif_pattern : regex pattern for the motif of interest

    Returns
    -------
    dict with phylop_at_motif, phylop_outside_motif, motif_count
    """
    phylop = get_scores(phylop_bw, chrom, start, end)

    seq = sequence.upper()
    motif_mask = np.zeros(len(seq), dtype=bool)

    # Find motif on forward strand
    for m in re.finditer(motif_pattern, seq):
        motif_mask[m.start():m.end()] = True

    # Find motif on reverse strand (complement the pattern)
    rc_map = str.maketrans("ACGTRYMKBDHV[]", "TGCAYRKMVHDB][")
    rc_pattern = motif_pattern.translate(rc_map)[::-1]
    try:
        for m in re.finditer(rc_pattern, seq):
            motif_mask[m.start():m.end()] = True
    except re.error:
        pass  # Skip if RC pattern is invalid regex

    # Trim mask and scores to same length
    min_len = min(len(phylop), len(motif_mask))
    phylop = phylop[:min_len]
    motif_mask = motif_mask[:min_len]

    valid = ~np.isnan(phylop)
    motif_positions = valid & motif_mask
    nonmotif_positions = valid & ~motif_mask

    motif_count = 0
    for m in re.finditer(motif_pattern, seq):
        motif_count += 1

    result = {
        "motif_count": motif_count,
        "motif_bases": int(motif_positions.sum()),
    }

    if motif_positions.sum() > 0:
        result["phylop_at_motif"] = float(np.mean(phylop[motif_positions]))
    else:
        result["phylop_at_motif"] = np.nan

    if nonmotif_positions.sum() > 0:
        result["phylop_outside_motif"] = float(np.mean(phylop[nonmotif_positions]))
    else:
        result["phylop_outside_motif"] = np.nan

    return result


def positional_conservation_profile(
    regions: list[tuple[str, int, int]],
    phylop_bw,
    n_bins: int = 50,
) -> np.ndarray:
    """Compute average positional conservation profile across regions.

    Each region is divided into n_bins equal-sized bins, and the mean
    PhyloP score per bin is averaged across all regions.

    Returns array of shape (n_bins,) with mean conservation per position.
    """
    all_profiles = []
    for chrom, start, end in regions:
        scores = get_scores(phylop_bw, chrom, start, end)
        valid = scores[~np.isnan(scores)]
        if len(valid) < n_bins:
            continue
        # Bin the scores
        binned = np.array_split(valid, n_bins)
        profile = np.array([np.mean(b) for b in binned])
        all_profiles.append(profile)

    if not all_profiles:
        return np.full(n_bins, np.nan)

    return np.mean(all_profiles, axis=0)


def parse_coordinates(header: str) -> tuple[str, int, int] | None:
    """Parse chrom:start-end from FASTA header or BED-like string."""
    # Try pattern like ::chr9:32550561-32552586
    match = re.search(r"(chr[\dXY]+):(\d+)-(\d+)", header)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None


def run_conservation_analysis(
    cand_seqs: dict[str, str],
    known_seqs: dict[str, str],
    ctrl_seqs: dict[str, str],
    known_coords: dict[str, dict],
) -> dict:
    """Run complete conservation analysis on all three groups.

    Parameters
    ----------
    cand_seqs, known_seqs, ctrl_seqs : sequence dicts (header -> sequence)
    known_coords : dict from config.KNOWN_UCOES with chrom/start/end

    Returns
    -------
    dict with 'candidates', 'known', 'controls' keys,
    each containing list of per-region metrics dicts
    """
    logger.info("Opening remote BigWig files (PhyloP and PhastCons 100-way)...")
    phylop_bw = open_bigwig(PHYLOP_URL)
    phastcons_bw = open_bigwig(PHASTCONS_URL)

    results = {"candidates": [], "known": [], "controls": []}

    # --- Candidates ---
    logger.info("Computing conservation for %d candidates...", len(cand_seqs))
    for i, (header, seq) in enumerate(cand_seqs.items()):
        coords = parse_coordinates(header)
        if coords is None:
            logger.warning("Could not parse coordinates from: %s", header)
            continue
        chrom, start, end = coords

        metrics = conservation_metrics(chrom, start, end, phylop_bw, phastcons_bw)
        motif_metrics = conservation_at_motifs(
            chrom, start, end, seq, phylop_bw, motif_pattern="CGGAA[GA]"
        )
        metrics.update(motif_metrics)
        metrics["name"] = header.split("::")[0] if "::" in header else header
        metrics["chrom"] = chrom
        metrics["start"] = start
        metrics["end"] = end
        results["candidates"].append(metrics)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d candidates", i + 1, len(cand_seqs))

    # --- Known UCOEs ---
    logger.info("Computing conservation for %d known UCOEs...", len(known_coords))
    for name, info in known_coords.items():
        chrom, start, end = info["chrom"], info["start"], info["end"]
        seq = known_seqs.get(name, "")

        metrics = conservation_metrics(chrom, start, end, phylop_bw, phastcons_bw)
        if seq:
            motif_metrics = conservation_at_motifs(
                chrom, start, end, seq, phylop_bw, motif_pattern="CGGAA[GA]"
            )
            metrics.update(motif_metrics)
        metrics["name"] = name
        metrics["chrom"] = chrom
        metrics["start"] = start
        metrics["end"] = end
        results["known"].append(metrics)

    # --- Controls ---
    logger.info("Computing conservation for %d controls...", len(ctrl_seqs))
    for i, (header, seq) in enumerate(ctrl_seqs.items()):
        coords = parse_coordinates(header)
        if coords is None:
            continue
        chrom, start, end = coords

        metrics = conservation_metrics(chrom, start, end, phylop_bw, phastcons_bw)
        motif_metrics = conservation_at_motifs(
            chrom, start, end, seq, phylop_bw, motif_pattern="CGGAA[GA]"
        )
        metrics.update(motif_metrics)
        metrics["name"] = header
        metrics["chrom"] = chrom
        metrics["start"] = start
        metrics["end"] = end
        results["controls"].append(metrics)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d controls", i + 1, len(ctrl_seqs))

    phylop_bw.close()
    phastcons_bw.close()

    return results
