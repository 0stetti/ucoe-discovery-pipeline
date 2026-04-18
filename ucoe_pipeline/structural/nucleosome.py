"""
Nucleosome occupancy prediction from DNA sequence.

Implements a simplified dinucleotide-based model inspired by:
- Kaplan et al. (2009) Nature 458:362 — genome-wide nucleosome map
- Tillo & Hughes (2009) BMC Bioinformatics 10:442 — GC-based model
- Segal & Widom (2009) Nature Reviews Genetics 10:443 — sequence determinants

The model uses dinucleotide log-odds scores for nucleosomal vs. linker DNA
to predict per-position nucleosome formation potential, then summarizes
occupancy across the full sequence.

Key biological insight: poly(dA:dT) tracts strongly exclude nucleosomes,
while GC-rich DNA has complex behavior (high in-vitro affinity but
depleted in-vivo at CpG islands due to active remodeling).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dinucleotide nucleosome formation preference scores
# Log-odds ratio: log2(P(dinuc|nucleosome) / P(dinuc|linker))
# Derived from Kaplan et al. 2009 / Tillo & Hughes 2009 consensus.
# Positive = favors nucleosome; negative = disfavors nucleosome.
# ---------------------------------------------------------------------------
NUCLEOSOME_DINUC_PREFERENCE = {
    "AA": -0.20, "AC":  0.05, "AG":  0.08, "AT": -0.15,
    "CA":  0.10, "CC":  0.18, "CG":  0.22, "CT":  0.08,
    "GA":  0.12, "GC":  0.25, "GG":  0.18, "GT":  0.05,
    "TA": -0.30, "TC":  0.12, "TG":  0.10, "TT": -0.20,
}

NUCLEOSOME_LENGTH = 147  # bp wrapped around histone octamer


def compute_nucleosome_score_profile(seq: str) -> np.ndarray:
    """Compute per-position nucleosome formation score.

    For each position, averages the dinucleotide preference in a local
    window of NUCLEOSOME_LENGTH bp centered on that position.

    Returns array of length len(seq), with NaN at edges.
    """
    seq = seq.upper()
    n = len(seq)
    if n < NUCLEOSOME_LENGTH:
        # Sequence shorter than a nucleosome — compute global average
        scores = []
        for i in range(n - 1):
            di = seq[i:i + 2]
            if di in NUCLEOSOME_DINUC_PREFERENCE:
                scores.append(NUCLEOSOME_DINUC_PREFERENCE[di])
        return np.full(n, np.mean(scores) if scores else 0.0)

    # Step 1: compute raw dinucleotide scores
    raw = np.zeros(n - 1)
    for i in range(n - 1):
        di = seq[i:i + 2]
        raw[i] = NUCLEOSOME_DINUC_PREFERENCE.get(di, 0.0)

    # Step 2: sliding window average (147bp)
    half = NUCLEOSOME_LENGTH // 2
    profile = np.full(n, np.nan)
    cumsum = np.concatenate([[0], np.cumsum(raw)])

    for i in range(half, n - half):
        left = max(0, i - half)
        right = min(n - 1, i + half)
        window_sum = cumsum[right] - cumsum[left]
        window_len = right - left
        if window_len > 0:
            profile[i] = window_sum / window_len

    # Fill edges with nearest valid value
    first_valid = half
    last_valid = n - half - 1
    if first_valid < n:
        profile[:first_valid] = profile[first_valid]
    if last_valid >= 0:
        profile[last_valid + 1:] = profile[last_valid]

    return profile


def count_poly_at_tracts(seq: str, min_length: int = 5) -> list[tuple[int, int, str]]:
    """Find poly(dA) and poly(dT) tracts of at least min_length bp.

    These tracts are strong nucleosome exclusion signals.

    Returns list of (start, end, base) tuples.
    """
    seq = seq.upper()
    tracts = []
    i = 0
    while i < len(seq):
        if seq[i] in ("A", "T"):
            base = seq[i]
            j = i
            while j < len(seq) and seq[j] == base:
                j += 1
            if j - i >= min_length:
                tracts.append((i, j, base))
            i = j
        else:
            i += 1
    return tracts


def nucleosome_metrics(seq: str) -> dict:
    """Compute nucleosome-related summary metrics for a sequence.

    Returns dict with:
        nuc_score_mean: mean nucleosome formation score (higher = more nucleosome-prone)
        nuc_score_std: variability of nucleosome score along sequence
        nuc_depleted_fraction: fraction of positions with score < 0 (nucleosome-depleted)
        nuc_enriched_fraction: fraction of positions with score > 0.1 (strongly nucleosome-prone)
        poly_at_tracts: number of poly(dA:dT) tracts ≥ 5bp
        poly_at_coverage: fraction of sequence covered by poly(dA:dT) tracts
        nfr_score: nucleosome-free region score = nuc_depleted_fraction + poly_at_coverage
    """
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n < 10:
        return {}

    profile = compute_nucleosome_score_profile(seq)
    valid = profile[~np.isnan(profile)]

    tracts = count_poly_at_tracts(seq)
    tract_bases = sum(end - start for start, end, _ in tracts)

    depleted = float(np.sum(valid < 0) / len(valid)) if len(valid) > 0 else 0.0
    enriched = float(np.sum(valid > 0.1) / len(valid)) if len(valid) > 0 else 0.0

    return {
        "nuc_score_mean": float(np.mean(valid)) if len(valid) > 0 else 0.0,
        "nuc_score_std": float(np.std(valid)) if len(valid) > 0 else 0.0,
        "nuc_depleted_fraction": depleted,
        "nuc_enriched_fraction": enriched,
        "poly_at_tracts": len(tracts),
        "poly_at_coverage": tract_bases / n if n > 0 else 0.0,
        "nfr_score": depleted * 0.7 + (tract_bases / n) * 0.3 if n > 0 else 0.0,
    }
