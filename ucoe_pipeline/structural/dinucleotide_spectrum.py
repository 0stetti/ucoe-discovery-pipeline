"""
Dinucleotide frequency spectrum analysis.

Each DNA sequence is represented as a 16-dimensional vector of normalized
dinucleotide frequencies — a compositional "fingerprint" that captures
higher-order sequence structure beyond mononucleotide content (GC%).

This approach follows Baldi & Baisnée (2000) J Comput Biol 7:1, who showed
that dinucleotide composition captures biologically meaningful differences
between genomic regions that cannot be explained by base composition alone.

The spectrum enables:
    1. PCA/t-SNE visualization of compositional space
    2. Identification of which specific dinucleotides distinguish UCOEs
    3. Shannon entropy as a measure of compositional diversity
    4. Markov deviation: how much the observed dinucleotide frequencies
       deviate from expectations under a first-order Markov model
       (i.e., dinuc_freq vs. product of mononuc frequencies)

Literature:
    Baldi & Baisnée (2000) J Comput Biol 7:1 — dinucleotide signatures
    Karlin & Burge (1995) Trends Genet 11:283 — genomic signature (rho*)
    Nussinov (1984) Nucleic Acids Res 12:4125 — dinucleotide frequencies
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# All 16 dinucleotides in canonical order
ALL_DINUCLEOTIDES = [
    "AA", "AC", "AG", "AT",
    "CA", "CC", "CG", "CT",
    "GA", "GC", "GG", "GT",
    "TA", "TC", "TG", "TT",
]


def dinucleotide_frequencies(seq: str) -> dict[str, float]:
    """Compute normalized dinucleotide frequencies for a DNA sequence.

    Frequency of XY = count(XY) / (N-1), where N is the sequence length.
    Frequencies sum to 1.0 (within floating-point precision).

    Parameters
    ----------
    seq : uppercase DNA string (A/C/G/T)

    Returns
    -------
    freqs : dict mapping each of 16 dinucleotides to its frequency
    """
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n < 2:
        return {d: 0.0 for d in ALL_DINUCLEOTIDES}

    counts = {d: 0 for d in ALL_DINUCLEOTIDES}
    total = 0
    for i in range(n - 1):
        dinuc = seq[i:i + 2]
        if dinuc in counts:
            counts[dinuc] += 1
            total += 1

    if total == 0:
        return {d: 0.0 for d in ALL_DINUCLEOTIDES}

    return {d: counts[d] / total for d in ALL_DINUCLEOTIDES}


def mononucleotide_frequencies(seq: str) -> dict[str, float]:
    """Compute normalized mononucleotide frequencies."""
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n == 0:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for base in seq:
        if base in counts:
            counts[base] += 1

    return {b: counts[b] / n for b in "ACGT"}


def rho_star(seq: str) -> dict[str, float]:
    """Compute the genomic signature rho* (Karlin & Burge 1995).

    rho*(XY) = f(XY) / (f(X) * f(Y))

    where f(XY) is the dinucleotide frequency and f(X), f(Y) are
    mononucleotide frequencies.

    Values:
        rho* = 1.0 → dinucleotide occurs at expected rate
        rho* > 1.0 → overrepresented (e.g., CpG in CpG islands)
        rho* < 1.0 → underrepresented (e.g., CpG genome-wide)

    This metric captures deviations from a random model — the
    "Markov signature" of the sequence. Biologically, rho*(CG)
    reflects methylation-driven CpG suppression history.
    """
    dinuc = dinucleotide_frequencies(seq)
    mono = mononucleotide_frequencies(seq)

    rho = {}
    for d in ALL_DINUCLEOTIDES:
        x, y = d[0], d[1]
        expected = mono[x] * mono[y]
        if expected > 1e-10:
            rho[d] = dinuc[d] / expected
        else:
            rho[d] = np.nan

    return rho


def shannon_entropy(freqs: dict[str, float]) -> float:
    """Shannon entropy of dinucleotide frequency distribution.

    H = -Σ p(i) * log2(p(i))

    Maximum entropy (uniform distribution over 16 dinucs) = log2(16) = 4.0
    Lower entropy → more biased composition → fewer dinucleotide types dominate.
    """
    h = 0.0
    for f in freqs.values():
        if f > 1e-15:
            h -= f * np.log2(f)
    return h


def markov_deviation(seq: str) -> float:
    """Total deviation of observed dinucleotide frequencies from Markov expectation.

    D = Σ |f_obs(XY) - f_X * f_Y| for all 16 dinucleotides

    Higher D → sequence has stronger nearest-neighbor dependencies
    that cannot be explained by base composition alone.

    This captures "hidden" sequence grammar. Baldi & Baisnée (2000)
    showed this metric distinguishes functional genomic elements.
    """
    dinuc = dinucleotide_frequencies(seq)
    mono = mononucleotide_frequencies(seq)

    deviation = 0.0
    for d in ALL_DINUCLEOTIDES:
        x, y = d[0], d[1]
        expected = mono[x] * mono[y]
        deviation += abs(dinuc[d] - expected)

    return deviation


def spectrum_metrics(seq: str) -> dict:
    """Compute all dinucleotide spectrum metrics for a DNA sequence.

    Returns
    -------
    dict with keys:
        dinuc_{XY} : float (16 entries)
            Normalized frequency of each dinucleotide.
        rho_{XY} : float (16 entries)
            Karlin-Burge rho* for each dinucleotide.
        dinuc_entropy : float
            Shannon entropy of dinucleotide distribution (max 4.0).
        markov_deviation : float
            Total deviation from Markov expectation.
        rho_cg : float
            CpG rho* (key indicator of methylation history).
    """
    seq = seq.upper().replace("N", "")
    if len(seq) < 10:
        result = {}
        for d in ALL_DINUCLEOTIDES:
            result[f"dinuc_{d}"] = np.nan
            result[f"rho_{d}"] = np.nan
        result["dinuc_entropy"] = np.nan
        result["markov_deviation"] = np.nan
        result["rho_cg"] = np.nan
        return result

    freqs = dinucleotide_frequencies(seq)
    rho = rho_star(seq)

    result = {}
    for d in ALL_DINUCLEOTIDES:
        result[f"dinuc_{d}"] = freqs[d]
        result[f"rho_{d}"] = rho[d]

    result["dinuc_entropy"] = shannon_entropy(freqs)
    result["markov_deviation"] = markov_deviation(seq)
    result["rho_cg"] = rho.get("CG", np.nan)

    return result


def get_frequency_vector(seq: str) -> np.ndarray:
    """Return 16-dimensional frequency vector for PCA/clustering.

    Dinucleotides in canonical order (AA, AC, AG, AT, ..., TG, TT).
    Suitable for direct input to sklearn PCA, t-SNE, or UMAP.
    """
    freqs = dinucleotide_frequencies(seq)
    return np.array([freqs[d] for d in ALL_DINUCLEOTIDES])
