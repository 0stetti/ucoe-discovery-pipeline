"""
DNA flexibility and stiffness analysis from sequence.

Implements two complementary dinucleotide-based metrics:
1. Brukner flexibility index (DNase I sensitivity; Brukner et al. 1995, J Mol Biol 249:479)
2. Dinucleotide stiffness (persistence length; Geggier & Vologodskii 2010, PNAS 107:15421)

Higher flexibility = easier to bend = more amenable to nucleosome wrapping.
Higher stiffness = more rigid = resists nucleosome wrapping.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brukner dinucleotide flexibility parameters
# From Brukner et al. (1995) J Mol Biol 249:479-486
# Values represent relative DNase I cutting frequency (higher = more flexible)
# Complement pairs share the same value (e.g., AC = GT)
# ---------------------------------------------------------------------------
BRUKNER_FLEXIBILITY = {
    "AA": 0.026, "AC": 0.032, "AG": 0.031, "AT": 0.033,
    "CA": 0.035, "CC": 0.031, "CG": 0.028, "CT": 0.031,
    "GA": 0.032, "GC": 0.029, "GG": 0.031, "GT": 0.032,
    "TA": 0.038, "TC": 0.032, "TG": 0.035, "TT": 0.026,
}

# ---------------------------------------------------------------------------
# Dinucleotide stiffness parameters (bend + twist combined)
# Derived from Geggier & Vologodskii (2010) PNAS 107:15421-15426
# Values are relative bend stiffness (higher = stiffer = more rigid)
# Normalized so that genome average ≈ 1.0
# ---------------------------------------------------------------------------
DINUCLEOTIDE_STIFFNESS = {
    "AA": 1.20, "AC": 1.05, "AG": 0.90, "AT": 0.72,
    "CA": 0.80, "CC": 1.15, "CG": 0.85, "CT": 0.90,
    "GA": 1.08, "GC": 1.35, "GG": 1.15, "GT": 1.05,
    "TA": 0.60, "TC": 1.08, "TG": 0.80, "TT": 1.20,
}

# ---------------------------------------------------------------------------
# Trinucleotide bendability (Brukner et al. 1995; Vlahovicek et al. 2003)
# DNase I-derived bendability toward major groove (higher = more bendable)
# ---------------------------------------------------------------------------
TRINUCLEOTIDE_BENDABILITY = {
    "AAA": 0.026, "AAC": 0.030, "AAG": 0.031, "AAT": 0.028,
    "ACA": 0.036, "ACC": 0.034, "ACG": 0.031, "ACT": 0.033,
    "AGA": 0.033, "AGC": 0.032, "AGG": 0.032, "AGT": 0.033,
    "ATA": 0.038, "ATC": 0.034, "ATG": 0.037, "ATT": 0.028,
    "CAA": 0.032, "CAC": 0.036, "CAG": 0.035, "CAT": 0.034,
    "CCA": 0.034, "CCC": 0.030, "CCG": 0.030, "CCT": 0.032,
    "CGA": 0.030, "CGC": 0.028, "CGG": 0.030, "CGT": 0.030,
    "CTA": 0.035, "CTC": 0.032, "CTG": 0.035, "CTT": 0.031,
    "GAA": 0.033, "GAC": 0.035, "GAG": 0.034, "GAT": 0.033,
    "GCA": 0.032, "GCC": 0.031, "GCG": 0.028, "GCT": 0.032,
    "GGA": 0.032, "GGC": 0.031, "GGG": 0.030, "GGT": 0.034,
    "GTA": 0.034, "GTC": 0.033, "GTG": 0.036, "GTT": 0.030,
    "TAA": 0.035, "TAC": 0.037, "TAG": 0.036, "TAT": 0.038,
    "TCA": 0.035, "TCC": 0.033, "TCG": 0.030, "TCT": 0.033,
    "TGA": 0.035, "TGC": 0.032, "TGG": 0.034, "TGT": 0.036,
    "TTA": 0.035, "TTC": 0.033, "TTG": 0.032, "TTT": 0.026,
}


def compute_flexibility_profile(seq: str) -> np.ndarray:
    """Compute per-position Brukner flexibility along a DNA sequence.

    Parameters
    ----------
    seq : uppercase DNA string (A/C/G/T only; N positions → NaN)

    Returns
    -------
    profile : array of length len(seq)-1, one value per dinucleotide step.
    """
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n < 2:
        return np.array([])
    profile = np.full(n - 1, np.nan)
    for i in range(n - 1):
        di = seq[i:i + 2]
        if di in BRUKNER_FLEXIBILITY:
            profile[i] = BRUKNER_FLEXIBILITY[di]
    return profile


def compute_stiffness_profile(seq: str) -> np.ndarray:
    """Compute per-position dinucleotide stiffness along a DNA sequence."""
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n < 2:
        return np.array([])
    profile = np.full(n - 1, np.nan)
    for i in range(n - 1):
        di = seq[i:i + 2]
        if di in DINUCLEOTIDE_STIFFNESS:
            profile[i] = DINUCLEOTIDE_STIFFNESS[di]
    return profile


def compute_bendability_profile(seq: str) -> np.ndarray:
    """Compute per-position trinucleotide bendability (Brukner/Vlahovicek)."""
    seq = seq.upper().replace("N", "")
    n = len(seq)
    if n < 3:
        return np.array([])
    profile = np.full(n - 2, np.nan)
    for i in range(n - 2):
        tri = seq[i:i + 3]
        if tri in TRINUCLEOTIDE_BENDABILITY:
            profile[i] = TRINUCLEOTIDE_BENDABILITY[tri]
    return profile


def sequence_metrics(seq: str) -> dict:
    """Compute summary structural metrics for a single DNA sequence.

    Returns dict with:
        flexibility_mean, flexibility_std
        stiffness_mean, stiffness_std
        bendability_mean, bendability_std
        gc_content
        cpg_obs_exp (CpG observed/expected ratio)
        cpg_density (CpGs per kb)
        poly_at_fraction (fraction of sequence in poly-dA:dT tracts ≥5bp)
    """
    seq = seq.upper()
    # Remove Ns for computation
    clean = seq.replace("N", "")
    n = len(clean)
    if n < 10:
        return {}

    # Flexibility
    flex = compute_flexibility_profile(clean)
    stiff = compute_stiffness_profile(clean)
    bend = compute_bendability_profile(clean)

    # GC content
    gc = (clean.count("G") + clean.count("C")) / n

    # CpG observed/expected
    c_count = clean.count("C")
    g_count = clean.count("G")
    cpg_count = clean.count("CG")
    if c_count > 0 and g_count > 0:
        cpg_oe = (cpg_count * n) / (c_count * g_count)
    else:
        cpg_oe = 0.0

    cpg_density = (cpg_count / n) * 1000  # per kb

    # Poly(dA:dT) tracts ≥ 5bp — nucleosome exclusion signals
    poly_at_bases = 0
    i = 0
    while i < n:
        if clean[i] in ("A", "T"):
            j = i
            while j < n and clean[j] == clean[i]:
                j += 1
            if j - i >= 5:
                poly_at_bases += j - i
            i = j
        else:
            i += 1

    return {
        "flexibility_mean": float(np.nanmean(flex)),
        "flexibility_std": float(np.nanstd(flex)),
        "stiffness_mean": float(np.nanmean(stiff)),
        "stiffness_std": float(np.nanstd(stiff)),
        "bendability_mean": float(np.nanmean(bend)),
        "bendability_std": float(np.nanstd(bend)),
        "gc_content": gc,
        "cpg_obs_exp": cpg_oe,
        "cpg_density": cpg_density,
        "poly_at_fraction": poly_at_bases / n,
        "seq_length": n,
    }
