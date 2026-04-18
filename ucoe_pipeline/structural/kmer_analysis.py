"""
K-mer enrichment analysis for UCOE candidate sequences.

Compares k-mer frequencies between UCOE candidates and CpG island controls
to identify short sequence motifs specifically enriched in UCOE regions.
Also identifies k-mers shared among the three known UCOEs.

Literature:
    Schones et al. (2008) Cell 132:887 — k-mer signatures in regulatory regions
    Xie et al. (2009) PNAS 106:3830 — motif enrichment at promoters
    Hartl et al. (2019) Nat Rev Mol Cell Biol 20:5 — CpG island TF recognition
    Thomson et al. (2010) Nature 464:1082 — CFP1/CXXC1 CpG island binding

Approach:
    1. Count all k-mers (k=4,5,6) in each sequence group
    2. Normalize by total k-mer count per group
    3. Compute fold enrichment (candidates/controls) and statistical significance
    4. Identify k-mers shared among known UCOEs and enriched in candidates
    5. Map enriched k-mers to known TF binding consensus motifs

Design decisions:
    - Both strands: each k-mer is counted along with its reverse complement,
      then collapsed to the lexicographically smaller of the pair (canonical form).
      This avoids strand bias in enrichment calculations.
    - Background: CpG island controls (not whole genome), so enrichment is
      relative to CpG islands, controlling for GC bias.
    - Multiple testing: Benjamini-Hochberg FDR on all k-mers simultaneously.
"""

import logging
from collections import Counter
from itertools import product

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Known TF binding consensus motifs relevant to CpG islands and UCOEs
# Sources: JASPAR 2024, Hartl et al. (2019), Thomson et al. (2010)
KNOWN_TF_MOTIFS = {
    # Sp1/Sp3 family — GC-box, major CpG island regulators
    "GGGGCGGGG": "Sp1 (GC-box)",
    "GGGCGG": "Sp1 (core)",
    "CCGCCC": "Sp1 (core, RC)",
    # NRF1 — bidirectional promoter motif
    "GCGCATGCGC": "NRF1",
    "GCATGCGC": "NRF1 (core)",
    "GCGCATGC": "NRF1 (core)",
    "TGCGCATGCGC": "NRF1 (extended)",
    # YY1 — CpG island associated, Polycomb recruitment
    "CCGCCATNTG": "YY1",
    "CCGCCAT": "YY1 (core)",
    "GCCATC": "YY1 (partial)",
    # ETS family — ubiquitous TFs at promoters
    "CCGGAA": "ETS (core)",
    "TTCCGG": "ETS (core, RC)",
    "GGAA": "ETS (minimal)",
    # E-box — bHLH TFs (MYC, USF)
    "CACGTG": "E-box (CACGTG)",
    "CAGCTG": "E-box (CAGCTG)",
    # CpG-containing motifs — CFP1/CXXC1 recognition
    "CGCG": "CpG repeat",
    "CGGCGG": "CpG-rich repeat",
    "GCGCGC": "CpG tandem",
    # CTCF — insulator
    "CCCTC": "CTCF (partial)",
    "GAGGG": "CTCF (partial, RC)",
    # NFY/CCAAT-box
    "CCAAT": "NFY/CCAAT-box",
    "ATTGG": "NFY (RC)",
    # Kaiso — methyl-CpG binding, recognizes unmethylated CGCG
    "TCCTGCNA": "Kaiso",
    "CGCGCG": "Kaiso (CpG)",
}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def canonical_kmer(kmer: str) -> str:
    """Return the canonical (lexicographically smaller) form of a k-mer.

    Collapses forward and reverse complement into a single representative,
    avoiding strand bias in counting.
    """
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def count_kmers(seq: str, k: int, canonical: bool = True) -> Counter:
    """Count all k-mers in a sequence.

    Parameters
    ----------
    seq : uppercase DNA string
    k : k-mer length
    canonical : if True, collapse reverse complements

    Returns
    -------
    Counter mapping k-mer strings to counts
    """
    seq = seq.upper().replace("N", "")
    counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if all(b in "ACGT" for b in kmer):
            if canonical:
                kmer = canonical_kmer(kmer)
            counts[kmer] += 1
    return counts


def group_kmer_frequencies(
    sequences: dict[str, str],
    k: int,
    canonical: bool = True,
) -> tuple[Counter, int]:
    """Compute pooled k-mer counts across a group of sequences.

    Returns total counts and total number of k-mers counted.
    """
    total_counts = Counter()
    total_kmers = 0
    for seq in sequences.values():
        counts = count_kmers(seq, k, canonical=canonical)
        total_counts += counts
        total_kmers += sum(counts.values())
    return total_counts, total_kmers


def kmer_enrichment(
    cand_counts: Counter,
    cand_total: int,
    ctrl_counts: Counter,
    ctrl_total: int,
    min_count: int = 10,
) -> list[dict]:
    """Compute enrichment of each k-mer in candidates vs controls.

    Uses Fisher's exact test for significance (more appropriate than
    chi-square for k-mers with low counts).

    Parameters
    ----------
    cand_counts, ctrl_counts : pooled k-mer counts
    cand_total, ctrl_total : total k-mers per group
    min_count : minimum total count across both groups to test

    Returns
    -------
    list of dicts with: kmer, cand_freq, ctrl_freq, fold_enrichment,
        odds_ratio, p_value, q_value
    """
    all_kmers = set(cand_counts.keys()) | set(ctrl_counts.keys())

    results = []
    for kmer in sorted(all_kmers):
        c_cand = cand_counts.get(kmer, 0)
        c_ctrl = ctrl_counts.get(kmer, 0)

        if c_cand + c_ctrl < min_count:
            continue

        freq_cand = c_cand / cand_total if cand_total > 0 else 0
        freq_ctrl = c_ctrl / ctrl_total if ctrl_total > 0 else 0

        # Fold enrichment (with pseudocount to avoid division by zero)
        fe = (freq_cand + 1e-8) / (freq_ctrl + 1e-8)

        # Fisher's exact test (2×2 contingency table)
        table = np.array([
            [c_cand, cand_total - c_cand],
            [c_ctrl, ctrl_total - c_ctrl],
        ])
        _, p_val = stats.fisher_exact(table, alternative="two-sided")

        results.append({
            "kmer": kmer,
            "cand_count": c_cand,
            "ctrl_count": c_ctrl,
            "cand_freq": freq_cand,
            "ctrl_freq": freq_ctrl,
            "fold_enrichment": fe,
            "p_value": p_val,
        })

    # FDR correction
    if results:
        p_values = np.array([r["p_value"] for r in results])
        _, q_values, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)
        for r, q in zip(results, q_values):
            r["q_value"] = q
            r["significant"] = q < 0.05

    return results


def find_shared_ucoe_kmers(
    known_seqs: dict[str, str],
    k: int,
    min_freq_percentile: float = 75.0,
) -> set[str]:
    """Find k-mers that are enriched in ALL known UCOEs.

    A k-mer is "shared" if its frequency is above the given percentile
    in each of the known UCOE sequences.

    This identifies k-mers that are consistently present across
    the three structurally diverse known UCOEs.
    """
    per_ucoe_freqs = {}
    for name, seq in known_seqs.items():
        counts = count_kmers(seq, k, canonical=True)
        total = sum(counts.values())
        if total > 0:
            per_ucoe_freqs[name] = {km: c / total for km, c in counts.items()}
        else:
            per_ucoe_freqs[name] = {}

    # For each k-mer, check if it's above threshold in ALL UCOEs
    all_kmers = set()
    for freqs in per_ucoe_freqs.values():
        all_kmers.update(freqs.keys())

    # Compute threshold per UCOE
    thresholds = {}
    for name, freqs in per_ucoe_freqs.items():
        if freqs:
            vals = list(freqs.values())
            thresholds[name] = np.percentile(vals, min_freq_percentile)
        else:
            thresholds[name] = 0

    shared = set()
    for kmer in all_kmers:
        in_all = True
        for name, freqs in per_ucoe_freqs.items():
            if freqs.get(kmer, 0) < thresholds[name]:
                in_all = False
                break
        if in_all:
            shared.add(kmer)

    return shared


def match_tf_motifs(kmer: str, motif_db: dict[str, str] = KNOWN_TF_MOTIFS) -> list[str]:
    """Check if a k-mer matches or is contained in known TF binding motifs.

    Returns list of matching TF names.
    """
    matches = []
    kmer_up = kmer.upper()
    rc = reverse_complement(kmer_up)

    for motif_seq, tf_name in motif_db.items():
        motif_up = motif_seq.upper()
        # Check if k-mer contains or is contained in motif
        if kmer_up in motif_up or motif_up in kmer_up:
            matches.append(tf_name)
        elif rc in motif_up or motif_up in rc:
            matches.append(tf_name)
    return list(set(matches))


def run_kmer_analysis(
    cand_seqs: dict[str, str],
    known_seqs: dict[str, str],
    ctrl_seqs: dict[str, str],
    k_values: list[int] = None,
) -> dict:
    """Run complete k-mer enrichment analysis.

    Parameters
    ----------
    cand_seqs, known_seqs, ctrl_seqs : sequence dicts
    k_values : list of k values to analyze (default [4, 5, 6])

    Returns
    -------
    dict with results per k value
    """
    if k_values is None:
        k_values = [4, 5, 6]

    all_results = {}

    for k in k_values:
        logger.info("Analyzing %d-mers...", k)

        # Pool counts
        cand_counts, cand_total = group_kmer_frequencies(cand_seqs, k)
        ctrl_counts, ctrl_total = group_kmer_frequencies(ctrl_seqs, k)

        # Enrichment analysis
        enrichment = kmer_enrichment(cand_counts, cand_total, ctrl_counts, ctrl_total)

        # Shared UCOEs k-mers
        shared = find_shared_ucoe_kmers(known_seqs, k)

        # Annotate with TF matches and shared status
        for r in enrichment:
            r["shared_in_known_ucoes"] = r["kmer"] in shared
            r["tf_matches"] = match_tf_motifs(r["kmer"])
            r["tf_annotation"] = "; ".join(r["tf_matches"]) if r["tf_matches"] else ""

        # Sort by fold enrichment
        enrichment.sort(key=lambda x: x["fold_enrichment"], reverse=True)

        # Summary stats
        sig_enriched = [r for r in enrichment if r.get("significant") and r["fold_enrichment"] > 1]
        sig_depleted = [r for r in enrichment if r.get("significant") and r["fold_enrichment"] < 1]
        sig_and_shared = [r for r in sig_enriched if r["shared_in_known_ucoes"]]

        logger.info(
            "  k=%d: %d total k-mers tested, %d significantly enriched, "
            "%d significantly depleted, %d enriched AND shared in known UCOEs",
            k, len(enrichment), len(sig_enriched), len(sig_depleted), len(sig_and_shared),
        )

        all_results[k] = {
            "enrichment": enrichment,
            "shared_ucoe_kmers": shared,
            "n_tested": len(enrichment),
            "n_sig_enriched": len(sig_enriched),
            "n_sig_depleted": len(sig_depleted),
            "n_sig_shared": len(sig_and_shared),
        }

    return all_results
