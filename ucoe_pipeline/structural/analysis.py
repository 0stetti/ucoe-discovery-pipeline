"""
Structural analysis — orchestration module.

Loads FASTA sequences for UCOE candidates, known UCOEs, and random CpG island
controls, computes structural metrics, performs statistical comparisons, and
generates output tables and plots.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ucoe_pipeline.config import (
    KNOWN_UCOES, GENOME_FASTA, CPG_ISLANDS_FILE, OUTPUT_DIR,
)
from ucoe_pipeline.structural.flexibility import sequence_metrics
from ucoe_pipeline.structural.nucleosome import nucleosome_metrics
from ucoe_pipeline.structural.periodicity import periodicity_metrics
from ucoe_pipeline.structural.dinucleotide_spectrum import spectrum_metrics
from ucoe_pipeline.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


# ── FASTA I/O ────────────────────────────────────────────────────────────────

def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into {header: sequence} dict."""
    sequences = {}
    current_header = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper())
    if current_header is not None:
        sequences[current_header] = "".join(current_seq)
    return sequences


def extract_known_ucoe_sequences(genome_fasta: Path = GENOME_FASTA) -> dict[str, str]:
    """Extract known UCOE sequences from the reference genome using pysam."""
    import pysam

    sequences = {}
    if not genome_fasta.exists():
        logger.warning("Genome FASTA not found: %s", genome_fasta)
        return sequences

    fasta = pysam.FastaFile(str(genome_fasta))
    for name, info in KNOWN_UCOES.items():
        try:
            seq = fasta.fetch(info["chrom"], info["start"], info["end"])
            sequences[name] = seq.upper()
            logger.info("Extracted %s: %s:%d-%d (%d bp)",
                        name, info["chrom"], info["start"], info["end"], len(seq))
        except Exception as e:
            logger.warning("Failed to extract %s: %s", name, e)
    fasta.close()
    return sequences


def sample_random_cpg_islands(
    n_samples: int = 200,
    genome_fasta: Path = GENOME_FASTA,
    cpg_file: Path = CPG_ISLANDS_FILE,
    exclude_regions: list[tuple[str, int, int]] | None = None,
    min_length: int = 500,
    max_length: int = 5000,
    seed: int = 42,
) -> dict[str, str]:
    """Sample random CpG islands from the genome as structural controls.

    Excludes any CpG islands overlapping with UCOE candidates or known UCOEs.
    """
    import gzip
    import pysam

    if not genome_fasta.exists() or not cpg_file.exists():
        logger.warning("Missing genome or CpG island file for controls")
        return {}

    # Load CpG islands
    cpg_islands = []
    opener = gzip.open if str(cpg_file).endswith(".gz") else open
    with opener(cpg_file, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Skip header
            if parts[0].startswith("#") or parts[0] == "bin":
                continue
            try:
                chrom = parts[1]
                start = int(parts[2])
                end = int(parts[3])
            except (ValueError, IndexError):
                continue
            length = end - start
            if min_length <= length <= max_length and chrom.startswith("chr") and "_" not in chrom:
                cpg_islands.append((chrom, start, end))

    # Exclude regions overlapping with candidates or known UCOEs
    if exclude_regions:
        exclude_set = set()
        for ec, es, ee in exclude_regions:
            exclude_set.add((ec, es, ee))

        filtered = []
        for chrom, start, end in cpg_islands:
            overlaps = False
            for ec, es, ee in exclude_regions:
                if chrom == ec and start < ee and end > es:
                    overlaps = True
                    break
            if not overlaps:
                filtered.append((chrom, start, end))
        cpg_islands = filtered

    # Random sample
    rng = np.random.default_rng(seed)
    if len(cpg_islands) > n_samples:
        indices = rng.choice(len(cpg_islands), size=n_samples, replace=False)
        cpg_islands = [cpg_islands[i] for i in indices]

    # Extract sequences
    sequences = {}
    fasta = pysam.FastaFile(str(genome_fasta))
    for chrom, start, end in cpg_islands:
        try:
            seq = fasta.fetch(chrom, start, end)
            header = f"control_cpg::{chrom}:{start}-{end}"
            sequences[header] = seq.upper()
        except Exception:
            pass
    fasta.close()
    logger.info("Sampled %d random CpG island controls", len(sequences))
    return sequences


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics(sequences: dict[str, str]) -> pd.DataFrame:
    """Compute structural + nucleosome metrics for all sequences.

    Returns DataFrame with one row per sequence.
    """
    records = []
    for header, seq in sequences.items():
        flex = sequence_metrics(seq)
        nuc = nucleosome_metrics(seq)
        period = periodicity_metrics(seq)
        spectrum = spectrum_metrics(seq)
        record = {"header": header, **flex, **nuc, **period, **spectrum}
        records.append(record)
    return pd.DataFrame(records)


def parse_candidate_header(header: str) -> dict:
    """Parse candidate FASTA header to extract rank, genes, score, coordinates."""
    result = {"header": header}
    try:
        # Format: rank{N:03d}_{gene1}_{gene2}_score{score:.4f}::{chrom}:{start}-{end}
        if "::" in header:
            name_part, coord_part = header.split("::", 1)
            result["coords"] = coord_part
            parts = name_part.split("_")
            if parts[0].startswith("rank"):
                result["rank"] = int(parts[0][4:])
            if len(parts) >= 3:
                result["gene1"] = parts[1]
                result["gene2"] = parts[2]
            for p in parts:
                if p.startswith("score"):
                    result["score"] = float(p[5:])
    except Exception:
        pass
    return result


# ── Statistical comparison ────────────────────────────────────────────────────

def compare_groups(
    candidates_df: pd.DataFrame,
    known_df: pd.DataFrame,
    controls_df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Run Mann-Whitney U tests comparing metric distributions between groups.

    Returns DataFrame with columns: metric, candidates_median, known_median,
    controls_median, U_cand_vs_ctrl, p_cand_vs_ctrl, effect_size.
    """
    results = []
    for metric in metrics:
        cand_vals = candidates_df[metric].dropna()
        ctrl_vals = controls_df[metric].dropna()
        known_vals = known_df[metric].dropna()

        # Mann-Whitney: candidates vs controls
        if len(cand_vals) > 1 and len(ctrl_vals) > 1:
            u_stat, p_val = stats.mannwhitneyu(cand_vals, ctrl_vals, alternative="two-sided")
            # Effect size: rank-biserial correlation
            n1, n2 = len(cand_vals), len(ctrl_vals)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
        else:
            u_stat, p_val, effect_size = np.nan, np.nan, np.nan

        results.append({
            "metric": metric,
            "candidates_median": float(cand_vals.median()) if len(cand_vals) > 0 else np.nan,
            "candidates_mean": float(cand_vals.mean()) if len(cand_vals) > 0 else np.nan,
            "known_ucoe_median": float(known_vals.median()) if len(known_vals) > 0 else np.nan,
            "known_ucoe_mean": float(known_vals.mean()) if len(known_vals) > 0 else np.nan,
            "controls_median": float(ctrl_vals.median()) if len(ctrl_vals) > 0 else np.nan,
            "controls_mean": float(ctrl_vals.mean()) if len(ctrl_vals) > 0 else np.nan,
            "U_statistic": u_stat,
            "p_value": p_val,
            "effect_size_r": effect_size,
        })

    df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction for multiple testing
    p_values = df["p_value"].values
    valid_mask = ~np.isnan(p_values)
    if valid_mask.sum() > 1:
        _, q_values_valid, _, _ = multipletests(
            p_values[valid_mask], method="fdr_bh", alpha=0.05
        )
        q_values = np.full_like(p_values, np.nan)
        q_values[valid_mask] = q_values_valid
        df["q_value_bh"] = q_values
        df["significant_fdr05"] = df["q_value_bh"] < 0.05
        logger.info("Applied Benjamini-Hochberg FDR correction to %d p-values", valid_mask.sum())
    else:
        df["q_value_bh"] = df["p_value"]
        df["significant_fdr05"] = df["p_value"] < 0.05

    return df


# ── Main analysis ─────────────────────────────────────────────────────────────

STRUCTURAL_METRICS = [
    "flexibility_mean", "stiffness_mean", "bendability_mean",
    "gc_content", "cpg_obs_exp", "cpg_density", "poly_at_fraction",
    "nuc_score_mean", "nuc_depleted_fraction", "nuc_enriched_fraction",
    "poly_at_tracts", "poly_at_coverage", "nfr_score",
]

# Periodicity metrics — Segal et al. (2006), Trifonov & Sussman (1980)
PERIODICITY_METRICS = [
    "ww_periodicity_snr", "ss_periodicity_snr",
    "ww_peak_period", "ss_peak_period",
    "ww_ss_phase_diff",
    "ww_autocorr_10bp", "ss_autocorr_10bp",
    "ww_fraction", "ss_fraction",
]

# Dinucleotide spectrum metrics — Baldi & Baisnée (2000), Karlin & Burge (1995)
SPECTRUM_METRICS = [
    "dinuc_entropy", "markov_deviation", "rho_cg",
]

# All metrics for statistical comparison
ALL_ANALYSIS_METRICS = STRUCTURAL_METRICS + PERIODICITY_METRICS + SPECTRUM_METRICS


def run_structural_analysis(
    candidates_fasta: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run the full structural analysis pipeline.

    Parameters
    ----------
    candidates_fasta : path to FASTA with candidate sequences.
        Defaults to output/ucoe_sequences.fa
    output_dir : output directory. Defaults to output/structural/

    Returns
    -------
    dict with keys: candidates_df, known_df, controls_df, comparison_df
    """
    if candidates_fasta is None:
        candidates_fasta = OUTPUT_DIR / "ucoe_sequences.fa"
    if output_dir is None:
        output_dir = OUTPUT_DIR / "structural"

    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("STRUCTURAL ANALYSIS — DNA Topology & Nucleosome Prediction")
    logger.info("=" * 60)

    # 1. Load candidate sequences
    logger.info("Loading candidate sequences from %s", candidates_fasta)
    candidates_seqs = read_fasta(candidates_fasta)
    logger.info("Loaded %d candidate sequences", len(candidates_seqs))

    # 2. Extract known UCOE sequences
    logger.info("Extracting known UCOE sequences from reference genome")
    known_seqs = extract_known_ucoe_sequences()
    logger.info("Extracted %d known UCOE sequences", len(known_seqs))

    # 3. Build exclusion list (candidates + known UCOEs)
    exclude_regions = []
    for info in KNOWN_UCOES.values():
        exclude_regions.append((info["chrom"], info["start"], info["end"]))
    for header in candidates_seqs:
        if "::" in header:
            coord = header.split("::")[1]
            try:
                chrom, pos = coord.split(":")
                start, end = pos.split("-")
                exclude_regions.append((chrom, int(start), int(end)))
            except Exception:
                pass

    # 4. Sample random CpG island controls
    logger.info("Sampling random CpG island controls")
    controls_seqs = sample_random_cpg_islands(
        n_samples=200,
        exclude_regions=exclude_regions,
    )
    logger.info("Sampled %d control CpG islands", len(controls_seqs))

    # 5. Compute metrics for all three groups
    logger.info("Computing structural metrics for candidates...")
    candidates_df = compute_all_metrics(candidates_seqs)
    candidates_df["group"] = "UCOE_candidates"

    # Parse rank and gene info from headers
    parsed = candidates_df["header"].apply(parse_candidate_header)
    for col in ["rank", "gene1", "gene2", "score", "coords"]:
        candidates_df[col] = parsed.apply(lambda x: x.get(col))

    logger.info("Computing structural metrics for known UCOEs...")
    known_df = compute_all_metrics(known_seqs)
    known_df["group"] = "Known_UCOEs"

    logger.info("Computing structural metrics for controls...")
    controls_df = compute_all_metrics(controls_seqs)
    controls_df["group"] = "CpG_island_controls"

    # 6. Statistical comparison (all metrics: structural + periodicity + spectrum)
    logger.info("Running statistical comparisons...")
    comparison_df = compare_groups(candidates_df, known_df, controls_df, ALL_ANALYSIS_METRICS)

    # 7. Save results
    candidates_df.to_csv(output_dir / "candidates_structural.tsv", sep="\t", index=False)
    known_df.to_csv(output_dir / "known_ucoes_structural.tsv", sep="\t", index=False)
    controls_df.to_csv(output_dir / "controls_structural.tsv", sep="\t", index=False)
    comparison_df.to_csv(output_dir / "statistical_comparison.tsv", sep="\t", index=False)

    # 8. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("STRUCTURAL ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("Candidates: %d sequences", len(candidates_df))
    logger.info("Known UCOEs: %d sequences", len(known_df))
    logger.info("Controls: %d CpG islands", len(controls_df))
    logger.info("")

    for _, row in comparison_df.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
        logger.info(
            "  %-25s  cand=%.4f  known=%.4f  ctrl=%.4f  p=%.2e %s",
            row["metric"],
            row["candidates_median"],
            row["known_ucoe_median"],
            row["controls_median"],
            row["p_value"],
            sig,
        )

    # Combined DataFrame for plotting
    combined_df = pd.concat([candidates_df, known_df, controls_df], ignore_index=True)
    combined_df.to_csv(output_dir / "all_groups_structural.tsv", sep="\t", index=False)

    return {
        "candidates_df": candidates_df,
        "known_df": known_df,
        "controls_df": controls_df,
        "comparison_df": comparison_df,
        "combined_df": combined_df,
    }
