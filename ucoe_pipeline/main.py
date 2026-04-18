"""
UCOE Discovery Pipeline — main entry point.

Orchestrates Phase I (rule-based filtering) → Phase II (similarity ranking).

Usage:
    python -m ucoe_pipeline.main
    python -m ucoe_pipeline.main --skip-to-phase2 output/phase1_candidates.tsv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ucoe_pipeline.config import (
    OUTPUT_DIR,
    RANKING_WEIGHTS,
    setup_logging,
)
from ucoe_pipeline.phase1.filter_divergent_hkg import run_filter1, check_known_ucoes_recovered
from ucoe_pipeline.phase1.filter_cpg_islands import run_filter2
from ucoe_pipeline.phase1.filter_histone_marks import run_filter3, run_filter4
from ucoe_pipeline.phase1.filter_methylation import run_filter5
from ucoe_pipeline.phase1.filter_dnase import run_filter6
from ucoe_pipeline.phase2.feature_extraction import extract_all_features
from ucoe_pipeline.phase2.reference_profile import build_reference_profile
from ucoe_pipeline.phase2.composite_score import compute_composite_scores, sensitivity_analysis
from ucoe_pipeline.phase2.validation import sanity_check, leave_one_out_validation
from ucoe_pipeline.visualization.filter_summary import plot_filter_funnel
from ucoe_pipeline.visualization.radar_plots import plot_top_candidates_radar
from ucoe_pipeline.visualization.ranking_plots import (
    plot_metric_comparison,
    plot_score_distribution,
    plot_top_ranked_bar,
)
from ucoe_pipeline.utils.io_utils import ensure_dir, save_bed, save_candidates

logger = logging.getLogger(__name__)


def _annotate_dnase(candidates: pd.DataFrame) -> pd.DataFrame:
    """Annotate candidates with DNase signal (informational, not a hard filter).

    Extracts DNase-seq fold-change signal across all cell lines and adds
    summary columns (mean, CV, fraction above threshold) without filtering.
    """
    from ucoe_pipeline.phase1.filter_dnase import run_filter6 as _run_dnase
    import numpy as np

    logger.info("=" * 60)
    logger.info("DNase annotation (informational — not filtering)")
    logger.info("=" * 60)

    # Use a very low threshold (0.0) and ubiquity (0.0) so nothing is filtered out
    annotated = _run_dnase(candidates, threshold=0.0, ubiquity=0.0)
    return annotated


def run_phase1() -> tuple[pd.DataFrame, dict[str, int]]:
    """Execute all Phase I filters sequentially. Returns (candidates, filter_counts)."""
    filter_counts = {}

    # Filter 1: Divergent HKG pairs
    candidates = run_filter1()
    filter_counts["Filter 1: Divergent HKG pairs"] = len(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "after_filter1.tsv", "Filter 1")

    if candidates.empty:
        logger.error("No candidates after Filter 1 — aborting")
        return candidates, filter_counts

    # Filter 2: CpG island overlap
    candidates = run_filter2(candidates)
    filter_counts["Filter 2: CpG island overlap"] = len(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "after_filter2.tsv", "Filter 2")

    if candidates.empty:
        logger.error("No candidates after Filter 2 — aborting")
        return candidates, filter_counts

    # Filter 3: Active histone marks
    candidates = run_filter3(candidates)
    filter_counts["Filter 3: Active histone marks"] = len(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "after_filter3.tsv", "Filter 3")

    if candidates.empty:
        logger.error("No candidates after Filter 3 — aborting")
        return candidates, filter_counts

    # Filter 4: Repressive mark absence
    candidates = run_filter4(candidates)
    filter_counts["Filter 4: Repressive mark absence"] = len(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "after_filter4.tsv", "Filter 4")

    if candidates.empty:
        logger.error("No candidates after Filter 4 — aborting")
        return candidates, filter_counts

    # Filter 5: Hypomethylation
    candidates = run_filter5(candidates)
    filter_counts["Filter 5: Hypomethylation"] = len(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "after_filter5.tsv", "Filter 5")

    if candidates.empty:
        logger.error("No candidates after Filter 5 — aborting")
        return candidates, filter_counts

    # DNase accessibility — extracted as informational feature but NOT used
    # as a hard filter. Known UCOEs show DNase FC ≈ 0.5–1.3, below any
    # reasonable threshold, because: (1) UCOEs maintain open chromatin via
    # histone marks and hypomethylation, not via DNase hypersensitivity;
    # (2) GC-bias in input/control samples deflates fold-change at CpG islands.
    # DNase signal is still extracted for Phase II ranking.
    candidates = _annotate_dnase(candidates)
    save_candidates(candidates, OUTPUT_DIR / "phase1" / "phase1_final.tsv", "Phase I final")

    # Sanity check after all Phase I filters
    check_known_ucoes_recovered(candidates)

    return candidates, filter_counts


def run_phase2(candidates: pd.DataFrame) -> pd.DataFrame:
    """Execute Phase II: feature extraction, reference building, scoring, validation."""
    # Build reference profile from known UCOEs
    ref_df, centroid, ref_matrix = build_reference_profile()
    save_candidates(ref_df, OUTPUT_DIR / "phase2" / "reference_profile.tsv", "Reference profile")

    # Extract full features for candidates
    candidates = extract_all_features(candidates)

    # Compute composite scores
    scored = compute_composite_scores(candidates, ref_matrix, centroid, RANKING_WEIGHTS)
    save_candidates(scored, OUTPUT_DIR / "phase2" / "scored_candidates.tsv", "Scored candidates")

    # Save BED file for genome browser
    save_bed(scored, OUTPUT_DIR / "ucoe_candidates.bed", name_col="gene1")

    # Validation
    sanity_results = sanity_check(scored)
    loo_results = leave_one_out_validation(scored)
    save_candidates(loo_results, OUTPUT_DIR / "phase2" / "loo_validation.tsv", "LOO validation")

    # Sensitivity analysis
    stability = sensitivity_analysis(scored, ref_matrix, centroid)
    save_candidates(stability, OUTPUT_DIR / "phase2" / "sensitivity_analysis.tsv", "Sensitivity")

    return scored


def generate_visualizations(
    scored: pd.DataFrame,
    filter_counts: dict[str, int],
    centroid,
):
    """Generate all publication-quality figures."""
    fig_dir = OUTPUT_DIR / "figures"
    ensure_dir(fig_dir)

    # Filter funnel
    plot_filter_funnel(filter_counts, fig_dir / "filter_funnel.png")

    # Score distribution
    plot_score_distribution(scored, fig_dir / "score_distribution.png")

    # Top ranked bar chart
    plot_top_ranked_bar(scored, fig_dir / "top_ranked.png")

    # Metric comparison scatter
    if "mahalanobis_score" in scored.columns and "cosine_score" in scored.columns:
        plot_metric_comparison(scored, fig_dir / "metric_comparison.png")

    # Radar plots for top 10
    plot_top_candidates_radar(scored, centroid, fig_dir / "radar")


def generate_summary_report(
    scored: pd.DataFrame,
    filter_counts: dict[str, int],
    sanity_results: dict | None = None,
):
    """Write a text summary of the pipeline run."""
    report_path = OUTPUT_DIR / "pipeline_summary.txt"
    ensure_dir(report_path.parent)

    lines = [
        "=" * 70,
        "UCOE DISCOVERY PIPELINE — SUMMARY REPORT",
        "=" * 70,
        "",
        "PHASE I — FILTER SUMMARY",
        "-" * 40,
    ]
    for name, count in filter_counts.items():
        lines.append(f"  {name}: {count:,} candidates")

    lines += [
        "",
        "PHASE II — SCORING SUMMARY",
        "-" * 40,
        f"  Total scored candidates: {len(scored):,}",
        f"  Composite score range: {scored['composite_score'].min():.4f} – {scored['composite_score'].max():.4f}",
        f"  Median composite score: {scored['composite_score'].median():.4f}",
        "",
        "TOP 20 CANDIDATES",
        "-" * 40,
    ]
    for i, (_, row) in enumerate(scored.head(20).iterrows()):
        gene = f"{row.get('gene1', '?')}/{row.get('gene2', '?')}"
        lines.append(
            f"  #{i+1:2d}: {row['chrom']}:{row['start']}-{row['end']} "
            f"({gene}) — score={row['composite_score']:.4f}"
        )

    lines.append("")
    report = "\n".join(lines)
    report_path.write_text(report)
    logger.info("Summary report saved to %s", report_path)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="UCOE Discovery Pipeline")
    parser.add_argument(
        "--skip-to-phase2",
        type=str,
        default=None,
        help="Path to Phase I output TSV to skip directly to Phase II",
    )
    parser.add_argument(
        "--filter1-only",
        action="store_true",
        help="Run only Filter 1 (for testing)",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run all Phase I filters but skip Phase II",
    )
    args = parser.parse_args()

    logger = setup_logging()
    ensure_dir(OUTPUT_DIR)

    if args.skip_to_phase2:
        logger.info("Loading Phase I results from %s", args.skip_to_phase2)
        candidates = pd.read_csv(args.skip_to_phase2, sep="\t")
        filter_counts = {"Loaded from file": len(candidates)}
    elif args.filter1_only:
        candidates = run_filter1()
        save_candidates(candidates, OUTPUT_DIR / "phase1" / "filter1_candidates.tsv", "Filter 1")
        logger.info("Filter 1 complete. %d candidates found.", len(candidates))
        return
    else:
        candidates, filter_counts = run_phase1()

    if candidates.empty:
        logger.error("No candidates survived Phase I — pipeline cannot continue")
        sys.exit(1)

    if args.phase1_only:
        logger.info("Phase I complete. %d candidates survived all filters.", len(candidates))
        logger.info("Output in %s", OUTPUT_DIR / "phase1")
        # Print filter funnel
        print("\n" + "=" * 60)
        print("PHASE I — FILTER FUNNEL")
        print("=" * 60)
        for name, count in filter_counts.items():
            print(f"  {name}: {count:,} candidates")
        print(f"\nFinal Phase I candidates: {len(candidates):,}")
        print(f"Output directory: {OUTPUT_DIR / 'phase1'}")
        return

    # Phase II
    scored = run_phase2(candidates)

    # Build reference again for visualization
    _, centroid, _ = build_reference_profile()

    # Visualizations
    generate_visualizations(scored, filter_counts, centroid)

    # Sequence extraction
    try:
        from ucoe_pipeline.utils.sequence_extraction import extract_sequences_from_scored
        extract_sequences_from_scored(top_n=20)
        extract_sequences_from_scored()
    except Exception as e:
        logger.warning("Sequence extraction skipped: %s", e)

    # Summary
    generate_summary_report(scored, filter_counts)

    logger.info("Pipeline complete. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
