#!/usr/bin/env python3
"""
Run the dinucleotide periodicity and spectrum analysis.

This script re-runs the structural analysis with the new periodicity
and dinucleotide spectrum modules integrated, then prints a summary
of the key findings.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from ucoe_pipeline.structural.analysis import run_structural_analysis, PERIODICITY_METRICS, SPECTRUM_METRICS

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("DINUCLEOTIDE PERIODICITY & SPECTRUM ANALYSIS")
    logger.info("=" * 70)

    results = run_structural_analysis()

    candidates_df = results["candidates_df"]
    known_df = results["known_df"]
    controls_df = results["controls_df"]
    comparison_df = results["comparison_df"]

    # ── Print periodicity results ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PERIODICITY ANALYSIS RESULTS")
    print("=" * 70)

    period_comparison = comparison_df[comparison_df["metric"].isin(PERIODICITY_METRICS + SPECTRUM_METRICS)]
    for _, row in period_comparison.iterrows():
        sig = "***" if row.get("q_value_bh", row["p_value"]) < 0.001 else \
              "**" if row.get("q_value_bh", row["p_value"]) < 0.01 else \
              "*" if row.get("q_value_bh", row["p_value"]) < 0.05 else "ns"
        print(f"  {row['metric']:30s}  cand={row['candidates_median']:.4f}  "
              f"known={row['known_ucoe_median']:.4f}  ctrl={row['controls_median']:.4f}  "
              f"q={row.get('q_value_bh', row['p_value']):.2e} {sig}  "
              f"r={row['effect_size_r']:.3f}")

    # ── Known UCOEs detail ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("KNOWN UCOEs — Periodicity Detail")
    print("-" * 70)
    period_cols = [c for c in known_df.columns if c in PERIODICITY_METRICS + SPECTRUM_METRICS + ["header"]]
    for _, row in known_df.iterrows():
        print(f"\n  {row['header']}:")
        for col in PERIODICITY_METRICS + SPECTRUM_METRICS:
            if col in row:
                print(f"    {col:30s} = {row[col]:.4f}")

    # ── Top 10 candidates by WW periodicity SNR ─────────────────────────
    print("\n" + "-" * 70)
    print("TOP 10 CANDIDATES BY WW PERIODICITY SNR (lowest = least nucleosomal)")
    print("-" * 70)
    if "ww_periodicity_snr" in candidates_df.columns:
        sorted_df = candidates_df.nsmallest(10, "ww_periodicity_snr")
        for _, row in sorted_df.iterrows():
            header = row.get("header", "?")
            short = header[:60] if len(header) > 60 else header
            print(f"  SNR={row['ww_periodicity_snr']:.3f}  phase={row.get('ww_ss_phase_diff', 0):.2f}  "
                  f"ww_frac={row.get('ww_fraction', 0):.3f}  {short}")

    # ── Save full results ────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent.parent / "output" / "structural"
    comparison_df.to_csv(out_dir / "statistical_comparison.tsv", sep="\t", index=False)
    print(f"\nFull comparison saved to {out_dir / 'statistical_comparison.tsv'}")
    print(f"All group data saved to {out_dir / 'all_groups_structural.tsv'}")

    return results


if __name__ == "__main__":
    main()
