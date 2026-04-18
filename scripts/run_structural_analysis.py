#!/usr/bin/env python3
"""
Run the structural analysis on UCOE candidate sequences.

This is an independent post-pipeline analysis that compares DNA structural
properties (flexibility, stiffness, nucleosome occupancy) across three groups:
  1. UCOE candidates (from Phase II output)
  2. Known UCOEs (A2UCOE, TBP/PSMB1, SRF-UCOE)
  3. Random CpG island controls

Usage:
    python run_structural_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
from ucoe_pipeline.config import setup_logging, OUTPUT_DIR
from ucoe_pipeline.structural.analysis import run_structural_analysis
from ucoe_pipeline.visualization.structural_plots import generate_structural_plots

logger = setup_logging()


def main():
    logger.info("Starting structural analysis...")

    results = run_structural_analysis()

    logger.info("Generating structural plots...")
    generate_structural_plots(
        combined_df=results["combined_df"],
        comparison_df=results["comparison_df"],
        output_dir=OUTPUT_DIR / "figures" / "structural",
    )

    # Print key findings
    comp = results["comparison_df"]
    print("\n" + "=" * 70)
    print("KEY FINDINGS — DNA Structural Analysis")
    print("=" * 70)

    for _, row in comp.iterrows():
        p = row["p_value"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        direction = ""
        if row["candidates_median"] > row["controls_median"]:
            direction = "HIGHER in candidates"
        elif row["candidates_median"] < row["controls_median"]:
            direction = "LOWER in candidates"

        print(f"  {row['metric']:30s}  {sig:4s}  {direction}")

    print(f"\nResults saved to: {OUTPUT_DIR / 'structural'}")
    print(f"Plots saved to:   {OUTPUT_DIR / 'figures' / 'structural'}")


if __name__ == "__main__":
    main()
