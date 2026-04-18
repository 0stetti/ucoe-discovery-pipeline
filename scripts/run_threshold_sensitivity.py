#!/usr/bin/env python3
"""
Sensitivity analysis for the inter-TSS distance threshold (Filter 1).

Tests multiple thresholds (1 kb, 2 kb, 3 kb, 4 kb, 5 kb) and reports:
- Number of divergent HKG pairs at each threshold
- Whether each known UCOE is recovered
- Number of candidates surviving all Phase I filters (estimated)

This analysis addresses the methodological concern that the 5 kb threshold
lacks empirical justification beyond recovering all known UCOEs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
import pandas as pd
from ucoe_pipeline.config import KNOWN_UCOES, setup_logging, OUTPUT_DIR
from ucoe_pipeline.phase1.filter_divergent_hkg import (
    parse_gencode_tss, load_housekeeping_genes,
    find_divergent_hkg_pairs, check_known_ucoes_recovered,
)
from ucoe_pipeline.config import GENCODE_GTF, HK_GENES_FILE
from ucoe_pipeline.utils.io_utils import ensure_dir

logger = setup_logging()


def main():
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS — Inter-TSS Distance Threshold")
    logger.info("=" * 60)

    # Parse gene annotations once
    tss_df = parse_gencode_tss(GENCODE_GTF)
    hk_genes = load_housekeeping_genes(HK_GENES_FILE)

    # Known UCOE inter-TSS distances for reference
    known_distances = {}
    for name, info in KNOWN_UCOES.items():
        dist = info["end"] - info["start"]
        known_distances[name] = dist
        logger.info("Known UCOE %s: inter-TSS distance = %d bp", name, dist)

    # Test multiple thresholds
    thresholds = [1000, 2000, 3000, 4000, 5000, 7500, 10000]
    results = []

    for threshold in thresholds:
        logger.info("\n--- Threshold: %d bp ---", threshold)
        candidates = find_divergent_hkg_pairs(tss_df, hk_genes, max_distance=threshold)
        n_candidates = len(candidates)

        # Check known UCOE recovery
        recovery = check_known_ucoes_recovered(candidates)

        result = {
            "threshold_bp": threshold,
            "n_divergent_pairs": n_candidates,
            "A2UCOE_recovered": recovery.get("A2UCOE_HNRNPA2B1_CBX3", False),
            "TBP_PSMB1_recovered": recovery.get("TBP_PSMB1", False),
            "SRF_UCOE_recovered": recovery.get("SRF_UCOE_SURF1_SURF2", False),
            "all_recovered": all(recovery.values()),
        }
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "phase1" / "threshold_sensitivity.tsv"
    ensure_dir(out_path.parent)
    df.to_csv(out_path, sep="\t", index=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("INTER-TSS DISTANCE THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nKnown UCOE inter-TSS distances:")
    for name, dist in known_distances.items():
        print(f"  {name}: {dist} bp")

    print(f"\n{'Threshold':>12s} | {'Pairs':>7s} | {'A2UCOE':>7s} | {'TBP/PSMB1':>9s} | {'SRF-UCOE':>9s} | {'All recovered':>14s}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(
            f"{row['threshold_bp']:>10d} bp | {row['n_divergent_pairs']:>7d} | "
            f"{'Yes' if row['A2UCOE_recovered'] else 'No':>7s} | "
            f"{'Yes' if row['TBP_PSMB1_recovered'] else 'No':>9s} | "
            f"{'Yes' if row['SRF_UCOE_recovered'] else 'No':>9s} | "
            f"{'Yes' if row['all_recovered'] else 'No':>14s}"
        )

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
