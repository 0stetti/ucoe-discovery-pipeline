#!/usr/bin/env python3
"""
Pre-compute PhyloP conservation scores for all 599 UCOE candidates.
Saves per-candidate summary + per-base arrays for top 50.
Output: webapp/data/conservation_summary.tsv + conservation_top50.npz
"""

import sys
import time
import numpy as np
import pandas as pd
import pyBigWig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "phase2"
OUT_DIR = ROOT / "webapp" / "data"

PHYLOP_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw"

def main():
    scored = pd.read_csv(DATA_DIR / "scored_candidates.tsv", sep="\t")
    print(f"Loading {len(scored)} candidates...")

    print("Connecting to UCSC PhyloP BigWig...")
    bw = pyBigWig.open(PHYLOP_URL)

    results = []
    per_base = {}  # only for top 50

    for idx, row in scored.iterrows():
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        rank = int(row["composite_rank"])
        label = f"{row['gene1']}/{row['gene2']}"

        try:
            vals = np.array(bw.values(chrom, start, end), dtype=np.float64)
        except Exception as e:
            print(f"  FAIL {label}: {e}")
            vals = np.full(end - start, np.nan)

        phylop_mean = np.nanmean(vals)
        phylop_median = np.nanmedian(vals)
        frac_positive = np.nanmean(vals > 0)
        frac_gt1 = np.nanmean(vals > 1)
        frac_gt2 = np.nanmean(vals > 2)

        results.append({
            "chrom": chrom, "start": start, "end": end,
            "gene1": row["gene1"], "gene2": row["gene2"],
            "composite_rank": rank,
            "phylop_mean": round(phylop_mean, 4),
            "phylop_median": round(phylop_median, 4),
            "phylop_frac_positive": round(frac_positive, 4),
            "phylop_frac_gt1": round(frac_gt1, 4),
            "phylop_frac_gt2": round(frac_gt2, 4),
        })

        # Save per-base for top 50
        if rank <= 50:
            key = f"{row['gene1']}_{row['gene2']}"
            per_base[key] = vals.astype(np.float32)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(scored)}...")
            time.sleep(0.5)  # be nice to UCSC

    bw.close()

    # Save summary
    df = pd.DataFrame(results)
    summary_path = OUT_DIR / "conservation_summary.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"Saved: {summary_path} ({len(df)} rows)")

    # Save per-base arrays
    npz_path = OUT_DIR / "conservation_top50.npz"
    np.savez_compressed(npz_path, **per_base)
    print(f"Saved: {npz_path} ({len(per_base)} candidates)")

    print("Done!")


if __name__ == "__main__":
    main()
