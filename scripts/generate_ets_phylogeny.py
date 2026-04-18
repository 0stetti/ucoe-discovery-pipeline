#!/usr/bin/env python3
"""
Phylogenetic conservation analysis of individual ETS motifs in known UCOEs.

For each ETS motif (CGGAA[GA]) in the 3 known human UCOEs:
1. Extract per-base PhyloP and PhastCons scores (100 vertebrates)
2. Compare conservation at ETS motifs vs flanking regions
3. Generate a heatmap showing conservation of each motif across the locus
4. Build a conservation summary per motif site

Uses remote BigWig access to UCSC (no large downloads needed).
"""

import re
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pyBigWig
import pysam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME_FA = os.path.join(PROJECT_ROOT, "ucoe_data/annotation/hg38.fa")
OUT_DIR = os.path.join(PROJECT_ROOT, "output/paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# UCSC BigWig URLs
PHYLOP_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw"
PHASTCONS_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw"

# ETS motif patterns
ETS_FWD = re.compile(r"CGGAA[GA]")
ETS_REV = re.compile(r"[TC]TTCCG")

# Known UCOEs
UCOES = {
    "A2UCOE": {"chrom": "chr7", "start": 26_199_798, "end": 26_202_442},
    "TBP/PSMB1": {"chrom": "chr6", "start": 170_553_036, "end": 170_554_735},
    "SRF-UCOE": {"chrom": "chr9", "start": 133_356_273, "end": 133_357_090},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def find_ets_motifs(seq):
    """Find all ETS motifs in sequence, return list of (start, end, strand, motif)."""
    motifs = []
    for m in ETS_FWD.finditer(seq):
        motifs.append((m.start(), m.end(), "+", m.group()))
    for m in ETS_REV.finditer(seq):
        motifs.append((m.start(), m.end(), "-", m.group()))
    motifs.sort(key=lambda x: x[0])
    return motifs


def get_conservation_scores(bw, chrom, start, end):
    """Get per-base scores from remote BigWig."""
    try:
        vals = bw.values(chrom, start, end)
        return np.array(vals, dtype=np.float64)
    except Exception as e:
        print(f"  Warning: failed to get scores for {chrom}:{start}-{end}: {e}")
        return np.full(end - start, np.nan)


def analyze_ucoe_ets_conservation(ucoe_name, info, phylop_bw, phastcons_bw, genome):
    """Analyze conservation of each ETS motif in one UCOE."""
    chrom = info["chrom"]
    start = info["start"]
    end = info["end"]
    length = end - start

    # Extract sequence
    seq = genome.fetch(chrom, start, end).upper()

    # Find ETS motifs
    motifs = find_ets_motifs(seq)
    print(f"\n{ucoe_name}: {len(motifs)} ETS motifs in {length} bp")

    # Get conservation scores for entire region
    phylop = get_conservation_scores(phylop_bw, chrom, start, end)
    phastcons = get_conservation_scores(phastcons_bw, chrom, start, end)

    # Per-motif conservation
    motif_data = []
    for i, (ms, me, strand, motif_seq) in enumerate(motifs):
        # Scores at motif
        pp_motif = phylop[ms:me]
        pc_motif = phastcons[ms:me]

        # Flanking 50bp on each side
        flank_l = max(0, ms - 50)
        flank_r = min(length, me + 50)
        pp_flank = np.concatenate([phylop[flank_l:ms], phylop[me:flank_r]])
        pc_flank = np.concatenate([phastcons[flank_l:ms], phastcons[me:flank_r]])

        motif_info = {
            "ucoe": ucoe_name,
            "motif_id": i + 1,
            "position_rel": ms,
            "position_abs": start + ms,
            "strand": strand,
            "sequence": motif_seq,
            "phylop_mean": np.nanmean(pp_motif),
            "phylop_max": np.nanmax(pp_motif),
            "phylop_min": np.nanmin(pp_motif),
            "phylop_per_base": [round(x, 3) for x in pp_motif if not np.isnan(x)],
            "phastcons_mean": np.nanmean(pc_motif),
            "phastcons_max": np.nanmax(pc_motif),
            "flanking_phylop_mean": np.nanmean(pp_flank),
            "flanking_phastcons_mean": np.nanmean(pc_flank),
            "enrichment_phylop": np.nanmean(pp_motif) / max(np.nanmean(pp_flank), 0.001),
            "enrichment_phastcons": np.nanmean(pc_motif) / max(np.nanmean(pc_flank), 0.001),
        }
        motif_data.append(motif_info)

        print(f"  ETS #{i+1} ({strand}) pos={ms}: seq={motif_seq}  "
              f"PhyloP={motif_info['phylop_mean']:.3f} (flank={motif_info['flanking_phylop_mean']:.3f})  "
              f"PhastCons={motif_info['phastcons_mean']:.3f} (flank={motif_info['flanking_phastcons_mean']:.3f})")

    # Background (non-motif positions)
    motif_mask = np.zeros(length, dtype=bool)
    for ms, me, _, _ in motifs:
        motif_mask[ms:me] = True
    bg_phylop = np.nanmean(phylop[~motif_mask])
    bg_phastcons = np.nanmean(phastcons[~motif_mask])
    print(f"  Background: PhyloP={bg_phylop:.3f}, PhastCons={bg_phastcons:.3f}")

    return motif_data, phylop, phastcons, motifs


def plot_ets_conservation_profile(ucoe_name, info, phylop, phastcons, motifs, out_dir):
    """Plot conservation profile along UCOE with ETS motifs highlighted."""
    length = info["end"] - info["start"]
    positions = np.arange(length)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})

    # PhyloP profile
    ax1 = axes[0]
    ax1.fill_between(positions, phylop, 0, where=phylop > 0,
                     color="#2196F3", alpha=0.4, label="Conservado")
    ax1.fill_between(positions, phylop, 0, where=phylop < 0,
                     color="#FF5722", alpha=0.4, label="Acelerado")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Highlight ETS motifs
    for ms, me, strand, seq in motifs:
        ax1.axvspan(ms, me, color="#FFD700", alpha=0.5, zorder=0)
        ax1.annotate(f"ETS ({strand})", xy=((ms + me) / 2, ax1.get_ylim()[1] * 0.85),
                     fontsize=7, ha="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFD700", alpha=0.8))

    ax1.set_ylabel("PhyloP\n(100 vertebrados)", fontsize=10)
    ax1.set_title(f"{ucoe_name} — Conservacao evolutiva por posicao\n"
                  f"({info['chrom']}:{info['start']:,}-{info['end']:,})",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # PhastCons profile
    ax2 = axes[1]
    ax2.fill_between(positions, phastcons, color="#4CAF50", alpha=0.5)
    for ms, me, strand, seq in motifs:
        ax2.axvspan(ms, me, color="#FFD700", alpha=0.5, zorder=0)

    ax2.set_ylabel("PhastCons\n(prob. conservacao)", fontsize=10)
    ax2.set_xlabel(f"Posicao no {ucoe_name} (pb)", fontsize=10)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_ets_phylogeny_{ucoe_name.replace('/', '_')}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: fig_ets_phylogeny_{ucoe_name.replace('/', '_')}.png")


def plot_summary_heatmap(all_motif_data, out_dir):
    """Plot summary heatmap of PhyloP scores per ETS motif across all UCOEs."""
    df = pd.DataFrame(all_motif_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

    # --- Left: bar chart of PhyloP per motif vs flanking ---
    ax1 = axes[0]
    labels = [f"{r['ucoe']}\nETS #{r['motif_id']} ({r['strand']})" for _, r in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df["phylop_mean"], width,
                    color="#2196F3", alpha=0.8, label="Motivo ETS")
    bars2 = ax1.bar(x + width/2, df["flanking_phylop_mean"], width,
                    color="#BDBDBD", alpha=0.8, label="Flanqueamento (50 pb)")

    ax1.set_ylabel("PhyloP medio\n(100 vertebrados)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax1.legend(fontsize=9)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("Conservacao evolutiva: motivos ETS vs. flanqueamento", fontsize=12, fontweight="bold")

    # Add enrichment ratio on top of bars
    for i, (_, row) in enumerate(df.iterrows()):
        enrich = row["enrichment_phylop"]
        ax1.text(i - width/2, row["phylop_mean"] + 0.05,
                 f"{enrich:.1f}x", ha="center", fontsize=7, fontweight="bold", color="#1565C0")

    # --- Right: PhastCons comparison ---
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, df["phastcons_mean"], width,
                    color="#4CAF50", alpha=0.8, label="Motivo ETS")
    bars4 = ax2.bar(x + width/2, df["flanking_phastcons_mean"], width,
                    color="#BDBDBD", alpha=0.8, label="Flanqueamento")

    ax2.set_ylabel("PhastCons medio", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{r['motif_id']}" for _, r in df.iterrows()], fontsize=9)
    ax2.set_xlabel("Motivo ETS", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Probabilidade de conservacao", fontsize=12, fontweight="bold")

    # Color-code by UCOE
    ucoe_colors = {"A2UCOE": "#E74C3C", "TBP/PSMB1": "#2ECC71", "SRF-UCOE": "#3498DB"}
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.get_xticklabels()[i].set_color(ucoe_colors.get(row["ucoe"], "black"))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_ets_phylogeny_summary.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"\nSaved: fig_ets_phylogeny_summary.png")


def main():
    print("Opening genome reference...")
    genome = pysam.FastaFile(GENOME_FA)

    print("Connecting to UCSC BigWig servers (PhyloP + PhastCons 100 vertebrates)...")
    phylop_bw = pyBigWig.open(PHYLOP_URL)
    phastcons_bw = pyBigWig.open(PHASTCONS_URL)
    print("  Connected successfully.")

    all_motif_data = []

    for ucoe_name, info in UCOES.items():
        motif_data, phylop, phastcons, motifs = analyze_ucoe_ets_conservation(
            ucoe_name, info, phylop_bw, phastcons_bw, genome
        )
        all_motif_data.extend(motif_data)

        # Per-UCOE conservation profile
        plot_ets_conservation_profile(ucoe_name, info, phylop, phastcons, motifs, OUT_DIR)

    # Summary figure
    plot_summary_heatmap(all_motif_data, OUT_DIR)

    # Save table
    df = pd.DataFrame(all_motif_data)
    cols_to_save = [c for c in df.columns if c != "phylop_per_base"]
    tsv_path = os.path.join(OUT_DIR, "ets_motif_conservation_table.tsv")
    df[cols_to_save].to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSaved table: {tsv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESUMO: CONSERVACAO DOS MOTIVOS ETS NOS UCOEs CONHECIDOS")
    print("=" * 70)
    for ucoe_name in UCOES:
        subset = df[df["ucoe"] == ucoe_name]
        print(f"\n{ucoe_name}:")
        for _, row in subset.iterrows():
            print(f"  ETS #{row['motif_id']} ({row['strand']}) pos={row['position_rel']}: "
                  f"PhyloP={row['phylop_mean']:.3f} vs flank={row['flanking_phylop_mean']:.3f} "
                  f"({row['enrichment_phylop']:.1f}x)  "
                  f"PhastCons={row['phastcons_mean']:.3f} vs flank={row['flanking_phastcons_mean']:.3f}")

    mean_motif = df["phylop_mean"].mean()
    mean_flank = df["flanking_phylop_mean"].mean()
    print(f"\nMedia global: ETS PhyloP={mean_motif:.3f} vs Flanking={mean_flank:.3f} "
          f"(enriquecimento={mean_motif/max(mean_flank, 0.001):.1f}x)")

    phylop_bw.close()
    phastcons_bw.close()
    genome.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
