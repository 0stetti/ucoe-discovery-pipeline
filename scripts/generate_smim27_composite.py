#!/usr/bin/env python3
"""
Composite multi-panel figure for SMIM27/TOPORS (rank #1 candidate).

Panels:
A) Gene structure + ETS motif positions on both strands
B) PhyloP conservation profile (100 vertebrates) with ETS highlighted
C) PhastCons conservation probability
D) CpG island coverage (rectangle) + GC content profile
E) CpG dinucleotide density (sliding window)
F) ETS motif density (sliding window)
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import pyBigWig
import pysam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME_FA = os.path.join(PROJECT_ROOT, "ucoe_data/annotation/hg38.fa")
OUT_DIR = os.path.join(PROJECT_ROOT, "output/paper_figures")

PHYLOP_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw"
PHASTCONS_URL = "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw"

# SMIM27/TOPORS locus
CHROM = "chr9"
START = 32_550_561
END = 32_552_586
LENGTH = END - START  # 2025 bp

# Gene info (from GENCODE v44)
GENES = [
    {"name": "SMIM27", "strand": "+", "tss_rel": 582,
     "arrow_start": 582, "arrow_end": LENGTH, "color": "#2E86C1"},
    {"name": "TOPORS", "strand": "-", "tss_rel": 582 + 1443,
     "arrow_start": 0, "arrow_end": 582 + 1443, "color": "#E67E22"},
]

# CpG island (from UCSC — covers 97.4% of region)
CPG_ISLAND_START_REL = 0
CPG_ISLAND_END_REL = int(LENGTH * 0.974)  # ~1972 bp

ETS_FWD = re.compile(r"CGGAA[GA]")
ETS_REV = re.compile(r"[TC]TTCCG")

WINDOW = 100  # sliding window for profiles


def sliding_window_gc(seq, window=WINDOW):
    """GC content in sliding window."""
    gc = []
    positions = []
    for i in range(0, len(seq) - window + 1, window // 2):
        w = seq[i:i + window]
        gc_pct = (w.count("G") + w.count("C")) / len(w) * 100
        gc.append(gc_pct)
        positions.append(i + window // 2)
    return np.array(positions), np.array(gc)


def sliding_window_cpg(seq, window=WINDOW):
    """CpG dinucleotide count in sliding window."""
    counts = []
    positions = []
    for i in range(0, len(seq) - window + 1, window // 2):
        w = seq[i:i + window]
        c = w.count("CG")
        counts.append(c)
        positions.append(i + window // 2)
    return np.array(positions), np.array(counts)


def sliding_window_ets(motifs, length, window=200):
    """ETS motif count in sliding window."""
    counts = []
    positions = []
    for i in range(0, length - window + 1, window // 2):
        c = sum(1 for s, e, _, _ in motifs if s >= i and s < i + window)
        counts.append(c)
        positions.append(i + window // 2)
    return np.array(positions), np.array(counts)


def main():
    print("Loading genome...")
    genome = pysam.FastaFile(GENOME_FA)
    seq = genome.fetch(CHROM, START, END).upper()

    print("Connecting to UCSC BigWig servers...")
    phylop_bw = pyBigWig.open(PHYLOP_URL)
    phastcons_bw = pyBigWig.open(PHASTCONS_URL)

    # Get conservation scores
    print("Fetching PhyloP scores...")
    phylop = np.array(phylop_bw.values(CHROM, START, END), dtype=np.float64)
    print("Fetching PhastCons scores...")
    phastcons = np.array(phastcons_bw.values(CHROM, START, END), dtype=np.float64)

    # Find ETS motifs
    fwd = [(m.start(), m.end(), "+", m.group()) for m in ETS_FWD.finditer(seq)]
    rev = [(m.start(), m.end(), "-", m.group()) for m in ETS_REV.finditer(seq)]
    motifs = sorted(fwd + rev, key=lambda x: x[0])
    print(f"Found {len(motifs)} ETS motifs")

    # Compute profiles
    gc_pos, gc_vals = sliding_window_gc(seq)
    cpg_pos, cpg_vals = sliding_window_cpg(seq)
    ets_pos, ets_vals = sliding_window_ets(motifs, LENGTH)

    # ── FIGURE ──
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.5, 1, 0.6, 1, 0.8],
                                          "hspace": 0.08})

    positions = np.arange(LENGTH)
    x_label = f"Posicao no locus SMIM27/TOPORS (pb)"

    # ── Panel A: Gene structure + ETS motifs ──
    ax_gene = axes[0]
    ax_gene.set_xlim(0, LENGTH)
    ax_gene.set_ylim(-1.5, 2.5)
    ax_gene.set_ylabel("")
    ax_gene.set_title(
        f"SMIM27/TOPORS (rank #1) — Analise integrada\n"
        f"({CHROM}:{START:,}-{END:,}; {LENGTH} pb)",
        fontsize=14, fontweight="bold", pad=10
    )

    # Gene arrows
    for gene in GENES:
        y = 1.5 if gene["strand"] == "+" else 0.0
        dx = gene["arrow_end"] - gene["arrow_start"]
        direction = 1 if gene["strand"] == "+" else -1
        arrow_x = gene["arrow_start"] if direction == 1 else gene["arrow_end"]

        ax_gene.annotate(
            "", xy=(arrow_x + dx * direction, y),
            xytext=(arrow_x, y),
            arrowprops=dict(arrowstyle="-|>", color=gene["color"],
                            lw=8, mutation_scale=15),
        )
        # Gene name
        name_x = (gene["arrow_start"] + gene["arrow_end"]) / 2
        ax_gene.text(name_x, y + 0.5 * direction, f'*{gene["name"]}*',
                     ha="center", va="center", fontsize=11, fontweight="bold",
                     fontstyle="italic", color=gene["color"])

    # TSS markers
    for gene in GENES:
        tss = gene["tss_rel"]
        ax_gene.axvline(tss, color=gene["color"], linewidth=1, linestyle=":", alpha=0.5)

    # ETS motifs on gene panel
    for i, (ms, me, strand, motif_seq) in enumerate(motifs):
        y_motif = -0.5 if strand == "+" else -1.0
        marker = "v" if strand == "+" else "^"
        color = "#E74C3C" if strand == "+" else "#9B59B6"
        ax_gene.plot(ms, y_motif, marker=marker, markersize=8, color=color, zorder=5)
        ax_gene.annotate(f"#{i+1}", xy=(ms, y_motif - 0.3),
                         fontsize=7, ha="center", color=color)

    # Legend for ETS
    ax_gene.plot([], [], "v", color="#E74C3C", markersize=8, label="ETS motivo (+)")
    ax_gene.plot([], [], "^", color="#9B59B6", markersize=8, label="ETS motivo (−)")
    ax_gene.legend(loc="upper right", fontsize=8, ncol=2)

    ax_gene.spines["bottom"].set_visible(False)
    ax_gene.spines["left"].set_visible(False)
    ax_gene.tick_params(left=False, labelleft=False)
    ax_gene.text(-0.02, 0.95, "A", transform=ax_gene.transAxes,
                 fontsize=16, fontweight="bold", va="top")

    # ── Panel B: PhyloP ──
    ax_phylop = axes[1]
    ax_phylop.fill_between(positions, phylop, 0, where=phylop > 0,
                           color="#2196F3", alpha=0.4)
    ax_phylop.fill_between(positions, phylop, 0, where=phylop < 0,
                           color="#FF5722", alpha=0.4)
    ax_phylop.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, (ms, me, strand, _) in enumerate(motifs):
        ax_phylop.axvspan(ms, me, color="#FFD700", alpha=0.6, zorder=0)

    ax_phylop.set_ylabel("PhyloP\n(100 vertebrados)", fontsize=10)
    ax_phylop.text(-0.02, 0.95, "B", transform=ax_phylop.transAxes,
                   fontsize=16, fontweight="bold", va="top")

    # Annotate exceptional ETS #6
    ets6_pos = motifs[5][0]
    ets6_phylop = np.nanmean(phylop[motifs[5][0]:motifs[5][1]])
    ax_phylop.annotate(
        f"ETS #6\nPhyloP = {ets6_phylop:.1f}",
        xy=(ets6_pos, ets6_phylop), xytext=(ets6_pos - 300, ets6_phylop + 1.5),
        fontsize=9, fontweight="bold", color="#D32F2F",
        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEB3B", alpha=0.9)
    )

    # ── Panel C: PhastCons ──
    ax_phast = axes[2]
    ax_phast.fill_between(positions, phastcons, color="#4CAF50", alpha=0.5)
    for ms, me, _, _ in motifs:
        ax_phast.axvspan(ms, me, color="#FFD700", alpha=0.6, zorder=0)
    ax_phast.set_ylabel("PhastCons\n(prob. conserv.)", fontsize=10)
    ax_phast.set_ylim(0, 1.05)
    ax_phast.text(-0.02, 0.95, "C", transform=ax_phast.transAxes,
                  fontsize=16, fontweight="bold", va="top")

    # ── Panel D: CpG island track + GC content ──
    ax_cpg_island = axes[3]

    # CpG island rectangle
    rect = mpatches.FancyBboxPatch(
        (CPG_ISLAND_START_REL, 0.1), CPG_ISLAND_END_REL - CPG_ISLAND_START_REL, 0.8,
        boxstyle="round,pad=0.01", facecolor="#81C784", edgecolor="#388E3C",
        linewidth=1.5, alpha=0.7
    )
    ax_cpg_island.add_patch(rect)
    ax_cpg_island.text(
        (CPG_ISLAND_START_REL + CPG_ISLAND_END_REL) / 2, 0.5,
        f"Ilha CpG (97,4% cobertura; obs/exp = 57,8)",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#1B5E20"
    )
    ax_cpg_island.set_ylim(0, 1)
    ax_cpg_island.set_ylabel("Ilha CpG", fontsize=10)
    ax_cpg_island.spines["left"].set_visible(False)
    ax_cpg_island.tick_params(left=False, labelleft=False)
    ax_cpg_island.text(-0.02, 0.95, "D", transform=ax_cpg_island.transAxes,
                       fontsize=16, fontweight="bold", va="top")

    # ── Panel E: GC content + CpG density ──
    ax_gc = axes[4]
    color_gc = "#00897B"
    color_cpg = "#FF8F00"

    ax_gc.fill_between(gc_pos, gc_vals, color=color_gc, alpha=0.3)
    ax_gc.plot(gc_pos, gc_vals, color=color_gc, linewidth=1.2, label="GC (%)")
    ax_gc.axhline(50, color=color_gc, linewidth=0.5, linestyle="--", alpha=0.5)
    ax_gc.set_ylabel("GC (%)", fontsize=10, color=color_gc)
    ax_gc.tick_params(axis="y", labelcolor=color_gc)

    # CpG density on secondary axis
    ax_cpg = ax_gc.twinx()
    ax_cpg.bar(cpg_pos, cpg_vals, width=WINDOW * 0.4, color=color_cpg, alpha=0.5, label="CpG count")
    ax_cpg.set_ylabel("CpGs / janela", fontsize=10, color=color_cpg)
    ax_cpg.tick_params(axis="y", labelcolor=color_cpg)

    # Combined legend
    lines1, labels1 = ax_gc.get_legend_handles_labels()
    lines2, labels2 = ax_cpg.get_legend_handles_labels()
    ax_gc.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax_gc.text(-0.02, 0.95, "E", transform=ax_gc.transAxes,
               fontsize=16, fontweight="bold", va="top")

    # ── Panel F: ETS density ──
    ax_ets = axes[5]
    ax_ets.bar(ets_pos, ets_vals, width=100, color="#E74C3C", alpha=0.7,
               edgecolor="#B71C1C", linewidth=0.5)
    ax_ets.set_ylabel("ETS motivos\n/ janela 200pb", fontsize=10)
    ax_ets.set_xlabel(x_label, fontsize=11)
    ax_ets.set_ylim(0, max(ets_vals) + 0.5)
    ax_ets.text(-0.02, 0.95, "F", transform=ax_ets.transAxes,
                fontsize=16, fontweight="bold", va="top")

    # ── Final adjustments ──
    for ax in axes:
        ax.set_xlim(0, LENGTH)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"fig_smim27_topors_composite.{ext}")
        fig.savefig(path)
        print(f"Saved: {path}")

    plt.close(fig)

    phylop_bw.close()
    phastcons_bw.close()
    genome.close()
    print("Done!")


if __name__ == "__main__":
    main()
