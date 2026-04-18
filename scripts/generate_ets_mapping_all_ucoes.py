#!/usr/bin/env python3
"""
Generate ETS motif distribution figures for all three known UCOEs:
- A2UCOE (HNRNPA2B1/CBX3)
- TBP/PSMB1
- SRF-UCOE (SURF1/SURF2)

Each figure shows:
A) Gene structure + ETS motif positions on both strands
B) Known functional fragments (where literature exists)
C) GC content profile (sliding window)
D) CpG density profile (sliding window)
E) ETS motif density per window
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pysam

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME_FA = os.path.join(PROJECT_ROOT, "ucoe_data/annotation/hg38.fa")

# ── UCOE definitions ──────────────────────────────────────────────────────────
UCOES = {
    "A2UCOE": {
        "chrom": "chr7",
        "start": 26_199_798,
        "end": 26_202_442,
        "genes": [
            {"name": "CBX3", "strand": "+", "tss_rel": 1364,
             "arrow_start": 1364, "arrow_end": 2644, "color": "#2E86C1"},
            {"name": "HNRNPA2B1", "strand": "-", "tss_rel": 1731,
             "arrow_start": 0, "arrow_end": 1731, "color": "#E67E22"},
        ],
        "fragments": [
            {"name": "Full A2UCOE (2.6 kb)", "start": 0, "end": 2644,
             "color": "#E8A87C", "activity": "Full activity"},
            {"name": "1.5 kb UCOE", "start": 0, "end": 1500,
             "color": "#E8A87C", "activity": "Full activity"},
            {"name": "455 bp core\n(Zhang et al. 2017)", "start": 1050, "end": 1505,
             "color": "#F4D03F", "activity": "Full activity"},
            {"name": "HNRNPA2B1 side (1.1 kb)", "start": 1500, "end": 2644,
             "color": "#ABEBC6", "activity": "CpG-dependent\n(Kunkiel et al.)"},
        ],
        "title": "ETS Motif Distribution Across the A2UCOE Locus",
        "subtitle": "chr7:26,199,798-26,202,442",
    },
    "TBP_PSMB1": {
        "chrom": "chr6",
        "start": 170_553_036,
        "end": 170_554_735,
        "genes": [
            {"name": "TBP", "strand": "+", "tss_rel": 1266,
             "arrow_start": 1266, "arrow_end": 1699, "color": "#2E86C1"},
            {"name": "PSMB1", "strand": "-", "tss_rel": 271,
             "arrow_start": 0, "arrow_end": 271, "color": "#E67E22"},
        ],
        "fragments": [],  # no published fragment studies
        "title": "ETS Motif Distribution Across the TBP/PSMB1 UCOE Locus",
        "subtitle": "chr6:170,553,036-170,554,735",
    },
    "SRF_UCOE": {
        "chrom": "chr9",
        "start": 133_356_273,
        "end": 133_357_090,
        "genes": [
            {"name": "SURF2", "strand": "+", "tss_rel": 277,
             "arrow_start": 277, "arrow_end": 817, "color": "#2E86C1"},
            {"name": "SURF1", "strand": "-", "tss_rel": 403,
             "arrow_start": 0, "arrow_end": 403, "color": "#E67E22"},
        ],
        "fragments": [],  # Rudina & Smolke 2019 tested full element
        "title": "ETS Motif Distribution Across the SRF-UCOE Locus",
        "subtitle": "chr9:133,356,273-133,357,090",
    },
}

ETS_PATTERN_FWD = re.compile(r"CGGAA[GA]", re.IGNORECASE)
ETS_PATTERN_REV = re.compile(r"[TC]TTCCG", re.IGNORECASE)


def fetch_sequence(chrom, start, end):
    """Fetch sequence from local hg38 FASTA."""
    fasta = pysam.FastaFile(GENOME_FA)
    seq = fasta.fetch(chrom, start, end).upper()
    fasta.close()
    print(f"  Extracted {chrom}:{start}-{end} ({len(seq)} bp) from local genome")
    return seq


def find_ets_motifs(seq):
    """Find ETS motifs on both strands. Returns list of (pos, strand, motif)."""
    motifs = []
    for m in ETS_PATTERN_FWD.finditer(seq):
        motifs.append((m.start(), "+", m.group()))
    for m in ETS_PATTERN_REV.finditer(seq):
        motifs.append((m.start(), "-", m.group()))
    return sorted(motifs, key=lambda x: x[0])


def sliding_window(seq, func, window=50, step=10):
    """Apply func to sliding windows across sequence."""
    positions = []
    values = []
    for i in range(0, len(seq) - window + 1, step):
        w = seq[i:i + window]
        positions.append(i + window // 2)
        values.append(func(w))
    return np.array(positions), np.array(values)


def gc_content(seq):
    """Calculate GC content as percentage."""
    gc = sum(1 for b in seq if b in "GC")
    return 100.0 * gc / len(seq) if len(seq) > 0 else 0


def cpg_count(seq):
    """Count CpG dinucleotides."""
    return sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")


def ets_density(motifs, seq_len, window=200):
    """Calculate ETS motif count per window."""
    bins = list(range(0, seq_len, window))
    counts = []
    for b in bins:
        n = sum(1 for pos, _, _ in motifs if b <= pos < b + window)
        counts.append(n)
    return np.array(bins), np.array(counts)


def generate_figure(ucoe_key, ucoe_info, seq):
    """Generate multi-panel ETS mapping figure for one UCOE."""
    motifs = find_ets_motifs(seq)
    seq_len = len(seq)
    n_ets = len(motifs)

    print(f"  Found {n_ets} ETS motifs: "
          f"{sum(1 for _,s,_ in motifs if s=='+')} fwd, "
          f"{sum(1 for _,s,_ in motifs if s=='-')} rev")

    has_fragments = len(ucoe_info["fragments"]) > 0
    n_panels = 5 if has_fragments else 4
    height_ratios = [1.5, 1.2, 1.2, 1.2, 1.2] if has_fragments else [1.5, 1.2, 1.2, 1.2]

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.0 * n_panels),
                              gridspec_kw={"height_ratios": height_ratios},
                              sharex=True)
    fig.suptitle(f"{ucoe_info['title']}\n({ucoe_info['subtitle']})",
                 fontsize=14, fontweight="bold", y=0.98)

    panel_idx = 0

    # ── Panel A: Gene structure + ETS positions ──
    ax = axes[panel_idx]
    ax.set_xlim(0, seq_len)
    ax.set_ylim(-2, 3.5)
    ax.set_ylabel("")
    ax.text(-0.02, 1.0, "A", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")

    # Draw locus bar
    ax.barh(0, seq_len, height=0.3, left=0, color="#D5D8DC", edgecolor="#AAB7B8")
    ax.text(seq_len / 2, 0, f"{ucoe_key.replace('_', '/')} — {seq_len} bp",
            ha="center", va="center", fontsize=9, style="italic")

    # Draw genes
    for i, gene in enumerate(ucoe_info["genes"]):
        y = 1.5 + i * 0.8
        color = gene["color"]
        s, e = gene["arrow_start"], gene["arrow_end"]
        if gene["strand"] == "+":
            ax.annotate("", xy=(e, y), xytext=(s, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        else:
            ax.annotate("", xy=(s, y), xytext=(e, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        mid = (s + e) / 2
        ax.text(mid, y + 0.35, f"*{gene['name']}*", ha="center", va="bottom",
                fontsize=11, fontstyle="italic", fontweight="bold", color=color)

    # Bidirectional promoter annotation
    tss_positions = [g["tss_rel"] for g in ucoe_info["genes"]]
    mid_tss = np.mean(tss_positions)
    ax.axvspan(min(tss_positions) - 50, max(tss_positions) + 50,
               alpha=0.15, color="#F39C12", zorder=0)
    ax.text(mid_tss, 3.3, "Bidirectional\npromoter", ha="center", va="top",
            fontsize=8, color="#E67E22",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7", edgecolor="#F39C12"))

    # Draw ETS motifs on strands
    for pos, strand, motif_seq in motifs:
        y_motif = -0.8 if strand == "+" else -1.5
        ax.plot(pos, y_motif, marker="v", color="#C0392B", markersize=10, zorder=5)
        ax.text(pos, y_motif - 0.45, motif_seq.upper(), ha="center", va="top",
                fontsize=6.5, color="#C0392B", fontweight="bold")

    ax.text(0, -0.5, "+", fontsize=10, fontweight="bold", va="center")
    ax.text(0, -1.2, "−", fontsize=10, fontweight="bold", va="center")

    # Legend
    legend_elements = [Line2D([0], [0], marker="v", color="w", markerfacecolor="#C0392B",
                              markersize=10, label="ETS motif (CGGAA[GA])")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    panel_idx += 1

    # ── Panel B: Functional fragments (only for A2UCOE) ──
    if has_fragments:
        ax = axes[panel_idx]
        ax.set_xlim(0, seq_len)
        ax.text(-0.02, 1.0, "B", transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")

        for i, frag in enumerate(ucoe_info["fragments"]):
            y = len(ucoe_info["fragments"]) - i - 1
            width = frag["end"] - frag["start"]
            ax.barh(y, width, left=frag["start"], height=0.6,
                    color=frag["color"], edgecolor="#AAB7B8", alpha=0.8)

            # Count ETS in fragment
            n_in = sum(1 for p, _, _ in motifs if frag["start"] <= p < frag["end"])
            ax.text(frag["start"] + 5, y, f"{n_in} ETS", va="center",
                    fontsize=8, fontweight="bold", color="#C0392B")
            ax.text(max(frag["start"] + width + 10, frag["end"] + 10), y,
                    frag["activity"], va="center", fontsize=8, color="#666666")
            ax.text(frag["start"] + width / 2, y + 0.35, frag["name"],
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_yticks([])
        ax.set_ylim(-0.5, len(ucoe_info["fragments"]) + 0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        panel_idx += 1

    # ── Panel C: GC content ──
    ax = axes[panel_idx]
    panel_letter = chr(ord("B") + panel_idx) if not has_fragments else chr(ord("A") + panel_idx)
    ax.text(-0.02, 1.0, panel_letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    pos_gc, val_gc = sliding_window(seq, gc_content, window=50, step=10)
    ax.fill_between(pos_gc, val_gc, alpha=0.6, color="#1ABC9C")
    ax.plot(pos_gc, val_gc, color="#16A085", linewidth=0.8)
    mean_gc = gc_content(seq)
    ax.axhline(mean_gc, color="#7F8C8D", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(seq_len * 0.98, mean_gc + 1, f"mean", ha="right", va="bottom",
            fontsize=8, color="#7F8C8D")
    ax.set_ylabel("GC (%)", fontsize=10)
    ax.set_ylim(30, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_idx += 1

    # ── Panel D: CpG density ──
    ax = axes[panel_idx]
    panel_letter = chr(ord("B") + panel_idx) if not has_fragments else chr(ord("A") + panel_idx)
    ax.text(-0.02, 1.0, panel_letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    pos_cpg, val_cpg = sliding_window(seq, cpg_count, window=50, step=10)
    ax.fill_between(pos_cpg, val_cpg, alpha=0.5, color="#82E0AA")
    ax.plot(pos_cpg, val_cpg, color="#27AE60", linewidth=0.8)
    ax.set_ylabel("CpG count\n(per 50 bp)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_idx += 1

    # ── Panel E: ETS density ──
    ax = axes[panel_idx]
    panel_letter = chr(ord("B") + panel_idx) if not has_fragments else chr(ord("A") + panel_idx)
    ax.text(-0.02, 1.0, panel_letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top")
    window_size = 200 if seq_len > 1000 else 100
    bins_ets, counts_ets = ets_density(motifs, seq_len, window=window_size)
    ax.bar(bins_ets + window_size / 2, counts_ets, width=window_size * 0.9,
           color="#C0392B", alpha=0.8, edgecolor="#922B21")
    ax.set_ylabel(f"ETS motifs\n(per {window_size} bp)", fontsize=10)
    ax.set_xlabel(f"Position in {ucoe_key.replace('_', '/')} (bp)", fontsize=11)
    ax.set_ylim(0, max(max(counts_ets) + 1, 4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight regions with motifs
    if has_fragments:
        core = ucoe_info["fragments"][2]  # 455bp core
        ax.axvspan(core["start"], core["end"], alpha=0.15, color="#F4D03F", zorder=0)
        ax.text((core["start"] + core["end"]) / 2, max(counts_ets) + 0.5,
                "455 bp core\n(full activity)", ha="center", va="top",
                fontsize=8, color="#B7950B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7",
                          edgecolor="#F4D03F", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.15)

    outpath = os.path.join(PROJECT_ROOT, f"output/paper_figures/fig_ets_mapping_{ucoe_key.lower()}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath, motifs


def print_summary(ucoe_key, motifs, seq):
    """Print summary statistics for the UCOE."""
    n_fwd = sum(1 for _, s, _ in motifs if s == "+")
    n_rev = sum(1 for _, s, _ in motifs if s == "-")
    density = len(motifs) / (len(seq) / 1000)
    print(f"\n  === {ucoe_key} Summary ===")
    print(f"  Sequence length: {len(seq)} bp")
    print(f"  GC content: {gc_content(seq):.1f}%")
    print(f"  CpG count: {seq.count('CG')}")
    print(f"  Total ETS motifs: {len(motifs)} ({n_fwd} fwd, {n_rev} rev)")
    print(f"  ETS density: {density:.1f} motifs/kb")
    print(f"  Motif positions:")
    for pos, strand, mseq in motifs:
        print(f"    pos {pos:>5d} ({strand}) {mseq}")


if __name__ == "__main__":
    results = {}

    for key, info in UCOES.items():
        print(f"\n{'='*60}")
        print(f"Processing {key}...")
        print(f"{'='*60}")

        seq = fetch_sequence(info["chrom"], info["start"], info["end"])
        outpath, motifs = generate_figure(key, info, seq)
        print_summary(key, motifs, seq)
        results[key] = {"motifs": motifs, "seq_len": len(seq), "path": outpath}

    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*60}")
    print(f"{'UCOE':<20} {'Length':>8} {'ETS motifs':>12} {'Density':>12}")
    print("-" * 55)
    for key, r in results.items():
        density = len(r["motifs"]) / (r["seq_len"] / 1000)
        print(f"{key:<20} {r['seq_len']:>8} bp {len(r['motifs']):>10} "
              f"{density:>9.1f}/kb")
