#!/usr/bin/env python3
"""
Run k-mer enrichment analysis on UCOE candidate sequences.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ucoe_pipeline.structural.analysis import (
    read_fasta, extract_known_ucoe_sequences, sample_random_cpg_islands,
)
from ucoe_pipeline.structural.kmer_analysis import (
    run_kmer_analysis, reverse_complement,
)
from ucoe_pipeline.config import KNOWN_UCOES, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = OUTPUT_DIR / "kmer_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sequences():
    cand_seqs = read_fasta(OUTPUT_DIR / "ucoe_sequences.fa")
    known_seqs = extract_known_ucoe_sequences()
    exclude = [(info["chrom"], info["start"], info["end"]) for info in KNOWN_UCOES.values()]
    for h in cand_seqs:
        if "::" in h:
            coord = h.split("::")[1]
            try:
                c, p = coord.split(":")
                s, e = p.split("-")
                exclude.append((c, int(s), int(e)))
            except Exception:
                pass
    ctrl_seqs = sample_random_cpg_islands(n_samples=200, exclude_regions=exclude)
    return cand_seqs, known_seqs, ctrl_seqs


def fig_volcano(enrichment, k, out_dir):
    """Volcano plot: fold enrichment vs -log10(q-value)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fes = [r["fold_enrichment"] for r in enrichment]
    qvals = [r.get("q_value", 1.0) for r in enrichment]
    log2fe = [np.log2(fe) for fe in fes]
    neg_log_q = [-np.log10(max(q, 1e-300)) for q in qvals]

    shared = [r.get("shared_in_known_ucoes", False) for r in enrichment]
    sig = [r.get("significant", False) for r in enrichment]
    tf = [bool(r.get("tf_matches")) for r in enrichment]

    # Non-significant
    ns_x = [x for x, s in zip(log2fe, sig) if not s]
    ns_y = [y for y, s in zip(neg_log_q, sig) if not s]
    ax.scatter(ns_x, ns_y, c="lightgray", s=8, alpha=0.5, label="Not significant")

    # Significant but not shared
    sig_x = [x for x, s, sh in zip(log2fe, sig, shared) if s and not sh]
    sig_y = [y for y, s, sh in zip(neg_log_q, sig, shared) if s and not sh]
    ax.scatter(sig_x, sig_y, c="#2196F3", s=15, alpha=0.6, label="Significant (FDR<0.05)")

    # Significant AND shared in known UCOEs
    sh_x = [x for x, s, sh in zip(log2fe, sig, shared) if s and sh]
    sh_y = [y for y, s, sh in zip(neg_log_q, sig, shared) if s and sh]
    ax.scatter(sh_x, sh_y, c="#E53935", s=30, alpha=0.8, edgecolor="black", lw=0.5,
              label="Significant + shared in known UCOEs")

    # Label top shared k-mers
    shared_enriched = [(r, np.log2(r["fold_enrichment"]), -np.log10(max(r.get("q_value", 1e-300), 1e-300)))
                       for r in enrichment if r.get("significant") and r.get("shared_in_known_ucoes")]
    shared_enriched.sort(key=lambda x: x[2], reverse=True)
    for r, x, y in shared_enriched[:15]:
        kmer = r["kmer"]
        tf_str = f" ({r['tf_annotation']})" if r["tf_annotation"] else ""
        ax.annotate(f"{kmer}{tf_str}", (x, y), fontsize=6, ha="left",
                   xytext=(4, 2), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("log₂(Fold Enrichment: Candidates / Controls)")
    ax.set_ylabel("-log₁₀(q-value, BH-FDR)")
    ax.set_title(f"K-mer Enrichment Volcano Plot (k={k})", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / f"volcano_k{k}.png")
    fig.savefig(out_dir / f"volcano_k{k}.pdf")
    plt.close(fig)


def fig_top_kmers_bar(enrichment, k, out_dir, n_top=25):
    """Bar plot of top enriched and depleted k-mers."""
    sig = [r for r in enrichment if r.get("significant")]
    enriched = sorted([r for r in sig if r["fold_enrichment"] > 1],
                      key=lambda x: x["fold_enrichment"], reverse=True)[:n_top]
    depleted = sorted([r for r in sig if r["fold_enrichment"] < 1],
                      key=lambda x: x["fold_enrichment"])[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Enriched
    if enriched:
        labels = []
        for r in enriched:
            lbl = r["kmer"]
            if r.get("shared_in_known_ucoes"):
                lbl += " ★"
            if r.get("tf_annotation"):
                lbl += f" [{r['tf_annotation']}]"
            labels.append(lbl)
        vals = [np.log2(r["fold_enrichment"]) for r in enriched]
        colors = ["#E53935" if r.get("shared_in_known_ucoes") else "#2196F3" for r in enriched]
        y_pos = range(len(enriched))
        axes[0].barh(y_pos, vals, color=colors, edgecolor="black", lw=0.3)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xlabel("log₂(Fold Enrichment)")
        axes[0].set_title(f"Top {n_top} Enriched {k}-mers in UCOE Candidates", fontweight="bold")
        axes[0].invert_yaxis()

    # Depleted
    if depleted:
        labels = [r["kmer"] for r in depleted]
        vals = [np.log2(r["fold_enrichment"]) for r in depleted]
        axes[1].barh(range(len(depleted)), vals, color="#9E9E9E", edgecolor="black", lw=0.3)
        axes[1].set_yticks(range(len(depleted)))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlabel("log₂(Fold Enrichment)")
        axes[1].set_title(f"Top {n_top} Depleted {k}-mers in UCOE Candidates", fontweight="bold")
        axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / f"top_kmers_k{k}.png")
    fig.savefig(out_dir / f"top_kmers_k{k}.pdf")
    plt.close(fig)


def fig_shared_ucoe_heatmap(results, known_seqs, k, out_dir):
    """Heatmap of shared k-mers across known UCOEs and candidates."""
    from ucoe_pipeline.structural.kmer_analysis import count_kmers

    enrichment = results["enrichment"]
    shared = results["shared_ucoe_kmers"]

    # Get shared k-mers that are also significantly enriched
    sig_shared = [r for r in enrichment
                  if r.get("significant") and r.get("shared_in_known_ucoes") and r["fold_enrichment"] > 1]
    sig_shared.sort(key=lambda x: x["fold_enrichment"], reverse=True)

    if not sig_shared:
        logger.info("No significantly enriched shared k-mers for k=%d", k)
        return

    top_kmers = [r["kmer"] for r in sig_shared[:30]]

    # Compute per-UCOE frequencies
    ucoe_names = list(known_seqs.keys())
    short_names = [n.split("_")[0] for n in ucoe_names]

    matrix = np.zeros((len(top_kmers), len(ucoe_names)))
    for j, (name, seq) in enumerate(known_seqs.items()):
        counts = count_kmers(seq, k, canonical=True)
        total = sum(counts.values())
        for i, kmer in enumerate(top_kmers):
            matrix[i, j] = counts.get(kmer, 0) / total * 1000 if total > 0 else 0

    fig, ax = plt.subplots(figsize=(6, max(4, len(top_kmers) * 0.3)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(ucoe_names)))
    ax.set_xticklabels(short_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(top_kmers)))

    ylabels = []
    for r in sig_shared[:30]:
        lbl = r["kmer"]
        if r.get("tf_annotation"):
            lbl += f" [{r['tf_annotation']}]"
        ylabels.append(lbl)
    ax.set_yticklabels(ylabels, fontsize=7)

    for i in range(len(top_kmers)):
        for j in range(len(ucoe_names)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6,
                   color="white" if matrix[i, j] > np.max(matrix) * 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Frequency (per 1000 k-mers)")
    ax.set_title(f"Enriched {k}-mers Shared Across Known UCOEs", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / f"shared_heatmap_k{k}.png")
    fig.savefig(out_dir / f"shared_heatmap_k{k}.pdf")
    plt.close(fig)


def main():
    print("Loading sequences...")
    cand_seqs, known_seqs, ctrl_seqs = load_sequences()
    print(f"  Candidates: {len(cand_seqs)}, Known: {len(known_seqs)}, Controls: {len(ctrl_seqs)}")

    print("\nRunning k-mer analysis...")
    results = run_kmer_analysis(cand_seqs, known_seqs, ctrl_seqs, k_values=[4, 5, 6])

    # Save results as TSV
    for k, res in results.items():
        df = pd.DataFrame(res["enrichment"])
        df.to_csv(OUT_DIR / f"kmer_enrichment_k{k}.tsv", sep="\t", index=False)

        print(f"\n{'='*70}")
        print(f"K={k} RESULTS")
        print(f"{'='*70}")
        print(f"  Tested: {res['n_tested']}")
        print(f"  Significantly enriched: {res['n_sig_enriched']}")
        print(f"  Significantly depleted: {res['n_sig_depleted']}")
        print(f"  Enriched + shared in known UCOEs: {res['n_sig_shared']}")

        # Top enriched shared k-mers
        sig_shared = [r for r in res["enrichment"]
                      if r.get("significant") and r.get("shared_in_known_ucoes") and r["fold_enrichment"] > 1]
        sig_shared.sort(key=lambda x: x["fold_enrichment"], reverse=True)
        if sig_shared:
            print(f"\n  Top enriched k-mers shared in all known UCOEs:")
            for r in sig_shared[:20]:
                rc = reverse_complement(r["kmer"])
                tf = f"  [{r['tf_annotation']}]" if r["tf_annotation"] else ""
                print(f"    {r['kmer']} (RC: {rc})  FE={r['fold_enrichment']:.2f}  "
                      f"q={r['q_value']:.2e}{tf}")

        # Top enriched overall
        sig_all = [r for r in res["enrichment"] if r.get("significant") and r["fold_enrichment"] > 1]
        sig_all.sort(key=lambda x: x["fold_enrichment"], reverse=True)
        if sig_all:
            print(f"\n  Top 15 enriched k-mers (all):")
            for r in sig_all[:15]:
                shared_mark = " ★" if r.get("shared_in_known_ucoes") else ""
                tf = f"  [{r['tf_annotation']}]" if r["tf_annotation"] else ""
                print(f"    {r['kmer']}{shared_mark}  FE={r['fold_enrichment']:.2f}  "
                      f"q={r['q_value']:.2e}{tf}")

    # Generate figures
    print("\nGenerating figures...")
    for k, res in results.items():
        fig_volcano(res["enrichment"], k, OUT_DIR)
        print(f"  Volcano plot k={k} saved.")

        fig_top_kmers_bar(res["enrichment"], k, OUT_DIR)
        print(f"  Top k-mers bar chart k={k} saved.")

        fig_shared_ucoe_heatmap(res, known_seqs, k, OUT_DIR)
        print(f"  Shared UCOE heatmap k={k} saved.")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
