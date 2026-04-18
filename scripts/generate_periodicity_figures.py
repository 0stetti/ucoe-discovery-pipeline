#!/usr/bin/env python3
"""
Generate publication-quality figures for the dinucleotide periodicity
and spectrum analysis.

Figures:
    1. Average autocorrelation curves (WW and SS) — candidates vs controls vs known UCOEs
    2. Power spectral density comparison
    3. Dinucleotide frequency spectrum heatmap (16 dinucs × 3 groups)
    4. Boxplots of key periodicity/composition metrics
    5. PCA of dinucleotide frequencies (16-dim → 2D)
    6. Karlin-Burge rho* profile comparison
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucoe_pipeline.structural.analysis import (
    read_fasta, extract_known_ucoe_sequences, sample_random_cpg_islands,
)
from ucoe_pipeline.structural.periodicity import (
    _binary_signal, autocorrelation, power_spectral_density,
    WW_DINUCS, SS_DINUCS, MAX_LAG,
)
from ucoe_pipeline.structural.dinucleotide_spectrum import (
    ALL_DINUCLEOTIDES, dinucleotide_frequencies, rho_star, get_frequency_vector,
)
from ucoe_pipeline.config import KNOWN_UCOES, OUTPUT_DIR

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLOR_CAND = "#2196F3"    # blue
COLOR_CTRL = "#9E9E9E"    # grey
COLOR_KNOWN = "#E53935"   # red
COLOR_A2 = "#D32F2F"
COLOR_TBP = "#FF5722"
COLOR_SRF = "#FF9800"

OUT_DIR = OUTPUT_DIR / "periodicity_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sequences():
    """Load all three groups of sequences."""
    candidates_fasta = OUTPUT_DIR / "ucoe_sequences.fa"
    cand_seqs = read_fasta(candidates_fasta)
    known_seqs = extract_known_ucoe_sequences()

    # Build exclusion list
    exclude_regions = []
    for info in KNOWN_UCOES.values():
        exclude_regions.append((info["chrom"], info["start"], info["end"]))
    for header in cand_seqs:
        if "::" in header:
            coord = header.split("::")[1]
            try:
                chrom, pos = coord.split(":")
                start, end = pos.split("-")
                exclude_regions.append((chrom, int(start), int(end)))
            except Exception:
                pass

    ctrl_seqs = sample_random_cpg_islands(n_samples=200, exclude_regions=exclude_regions)

    return cand_seqs, known_seqs, ctrl_seqs


def compute_group_acf(sequences, dinuc_set, max_lag=MAX_LAG):
    """Compute average autocorrelation for a group of sequences."""
    acfs = []
    for seq in sequences.values():
        seq = seq.upper().replace("N", "")
        if len(seq) < max_lag + 10:
            continue
        sig = _binary_signal(seq, dinuc_set)
        acf = autocorrelation(sig, max_lag)
        if len(acf) == max_lag + 1:
            acfs.append(acf)
    if not acfs:
        return np.zeros(max_lag + 1)
    return np.mean(acfs, axis=0), np.std(acfs, axis=0) / np.sqrt(len(acfs))


def compute_group_psd(sequences, dinuc_set):
    """Compute average PSD for a group, interpolated to common grid."""
    common_periods = np.linspace(3, 50, 500)
    all_psd = []
    for seq in sequences.values():
        seq = seq.upper().replace("N", "")
        if len(seq) < 100:
            continue
        sig = _binary_signal(seq, dinuc_set)
        periods, psd = power_spectral_density(sig)
        if len(periods) > 10:
            psd_interp = np.interp(common_periods, periods[::-1], psd[::-1])
            all_psd.append(psd_interp)
    if not all_psd:
        return common_periods, np.zeros_like(common_periods)
    return common_periods, np.mean(all_psd, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Autocorrelation curves
# ══════════════════════════════════════════════════════════════════════════════
def fig1_autocorrelation(cand_seqs, known_seqs, ctrl_seqs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    lags = np.arange(MAX_LAG + 1)

    for ax, dinuc_set, label in [
        (axes[0], WW_DINUCS, "WW (AA, AT, TA, TT)"),
        (axes[1], SS_DINUCS, "SS (CC, CG, GC, GG)"),
    ]:
        cand_mean, cand_se = compute_group_acf(cand_seqs, dinuc_set)
        ctrl_mean, ctrl_se = compute_group_acf(ctrl_seqs, dinuc_set)

        ax.fill_between(lags, cand_mean - 1.96*cand_se, cand_mean + 1.96*cand_se,
                        alpha=0.15, color=COLOR_CAND)
        ax.fill_between(lags, ctrl_mean - 1.96*ctrl_se, ctrl_mean + 1.96*ctrl_se,
                        alpha=0.15, color=COLOR_CTRL)

        ax.plot(lags, cand_mean, color=COLOR_CAND, lw=1.5, label="UCOE candidates (n=599)")
        ax.plot(lags, ctrl_mean, color=COLOR_CTRL, lw=1.5, label="CpG island controls (n=200)")

        # Individual known UCOEs
        colors_known = [COLOR_A2, COLOR_TBP, COLOR_SRF]
        for (name, seq), c in zip(known_seqs.items(), colors_known):
            seq_clean = seq.upper().replace("N", "")
            sig = _binary_signal(seq_clean, dinuc_set)
            acf = autocorrelation(sig, MAX_LAG)
            short_name = name.split("_")[0] if "_" in name else name
            ax.plot(lags[:len(acf)], acf, color=c, lw=1.0, ls="--",
                    alpha=0.7, label=short_name)

        # Mark helical repeat
        for mult in [10, 20, 30, 40, 50]:
            ax.axvline(mult, color="gray", ls=":", lw=0.5, alpha=0.4)

        ax.set_xlabel("Lag (bp)")
        ax.set_title(f"{label} Dinucleotide Autocorrelation")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, MAX_LAG)

    axes[0].set_ylabel("Normalized autocorrelation R(k)")
    fig.suptitle("Dinucleotide Periodicity: Autocorrelation at Helical Repeat (~10.5 bp)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_autocorrelation.png")
    fig.savefig(OUT_DIR / "fig1_autocorrelation.pdf")
    plt.close(fig)
    print("  Figure 1: Autocorrelation curves saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Power Spectral Density
# ══════════════════════════════════════════════════════════════════════════════
def fig2_psd(cand_seqs, known_seqs, ctrl_seqs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, dinuc_set, label in [
        (axes[0], WW_DINUCS, "WW Dinucleotides"),
        (axes[1], SS_DINUCS, "SS Dinucleotides"),
    ]:
        periods_c, psd_c = compute_group_psd(cand_seqs, dinuc_set)
        periods_k, psd_k = compute_group_psd(ctrl_seqs, dinuc_set)

        ax.plot(periods_c, psd_c, color=COLOR_CAND, lw=1.5, label="UCOE candidates")
        ax.plot(periods_k, psd_k, color=COLOR_CTRL, lw=1.5, label="CpG controls")

        # Highlight nucleosomal period band
        ax.axvspan(9, 12, alpha=0.1, color="gold", label="Nucleosomal band (9–12 bp)")
        ax.axvline(10.4, color="gold", ls="--", lw=1, alpha=0.6)

        ax.set_xlabel("Period (bp)")
        ax.set_title(f"Power Spectral Density — {label}")
        ax.set_xlim(3, 30)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Power Spectral Density")
    fig.suptitle("Fourier Analysis of Dinucleotide Periodicity",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_psd.png")
    fig.savefig(OUT_DIR / "fig2_psd.pdf")
    plt.close(fig)
    print("  Figure 2: Power spectral density saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Dinucleotide frequency spectrum heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig3_spectrum_heatmap(cand_seqs, known_seqs, ctrl_seqs):
    groups = {
        "UCOE candidates": cand_seqs,
        "Known UCOEs": known_seqs,
        "CpG controls": ctrl_seqs,
    }
    freq_matrix = []
    group_labels = []

    for gname, seqs in groups.items():
        group_freqs = []
        for seq in seqs.values():
            freqs = dinucleotide_frequencies(seq)
            group_freqs.append([freqs[d] for d in ALL_DINUCLEOTIDES])
        mean_freqs = np.mean(group_freqs, axis=0)
        freq_matrix.append(mean_freqs)
        group_labels.append(gname)

    freq_matrix = np.array(freq_matrix)

    # Compute difference (candidates - controls) for annotation
    diff = freq_matrix[0] - freq_matrix[2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 2]})

    # Heatmap
    im = axes[0].imshow(freq_matrix, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(16))
    axes[0].set_xticklabels(ALL_DINUCLEOTIDES, fontsize=8, fontweight="bold")
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(group_labels, fontsize=10)

    # Annotate cells with frequencies
    for i in range(3):
        for j in range(16):
            axes[0].text(j, i, f"{freq_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if freq_matrix[i, j] > 0.08 else "black")

    plt.colorbar(im, ax=axes[0], shrink=0.6, label="Frequency")
    axes[0].set_title("Dinucleotide Frequency Spectrum by Group", fontweight="bold")

    # Difference bar plot
    colors = [COLOR_CAND if d > 0 else COLOR_CTRL for d in diff]
    bars = axes[1].bar(range(16), diff * 1000, color=colors, edgecolor="black", lw=0.3)
    axes[1].set_xticks(range(16))
    axes[1].set_xticklabels(ALL_DINUCLEOTIDES, fontsize=8, fontweight="bold")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("Δ Frequency (×10³)\nCandidate − Control")
    axes[1].set_title("Dinucleotide Enrichment in UCOE Candidates vs. Controls", fontweight="bold")

    # Highlight WW and SS
    for i, d in enumerate(ALL_DINUCLEOTIDES):
        if d in WW_DINUCS:
            axes[1].get_xticklabels()[i].set_color("#1565C0")
            axes[1].get_xticklabels()[i].set_fontweight("bold")
        elif d in SS_DINUCS:
            axes[1].get_xticklabels()[i].set_color("#C62828")
            axes[1].get_xticklabels()[i].set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_spectrum_heatmap.png")
    fig.savefig(OUT_DIR / "fig3_spectrum_heatmap.pdf")
    plt.close(fig)
    print("  Figure 3: Dinucleotide spectrum heatmap saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Boxplots of key metrics
# ══════════════════════════════════════════════════════════════════════════════
def fig4_boxplots(cand_seqs, known_seqs, ctrl_seqs):
    from ucoe_pipeline.structural.periodicity import periodicity_metrics
    from ucoe_pipeline.structural.dinucleotide_spectrum import spectrum_metrics

    # Compute metrics for all sequences
    groups_data = {}
    for gname, seqs, color in [
        ("UCOE candidates", cand_seqs, COLOR_CAND),
        ("CpG controls", ctrl_seqs, COLOR_CTRL),
    ]:
        records = []
        for seq in seqs.values():
            pm = periodicity_metrics(seq)
            sm = spectrum_metrics(seq)
            records.append({**pm, **sm})
        groups_data[gname] = (pd.DataFrame(records), color)

    # Known UCOE individual points
    known_records = []
    known_names = []
    for name, seq in known_seqs.items():
        pm = periodicity_metrics(seq)
        sm = spectrum_metrics(seq)
        known_records.append({**pm, **sm})
        known_names.append(name.split("_")[0])
    known_df = pd.DataFrame(known_records)

    metrics_to_plot = [
        ("ww_fraction", "WW Fraction", "Higher = more A/T dinucleotides"),
        ("ss_fraction", "SS Fraction", "Higher = more G/C dinucleotides"),
        ("ww_autocorr_10bp", "WW Autocorr. at 10 bp", "Higher = stronger helical periodicity"),
        ("dinuc_entropy", "Dinucleotide Entropy", "Higher = more diverse composition"),
        ("markov_deviation", "Markov Deviation", "Higher = stronger neighbor dependencies"),
        ("rho_cg", "CpG ρ* (Karlin-Burge)", "Higher = less CpG suppression"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (metric, title, subtitle) in zip(axes, metrics_to_plot):
        data_list = []
        labels = []
        colors = []
        for gname, (df, color) in groups_data.items():
            vals = df[metric].dropna()
            data_list.append(vals.values)
            labels.append(gname)
            colors.append(color)

        bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True,
                       widths=0.6, showfliers=False,
                       medianprops=dict(color="black", lw=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Overlay known UCOEs as individual points
        for i, (name, val) in enumerate(zip(known_names, known_df[metric])):
            if not np.isnan(val):
                ax.scatter(1.5, val, color=[COLOR_A2, COLOR_TBP, COLOR_SRF][i],
                          s=60, zorder=5, marker="D", edgecolor="black", lw=0.5)
                ax.annotate(name, (1.5, val), fontsize=6, ha="left",
                           xytext=(5, 0), textcoords="offset points")

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel(subtitle, fontsize=7, style="italic")

    fig.suptitle("Dinucleotide Composition & Periodicity: UCOE Candidates vs. CpG Controls",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_boxplots.png")
    fig.savefig(OUT_DIR / "fig4_boxplots.pdf")
    plt.close(fig)
    print("  Figure 4: Key metric boxplots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: PCA of dinucleotide frequencies
# ══════════════════════════════════════════════════════════════════════════════
def fig5_pca(cand_seqs, known_seqs, ctrl_seqs):
    # Build frequency vectors
    all_vectors = []
    all_labels = []
    all_groups = []

    for name, seq in cand_seqs.items():
        vec = get_frequency_vector(seq)
        if not np.any(np.isnan(vec)):
            all_vectors.append(vec)
            all_labels.append("candidate")
            all_groups.append(0)

    for name, seq in ctrl_seqs.items():
        vec = get_frequency_vector(seq)
        if not np.any(np.isnan(vec)):
            all_vectors.append(vec)
            all_labels.append("control")
            all_groups.append(1)

    known_vectors = []
    known_labels_list = []
    for name, seq in known_seqs.items():
        vec = get_frequency_vector(seq)
        known_vectors.append(vec)
        known_labels_list.append(name.split("_")[0])

    X = np.array(all_vectors + known_vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    n_main = len(all_vectors)
    X_cand = X_pca[:len(cand_seqs)]
    X_ctrl = X_pca[len(cand_seqs):n_main]
    X_known = X_pca[n_main:]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_ctrl[:, 0], X_ctrl[:, 1], c=COLOR_CTRL, s=15, alpha=0.4,
              label=f"CpG controls (n={len(X_ctrl)})", zorder=2)
    ax.scatter(X_cand[:, 0], X_cand[:, 1], c=COLOR_CAND, s=15, alpha=0.3,
              label=f"UCOE candidates (n={len(X_cand)})", zorder=3)

    # Known UCOEs
    colors_k = [COLOR_A2, COLOR_TBP, COLOR_SRF]
    for i, (xk, lbl) in enumerate(zip(X_known, known_labels_list)):
        ax.scatter(xk[0], xk[1], c=colors_k[i], s=120, marker="*",
                  edgecolor="black", lw=0.8, zorder=5, label=lbl)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA of Dinucleotide Frequency Spectrum (16-dimensional)",
                fontweight="bold")
    ax.legend(loc="best", fontsize=8)

    # Loadings annotation
    loadings = pca.components_
    # Top 4 contributing dinucleotides for PC1
    top_pc1 = np.argsort(np.abs(loadings[0]))[-4:]
    for idx in top_pc1:
        ax.annotate(ALL_DINUCLEOTIDES[idx],
                   xy=(0, 0), xytext=(loadings[0, idx]*15, loadings[1, idx]*15),
                   fontsize=7, color="darkred", alpha=0.6,
                   arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.3))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_pca_dinucleotide.png")
    fig.savefig(OUT_DIR / "fig5_pca_dinucleotide.pdf")
    plt.close(fig)
    print(f"  Figure 5: PCA plot saved. PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Karlin-Burge rho* profile
# ══════════════════════════════════════════════════════════════════════════════
def fig6_rho_profile(cand_seqs, known_seqs, ctrl_seqs):
    groups = {
        "UCOE candidates": (cand_seqs, COLOR_CAND),
        "Known UCOEs": (known_seqs, COLOR_KNOWN),
        "CpG controls": (ctrl_seqs, COLOR_CTRL),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(16)
    width = 0.25

    for i, (gname, (seqs, color)) in enumerate(groups.items()):
        all_rho = []
        for seq in seqs.values():
            rho = rho_star(seq)
            all_rho.append([rho[d] for d in ALL_DINUCLEOTIDES])
        mean_rho = np.nanmean(all_rho, axis=0)
        se_rho = np.nanstd(all_rho, axis=0) / np.sqrt(len(all_rho))

        ax.bar(x + i * width, mean_rho, width, label=gname,
               color=color, alpha=0.7, edgecolor="black", lw=0.3,
               yerr=se_rho, capsize=2, error_kw={"lw": 0.5})

    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ALL_DINUCLEOTIDES, fontsize=9, fontweight="bold")
    ax.set_ylabel("ρ* (observed / expected)")
    ax.set_title("Karlin-Burge Genomic Signature (ρ*) — Dinucleotide Over/Under-representation",
                fontweight="bold")
    ax.legend(fontsize=9)

    # Color WW/SS labels
    for i, d in enumerate(ALL_DINUCLEOTIDES):
        if d in WW_DINUCS:
            ax.get_xticklabels()[i].set_color("#1565C0")
        elif d in SS_DINUCS:
            ax.get_xticklabels()[i].set_color("#C62828")

    # Annotate CG
    cg_idx = ALL_DINUCLEOTIDES.index("CG")
    ax.annotate("CpG\nsuppression", xy=(cg_idx + width, 0.85),
               fontsize=7, ha="center", style="italic", color="darkred")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_rho_profile.png")
    fig.savefig(OUT_DIR / "fig6_rho_profile.pdf")
    plt.close(fig)
    print("  Figure 6: Karlin-Burge rho* profile saved.")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading sequences...")
    cand_seqs, known_seqs, ctrl_seqs = load_sequences()
    print(f"  Candidates: {len(cand_seqs)}, Known UCOEs: {len(known_seqs)}, Controls: {len(ctrl_seqs)}")

    print("\nGenerating figures...")
    fig1_autocorrelation(cand_seqs, known_seqs, ctrl_seqs)
    fig2_psd(cand_seqs, known_seqs, ctrl_seqs)
    fig3_spectrum_heatmap(cand_seqs, known_seqs, ctrl_seqs)
    fig4_boxplots(cand_seqs, known_seqs, ctrl_seqs)
    fig5_pca(cand_seqs, known_seqs, ctrl_seqs)
    fig6_rho_profile(cand_seqs, known_seqs, ctrl_seqs)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
