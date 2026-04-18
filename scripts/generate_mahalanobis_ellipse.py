#!/usr/bin/env python3
"""
Generate a Mahalanobis distance visualization with confidence ellipses
for UCOE candidates projected onto PCA space (2D).

Shows:
- All 599 candidates as scatter points colored by Mahalanobis distance
- Known UCOEs as labeled stars
- Confidence ellipses at 1σ, 2σ, and 3σ from the UCOE centroid
- Top-N candidates highlighted
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ucoe_pipeline.config import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FEATURES = [
    "H3K4me3_mean", "H3K4me3_cv",
    "H3K27ac_mean", "H3K27ac_cv",
    "H3K27me3_mean", "H3K27me3_cv",
    "H3K9me3_mean", "H3K9me3_cv",
    "meth_mean", "meth_cv",
    "DNase_mean", "DNase_cv",
    "H3K9ac_mean", "H3K9ac_cv",
    "H3K36me3_mean", "H3K36me3_cv",
    "repliseq_mean", "CTCF_n_peaks",
]

KNOWN_UCOES = {
    "A2UCOE": ("CBX3", "HNRNPA2B1"),
    "TBP/PSMB1": ("PSMB1", "TBP"),
    "SRF-UCOE": ("SURF2", "SURF1"),
}


def identify_known_ucoes(df):
    known_idx = {}
    for name, (g1, g2) in KNOWN_UCOES.items():
        mask = (
            ((df["gene1"] == g1) & (df["gene2"] == g2)) |
            ((df["gene1"] == g2) & (df["gene2"] == g1))
        )
        matches = df[mask]
        if len(matches) >= 1:
            known_idx[name] = matches.index[0]
    return known_idx


def draw_confidence_ellipse(ax, mean, cov, n_std, **kwargs):
    """Draw a confidence ellipse based on covariance matrix in 2D."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # chi2 critical value for 2 degrees of freedom
    chi2_val = chi2.ppf(0.6827 if n_std == 1 else
                        0.9545 if n_std == 2 else
                        0.9973, df=2)
    width = 2 * np.sqrt(eigenvalues[0] * chi2_val)
    height = 2 * np.sqrt(eigenvalues[1] * chi2_val)

    ellipse = patches.Ellipse(
        xy=mean, width=width, height=height, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def main():
    # Load data
    scored = pd.read_csv(OUTPUT_DIR / "phase2" / "scored_candidates.tsv", sep="\t")
    known_idx = identify_known_ucoes(scored)

    if len(known_idx) < 3:
        print(f"Warning: found only {len(known_idx)} known UCOEs")

    # Prepare features
    X = scored[FEATURES].copy()
    for col in FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Standardize
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # PCA to 2D
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    # Centroid of known UCOEs in PCA space
    ref_indices = list(known_idx.values())
    centroid_pca = Z_pca[ref_indices].mean(axis=0)

    # Covariance of ALL candidates in PCA space (for ellipse shape)
    # Use the candidate distribution covariance centered on the UCOE centroid
    cov_pca = np.cov(Z_pca.T)

    # Mahalanobis distance in full feature space (already computed)
    mahal_dist = scored["mahalanobis_dist"].values

    # Compute Mahalanobis-like distance in PCA space from centroid
    diff = Z_pca - centroid_pca
    cov_inv = np.linalg.inv(cov_pca)
    mahal_pca = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by Mahalanobis distance (original, from pipeline)
    norm = Normalize(vmin=0, vmax=np.percentile(mahal_dist, 95))
    colors = cm.viridis_r(norm(mahal_dist))

    # Scatter all candidates
    scatter = ax.scatter(
        Z_pca[:, 0], Z_pca[:, 1],
        c=mahal_dist, cmap="viridis_r", norm=norm,
        s=15, alpha=0.6, edgecolors="none", zorder=2,
    )

    # Confidence ellipses (1σ, 2σ, 3σ)
    ellipse_styles = [
        (1, "#E74C3C", 2.0, "1σ (68.3%)"),
        (2, "#2ECC71", 1.5, "2σ (95.5%)"),
        (3, "#3498DB", 1.0, "3σ (99.7%)"),
    ]
    for n_std, color, lw, label in ellipse_styles:
        draw_confidence_ellipse(
            ax, centroid_pca, cov_pca, n_std,
            facecolor="none", edgecolor=color,
            linestyle="--", linewidth=lw, label=label, zorder=3,
        )

    # Plot centroid
    ax.scatter(
        centroid_pca[0], centroid_pca[1],
        marker="D", s=120, c="#8E44AD", edgecolors="white",
        linewidth=1.5, zorder=6, label="Centroide UCOE",
    )

    # Plot known UCOEs as stars
    ucoe_colors = {"A2UCOE": "#E74C3C", "TBP/PSMB1": "#2ECC71", "SRF-UCOE": "#3498DB"}
    for name, idx in known_idx.items():
        ax.scatter(
            Z_pca[idx, 0], Z_pca[idx, 1],
            marker="*", s=300, c=ucoe_colors[name],
            edgecolors="black", linewidth=0.8, zorder=7,
        )
        ax.annotate(
            name,
            (Z_pca[idx, 0], Z_pca[idx, 1]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            zorder=8,
        )

    # Highlight top 10 candidates (excluding known UCOEs)
    top_n = 10
    top_mask = np.ones(len(scored), dtype=bool)
    for idx in ref_indices:
        top_mask[idx] = False
    top_candidates = scored[top_mask].nsmallest(top_n, "composite_rank")

    for _, row in top_candidates.iterrows():
        i = row.name
        label_text = f"{row['gene1']}/{row['gene2']}"
        ax.scatter(
            Z_pca[i, 0], Z_pca[i, 1],
            marker="o", s=60, facecolors="none",
            edgecolors="#E67E22", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            label_text,
            (Z_pca[i, 0], Z_pca[i, 1]),
            xytext=(8, -8), textcoords="offset points",
            fontsize=7, color="#E67E22", fontstyle="italic",
            zorder=5,
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Distancia de Mahalanobis", fontsize=11)

    # Labels
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variancia)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variancia)", fontsize=12)
    ax.set_title(
        "Candidatos UCOE: Distancia de Mahalanobis no espaco epigenomico",
        fontsize=13, fontweight="bold",
    )

    # Legend
    ax.legend(
        loc="upper left", fontsize=9, framealpha=0.9,
        title="Elipses de confianca", title_fontsize=10,
    )

    # Count candidates inside each ellipse
    n_1s = np.sum(mahal_pca <= chi2.ppf(0.6827, 2) ** 0.5)
    n_2s = np.sum(mahal_pca <= chi2.ppf(0.9545, 2) ** 0.5)
    n_3s = np.sum(mahal_pca <= chi2.ppf(0.9973, 2) ** 0.5)

    stats_text = (
        f"Dentro de 1σ: {n_1s} candidatos\n"
        f"Dentro de 2σ: {n_2s} candidatos\n"
        f"Dentro de 3σ: {n_3s} candidatos\n"
        f"Total: {len(scored)} candidatos"
    )
    ax.text(
        0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=8, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", alpha=0.9),
    )

    # Save
    for ext in ("png", "pdf"):
        outpath = OUT_DIR / f"fig_mahalanobis_ellipse.{ext}"
        fig.savefig(outpath)
        print(f"Saved: {outpath}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
