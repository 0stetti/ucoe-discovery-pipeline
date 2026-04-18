#!/usr/bin/env python3
"""
Re-rank UCOE candidates using two alternative strategies and compare
against the original Mahalanobis+cosine ranking.

Strategy 1: Z-score normalization + Euclidean distance to centroid
Strategy 2: Z-score normalization + Nearest Neighbor distance

Both strategies avoid estimating a covariance matrix from only 3
reference points, which is the main limitation of the original
Mahalanobis-based approach.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ucoe_pipeline.config import OUTPUT_DIR

OUT_DIR = OUTPUT_DIR / "reranking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Epigenomic features used for ranking (same as Phase II)
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

# Known UCOEs identified by gene pairs
KNOWN_UCOES = {
    "A2UCOE": ("CBX3", "HNRNPA2B1"),
    "TBP/PSMB1": ("PSMB1", "TBP"),
    "SRF-UCOE": ("SURF2", "SURF1"),
}


def identify_known_ucoes(df):
    """Find rows corresponding to known UCOEs."""
    known_idx = {}
    for name, (g1, g2) in KNOWN_UCOES.items():
        mask = (
            ((df["gene1"] == g1) & (df["gene2"] == g2)) |
            ((df["gene1"] == g2) & (df["gene2"] == g1))
        )
        matches = df[mask]
        if len(matches) == 1:
            known_idx[name] = matches.index[0]
        elif len(matches) > 1:
            known_idx[name] = matches.index[0]
    return known_idx


def compute_rankings(df, features, known_idx):
    """Compute both ranking strategies."""
    X = df[features].copy()

    # Impute NaN with median (7 NaN in repliseq_mean)
    for col in features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Z-score normalization from full candidate distribution
    scaler = StandardScaler()
    Z = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=features,
    )

    # Reference centroid in z-score space
    ref_indices = list(known_idx.values())
    Z_ref = Z.loc[ref_indices]
    centroid = Z_ref.mean(axis=0).values

    # --- Strategy 1: Euclidean distance to centroid ---
    dist_centroid = np.sqrt(((Z.values - centroid) ** 2).sum(axis=1))
    df["dist_centroid"] = dist_centroid
    df["rank_centroid"] = df["dist_centroid"].rank(method="min").astype(int)

    # --- Strategy 2: Nearest neighbor distance ---
    ref_vectors = Z_ref.values  # shape (3, n_features)
    dist_nn = np.full(len(Z), np.inf)
    nearest_ucoe = [""] * len(Z)
    ucoe_names = list(known_idx.keys())

    for i, (name, idx) in enumerate(known_idx.items()):
        d = np.sqrt(((Z.values - ref_vectors[i]) ** 2).sum(axis=1))
        for j in range(len(Z)):
            if d[j] < dist_nn[j]:
                dist_nn[j] = d[j]
                nearest_ucoe[j] = name

    df["dist_nn"] = dist_nn
    df["nearest_ucoe"] = nearest_ucoe
    df["rank_nn"] = df["dist_nn"].rank(method="min").astype(int)

    # --- Combined ranking ---
    df["combined_rank_avg"] = (df["rank_centroid"] + df["rank_nn"]) / 2
    df["rank_combined"] = df["combined_rank_avg"].rank(method="min").astype(int)

    # --- PCA-based ranking (robustness check) ---
    pca = PCA(n_components=0.95)  # retain 95% variance
    Z_pca = pca.fit_transform(Z.values)
    n_comp = Z_pca.shape[1]
    centroid_pca = Z_pca[ref_indices].mean(axis=0)
    dist_centroid_pca = np.sqrt(((Z_pca - centroid_pca) ** 2).sum(axis=1))
    df["dist_centroid_pca"] = dist_centroid_pca
    df["rank_centroid_pca"] = df["dist_centroid_pca"].rank(method="min").astype(int)

    return df, Z, n_comp, pca


def compare_rankings(df, known_idx):
    """Compare original and new rankings."""
    print("=" * 70)
    print("RANKING COMPARISON")
    print("=" * 70)

    # Spearman correlations
    print("\n1. RANK CORRELATIONS (Spearman rho):")
    pairs = [
        ("Original (composite_rank)", "composite_rank", "Centroid", "rank_centroid"),
        ("Original (composite_rank)", "composite_rank", "Nearest Neighbor", "rank_nn"),
        ("Original (composite_rank)", "composite_rank", "Combined", "rank_combined"),
        ("Original (composite_rank)", "composite_rank", "PCA Centroid", "rank_centroid_pca"),
        ("Centroid", "rank_centroid", "Nearest Neighbor", "rank_nn"),
        ("Centroid", "rank_centroid", "PCA Centroid", "rank_centroid_pca"),
    ]
    for name1, col1, name2, col2 in pairs:
        rho, p = stats.spearmanr(df[col1], df[col2])
        print(f"  {name1} vs {name2}: rho={rho:.4f}, p={p:.2e}")

    # Top-N overlaps
    print("\n2. TOP-N OVERLAPS:")
    rank_cols = {
        "Original": "composite_rank",
        "Centroid": "rank_centroid",
        "NN": "rank_nn",
        "Combined": "rank_combined",
    }
    for n in [10, 25, 50]:
        print(f"\n  Top {n}:")
        top_sets = {}
        for name, col in rank_cols.items():
            top_sets[name] = set(df[df[col] <= n].index)
        for i, (n1, s1) in enumerate(top_sets.items()):
            for n2, s2 in list(top_sets.items())[i+1:]:
                overlap = len(s1 & s2)
                print(f"    {n1} ∩ {n2}: {overlap}/{n}")

    # High-confidence set (top 50 in ALL rankings)
    print("\n3. HIGH-CONFIDENCE CANDIDATES (top 50 in ALL rankings):")
    all_top50 = set(df.index)
    for col in ["composite_rank", "rank_centroid", "rank_nn"]:
        all_top50 &= set(df[df[col] <= 50].index)

    hc = df.loc[sorted(all_top50)].sort_values("rank_combined")
    for _, r in hc.iterrows():
        print(f"  {r['gene1']}/{r['gene2']}  orig=#{int(r['composite_rank'])}  "
              f"centroid=#{int(r['rank_centroid'])}  NN=#{int(r['rank_nn'])}  "
              f"combined=#{int(r['rank_combined'])}  nearest={r['nearest_ucoe']}")
    print(f"  Total: {len(hc)} candidates")

    # Known UCOEs positions
    print("\n4. KNOWN UCOE POSITIONS ACROSS RANKINGS:")
    for name, idx in known_idx.items():
        r = df.loc[idx]
        print(f"  {name}: orig=#{int(r['composite_rank'])}  centroid=#{int(r['rank_centroid'])}  "
              f"NN=#{int(r['rank_nn'])}  combined=#{int(r['rank_combined'])}  "
              f"PCA=#{int(r['rank_centroid_pca'])}")

    # Dramatic rank changes (>100 positions)
    print("\n5. DRAMATIC RANK CHANGES (|delta| > 100, original vs combined):")
    df["rank_delta"] = df["rank_combined"] - df["composite_rank"]
    big_changes = df[df["rank_delta"].abs() > 100].sort_values("rank_delta")
    print(f"  Total candidates with |delta| > 100: {len(big_changes)}")
    print(f"\n  IMPROVED (moved up >100 positions):")
    improved = big_changes[big_changes["rank_delta"] < 0].head(15)
    for _, r in improved.iterrows():
        print(f"    {r['gene1']}/{r['gene2']}  orig=#{int(r['composite_rank'])} -> "
              f"combined=#{int(r['rank_combined'])}  (delta={int(r['rank_delta'])})")
    print(f"\n  DROPPED (moved down >100 positions):")
    dropped = big_changes[big_changes["rank_delta"] > 0].head(15)
    for _, r in dropped.iterrows():
        print(f"    {r['gene1']}/{r['gene2']}  orig=#{int(r['composite_rank'])} -> "
              f"combined=#{int(r['rank_combined'])}  (delta={int(r['rank_delta'])})")

    return hc


def fig_scatter_rankings(df, known_idx, out_dir):
    """Scatter plot: original rank vs combined new rank."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # All candidates
    ax.scatter(
        df["composite_rank"], df["rank_combined"],
        c="#BBBBBB", s=8, alpha=0.4, label="Candidates",
    )

    # Highlight top 50 in both
    both_top50 = df[(df["composite_rank"] <= 50) & (df["rank_combined"] <= 50)]
    ax.scatter(
        both_top50["composite_rank"], both_top50["rank_combined"],
        c="#2196F3", s=20, alpha=0.7, label=f"Top 50 in both ({len(both_top50)})",
    )

    # Known UCOEs
    colors = {"A2UCOE": "#E53935", "TBP/PSMB1": "#4CAF50", "SRF-UCOE": "#FF9800"}
    for name, idx in known_idx.items():
        r = df.loc[idx]
        ax.scatter(
            r["composite_rank"], r["rank_combined"],
            c=colors[name], s=150, marker="*", edgecolor="black",
            linewidth=0.5, zorder=5, label=name,
        )
        ax.annotate(
            name, (r["composite_rank"], r["rank_combined"]),
            fontsize=8, fontweight="bold",
            xytext=(8, 8), textcoords="offset points",
        )

    # Diagonal
    ax.plot([1, 599], [1, 599], "k--", lw=0.5, alpha=0.3)

    ax.set_xlabel("Original Ranking (Mahalanobis + Cosine)")
    ax.set_ylabel("New Combined Ranking (Centroid + NN)")
    ax.set_title("Comparison of UCOE Candidate Rankings", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 610)
    ax.set_ylim(0, 610)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_dir / "ranking_comparison_scatter.png")
    fig.savefig(out_dir / "ranking_comparison_scatter.pdf")
    plt.close(fig)


def fig_rank_distributions(df, known_idx, out_dir):
    """Distribution of rank changes and feature heatmap of top candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: rank change histogram
    ax = axes[0]
    delta = df["rank_combined"] - df["composite_rank"]
    ax.hist(delta, bins=50, color="#2196F3", alpha=0.7, edgecolor="black", lw=0.3)
    ax.axvline(0, color="red", lw=1, ls="--")
    ax.set_xlabel("Rank change (Combined - Original)")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Distribution of Rank Changes", fontweight="bold")

    # Panel B: top 20 combined - feature heatmap
    ax = axes[1]
    top20 = df.nsmallest(20, "rank_combined")
    features_display = [
        "H3K4me3_mean", "H3K27ac_mean", "H3K9ac_mean",
        "H3K27me3_mean", "H3K9me3_mean",
        "meth_mean", "DNase_mean", "repliseq_mean",
    ]
    Z_display = top20[features_display].copy()
    for col in features_display:
        Z_display[col] = (Z_display[col] - df[col].mean()) / df[col].std()

    labels = [f"#{int(r['rank_combined'])} {r['gene1']}/{r['gene2']}"
              for _, r in top20.iterrows()]

    im = ax.imshow(Z_display.values, cmap="RdBu_r", aspect="auto",
                   vmin=-2, vmax=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(features_display)))
    ax.set_xticklabels([f.replace("_mean", "").replace("_", "\n")
                        for f in features_display],
                       fontsize=7, rotation=45, ha="right")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Z-score")
    ax.set_title("Top 20 Candidates (Combined Ranking)\nEpigenomic Profile",
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "ranking_distributions.png")
    fig.savefig(out_dir / "ranking_distributions.pdf")
    plt.close(fig)


def fig_pca_validation(df, Z, known_idx, pca_model, out_dir):
    """PCA scatter with rankings overlay."""
    pca2 = PCA(n_components=2)
    Z_2d = pca2.fit_transform(Z.values)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Color by combined rank
    sc = ax.scatter(
        Z_2d[:, 0], Z_2d[:, 1],
        c=df["rank_combined"], cmap="viridis_r",
        s=12, alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label="Combined Rank (lower = better)")

    # Known UCOEs
    colors = {"A2UCOE": "#E53935", "TBP/PSMB1": "#4CAF50", "SRF-UCOE": "#FF9800"}
    for name, idx in known_idx.items():
        i = df.index.get_loc(idx) if isinstance(idx, int) else idx
        ax.scatter(
            Z_2d[i, 0], Z_2d[i, 1],
            c=colors[name], s=200, marker="*",
            edgecolor="black", lw=0.8, zorder=5,
        )
        ax.annotate(
            name, (Z_2d[i, 0], Z_2d[i, 1]),
            fontsize=8, fontweight="bold",
            xytext=(8, 8), textcoords="offset points",
        )

    var1 = pca2.explained_variance_ratio_[0] * 100
    var2 = pca2.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("PCA of Epigenomic Features\nColored by Combined Rank",
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "pca_ranking_overlay.png")
    fig.savefig(out_dir / "pca_ranking_overlay.pdf")
    plt.close(fig)


def main():
    print("Loading scored candidates...")
    df = pd.read_csv(OUTPUT_DIR / "phase2" / "scored_candidates.tsv", sep="\t")
    print(f"  {len(df)} candidates, {len(FEATURES)} features")

    # Identify known UCOEs
    known_idx = identify_known_ucoes(df)
    print(f"  Found {len(known_idx)} known UCOEs:")
    for name, idx in known_idx.items():
        r = df.loc[idx]
        print(f"    {name}: {r['gene1']}/{r['gene2']} (original rank #{int(r['composite_rank'])})")

    # Compute new rankings
    print("\nComputing new rankings...")
    df, Z, n_pca_comp, pca_model = compute_rankings(df, FEATURES, known_idx)
    print(f"  PCA retained {n_pca_comp} components (95% variance)")

    # Compare
    compare_rankings(df, known_idx)

    # Save full re-ranked table
    output_cols = [
        "chrom", "start", "end", "gene1", "gene2",
        "composite_rank", "composite_score",
        "rank_centroid", "dist_centroid",
        "rank_nn", "dist_nn", "nearest_ucoe",
        "rank_combined", "combined_rank_avg",
        "rank_centroid_pca", "dist_centroid_pca",
    ]
    df[output_cols].to_csv(OUT_DIR / "reranked_candidates.tsv", sep="\t", index=False)
    print(f"\nFull re-ranked table saved to {OUT_DIR / 'reranked_candidates.tsv'}")

    # Figures
    print("\nGenerating figures...")
    fig_scatter_rankings(df, known_idx, OUT_DIR)
    print("  Scatter plot saved.")
    fig_rank_distributions(df, known_idx, OUT_DIR)
    print("  Rank distributions saved.")
    fig_pca_validation(df, Z, known_idx, pca_model, OUT_DIR)
    print("  PCA overlay saved.")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
