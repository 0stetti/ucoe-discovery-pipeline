#!/usr/bin/env python3
"""
Generate all publication-quality figures for the Bioinformatics article.
All text in English. Figures saved to output/paper_figures/.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
FIG_DIR = OUTPUT / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ==========================================================================
# FIGURE 1 — Pipeline Overview Schematic
# ==========================================================================
def fig1_pipeline_overview():
    """Schematic diagram of the two-phase pipeline."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    c_input = "#E8D5B7"
    c_phase1 = "#4A90D9"
    c_phase2 = "#E67E22"
    c_struct = "#27AE60"
    c_output = "#8E44AD"
    c_arrow = "#2C3E50"

    def add_box(x, y, w, h, text, color, fontsize=8, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#2C3E50", linewidth=1.2, alpha=0.9)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True,
                color="white" if color in [c_phase1, c_phase2, c_struct, c_output] else "#2C3E50")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.5))

    # Input data
    add_box(0.2, 7.0, 2.2, 0.7, "ENCODE\n(11 cell lines)", c_input, 7)
    add_box(2.7, 7.0, 2.2, 0.7, "GENCODE v44\nGene annotations", c_input, 7)
    add_box(5.2, 7.0, 2.2, 0.7, "HPA v23\n8,528 HKGs", c_input, 7)
    add_box(7.7, 7.0, 2.0, 0.7, "UCSC\nhg38 / CpG islands", c_input, 7)

    # Phase I header
    add_box(0.5, 5.8, 4.0, 0.65, "PHASE I — Rule-based Filtering", c_phase1, 9, bold=True)

    # Phase I filters
    filters = [
        "F1: Divergent HKG pairs (<=5 kb)  789",
        "F2: CpG island overlap (>=40%)  692",
        "F3: Active marks (H3K4me3, H3K27ac)  647",
        "F4: No repressive marks  645",
        "F5: Hypomethylation (<10%)  599",
    ]
    for i, f in enumerate(filters):
        y = 5.2 - i * 0.42
        ax.text(0.7, y, f, fontsize=6.5, va="center", color="#2C3E50",
                fontfamily="monospace")

    # Arrow from inputs to Phase I
    arrow(2.5, 7.0, 2.5, 6.5)
    arrow(5.0, 7.0, 2.5, 6.5)

    # Phase II header
    add_box(5.5, 5.8, 4.0, 0.65, "PHASE II — Similarity Ranking", c_phase2, 9, bold=True)

    # Phase II components
    p2_items = [
        "21-feature epigenomic vector",
        "Reference: 3 known UCOEs (centroid)",
        "Mahalanobis distance (40%)",
        "Cosine similarity (30%)",
        "Percentile rank (30%)",
        "Sensitivity: 29 weight combinations",
    ]
    for i, item in enumerate(p2_items):
        y = 5.2 - i * 0.38
        ax.text(5.7, y, item, fontsize=6.5, va="center", color="#2C3E50")

    # Arrow Phase I -> Phase II
    arrow(4.5, 4.3, 5.5, 4.3)
    ax.text(5.0, 4.45, "599", fontsize=8, ha="center", fontweight="bold", color=c_phase1)

    # Structural analysis
    add_box(0.5, 2.0, 4.0, 0.65, "DNA Structural Analysis", c_struct, 9, bold=True)
    s_items = [
        "Brukner flexibility index",
        "Dinucleotide stiffness",
        "Nucleosome occupancy prediction",
        "vs. 200 random CpG island controls",
    ]
    for i, item in enumerate(s_items):
        y = 1.55 - i * 0.35
        ax.text(0.7, y, item, fontsize=6.5, va="center", color="#2C3E50")

    arrow(2.5, 3.2, 2.5, 2.7)

    # Output
    add_box(5.5, 1.5, 4.0, 1.2, "OUTPUT\n599 ranked candidates\n13 with >80% stability\n6 with 100% stability\n+ structural characterization", c_output, 7.5, bold=False)

    arrow(7.5, 3.2, 7.5, 2.75)

    # Title
    ax.text(5.0, 7.95, "Computational Pipeline for De Novo UCOE Discovery",
            ha="center", fontsize=12, fontweight="bold", color="#2C3E50")

    fig.savefig(FIG_DIR / "fig1_pipeline_overview.png")
    fig.savefig(FIG_DIR / "fig1_pipeline_overview.pdf")
    plt.close(fig)
    logger.info("Fig 1: Pipeline overview saved")


# ==========================================================================
# FIGURE 2 — Filtering Funnel (English)
# ==========================================================================
def fig2_filter_funnel():
    """Phase I filtering funnel in English."""
    labels = [
        "Divergent HKG pairs (inter-TSS <= 5 kb)",
        "CpG island overlap >= 40%",
        "Ubiquitous active marks\n(H3K4me3, H3K27ac in >= 80% cell lines)",
        "Absence of repressive marks\n(H3K27me3, H3K9me3 in >= 80% cell lines)",
        "Constitutive hypomethylation\n(mean methylation < 10%)",
    ]
    counts = [789, 692, 647, 645, 599]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.6 * i / (n - 1)) for i in range(n)]

    bars = ax.barh(range(n), counts, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Candidate Regions", fontsize=10)

    max_count = max(counts)
    for i, (count, bar) in enumerate(zip(counts, bars)):
        pct = ""
        if i > 0 and counts[i - 1] > 0:
            pct = f" ({count/counts[i-1]*100:.0f}%)"
        ax.text(count + max_count * 0.01, i, f"{count:,}{pct}",
                va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(0, max_count * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mark known UCOEs
    ax.text(max_count * 0.6, n - 0.3,
            "All 3 known UCOEs pass all filters",
            fontsize=7, fontstyle="italic", color="#27AE60", ha="center")

    fig.savefig(FIG_DIR / "fig2_filter_funnel.png")
    fig.savefig(FIG_DIR / "fig2_filter_funnel.pdf")
    plt.close(fig)
    logger.info("Fig 2: Filter funnel saved")


# ==========================================================================
# FIGURE 3 — Known UCOE Reference Profiles (Heatmap)
# ==========================================================================
def fig3_reference_heatmap():
    """Heatmap of epigenomic features for the 3 known UCOEs."""
    ref = pd.read_csv(OUTPUT / "phase2" / "reference_profile.tsv", sep="\t")

    features = [
        "H3K4me3_mean", "H3K27ac_mean", "H3K9ac_mean", "H3K36me3_mean",
        "H3K27me3_mean", "H3K9me3_mean",
        "H3K4me3_cv", "H3K27ac_cv", "H3K9ac_cv",
        "meth_mean", "meth_cv",
        "DNase_mean", "repliseq_mean", "CTCF_n_peaks",
    ]
    feature_labels = [
        "H3K4me3 (FC)", "H3K27ac (FC)", "H3K9ac (FC)", "H3K36me3 (FC)",
        "H3K27me3 (FC)", "H3K9me3 (FC)",
        "H3K4me3 (CV)", "H3K27ac (CV)", "H3K9ac (CV)",
        "Methylation (%)", "Methylation (CV)",
        "DNase (FC)", "Repli-seq (E/L)", "CTCF peaks",
    ]

    available = [f for f in features if f in ref.columns]
    labels_avail = [feature_labels[i] for i, f in enumerate(features) if f in ref.columns]

    data = ref.set_index("ucoe_name")[available].T
    data.index = labels_avail
    data.columns = ["A2UCOE", "TBP/PSMB1", "SRF-UCOE"]

    # Z-score normalize rows for visualization
    data_z = data.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 5),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: raw values heatmap
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax1,
                linewidths=0.5, cbar_kws={"shrink": 0.6, "label": "Value"})
    ax1.set_title("(A) Raw Feature Values", fontsize=10, fontweight="bold")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=7)

    # Right: z-score heatmap
    sns.heatmap(data_z, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax2,
                linewidths=0.5, cbar_kws={"shrink": 0.6, "label": "Z-score"})
    ax2.set_title("(B) Z-score Normalized", fontsize=10, fontweight="bold")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=7)
    ax2.set_ylabel("")

    fig.suptitle("Epigenomic Profiles of Known Human UCOEs", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_reference_heatmap.png")
    fig.savefig(FIG_DIR / "fig3_reference_heatmap.pdf")
    plt.close(fig)
    logger.info("Fig 3: Reference heatmap saved")


# ==========================================================================
# FIGURE 4 — Score Distribution + Known UCOEs
# ==========================================================================
def fig4_score_distribution():
    """Score distribution with known UCOEs highlighted."""
    scored = pd.read_csv(OUTPUT / "phase2" / "scored_candidates.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.hist(scored["composite_score"], bins=40, color="#4A90D9", edgecolor="white",
            alpha=0.8, label="599 candidates")

    # Known UCOEs — find them
    known_genes = {"HNRNPA2B1": "A2UCOE", "TBP": "TBP/PSMB1", "SURF1": "SRF-UCOE"}
    for gene, name in known_genes.items():
        match = scored[scored["gene1"].str.upper() == gene.upper()]
        if match.empty:
            match = scored[scored["gene2"].str.upper() == gene.upper()]
        if not match.empty:
            score = match.iloc[0]["composite_score"]
            rank = int(match.iloc[0]["composite_rank"])
            ax.axvline(score, color="#E74C3C", linewidth=2, linestyle="--", alpha=0.8)
            ax.text(score + 0.005, ax.get_ylim()[1] * 0.85,
                    f"{name}\n(#{rank})", fontsize=7, color="#E74C3C",
                    fontweight="bold", va="top")

    ax.set_xlabel("Composite UCOE Score", fontsize=10)
    ax.set_ylabel("Number of Candidates", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", fontsize=8)

    fig.savefig(FIG_DIR / "fig4_score_distribution.png")
    fig.savefig(FIG_DIR / "fig4_score_distribution.pdf")
    plt.close(fig)
    logger.info("Fig 4: Score distribution saved")


# ==========================================================================
# FIGURE 5 — Top 20 Candidates + Selected Radar Plots
# ==========================================================================
def fig5_top_candidates():
    """Top 20 candidates bar chart."""
    scored = pd.read_csv(OUTPUT / "phase2" / "scored_candidates.tsv", sep="\t")
    top20 = scored.nsmallest(20, "composite_rank")

    # Also get known UCOEs in the ranking for reference
    known_genes = ["HNRNPA2B1", "TBP", "SURF1"]
    known_rows = []
    for g in known_genes:
        m = scored[(scored["gene1"].str.upper() == g) | (scored["gene2"].str.upper() == g)]
        if not m.empty:
            known_rows.append(m.iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = [f"{row['gene1']}/{row['gene2']}" for _, row in top20.iterrows()]
    scores = top20["composite_score"].values
    ranks = top20["composite_rank"].values.astype(int)

    # Load sensitivity data to color by stability
    sens = pd.read_csv(OUTPUT / "phase2" / "sensitivity_analysis.tsv", sep="\t")
    stability_map = {}
    for _, srow in sens.iterrows():
        stability_map[srow["region"]] = srow["stability_pct"]

    colors = []
    for _, row in top20.iterrows():
        region = f"{row['chrom']}:{row['start']}-{row['end']}"
        stab = stability_map.get(region, 0)
        if stab >= 100:
            colors.append("#27AE60")
        elif stab >= 80:
            colors.append("#F39C12")
        else:
            colors.append("#3498DB")

    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Composite UCOE Score", fontsize=10)

    for i, (score, rank) in enumerate(zip(scores, ranks)):
        ax.text(score + 0.003, i, f"{score:.3f}", va="center", fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#27AE60", label="100% stability"),
        mpatches.Patch(color="#F39C12", label="80-99% stability"),
        mpatches.Patch(color="#3498DB", label="<80% stability"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "fig5_top_candidates.png")
    fig.savefig(FIG_DIR / "fig5_top_candidates.pdf")
    plt.close(fig)
    logger.info("Fig 5: Top candidates saved")


# ==========================================================================
# FIGURE 6 — Sensitivity Analysis
# ==========================================================================
def fig6_sensitivity():
    """Sensitivity analysis — stability bar chart for top candidates."""
    sens = pd.read_csv(OUTPUT / "phase2" / "sensitivity_analysis.tsv", sep="\t")
    scored = pd.read_csv(OUTPUT / "phase2" / "scored_candidates.tsv", sep="\t")

    # Map regions to gene names
    region_to_genes = {}
    for _, row in scored.iterrows():
        region = f"{row['chrom']}:{row['start']}-{row['end']}"
        region_to_genes[region] = f"{row['gene1']}/{row['gene2']}"

    # Top 25 by stability
    top_sens = sens.nlargest(25, "stability_pct")
    labels = [region_to_genes.get(r, r) for r in top_sens["region"]]
    stability = top_sens["stability_pct"].values

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#27AE60" if s >= 100 else "#F39C12" if s >= 80 else "#E74C3C" for s in stability]

    bars = ax.barh(range(len(labels)), stability, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Stability (% of 29 weight combinations in top 20)", fontsize=9)
    ax.axvline(80, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(81, len(labels) - 0.5, "80% threshold", fontsize=7, color="#E74C3C", va="bottom")

    for i, s in enumerate(stability):
        ax.text(s + 0.5, i, f"{s:.0f}%", va="center", fontsize=7)

    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        mpatches.Patch(color="#27AE60", label="100% stable (6 candidates)"),
        mpatches.Patch(color="#F39C12", label="80-99% stable (7 candidates)"),
        mpatches.Patch(color="#E74C3C", label="<80% stable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.savefig(FIG_DIR / "fig6_sensitivity.png")
    fig.savefig(FIG_DIR / "fig6_sensitivity.pdf")
    plt.close(fig)
    logger.info("Fig 6: Sensitivity analysis saved")


# ==========================================================================
# FIGURE 7 — Structural Analysis (Panel: boxplots + scatter)
# ==========================================================================
def fig7_structural():
    """Structural analysis panel: boxplots + stiffness vs nucleosome scatter."""
    combined = pd.read_csv(OUTPUT / "structural" / "all_groups_structural.tsv", sep="\t")

    COLORS = {
        "UCOE_candidates": "#3274A1",
        "Known_UCOEs": "#E1812C",
        "CpG_island_controls": "#78B7B2",
    }
    GROUP_ORDER = ["UCOE_candidates", "Known_UCOEs", "CpG_island_controls"]
    GROUP_LABELS = {
        "UCOE_candidates": "UCOE Candidates",
        "Known_UCOEs": "Known UCOEs",
        "CpG_island_controls": "Random CpG Islands",
    }

    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    metrics = [
        ("stiffness_mean", "DNA Stiffness Index"),
        ("nuc_score_mean", "Nucleosome Formation\nScore"),
        ("nuc_enriched_fraction", "Nucleosome-Enriched\nFraction"),
        ("gc_content", "GC Content"),
        ("cpg_obs_exp", "CpG Obs/Exp Ratio"),
    ]

    for idx, (metric, label) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        data = combined[["group", metric]].dropna()
        groups_present = [g for g in GROUP_ORDER if g in data["group"].values]
        palette = [COLORS[g] for g in groups_present]

        parts = ax.boxplot(
            [data[data["group"] == g][metric].values for g in groups_present],
            labels=[GROUP_LABELS[g].replace(" ", "\n") for g in groups_present],
            patch_artist=True, showfliers=False, widths=0.5,
        )
        for patch, color in zip(parts["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay known UCOEs as diamonds
        known_data = data[data["group"] == "Known_UCOEs"][metric]
        if not known_data.empty:
            known_idx = groups_present.index("Known_UCOEs") + 1
            ax.scatter([known_idx] * len(known_data), known_data,
                       color=COLORS["Known_UCOEs"], marker="D", s=60,
                       edgecolors="black", linewidths=0.8, zorder=5)

        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(axis="x", labelsize=6.5)
        letter = chr(65 + idx)  # A, B, C, ...
        ax.set_title(f"({letter})", fontsize=9, fontweight="bold", loc="left")

    # Panel F: scatter stiffness vs nucleosome
    ax = fig.add_subplot(gs[1, 2])
    for group in ["CpG_island_controls", "UCOE_candidates", "Known_UCOEs"]:
        subset = combined[combined["group"] == group]
        if subset.empty:
            continue
        marker = "D" if group == "Known_UCOEs" else "o"
        size = 80 if group == "Known_UCOEs" else 12
        alpha = 1.0 if group == "Known_UCOEs" else 0.4
        zorder = 10 if group == "Known_UCOEs" else 2
        edgecolor = "black" if group == "Known_UCOEs" else "none"

        ax.scatter(
            subset["stiffness_mean"], subset["nuc_score_mean"],
            c=COLORS[group], label=GROUP_LABELS[group],
            marker=marker, s=size, alpha=alpha, zorder=zorder,
            edgecolors=edgecolor, linewidths=0.8,
        )

        if group == "Known_UCOEs":
            for _, row in subset.iterrows():
                h = row["header"]
                if "A2UCOE" in h:
                    name = "A2UCOE"
                elif "TBP" in h:
                    name = "TBP/PSMB1"
                else:
                    name = "SRF-UCOE"
                ax.annotate(name, (row["stiffness_mean"], row["nuc_score_mean"]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6, fontweight="bold")

    ax.set_xlabel("Stiffness", fontsize=8)
    ax.set_ylabel("Nucleosome Score", fontsize=8)
    ax.set_title("(F)", fontsize=9, fontweight="bold", loc="left")
    ax.legend(fontsize=5.5, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.savefig(FIG_DIR / "fig7_structural.png")
    fig.savefig(FIG_DIR / "fig7_structural.pdf")
    plt.close(fig)
    logger.info("Fig 7: Structural analysis saved")


# ==========================================================================
# FIGURE 8 — Chromosome Distribution
# ==========================================================================
def fig8_chromosome_distribution():
    """Distribution of 599 candidates across chromosomes."""
    scored = pd.read_csv(OUTPUT / "phase2" / "scored_candidates.tsv", sep="\t")

    # Count per chromosome
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    counts = scored["chrom"].value_counts()
    chrom_counts = [counts.get(c, 0) for c in chrom_order]
    chrom_labels = [c.replace("chr", "") for c in chrom_order]

    # Known UCOE chromosomes
    known_chroms = {"chr7": "A2UCOE", "chr6": "TBP/PSMB1", "chr9": "SRF-UCOE"}

    fig, ax = plt.subplots(figsize=(7, 3))

    colors = ["#E74C3C" if chrom_order[i] in known_chroms else "#4A90D9"
              for i in range(len(chrom_order))]

    bars = ax.bar(range(len(chrom_labels)), chrom_counts, color=colors,
                  edgecolor="white", width=0.7)
    ax.set_xticks(range(len(chrom_labels)))
    ax.set_xticklabels(chrom_labels, fontsize=7)
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("Number of UCOE Candidates", fontsize=10)

    # Annotate known UCOE chromosomes
    for chrom, name in known_chroms.items():
        if chrom in chrom_order:
            idx = chrom_order.index(chrom)
            ax.text(idx, chrom_counts[idx] + 1, name, ha="center",
                    fontsize=6, color="#E74C3C", fontweight="bold", rotation=45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Contains known UCOE"),
        mpatches.Patch(color="#4A90D9", label="Novel candidates only"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.savefig(FIG_DIR / "fig8_chromosome_distribution.png")
    fig.savefig(FIG_DIR / "fig8_chromosome_distribution.pdf")
    plt.close(fig)
    logger.info("Fig 8: Chromosome distribution saved")


# ==========================================================================
# SUPPLEMENTARY — Metric Comparison (Mahalanobis vs Cosine)
# ==========================================================================
def sfig_metric_comparison():
    """Scatter: Mahalanobis vs Cosine scores colored by percentile."""
    scored = pd.read_csv(OUTPUT / "phase2" / "scored_candidates.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        scored["mahalanobis_score"], scored["cosine_score"],
        c=scored["percentile_score"], cmap="viridis", s=15, alpha=0.6,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Percentile Score", fontsize=9)

    # Mark known UCOEs
    known_genes = {"HNRNPA2B1": "A2UCOE", "TBP": "TBP/PSMB1", "SURF1": "SRF-UCOE"}
    for gene, name in known_genes.items():
        match = scored[(scored["gene1"].str.upper() == gene) | (scored["gene2"].str.upper() == gene)]
        if not match.empty:
            row = match.iloc[0]
            ax.scatter(row["mahalanobis_score"], row["cosine_score"],
                       c="red", marker="*", s=200, zorder=10, edgecolors="black", linewidths=0.8)
            ax.annotate(name, (row["mahalanobis_score"], row["cosine_score"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8, fontweight="bold", color="red")

    ax.set_xlabel("Mahalanobis Score", fontsize=10)
    ax.set_ylabel("Cosine Similarity Score", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIG_DIR / "sfig_metric_comparison.png")
    fig.savefig(FIG_DIR / "sfig_metric_comparison.pdf")
    plt.close(fig)
    logger.info("SFig: Metric comparison saved")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    logger.info("Generating publication figures...")
    fig1_pipeline_overview()
    fig2_filter_funnel()
    fig3_reference_heatmap()
    fig4_score_distribution()
    fig5_top_candidates()
    fig6_sensitivity()
    fig7_structural()
    fig8_chromosome_distribution()
    sfig_metric_comparison()
    logger.info(f"\nAll figures saved to {FIG_DIR}")
