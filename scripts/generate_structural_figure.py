#!/usr/bin/env python3
"""
Generate corrected structural figure for the FAPESP report.

Shows the five metrics that are both statistically significant AND
directly mentioned in the report text:
  A  ww_fraction           — fraction of WW dinucleotides
  B  ss_fraction           — fraction of SS dinucleotides
  C  dinuc_entropy         — dinucleotide compositional entropy
  D  stiffness_mean        — DNA stiffness index
  E  nuc_enriched_fraction — fraction of sites with high nucleosome occupancy

Saves to: output/paper_figures/fig7_structural.png (overwrites old version)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
STRUCTURAL_DIR = ROOT / "output" / "structural"
OUT_PATH = ROOT / "output" / "paper_figures" / "fig7_structural.png"

COLORS = {
    "UCOE_candidates":    "#3274A1",
    "Known_UCOEs":        "#E1812C",
    "CpG_island_controls":"#78B7B2",
}
GROUP_LABELS = {
    "UCOE_candidates":     "Candidatos UCOE\n(n=599)",
    "Known_UCOEs":         "UCOEs Conhecidos\n(n=3)",
    "CpG_island_controls": "Controles CpG\n(n=200)",
}
METRICS = [
    "ww_fraction",
    "ss_fraction",
    "dinuc_entropy",
    "stiffness_mean",
    "nuc_enriched_fraction",
]
METRIC_LABELS = {
    "ww_fraction":           "Fração WW\n(AA/AT/TA/TT)",
    "ss_fraction":           "Fração SS\n(CC/CG/GC/GG)",
    "dinuc_entropy":         "Entropia Dinucleotídica\n(bits)",
    "stiffness_mean":        "Rigidez do DNA\n(índice de Brukner)",
    "nuc_enriched_fraction": "Fração de Sítios com Alta\nOcupação Nucleossômica",
}
PANEL_LETTERS = list("ABCDE")

Q_VALUES = {
    "ww_fraction":           7.82e-20,
    "ss_fraction":           3.02e-11,
    "dinuc_entropy":         7.82e-20,
    "stiffness_mean":        7.38e-7,
    "nuc_enriched_fraction": 5.87e-17,
}

def q_to_stars(q):
    if q < 0.001:
        return "***"
    elif q < 0.01:
        return "**"
    elif q < 0.05:
        return "*"
    return "ns"


def main():
    # Load data
    all_df = pd.read_csv(STRUCTURAL_DIR / "all_groups_structural.tsv", sep="\t")

    # Load group labels from individual files
    cands = pd.read_csv(STRUCTURAL_DIR / "candidates_structural.tsv", sep="\t")
    cands["group"] = "UCOE_candidates"
    ctrls = pd.read_csv(STRUCTURAL_DIR / "controls_structural.tsv", sep="\t")
    ctrls["group"] = "CpG_island_controls"
    known = pd.read_csv(STRUCTURAL_DIR / "known_ucoes_structural.tsv", sep="\t")
    known["group"] = "Known_UCOEs"

    combined = pd.concat([cands, ctrls, known], ignore_index=True)

    # Verify all metrics exist
    for m in METRICS:
        if m not in combined.columns:
            print(f"WARNING: {m} not in combined dataframe — columns: {list(combined.columns)[:20]}")
            return

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")

    GROUP_ORDER = ["UCOE_candidates", "CpG_island_controls", "Known_UCOEs"]

    for i, metric in enumerate(METRICS):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(2, 3, i + 1)

        plot_data = combined[["group", metric]].dropna()
        data_groups = [plot_data[plot_data["group"] == g][metric].values
                       for g in GROUP_ORDER]

        # Draw boxplots manually for full control
        bp = ax.boxplot(
            data_groups,
            patch_artist=True,
            showfliers=False,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            boxprops=dict(linewidth=1.2),
        )
        colors_list = [COLORS[g] for g in GROUP_ORDER]
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for whisker in bp["whiskers"]:
            whisker.set_color("gray")
        for cap in bp["caps"]:
            cap.set_color("gray")

        # Overlay known UCOEs as diamonds
        known_vals = plot_data[plot_data["group"] == "Known_UCOEs"][metric].values
        ax.scatter(
            [3] * len(known_vals), known_vals,
            marker="D", s=80, color=COLORS["Known_UCOEs"],
            edgecolors="black", linewidths=0.8, zorder=6,
        )

        # Significance annotation: candidates vs controls (positions 1 vs 2)
        q = Q_VALUES[metric]
        stars = q_to_stars(q)
        y_max = max(
            np.nanpercentile(data_groups[0], 95) if len(data_groups[0]) else 0,
            np.nanpercentile(data_groups[1], 95) if len(data_groups[1]) else 0,
        )
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        # draw bracket
        y_top = y_max + y_range * 0.08
        ax.plot([1, 1, 2, 2], [y_top - y_range*0.02, y_top, y_top, y_top - y_range*0.02],
                color="black", linewidth=1.0)
        ax.text(1.5, y_top + y_range * 0.01, stars, ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(1.5, y_top - y_range * 0.04,
                f"q = {q:.2e}", ha="center", va="top", fontsize=7, color="#555555")

        # Panel letter
        ax.text(-0.18, 1.08, PANEL_LETTERS[i], transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

        ax.set_title(METRIC_LABELS[metric], fontsize=9.5, fontweight="bold", pad=4)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER],
                           fontsize=7.5, ha="center")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide 6th subplot (2×3 grid, 5 panels)
    ax_hidden = fig.add_subplot(2, 3, 6)
    ax_hidden.set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["UCOE_candidates"],    label="Candidatos UCOE (n=599)"),
        mpatches.Patch(color=COLORS["CpG_island_controls"],label="Controles CpG (n=200)"),
        mpatches.Patch(color=COLORS["Known_UCOEs"],        label="UCOEs Conhecidos (n=3)"),
    ]
    fig.legend(handles=patches, loc="lower right",
               bbox_to_anchor=(0.98, 0.04), fontsize=9,
               framealpha=0.9, edgecolor="#cccccc")

    fig.suptitle(
        "Propriedades Biofísicas Intrínsecas do DNA: Candidatos UCOE vs. Controles CpG\n"
        "(Teste de Mann-Whitney U; *** q < 0,001; ** q < 0,01; * q < 0,05; FDR Benjamini-Hochberg)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.0, 1, 0.99])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
