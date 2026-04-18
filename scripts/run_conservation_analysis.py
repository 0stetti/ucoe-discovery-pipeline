#!/usr/bin/env python3
"""
Run evolutionary conservation analysis on UCOE candidate sequences.

Compares PhyloP and PhastCons scores (100 vertebrates) between
UCOE candidates, known UCOEs, and CpG island controls.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ucoe_pipeline.structural.analysis import (
    read_fasta, extract_known_ucoe_sequences, sample_random_cpg_islands,
)
from ucoe_pipeline.structural.conservation import (
    run_conservation_analysis, parse_coordinates,
    open_bigwig, positional_conservation_profile, PHYLOP_URL,
)
from ucoe_pipeline.config import KNOWN_UCOES, OUTPUT_DIR

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = OUTPUT_DIR / "conservation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sequences():
    """Load candidate, known UCOE, and control sequences."""
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


def statistical_comparison(results: dict) -> pd.DataFrame:
    """Compare conservation metrics between candidates and controls."""
    metrics = [
        "phylop_mean", "phylop_median",
        "phylop_positive_frac", "phylop_gt1_frac", "phylop_gt2_frac",
        "phastcons_mean", "phastcons_gt05_frac", "phastcons_gt09_frac",
    ]

    cand_df = pd.DataFrame(results["candidates"])
    ctrl_df = pd.DataFrame(results["controls"])

    rows = []
    for m in metrics:
        cand_vals = cand_df[m].dropna()
        ctrl_vals = ctrl_df[m].dropna()
        if len(cand_vals) < 5 or len(ctrl_vals) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(cand_vals, ctrl_vals, alternative="two-sided")
        n1, n2 = len(cand_vals), len(ctrl_vals)
        effect_r = (u_stat - (n1 * n2 / 2)) / (n1 * n2 / 2)

        rows.append({
            "metric": m,
            "cand_median": cand_vals.median(),
            "cand_mean": cand_vals.mean(),
            "ctrl_median": ctrl_vals.median(),
            "ctrl_mean": ctrl_vals.mean(),
            "U_statistic": u_stat,
            "p_value": p_val,
            "effect_size_r": effect_r,
        })

        # Known UCOEs
        known_vals = pd.DataFrame(results["known"])[m].dropna() if results["known"] else pd.Series()
        if len(known_vals) > 0:
            rows[-1]["known_median"] = known_vals.median()
            rows[-1]["known_mean"] = known_vals.mean()

    df = pd.DataFrame(rows)
    if len(df) > 0:
        _, q_values, _, _ = multipletests(df["p_value"], method="fdr_bh", alpha=0.05)
        df["q_value"] = q_values
        df["significant"] = q_values < 0.05

    return df


def fig_conservation_boxplots(results: dict, stat_df: pd.DataFrame, out_dir: Path):
    """Boxplots of key conservation metrics."""
    cand_df = pd.DataFrame(results["candidates"])
    ctrl_df = pd.DataFrame(results["controls"])
    known_df = pd.DataFrame(results["known"]) if results["known"] else pd.DataFrame()

    metrics = [
        ("phylop_mean", "PhyloP médio", "PhyloP (100 vertebrados)"),
        ("phastcons_mean", "PhastCons médio", "Prob. de elemento conservado"),
        ("phylop_gt1_frac", "Fração PhyloP > 1", "Fração de bases"),
        ("phastcons_gt05_frac", "Fração PhastCons > 0.5", "Fração de bases"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        cand_vals = cand_df[metric].dropna()
        ctrl_vals = ctrl_df[metric].dropna()

        bp = ax.boxplot(
            [cand_vals, ctrl_vals],
            labels=["Candidatas\nUCOE", "Controles\nCpG"],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#9E9E9E")
        bp["boxes"][1].set_alpha(0.5)

        # Overlay known UCOEs
        if len(known_df) > 0 and metric in known_df.columns:
            known_vals = known_df[metric].dropna()
            for val in known_vals:
                ax.plot(1, val, "r*", markersize=12, zorder=5)
            if len(known_vals) > 0:
                names = known_df["name"].tolist()
                for j, (name, val) in enumerate(zip(names, known_vals)):
                    short = name.split("_")[0]
                    ax.annotate(short, (1.15, val), fontsize=7, color="red")

        # Significance annotation
        row = stat_df[stat_df["metric"] == metric]
        if len(row) > 0:
            q = row.iloc[0]["q_value"]
            if q < 0.001:
                sig_text = f"q = {q:.2e} ***"
            elif q < 0.01:
                sig_text = f"q = {q:.2e} **"
            elif q < 0.05:
                sig_text = f"q = {q:.3f} *"
            else:
                sig_text = f"q = {q:.3f} ns"
            ax.set_title(f"{title}\n{sig_text}", fontweight="bold", fontsize=10)
        else:
            ax.set_title(title, fontweight="bold")

        ax.set_ylabel(ylabel)

    fig.suptitle(
        "Conservação Evolutiva: Candidatas UCOE vs. Controles CpG\n"
        "(PhyloP e PhastCons, 100 vertebrados)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "conservation_boxplots.png")
    fig.savefig(out_dir / "conservation_boxplots.pdf")
    plt.close(fig)


def fig_conservation_profile(results: dict, out_dir: Path):
    """Positional conservation profile across regions."""
    logger.info("Computing positional conservation profiles...")
    phylop_bw = open_bigwig(PHYLOP_URL)

    n_bins = 50

    cand_coords = []
    for r in results["candidates"]:
        cand_coords.append((r["chrom"], r["start"], r["end"]))
    ctrl_coords = []
    for r in results["controls"]:
        ctrl_coords.append((r["chrom"], r["start"], r["end"]))

    cand_profile = positional_conservation_profile(cand_coords, phylop_bw, n_bins)
    ctrl_profile = positional_conservation_profile(ctrl_coords, phylop_bw, n_bins)

    phylop_bw.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 100, n_bins)

    ax.plot(x, cand_profile, color="#2196F3", linewidth=2, label="Candidatas UCOE")
    ax.plot(x, ctrl_profile, color="#9E9E9E", linewidth=2, label="Controles CpG")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    ax.set_xlabel("Posição relativa na região (%)")
    ax.set_ylabel("PhyloP médio (100 vertebrados)")
    ax.set_title(
        "Perfil Posicional de Conservação Evolutiva",
        fontweight="bold",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "conservation_profile.png")
    fig.savefig(out_dir / "conservation_profile.pdf")
    plt.close(fig)


def fig_motif_conservation(results: dict, out_dir: Path):
    """Compare conservation at ETS motif positions vs. non-motif."""
    cand_df = pd.DataFrame(results["candidates"])
    ctrl_df = pd.DataFrame(results["controls"])

    # Filter to regions with at least one motif
    cand_with = cand_df[cand_df["motif_count"] > 0].copy()
    ctrl_with = ctrl_df[ctrl_df["motif_count"] > 0].copy()

    if len(cand_with) < 5:
        logger.warning("Too few candidates with ETS motifs for motif conservation plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: motif vs non-motif conservation in candidates
    ax = axes[0]
    at_motif = cand_with["phylop_at_motif"].dropna()
    outside = cand_with["phylop_outside_motif"].dropna()

    bp = ax.boxplot(
        [at_motif, outside],
        labels=["No motivo\nETS", "Fora do\nmotivo ETS"],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    bp["boxes"][0].set_facecolor("#E53935")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#2196F3")
    bp["boxes"][1].set_alpha(0.4)

    if len(at_motif) > 5 and len(outside) > 5:
        _, p = stats.mannwhitneyu(at_motif, outside, alternative="two-sided")
        ax.set_title(
            f"Conservação em Candidatas UCOE\n(p = {p:.2e})",
            fontweight="bold",
        )
    else:
        ax.set_title("Conservação em Candidatas UCOE", fontweight="bold")
    ax.set_ylabel("PhyloP médio")

    # Panel B: motif counts distribution
    ax = axes[1]
    cand_counts = cand_df["motif_count"].dropna()
    ctrl_counts = ctrl_df["motif_count"].dropna()

    max_count = int(max(cand_counts.max(), ctrl_counts.max())) + 1
    bins = np.arange(-0.5, max_count + 0.5, 1)
    ax.hist(cand_counts, bins=bins, alpha=0.6, color="#2196F3",
            label=f"Candidatas (média={cand_counts.mean():.1f})", density=True)
    ax.hist(ctrl_counts, bins=bins, alpha=0.6, color="#9E9E9E",
            label=f"Controles (média={ctrl_counts.mean():.1f})", density=True)
    ax.set_xlabel("Número de motivos ETS (CGGAA[GA]) por região")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de Motivos ETS", fontweight="bold")
    ax.legend()

    fig.suptitle(
        "Conservação Evolutiva nos Motivos ETS",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "motif_conservation.png")
    fig.savefig(out_dir / "motif_conservation.pdf")
    plt.close(fig)


def fig_scatter_conservation_vs_score(results: dict, out_dir: Path):
    """Scatter plot: composite UCOE score vs. conservation."""
    import re

    cand_df = pd.DataFrame(results["candidates"])

    # Extract score from name
    scores = []
    for name in cand_df["name"]:
        m = re.search(r"score([\d.]+)", str(name))
        if m:
            scores.append(float(m.group(1)))
        else:
            scores.append(np.nan)
    cand_df["ucoe_score"] = scores

    valid = cand_df.dropna(subset=["ucoe_score", "phylop_mean"])
    if len(valid) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        valid["ucoe_score"], valid["phylop_mean"],
        c=valid["phastcons_mean"], cmap="YlOrRd",
        s=15, alpha=0.6, edgecolor="none",
    )
    plt.colorbar(sc, ax=ax, label="PhastCons médio")

    # Correlation
    r, p = stats.spearmanr(valid["ucoe_score"], valid["phylop_mean"])
    ax.set_xlabel("Score UCOE composto")
    ax.set_ylabel("PhyloP médio (100 vertebrados)")
    ax.set_title(
        f"Conservação vs. Score UCOE\n(Spearman ρ = {r:.3f}, p = {p:.2e})",
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_dir / "conservation_vs_score.png")
    fig.savefig(out_dir / "conservation_vs_score.pdf")
    plt.close(fig)


def main():
    print("Loading sequences...")
    cand_seqs, known_seqs, ctrl_seqs = load_sequences()
    print(f"  Candidates: {len(cand_seqs)}, Known: {len(known_seqs)}, Controls: {len(ctrl_seqs)}")

    print("\nRunning conservation analysis (querying UCSC BigWig files remotely)...")
    print("  This may take 10-15 minutes for ~800 regions...")
    results = run_conservation_analysis(cand_seqs, known_seqs, ctrl_seqs, KNOWN_UCOES)

    # Save raw results
    for group in ["candidates", "known", "controls"]:
        if results[group]:
            df = pd.DataFrame(results[group])
            df.to_csv(OUT_DIR / f"conservation_{group}.tsv", sep="\t", index=False)
            print(f"  Saved {len(df)} {group} records")

    # Statistical comparison
    print("\nStatistical comparison (Mann-Whitney U, BH-FDR):")
    stat_df = statistical_comparison(results)
    stat_df.to_csv(OUT_DIR / "conservation_statistics.tsv", sep="\t", index=False)

    print(f"\n{'='*70}")
    print("CONSERVATION RESULTS")
    print(f"{'='*70}")

    cand_df = pd.DataFrame(results["candidates"])
    ctrl_df = pd.DataFrame(results["controls"])
    known_df = pd.DataFrame(results["known"]) if results["known"] else pd.DataFrame()

    for _, row in stat_df.iterrows():
        sig = "***" if row["q_value"] < 0.001 else "**" if row["q_value"] < 0.01 else "*" if row["q_value"] < 0.05 else "ns"
        print(f"  {row['metric']:25s}  cand={row['cand_median']:.4f}  ctrl={row['ctrl_median']:.4f}  "
              f"q={row['q_value']:.2e} {sig}")

    if len(known_df) > 0:
        print(f"\n  Known UCOEs:")
        for _, row in known_df.iterrows():
            print(f"    {row['name']:30s}  phyloP={row['phylop_mean']:.4f}  "
                  f"phastCons={row['phastcons_mean']:.4f}")

    # ETS motif conservation
    cand_with_motif = cand_df[cand_df["motif_count"] > 0]
    if len(cand_with_motif) > 0:
        at_motif = cand_with_motif["phylop_at_motif"].dropna()
        outside = cand_with_motif["phylop_outside_motif"].dropna()
        print(f"\n  ETS motif conservation (candidates with ≥1 motif: {len(cand_with_motif)}):")
        print(f"    At ETS motif:      PhyloP = {at_motif.mean():.4f} ± {at_motif.std():.4f}")
        print(f"    Outside ETS motif: PhyloP = {outside.mean():.4f} ± {outside.std():.4f}")
        if len(at_motif) > 5 and len(outside) > 5:
            _, p = stats.mannwhitneyu(at_motif, outside, alternative="two-sided")
            print(f"    Mann-Whitney p = {p:.2e}")

    # Generate figures
    print("\nGenerating figures...")
    fig_conservation_boxplots(results, stat_df, OUT_DIR)
    print("  Conservation boxplots saved.")

    fig_conservation_profile(results, OUT_DIR)
    print("  Positional profile saved.")

    fig_motif_conservation(results, OUT_DIR)
    print("  Motif conservation figure saved.")

    fig_scatter_conservation_vs_score(results, OUT_DIR)
    print("  Conservation vs. score scatter saved.")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
