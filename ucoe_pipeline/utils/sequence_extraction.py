"""
Extract FASTA sequences for UCOE candidate regions from the hg38 reference genome.

Uses pysam (samtools faidx wrapper) for indexed FASTA access, which is faster
and more memory-efficient than loading the entire genome into memory.
"""

import logging
import subprocess
from pathlib import Path

import pandas as pd

from ucoe_pipeline.config import GENOME_FASTA, OUTPUT_DIR

logger = logging.getLogger(__name__)

GENOME_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
)


def ensure_genome_fasta(genome_path: Path = GENOME_FASTA) -> Path:
    """Download and index hg38 reference genome if not present."""
    if genome_path.exists() and genome_path.with_suffix(".fa.fai").exists():
        logger.info("Genome FASTA already present: %s", genome_path)
        return genome_path

    gz_path = genome_path.with_suffix(".fa.gz")

    if not genome_path.exists():
        if not gz_path.exists():
            logger.info("Downloading hg38 reference genome (~3 GB)...")
            subprocess.run(
                ["curl", "-L", "-o", str(gz_path), GENOME_URL],
                check=True,
            )
            logger.info("Download complete: %s", gz_path)

        logger.info("Decompressing genome...")
        subprocess.run(["gunzip", "-k", str(gz_path)], check=True)
        logger.info("Decompression complete: %s", genome_path)

    # Index with samtools
    fai_path = genome_path.with_suffix(".fa.fai")
    if not fai_path.exists():
        logger.info("Indexing genome with samtools faidx...")
        subprocess.run(["samtools", "faidx", str(genome_path)], check=True)
        logger.info("Index created: %s", fai_path)

    return genome_path


def extract_sequences_bedtools(
    bed_path: Path,
    genome_path: Path = GENOME_FASTA,
    output_fasta: Path | None = None,
) -> Path:
    """Extract FASTA sequences for regions in a BED file using bedtools getfasta.

    Parameters
    ----------
    bed_path : Path to BED file with candidate regions
    genome_path : Path to indexed hg38.fa
    output_fasta : Output FASTA path (default: output/ucoe_sequences.fa)

    Returns
    -------
    Path to the output FASTA file
    """
    if output_fasta is None:
        output_fasta = OUTPUT_DIR / "ucoe_sequences.fa"

    ensure_genome_fasta(genome_path)

    logger.info(
        "Extracting sequences: %s -> %s", bed_path, output_fasta
    )

    subprocess.run(
        [
            "bedtools", "getfasta",
            "-fi", str(genome_path),
            "-bed", str(bed_path),
            "-fo", str(output_fasta),
            "-name",
        ],
        check=True,
    )

    # Count sequences
    n_seqs = sum(1 for line in open(output_fasta) if line.startswith(">"))
    logger.info("Extracted %d sequences to %s", n_seqs, output_fasta)

    return output_fasta


def extract_sequences_from_scored(
    scored_path: Path | None = None,
    genome_path: Path = GENOME_FASTA,
    output_fasta: Path | None = None,
    top_n: int | None = None,
) -> Path:
    """Extract FASTA sequences for scored candidates.

    Creates a properly formatted BED file from the scored candidates TSV,
    then extracts sequences. Includes gene names and rank in FASTA headers.

    Parameters
    ----------
    scored_path : Path to scored_candidates.tsv (default: output/phase2/scored_candidates.tsv)
    genome_path : Path to indexed hg38.fa
    output_fasta : Output FASTA path
    top_n : If set, extract only the top N candidates
    """
    if scored_path is None:
        scored_path = OUTPUT_DIR / "phase2" / "scored_candidates.tsv"
    if output_fasta is None:
        suffix = f"_top{top_n}" if top_n else ""
        output_fasta = OUTPUT_DIR / f"ucoe_sequences{suffix}.fa"

    ensure_genome_fasta(genome_path)

    df = pd.read_csv(scored_path, sep="\t")
    if top_n:
        df = df.head(top_n)

    # Create BED with informative names
    tmp_bed = OUTPUT_DIR / "_tmp_extract.bed"
    with open(tmp_bed, "w") as f:
        for _, row in df.iterrows():
            rank = int(row.get("composite_rank", 0))
            gene1 = row.get("gene1", "?")
            gene2 = row.get("gene2", "?")
            score = row.get("composite_score", 0)
            name = f"rank{rank:03d}_{gene1}_{gene2}_score{score:.4f}"
            f.write(
                f"{row['chrom']}\t{row['start']}\t{row['end']}\t{name}\t0\t.\n"
            )

    result = extract_sequences_bedtools(tmp_bed, genome_path, output_fasta)
    tmp_bed.unlink(missing_ok=True)

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    out = extract_sequences_from_scored(top_n=top_n)
    print(f"Done: {out}")
