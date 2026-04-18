#!/bin/bash
# Extract FASTA sequences for UCOE candidates from hg38 reference genome.
# Usage:
#   ./extract_sequences.sh              # Extract all 599 candidates
#   ./extract_sequences.sh 20           # Extract top 20 candidates only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ANNOTATION_DIR="$PROJECT_DIR/ucoe_data/annotation"
OUTPUT_DIR="$PROJECT_DIR/output"
GENOME="$ANNOTATION_DIR/hg38.fa"
GENOME_GZ="$ANNOTATION_DIR/hg38.fa.gz"
GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

TOP_N="${1:-}"

# 1. Ensure genome is downloaded and indexed
if [ ! -f "$GENOME" ]; then
    if [ ! -f "$GENOME_GZ" ]; then
        echo "Downloading hg38 reference genome (~3 GB)..."
        curl -L -o "$GENOME_GZ" "$GENOME_URL"
    fi
    echo "Decompressing genome..."
    gunzip -k "$GENOME_GZ"
fi

if [ ! -f "${GENOME}.fai" ]; then
    echo "Indexing genome..."
    samtools faidx "$GENOME"
fi

# 2. Extract sequences
BED="$OUTPUT_DIR/ucoe_candidates.bed"
if [ ! -f "$BED" ]; then
    echo "ERROR: BED file not found: $BED"
    echo "Run the pipeline first to generate candidate regions."
    exit 1
fi

if [ -n "$TOP_N" ]; then
    OUT="$OUTPUT_DIR/ucoe_sequences_top${TOP_N}.fa"
    echo "Extracting top $TOP_N sequences..."
    head -n "$TOP_N" "$BED" > "$OUTPUT_DIR/_tmp_top.bed"
    bedtools getfasta -fi "$GENOME" -bed "$OUTPUT_DIR/_tmp_top.bed" -fo "$OUT" -name
    rm -f "$OUTPUT_DIR/_tmp_top.bed"
else
    OUT="$OUTPUT_DIR/ucoe_sequences.fa"
    echo "Extracting all sequences..."
    bedtools getfasta -fi "$GENOME" -bed "$BED" -fo "$OUT" -name
fi

N_SEQS=$(grep -c "^>" "$OUT")
echo "Done: $N_SEQS sequences extracted to $OUT"
