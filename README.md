# UCOE Discovery Pipeline

A two-phase computational pipeline for *de novo* identification of **Ubiquitous Chromatin Opening Elements (UCOEs)** in the human genome (GRCh38/hg38).

> Elton Roger Ostetti — Universidade de São Paulo (USP)

---

## Overview

UCOEs are regulatory DNA elements that maintain open chromatin and active transcription across diverse cell types. This pipeline identifies novel UCOE candidates genome-wide by integrating multi-tissue epigenomic data from ENCODE:

- Bidirectional transcription from housekeeping gene pairs
- CpG island overlap
- Histone modification profiles (H3K4me3, H3K27ac, H3K9ac, H3K27me3, H3K9me3, H3K36me3)
- DNase-seq hypersensitivity
- DNA methylation (WGBS)
- DNA structural properties (flexibility, nucleosome occupancy, k-mer periodicity)

---

## Repository Structure

```
ucoe-pipeline/
├── ucoe_pipeline/        # Main Python package
│   ├── config.py         # Central configuration (thresholds, known UCOEs)
│   ├── main.py           # Pipeline entry point
│   ├── phase1/           # Phase I — Epigenomic filtering (5 filters)
│   ├── phase2/           # Phase II — Multivariate similarity ranking
│   ├── structural/       # DNA structural analysis modules
│   ├── visualization/    # Figure generation
│   └── utils/            # I/O, BigWig, BED, sequence utilities
│
├── scripts/              # Standalone analysis scripts
│   ├── download_encode_ucoe.py     # ENCODE data download
│   ├── run_structural_analysis.py
│   ├── run_conservation_analysis.py
│   ├── run_kmer_analysis.py
│   ├── run_periodicity_analysis.py
│   ├── run_reranking.py
│   ├── run_threshold_sensitivity.py
│   └── extract_sequences.sh
│
├── webapp/               # Interactive Streamlit web application
│   ├── app.py
│   ├── requirements.txt
│   └── data/             # Precomputed data for the webapp
│
├── output/               # Pipeline results
│   ├── ucoe_candidates.bed               # Candidate genomic coordinates (BED)
│   ├── ucoe_sequences.fa                 # Candidate FASTA sequences
│   ├── ucoe_integrated_classification.tsv
│   ├── phase1/           # Candidate sets after each filter (TSV)
│   ├── phase2/           # Rankings, LOO validation, sensitivity (TSV)
│   ├── structural/       # Structural properties per candidate (TSV)
│   └── reranking/        # PCA-based reranking results (TSV)
│
├── ucoe_data/            # Raw genomic data — download separately (see below)
│   ├── annotation/       # hg38.chrom.sizes, housekeeping gene lists
│   ├── chipseq/          # ChIP-seq bigWig files (ENCODE)
│   ├── dnase/            # DNase-seq (ENCODE)
│   ├── methylation/      # WGBS (ENCODE)
│   └── repliseq/         # Repli-seq (ENCODE)
│
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/0stetti/ucoe-pipeline.git
cd ucoe-pipeline
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, pyBigWig, pybedtools, NumPy, SciPy, pandas, scikit-learn, matplotlib, seaborn, samtools, bedtools.

---

## Usage

### Full pipeline

```bash
python -m ucoe_pipeline.main
```

### Phase II only (from existing Phase I results)

```bash
python -m ucoe_pipeline.main --skip-to-phase2 output/phase1/phase1_final.tsv
```

### Individual analyses

```bash
python scripts/run_structural_analysis.py
python scripts/run_kmer_analysis.py
python scripts/run_periodicity_analysis.py
python scripts/run_conservation_analysis.py
python scripts/run_threshold_sensitivity.py
```

### Web application

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

---

## Pipeline Description

### Phase I — Epigenomic Filtering (789 → 599 candidates)

| Filter | Criterion | Cutoff |
|--------|-----------|--------|
| 1 | Divergent housekeeping gene promoter pairs | ≤ 5 kb apart |
| 2 | CpG island overlap (±500 bp extension) | ≥ 40% |
| 3 | Active histone marks (H3K4me3 + H3K27ac) across cell lines | FC > 2.0 in ≥ 80% |
| 4 | Repressive mark absence (H3K27me3 + H3K9me3) | FC < 2.0 in ≥ 80% |
| 5 | Constitutive DNA hypomethylation (WGBS) | Mean < 10% |

### Phase II — Multivariate Similarity Ranking

Each candidate is scored against the centroid of the 3 known human UCOEs (HNRPA2B1-CBX3, DNAPTP6-ATP5A1, DHFR) using a 21-feature epigenomic vector:

| Metric | Weight |
|--------|--------|
| Mahalanobis distance | 40% |
| Cosine similarity | 30% |
| Percentile scoring | 30% |

Validated by leave-one-out cross-validation and sensitivity analysis across 29 weight combinations.

---

## Raw Data

Raw epigenomic datasets (~47 GB) must be downloaded before running the pipeline:

```bash
python scripts/download_encode_ucoe.py --output-dir ucoe_data/
```

**Sources:**
- [ENCODE Project](https://www.encodeproject.org) — ChIP-seq, DNase-seq, WGBS, Repli-seq
- [GENCODE v44](https://www.gencodegenes.org) — Gene annotation
- [UCSC](https://genome.ucsc.edu) — hg38 genome, CpG islands
- [Human Protein Atlas v23](https://www.proteinatlas.org) — Housekeeping gene list

---

## Citation

```bibtex
@software{ostetti_ucoe_2026,
  author    = {Ostetti, Elton Roger},
  title     = {UCOE Discovery Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/0stetti/ucoe-pipeline}
}
```

## License

[MIT](LICENSE)
