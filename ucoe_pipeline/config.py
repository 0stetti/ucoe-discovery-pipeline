"""
Central configuration for the UCOE discovery pipeline.
All paths, thresholds, cell line lists, and known UCOE coordinates.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "ucoe_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data subdirectories
CHIPSEQ_DIR = DATA_DIR / "chipseq"
DNASE_DIR = DATA_DIR / "dnase"
METHYLATION_DIR = DATA_DIR / "methylation"
REPLISEQ_DIR = DATA_DIR / "repliseq"
ANNOTATION_DIR = DATA_DIR / "annotation"

# Annotation files
GENCODE_GTF = ANNOTATION_DIR / "gencode.v44.annotation.gtf.gz"
CPG_ISLANDS_FILE = ANNOTATION_DIR / "cpgIslandExt.txt.gz"
CHROM_SIZES_FILE = ANNOTATION_DIR / "hg38.chrom.sizes"
GENOME_FASTA = ANNOTATION_DIR / "hg38.fa"
HK_GENES_FILE = ANNOTATION_DIR / "housekeeping_genes_eisenberg2013.txt"

# ── Cell lines ───────────────────────────────────────────────────────────────
CELL_LINES = [
    "GM12878",
    "K562",
    "HepG2",
    "H1-hESC",
    "HUVEC",
    "HSMM",
    "NHLF",
    "NHEK",
    "HMEC",
    "IMR-90",
    "A549",
]

# ── Histone marks ────────────────────────────────────────────────────────────
ACTIVE_MARKS = ["H3K4me3", "H3K27ac"]
REPRESSIVE_MARKS = ["H3K27me3", "H3K9me3"]
ALL_MARKS_FOR_RANKING = [
    "H3K4me3", "H3K27ac", "H3K9ac", "H3K36me3",
    "H3K27me3", "H3K9me3",
]

# ── Thresholds ───────────────────────────────────────────────────────────────
# Fold-change signal threshold (ENCODE standard)
SIGNAL_PRESENT_THRESHOLD = 2.0   # mark is "present" if mean FC > this
SIGNAL_ABSENT_THRESHOLD = 2.0    # repressive mark is "absent" if mean FC < this

# DNase accessibility threshold — lower than histone marks because UCOEs
# maintain constitutively open (not hyper-accessible) chromatin.
# Known UCOEs show DNase FC ≈ 0.5–1.9, so FC > 1.0 captures accessibility
# above background without requiring hypersensitivity.
DNASE_PRESENT_THRESHOLD = 1.0

# Ubiquity: fraction of cell lines that must satisfy the criterion
UBIQUITY_FRACTION = 0.80

# CpG island overlap
# CpG overlap threshold — set to 40% to accommodate UCOEs like TBP/PSMB1
# where two CpG islands are separated by a short gap at the bidirectional
# promoter (46% overlap after region extension).
CPG_OVERLAP_FRACTION = 0.40

# Methylation
HYPOMETHYLATION_THRESHOLD = 10.0  # mean methylation < 10%

# Divergent gene pair distance
MAX_INTER_TSS_DISTANCE = 5000  # ≤5 kb between TSSs of divergent pair

# ── Known human UCOEs (hg38 coordinates) ─────────────────────────────────────
KNOWN_UCOES = {
    "A2UCOE_HNRNPA2B1_CBX3": {
        "chrom": "chr7",
        "start": 26_199_798,
        "end": 26_202_442,
        "genes": ("HNRNPA2B1", "CBX3"),
        "description": (
            "A2UCOE — 2,644 bp CpG island spanning the bidirectional promoter "
            "of CBX3(+)/HNRNPA2B1(-) on chr7 (Williams et al., 2005)"
        ),
    },
    "TBP_PSMB1": {
        "chrom": "chr6",
        "start": 170_553_036,
        "end": 170_554_735,
        "genes": ("TBP", "PSMB1"),
        "description": (
            "TBP/PSMB1 UCOE — bidirectional promoter region between "
            "PSMB1(-)/TBP(+) on chr6, spanning two CpG islands (Benton et al., 2002)"
        ),
    },
    "SRF_UCOE_SURF1_SURF2": {
        "chrom": "chr9",
        "start": 133_356_273,
        "end": 133_357_090,
        "genes": ("SURF1", "SURF2"),
        "description": (
            "SRF-UCOE — 817 bp CpG island at the bidirectional promoter "
            "of SURF2(+)/SURF1(-) on chr9 (Rudina & Smolke, 2019)"
        ),
    },
}

# ── Phase II ranking weights ─────────────────────────────────────────────────
RANKING_WEIGHTS = {
    "mahalanobis": 0.4,
    "cosine": 0.3,
    "percentile": 0.3,
}

# ── Logging ──────────────────────────────────────────────────────────────────
import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging():
    """Configure root logger for the pipeline."""
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
    return logging.getLogger("ucoe_pipeline")
