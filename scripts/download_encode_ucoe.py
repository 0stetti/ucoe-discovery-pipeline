#!/usr/bin/env python3
"""
================================================================================
UCOE Discovery Pipeline — Script de Download Automático de Dados ENCODE
================================================================================

Este script consulta a API REST do ENCODE Portal e baixa automaticamente todos
os datasets necessários para o pipeline de identificação de UCOEs:

  1. ChIP-seq (histonas): H3K4me3, H3K27ac, H3K9ac, H3K36me3, H3K27me3, H3K9me3
  2. ChIP-seq (CTCF)
  3. DNase-seq
  4. DNA methylation (WGBS / RRBS)
  5. Repli-seq (replication timing)
  6. Anotação de genes (GENCODE v44)
  7. Ilhas CpG (UCSC Genome Browser)
  8. Lista de housekeeping genes (Eisenberg & Levanon, 2013)

Linhagens celulares alvo (ENCODE Tier 1/2):
  GM12878, K562, HepG2, H1-hESC, HUVEC, HSMM, NHLF, NHEK, HMEC, IMR-90, A549

Uso:
  python download_encode_ucoe.py [--output-dir DIR] [--dry-run] [--no-verify]
                                  [--max-retries N] [--threads N]

Autor: Projeto de Doutorado — Pipeline UCOE Discovery
Data: Março 2026
================================================================================
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Genoma de referência
GENOME_ASSEMBLY = "GRCh38"

# Linhagens celulares do ENCODE (Tier 1/2) e suas variantes de nome.
# ENCODE API uses ontology term_names, not legacy abbreviations for some cell types.
CELL_LINES = {
    "GM12878":  ["GM12878"],
    "K562":     ["K562"],
    "HepG2":    ["HepG2"],
    "H1-hESC":  ["H1-hESC", "H1"],
    "HUVEC":    ["endothelial cell of umbilical vein", "HUVEC"],
    "HSMM":     ["myotube", "skeletal muscle myoblast", "HSMM"],
    "NHLF":     ["fibroblast of lung", "NHLF"],
    "NHEK":     ["keratinocyte", "NHEK"],
    "HMEC":     ["mammary epithelial cell", "HMEC"],
    "IMR-90":   ["IMR-90", "IMR90"],
    "A549":     ["A549"],
}

# Alvos de ChIP-seq de histonas
HISTONE_TARGETS = [
    "H3K4me3",   # promotor ativo
    "H3K27ac",   # ativador / enhancer ativo
    "H3K9ac",    # promotor ativo
    "H3K36me3",  # corpo de gene ativo
    "H3K27me3",  # repressivo (Polycomb)
    "H3K9me3",   # heterocromatina constitutiva
]

# URL base do ENCODE
ENCODE_BASE = "https://www.encodeproject.org"

# URLs estáticas para dados não-ENCODE
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
GENCODE_GFF3_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gff3.gz"
CHROM_SIZES_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"
CPG_ISLANDS_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"

# Eisenberg & Levanon HKG — URL do suplemento (Trends in Genetics 2013)
# Nota: o arquivo suplementar precisa ser baixado manualmente do publisher.
# O script gera instruções para download manual se necessário.
HKG_DOI = "10.1016/j.tig.2013.05.010"


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir):
    """Configura logging com arquivo e console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"

    # Formato
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler de arquivo
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Handler de console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger = logging.getLogger("ucoe_download")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log salvo em: {log_file}")
    return logger


# =============================================================================
# ENCODE REST API
# =============================================================================

def encode_search(params, limit="all"):
    """
    Consulta a API REST do ENCODE e retorna resultados em JSON.

    Args:
        params: dict com parâmetros de busca
        limit: número de resultados ou 'all'

    Returns:
        list de objetos JSON (experiments ou files)
    """
    params["format"] = "json"
    params["limit"] = limit

    url = f"{ENCODE_BASE}/search/?{urllib.parse.urlencode(params, doseq=True)}"

    headers = {"Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("@graph", [])
    except Exception as e:
        logging.getLogger("ucoe_download").error(f"Erro na busca ENCODE: {e}")
        logging.getLogger("ucoe_download").debug(f"URL: {url}")
        return []


def encode_get_object(path):
    """
    Obtém um objeto específico do ENCODE por seu path (ex: /experiments/ENCSR...).
    """
    url = f"{ENCODE_BASE}{path}?format=json"
    headers = {"Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logging.getLogger("ucoe_download").error(f"Erro ao obter {path}: {e}")
        return None


def search_encode_files(assay_title, target_label=None, biosample_term_name=None,
                        file_format="bigWig", output_type=None, assembly=GENOME_ASSEMBLY,
                        status="released"):
    """
    Busca arquivos processados no ENCODE com filtros específicos.

    Retorna lista de dicts com: accession, href, md5sum, file_size,
    biosample, target, output_type, assembly.
    """
    logger = logging.getLogger("ucoe_download")

    # NOTE: ENCODE API uses direct field names for File searches,
    # NOT 'dataset.' prefixed names (those cause 404 errors).
    params = {
        "type": "File",
        "assay_title": assay_title,
        "file_format": file_format,
        "assembly": assembly,
        "status": status,
    }

    if target_label:
        params["target.label"] = target_label
    if biosample_term_name:
        params["biosample_ontology.term_name"] = biosample_term_name
    if output_type:
        params["output_type"] = output_type

    results = encode_search(params)

    files = []
    for r in results:
        file_info = {
            "accession": r.get("accession", "unknown"),
            "href": r.get("href", ""),
            "md5sum": r.get("md5sum", ""),
            "file_size": r.get("file_size", 0),
            "file_format": r.get("file_format", ""),
            "output_type": r.get("output_type", ""),
            "assembly": r.get("assembly", ""),
            "dataset": r.get("dataset", ""),
            "biological_replicates": r.get("biological_replicates", []),
        }
        files.append(file_info)

    logger.debug(
        f"  Busca: assay={assay_title}, target={target_label}, "
        f"biosample={biosample_term_name}, format={file_format}, "
        f"output_type={output_type} → {len(files)} arquivos"
    )
    return files


def get_biosample_from_dataset(dataset_path):
    """Extrai o nome do biosample a partir do dataset."""
    obj = encode_get_object(dataset_path)
    if obj:
        bio = obj.get("biosample_ontology", {})
        if isinstance(bio, dict):
            return bio.get("term_name", "unknown")
        elif isinstance(bio, str):
            bio_obj = encode_get_object(bio)
            if bio_obj:
                return bio_obj.get("term_name", "unknown")
    return "unknown"


# =============================================================================
# FUNÇÕES DE DOWNLOAD
# =============================================================================

def download_file(url, dest_path, md5_expected=None, max_retries=3, verify=True):
    """
    Baixa um arquivo com retry e verificação de MD5.

    Args:
        url: URL completa do arquivo
        dest_path: Path de destino
        md5_expected: hash MD5 esperado (opcional)
        max_retries: número máximo de tentativas
        verify: se True, verifica MD5 após download

    Returns:
        tuple (success: bool, message: str)
    """
    logger = logging.getLogger("ucoe_download")
    dest_path = Path(dest_path)

    # Se já existe e MD5 bate, pula
    if dest_path.exists() and verify and md5_expected:
        existing_md5 = compute_md5(dest_path)
        if existing_md5 == md5_expected:
            logger.info(f"  ✓ Já existe (MD5 ok): {dest_path.name}")
            return (True, "already_exists")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  ↓ Baixando: {dest_path.name} (tentativa {attempt}/{max_retries})")

            req = urllib.request.Request(url)
            req.add_header("User-Agent", "UCOE-Pipeline-Downloader/1.0")

            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1 MB

                with open(dest_path, "wb") as f:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)

                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            # Log a cada ~20%
                            if int(pct) % 20 == 0:
                                logger.debug(
                                    f"    {size_mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)"
                                )

            # Verificação MD5
            if verify and md5_expected:
                actual_md5 = compute_md5(dest_path)
                if actual_md5 != md5_expected:
                    logger.warning(
                        f"  ✗ MD5 não confere: {dest_path.name} "
                        f"(esperado={md5_expected}, obtido={actual_md5})"
                    )
                    dest_path.unlink(missing_ok=True)
                    if attempt < max_retries:
                        time.sleep(5 * attempt)
                        continue
                    return (False, "md5_mismatch")
                else:
                    logger.info(f"  ✓ MD5 verificado: {dest_path.name}")

            size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Concluído: {dest_path.name} ({size_mb:.1f} MB)")
            return (True, "downloaded")

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning(f"  ✗ Erro (tentativa {attempt}): {e}")
            dest_path.unlink(missing_ok=True)
            if attempt < max_retries:
                wait = 10 * attempt
                logger.info(f"    Aguardando {wait}s antes de retry...")
                time.sleep(wait)

    return (False, "max_retries_exceeded")


def compute_md5(filepath, chunk_size=1024*1024):
    """Calcula MD5 de um arquivo."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


# =============================================================================
# COLETA DE METADADOS — ENCODE
# =============================================================================

def collect_chipseq_histone_files(cell_lines, targets, logger):
    """
    Coleta metadados de ChIP-seq de histonas para todas as linhagens e alvos.

    Prioridade de output_type:
      1. "fold change over control" (bigWig) — sinal normalizado pelo input
      2. "signal p-value" (bigWig) — alternativa
    """
    logger.info("=" * 70)
    logger.info("COLETANDO METADADOS: ChIP-seq Histonas")
    logger.info("=" * 70)

    all_files = []

    for target in targets:
        logger.info(f"\n  Alvo: {target}")

        for cell_name, cell_aliases in cell_lines.items():
            found = False

            for alias in cell_aliases:
                # Buscar fold change over control (preferido)
                files = search_encode_files(
                    assay_title="Histone ChIP-seq",
                    target_label=target,
                    biosample_term_name=alias,
                    file_format="bigWig",
                    output_type="fold change over control",
                )

                if not files:
                    # Fallback: signal p-value
                    files = search_encode_files(
                        assay_title="Histone ChIP-seq",
                        target_label=target,
                        biosample_term_name=alias,
                        file_format="bigWig",
                        output_type="signal p-value",
                    )

                if files:
                    # Pegar o primeiro (geralmente o mais recente/melhor qualidade)
                    best = files[0]
                    best["cell_line"] = cell_name
                    best["target"] = target
                    best["data_type"] = "chipseq_histone"
                    all_files.append(best)

                    size_mb = best["file_size"] / (1024*1024) if best["file_size"] else 0
                    logger.info(
                        f"    {cell_name}: {best['accession']} "
                        f"({best['output_type']}, {size_mb:.1f} MB)"
                    )
                    found = True
                    break

            if not found:
                logger.warning(f"    {cell_name}: NENHUM ARQUIVO ENCONTRADO para {target}")

    logger.info(f"\n  Total ChIP-seq histonas: {len(all_files)} arquivos")
    return all_files


def collect_ctcf_files(cell_lines, logger):
    """Coleta metadados de ChIP-seq CTCF."""
    logger.info("=" * 70)
    logger.info("COLETANDO METADADOS: ChIP-seq CTCF")
    logger.info("=" * 70)

    all_files = []

    for cell_name, cell_aliases in cell_lines.items():
        found = False
        for alias in cell_aliases:
            # bigWig signal
            files_bw = search_encode_files(
                assay_title="TF ChIP-seq",
                target_label="CTCF",
                biosample_term_name=alias,
                file_format="bigWig",
                output_type="fold change over control",
            )

            # narrowPeak (picos) — try multiple output_type values
            files_np = search_encode_files(
                assay_title="TF ChIP-seq",
                target_label="CTCF",
                biosample_term_name=alias,
                file_format="bed",
                output_type="IDR thresholded peaks",
            )

            if not files_np:
                files_np = search_encode_files(
                    assay_title="TF ChIP-seq",
                    target_label="CTCF",
                    biosample_term_name=alias,
                    file_format="bed",
                    output_type="optimal IDR thresholded peaks",
                )

            if not files_np:
                files_np = search_encode_files(
                    assay_title="TF ChIP-seq",
                    target_label="CTCF",
                    biosample_term_name=alias,
                    file_format="bed",
                    output_type="pseudoreplicated IDR thresholded peaks",
                )

            if files_bw:
                best = files_bw[0]
                best["cell_line"] = cell_name
                best["target"] = "CTCF"
                best["data_type"] = "chipseq_ctcf_signal"
                all_files.append(best)
                logger.info(f"    {cell_name} (signal): {best['accession']}")
                found = True

            if files_np:
                best_np = files_np[0]
                best_np["cell_line"] = cell_name
                best_np["target"] = "CTCF"
                best_np["data_type"] = "chipseq_ctcf_peaks"
                all_files.append(best_np)
                logger.info(f"    {cell_name} (peaks): {best_np['accession']}")
                found = True

            if found:
                break

        if not found:
            logger.warning(f"    {cell_name}: NENHUM arquivo CTCF encontrado")

    logger.info(f"\n  Total CTCF: {len(all_files)} arquivos")
    return all_files


def collect_dnase_files(cell_lines, logger):
    """Coleta metadados de DNase-seq."""
    logger.info("=" * 70)
    logger.info("COLETANDO METADADOS: DNase-seq")
    logger.info("=" * 70)

    all_files = []

    for cell_name, cell_aliases in cell_lines.items():
        found = False
        for alias in cell_aliases:
            # bigWig signal — try multiple output_type values in priority order
            files_bw = None
            for ot in [
                "read-depth normalized signal",
                "signal of unique reads",
                "signal of all reads",
                "signal",
                "fold change over control",
            ]:
                files_bw = search_encode_files(
                    assay_title="DNase-seq",
                    biosample_term_name=alias,
                    file_format="bigWig",
                    output_type=ot,
                )
                if files_bw:
                    break

            # Peak calls — try multiple output types
            files_pk = None
            for ot_pk in [
                "peaks",
                "IDR thresholded peaks",
                "optimal IDR thresholded peaks",
            ]:
                files_pk = search_encode_files(
                    assay_title="DNase-seq",
                    biosample_term_name=alias,
                    file_format="bed",
                    output_type=ot_pk,
                )
                if files_pk:
                    break

            if files_bw:
                best = files_bw[0]
                best["cell_line"] = cell_name
                best["target"] = "DNase"
                best["data_type"] = "dnase_signal"
                all_files.append(best)
                logger.info(f"    {cell_name} (signal): {best['accession']} [{best['output_type']}]")
                found = True

            if files_pk:
                best_pk = files_pk[0]
                best_pk["cell_line"] = cell_name
                best_pk["target"] = "DNase"
                best_pk["data_type"] = "dnase_peaks"
                all_files.append(best_pk)
                logger.info(f"    {cell_name} (peaks): {best_pk['accession']}")

            if found:
                break

        if not found:
            logger.warning(f"    {cell_name}: NENHUM arquivo DNase encontrado")

    logger.info(f"\n  Total DNase-seq: {len(all_files)} arquivos")
    return all_files


def collect_methylation_files(cell_lines, logger):
    """Coleta metadados de WGBS e RRBS.

    FIX: ENCODE uses 'whole-genome shotgun bisulfite sequencing' as
    assay_title for WGBS, not 'WGBS'. We try both. We also restrict
    to bed format (not bigBed) since the pipeline reads BED files.
    """
    logger.info("=" * 70)
    logger.info("COLETANDO METADADOS: DNA Methylation (WGBS/RRBS)")
    logger.info("=" * 70)

    all_files = []

    # Assay title variants — ENCODE changed naming over the years
    WGBS_ASSAY_TITLES = [
        "whole-genome shotgun bisulfite sequencing",
        "WGBS",
    ]
    RRBS_ASSAY_TITLES = [
        "RRBS",
    ]

    for cell_name, cell_aliases in cell_lines.items():
        found = False
        for alias in cell_aliases:
            # Tentar WGBS primeiro (cobertura genômica completa), then RRBS
            assay_groups = [
                ("WGBS", WGBS_ASSAY_TITLES),
                ("RRBS", RRBS_ASSAY_TITLES),
            ]

            for assay_label, assay_titles in assay_groups:
                if found:
                    break
                for assay_title in assay_titles:
                    # Prefer bed format (pipeline can read it directly)
                    files = search_encode_files(
                        assay_title=assay_title,
                        biosample_term_name=alias,
                        file_format="bed",
                        output_type="methylation state at CpG",
                    )

                    if files:
                        best = files[0]
                        best["cell_line"] = cell_name
                        best["target"] = "methylation"
                        best["data_type"] = f"methylation_{assay_label.lower()}"
                        all_files.append(best)

                        size_mb = best["file_size"] / (1024*1024) if best["file_size"] else 0
                        logger.info(
                            f"    {cell_name} ({assay_label}): {best['accession']} "
                            f"({size_mb:.1f} MB) [assay_title={assay_title}]"
                        )
                        found = True
                        break

            if found:
                break

        if not found:
            logger.warning(f"    {cell_name}: NENHUM dado de metilação encontrado")

    logger.info(f"\n  Total Methylation: {len(all_files)} arquivos")
    return all_files


def collect_repliseq_files(cell_lines, logger):
    """Coleta metadados de Repli-seq.

    NOTE: ENCODE Repli-seq data is only available aligned to hg19, not GRCh38.
    We download hg19 files and the pipeline will use liftOver or direct
    coordinate mapping. We select ONE file per cell line, preferring
    wavelet-smoothed signal (E/L ratio).
    """
    logger.info("=" * 70)
    logger.info("COLETANDO METADADOS: Repli-seq (Replication Timing)")
    logger.info("=" * 70)
    logger.info("  NOTA: Repli-seq no ENCODE está apenas em hg19. Baixando hg19 para liftOver.")

    all_files = []

    # Preferred output types in priority order
    REPLISEQ_OUTPUT_TYPES = [
        "wavelet-smoothed signal",
        "percentage normalized signal",
        "signal",
        "read-depth normalized signal",
    ]

    for cell_name, cell_aliases in cell_lines.items():
        found = False
        for alias in cell_aliases:
            # Try each output type in priority order — use hg19 assembly
            for ot in REPLISEQ_OUTPUT_TYPES:
                files = search_encode_files(
                    assay_title="Repli-seq",
                    biosample_term_name=alias,
                    file_format="bigWig",
                    output_type=ot,
                    assembly="hg19",
                )
                if files:
                    best = files[0]
                    best["cell_line"] = cell_name
                    best["target"] = "replication_timing"
                    best["data_type"] = "repliseq"
                    all_files.append(best)
                    logger.info(
                        f"    {cell_name}: {best['accession']} [{ot}] (hg19)"
                    )
                    found = True
                    break

            # Fallback: any bigWig from Repli-seq in hg19
            if not found:
                files = search_encode_files(
                    assay_title="Repli-seq",
                    biosample_term_name=alias,
                    file_format="bigWig",
                    assembly="hg19",
                )
                if files:
                    best = files[0]
                    best["cell_line"] = cell_name
                    best["target"] = "replication_timing"
                    best["data_type"] = "repliseq"
                    all_files.append(best)
                    logger.info(
                        f"    {cell_name}: {best['accession']} (fallback, "
                        f"output_type={best.get('output_type', '?')}) (hg19)"
                    )
                    found = True

            if found:
                break

        if not found:
            logger.warning(f"    {cell_name}: NENHUM dado Repli-seq encontrado")

    logger.info(f"\n  Total Repli-seq: {len(all_files)} arquivos")
    if all_files:
        logger.info("  IMPORTANTE: Arquivos em hg19 — necessário liftOver para hg38 antes de usar no pipeline")
    return all_files


# =============================================================================
# ORGANIZAÇÃO E DOWNLOAD
# =============================================================================

def _get_file_extension(file_info):
    """Derive a clean file extension from ENCODE file_format and href.

    ENCODE's file_format field can be 'bigWig', 'bed', 'bigBed', etc.
    The href usually has the real extension. Use href when available.
    """
    href = file_info.get("href", "")
    if href:
        # href like /files/ENCFF123ABC/@@download/ENCFF123ABC.bigWig
        fname = href.rstrip("/").split("/")[-1]
        # Return everything after the accession prefix
        parts = fname.split(".", 1)
        if len(parts) == 2:
            return parts[1]  # e.g. "bigWig", "bed.gz", "bigBed"

    # Fallback to file_format
    fmt = file_info.get("file_format", "")
    if fmt == "bigWig":
        return "bigWig"
    elif fmt == "bed":
        return "bed.gz"
    elif fmt == "bigBed":
        return "bigBed"
    return fmt


def organize_downloads(all_files, output_dir):
    """
    Organiza os arquivos por tipo de dado em subpastas.

    Estrutura:
        output_dir/
        ├── chipseq/
        │   ├── H3K4me3/
        │   │   ├── GM12878_H3K4me3_ENCFFXXXXXX.bigWig
        │   │   └── ...
        │   ├── H3K27ac/
        │   ├── CTCF/
        │   │   ├── signal/
        │   │   └── peaks/
        │   └── ...
        ├── dnase/
        │   ├── signal/
        │   └── peaks/
        ├── methylation/
        ├── repliseq/
        ├── annotation/
        ├── logs/
        └── metadata/
            └── download_manifest.json
    """
    download_plan = []

    for f in all_files:
        dt = f.get("data_type", "other")
        cell = f.get("cell_line", "unknown")
        target = f.get("target", "unknown")
        acc = f.get("accession", "unknown")
        ext = _get_file_extension(f)

        # Definir subpasta e filename
        if dt == "chipseq_histone":
            subdir = output_dir / "chipseq" / target
            filename = f"{cell}_{target}_{acc}.{ext}"
        elif dt == "chipseq_ctcf_signal":
            subdir = output_dir / "chipseq" / "CTCF" / "signal"
            filename = f"{cell}_CTCF_signal_{acc}.{ext}"
        elif dt == "chipseq_ctcf_peaks":
            subdir = output_dir / "chipseq" / "CTCF" / "peaks"
            filename = f"{cell}_CTCF_peaks_{acc}.{ext}"
        elif dt == "dnase_signal":
            subdir = output_dir / "dnase" / "signal"
            filename = f"{cell}_DNase_signal_{acc}.{ext}"
        elif dt == "dnase_peaks":
            subdir = output_dir / "dnase" / "peaks"
            filename = f"{cell}_DNase_peaks_{acc}.{ext}"
        elif dt.startswith("methylation"):
            assay_type = dt.split("_")[1] if "_" in dt else "wgbs"
            subdir = output_dir / "methylation"
            # Use the real extension from ENCODE (usually .bed.gz)
            filename = f"{cell}_{assay_type}_{acc}.{ext}"
        elif dt == "repliseq":
            subdir = output_dir / "repliseq"
            filename = f"{cell}_repliseq_{acc}.{ext}"
        else:
            subdir = output_dir / "other"
            filename = f"{acc}.{ext}"

        url = f"{ENCODE_BASE}{f['href']}" if f.get("href") else None

        download_plan.append({
            "url": url,
            "dest": subdir / filename,
            "md5": f.get("md5sum", ""),
            "size": f.get("file_size", 0),
            "accession": acc,
            "cell_line": cell,
            "target": target,
            "data_type": dt,
            "output_type": f.get("output_type", ""),
        })

    return download_plan


def add_static_downloads(output_dir):
    """Adiciona downloads estáticos (GENCODE, CpG islands, etc.)."""
    annotation_dir = output_dir / "annotation"

    static = [
        {
            "url": GENCODE_URL,
            "dest": annotation_dir / "gencode.v44.annotation.gtf.gz",
            "md5": "",
            "size": 0,
            "accession": "GENCODE_v44_GTF",
            "cell_line": "N/A",
            "target": "gene_annotation",
            "data_type": "annotation",
            "output_type": "GTF",
        },
        {
            "url": GENCODE_GFF3_URL,
            "dest": annotation_dir / "gencode.v44.annotation.gff3.gz",
            "md5": "",
            "size": 0,
            "accession": "GENCODE_v44_GFF3",
            "cell_line": "N/A",
            "target": "gene_annotation",
            "data_type": "annotation",
            "output_type": "GFF3",
        },
        {
            "url": CHROM_SIZES_URL,
            "dest": annotation_dir / "hg38.chrom.sizes",
            "md5": "",
            "size": 0,
            "accession": "hg38_chrom_sizes",
            "cell_line": "N/A",
            "target": "reference",
            "data_type": "annotation",
            "output_type": "chrom.sizes",
        },
        {
            "url": CPG_ISLANDS_URL,
            "dest": annotation_dir / "cpgIslandExt.txt.gz",
            "md5": "",
            "size": 0,
            "accession": "CpG_Islands_hg38",
            "cell_line": "N/A",
            "target": "cpg_islands",
            "data_type": "annotation",
            "output_type": "BED",
        },
        {
            "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
            "dest": annotation_dir / "hg19ToHg38.over.chain.gz",
            "md5": "",
            "size": 0,
            "accession": "liftOver_hg19_to_hg38",
            "cell_line": "N/A",
            "target": "liftover",
            "data_type": "annotation",
            "output_type": "chain",
        },
    ]

    return static


def generate_hkg_instructions(output_dir, logger):
    """
    Gera instruções para download manual da lista de housekeeping genes.
    (Suplemento de artigo requer acesso ao publisher.)
    """
    instructions_file = output_dir / "annotation" / "README_housekeeping_genes.txt"
    instructions_file.parent.mkdir(parents=True, exist_ok=True)

    text = f"""================================================================================
INSTRUÇÕES PARA DOWNLOAD DA LISTA DE HOUSEKEEPING GENES
================================================================================

Referência: Eisenberg, E. & Levanon, E.Y. (2013). Human housekeeping genes,
revisited. Trends in Genetics, 29(10), 569-574.
DOI: {HKG_DOI}

A lista de ~3.800 housekeeping genes está disponível como material suplementar
(Table S1) no artigo acima.

OPÇÕES DE DOWNLOAD:

1. Acesse: https://doi.org/{HKG_DOI}
   → Clique em "Supplementary Material" ou "Appendix"
   → Baixe a Table S1 (lista de HKGs)
   → Salve como: {output_dir}/annotation/housekeeping_genes_eisenberg2013.txt

2. Alternativa (se disponível no repositório):
   → Acesse: https://www.tau.ac.il/~eli101/HKG/
   → Baixe a lista completa de HKGs

3. O arquivo deve conter gene symbols (uma coluna com nomes de genes).
   O pipeline espera um arquivo TSV/TXT com pelo menos uma coluna "Gene"
   contendo HGNC gene symbols (ex: ACTB, GAPDH, TBP, etc.)

FORMATO ESPERADO PELO PIPELINE:
Gene
ACTB
GAPDH
TBP
...
(um gene symbol por linha, sem header obrigatório)

Após o download, verifique que o arquivo está no diretório:
  {output_dir}/annotation/housekeeping_genes_eisenberg2013.txt

================================================================================
"""

    with open(instructions_file, "w") as f:
        f.write(text)

    logger.info(f"  Instruções HKG salvas em: {instructions_file}")


def save_manifest(download_plan, output_dir, logger):
    """Salva manifesto JSON com todos os metadados dos downloads."""
    manifest_dir = output_dir / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = manifest_dir / "download_manifest.json"

    manifest = {
        "pipeline": "UCOE Discovery Pipeline",
        "genome_assembly": GENOME_ASSEMBLY,
        "download_date": datetime.now().isoformat(),
        "total_files": len(download_plan),
        "total_size_bytes": sum(d["size"] for d in download_plan if d["size"]),
        "total_size_gb": round(sum(d["size"] for d in download_plan if d["size"]) / (1024**3), 2),
        "cell_lines": list(CELL_LINES.keys()),
        "histone_targets": HISTONE_TARGETS,
        "files": [
            {
                "accession": d["accession"],
                "url": d["url"],
                "dest": str(d["dest"]),
                "md5": d["md5"],
                "size_bytes": d["size"],
                "cell_line": d["cell_line"],
                "target": d["target"],
                "data_type": d["data_type"],
                "output_type": d["output_type"],
            }
            for d in download_plan
        ]
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"  Manifesto salvo em: {manifest_file}")
    return manifest


def print_summary(manifest, logger):
    """Imprime resumo do plano de download."""
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO DO PLANO DE DOWNLOAD")
    logger.info("=" * 70)
    logger.info(f"  Genoma: {manifest['genome_assembly']}")
    logger.info(f"  Linhagens celulares: {len(manifest['cell_lines'])}")
    logger.info(f"  Total de arquivos: {manifest['total_files']}")
    logger.info(f"  Tamanho estimado: {manifest['total_size_gb']:.1f} GB")
    logger.info("")

    # Contagem por tipo
    type_counts = {}
    type_sizes = {}
    for f in manifest["files"]:
        dt = f["data_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
        type_sizes[dt] = type_sizes.get(dt, 0) + (f["size_bytes"] or 0)

    logger.info(f"  {'Tipo de dado':<30} {'Arquivos':>10} {'Tamanho':>12}")
    logger.info(f"  {'-'*30} {'-'*10} {'-'*12}")
    for dt, count in sorted(type_counts.items()):
        size_gb = type_sizes.get(dt, 0) / (1024**3)
        logger.info(f"  {dt:<30} {count:>10} {size_gb:>10.2f} GB")

    logger.info(f"  {'-'*30} {'-'*10} {'-'*12}")
    logger.info(
        f"  {'TOTAL':<30} {manifest['total_files']:>10} "
        f"{manifest['total_size_gb']:>10.2f} GB"
    )
    logger.info("=" * 70)


def execute_downloads(download_plan, max_retries=3, verify=True, threads=2, logger=None):
    """
    Executa o download de todos os arquivos.

    Usa ThreadPoolExecutor para paralelização leve (ENCODE pede ≤10 req/s).
    """
    if logger is None:
        logger = logging.getLogger("ucoe_download")

    total = len(download_plan)
    results = {"success": 0, "skipped": 0, "failed": 0, "errors": []}

    logger.info("\n" + "=" * 70)
    logger.info("INICIANDO DOWNLOADS")
    logger.info("=" * 70)

    def _do_download(item_with_idx):
        idx, item = item_with_idx
        url = item["url"]
        dest = item["dest"]
        md5 = item["md5"]

        if not url:
            return (item, False, "no_url")

        logger.info(f"\n[{idx}/{total}] {item['accession']} → {Path(dest).name}")
        success, msg = download_file(url, dest, md5, max_retries, verify)
        return (item, success, msg)

    # Download sequencial ou paralelo leve
    # (ENCODE recomenda ≤10 requests/segundo)
    effective_threads = min(threads, 3)  # Limitar a 3 para respeitar ENCODE

    indexed_plan = list(enumerate(download_plan, start=1))

    with ThreadPoolExecutor(max_workers=effective_threads) as executor:
        futures = {
            executor.submit(_do_download, item): item
            for item in indexed_plan
        }

        for future in as_completed(futures):
            item, success, msg = future.result()
            if success:
                if msg == "already_exists":
                    results["skipped"] += 1
                else:
                    results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "accession": item["accession"],
                    "url": item["url"],
                    "error": msg,
                })

            # Rate limiting
            time.sleep(0.5)

    # Resumo final
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADO DOS DOWNLOADS")
    logger.info("=" * 70)
    logger.info(f"  ✓ Baixados com sucesso:  {results['success']}")
    logger.info(f"  → Já existiam (skip):    {results['skipped']}")
    logger.info(f"  ✗ Falhas:                {results['failed']}")

    if results["errors"]:
        logger.warning("\n  Arquivos com falha:")
        for err in results["errors"]:
            logger.warning(f"    - {err['accession']}: {err['error']}")

    logger.info("=" * 70)
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UCOE Pipeline — Download automático de dados ENCODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Download completo com verificação MD5
  python download_encode_ucoe.py --output-dir ./ucoe_data

  # Apenas gerar manifesto (sem baixar)
  python download_encode_ucoe.py --output-dir ./ucoe_data --dry-run

  # Download sem verificação MD5 (mais rápido)
  python download_encode_ucoe.py --output-dir ./ucoe_data --no-verify

  # Download com 3 threads paralelas
  python download_encode_ucoe.py --output-dir ./ucoe_data --threads 3
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./ucoe_data",
        help="Diretório raiz para salvar os dados (default: ./ucoe_data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas consultar ENCODE e gerar manifesto, sem baixar"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Não verificar MD5 após download (mais rápido)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Número máximo de tentativas por arquivo (default: 3)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Número de downloads paralelos (default: 2, max: 3)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retomar downloads usando manifesto existente"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("UCOE DISCOVERY PIPELINE — DOWNLOAD DE DADOS")
    logger.info("=" * 70)
    logger.info(f"  Diretório de saída: {output_dir}")
    logger.info(f"  Genoma: {GENOME_ASSEMBLY}")
    logger.info(f"  Linhagens: {', '.join(CELL_LINES.keys())}")
    logger.info(f"  Alvos de histonas: {', '.join(HISTONE_TARGETS)}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Verificação MD5: {not args.no_verify}")
    logger.info(f"  Max retries: {args.max_retries}")
    logger.info(f"  Threads: {args.threads}")
    logger.info("")

    # --- FASE 1: Coleta de metadados ---
    if args.resume and (output_dir / "metadata" / "download_manifest.json").exists():
        logger.info("Retomando a partir de manifesto existente...")
        with open(output_dir / "metadata" / "download_manifest.json") as f:
            manifest = json.load(f)

        download_plan = []
        for f_info in manifest["files"]:
            download_plan.append({
                "url": f_info["url"],
                "dest": Path(f_info["dest"]),
                "md5": f_info["md5"],
                "size": f_info["size_bytes"],
                "accession": f_info["accession"],
                "cell_line": f_info["cell_line"],
                "target": f_info["target"],
                "data_type": f_info["data_type"],
                "output_type": f_info["output_type"],
            })
    else:
        logger.info("FASE 1: Consultando ENCODE Portal API...")
        logger.info("(Isso pode levar alguns minutos dependendo da conexão)")
        logger.info("")

        all_encode_files = []

        # 1. ChIP-seq histonas
        all_encode_files.extend(
            collect_chipseq_histone_files(CELL_LINES, HISTONE_TARGETS, logger)
        )
        time.sleep(1)  # Rate limiting

        # 2. CTCF
        all_encode_files.extend(collect_ctcf_files(CELL_LINES, logger))
        time.sleep(1)

        # 3. DNase-seq
        all_encode_files.extend(collect_dnase_files(CELL_LINES, logger))
        time.sleep(1)

        # 4. DNA methylation
        all_encode_files.extend(collect_methylation_files(CELL_LINES, logger))
        time.sleep(1)

        # 5. Repli-seq
        all_encode_files.extend(collect_repliseq_files(CELL_LINES, logger))

        # Organizar
        download_plan = organize_downloads(all_encode_files, output_dir)

        # Adicionar downloads estáticos (GENCODE, CpG islands, etc.)
        download_plan.extend(add_static_downloads(output_dir))

        # Gerar instruções HKG
        generate_hkg_instructions(output_dir, logger)

        # Salvar manifesto
        manifest = save_manifest(download_plan, output_dir, logger)

    # Resumo
    print_summary(manifest, logger)

    # --- FASE 2: Download ---
    if args.dry_run:
        logger.info("\n  *** DRY RUN — Nenhum arquivo será baixado ***")
        logger.info(f"  Manifesto salvo em: {output_dir}/metadata/download_manifest.json")
        logger.info("  Para executar o download, rode sem --dry-run")
    else:
        results = execute_downloads(
            download_plan,
            max_retries=args.max_retries,
            verify=not args.no_verify,
            threads=args.threads,
            logger=logger,
        )

        # Salvar resultado final
        results_file = output_dir / "metadata" / "download_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n  Resultados salvos em: {results_file}")

    logger.info("\nConcluído!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
