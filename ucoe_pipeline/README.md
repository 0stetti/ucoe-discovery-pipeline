# ucoe_pipeline — Código-fonte

Pacote Python com a implementação do pipeline de descoberta de UCOEs.

## Módulos principais

| Arquivo | Descrição |
|---|---|
| `config.py` | Configuração central: caminhos, limiares, linhagens celulares, coordenadas dos UCOEs conhecidos, pesos do ranqueamento |
| `main.py` | Orquestração do pipeline. Aceita flags `--skip-to-phase2`, `--phase1-only`, `--filter1-only` |

## Subdiretórios

### phase1/ — Filtragem baseada em regras
| Módulo | Filtro | O que faz |
|---|---|---|
| `filter_divergent_hkg.py` | Filtro 1 | Identifica pares de genes HKG com TSSs divergentes ≤5 kb |
| `filter_cpg_islands.py` | Filtro 2 | Verifica sobreposição ≥40% com ilhas CpG (extensão ±500 pb) |
| `filter_histone_marks.py` | Filtros 3-4 | Presença ubíqua de marcas ativas; ausência de repressivas |
| `filter_methylation.py` | Filtro 5 | Hipometilação constitutiva (<10%) via WGBS |
| `filter_dnase.py` | Anotação | Anota acessibilidade DNase-seq (informativo, não eliminatório) |

### phase2/ — Ranqueamento por similaridade
| Módulo | O que faz |
|---|---|
| `feature_extraction.py` | Extrai vetor de 21 features epigenômicas por candidato |
| `reference_profile.py` | Constrói perfil de referência (centroide dos 3 UCOEs conhecidos) |
| `similarity_metrics.py` | Computa Mahalanobis, cosseno e percentil |
| `composite_score.py` | Combina métricas em escore composto + análise de sensibilidade |
| `validation.py` | Sanity check e validação leave-one-out |

### structural/ — Análise estrutural do DNA
| Módulo | O que faz |
|---|---|
| `flexibility.py` | Flexibilidade (Brukner), rigidez (Geggier-Vologodskii), bendabilidade trinucleotídica |
| `nucleosome.py` | Predição de ocupação nucleossômica baseada em preferências dinucleotídicas |
| `analysis.py` | Orquestração: carrega sequências, computa métricas, testes Mann-Whitney U |

### visualization/ — Figuras
| Módulo | O que gera |
|---|---|
| `filter_summary.py` | Gráfico de funil (candidatos por filtro) |
| `radar_plots.py` | Gráficos de radar (perfil epigenômico dos top candidatos) |
| `ranking_plots.py` | Distribuição de escores, bar chart dos top candidatos, scatter Mahalanobis vs Cosseno |
| `structural_plots.py` | Boxplots estruturais, scatter rigidez vs nucleossomo, tabela estatística |

### utils/ — Utilitários
| Módulo | O que faz |
|---|---|
| `bigwig_utils.py` | Extração de sinal de arquivos BigWig (pyBigWig) |
| `bed_utils.py` | Operações com BED via pybedtools |
| `io_utils.py` | I/O genérico (diretórios, leitura gzip, salvamento TSV/BED) |
| `sequence_extraction.py` | Extração de sequências FASTA do genoma hg38 via bedtools |
