# output/ — Resultados do Pipeline

Todos os arquivos neste diretório são gerados automaticamente pelo pipeline. Para regenerar, execute `python3 -m ucoe_pipeline.main`.

## Arquivos raiz

| Arquivo | Descrição |
|---|---|
| `ucoe_candidates.bed` | Coordenadas das 599 candidatas em formato BED (carregável no UCSC Genome Browser) |
| `ucoe_sequences.fa` | Sequências FASTA de todas as 599 candidatas |
| `ucoe_sequences_top20.fa` | Sequências FASTA dos 20 candidatos melhor ranqueados |
| `pipeline_summary.txt` | Relatório-texto com resumo do pipeline e top 20 candidatos |

## phase1/ — Resultados da Fase I (Filtragem)

Cada arquivo contém os candidatos remanescentes após o filtro correspondente, em formato TSV.

| Arquivo | Filtro | Candidatos |
|---|---|---|
| `filter1_candidates.tsv` | Pares HKG divergentes (≤5 kb) | 789 |
| `after_filter1.tsv` | Idem (formato intermediário) | 789 |
| `after_filter2.tsv` | + Sobreposição CpG ≥40% | 692 |
| `after_filter3.tsv` | + Marcas ativas ubíquas | 647 |
| `after_filter4.tsv` | + Ausência de repressivas | 645 |
| `after_filter5.tsv` | + Hipometilação (<10%) | 599 |
| `after_filter6.tsv` | + Anotação DNase (informativa) | 599 |
| `phase1_final.tsv` | Resultado final da Fase I | 599 |

### Colunas principais (phase1_final.tsv)
- `chrom`, `start`, `end`: coordenadas genômicas (hg38)
- `gene1`, `gene2`: genes housekeeping flanqueadores
- `inter_tss_distance`: distância entre TSSs (pb)
- `cpg_overlap_fraction`: fração de sobreposição com ilhas CpG
- `H3K4me3_mean`, `H3K27ac_mean`, etc.: fold change médio por marca de histona
- `meth_mean`: metilação média (%) via WGBS

## phase2/ — Resultados da Fase II (Ranqueamento)

| Arquivo | Descrição |
|---|---|
| `scored_candidates.tsv` | 599 candidatos com escores e ranqueamento |
| `reference_profile.tsv` | Perfil de features dos 3 UCOEs de referência |
| `sensitivity_analysis.tsv` | Estabilidade de cada candidato em 29 combinações de pesos |
| `loo_validation.tsv` | Resultados da validação leave-one-out |

### Colunas principais (scored_candidates.tsv)
- `composite_score`: escore UCOE composto (0–1, maior = mais similar ao perfil UCOE)
- `composite_rank`: posição no ranqueamento (1 = melhor)
- `mahalanobis_dist`, `mahalanobis_score`: distância e escore de Mahalanobis
- `cosine_sim`, `cosine_score`: similaridade cosseno e escore
- `percentile_score`: escore percentil composto
- Features: `H3K4me3_mean`, `H3K4me3_cv`, `meth_mean`, `meth_cv`, `DNase_mean`, `repliseq_mean`, `ctcf_peaks`, `cpg_obs_exp`, `cpg_gc_pct`, `inter_tss_distance`

## structural/ — Análise Estrutural do DNA

Gerado por `python3 run_structural_analysis.py`.

| Arquivo | Descrição |
|---|---|
| `candidates_structural.tsv` | Métricas estruturais das 599 candidatas |
| `known_ucoes_structural.tsv` | Métricas estruturais dos 3 UCOEs conhecidos |
| `controls_structural.tsv` | Métricas estruturais de 200 ilhas CpG controle |
| `all_groups_structural.tsv` | Todos os grupos combinados (para plotagem) |
| `statistical_comparison.tsv` | Testes Mann-Whitney U: candidatas vs. controles |

### Métricas estruturais
- `flexibility_mean`: índice de flexibilidade de Brukner (maior = mais flexível)
- `stiffness_mean`: rigidez dinucleotídica (maior = mais rígido)
- `bendability_mean`: bendabilidade trinucleotídica
- `gc_content`: conteúdo GC
- `cpg_obs_exp`: razão CpG observado/esperado
- `nuc_score_mean`: escore de formação nucleossômica (maior = mais nucleossomo-propenso)
- `nuc_depleted_fraction`: fração de posições nucleossomo-depletadas
- `nfr_score`: escore de região livre de nucleossomos

## figures/ — Figuras

| Arquivo | Descrição |
|---|---|
| `filter_funnel.png` | Funil de filtragem (Fase I): candidatos por etapa |
| `score_distribution.png` | Histograma de escores compostos com UCOEs marcados |
| `top_ranked.png` | Bar chart dos top 30 candidatos |
| `metric_comparison.png` | Scatter: Mahalanobis vs. Cosseno, colorido por percentil |
| `radar/` | Gráficos de radar dos top 10 candidatos |
| `structural/` | Figuras da análise estrutural (boxplots, scatter, tabela) |
