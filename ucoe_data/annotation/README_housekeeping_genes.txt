================================================================================
INSTRUÇÕES PARA DOWNLOAD DA LISTA DE HOUSEKEEPING GENES
================================================================================

Referência: Eisenberg, E. & Levanon, E.Y. (2013). Human housekeeping genes,
revisited. Trends in Genetics, 29(10), 569-574.
DOI: 10.1016/j.tig.2013.05.010

A lista de ~3.800 housekeeping genes está disponível como material suplementar
(Table S1) no artigo acima.

OPÇÕES DE DOWNLOAD:

1. Acesse: https://doi.org/10.1016/j.tig.2013.05.010
   → Clique em "Supplementary Material" ou "Appendix"
   → Baixe a Table S1 (lista de HKGs)
   → Salve como: /Users/eltonostetti/Documents/USP/Tese/Algoritmo UCOE/ucoe_data/annotation/housekeeping_genes_eisenberg2013.txt

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
  /Users/eltonostetti/Documents/USP/Tese/Algoritmo UCOE/ucoe_data/annotation/housekeeping_genes_eisenberg2013.txt

================================================================================
