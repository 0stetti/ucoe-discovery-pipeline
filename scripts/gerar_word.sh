#!/bin/bash
# Converte a monografia Markdown para Word (.docx) com figuras incorporadas.
# As imagens referenciadas no Markdown são automaticamente embutidas no DOCX.
#
# Uso:
#   ./scripts/gerar_word.sh
#
# Requer: pandoc (brew install pandoc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT="$PROJECT_DIR/docs/monografia_ucoe.md"
OUTPUT="$PROJECT_DIR/output/monografia_ucoe.docx"

echo "Convertendo Markdown -> Word..."
echo "  Entrada: $INPUT"
echo "  Saída:   $OUTPUT"

pandoc "$INPUT" \
    -o "$OUTPUT" \
    --from markdown \
    --to docx \
    --resource-path="$PROJECT_DIR" \
    --standalone \
    --toc \
    --toc-depth=3 \
    \
    --reference-doc="$PROJECT_DIR/reference.docx" 2>/dev/null \
    || pandoc "$INPUT" \
        -o "$OUTPUT" \
        --from markdown \
        --to docx \
        --resource-path="$PROJECT_DIR" \
        --standalone \
        --toc \
        --toc-depth=3 \
        --number-sections

echo ""
echo "Arquivo gerado: $OUTPUT"
echo ""
echo "Notas:"
echo "  - Todas as figuras foram incorporadas automaticamente no .docx"
echo "  - Sumário (TOC) gerado com 3 níveis de profundidade"
echo "  - Seções com numeração manual (ABNT)"
echo "  - Para formatação ABNT completa, abra no Word e aplique o template ABNT"
echo "  - Fórmulas LaTeX aparecem como texto — edite no Word com o editor de equações"
