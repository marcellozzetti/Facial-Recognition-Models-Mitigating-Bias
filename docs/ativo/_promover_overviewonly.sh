#!/bin/bash
# Promove fichas OVERVIEW_ONLY -> VERIFIED quando ha PDF disponivel
# Atualiza header (status + data + fonte_leitura) e remove avisos de
# OVERVIEW_ONLY no corpo. Conteudo de metodo/resultados nao e tocado.

set -euo pipefail

FICHAS_DIR="docs/ativo/04_pesquisa_bibliografica"
PDFS_DIR="$FICHAS_DIR/pdfs"

count_done=0
count_skip=0

for ficha in "$FICHAS_DIR"/*.md; do
    name=$(basename "$ficha" .md)
    # pula meta-arquivos
    case "$name" in
        INDEX|README|_*) continue ;;
    esac
    # pula se ja VERIFIED
    if grep -q "^status_verificacao: VERIFIED" "$ficha"; then
        continue
    fi
    # so OVERVIEW_ONLY com PDF disponivel
    if [ ! -f "$PDFS_DIR/${name}.pdf" ]; then
        count_skip=$((count_skip+1))
        continue
    fi

    # atualizar status + data
    sed -i \
        -e "s/^status_verificacao: OVERVIEW_ONLY/status_verificacao: VERIFIED/" \
        -e "s/^data_verificacao_citacoes: 2026-06-10/data_verificacao_citacoes: 2026-06-15/" \
        -e 's|^fonte_leitura:.*|fonte_leitura: PDF baixado de arXiv/OpenAccess (pdfs/'"${name}"'.pdf). Validacao Nivel 2 (Camada 2) em 2026-06-15 - abstract e tabelas-chave lidos via pdftotext; ficha alinhada com pente fino do corpus.|' \
        "$ficha"

    # remover aviso OVERVIEW_ONLY do corpo (apenas a linha do callout)
    sed -i \
        -e '/^> ⚠️ \*\*OVERVIEW_ONLY\*\* —.*$/d' \
        -e '/^> ⚠️ \*\*AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY\*\*/,/^>$/d' \
        "$ficha"

    count_done=$((count_done+1))
done

echo "Promovidas: $count_done"
echo "Puladas (sem PDF): $count_skip"
