# Scripts operacionais — usados na preparação do corpus (jun/2026)

Scripts one-shot executados durante a construção e revisão do corpus
bibliográfico de 104 fichas. Arquivados em julho/2026 após corpus
consolidado. Mantidos para referência histórica e eventual
re-execução em rodadas futuras.

## Arquivos

- `_auditar_fichas.py` — auditoria de qualidade das fichas (A/B/C/D)
- `_baixar_pdfs_arxiv.py` — download automático de PDFs de arXiv (61 papers)
- `_baixar_pdfs_extras.py` — download de PDFs individuais (Bian, Zhao, Counterfactual, CosFace)
- `_corrigir_autores.sh` — batch de correção de autoria via sed (27 fichas)
- `_inventariar_pdfs.py` — script de inventário dos PDFs
- `_organizar_pdfs_por_tier.py` — organização em 4 tiers para NotebookLM
- `_promover_overviewonly.sh` — batch de promoção de fichas OVERVIEW_ONLY → VERIFIED

## Contexto

Todos executados entre 15/jun e 30/jun de 2026. Resultados
incorporados aos documentos ativos. Se precisar re-executar (por
exemplo, adicionar novo tier de fichas), voltar aqui.
