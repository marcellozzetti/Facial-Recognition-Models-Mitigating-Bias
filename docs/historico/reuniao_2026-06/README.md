# Material arquivado da reunião com orientador — Junho 2026

> **Reunião realizada em 2026-06-04** (Prof. Marcos Quiles).
> Material preservado para rastreabilidade. **NÃO deve ser citado em
> novos materiais sem revisão** — feedback do orientador motivou
> reformulação para tese v3.2.

## Arquivos preservados

| Arquivo | Descrição |
|---|---|
| `material_reuniao_orientador_2026-06.md` | Documento original em 5 capítulos + tabela de relevância dos 23 papers + (antes Anexo A com 25 perguntas — removido na refatoração) |
| `material_reuniao_orientador_2026-06.pptx` | Apresentação PowerPoint usada na reunião (36 slides, refatorados conforme feedback do usuário pré-reunião) |
| `_gerar_apresentacao.py` | Script Python que gerou o PPTX a partir do .md |

## Resultado da reunião

Orientador validou: (1) evolução metodológica, (2) FaceScanPaliGemma
como SOTA, (3) linha de pesquisa proposta.

**Pediu ampliação em 4 frentes:**

1. **Revisar método** de cada artigo com mais profundidade.
2. **Ampliar pesquisa** para venues de ML / Redes Neurais.
3. **Double-check do SOTA** com janela expandida 2025-2026.
4. **Reformular tese como prescritiva**:
   - Treinar classifier de tonalidade (MST)
   - Aplicar ao FairFace e catalogar como nova feature
   - Re-treinar race classifier considerando composição de tom
   - Avaliar melhora em métricas de fairness
   - **Estender a face recognition** (ex.: RFW, BFW)
   - Verificar se reconhecimento de grupos sub-representados (Black) melhora

## Encaminhamentos pós-reunião

- Rodada 5 executada (6 papers de ML/redes neurais — Hardt 2016, FiLM,
  Zemel 2013, LAFTR, Zhang 2018, Kleinberg 2017) — ver
  `docs/ativo/04_pesquisa_bibliografica/`.
- Tese v3.1 → **v3.2** (pendente de redação completa).
- Revisão crítica de método nas fichas existentes (pendente).
- SOTA double-check (pendente).
- Atualização do PPTX para v3.2 (após o item acima).

## Política de uso deste diretório

- **Consulta**: livre, para rastreabilidade do processo.
- **Reuso**: somente após revisão crítica + atualização para v3.2.
- **Status**: arquivado em 2026-06-04 após sanitização do `docs/ativo/`.
