---
data: 2026-06-15
tipo: material-apresentacao
participantes: [Marcello Ozzetti, Prof. Marcos Quiles]
reuniao_anterior: _reuniao_2026-06-08.md
status: preparacao
---

# Reunião 15/jun/2026 — Evolução da semana

> **1 semana após** a reunião de 08/jun que registrou 8 decisões.
> **30 dias** restantes até o deadline de qualificação (15/jul/2026).

---

## 1. Status das 8 decisões da reunião anterior

| # | Decisão (08/jun) | Status | Evidência |
|---|---|---|---|
| 1 | Pipeline 6 etapas aprovado | ✅ Consolidado | `_validacao_cientifica_pipeline.md` |
| 2 | Usar SkinToneNet pré-treinado | ✅ Adotado | Etapa 1 do pipeline; ficha [pereira_2026.md](04_pesquisa_bibliografica/pereira_2026.md) **VERIFIED** após leitura HTML integral |
| 3 | Corpus ≥100 artigos (filtros menos restritivos) | ✅ **101 fichas** | +46 fichas na R7 (de 55 → 101) |
| 4 | ≥20 artigos de 2025-2026 | ✅ **25 fichas** | `_corpus_distribuicao_ano.md` |
| 5 | Adicionar CLIP/BLIP como conditioning alternativo ao FiLM | ✅ **Track I criada** (14 fichas) + análise técnica entregue | `_revisao_critica_corpus_v2.md` §3 |
| 6 | Submissão qualificação até 15/jul/2026 | 🟡 Em execução | 30 dias restantes; roadmap 4 semanas definido |
| 7 | Usar LaTeX em Overleaf | 🟡 Skeleton pronto | `docs/tese/main.tex` + 5 capítulos `.tex` |
| 8 | Storytelling: Contexto → Problemas → Estado da Arte → Gap → Objetivo → Método | ✅ Narrativa pronta | `_pre_qualificacao_narrativa.md` (~5500 palavras) |

**Resultado:** 6/8 ✅ concluídas, 2/8 🟡 em execução conforme planejado.

---

## 2. Entregáveis principais da semana

### 2.1 Corpus bibliográfico — Rodada 7 completa
- **101 fichas** distribuídas em **11 tracks** (cresceu de 6 → 11)
- **25 fichas de 2025-2026** ✅ meta atingida
- 46 novas fichas em 3 levas (top venues CVPR/ICCV + FR fundadores + complementares)

### 2.2 Pente fino crítico do corpus (`_revisao_critica_corpus_v2.md`)
Classificação por impacto na tese:

| Categoria | Fichas | % |
|---|---|---|
| 🟢🟢 Forte favorável | 12 | 11.9% |
| 🟢 Favorável | 38 | 37.6% |
| 🟡 Neutra/contextual | 18 | 17.8% |
| 🔴 Conflito moderado | 5 | 5.0% |
| 🔴🔴 **Conflito forte** | **2** | 2.0% |
| 🟣 Caminho alternativo | 26 | 25.7% |

**Conflitos fortes (apenas 2)**: Pangelinan 2023 (pixel info → endereçado via H6) e Neto 2025 (reconhecido como limitação). **Nenhuma refutação categórica sem resposta defensiva**.

### 2.3 Justificativa dos 11 tracks
- **A-G** (R1-R5): esqueleto original
- **I, J, K** (R7): novos tracks justificados
  - **I**: VLM/CLIP/BLIP fairness (14) — **resposta direta ao orientador**
  - **J**: Conditioning moderno LoRA/ViT (5)
  - **K**: FR fundadores ArcFace/FaceNet/AdaFace (6) — preenche gap óbvio
- **L**: Auxiliar/complementar (13)

### 2.4 Análise técnica CLIP vs FiLM (resposta ao orientador)
Documento crítico endereça a observação "FiLM parece mais antigo que CLIP":

**Esclarecimento conceitual:** FiLM ≠ CLIP — categorias distintas
- **FiLM** = mecanismo/camada (modula features)
- **CLIP** = modelo/embedding (produz representações)
- **NÃO são comparáveis no mesmo nível**

**Por que mantemos FiLM como mecanismo central:**
| Critério | FiLM | Cross-attn (CLIP) |
|---|---|---|
| Adequação a baixa-dim (10 MST) | ✅ sweet spot | ❌ overkill |
| Parâmetros adicionais | ~1% | ~3% |
| Interpretabilidade γ, β | ✅ direta | ⚠️ menor |
| Compatibilidade com ablation | ✅ limpa | ✅ |

**Decisão proposta:** Manter FiLM como mecanismo central + **adicionar CLIP-conditioning como Contribuição C7 (ablation arquitetural Cap 2)** com 5 configurações comparadas. Se CLIP superar → reportamos (informação útil, não invalidação).

**Cronologia:** FaceNet 2015, ArcFace 2019, FiLM 2018, CLIP 2021, LoRA 2021 — todos ainda padrão em 2024-2026. **Idade ≠ datedness**.

---

## 3. Métricas da semana (números)

| Métrica | Valor |
|---|---|
| Commits na semana | **25** |
| Fichas adicionadas | +46 (55 → **101**) |
| Tracks criadas | +5 novos tracks (I, J, K + reorg) |
| Documentos críticos produzidos | 4 (consolidada, distribuição ano, pente fino, evolução) |
| Conflitos identificados | 7 (5 moderados + 2 fortes) — **todos com resposta defensiva** |

---

## 4. Pontos para discussão hoje

### 4.1 Validação da resposta sobre CLIP vs FiLM
- O orientador concorda com o esclarecimento conceitual (FiLM ≠ CLIP)?
- Está OK manter FiLM central + CLIP-conditioning como C7 (ablation)?

### 4.2 Validação do corpus consolidado
- **101 fichas** + **11 tracks** atendem ao critério?
- Algum track precisa de mais fichas?

### 4.3 Roadmap LaTeX (30 dias)
- Semana 1 (15-21/jun): Setup Overleaf + Cap 1 (Introdução)
- Semana 2 (22-28/jun): Cap 2 (Revisão literatura — 11 tracks)
- Semana 3 (29/jun-5/jul): Cap 3 (Objetivos) + Cap 4 (Metodologia)
- Semana 4 (6-12/jul): Cap 5 (Cronograma) + revisão final
- Buffer: 13-15/jul

### 4.4 Itens operacionais
- Template Overleaf Unifesp/ICT — já existe? Precisa solicitar?
- Co-orientador formalizado?
- Banca preliminar — sugestões?

---

## 5. Próximos passos imediatos (próxima semana)

1. **Setup Overleaf** com template institucional
2. **Gerar `.bib` consolidado** a partir das 101 fichas
3. **Escrever Cap 1 — Introdução** usando `_pre_qualificacao_narrativa.md` como base
4. **Importar imagens** já produzidas (FiLM pipeline, ConvNeXt-T, métricas Hardt)

---

## 6. Riscos identificados

| Risco | Mitigação |
|---|---|
| Atraso no template Overleaf | Começar com `\documentclass{article}` padrão e migrar depois |
| Promoção OVERVIEW_ONLY → VERIFIED demanda tempo | Priorizar apenas as 12 forte favorável + 2 conflito forte |
| Conflito Pangelinan/Neto | Já tratado via H6 + limitação reconhecida |

---

## Anexos (documentos produzidos)

- [_revisao_critica_corpus_v2.md](_revisao_critica_corpus_v2.md) — pente fino + 11 tracks + CLIP vs FiLM
- [_corpus_analise_consolidada.md](_corpus_analise_consolidada.md) — 101 fichas + 7 conflitos + 8 decisões
- [_corpus_distribuicao_ano.md](_corpus_distribuicao_ano.md) — distribuição temporal
- [_pre_qualificacao_narrativa.md](_pre_qualificacao_narrativa.md) — narrativa storytelling ~5500 palavras
- [_reuniao_2026-06-08.md](_reuniao_2026-06-08.md) — ata reunião anterior
