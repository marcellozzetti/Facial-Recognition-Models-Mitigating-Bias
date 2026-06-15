---
data: 2026-06-15
tipo: cola-apresentacao
status: ativo
---

# Cola de bolso — reunião 15/jun/2026 (1 página)

## Abertura (30s)
> "Professor, dos 8 itens da nossa reunião passada, **6 fechei** e **2 estão em execução conforme planejado**. Trouxe a evolução em 4 pontos."

## Ponto 1 — Corpus expandido para 101 fichas (1 min)
- Meta era ≥100 → **entreguei 101**
- Meta era ≥20 de 2025-2026 → **entreguei 25**
- Cresceu de 6 → **11 tracks temáticos**

## Ponto 2 — Pente fino crítico do corpus (1 min)
- Classifiquei as **101 fichas em 6 categorias** por impacto na tese
- **12 forte favorável**, 38 favorável, 18 neutra, 26 caminho alternativo
- Apenas **2 em conflito forte** (Pangelinan + Neto) — **ambos com resposta defensiva**
- **Conclusão**: corpus está sólido para a qualificação

## Ponto 3 — Resposta sobre CLIP vs FiLM (2-3 min) ⭐ CRÍTICO
> "Sobre sua observação de que FiLM parece mais antigo que CLIP — fui a fundo."

**Esclarecimento que vou apresentar:**
1. **FiLM ≠ CLIP — são objetos de categorias diferentes**
   - FiLM = **mecanismo** (camada que modula features)
   - CLIP = **modelo** (produz embeddings)
   - Não substituem um ao outro
2. **No nosso caso (vetor MST 10-dim baixa-dim):**
   - FiLM é o sweet spot: ~1% de parâmetros, interpretável
   - Cross-attn com CLIP seria overkill (3× mais parâmetros)
3. **Idade não é problema:** FaceNet 2015, ArcFace 2019, FiLM 2018 — todos padrão em 2026
4. **Proposta concreta:** mantenho FiLM como mecanismo central E **adiciono CLIP-conditioning como Contribuição C7** (ablation arquitetural no Cap 2)
   - 5 configurações comparadas
   - Se CLIP superar → reporto como informação útil (não invalidação)

> "Faz sentido para o senhor?"

## Ponto 4 — Roadmap LaTeX 30 dias (1 min)
- **Semana 1** (esta): Setup Overleaf + Cap 1 (Introdução)
- **Semana 2**: Cap 2 (Revisão literatura — 11 tracks)
- **Semana 3**: Cap 3 (Objetivos) + Cap 4 (Metodologia)
- **Semana 4**: Cap 5 (Cronograma) + revisão final
- **Buffer**: 13-15/jul

## Perguntas que vou fazer

1. "Concorda com manter FiLM central + CLIP como C7 (ablation)?"
2. "11 tracks estão bem? Algum precisa de mais fichas?"
3. "Tem template Overleaf da Unifesp/ICT ou usamos o padrão?"
4. "Co-orientador — já podemos formalizar?"
5. "Sugestões para banca preliminar?"

## Se houver tempo (extra)

- Mostrar `_revisao_critica_corpus_v2.md` na tela (documento técnico)
- Mostrar a tabela de distribuição por ano
- Mostrar imagens já produzidas (pipeline FiLM, ConvNeXt-T, métricas Hardt)

## Fechamento (30s)
> "Próxima semana eu volto com o Cap 1 escrito em LaTeX + bibliografia consolidada. Qualquer ajuste do que conversamos hoje, ajusto antes."

---

**Tempo total estimado**: 6-8 min de apresentação + Q&A.

**Documento completo**: [_reuniao_2026-06-15_evolucao.md](_reuniao_2026-06-15_evolucao.md)
