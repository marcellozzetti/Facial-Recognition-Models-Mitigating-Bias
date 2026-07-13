---
data: 2026-07-13
tipo: material-apresentacao
participantes: [Marcello Ozzetti, Prof. Marcos Quiles]
reuniao_anterior_realizada: 2026-06-15
proxima_entrega: primeira revisao ja disponivel no Overleaf
status: preparacao
marco_1: PRONTO (antecipado em 2 dias)
---

# Reunião com o orientador — 13/jul/2026

> **Marco 1 antecipado**: a primeira revisão da qualificação, prevista
> originalmente para 15/jul, está pronta e no Overleaf a partir de hoje.
> Convite de compartilhamento enviado ao orientador.

---

## 1. Plano até a defesa — três marcos formais

| # | Marco | Data | Status |
|---|---|---|---|
| 1 | Primeira revisão da qualificação ao orientador | 15/jul/2026 | Overleaf compartilhado hoje; Caps 1, 3, 4, 5 em versão inicial completa; **Cap 2 em revisão ativa** |
| 2 | Pedido formal de qualificação ao PPG-CC / ICT | 30/jul/2026 | Prazo regimental — mantido |
| 3 | Defesa da qualificação | outubro/2026 | Ajuste de agosto → outubro (extensão de 2 meses solicitada) |

---

## 2. Estado da escrita — qualificação migrada para o Overleaf

### 2.1 O que foi feito esta semana

A base narrativa em Markdown (aproximadamente 16 mil palavras acumuladas
em documentos temáticos ao longo dos últimos meses) foi **transposta
manualmente para LaTeX no Overleaf**, capítulo por capítulo, ao longo
dos últimos dias. A bibliografia consolidada (arquivo `referencias.bib`
com 104 entradas) foi importada e todas as referências cruzadas foram
testadas.

### 2.2 O que o orientador encontrará no Overleaf

| Capítulo | Conteúdo consolidado |
|---|---|
| Cap 1 — Introdução | **Versão inicial completa.** Contexto do reconhecimento facial, disparidade documentada (Black 90 % × Latinx 60 % em SOTA), pergunta de pesquisa, gap na literatura, objetivo geral. |
| Cap 2 — Revisão bibliográfica | **Em escrita e revisão ativa** — é o capítulo mais denso. Estrutura pronta: fairness formal (Buolamwini 2018, Hardt 2016, Kleinberg 2017), MST (Schumann 2023, Pereira 2026), FiLM (Perez 2018), refutação central (Pangelinan 2023), tripé Latinx (Telles 2014, Bryc 2015, Pew 2017). Densificação com as 104 referências mapeadas está em curso. |
| Cap 3 — Objetivos | **Versão inicial completa.** Objetivo geral + 6 objetivos específicos + 6 hipóteses testáveis + 7 contribuições esperadas. |
| Cap 4 — Metodologia | **Versão inicial completa.** Pipeline em 6 etapas, 4 configurações comparativas (A/B/C/D), rigor experimental, adequação ética CEP formalizada. |
| Cap 5 — Cronograma | **Versão inicial completa.** Marcos consolidados conforme plano de trabalho. |

### 2.3 Postura sobre uso de LLM

Ponto crítico para ser explicitado ao orientador antes da defesa:

> A LLM (Claude Code) foi utilizada como **ferramenta operacional** —
> assistência em resumos após leitura dos papers, formatação padronizada
> de fichas, revisão de coerência entre capítulos. Mas **cada afirmação
> central da tese** — pergunta de pesquisa, escolha do backbone, mecanismo
> FiLM, ajuste ético do OE-1, argumento de heterogeneidade intra-Latinx —
> **nasceu de leitura minha e de conversas nesta sala**. A transposição
> para o Overleaf foi manual, linha a linha, precisamente para forçar
> essa apropriação. **O orientador poderá conferir isso lendo o texto
> entregue hoje.**

### 2.4 Metodologia de trabalho hoje

- Corpo LaTeX no Overleaf — **compartilhado ao orientador nesta reunião**
- Repositório GitHub aberto com histórico completo de commits e edições
- Modo de trabalho da próxima semana: **escuta e revisão** conforme
  comentários chegarem pelo Overleaf

---

## 3. Já resolvido — Declaração de Responsabilidade CEP submetida em 12/jul

A Declaração de Responsabilidade prevista no parágrafo único do
Art. 8º da Resolução 200/2021/CONSU Unifesp foi **submetida ontem
(12 de julho de 2026)**. Contexto rápido:

- **Enquadramento**: Art. 8º — pesquisa puramente computacional
  sobre datasets públicos secundários (FairFace, RFW, BFW), sem
  coleta primária nem crowdsourcing. **Dispensa de submissão ao
  CEP**.
- **Ajuste técnico que viabilizou o enquadramento**: OE-1 mudou de
  validação MST via Prolific (crowdsourcing pago externo) para
  validação interna pela equipe acadêmica — Mestrando + Orientador,
  em subset estratificado de aproximadamente 200 a 300 imagens.
  Documentado como versão 3.6 do objetivo.
- **Assinaturas coletadas**: Mestrando + Orientador + Chefe do
  Departamento.
- **Status administrativo**: nenhum bloqueio em aberto para a
  qualificação. A Declaração assinada será anexada à entrega
  formal em 30/jul.

Documentos-fonte: `_checklist_etica_cep.md`,
`_projeto_declaracao_responsabilidade.md`,
`projeto_declaracao_responsabilidade.docx` / `.pdf`.

---

## 4. Solicitação pessoal — extensão de 2 meses

Reiteração formal da solicitação já apresentada:

- **Motivo**: nascimento de minha filha em 28/março/2026 e período de
  afastamento subsequente para cuidado parental.
- **Marcos preservados**: primeira revisão (15/jul, **já entregue hoje**)
  e pedido formal (30/jul) permanecem intactos.
- **Ajuste solicitado**: prazo da defesa da qualificação passa de
  agosto para outubro/2026.
- **Documento**: carta em dois parágrafos redigida, pronta para envio
  via SEI ou conforme trâmite indicado pelo Programa.

---

## 5. Perguntas objetivas ao orientador

1. **Sobre o Overleaf**: há capítulo que o senhor gostaria de ver com
   tratamento diferente antes da minha próxima rodada de revisão?
2. **Trâmite da extensão de 2 meses**: carta direta ao orientador,
   requerimento via SEI ao PPG, ou processo formal com anexos?
3. **Template Overleaf**: o padrão que está sendo usado atende ao PPG,
   ou é necessário migrar para template institucional específico do
   Unifesp / ICT?
4. **Co-orientador**: alguma definição?
5. **Composição da banca preliminar**: alguma sugestão?

---

## 6. Próximos passos — semana 13-15/jul

1. **Fechar a densificação do Cap 2** (Revisão bibliográfica) com as
   104 referências mapeadas por frente teórica.
2. **Aguardar comentários no Overleaf** (modo escuta e revisão).
3. **Protocolar carta de extensão** conforme trâmite indicado hoje.
4. **Preparar versão 2 do documento** até 30/jul, incorporando o
   feedback.

---

## Anexos

- [_reuniao_2026-07-13_cola.md](_reuniao_2026-07-13_cola.md) — cola de bolso de 1 página
- [_objetivo_tese_v3.3.md](_objetivo_tese_v3.3.md) — v3.6 com adequação ética
- [_checklist_etica_cep.md](_checklist_etica_cep.md) — enquadramento Art. 8º
- [projeto_declaracao_responsabilidade.docx](projeto_declaracao_responsabilidade.docx) — projeto formal em Word
- [projeto_declaracao_responsabilidade.pdf](projeto_declaracao_responsabilidade.pdf) — projeto formal em PDF
- `docs/tese/` — Overleaf sincronizado com repositório GitHub
- `docs/tese/referencias.bib` — bibliografia consolidada (104 entradas)
