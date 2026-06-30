---
data: 2026-06-28
tipo: cola-apresentacao
status: ativo
---

# Cola de bolso — reunião 28/jun/2026 (1 página)

## Abertura (30s)
> "Professor, das 4 decisões da reunião passada, **todas 4 fechei**. Trouxe a evolução em 5 pontos, incluindo a adequação ética do projeto à Resolução 200/2021 da Unifesp."

## Ponto 1 — Revisão bibliográfica concluída (1 min)
- **103 de 104 fichas VERIFIED** (99 %) — corpus blindado
- **103 PDFs** no repositório com autoria correta verificada via leitura
- Auditoria de qualidade aplicada: 29 A + 14 B + 57 C + 4 D
- **Bibliografia consolidada** em `referencias.bib` com 104 entradas — pronta para Overleaf

## Ponto 2 — Cross-reference sistemático (1-2 min)
> "Fiz uma auditoria cruzando cada elemento da tese contra as 104 fichas."

- Objetivo geral + 6 OEs + 7 contribuições + 6 hipóteses + storytelling cruzados
- **Veredito**: nenhum conflito não endereçado; 7 conflitos identificados, todos com resposta defensiva mapeada
- Documento: `_validacao_cross_reference_v3.md`

## Ponto 3 — Validação externa via NotebookLM ⭐ (2-3 min)
> "Para ter segunda opinião externa, conduzi análise paralela no NotebookLM Google AI Plus."

**Procedimento:**
- Importei o corpus organizado em 4 tiers de relevância
- Submeti 7 perguntas-chave (originalidade, valor científico, divergências, gaps, recência, defensibilidade do pipeline)

**Resultado convergente** com nossa análise interna em 6 dimensões. Veredito do NotebookLM:
> *"O trabalho está muito bem estruturado, possui contribuições claras e está fundamentado em literatura de ponta. O ponto crítico da defesa será a decomposição do erro Latinx — onde reside sua maior contribuição intelectual."*

**2 sugestões novas — incorporadas:**
1. **Heterogeneidade intra-Latinx** → Rodada 8 com 3 fichas integradas
2. **Sensitivity analysis SkinToneNet** → OE-2 v3.5 com validação a 2-3 classificadores MST

## Ponto 4 — Rodada 8: fundamentação Latinx (1-2 min)
> "Latinx F1 ≈ 60 % persistente em SOTA não tinha fundamentação empírica explícita de heterogeneidade. Fechei isso com 3 fichas."

**Tripé empírico consolidado:**
- 🌎 **Antropologia**: Telles 2014 (PERLA — 4 países LatAm, pigmentocracia)
- 🧬 **Genética**: Bryc 2015 (AJHG — 162k indivíduos, heterogeneidade Native+European+African)
- 👥 **Sociologia**: Pew 2017 (97 % → 50 % identidade Hispanic cross gerações)

**Sustenta empíricamente:**
- H3: Latinx spread MST ≥ 5 das 10 classes
- C6: decomposição fenotípico × algorítmico do erro Latinx

## Ponto 5 — Adequação ética CEP/Unifesp (1 min)
> "Avaliei a Resolução 200/2021 do Conselho Universitário. O projeto se enquadra no Art. 8º — pesquisa sem envolvimento direto ou indireto de seres humanos."

- **Decisão**: removida validação Prolific (crowdsourcing externo) do OE-1 → substituída por **validação interna** (Mestrando + Orientador, ~200-300 imgs estratificadas)
- **Resultado**: dispensa de submissão ao CEP — exige apenas **Declaração de Responsabilidade**
- **3 assinaturas necessárias**: Mestrando + Orientador + **Chefe de Departamento** (a confirmar nesta reunião)
- Documento atualizado: `_objetivo_tese` v3.6 + `_checklist_etica_cep.md`

## Solicitação pessoal (1 min)
> "Professor, preciso solicitar uma extensão de 2 meses no prazo."

- Nascimento da minha filha em **28/março/2026**
- Período de afastamento nas semanas seguintes para cuidar dela e da mãe
- Solicitação: prorrogação de **15/jul → 15/set/2026**
- Carta formal já redigida

## Perguntas que vou fazer

1. **Concorda em promover H6 para OE-6 formal** (decomposição quantitativa pixel info × skin tone)?
2. **Estudo comparativo com 4 configurações** (A baseline / B FiLM-MST / C Gated FiLM / D FiLM-CLIP) atende à recomendação anterior?
3. Alguma orientação sobre **template Overleaf institucional** ou padrão ABNT específico Unifesp/ICT?
4. **Co-orientador** — alguma definição?
5. **Chefe do Departamento** — quem deve assinar a Declaração de Responsabilidade do CEP?
6. **Extensão de 2 meses** — qual o trâmite formal (carta, requerimento via SEI, processo)?

## Fechamento (30s)
> "Próxima semana eu volto com o Capítulo 1 escrito. Qualquer ajuste do que conversarmos hoje, ajusto antes."

---

## Se houver tempo (extra)

- Mostrar `_validacao_cross_reference_v3.md` na tela
- Mostrar tabela com 4 configurações do Cap 2
- Mostrar tripé empírico Latinx (Telles + Bryc + Pew)
- Mostrar mapa de citações por capítulo

---

**Tempo estimado**: 10-12 min apresentação + Q&A.

**Documento completo**: [_reuniao_2026-06-28_evolucao.md](_reuniao_2026-06-28_evolucao.md)
