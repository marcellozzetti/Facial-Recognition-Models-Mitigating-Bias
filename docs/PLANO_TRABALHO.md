# Plano de Trabalho — Mestrado (até qualificação e defesa)

**Aluno:** Marcello Ozzetti · **Orientador:** Prof. `[nome do orientador]`
**Programa:** Mestrado em Ciência da Computação — Unifesp/ICT
**Kickoff:** 2026-05-11 · **Qualificação alvo:** 2026-08-24 · **Defesa final:** ~2027 (a definir)
**Última atualização deste plano:** 2026-05-14

> Este documento substitui a Parte V da [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md).
> O objetivo do trabalho é **provisório até o SOTA review** (ver §1) — esta é
> uma decisão metodológica, não indecisão.

---

## 1. Premissa central — o SOTA é o gate do objetivo

O objetivo definitivo da dissertação **não pode ser travado antes da
revisão do estado da arte**, porque o objetivo precisa ser um **delta
publicável** sobre o que já existe na literatura 2024–2026. A cadeia de
dependência é:

```
SOTA review  →  mapa do que já existe  →  identificação do delta  →  OBJETIVO TRAVADO  →  programa experimental restante
   (Fase 0)                                  (gate de decisão)
```

### Gate G0 — RESOLVIDO em 2026-05-15 (objetivo TRAVADO)

O SOTA review + auditoria semântica de 555 papers
([sota_review.md](sota_review.md) §5.0, §5.0-crosscheck;
[literature_semantic_audit.md](literature_semantic_audit.md) §5)
estabeleceram que:

- ❌ "HPO/NAS multi-objetivo para fairness facial" **já existe**
  (NeurIPS 2023, FairGRAPE 2022, FineFACE 2024) → não é contribuição.
- ✅ Deltas defensáveis: **critério Pareto-aware best-epoch** (zero
  método igual em 555 docs) + **decomposição experimental controlada**
  (lacuna explícita do survey 2025).

**Decisão (Linha A — Atribuição Causal de Viés):** a tese deixa de ser
"mais uma mitigação" e passa a ser uma **metodologia de atribuição
causal de disparidade demográfica**. Reframe central: não responder
*se* a IA pode ser mais justa, mas **onde intervir** para torná-la
justa, quantificando a contribuição marginal de cada fator.

### Objetivo DEFINITIVO (travado, gate G0)

> "Propor uma metodologia de **atribuição causal de disparidade
> demográfica** em reconhecimento facial, baseada em decomposição
> experimental controlada e em otimização multi-objetivo Pareto-aware,
> que quantifica a contribuição marginal de cada fator — integridade do
> dataset, topologia do classificador, função de perda, paradigma de
> aprendizado e backbone — para a (in)justiça do sistema, respondendo
> não *se* a IA pode ser mais justa, mas *onde intervir* para
> torná-la justa."

**Reenquadramento conceitual dos eixos experimentais:** o que antes era
"testar contrastivo / losses / backbones" agora são **fatores da
decomposição causal**. Cada eixo do programa experimental deixa de ser
"mais um experimento" e passa a ser "um fator cuja contribuição
marginal para a disparidade é isolada e quantificada via o protocolo
controlado (mesma seed, mesmo split, um fator por vez) + frente de
Pareto F1×IR".

**Outputs de publicação derivados (Linha A como espinha):**
- Linha B (critério Pareto-aware) → paper de métodos (AutoML/NeurIPS-W).
- Linha C (viés de cena correlacionado a raça) → capítulo + paper curto.

---

## 2. Escopo — o que ENTRA na dissertação

Diferente da versão anterior do plano, **todos os eixos abaixo entram**
(qualificação = subconjunto + plano; defesa final = tudo):

| Eixo experimental | Diretriz | Qualificação (24/08) | Defesa final |
|---|---|---|---|
| Pipeline auditável (refatoração + correção MBA) | — | ✅ feito | ✅ |
| Auditoria multi-face + R2 (efeito limpeza) | 4 | ✅ feito | ✅ |
| HPO topologia de head (R1/R2) + Pareto-aware | 2,3 | ✅ feito | ✅ |
| Refit Pareto winners 25 epochs (Fase 4) | — | ✅ a fazer (curto) | ✅ |
| **SOTA review 2024–2026** | 1 | ✅ **obrigatório** | ✅ expandido |
| **Aprendizado contrastivo** (SimCLR, SupCon, CLIP) | 5 | 🟡 1 piloto + plano | ✅ completo |
| **Famílias de loss** (AdaFace, MagFace, KP-RPE) | 6 | 🟡 1 piloto + plano | ✅ completo |
| **Múltiplos backbones** (ViT, ConvNeXt, IR-SE) | — | 🟡 1 piloto + plano | ✅ completo |
| Pesquisa semântica ~900 refs | 8 | 🟡 se corpus chegar | ✅ |

**Legenda:** ✅ obrigatório/feito · 🟡 parcial/piloto.

A lógica: a qualificação apresenta **problema + SOTA + resultados
preliminares fortes + plano completo + pilotos que provam viabilidade**.
A defesa final tem o programa experimental completo.

---

## 3. Trilhos paralelos

O trabalho roda em **2 trilhos simultâneos** para caber no calendário:

- **Trilho A — Escrita & Literatura** (contínuo, baixo uso de GPU):
  SOTA review, escrita da qualificação, organização de documentação.
- **Trilho B — Experimentos** (intensivo em GPU, RTX 4070 SUPER):
  pilotos e rodadas completas dos eixos experimentais.

Trilho A nunca bloqueia Trilho B e vice-versa, exceto no **gate da Fase 0**
(SOTA trava o objetivo, que orienta a priorização do Trilho B).

---

## 4. Cronograma até a qualificação (2026-05-14 → 2026-08-24, ~15 semanas)

### Semana 1 (14–18 mai) — Fechamento da fundação
- **B:** Fase 4 — refit dos 2 vencedores Pareto R2 em 25 epochs (3 seeds). [~3h GPU]
- **A:** Commit + organização de toda a documentação (HISTORICO, GLOSSARIO, slides).
- **A:** Início do SOTA review — montar protocolo de busca (queries, venues, anos).
- **Entrega:** evidência empírica fechada para os 3 movimentos atuais.

### Semanas 2–4 (19 mai – 8 jun) — SOTA review (Fase 0, gate)
- **A:** Revisão sistemática 2024–2026. Venues alvo: CVPR/ICCV/ECCV/WACV
  workshops on fairness, IJCB, BMVC, ACM FAccT, IEEE TBIOM, TPAMI.
- **A:** Eixos a mapear no SOTA:
  1. Fairness em face recognition (métricas, datasets, mitigação)
  2. Dataset cleaning / label noise em fairness
  3. HPO multi-objetivo para fairness (existe critério Pareto-aware?)
  4. Contrastive learning para fairness facial
  5. Quality-adaptive losses (AdaFace/MagFace/KP-RPE) em contexto de bias
  6. Backbones modernos (ViT/ConvNeXt) em fairness facial
- **A:** Se o corpus de ~900 refs do MBA chegar → rodar
  `scripts/semantic_search_corpus.py` (Diretriz 8).
- **GATE (fim da Semana 4):** reunião com orientador → **travar objetivo
  definitivo** + priorizar eixos do Trilho B conforme o delta identificado.
- **Entrega:** capítulo de revisão de literatura (rascunho) + objetivo travado.

> **Protocolo comum a todos os fatores (3–5):** cada fator é medido
> isoladamente — varia-se **apenas o fator**, tudo o mais fixo no
> melhor ponto consolidado dos fatores já medidos. **3 seeds (42,1,2)
> sempre**, base de comparação casada, resultado em média ± dp; ganho
> dentro de 1 dp é declarado não-significativo. Mesmo rigor do Fator 1
> (dataset, 12-config batch) e Fator 2 (topologia, Fase 4). Regra
> registrada em memória do projeto.

### Semanas 5–6 (9–22 jun) — Fator 4: Paradigma de aprendizado (Diretriz 5)
- **Pergunta:** qual a contribuição marginal do aprendizado contrastivo
  para a disparidade, isolado dos demais fatores?
- **B:** SupCon (Supervised Contrastive) sobre backbone ResNet50 fixo +
  melhor head/dataset dos fatores 1–2. **3 seeds.** [~impl + ~9h GPU]
- **B:** SimCLR (self-supervised) como segundo ponto do fator. **3 seeds.**
- **Controle:** braço "sem contrastivo" (normal) com **3 seeds**, base
  casada → Δ atribuível só ao paradigma.
- **A:** `docs/factor4_contrastive_results.md` (média ± dp, Δ vs controle).
- **G1 (fim Sem. 6):** contrastivo viável? Senão → só SupCon, documentar.

### Semanas 7–8 (23 jun – 6 jul) — Fator 3: Função de perda (Diretriz 6)
- **Pergunta:** qual a contribuição marginal de losses quality-adaptive
  vs a loss-base, isolada da topologia e do dataset?
- **B:** AdaFace e MagFace como losses plugáveis (mesma interface de
  `ArcMarginProduct`). [~impl]
- **B:** Cada loss × **3 seeds** no melhor ponto dos fatores 1–2;
  braço-controle = loss-base (CE) **3 seeds**, base casada.
- **A:** `docs/factor3_loss_results.md` (média ± dp, Δ vs controle).
- **G2 (fim Sem. 8):** AdaFace/MagFace estáveis? Senão → documentar
  instabilidade como achado (coerente com o achado ArcFace do Fator 1).

### Semanas 9–10 (7–20 jul) — Fator 5: Backbone
- **Pergunta:** qual a contribuição marginal do backbone (capacidade /
  indutive bias) para a disparidade, mantido o melhor head?
- **B:** ResNet-50 (controle) vs ViT-B/16 vs ConvNeXt-T, cada um ×
  **3 seeds**, base casada, melhor head/loss/dataset dos fatores 1–4.
- **A:** `docs/factor5_backbone_results.md` (média ± dp, Δ vs ResNet-50).
- **G3 (fim Sem. 10):** ViT/ConvNeXt cabem em 12GB? Senão → batch 64 +
  grad-accum, ou só ConvNeXt-T; documentar a restrição.

> **Saída da decomposição:** uma tabela única "contribuição marginal de
> cada fator (Δ F1, Δ IR, ± dp, significância)" — Fatores 1 (dataset) e
> 2 (topologia) já medidos; 3–5 nas Semanas 5–10. É o **resultado
> central da tese** (mapa de *onde intervir*). Mais um experimento de
> **interação** (par de fatores com maior |Δ|) se houver orçamento.

### Semanas 11–13 (21 jul – 10 ago) — Escrita da qualificação
- **A:** Consolidar todos os capítulos:
  1. Introdução + motivação regulatória
  2. Revisão de literatura (SOTA da Fase 0)
  3. Pipeline auditável
  4. Movimento 1: auditoria do dataset (R2)
  5. Movimento 2: HPO de topologia (R1/R2)
  6. Movimento 3: critério Pareto-aware
  7. Pipeline integrado + decomposição
  8. Resultados preliminares dos 3 pilotos (contrastivo/loss/backbone)
  9. Plano de trabalho para a defesa final
- **A:** Preparar slide deck final (a partir de
  [slides_qualificacao.md](slides_qualificacao.md)).
- **Entrega:** documento de qualificação completo + apresentação.

### Semanas 14–15 (11–24 ago) — Revisão e ensaio
- **A:** Revisão com orientador (sessões semanais de segunda).
- **A:** Ensaio da apresentação, ajustes finais.
- **24/08:** **Qualificação.**

---

## 5. Cronograma pós-qualificação (defesa final, ~2027)

Após a qualificação, o programa experimental completo é executado:

| Bloco | Conteúdo | Estimativa |
|---|---|---|
| Contrastivo completo | SupCon + SimCLR + CLIP × dataset limpo, com HPO | ~4 semanas |
| Losses completo | AdaFace + MagFace + KP-RPE × HPO, com auditoria de fairness | ~4 semanas |
| Backbones completo | ViT + ConvNeXt + IR-SE × melhor recipe, com HPO | ~3 semanas |
| Matriz cruzada | Melhores combinações (backbone × loss × head × contrastivo) | ~3 semanas |
| Generalização cross-dataset | Validar em RFW / BUPT-Balancedface | ~3 semanas |
| Escrita da dissertação final + paper | Consolidação + submissão | ~6 semanas |

Os achados de cada bloco entram no paper alvo (ver §7).

---

## 6. Gates de decisão e contingência

| Gate | Quando | Critério | Se falhar |
|---|---|---|---|
| **G0 — Objetivo travado** | fim Semana 4 (pós-SOTA) | Delta publicável identificado vs SOTA | Pivotar para o eixo menos explorado (provável: Pareto-aware + contrastivo) |
| **G1 — Contrastivo viável** | fim Semana 6 | Piloto roda e produz números coerentes | Reduzir a SupCon apenas; SimCLR/CLIP só na defesa |
| **G2 — Losses viável** | fim Semana 8 | AdaFace/MagFace plugam sem instabilidade catastrófica | Documentar instabilidade como achado; manter ArcFace baseline |
| **G3 — Backbones viável** | fim Semana 10 | ViT/ConvNeXt treinam na 4070 12GB | Reduzir batch / usar gradient accumulation; ou só ConvNeXt-T |
| **G4 — Escrita no prazo** | fim Semana 13 | Rascunho completo de qualificação | Cortar pilotos mais fracos da apresentação, manter no plano |

**Risco principal:** GPU única (RTX 4070 SUPER 12GB) é gargalo. Mitigação:
AMP sempre ligado em experimentos novos (3× speedup já comprovado), HPO
com orçamento curto (8 epochs) + refit só dos vencedores, early stopping
agressivo (patience=5).

**Risco secundário:** ViT-B/16 pode não caber em 12GB com batch 128.
Mitigação: batch 64 + gradient accumulation, ou ViT-S/16.

---

## 7. Objetivo de publicação

| Foro | Tipo | Conteúdo | Janela |
|---|---|---|---|
| Qualificação Unifesp | Marco acadêmico | Problema + SOTA + prelim + plano | 2026-08-24 |
| TDC SP 2026 | Palestra | Pipeline de auditoria de fairness (já preparada) | CFP aberto |
| Workshop tier-A (WACV/ICCV/ECCV Fair CV; ou IJCB 2027; ou ACM FAccT 2027) | Paper | Delta confirmado pós-SOTA (provável: "Pareto-aware HPO para fairness demográfica + decomposição limpeza/topologia/contrastivo") | 2027 H1 |
| Defesa final | Dissertação | Programa completo | ~2027 |

**O paper só será submetido após o SOTA confirmar o delta.** Se o SOTA
mostrar que o delta principal já existe, o paper pivota para a
contribuição metodológica (critério Pareto-aware) ou para a decomposição
empírica multi-eixo, que são menos prováveis de já estarem publicadas.

---

## 8. Ações imediatas (esta semana)

1. **[B]** Fase 4 — refit Pareto winners R2 em 25 epochs (pode disparar já).
2. **[A]** Iniciar o SOTA review — montar o protocolo de busca e começar
   a coleta. **Este é o item de caminho crítico** (gate G0).
3. **[A]** Commit da documentação de organização (HISTORICO/GLOSSARIO/
   slides/este plano).

A definição **definitiva** do objetivo sai no **gate G0 (fim da Semana 4)**,
após o SOTA. Até lá, o objetivo provisório (§1) orienta o trabalho sem
travá-lo.

---

## Apêndice — mapeamento diretrizes do orientador → fases deste plano

| Diretriz (kickoff 2026-05-11) | Onde é endereçada |
|---|---|
| 0 — Tese positiva | Objetivo provisório §1 (claim positivo) |
| 1 — TOP Line / SOTA | **Fase 0, Semanas 2–4 (gate G0)** |
| 2 — MLP head | ✅ feito (HPO R1/R2) |
| 3 — Optuna | ✅ feito (HPO R1/R2) |
| 4 — Limpeza multi-face | ✅ feito (auditoria + R2) |
| 5 — Aprendizado contrastivo | Semanas 5–6 (piloto) + defesa final (completo) |
| 6 — Famílias de loss | Semanas 7–8 (piloto) + defesa final (completo) |
| 7 — Docs em `docs/` | ✅ contínuo |
| 8 — Pesquisa semântica ~900 refs | Fase 0 (se corpus disponível) |
