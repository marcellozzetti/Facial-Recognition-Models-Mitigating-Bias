# Pauta da Sessão com o Orientador — 2026-05-18 (segunda-feira)

**Tipo:** sessão semanal de orientação (não é a qualificação).
**Objetivo desta sessão:** validar as decisões metodológicas tomadas
desde o kickoff (2026-05-11) e alinhar 1 ponto técnico novo levantado
pelo próprio orientador.

> Ordenado por prioridade. Itens ⭐ exigem decisão/validação do
> orientador antes de seguir.

---

## ⭐ 1. Ponto novo do orientador — "função de pesos" vs undersampling

**Origem:** o orientador comentou sobre **função de pesos**; eu havia
falado de **função de perda** — não são a mesma coisa.

- **Função de perda** = a fórmula que mede o erro (CrossEntropy, ArcFace).
- **Função de pesos / loss ponderada** = multiplicador por classe
  (ou por amostra) **dentro** da loss; mantém todos os dados mas faz a
  classe minoritária "pesar mais" (∝ 1/frequência).

**Por que isso importa para a tese:** são **duas alavancas distintas e
clássicas** para o mesmo problema de desbalanceamento:

1. **Undersampling** — o que o MBA fez (joga fora dados da majoritária).
2. **Weighted loss** — nunca testamos (mantém os dados, repondera o erro).

**Perguntas para confirmar com o orientador:**
- Confirmar a interpretação: ele se referia a **ponderação de classes
  na loss (weighted loss)** como alternativa/complemento ao
  undersampling? (ou à matriz de pesos do ArcFace, ou outra coisa?)
- Se sim: **incluir "estratégia de balanceamento" como um fator da
  decomposição causal** — comparar `undersampling` vs `weighted-loss`
  vs `nenhum`, no protocolo padrão (3 seeds, base casada). É uma
  alavanca que falta no estudo e encaixa direto na Linha A.

**Preparado:** esboço de configuração desse fator pronto para gerar
assim que ele confirmar (mesmo gerador/protocolo dos Fatores 1-2).

---

## ⭐ 2. Validar o gate G0 — objetivo travado (Linha A)

Decisão tomada em 2026-05-15 após revisão sistemática + auditoria
semântica de 555 papers. **Precisa do endosso do orientador.**

- **Antes:** "mais uma mitigação de viés" (espaço saturado — NeurIPS
  2023, FairGRAPE, FineFACE já fazem HPO/NAS multi-objetivo para
  fairness).
- **Agora (objetivo definitivo):** **metodologia de atribuição causal**
  — quantificar a contribuição marginal de cada fator (dataset,
  topologia, loss, paradigma, backbone) para a disparidade. Responder
  *onde intervir*, não *se* dá pra melhorar.
- **Delta validado contra 555 papers:** critério Pareto-aware
  best-epoch (sem método igual no corpus) + decomposição controlada
  (lacuna explícita do survey 2025).

**Pergunta:** ele endossa o reposicionamento de "mitigação" para
"atribuição causal"? (referência: `docs/PROPOSTA_MESTRADO.md` §1,
`docs/sota_review.md` §5.3, `docs/PLANO_TRABALHO.md` §1).

---

## 3. Reportar — decisões metodológicas de rigor (informe, não decisão)

- **Regra dos 3 seeds:** todo experimento agora roda 3 seeds (42,1,2),
  base casada, média ± dp; ganho < 1 dp = não-significativo. Motivada
  pela Fase 4 (variância entre seeds ≈ 0.168 > efeito do fator ≈ 0.068).
- **Confound detectado e corrigido:** R1 (torch 2.5+cv2) vs R2 (torch
  2.12+PIL) confundia limpeza de dataset com versão de framework. Em
  correção: batch casado de 12 configs (R1ctrl vs R2base, 3 seeds, mesmo
  ambiente). Mencionar como exemplo de maturidade metodológica (vira
  subseção "ameaças à validade e correções" na dissertação).

---

## 4. Reportar — resultados defensáveis (2/5 fatores) + 3 reversões

Batch casado 12-config 3-seed (ambiente único) **concluído**. Reportar
as 3 reversões — é o ponto mais forte da sessão (mostra o método
funcionando):

- **Fator 1 (Dataset):** limpeza melhora **acurácia no CE (+1,35pp,
  signif.)**, **efeito nulo sobre IR** em ambos os recipes. O
  "ArcFace −54% IR" reportado antes era **artefato de confound**
  (torch/decode + 1-seed) — **não se confirma**.
- **Fator 2 (Topologia):** vs baseline linear de **3 seeds**, MLP
  `[256] GELU drop=0,52` **reduz IR −0,11 (estatisticamente
  significativo)**. A conclusão anterior "não-significativo" usava
  baseline de 1-seed (pessimista). **A topologia é a alavanca de
  fairness defensável, não a limpeza.**
- **Mensagem-chave para o orientador:** a metodologia de atribuição
  **matou um efeito falso e revelou um real** — isso *é* a tese (Linha
  A) se provando. Vira a subseção "ameaças à validade e correções".
- **Achado lateral (Linha C):** taxa multi-face correlacionada com raça
  (spread 9,3pp) — viés de cena, robusto.

---

## 5. Pontos de alinhamento de cronograma

- Qualificação alvo: **2026-08-24** (confirmar se segue de pé).
- Pendência da Diretriz 8: corpus de 479 papers (cited-by FairFace via
  OpenAlex) já extraído e auditado semanticamente — perguntar se ele
  quer os ~900 do Google Scholar como complemento ou se 479
  deduplicados bastam.
- 5 PDFs sinalizados pela auditoria para leitura pré-submissão de paper
  (Utility-Fairness 2024, DSAP 2024, Fairness-in-Details 2025,
  FairGRAPE 2022, FineFACE 2024) — alinhar prioridade.

---

## Resumo de 1 minuto (abertura da reunião)

> "Desde o kickoff: travei o objetivo como **atribuição causal** (não
> mitigação) — validado contra 555 papers. Decompondo a disparidade
> fator a fator com 3 seeds e base casada. **2 fatores já medidos e
> defensáveis: a topologia do classificador reduz IR de forma
> significativa (−0,11); a limpeza do dataset NÃO afeta fairness (só
> acurácia).** O método derrubou um efeito falso (ArcFace −54%, que era
> confound) e revelou um real — isso é a tese se provando. O senhor
> levantou **função de pesos** — quero confirmar se é weighted-loss e,
> se for, adicionar 'estratégia de balanceamento' como fator novo."
