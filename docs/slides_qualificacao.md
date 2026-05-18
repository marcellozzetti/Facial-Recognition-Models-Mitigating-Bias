---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Mitigação de Viés Racial em Reconhecimento Facial · Marcello Ozzetti · Mestrado Unifesp/ICT'
footer: 'Qualificação 2026-08-24 · github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias'
---

# Mitigação de Viés Racial em Reconhecimento Facial

### Da dissertação de MBA à pipeline auditável

**Marcello Ozzetti** · marcello.ozzetti@gmail.com
Mestrado em Ciência da Computação · Unifesp/ICT
Orientador: Prof. Marcos Quiles

Qualificação · 24 de agosto de 2026

---

## Roteiro

1. Motivação e contexto regulatório
2. Origem: dissertação de MBA (USP, 2024)
3. Reframing positivo da tese (kickoff 2026-05-11)
4. Arquitetura técnica
5. Pipeline experimental
6. Resultado 1: efeito isolado da limpeza do dataset
7. Resultado 2: otimização da topologia do classificador (HPO)
8. Resultado 3: pipeline integrado vs baseline MBA
9. Decomposição dos ganhos
10. Próximos passos e cronograma até a defesa

---

## 1. Motivação

Sistemas de reconhecimento facial **falham desigualmente entre grupos
demográficos**:

- Buolamwini & Gebru (2018) — Gender Shades: 34.7 pp de gap entre
  mulheres negras e homens brancos em comerciais.
- Kotwal & Marcel (2025) — gaps demográficos persistem em sistemas
  modernos.

**E agora a regulação chegou:**

- **EU AI Act** (em vigor): reconhecimento facial = alto risco, exige
  evidência empírica de fairness antes do deploy.
- **NIST FRTE 2024**: protocolos formais de auditoria demográfica.

→ **Como auditar e mitigar viés de forma reprodutível?**

---

## 2. Origem — dissertação de MBA (USP, 2024)

**Tema:** mitigação de viés racial em reconhecimento facial usando
**undersampling**.

**Stack:**
- Backbone: **ResNet50** (pré-treinada ImageNet)
- Head: `Linear(2048 → 7)` (cabeçote ingênuo)
- Dataset: **FairFace** (108k imagens, 7 raças)
- Matriz: **11 experimentos** variando loss, optimizer, scheduler,
  filtro de classes.

**Conclusão da época:** undersampling apresenta variabilidade; F1 ≈ 0.67
no problema de 7 raças.

---

## 3. Revisão crítica do código MBA (2026-04)

Após auditoria sistemática, **14 bugs identificados**. O mais grave:

> **§2.2 (Crítico):** `ArcFaceLoss.forward()` retornava sempre
> `F.cross_entropy(...)` — **todos os experimentos "ArcFace" do MBA eram
> Cross-Entropy puro**.

Outros: normalização errada (§2.3), chaves inconsistentes (§2.4),
landmarks absolutos (§2.14), antipattern `sys.path.append` (§2.7), etc.

**4 sprints de refatoração** (A: empacotamento, B: bug fixes, C: trainer
+ evaluator, D: testes) → **94 testes verdes + pipeline auditável**.

---

## 4. Re-execução dos 11 experimentos do MBA (2026-05-08)

Re-rodados com bugs corrigidos. Wall-clock total: **9h 38min**.

| Família | MBA reportou | Clean run | Δ |
|---|---|---|---|
| Cross-Entropy (Exp 1, 3, **5**, 9, 11) | 0.62–0.68 | 0.61–0.67 | ~0 |
| ArcFace (Exp 2, 4, 10) | 0.58–0.61 | **0.17–0.46** | **−0.34** |
| ArcFace + Cosine (Exp 6) | (zeros) | 0.555 | recupera |
| Black/White binário (Exp 7, 8) | 0.94–0.95 | 0.92–0.93 | −0.02 |

**O recipe vencedor (Exp 5) — CE + AdamW + Cosine — é o baseline do
mestrado:** F1=0.665, IR=1.76.

---

## 5. Reunião de kickoff com o orientador (2026-05-11)

**8 diretrizes** estabelecidas. Recorte das principais:

| # | Diretriz | Status atual |
|---|---|---|
| 0 | A **tese precisa ser positiva** (provar algo que funciona) | Adotado |
| 2 | Adicionar **MLP head** ao classificador (sair do linear) | ✓ Feito |
| 3 | Usar **Optuna** como critério metodológico | ✓ Feito 2× |
| 4 | **Limpar imagens com mais de 1 face** (MTCNN) | ✓ Feito |
| 6 | Variação de loss aprovada (ArcFace, AdaFace, MagFace) | Pendente |
| 5 | Aprendizado contrastivo (SimCLR, Sup-SimCLR, CLIP) | Pendente |

---

## 6. Reframing positivo da tese

**Antes:** "Vamos mostrar que undersampling NÃO funciona em N condições."

❌ Vulnerável: a banca pode citar 1 paper que mostra undersampling
funcionando e derruba a tese.

**Depois:** "Vamos demonstrar empiricamente uma topologia de
classificador + preprocessamento que **melhoram simultaneamente F1 e
fairness** sobre a baseline."

✓ Defensável: o claim é positivo, baseado em evidência empírica, com
artefatos reprodutíveis.

---

## 7. Arquitetura técnica adotada

```
Imagem 224×224
     ↓
┌─────────────────────────┐
│   ResNet50 backbone     │ ← pré-treinada ImageNet
│   (~23M parâmetros)     │   fc original substituída
└─────────────────────────┘
     ↓ embedding (2048-d)
┌─────────────────────────┐
│   Dropout(p=0.2)        │
└─────────────────────────┘
     ↓
┌─────────────────────────┐
│   MLPHead configurável  │ ← otimizado por Optuna
│   (depth, width, act,   │
│    norm, dropout)       │
└─────────────────────────┘
     ↓
Logits (7 raças)
```

---

## 8. O espaço de busca de Optuna

Optuna otimiza **apenas o cabeçote** (backbone e training recipe
congelados). 6 dimensões:

| Variável | Valores | Tipo |
|---|---|---|
| `depth` | 1, 2 ou 3 | int |
| `hidden_dims[i]` | {128, 256, 512, 1024, 2048} | categórico |
| `activation` | relu, gelu, silu | categórico |
| `dropout` | [0, 0.6] | uniforme |
| `norm` | none, batchnorm, layernorm | categórico |
| `learning_rate` | [1e-4, 5e-3] | log-uniforme |

**Multi-objetivo:** maximizar F1 macro **E** minimizar Inequity Rate.
**Sampler:** TPE multivariate (Akiba et al., 2019).

---

## 9. Design experimental rigoroso

3 pontos de referência **independentes**:

| Ponto | Dataset | Head | Para que serve |
|---|---|---|---|
| **R1** | original 97k | Linear | Baseline MBA |
| **R2** | clean 72k (n_faces=1) | Linear | Isola **efeito da limpeza** |
| **R3** | clean 72k | MLP (otimizado) | Isola **efeito da topologia** + combinação |

→ **3 claims positivos independentes** na qualificação:

1. Limpeza melhora ArcFace.
2. Topologia MLP melhora F1 e IR.
3. Pipeline integrado é melhor que MBA original.

---

## 10. Auditoria multi-face (diretriz 4)

MTCNN aplicado sobre as 97 698 imagens originais. Distribuição:

| n_faces | count | % |
|---|---|---|
| 0 (não detectou) | 418 | 0.43% |
| **1 (limpo)** | **72 749** | **74.46%** |
| 2 | 18 675 | 19.12% |
| 3+ | 5 856 | 6.00% |

**E NÃO é uniforme entre raças:**

- Black: 30.0% multi-face
- East Asian: 20.7% multi-face
- **Spread: 9.3 pp**

→ Achado relevante por si só: viés de cena correlacionado com raça.

---

## 11. Resultado 1 — Fator Dataset (3-seed, ambiente único)

Batch casado de 12 execuções (r1ctrl orig × r2base clean × CE/ArcFace
× 3 seeds), ambiente idêntico, média ± desvio.

| Recipe | Δ F1 (clean−orig) | Δ IR (clean−orig) |
|---|---|---|
| **CE** | **+1,35 pp** (significativo, |Δ|>σ) | +0,05 (dentro de 1σ → não-signif.) |
| **ArcFace** | −3,2 pp (não-signif.) | −0,006 (não-signif.; σ_IR=±1,24) |

**Interpretação:** a limpeza do dataset melhora **acurácia no recipe
CE** e **não tem efeito significativo sobre a disparidade (IR)** em
nenhum recipe. ArcFace tem variância de seed alta (σ_IR ±1,24) — sua
instabilidade é intrínseca ao recipe.

---

## 12. Por que 2 experimentos e não 11?

**Pergunta da banca:** "Por que não rodou os 11 no dataset limpo?"

**Resposta:** porque dataset cleaning é um **passo de preprocessamento
determinístico**, não um hiperparâmetro otimizado.

- A tese fixa optimizer (AdamW), scheduler (Cosine), dropout (0.2),
  baseado nos resultados do clean run.
- A tese **só varia 2 dimensões downstream**: loss family e head topology.
- Logo, **basta testar a limpeza nas 2 famílias de loss** (CE = Exp 5,
  ArcFace = Exp 6).

→ Design mínimo suficiente: 2 experimentos, ~2h GPU, defensível.

---

## 13. Resultado 2 — HPO da topologia (R1 vs R2)

20 trials × 8 epochs em cada round, mesmo espaço de busca, seed=42.

**Frentes de Pareto:**

| | R1 Pareto (4 trials) | R2 Pareto (2 trials) |
|---|---|---|
| Best F1 | 0.6817 | **0.6935** |
| Best IR | **1.529** | 1.591 |

**O melhor F1 do R2 (0.6935) supera todos os 20 trials do R1.**

R2 best IR (1.591) é levemente pior que R1 (1.529) — mas R1 IR=1.529 era
artefato de "epoch sortuda" com label ruidoso; no dataset limpo, a
convergência é mais suave.

---

## 14. Resultado 2 — Topologias vencedoras (R2)

| Trial | F1↑ | IR↓ | Topologia |
|---|---:|---:|---|
| **4** | **0.6935** | 1.638 | `[256] GELU drop=0.52 norm=none` |
| **10** | 0.6886 | **1.591** | `[1024, 1024, 2048] SiLU drop=0.087 layernorm` (depth=3!) |

**Padrões descobertos pelo TPE:**

- GELU > ReLU > SiLU para shallow heads.
- Width 256 ainda é vencedora.
- **Topologias profundas voltam a aparecer no dataset limpo** (R1 só
  tinha depth=1 no Pareto).

→ Limpeza permite arquiteturas mais expressivas sem penalidade.

---

## 15. Critério metodológico — Pareto-aware best epoch

**Problema descoberto no Round 1:** o script reportava `max F1 do trial`
+ IR daquela epoch. Quando F1 sobe enquanto IR piora, perde-se o ponto
Pareto-ótimo.

**Solução implementada:**

1. Tracking de todas as 8 epochs do trial.
2. Pareto-local: epochs não-dominadas internamente.
3. Tie-break: **menor IR** (fairness-favoring).

**Resultado:** 4 trials adicionais entraram no Pareto reanalisado do R1.

→ Script: `scripts/reanalyze_hpo_round1.py` (reusável, não toca o SQLite).

---

## 16. Resultado 3 — Mapa de atribuição (defensável, 3-seed)

Contribuição marginal por fator (2 de 5 medidos, protocolo controlado):

| Fator | → Acurácia (F1) | → **Fairness (IR)** |
|---|---|---|
| **1. Dataset** (limpeza multi-face) | +1,35pp CE (signif.) | **nula** (não-signif.) |
| **2. Topologia** (linear→MLP `[256] GELU`) | +0,8pp (n.s.) | **−0,11 (SIGNIF.)** |
| 3-5 (loss/contrastivo/backbone) | a medir | a medir |

→ **Tese: a alavanca de equidade defensável é a TOPOLOGIA do
classificador; a limpeza do dataset contribui para acurácia.** Cada
fator paga em um eixo distinto — o resultado que a metodologia de
atribuição (Linha A) existe para produzir.

---

## 17. Eficiência do pipeline

Otimizações aplicadas antes do R2:

| Win | Otimização | Speedup observado |
|---|---|---|
| #1 | AMP (fp16 autocast + GradScaler) | ~2× |
| #2 | `persistent_workers` + `prefetch_factor=4` | ~10% |
| #4 | Pré-encode labels (sklearn LabelEncoder uma vez) | ~3% |
| #5 | `PIL.Image.open` direto (substitui cv2) | ~10% |
| **Total combinado** | — | **~2.7×** |

**HPO Round 1 (fp32):** 12h. **HPO Round 2 (AMP + clean):** 3h 27min.

→ 3.5× mais rápido, sem reduzir qualidade do estudo.

---

## 18. Engenharia e qualidade do pipeline

- **94 testes** verdes (unit + integration + smoke + GPU).
- **Pydantic** para validação de configs YAML.
- **MLflow** para tracking de experimentos.
- **DVC** preparado para versionamento de dados.
- **24 CVEs fechadas** em 2 ondas de patches Dependabot.
- **Pré-commit** com detect-secrets, ruff, black.
- **`scripts/run_all_experiments.py`** orquestra subprocess train + eval
  com persistência de resultados (resumível).
- **`scripts/hpo_head.py`** com Optuna SQLite-backed (resumível).

→ Pipeline reproduzível, auditável, defensível em banca.

---

## 19. Resumo dos artefatos produzidos

**Código (~7k LoC novas):**
- `src/face_bias/` — pacote instalável
- `scripts/` — orquestradores reusáveis
- `tests/` — 94 testes

**Documentação (em `docs/`):**
- [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md) — proposta original
- [HISTORICO.md](HISTORICO.md) — este histórico cronológico
- [GLOSSARIO.md](GLOSSARIO.md) — glossário de termos
- [clean_results.md](clean_results.md) — 11 experimentos MBA reproduzidos
- [hpo_round1_results.md](hpo_round1_results.md) — HPO original
- [r2_clean_dataset_results.md](r2_clean_dataset_results.md) — efeito da limpeza
- [hpo_round2_results.md](hpo_round2_results.md) — HPO no dataset limpo
- [security_audit.md](security_audit.md) — trilha de CVEs
- [meeting_2026-05-11_kickoff.md](meeting_2026-05-11_kickoff.md) — reunião kickoff

---

## 20. Próximos passos antes da qualificação

**Fase 4 (imediata, ~50 min):** refit dos 2 vencedores do Pareto R2 em
**25 epochs** para confirmar que os ganhos sobrevivem ao orçamento
completo.

**Semanas 4-6:** SOTA review (diretriz 1) — comparar nossos números com
o estado da arte 2024-2026.

**Semanas 7-10:** experimentos com **contrastive learning** (diretriz 5):
SimCLR, Sup-SimCLR, CLIP como axis alternativo ao classification head.

**Semanas 11-13:** variação de loss (diretriz 6): AdaFace, MagFace.

**Semanas 14-15:** consolidação dos resultados + escrita da qualificação.

---

## 21. Contribuições científicas defendíveis

1. **Metodológica:** primeira aplicação documentada do critério
   **Pareto-aware best epoch** em HPO multi-objetivo de fairness —
   descoberta empírica do problema "best by F1" durante o Round 1.

2. **Empírica (dataset):** sob protocolo controlado (3-seed, ambiente
   único), a limpeza de imagens multi-face contribui para **acurácia
   (+1,35pp CE, significativo)** e **não** para a equidade.

3. **Empírica (topologia):** vs o baseline linear de 3 seeds, a
   topologia MLP `[256] GELU drop=0,52` **reduz o IR em −0,11 de forma
   estatisticamente significativa** — a **alavanca de equidade
   defensável** é a topologia do classificador.

4. **Engenharia:** pipeline auditável, reproduzível, instrumentado com
   testes — material para um paper de "ML systems for fairness audit".

---

## 22. Limitações conhecidas

- **Apenas FairFace.** Generalização para outros datasets (BUPT, RFW)
  ainda não testada.
- **Apenas 7 classes raciais.** Atributos protegidos como gênero e
  idade não são alvo desta fase.
- **Apenas ResNet50.** Backbones modernos (ViT, ConvNeXt) ficam para
  trabalho futuro.
- **Apenas Inequity Rate.** Outras métricas (Demographic Parity,
  Equalized Odds) reportadas mas não otimizadas diretamente.
- **ArcFace ainda instável** mesmo no dataset limpo — fora do escopo
  desta tese.

---

## 23. Plano de publicação

| Foro | Material | Prazo alvo |
|---|---|---|
| Qualificação (Unifesp) | Dissertação parcial | **2026-08-24** |
| TDC SP 2026 | Palestra: "MLOps para Responsible AI" (Eng IA) ou "Auditoria de viés com HPO multi-objetivo" (Governança) | Submissão CFP aberta |
| Workshop ICCV / ECCV | Paper curto: "Pareto-aware HPO for Demographic Fairness" | 2026 H2 |
| Defesa final | Dissertação completa | ~2027 Q1-Q2 |

---

## 24. Q&A — respostas para perguntas previsíveis da banca

**"Por que não testou outros backbones?"**
→ Backbone está fixo por design — a tese estuda o head. Trabalho futuro.

**"E se ArcFace funcionar com warmup?"**
→ Não testado nesta fase. ArcFace é apresentado como recipe **alternativo
e instável**, não como recipe principal.

**"O speedup do AMP não compromete os resultados?"**
→ Não. Validamos com sanity check: trajetória qualitativamente idêntica,
ruído numérico de ~0.5 pp por epoch (aceitável para HPO).

**"O critério Pareto-aware é arbitrário?"**
→ Não — é o **mínimo necessário** para multi-objetivo. O critério "best
F1" era o erro implícito.

---

## 25. Obrigado

**Materiais e código:**
[github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias](https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias)

**Contato:** marcello.ozzetti@gmail.com

---

<!-- Notas para conversão para slides reais (Marp, Reveal.js, PowerPoint):

1. Cada `---` marca o fim de um slide.
2. Para gerar PDF/HTML com Marp:
   `npx @marp-team/marp-cli@latest docs/slides_qualificacao.md -o slides.pdf`
3. Para Reveal.js: usar Pandoc com `--to revealjs`.
4. Para PowerPoint: usar Pandoc com `--to pptx`.

Convenções de notação visual sugeridas para o slide deck final:

- Cor primária: azul ICT/Unifesp (#003D7A) ou similar institucional.
- Cor de destaque: laranja (#E07B00) para deltas positivos.
- Cor de alerta: vermelho (#C8102E) para bugs e CVEs.
- Cor de fundo das tabelas: cinza claro (#F4F4F4).
- Fonte: sans-serif moderna (Inter, IBM Plex Sans, Helvetica).
- Code blocks: usar fonte mono (Fira Code, JetBrains Mono) com syntax highlight.

Slides com tabelas grandes (4, 11, 13, 16) podem ser quebrados em 2 slides
se necessário ao apresentar.

Numeração de seções (1-25) é só para navegação aqui no .md — não imprimir
nos slides finais. -->
