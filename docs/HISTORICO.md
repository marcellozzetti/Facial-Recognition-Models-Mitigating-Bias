# Histórico do Projeto — MBA (USP) → Mestrado (Unifesp/ICT)

Este documento é a linha do tempo cronológica de tudo que foi feito desde a
dissertação de MBA até o estado atual do trabalho de mestrado. Cada marco
referencia o(s) documento(s) técnico(s) detalhado(s) correspondente(s),
permitindo navegar do panorama → detalhe sem perder contexto.

**Aluno:** Marcello Ozzetti
**Orientador (mestrado):** Prof. (a definir no documento — `[nome do orientador]`)
**Programa:** Mestrado em Ciência da Computação, Unifesp/ICT
**Origem:** Dissertação de MBA em IA, USP (2024)
**Repositório:** https://github.com/marcellozzetti/Facial-Recognition-Models-Mitigating-Bias

---

## Objetivo do trabalho (TRAVADO em 2026-05-15 — gate G0)

> Propor uma metodologia de **atribuição causal de disparidade
> demográfica** em reconhecimento facial, baseada em decomposição
> experimental controlada e em otimização multi-objetivo Pareto-aware,
> que quantifica a contribuição marginal de cada fator — integridade do
> dataset, topologia do classificador, função de perda, paradigma de
> aprendizado e backbone — para a (in)justiça do sistema, respondendo
> não *se* a IA pode ser mais justa, mas **onde intervir** para
> torná-la justa.

Motivador (herdado do MBA): *como tornar a IA mais justa*. O reframing
para **atribuição causal** (em vez de "mais uma mitigação") é o
diferencial validado contra 555 papers (ver
[sota_review.md](sota_review.md) §5.3 e
[literature_semantic_audit.md](literature_semantic_audit.md) §5).
Decisão registrada como Linha A em [PLANO_TRABALHO.md](PLANO_TRABALHO.md) §1.

---

## Linha do tempo resumida

| Período | Marco | Documentação |
|---|---|---|
| 2024 | Dissertação de MBA concluída (USP) | `notebooks/MBA_IA_USP_marcello_ozzetti.ipynb` |
| 2026-04 | Avaliação completa dos 5 capítulos da dissertação MBA | — |
| 2026-04 | Pesquisa de literatura 2024-2026 (estado da arte) | — |
| 2026-04 | Criação da `PROPOSTA_MESTRADO.md` | [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md) |
| 2026-04 | Sprint A: empacotamento, src layout, CLIs, schemas Pydantic | — |
| 2026-04 | Sprint B: correção de 14 bugs identificados na revisão | `REVIEW_AND_PLAN.md` |
| 2026-04 | Sprint C: trainer, evaluator, fairness audit, MLflow, DVC, callbacks | — |
| 2026-04 | Sprint D: testes (unit + integration + smoke + gpu) | — |
| 2026-05-08 | **Clean run dos 11 experimentos do MBA reproduzidos** (9h 38min) | [clean_results.md](clean_results.md) |
| 2026-05-09 | Documento do exp01 vs MBA original (diagnóstico) | [exp01_vs_mba.md](exp01_vs_mba.md) |
| 2026-05-11 | **Reunião de kickoff com o orientador** | [meeting_2026-05-11_kickoff.md](meeting_2026-05-11_kickoff.md) |
| 2026-05-11 | Wave 1 de patches de segurança: 20/21 CVEs fechadas | [security_audit.md](security_audit.md) |
| 2026-05-12 | Implementação do `MLPHead` + integração no `LResNet50E_IR` | — |
| 2026-05-13 | Script `scripts/hpo_head.py` com Optuna multi-objetivo | [hpo_round1_results.md](hpo_round1_results.md) |
| 2026-05-13/14 | **HPO Round 1** rodado (20 trials × 8 epochs, dataset original, fp32) | [hpo_round1_results.md](hpo_round1_results.md) |
| 2026-05-14 | Wave 2 de patches: torch 2.5→2.12 (4 CVEs PyTorch fechadas) | [security_audit.md](security_audit.md) |
| 2026-05-14 | Wins de eficiência aplicados (AMP, persistent_workers, PIL, pre-encode) | — |
| 2026-05-14 | Auditoria multi-face (MTCNN sobre 97k imagens) | `outputs/audit/multi_face_audit_summary.md` |
| 2026-05-14 | Decisão: Opção A (excluir todas as imagens com >1 face) | [r2_clean_dataset_results.md](r2_clean_dataset_results.md) |
| 2026-05-14 | **R2: Exp 5 + Exp 6 rodados no dataset limpo** (2h) | [r2_clean_dataset_results.md](r2_clean_dataset_results.md) |
| 2026-05-14 | **HPO Round 2** rodado (20 trials × 8 epochs, dataset limpo, AMP) | [hpo_round2_results.md](hpo_round2_results.md) |
| **Próximo** | Fase 4: refit dos vencedores do Pareto R2 em 25 epochs | (a fazer) |
| 2026-08-24 | **Qualificação alvo** | — |

---

## 1. Origem — MBA em IA / USP (2024)

A dissertação de MBA tinha como tema **mitigação de viés racial em
reconhecimento facial**. A abordagem adotada na época foi:

- **Backbone:** ResNet50 pré-treinada no ImageNet, com a `fc` (1000 classes
  ImageNet) substituída por `Linear(2048, num_classes)`.
- **Dataset:** FairFace (108k imagens, 7 raças, com anotações de gênero,
  idade, raça).
- **Estratégia de mitigação:** **undersampling** — balancear o dataset
  reduzindo todas as classes ao tamanho da minoria.
- **Matriz experimental:** 11 experimentos variando 4 fatores: função de
  perda (Cross-Entropy vs ArcFace), otimizador (SGD vs AdamW), scheduler
  (OneCycleLR vs Cosine), e filtro de classes (todas 7 raças vs binário
  Black/White), com variações adicionais de dropout e epochs.
- **Métricas:** acurácia, F1 macro, e métricas de fairness por classe.

A dissertação reportou que o undersampling apresentou variabilidade nos
resultados entre experimentos, com algumas configurações alcançando F1 ≈
0.67 no problema de 7 raças.

---

## 2. Sprint inicial (abril 2026) — refatoração e revisão crítica

Antes de avançar para o mestrado, foi feita uma auditoria completa do
código do MBA. **14 bugs foram identificados** (`REVIEW_AND_PLAN.md`), dos
quais os mais críticos:

- **§2.2 (Crítico):** `ArcFaceLoss.forward()` retornava sempre
  `F.cross_entropy(logits, labels)` — ou seja, **todos os experimentos
  "ArcFace" do MBA eram, na prática, Cross-Entropy puro**. A margem
  angular nunca foi aplicada.
- **§2.3:** `image_std` usava `image_mean` duplicado na normalização.
- **§2.4:** Chave `input_size` vs `image_size` inconsistente entre módulos.
- **§2.5:** `class_to_idx` não existia no `setup_dataset`.
- **§2.7:** `sys.path.append` antipattern; código não era um pacote
  instalável.
- **§2.12:** `cropping_procedure` produzia índices negativos sem clamping.
- **§2.14:** `alignment_procedure` aplicava landmarks absolutos.

**Sprints A-D** (~1 mês de trabalho) endereçaram tudo:

- **Sprint A (empacotamento):** `src/face_bias/` instalável via
  `pyproject.toml`, CLIs (`face-bias-train`, `face-bias-evaluate`,
  `face-bias-tsne`, `face-bias-gradcam`, etc.), schemas Pydantic para
  validação de configs YAML.
- **Sprint B (correções):** todos os bugs §2.2–§2.14 fechados, com testes
  de regressão (`tests/smoke/test_arcface_integration.py`).
- **Sprint C (training & evaluation):** classe `Trainer` com MLflow opcional,
  checkpoint de melhor modelo + último modelo, early stopping,
  gradient clipping; `Evaluator` com auditoria de fairness completa
  (Inequity Rate, max-min disparity, Gini, FDR, CEI por classe).
- **Sprint D (testes):** 94 testes unitários + integração + smoke + GPU,
  estruturados com pytest markers.

---

## 3. Clean run dos 11 experimentos do MBA (2026-05-08)

Com o pipeline corrigido, **os 11 experimentos do MBA foram re-rodados do
zero** em condições idênticas mas **com os bugs corrigidos**.

**Resultado:** [clean_results.md](clean_results.md).

Principais achados:

- A família **Cross-Entropy reproduz bem** os números do MBA (deltas ≈ 0).
- A família **ArcFace colapsa** — agora que a margem é de fato aplicada,
  com `m=0.5` e LR padrão, pelo menos uma classe demográfica nunca é
  predita (IR=∞, F1=0). Confirma empiricamente que os "ganhos de ArcFace"
  reportados no MBA eram artefato do bug §2.2.
- **Exp 5 (CE + AdamW + Cosine, dropout=0.2)** emerge como o **melhor
  recipe**: F1=0.665, IR=1.76 — o ponto de referência para todo o resto do
  trabalho de mestrado.

---

## 4. Proposta do mestrado (2026-04-05)

A `PROPOSTA_MESTRADO.md` foi escrita para apresentar ao coordenador e ao
orientador. Estrutura:

1. Motivação (EU AI Act, NIST FRTE, viés algorítmico)
2. Trabalho do MBA como ponto de partida
3. Hipóteses H1-H5 com framing positivo
4. Plano de trabalho semestral (6 meses, dezembro 2026)
5. Parte III.5: plano de publicações

Vale notar que após a reunião de kickoff (próxima seção), a proposta passou
por reframing positivo significativo — passou de "undersampling não basta,
precisamos mostrar limites" para "vamos demonstrar empiricamente método X
que funciona".

Ver também:

- [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md)
- [meeting_prep_2026-05-11.md](meeting_prep_2026-05-11.md) — preparação para
  reunião de kickoff

---

## 5. Reunião de kickoff com o orientador (2026-05-11)

Marco mais importante até hoje. **8 diretrizes** foram definidas pelo
orientador, registradas integralmente em
[meeting_2026-05-11_kickoff.md](meeting_2026-05-11_kickoff.md):

| # | Diretriz | Status |
|---|---|---|
| 0 | **A tese precisa ser positiva** (demonstrar algo que funciona, não algo que falha) | Adotado, framing reescrito |
| 1 | Definir o TOP Line / SOTA de reconhecimento facial atual | (pendente — Semana 1) |
| 2 | Adicionar camadas densas (MLP head) ao classificador, sair do linear puro | **Implementado e testado em R1 + R2** |
| 3 | Usar **Optuna** como critério metodológico de busca de topologia | **Implementado e rodado 2 vezes** |
| 4 | **Limpar imagens com mais de uma face** via MTCNN | **Auditado, decisão Opção A tomada, R2 rodado** |
| 5 | Avaliar aprendizado contrastivo (SimCLR, Sup-SimCLR, CLIP) | (pendente — Round 3+) |
| 6 | Variação de loss aprovada (ArcFace, AdaFace, MagFace, KP-RPE) | (pendente — Round 3+) |
| 7 | Toda documentação fica na pasta `docs/` | Adotado |
| 8 | Pesquisa semântica sobre as ~900 referências bibliográficas do MBA | (pendente — quando corpus chegar) |

Cronograma: **qualificação em 2026-08-24** (15 semanas a partir do kickoff,
com reuniões semanais às segundas-feiras).

---

## 6. Implementação do MLP head + Optuna (diretrizes 2 e 3)

**`MLPHead` (`src/face_bias/models/mlp_head.py`):** cabeçote totalmente
configurável (profundidade, largura por camada, ativação, normalização,
dropout). Integrado ao `LResNet50E_IR` via `head="mlp"` — **sem quebrar**
os caminhos `head="linear"` (baseline MBA) e `head="arcface"`. Todos os 11
YAMLs do MBA continuam carregando sem alteração.

**`scripts/hpo_head.py`:** estudo Optuna multi-objetivo (maximizar F1 macro,
minimizar Inequity Rate) sobre a topologia do MLP head + learning rate.

- Sampler: **TPE multivariate** com seed=42 (Akiba et al. 2019).
- Sem pruning (não suportado em multi-objetivo no Optuna).
- Storage: SQLite local, estudo resumível.
- Critério de "best epoch" do trial: **Pareto-local não-dominado**,
  tie-break pela menor IR (fairness-favoring).

---

## 7. HPO Round 1 (2026-05-13/14)

**Configuração:**
- Base: `configs/experiments/exp05_ce_adamw_cosine.yaml` (dataset original
  97k, fp32, sem AMP).
- 20 trials × 8 epochs (orçamento curto; refit final em 25 epochs).
- Hardware: RTX 4070 SUPER 12 GB.

**Custo:** ~12 h de wall clock (uma reinicialização no meio, retomada
via SQLite).

**Resultado** (Pareto-aware reanalysis):

| Trial | F1↑ | IR↓ | Topologia |
|---|---:|---:|---|
| 4 | 0.6817 | 1.564 | `[256] GELU drop=0.52 norm=none` |
| 8 | 0.6808 | 1.544 | `[256, 1024, 512] GELU drop=0.40 norm=none` |
| 12 | 0.6617 | 1.529 | `[256] GELU drop=0.27 norm=none` |
| 13 | 0.6730 | 1.539 | `[256] SiLU drop=0.42 norm=none` |

Detalhes completos em [hpo_round1_results.md](hpo_round1_results.md).

**Achado metodológico importante:** o critério inicial "best epoch =
maior F1 do trial" descartava silenciosamente epochs onde o IR era muito
menor com pequeno custo em F1. O **critério Pareto-aware** (não-dominado
localmente, tie-break por menor IR) foi implementado no
`scripts/reanalyze_hpo_round1.py` (post-hoc, sem mexer no SQLite original)
e depois embutido no próprio `hpo_head.py` para rounds futuros.

---

## 8. Wins de eficiência (2026-05-14)

Antes do Round 2, **4 otimizações de pipeline** foram aplicadas para
reduzir tempo de epoch sem alterar resultado em fp32 (e ~3× em AMP):

| Win | O que muda | Estado |
|---|---|---|
| #1 | **AMP** (autocast fp16 + GradScaler) | Sempre ON no HPO; opt-in em `training.use_amp` no Trainer |
| #2 | `persistent_workers=True` + `prefetch_factor=4` | Sempre ON quando `num_workers > 0` |
| #4 | Pré-encodar labels uma vez no `setup_dataset` (sklearn LabelEncoder fora do `__getitem__`) | Sempre ON |
| #5 | `PIL.Image.open(path).convert("RGB")` direto (substitui cv2→fromarray) | Sempre ON |

Resultado empírico no sanity check: **~2.7× speedup** combinado em fp16.
Em fp32 (para apples-to-apples com R1), só wins #2/#4/#5 atuam → ~10%
mais rápido.

---

## 9. Auditoria multi-face (2026-05-14, diretriz 4)

**`scripts/audit_multi_face.py`:** roda MTCNN sobre as 97 698 imagens
originais (não as alinhadas) e conta quantas faces cada uma tem.

Custo: ~45 min na 4070 SUPER.

Resultado em `outputs/audit/multi_face_audit_summary.md`:

| n_faces | count | % do total |
|---|---|---|
| 0 (MTCNN não detectou) | 418 | 0.43% |
| **1 (caso "limpo")** | **72 749** | **74.46%** |
| 2 | 18 675 | 19.12% |
| 3+ | 5 856 | 6.00% |

A distribuição de imagens multi-face **não é uniforme por raça**:

- Black: 30.0% multi-face
- East Asian: 20.7% multi-face
- Spread: 9.3 pp

Isso é um achado relevante por si só — sugere que a coleta original do
FairFace tem **viés de cena** correlacionado com raça (foto de grupo,
contexto, ambiente).

---

## 10. Decisão de dataset — Opção A (2026-05-14)

Após análise das 4 opções (manter como está, excluir n_faces>2, excluir
n_faces≥1, verificar correspondência label-vs-face escolhida), foi adotada
a **Opção A: excluir todas as imagens com n_faces ≠ 1**.

Justificativa científica:

- Dataset com 1 face por imagem ⇒ **rótulo unívoco**.
- Remove ambiguidade sistemática.
- Custo: −25.54% das amostras → minoria pós-filtro = Middle Eastern com
  8 011 imagens → dataset balanceado de 56 077 imagens.

Implementação:

- `scripts/filter_dataset_clean.py` (idempotente, junta `audit.csv` com
  `fairface_labels.csv`).
- Output: `data/raw/fairface/fairface_labels_clean.csv`.

---

## 11. R2 — Efeito isolado da limpeza (2026-05-14)

**Pergunta:** a limpeza prejudica ou ajuda o recipe que vamos usar
downstream?

**Design experimental defendido em banca:** 2 experimentos cobrindo as
**duas famílias de loss** (CE = softmax-based, ArcFace = margin-based) —
não 11 experimentos, porque dataset cleaning é um passo de preprocessamento
determinístico, não um hiperparâmetro que precisa de varredura completa.

Configs:
- `configs/experiments_clean/exp05_ce_adamw_cosine.yaml`
- `configs/experiments_clean/exp06_arcface_adamw_cosine.yaml`

Wall clock: 2h 3min total (Exp 5 = 54 min, Exp 6 = 69 min, ambos em fp32).

**Resultado** ([r2_clean_dataset_results.md](r2_clean_dataset_results.md)):

| | R1 (original) | R2 (clean) | Δ F1 | Δ IR |
|---|---|---|---:|---:|
| Exp 5 (CE) | F1=0.665, IR=1.76 | F1=0.668, IR=1.737 | +0.3 pp | −1.3% |
| Exp 6 (ArcFace) | F1=0.529, IR=4.60 | F1=0.587, IR=2.114 | **+5.8 pp** | **−54%** |

**Conclusão:** a limpeza tem **efeito quase nulo em CE** (não prejudica) e
**efeito grande em ArcFace** (estabiliza). É um claim positivo defendável
e que dá segurança para usar o dataset limpo na busca de topologia.

---

## 12. HPO Round 2 (2026-05-14)

Mesma busca Optuna do Round 1, mas com base config = Exp 5 no dataset
limpo + AMP ON.

**Custo:** **3h 27min** (vs 12h+ no Round 1 — 3.5× speedup via AMP +
dataset menor + wins de eficiência).

**Resultado** ([hpo_round2_results.md](hpo_round2_results.md)):

Frente de Pareto = 2 trials.

| Trial | F1↑ | IR↓ | Topologia |
|---|---:|---:|---|
| 4 | **0.6935** | 1.638 | `[256] GELU drop=0.52 norm=none` |
| 10 | 0.6886 | **1.591** | `[1024, 1024, 2048] SiLU drop=0.087 layernorm` (depth=3!) |

**Decomposição dos ganhos** vs MBA Exp 5 baseline (F1=0.665, IR=1.76):

| Contribuinte | Δ F1 macro | Δ IR |
|---|---:|---:|
| Limpeza sozinha (R1→R2 baseline) | +0.3 pp | −1.3% |
| Topologia sozinha (R1 base → R1 HPO) | +1.7 pp | −11.1% |
| **Combinação (R1 base → R2 HPO #4)** | **+2.85 pp** | **−6.9%** |
| **Combinação (R1 base → R2 HPO #10)** | +2.36 pp | **−9.6%** |

**Implicação para a tese — três claims positivos independentes:**

1. "Limpeza melhora margin-based losses substancialmente (+5.8 pp em F1
   no ArcFace)" — claim sobre preprocessamento.
2. "Topologia MLP escolhida via Optuna multi-objetivo domina o baseline
   linear na frente de Pareto F1×IR" — claim sobre modelagem.
3. "Pipeline integrado (limpeza + MLP) reduz IR em 9.6% mantendo F1 +2.36
   pp sobre o baseline MBA original" — claim sobre o sistema todo.

---

## 13. Segurança e CVEs (Wave 1 + Wave 2)

Em paralelo ao trabalho técnico, **24 CVEs foram fechadas** em 2 ondas:

- **Wave 1 (2026-05-11):** 20 das 21 reportadas pelo Dependabot — bump
  Pillow, urllib3, idna, Jinja2, fonttools, filelock, pip, setuptools.
- **Wave 2 (2026-05-14):** 3 das 4 novas — torch 2.5.1+cu121 → 2.12.0+cu126
  (fecha CVE-2025-32434 Critical RCE, CVE-2026-24747 High RCE, GHSA-2rj9
  Moderate libuv).

Residual aceito: **diskcache 5.6.3** (CVE-2025-69872) sem patch upstream
disponível, com mitigações documentadas. Ver
[security_audit.md](security_audit.md).

---

## 14. Próximo passo — Fase 4: refit em 25 epochs

Os 2 vencedores do Pareto R2 (trials 4 e 10) ainda foram avaliados em
apenas 8 epochs. O **refit em 25 epochs** confirma se os ganhos
permanecem com o orçamento completo de treino.

Configs a gerar:

- `configs/experiments_clean/r4_trial4_refit.yaml`
- `configs/experiments_clean/r4_trial10_refit.yaml`

Custo estimado: ~50 min com AMP (1 seed), ~2h 30min (3 seeds para
intervalo de confiança).

Após Fase 4, todos os números estão fechados para a qualificação.

---

## 15. Próximas semanas (até 2026-08-24)

Pendências da diretriz original que ainda não foram tocadas:

- **Diretriz 1 (TOP Line / SOTA):** revisão de literatura recente (2024-2026)
  sobre reconhecimento facial; comparar números do nosso pipeline com
  SOTA documentado.
- **Diretriz 5 (aprendizado contrastivo):** experimentar SimCLR /
  Sup-SimCLR / CLIP como axis alternativo ao classification head.
- **Diretriz 6 (variação de loss):** AdaFace, MagFace, KP-RPE.
- **Diretriz 8 (pesquisa semântica):** rodar
  `scripts/semantic_search_corpus.py` quando o corpus de ~900 referências
  estiver disponível.

Cronograma proposto:

- **Semanas 4-6:** SOTA review + dataset audit estendido.
- **Semanas 7-10:** experimentos com contrastive learning (diretriz 5).
- **Semanas 11-13:** experimentos com losses alternativas (diretriz 6).
- **Semanas 14-15:** consolidação dos resultados + escrita da qualificação.

---

## Como navegar este histórico

- Para entender **o que foi feito**: leia este documento na ordem.
- Para entender **o porquê de cada decisão**: siga os links para os docs
  técnicos de cada marco.
- Para encontrar **termos técnicos**: ver [GLOSSARIO.md](GLOSSARIO.md).
- Para apresentar **em slides**: ver [slides_qualificacao.md](slides_qualificacao.md).
