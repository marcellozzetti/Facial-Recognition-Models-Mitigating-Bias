---
name: aldahoul-2024
status_verificacao: VERIFIED
autores: [Nouar AlDahoul, Myles Joshua Toledo Tan, Harishwar Reddy Kasireddy, Yasir Zaki]
ano: 2024
titulo: "Exploring Vision Language Models for Facial Attribute Recognition: Emotion, Race, Gender, and Age"
venue: "arXiv preprint 2410.24148 (NYU Abu Dhabi + University of Florida; follow-up Nature Scientific Reports 2026 a verificar separadamente)"
tipo_publicacao: preprint
arxiv_id: "2410.24148"
doi: null
url_primario: https://arxiv.org/abs/2410.24148
citacoes_google_scholar: null
citacoes_semantic_scholar: 11
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~75
lente_disrupcao: metodologica
fonte_leitura: PDF integral extraído via pypdf (pdfs/aldahoul_2024_vlm.pdf), 40+ páginas
---

# Exploring VLMs for Facial Attribute Recognition (AlDahoul, Tan, Kasireddy & Zaki, 2024)

> **Nota sobre autoria:** este é o paper anteriormente referido como
> "Hassanpour et al. 2024" na documentação do projeto. A atribuição
> "Hassanpour" foi um erro identificado em 2026-05-25; a autoria
> correta é AlDahoul, Tan, Kasireddy, Zaki, verificada via arXiv
> 2410.24148.

## 1. Resumo do problema atacado

Métodos tradicionais de classificação de atributos faciais (race,
gender, age, emotion) usam CNNs supervisionadas treinadas em datasets
labeled. O paper investiga se **vision language models (VLMs) —
GPT-4o, Gemini 1.5 flash, LLaVA-NeXT, PaliGemma, Florence-2 — podem
substituir ou complementar** essas abordagens, em duas direções:
(a) **zero-shot** via prompt engineering; (b) **fine-tuning** sobre
FairFace para criar uma versão especializada (FaceScanPaliGemma).
O paper é a **única referência publicada que avalia classificação de
raça em 7 categorias no FairFace contra baselines comparáveis**, o
que o torna referência crítica para nossa pesquisa.

## 2. Método

### 2.1 Formulação como Visual Question Answering

A tarefa de classificação é reformulada como VQA: imagem + prompt
("What is the race of the main person? Pick one of: [Black, East
Asian, Indian, Latinx or Hispanic, Middle Eastern, Southeast Asian,
White]. Answer using a single word.") → resposta do VLM.

### 2.2 VLMs avaliados

- **GPT-4o** e **GPT-4o-mini** (OpenAI).
- **Gemini 1.5 flash** (Google).
- **LLaVA-NeXT 7B** (open-source).
- **PaliGemma** (Google, pre-trained).
- **Florence-2** (Microsoft, Base e Large).

### 2.3 FaceScanPaliGemma (fine-tuning)

- **Base model:** PaliGemma (Pathways Language and Image + Gemma).
- **Dataset:** FairFace train split (86 744 images).
- **Split de fine-tuning:** 75% train, 25% validation.
- **Teste:** FairFace test split (10 954 images).
- **Fine-tuning separado por tarefa**: 4 versões (race, gender, age,
  emotion). **Não é multi-task simultaneamente.**
- **Disponibilizado open-source:**
  https://huggingface.co/NYUAD-ComNets/FaceScanPaliGemma_Race (e
  análogos para gender, age, emotion).

### 2.4 FaceScanGPT (multi-person via prompt)

GPT-4o com prompt engineering específico para imagens contendo
múltiplas pessoas. Avaliado no dataset proprietário **DiverseFaces**
(1 790 imagens, 4 pessoas por imagem composta a partir do UTKFace).

## 3. Datasets e setup experimental

- **FairFace:** 108 501 imagens, 7 raças.
  - Train: 86 744 (Black 12 233 | East Asian 12 287 | Indian 12 319 |
    Latinx/Hispanic 13 367 | Middle East 9 216 | SE Asian 10 795 |
    White 16 527).
  - Test: 10 954 (Black 1 556 | East Asian 1 550 | Indian 1 516 |
    Latinx 1 623 | Middle East 1 209 | SE Asian 1 415 | White 2 085).
  - **Importante:** distribuição train não é perfeitamente
    balanceada — White=19%, Latinx=15%, Middle East=11%. Isso afeta
    nossa interpretação do "FairFace é balanceado".
- **AffectNet:** ~400 K imagens, 8 emoções.
- **UTKFace:** 24 086 imagens, 5 raças (mescladas: White, Black,
  Asian, Indian, Others).
- **DiverseFaces (proposto pelos autores):** 1 790 imagens
  multi-pessoa.

## 4. Métricas reportadas

- **Accuracy, Precision, Recall, F1** (macro presumido — não
  declarado explicitamente).
- Reportados por classe (Tables 16, 17, 18) e overall.

## 5. Resultados principais (valores numéricos)

### 5.1 Race classification — **7 raças** (FairFace test, Tabela 10)

| Método | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| VGGFace-ResNet-50 + SVM | 72.6% | 72% | 72% | 72% |
| **FairFace ResNet-34 (Kärkkäinen & Joo, 2021)** | **72%** | 72% | 71% | **72%** |
| Google FaceNet + SVM | 68.9% | 69% | 68% | 68% |
| CLIP zero-shot | 64.2% | 67% | 65% | 65% |
| GPT-4o (zero-shot) | 68% | 69% | 66% | 65% |
| **FaceScanPaliGemma (proposto)** | **75.7%** | **75%** | **75%** | **75%** |

**Achado central:** **FaceScanPaliGemma supera o baseline ResNet-34
do FairFace original** (72% → 75.7% accuracy; +3.7 pp). É o **novo
estado da arte para 7-class race classification em FairFace**.

### 5.2 Race classification — **6 raças** (East+SE Asian merged, Tabela 9)

| Método | Accuracy |
|---|---|
| FairFace ResNet-34 | 77.7% |
| GPT-4o | 76.4% |
| Google FaceNet + SVM | 74.8% |
| VGGFace-ResNet-50 + SVM | 72.9% |
| CLIP zero-shot | 70.7% |
| **FaceScanPaliGemma** | **81.1%** |

Margem cresce para +3.4 pp sobre ResNet-34 em 6-class.

### 5.3 FaceScanPaliGemma — breakdown por raça (Table 16, 7-class)

| Raça | Precision | Recall | F1 |
|---|---|---|---|
| Black | 89% | 91% | **90%** |
| White | 79% | 81% | **80%** |
| Indian | 78% | 79% | 79% |
| East Asian | 76% | 80% | 78% |
| Middle Eastern | 75% | 72% | 73% |
| Southeast Asian | 68% | 66% | 67% |
| **Latinx/Hispanic** | **63%** | **58%** | **60%** |

**Disparidade observada:** **F1 max/min = 90% / 60% = 1.5×**.
Latinx/Hispanic mais difícil; Black mais fácil. **Padrão divergente
do FairFace original**, que reportava White facílimo e Latino
dificílimo. Aqui Black supera White.

### 5.4 Gender (Tabela 11)

| Método | Accuracy | F1 |
|---|---|---|
| FairFace ResNet-34 | 94.4% | 94% |
| GPT-4o | 95.9% | 96% |
| **FaceScanPaliGemma** | **95.8%** | **96%** |

VLMs **superam** baselines tradicionais em gender. Diferenças
pequenas mas consistentes.

### 5.5 Age groups (5 categorias: 0-9, 10-19, 20-39, 40-59, 60+)

| Método | Accuracy | F1 |
|---|---|---|
| FairFace ResNet-34 | 79% | 73% |
| GPT-4o | 77.4% | 69% |
| **FaceScanPaliGemma** | **80%** | **74%** |

### 5.6 UTKFace (cross-dataset, FaceScanPaliGemma)

- Race (5-class merged): **88.3% accuracy, F1=83%**.
- Gender: **97.4%, F1=97%**.
- Age: **81.9%, F1=78%**.

(Sem ajuste fino no UTKFace — generalização zero-shot do
FaceScanPaliGemma treinado em FairFace.)

### 5.7 Emotion (AffectNet)

VLMs pre-trained têm performance **decepcionante** (38-50%
accuracy). FaceScanPaliGemma sobe para 59.4% (F1=59%). FMAE
(baseline SOTA emoção) = 65%.

## 6. Limitações declaradas pelos autores

- **FaceScanPaliGemma assume single-person per image:** para
  multi-pessoa, recomendam person-detection upstream ou re-fine-tuning
  com DiverseFaces.
- **Apenas teste accuracy/F1/recall/precision sem desvio padrão**
  (single run, sem multi-seed).
- **Recursos computacionais para fine-tuning** não declarados em
  detalhe.
- **VLMs grandes (GPT-4o) têm custo via API** que limita uso em
  produção sensível a custo.
- **Categorias raciais herdadas do FairFace** — herdam suas
  limitações conceituais.

## 7. Limitações que identifiquei (leitura crítica)

- **Single seed / no statistical rigor.** Toda a tabela é resultado
  de um único treinamento. Sem desvios padrão, intervalos de
  confiança, ou comparação pareada com baselines. **Diferenças de
  3-4 pp podem estar dentro do ruído de treinamento.**
- **Critério de seleção de checkpoint não declarado.** Best val
  accuracy? Best val F1? Final epoch? Cada escolha afeta o resultado.
- **Hiperparâmetros de fine-tuning não detalhados:** learning rate,
  batch size, épocas, scheduler, weight decay — nada disso aparece.
  **Impossível reproduzir exatamente.**
- **Discrepância train distribution:** o paper diz "FairFace is
  imbalanced in terms of race" — White=19% vs Middle East=11%. Esta
  é uma **observação importante**: o "FairFace balanceado" é, na
  prática, **moderadamente desbalanceado em raça** (rotularia "soft
  balanced"). O paper original também observa isso. Implicação para
  nossa pesquisa: subamostragem ou class weights podem ser
  justificáveis.
- **Tabela de 7-class race tem F1=72% para ResNet-34**, mas o paper
  original do FairFace **não publica esse número** (publicou Average
  All=.815 sobre Twitter/Media/Protest external sets). AlDahoul
  reporta um número sobre o **FairFace test in-domain**, não o
  external. Isso explica a divergência aparente (0.815 vs 0.72) — são
  test sets diferentes. **Esta distinção é crítica e deve ser
  preservada.**
- **VLMs comerciais (GPT-4o, Gemini) podem mudar entre versões.** O
  paper roda em outubro 2024; replicação em 2026 pode dar números
  diferentes.
- **Latinx/Hispanic com F1=60%** é **muito próximo** dos .247 do
  FairFace external set (FairFace paper, Table 6), sugerindo que
  Latinx é **intrinsecamente difícil** mesmo com fine-tuning em VLM
  forte.
- **FaceScanPaliGemma é um modelo grande (PaliGemma é 3B
  parâmetros)**; comparar com ResNet-34 (21M params) é comparar
  modelos de classes muito diferentes. A **margem de 3.7 pp pode ser
  pequena dado o gap de capacidade**.

## 8. Relação com nossa pesquisa

**Centralidade extrema:** este é o **único paper publicado que avalia
classificação de raça em 7 categorias sobre FairFace test in-domain
contra múltiplos baselines**. É a **referência de ouro** para
posicionamento da nossa pesquisa.

**Pontos de ancoragem:**

1. **Baseline numérico inequívoco:** **FairFace ResNet-34 in-domain
   7-class = 72% accuracy / 72% F1**. Nossa pesquisa **deve relatar
   contra este número**, não contra o 0.815 do paper original do
   FairFace (que é external sets).
2. **Estado da arte explícito:** **FaceScanPaliGemma 75.7% accuracy
   / 75% F1**. Para nossa pesquisa **superar** este número, precisamos
   ou (a) técnicas algorítmicas além de fine-tuning vanilla, ou (b)
   modelos maiores. Para **igualar** ResNet-34 baseline, basta
   protocolo casado e best-practice.
3. **Métrica de F1 macro como padrão:** AlDahoul reporta F1 macro
   (presumido — eles dão "F1" sem especificar). Nossa pesquisa usa
   F1 macro explicitamente — alinhamento de métrica.
4. **Disparidade max/min = 1.5×** (FaceScanPaliGemma) → nossa razão
   de disparidade DR ~1.06-1.5 fica **comparável em magnitude**.
   Sugere que o gap absoluto entre raças "fáceis" (Black, White) e
   "difíceis" (Latinx, SE Asian) é **relativamente estável across
   métodos** — outra evidência para "balanceamento não basta".
5. **Modelo open-source disponível** em HuggingFace. Permite
   comparação experimental direta se quisermos rodar em nosso
   pipeline.
6. **Posicionamento ético:** o paper enfatiza ethical AI mas não
   reporta métricas de fairness explícitas (DR, equalized odds).
   **Esta é uma lacuna onde nossa pesquisa pode contribuir.**

## 9. Pontos para citar / posicionar

- *"AlDahoul, Tan, Kasireddy e Zaki (2024) avaliam vision language
  models e CNN baselines para classificação de raça em 7 categorias
  sobre o FairFace, estabelecendo que o classifier ResNet-34
  originalmente proposto por Kärkkäinen e Joo (2021) atinge 72% de
  accuracy e F1=72% em condição in-domain — número que difere do
  reportado no paper original (0.815) por ser avaliação sobre o test
  split do próprio FairFace, não sobre os datasets externos
  (Twitter/Media/Protest) usados na avaliação publicada inicialmente."*
- *"O atual estado da arte para classificação de raça em 7 categorias
  no FairFace é o FaceScanPaliGemma (AlDahoul et al., 2024), uma
  versão fine-tuned do PaliGemma de 3B parâmetros, com accuracy de
  75.7% e F1 macro de 75%. Comparações de método são reportadas em
  single seed, sem intervalo de confiança."*
- *"A disparidade entre raças no melhor modelo publicado para
  FairFace 7-class (FaceScanPaliGemma) é F1_max / F1_min = 0.90 /
  0.60 = 1.5×, com Black (F1=90%) como classe mais fácil e
  Latinx/Hispanic (F1=60%) como mais difícil. Esta disparidade
  persiste apesar do dataset balanceado e do modelo de capacidade
  alta, sustentando a observação de Kärkkäinen e Joo (2021) e Wang
  et al. (2019) de que balanceamento de dados não elimina viés
  demográfico."*

## 10. Arquivos relacionados

- PDF local: `pdfs/aldahoul_2024_vlm.pdf` (gitignored).
- Texto extraído: `pdfs/aldahoul_2024_vlm.txt` (gitignored).
- Modelo open-source: HuggingFace `NYUAD-ComNets/FaceScanPaliGemma_Race`.
- Entradas relacionadas: [[dataset_karkkainen_2021]] (dataset central),
  [[manzoor_2024]] (FineFACE — paper anterior auditado, confundido com
  classificação de raça mas é gênero),
  [[dehdashtian_2024]] (U-FaTE — teoriza trade-off complementar).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S1.
