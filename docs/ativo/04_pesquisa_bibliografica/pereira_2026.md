---
name: pereira-2026
status_verificacao: VERIFIED
autores: [Vitor Pereira Matias, Márcus Vinícius Lobo Costa, João Batista Neto, Tiago Novello de Brito]
ano: 2026
titulo: "Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild"
venue: "arXiv preprint (submetido março/2026)"
tipo_publicacao: preprint
arxiv_id: "2603.02475"
doi: null
url_primario: https://arxiv.org/abs/2603.02475
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-09
n_paginas: 12
lente_disrupcao: insumo
fonte_leitura: HTML integral via arxiv.org/html/2603.02475v1, lido em 2026-06-09. PDF binário não foi processado, mas o HTML do arXiv contém o conteúdo completo do paper.
---

# SkinToneNet + STW Dataset (Pereira Matias, Costa, Neto & Novello de Brito, 2026)

> **Estado-da-arte para classificação de tom de pele MST 10-classe in-the-wild.**
> Insumo direto da Etapa 1 do pipeline v3.2, MAS com caveat de
> **contaminação treino-teste**: STW é construído por agregação de
> 7 datasets faciais incluindo FairFace e CelebA. Auditoria MST do
> FairFace É feita pelos autores (Seção 7.1) — esta ficha esclarece
> as implicações para nossa contribuição C2 (matriz MST × race).

> ⚠️ **NOTA DE HISTÓRICO**: versões anteriores desta ficha (commits
> `667e882` e `a798a20`) afirmavam que o paper NÃO auditava FairFace —
> baseando-se apenas no abstract, que menciona apenas CelebA e
> VGGFace2. A leitura do HTML integral em 2026-06-09 corrige este
> erro: FairFace é uma das fontes do STW E é auditado em Seção 7.1.

## 1. Resumo do problema atacado

A literatura de fairness em IA facial usa skin tone como proxy de
diversidade demográfica, mas a infraestrutura para isso é limitada:

- A escala Fitzpatrick 6-tons "lacks visual representativeness"
  (cf. [[fitzpatrick_1988]]).
- Datasets MST públicos disponíveis são pequenos ou privados (~1.500
  imagens em MST-E).
- Pipelines clássicos de CV existem mas poucos usam deep learning
  para skin tone.
- Problemas conhecidos como train-test leakage e dataset imbalance
  são tipicamente ignorados.

Os autores propõem framework "comprehensive" em 4 componentes:
dataset (STW), benchmark clássico (SkinToneCCV), classificador SOTA
(SkinToneNet), e auditoria de 8 datasets faciais.

## 2. Método

### 2.1 Dataset STW (Skin Tone in the Wild)

**Fonte (Seção 3)**: 7 datasets faciais open-access agregados:

| Dataset-fonte | Contribuição |
|---|---|
| LFW (Labeled Faces in the Wild) | faces gerais |
| CASIA Face Africa (CFA) | faces africanas |
| CASIA V5 | faces asiáticas |
| FEI | faces brasileiras |
| Faces 94 & 95 | datasets antigos com diversidade |
| **FairFace** | **face attribute fairness — pré-existe na agregação** |
| **CelebA** | **celebridades — auditado depois** |

**Estatísticas finais**: 42.313 imagens de 3.564 indivíduos únicos,
anotados na escala MST 10-tons.

### 2.2 Protocolo de anotação (mais robusto do que o esperado)

- **1 anotador principal expert** rotulou todos os 3.564 indivíduos
  (para consistência).
- **2 anotadores independentes** validaram um subset estratificado de
  1.000 indivíduos (100 por classe MST).
- **Inter-annotator agreement (IAA)**:
  - Exact match: **38.8%** (consistente com subjetividade declarada de
    Schumann 2023).
  - Off-by-one accuracy: **88%**.
  - **ICC(3) = 0.939** (intraclass correlation — excelente).
  - **Krippendorff's α = 0.935** (excelente).
- **Protocolo "strict 10-page"** com interface custom split-window
  mostrando imagens de referência ao lado da imagem-alvo.

### 2.3 Splits

- **20% reservado para teste**, 80% restante dividido 80/20 para
  hyperparameter tuning.
- **5-fold cross-validation** no 80% de treino.
- **Duas estratégias de split**:
  - **IMG (Image-based)**: split por imagem.
  - **IND (Individual-based)**: split por indivíduo — **previne identity
    leakage**.
- Para deep learning: **máximo de 2 imagens por indivíduo** no treino
  (balanceamento por identidade).

### 2.4 SkinToneNet — arquitetura

- **Backbone**: **ViT-Small** (Vision Transformer Small).
- **Pré-treino**: **ImageNet** (declarado explicitamente).
- **Fine-tuning**: backbone completo fine-tuned em STW.
- **Input**: Full-image (FI) — não usa segmentação de pele.
- **Loss**: Cross-entropy.
- **Justificativa de arquitetura**: "architectures with global context
  (like ViT's attention) or dense feature reuse (DenseNet) are better".

### 2.5 Hiperparâmetros (parcialmente declarados)

| Hiperparâmetro | Valor declarado |
|---|---|
| Optimizers testados | SGD e Adam |
| Learning rate scheduler | reduz pela metade quando val performance plateau |
| Augmentation | horizontal/vertical flips, translations, rescaling, rotations, random brightness/contrast/hue/saturation, Gaussian blur, Gaussian noise, randomized grid shuffling com coarse dropout |
| Learning rate exato | ❌ não declarado |
| Batch size | ❌ não declarado |
| Épocas totais | ❌ não declarado |

## 3. Datasets e setup experimental

### 3.1 Datasets de avaliação

- **STW-test** (interno, split de teste).
- **MSTE** (Monk Skin Tone Example — dataset oficial Google).
- **MSTE-G** (MSTE-Gold, subset com gold-standard annotation).
- **CCv2** (Casual Conversations V2 — Meta 2022, ver [[dataset_hazirbas_2021]]).

### 3.2 Datasets auditados (Seção 7.1 — IMPORTANTE)

8 datasets faciais auditados via SkinToneNet em zero-shot:

| Dataset | Tipo |
|---|---|
| FACET | facial attribute |
| IMDB Faces | celebridades |
| CelebA | celebridades |
| VGGFace2 | face recognition large-scale |
| CASIA WebFace | face recognition |
| **FairFace** | **face attribute fairness** |
| FERET | face recognition antigo |
| LFW | face recognition padrão |

**Achado central da auditoria**: "Almost all evaluated datasets
exhibit a high absence of MST classes 6 through 10" com
"considerable imbalance for tones 0 and 1".

### 3.3 Comentário específico sobre FairFace

Os autores comentam que "FACET and Fairface were built to perform
fairness evaluation" mas "no dataset shows a significant presence of
skin tone 6 or higher". Isto é, mesmo FairFace, criado para fairness,
exibe subrepresentação severa de tons escuros.

## 4. Métricas reportadas

- **bAcc** (balanced accuracy) — média das accuracies por classe MST.
- **wOOAcc** (weighted off-one accuracy) — tolerância a ±1 classe MST,
  considerando subjetividade da anotação.
- **Acc / OOAcc** simples nas tabelas de comparação cross-dataset.
- Auditoria reporta distribuição percentual por classe MST (não
  publicado como tabela explícita no que extraímos do HTML — ver PDF
  para gráficos).

## 5. Resultados principais (valores numéricos)

### 5.1 Tabela 2 — SkinToneCCV (modelos clássicos)

| Modelo | Bins ITA | bAcc | wOOAcc |
|---|---|---|---|
| KNN | 8 | 0.310 | 0.603 |
| RF | 16 | 0.339 | 0.629 |
| DT | 64 | 0.303 | 0.614 |
| MLP | 8 | 0.310 | 0.643 |
| SVM | 8 | 0.329 | 0.676 |

**Achado**: modelos clássicos atingem bAcc ~31-34%, próximo ao random
para 10 classes (10%) só ligeiramente acima — confirma claim do
abstract de "near-random".

### 5.2 Tabela 4 — Arquiteturas de Deep Learning

Em STW-test / MSTE / CCv2 (sempre bAcc / wOOAcc):

| Modelo | STW bAcc | STW wOOAcc | MSTE Acc | MSTE OOAcc | CCv2 Acc | CCv2 OOAcc |
|---|---|---|---|---|---|---|
| DenseNet121 | 0.445 | 0.860 | 0.292 | 0.792 | 0.227 | 0.626 |
| DINOv2 | 0.373 | 0.820 | 0.329 | 0.766 | 0.276 | 0.650 |
| DINOv3 | 0.466 | 0.883 | 0.313 | 0.890 | 0.277 | 0.700 |
| LabNet | 0.328 | 0.777 | 0.227 | 0.588 | 0.195 | 0.507 |
| ResNet18 | 0.389 | 0.856 | 0.321 | 0.740 | 0.228 | 0.615 |
| ViT-Base | 0.414 | 0.880 | 0.396 | 0.862 | 0.327 | 0.686 |
| **ViT-Small** | **0.449** | **0.901** | **0.413** | **0.853** | **0.250** | **0.706** |

**Achados**:

- **ViT-Small (SkinToneNet)** é o melhor em **STW wOOAcc (0.901)** e
  **CCv2 OOAcc (0.706)**.
- **DINOv3** o supera em STW bAcc (0.466 vs 0.449) e MSTE OOAcc (0.890
  vs 0.853) — não é dominância absoluta.
- **Deep learning vs clássico**: melhor DL ~45% bAcc vs melhor clássico
  34% bAcc — gap de ~11pp, mas long way de "near-annotator accuracy".
- "Near-annotator accuracy" do abstract refere-se ao IAA exact 38.8% —
  bAcc 0.449 ESTÁ próximo do anotador.

### 5.3 Tabela 3 — Generalização vs prior work

SkinToneNet atinge Acc/OOAcc = 0.43/0.87 em STW (não detalhamos os
baselines aqui).

### 5.4 Auditoria FairFace (Seção 7.1)

Resultado qualitativo declarado:

- FairFace tem **distribuição agregada** com forte ausência de MST 6-10.
- "No dataset shows a significant presence of skin tone 6 or higher" —
  generalização que inclui FairFace.

**O que NÃO está no paper** (verificado por leitura HTML):

- Matriz cruzada **MST × race classes específicas do FairFace** (i.e.,
  como Latinx se distribui em MST 1-10 vs como East Asian se distribui).
- Análise per-race da auditoria.
- Confusion matrix MST × race.

## 6. Limitações declaradas pelos autores

- Dataset STW permanece **imbalanced image-wise** — extremos MST 1, 9,
  10 com menos imagens (declarado: "expected outcome given the
  logistics challenge associated with the labeling of the extremities").
- **IAA exact match = 38.8%** declarada como "consistent with the
  subjectivity of skin tone perception" (referência implícita a
  Schumann 2023 — 27% divergência inter-anotador).
- Modelos clássicos (SkinToneCCV) falham em ambientes não-controlados.

## 7. Limitações que identifiquei (leitura crítica)

### 7.1 Contaminação treino-teste para nossa C2

**CRÍTICO**: STW é agregado a partir de FairFace + CelebA + 5 outros.
Se nosso pipeline aplica SkinToneNet (treinado em STW que inclui
FairFace) para auditar FairFace, há **contaminação treino-teste**.
Implicações:

- Métricas de "generalização para FairFace" não são válidas —
  FairFace estava no treino.
- Distribuição MST × race que reportarmos pode refletir overfit aos
  rótulos que o anotador principal atribuiu, não distribuição
  empírica real.

### 7.2 Code and data "available soon"

No momento da leitura (2026-06-09), código e dataset declarados como
"available soon". **Não confirmado se pesos pré-treinados estão
publicamente disponíveis**. Esta é uma dependência crítica para nossa
decisão de "usar SkinToneNet pré-treinado".

### 7.3 ViT-Small parâmetros

ViT-Small tem ~22M parâmetros (não declarado pelos autores; conhecido
da literatura). Compatível com nosso budget compute, mas detalhes de
checkpoint (qual versão? qual fine-tune?) precisam ser confirmados
quando código sair.

### 7.4 Auditoria FairFace é qualitativa, não quantitativa

Os autores reportam que FairFace tem ausência de MST 6-10 mas
**não publicam**:
- Histograma exato MST do FairFace.
- Matriz cruzada MST × race classes.
- Análise comparativa entre as 7 raças do FairFace.

**Nossa C2 ainda pode ser contribuição original** desde que vá além
do "FairFace tem ausência de MST 6-10" e entregue **a matriz cruzada
MST × race** com análise comparativa por classe.

### 7.5 Hiperparâmetros incompletos

LR exato, batch size, épocas totais não declarados. Para reprodução
exata, esperar release do código.

### 7.6 Sem ablation crítica

Não há ablation de:
- ViT-Small vs ConvNeXt-T vs EfficientNet para skin tone
  classification — impede nossa decisão arquitetural ser totalmente
  validada por este paper.
- Estratégia de split (IMG vs IND) — implicações de identity leakage
  comparadas só qualitativamente.
- Trade-off entre tamanho do dataset (STW 42K) e qualidade da
  anotação (1 anotador principal).

## 8. Relação com nossa pesquisa

### 8.1 Decisão técnica revisada após leitura integral

**Antes (com base no abstract)**: "Usar SkinToneNet pré-treinado como
insumo da Etapa 1; C2 segue original porque autores não auditaram
FairFace."

**Agora (com base no HTML integral)**: Decisão **mais nuançada**.

| Decisão | Status |
|---|---|
| Usar SkinToneNet pré-treinado | ⚠️ Condicionada à liberação do código/pesos (paper diz "available soon") |
| C2 segue original? | ⚠️ **Apenas parcialmente** — autores fizeram auditoria QUALITATIVA, nossa C2 entrega análise QUANTITATIVA (matriz cruzada MST × race) |
| Contaminação treino-teste | ❌ **PROBLEMA NOVO**: STW inclui FairFace; precisa estratégia para validar SkinToneNet em FairFace independente |

### 8.2 Como nossa C2 pode permanecer original

Diferenciar nossa contribuição:

| Pereira 2026 (Seção 7.1) | Nossa C2 |
|---|---|
| Distribuição AGREGADA MST do FairFace (qualitativa: "ausência MST 6-10") | **Matriz MST × race classes** (quantitativa: % de cada raça em cada bin MST) |
| Auditoria comparativa entre datasets | Análise INTERNA do FairFace por raça |
| 8 datasets, FairFace é UM | FairFace é alvo PRINCIPAL |
| Sem confusion matrix | Confusion matrix MST × race + análise de overlap entre raças (importante para H4) |
| Skin tone como dimensão demográfica isolada | Skin tone como sinal AUXILIAR para race classifier (H1 + FiLM-conditioning) |

### 8.3 Mitigação da contaminação treino-teste

**Opções**:

1. **Validar SkinToneNet em subset FairFace anotado manualmente**
   por anotadores diversos via Prolific (~700 imgs × 3 anotadores).
   Comparar accuracy contra rótulos humanos para detectar overfit.
2. **Re-treinar SkinToneNet apenas com 6 datasets (excluindo
   FairFace)** se código for liberado. Valida que generalização não
   depende de ter visto FairFace.
3. **Documentar honestamente** que SkinToneNet viu FairFace durante
   treino — limitação metodológica reconhecida na dissertação.

Plano padrão: **opção 1** (validação independente) + **opção 3**
(documentação honesta).

### 8.4 Recomendação revisada

- **Curto prazo (Cap 1)**: usar SkinToneNet pré-treinado **se** código
  for liberado. Validar em subset FairFace via anotação manual
  independente.
- **Plano B**: se SkinToneNet não for liberado, treinar próprio com
  ViT-Small fine-tuned em STW (se dataset for liberado) ou MST-E +
  Casual Conversations (se STW também não disponível).
- **C2 reformulado**: matriz cruzada MST × race do FairFace,
  diferenciando-se da auditoria agregada de Pereira pela
  granularidade per-race.

## 9. Pontos para citar

- *"Pereira et al. (2026) introduzem o STW (Skin Tone in the Wild),
  dataset agregado de 42.313 imagens de 3.564 indivíduos anotados na
  escala Monk Skin Tone 10-classe, agregando 7 datasets faciais
  pré-existentes (LFW, CASIA Face Africa, CASIA V5, FEI, Faces 94/95,
  FairFace e CelebA). Anotação por anotador principal único validada
  por dois anotadores independentes em subset estratificado de 1.000
  indivíduos (100 por classe), atingindo ICC(3) = 0.939 e
  Krippendorff's α = 0.935."*

- *"SkinToneNet (Pereira et al., 2026) — Vision Transformer Small
  fine-tuned em STW a partir de pré-treino ImageNet — atinge balanced
  accuracy de 0.449 e off-one accuracy de 0.901 no test split do
  STW. Comparado em zero-shot ao Casual Conversations V2 e ao MSTE,
  mantém competitividade contra arquiteturas alternativas (DINOv3,
  ViT-Base, DenseNet121)."*

- *"A auditoria de Pereira et al. (2026, Seção 7.1) sobre 8 datasets
  faciais — incluindo FairFace — reporta ausência sistemática de
  classes MST 6-10. A presente dissertação estende essa observação
  ao reportar a distribuição cruzada MST × race classes para o
  FairFace, granularidade não publicada pelos autores e necessária
  para investigar a hipótese de overlap fenotípico entre categorias
  raciais (H4)."*

- *"Conforme reconhecido na Seção 7.1 de Pereira et al. (2026),
  SkinToneNet foi treinado em STW que inclui FairFace como dataset-
  fonte. Esta dependência de treino é mitigada nesta dissertação por
  validação independente do classificador em subset FairFace
  re-anotado manualmente por anotadores diversos via Prolific
  Academic, conforme protocolo descrito em [02_metodologia]."*

## 10. Arquivos relacionados

- PDF: pendente download para `pdfs/pereira_2026_skintonenet.pdf`.
  HTML integral foi a fonte da leitura ([arxiv.org/html/2603.02475v1](https://arxiv.org/html/2603.02475v1)).
- Código original: declarado "available soon" — não disponível em
  2026-06-09.
- Dataset STW: declarado open-access — pendente confirmação de URL
  pública.
- Licença: **CC BY 4.0**.

### Entradas relacionadas no corpus

- [[schumann_2023]] — MST scale + protocolo de consenso (escala de
  origem; IAA exact 38.8% de Pereira ≈ 27% de Schumann).
- [[dataset_hazirbas_2021]] — Casual Conversations (também é dataset
  de validação out-of-domain de Pereira via CCv2).
- [[dataset_karkkainen_2021]] — FairFace (fonte do STW + alvo de
  auditoria + alvo da nossa C2).
- [[buolamwini_2018]] — motivação histórica do uso de escalas de tom.
- [[dominguez_2024]] — DSAP (metodologia alternativa de auditoria).
- [[perez_2018]] — FiLM (consome saída do SkinToneNet).
- [[../_validacao_cientifica_pipeline]] — análise da Rodada 6 (precisa
  atualização para refletir achados desta leitura integral).

## 11. Trabalhos sugeridos pelos autores (Future Work)

Da seção final do paper:

- **Classificar tom de pele em outras modalidades** — books, vídeos, etc.
  ❌ Fora do nosso escopo.
- **Endereçar imbalance** adicionando mais indivíduos em MST 3, 4, 5, 6.
  ⚠ Direção paralela.
- **Explorar Colorimetric Skin Tone scale (CST)** como alternativa de
  anotação. ⚠ Direção alternativa de escala — escolhemos manter MST.

## 12. Análise crítica do método

### (a) Rigor formal

- Definição da tarefa (10-classe MST) é direta — sem formalismo
  complexo a auditar.
- **Anotação subjetiva**: IAA exact 38.8% é declarado e contextualizado
  contra Schumann 2023. **Postura epistemicamente honesta**.
- **ICC(3) = 0.939 e Krippendorff's α = 0.935** são valores fortes —
  indicam que apesar da subjetividade exact-match, o ordenamento
  relativo entre anotadores é consistente.

### (b) Reprodutibilidade

- **Positivo**: dataset declarado open-access, código declarado
  "available soon", split strategies (IMG/IND) explícitas, IAA
  reportados, 5-fold CV documentado.
- **A verificar**: liberação efetiva do código e pesos (não confirmado
  em 2026-06-09).
- **Limitação**: hiperparâmetros exatos não declarados (LR, batch,
  épocas).

### (c) Aplicabilidade ao pipeline v3.2

- **Etapa 1**: alta aplicabilidade SE código for liberado;
  **mitigação obrigatória** da contaminação treino-teste via validação
  independente em FairFace.
- **Etapa 2** (auditoria FairFace via matriz MST × race):
  **diferenciação obrigatória** de Pereira via análise per-race com
  confusion matrix MST × race.
- **Etapa 3** (FiLM-conditioning): SkinToneNet provê o vetor de
  contexto z. Decisão de design: usar softmax 10-dim por
  interpretabilidade.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| MST sobre Fitzpatrick | ✅ Justificada |
| ViT-Small como backbone | ⚠ Parcialmente — comparação com DINOv3 mostra que ViT-Small não é dominante; faltava comparar com ConvNeXt-T |
| Agregar 7 datasets em STW | ✅ Justificada por escala, mas implica contaminação treino-teste para usuários downstream |
| 1 anotador principal + 2 validadores | ⚠ Trade-off escala vs rigor — alternativa seria 3 anotadores em todos os 3.564 indivíduos. IAA fortes (ICC, α) sugerem que o trade-off foi defensável |
| Split IMG vs IND | ✅ Justificada (previne identity leakage) |
| Hiperparâmetros não declarados | ❌ Assumida — limita reprodutibilidade exata |

### (e) Conexão com R5/R6

- **Conversa com [[schumann_2023]]**: ambos endereçam MST scale.
  Pereira escala (28× mais imagens) mas usa anotação 1+2; Schumann
  estabelece protocolo de consenso mais robusto. **Complementares.**
- **Conversa com [[dominguez_2024]] (DSAP)**: ambos auditam datasets
  faciais. Pereira foca em MST; DSAP em demographic profile multi-axis.
  **Complementares.**
- **Conexão com [[perez_2018]] (FiLM)**: Pereira fornece o
  classificador cuja saída será usada como contexto FiLM. Combinação
  Pereira + Perez = operacionalização da Etapa 3.
- **Implicação para H4**: a auditoria qualitativa de Pereira ("FairFace
  tem ausência de MST 6-10") é compatível com a hipótese de que erros
  de race classification ocorrem em zonas de overlap MST. Nossa C2
  (matriz cruzada quantitativa) é necessária para testar H4 mais
  precisamente.
