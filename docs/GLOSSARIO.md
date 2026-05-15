# Glossário de Termos Técnicos

Glossário consolidado de todos os termos técnicos utilizados ao longo do
projeto. Organizado por categoria, com termo em inglês entre parênteses
quando relevante e referência para onde o termo aparece em uso.

---

## 1. Arquitetura de Modelos

**Backbone (espinha dorsal):** parte da rede neural responsável por
extrair features de alto nível da imagem de entrada. No projeto, é a
**ResNet50** pré-treinada no ImageNet, com a camada `fc` (1000 classes
ImageNet) substituída por `nn.Identity()` — preservando os 23M parâmetros
do extrator mas removendo a saída original. Produz vetores de **2048
dimensões**.

**Cabeçote / Head (classification head):** a parte "leve" da rede que pega
a feature extraída pelo backbone e produz os **logits** finais (uma
pontuação por classe). No projeto temos 3 cabeçotes:

- `head="linear"`: `nn.Linear(2048, 7)` — baseline do MBA.
- `head="arcface"`: `ArcMarginProduct(2048, 7)` — adiciona margem angular
  (Deng et al., 2019).
- `head="mlp"`: `MLPHead` — multicamada totalmente configurável (depth,
  width, activation, norm, dropout). Implementado para o mestrado.

**Embedding:** vetor de features produzido pelo backbone (antes do head).
No projeto, tem **2048 dimensões** (saída do `avgpool` da ResNet50).

**Logit:** valor de pontuação por classe, antes da aplicação do softmax. O
modelo classifica como `argmax(logits)`.

**LResNet50E_IR:** nomenclatura do projeto para o modelo composto
(`Backbone ResNet50 → Head configurável`). Definido em
`src/face_bias/models/resnet.py`.

**MLPHead (Multi-Layer Perceptron head):** cabeçote denso configurável
implementado para o mestrado. Aceita uma lista `hidden_dims` (e.g.,
`[256]`, `[128, 256]`, `[1024, 1024, 2048]`), `activation` (relu, gelu,
silu, tanh), `norm` (none, batchnorm, layernorm) e `dropout`. Última
camada é sempre `Linear(prev, num_classes)` emitindo logits puros.

**ArcMarginProduct (ArcFace head):** cabeçote que normaliza pesos e
features na esfera unitária, computa cosseno, e aplica margem angular
aditiva (`m=0.5`) ao ângulo entre feature e peso da classe correta. Faz
parte da família **margin-based losses**.

**ResNet50:** rede convolucional profunda com 50 camadas e ~23M parâmetros,
introduzida por He et al. (2016). Usa conexões residuais para permitir
treinamento de redes muito profundas. Pré-treinada no ImageNet (1.28M
imagens, 1000 classes).

**MTCNN (Multi-task Cascaded Convolutional Network):** detector de faces
em 3 estágios (P-Net, R-Net, O-Net) que retorna bounding boxes e
landmarks. Usado no projeto para **pré-processamento** (alinhamento de
faces) e para a **auditoria multi-face** (contar quantas faces há por
imagem).

---

## 2. Funções de Perda (Loss Functions)

**Cross-Entropy (CE) / Softmax-based loss:** loss padrão de classificação
multiclasse. Aplica softmax aos logits e calcula `-log(p_target)`. Loss
mais robusta a ruído de rótulo.

**ArcFace / Margin-based loss:** loss que adiciona margem angular
aditiva entre a feature e o peso da classe correta. Força o modelo a
aprender embeddings com maior separação angular. Mais sensível à
qualidade dos rótulos do que CE.

**AdaFace / MagFace:** variantes de ArcFace que ajustam dinamicamente a
margem com base na "qualidade" do embedding (magnitude). Não usadas
ainda no projeto, planejadas para a diretriz 6.

**KP-RPE (Keypoint Relative Position Encoding):** loss que incorpora
informação relativa entre keypoints faciais. Não usada ainda, planejada
para a diretriz 6.

---

## 3. Treinamento

**Otimizadores (Optimizers):**

- **SGD (Stochastic Gradient Descent):** otimizador clássico com momentum.
  Configurado com `momentum=0.9, weight_decay=5e-4`.
- **AdamW:** Adam com weight decay desacoplado da regra de atualização.
  Configurado com `weight_decay=5e-4`.

**Schedulers (LR schedulers):**

- **OneCycleLR:** ciclo único de subida e descida do learning rate ao
  longo do treino. Bom para treinos curtos.
- **CosineAnnealingWarmRestarts:** decaimento cosseno com restarts
  periódicos. **Usado no recipe vencedor (Exp 5)**.

**AMP (Automatic Mixed Precision):** treino em precisão mista (fp16 para
forward/backward, fp32 para acumulação de gradientes). Usa
`torch.amp.autocast` + `torch.amp.GradScaler` para evitar underflow do
gradiente. Ganho típico: **~1.5-2× speedup** + ~50% redução de memória
GPU. Ligado no HPO sempre; opt-in via `training.use_amp` no Trainer
(default `false` para reprodutibilidade do MBA).

**GradScaler:** escala o loss em fp16 para evitar underflow do gradiente.
Faz `scaler.scale(loss).backward()` e `scaler.unscale_(optimizer)` antes
do `clip_grad_norm_`, depois `scaler.step(optimizer)` e `scaler.update()`.

**Gradient Clipping (`grad_clip_norm`):** limita a norma global dos
gradientes para evitar explosão. Default no projeto: 5.0. Essencial para
estabilidade do ArcFace + AdamW.

**Early Stopping:** interrompe o treino quando `val_loss` para de melhorar
por N epochs (`patience`). Default no projeto: `patience=5`. Salvou
significativo tempo de compute em R2 (Exp 5 parou em ep 10 vs 25
budget).

**Dropout:** zera aleatoriamente uma fração `p` dos neurônios durante
treino. Regulariza contra overfit. No projeto, aparece em 2 lugares:
após o `avgpool` do backbone (`model.dropout=0.2`) e dentro do MLP head
(`mlp_dropout` ∈ [0, 0.6] na busca Optuna).

**BatchNorm1d:** normaliza features ao longo da dimensão de batch. Útil
para acelerar convergência, assume batch grande e iid.

**LayerNorm:** normaliza features ao longo da dimensão das features (por
amostra). Mais estável que BatchNorm para batches pequenos.

**Activation functions:**

- **ReLU:** `max(0, x)`. Não-diferenciável em zero, pode causar "dead
  neurons".
- **GELU (Gaussian Error Linear Unit):** suave, diferenciável em zero.
  Padrão em transformers. **Venceu o HPO Round 1 e 2**.
- **SiLU (Sigmoid Linear Unit) / Swish:** `x * sigmoid(x)`. Suave também.

**Weight decay:** regularização L2 sobre os parâmetros do modelo. Default
no projeto: 5e-4.

**Pretrained weights:** pesos do modelo aprendidos previamente em um
dataset grande (ImageNet, no caso). Servem como inicialização para
fine-tuning no FairFace.

**Fine-tuning:** continuar treinando um modelo pré-treinado em um novo
dataset, ajustando todos (ou parte) dos pesos.

---

## 4. Dataset e Pré-processamento

**FairFace:** dataset público de ~108k imagens de rostos com anotações de
**idade**, **gênero** e **raça** (7 categorias). Kärkkäinen & Joo (2021).
Distribuição mais balanceada por raça que a maioria dos datasets de
reconhecimento facial. Disponível em https://github.com/joojs/fairface.

**Classes de raça no FairFace:**
- White, Black, Latino_Hispanic, East Asian, Southeast Asian,
  Middle Eastern, Indian.

**Undersampling:** estratégia de balanceamento que reduz todas as
classes ao tamanho da minoria. Trade-off: dataset balanceado vs perda
de amostras. **Foi a estratégia adotada no MBA e mantida no mestrado.**

**Stratified split (split estratificado):** divisão treino/val/teste
preservando a proporção de classes em cada split. No projeto: 80/10/10
com `random_state=42`.

**Data augmentation:** transformações aleatórias aplicadas aos dados
durante treino para aumentar a variedade. No projeto, só `RandomHorizontalFlip`
no train.

**Image transforms:**
- Resize 224×224
- ToTensor (HWC uint8 [0,255] → CHW float [0,1])
- Normalize (com `image_mean` e `image_std` ImageNet)

**Multi-face image:** imagem que contém mais de uma face detectada pela
MTCNN. **24 531 imagens (25.11%) do FairFace** caem nessa categoria —
fonte potencial de ambiguidade de rótulo.

**Dataset limpo (clean dataset):** subconjunto do FairFace contendo
**apenas imagens com 1 face** (Opção A adotada na auditoria 2026-05-14).
72 749 imagens (vs 97 698 originais). Salvo em
`data/raw/fairface/fairface_labels_clean.csv`.

**Alinhamento de face (face alignment):** rotação e crop da imagem para
que olhos, nariz e boca fiquem em posições padronizadas. No projeto, feito
pelo `preprocess` que usa MTCNN para detectar landmarks.

---

## 5. Métricas de Classificação

**Accuracy (acurácia):** fração de predições corretas
(`predicted == true`).

**F1 score:** média harmônica de precision e recall. Mais robusto que
accuracy quando há desbalanceamento de classes.

**F1 macro:** média simples do F1 por classe. Trata todas as classes
igualmente — métrica preferida para problemas balanceados como o nosso
(pós undersampling).

**F1 weighted:** média do F1 ponderada pelo suporte (número de amostras)
de cada classe.

**Precision (precisão):** dos preditos como classe X, quantos são de fato
classe X. `TP / (TP + FP)`.

**Recall (sensibilidade):** das amostras reais da classe X, quantas o
modelo identificou. `TP / (TP + FN)`.

**Log loss:** entropia cruzada média (mesma fórmula da CE loss, calculada
nas probabilidades de teste).

**Confusion matrix (matriz de confusão):** tabela `true × predicted`
contando ocorrências. Diagonal = acertos.

---

## 6. Métricas de Fairness (Equidade)

**Inequity Rate (IR):** razão max/min do score (F1, precision, recall)
entre grupos demográficos.
- IR = 1.0 → paridade perfeita.
- IR = 2.0 → o grupo melhor servido tem o dobro do score do pior.
- IR = ∞ → pelo menos um grupo tem score 0 (uma classe nunca é predita).
- **A métrica primária de fairness neste projeto.**

**Max-Min disparity:** diferença absoluta `max - min` do score entre
grupos. Complemento do IR.

**Gini coefficient:** medida de desigualdade da distribuição entre grupos.
0 = paridade, 1 = um grupo monopoliza.

**FDR (False Discovery Rate):** `FP / (FP + TP)` — fração das predições
positivas que são erradas. Útil para reconhecimento biométrico.

**CEI (Classification Equity Index):** índice composto que combina F1 e
disparidade entre grupos. Definido localmente no projeto.

**Demographic Parity:** todas as classes têm a mesma taxa de predição
positiva.

**Equalized Odds:** todas as classes têm a mesma TPR (true positive rate)
e FPR (false positive rate).

**Pereira & Marcel (2024):** referência para o framework de Inequity Rate
aplicado a sistemas biométricos, adaptado neste projeto para
classificação per-class.

---

## 7. Otimização de Hiperparâmetros (HPO)

**HPO (Hyperparameter Optimization):** processo de buscar a melhor
combinação de hiperparâmetros (LR, dropout, profundidade, etc.) de forma
sistemática.

**Grid search:** testa todas as combinações de uma grade pré-definida.
Custo cresce exponencialmente com o número de eixos.

**Random search:** sorteia N combinações aleatórias do espaço. Bergstra
& Bengio (2012) mostraram que é mais eficiente que grid em alta dimensão.

**Optuna:** framework de HPO em Python (Akiba et al., 2019). Suporta TPE,
multi-objective, pruning, persistence em SQLite. **Usado no projeto para
buscar topologia do MLP head.**

**Study / Trial:** estrutura do Optuna.
- **Study:** experimento inteiro, com nome, storage, sampler.
- **Trial:** uma execução individual com uma combinação de
  hiperparâmetros.

**TPE (Tree-structured Parzen Estimator):** sampler bayesiano do Optuna.
Modela `p(params | bom)` vs `p(params | ruim)` via KDE, sorteia próximas
amostras maximizando `p(bom)/p(ruim)`.

**Multi-objective optimization:** otimizar mais de uma métrica
simultaneamente. No nosso caso: maximizar F1 macro **e** minimizar IR.

**Pareto front / Pareto frontier:** conjunto de trials **não-dominados**
— nenhum outro trial é estritamente melhor em todos os objetivos. Em
multi-objetivo, não há "melhor único", há frente de Pareto.

**Pareto-aware best epoch:** critério de seleção de "melhor epoch" dentro
de um trial, baseado em dominância local. Implementado no projeto após o
HPO Round 1 expor o problema do "best by F1" descartar epochs de baixa IR.

**Pruning:** descartar trials ruins cedo (Optuna `MedianPruner`,
`HyperbandPruner`). **Não funciona em multi-objetivo** — descoberto
empiricamente no Round 1, removido do script.

**NSGA-II (Non-dominated Sorting Genetic Algorithm II):** algoritmo
genético para multi-objetivo. Otuna usa internamente em
`multi_objective` quando especificado.

**Sampler `multivariate=True`:** permite que o TPE modele correlações
entre hiperparâmetros (não trata cada um independentemente).

**`n_startup_trials`:** quantos trials iniciais são puramente aleatórios
antes do TPE começar a aprender. Default: 10. Importante para evitar
exploitation prematura.

---

## 8. Software / Tooling

**PyTorch:** framework de deep learning principal do projeto. Versão atual:
**2.12.0+cu126** (atualizada de 2.5.1+cu121 na Wave 2 de segurança).

**torchvision:** complemento do PyTorch com modelos pré-treinados (ResNet,
EfficientNet, etc.) e transforms. Versão atual: **0.27.0+cu126**.

**CUDA:** plataforma de computação paralela da NVIDIA. Versão usada: 12.6
(cu126).

**Pydantic:** validação de schemas em Python. Usado para validar configs
YAML do projeto via `face_bias.config.schema`.

**MLflow:** tracking de experimentos (params, métricas, artefatos). Backend
local em `outputs/<run_id>/mlruns/`.

**DVC (Data Version Control):** versionamento de datasets e modelos. Não
ativamente usado ainda; instalação via `pip install face-bias[dvc]`.

**pytest:** framework de testes. Markers usados: `unit`, `integration`,
`smoke`, `gpu`.

**ruff, black:** linter e formatter Python. Pré-commit instalado.

**pre-commit:** hooks de Git que rodam linters antes de cada commit.
`detect-secrets` previne commit acidental de credenciais.

**pip-audit / Dependabot:** ferramentas que verificam vulnerabilidades
conhecidas (CVE) nas dependências. Dependabot do GitHub abre PRs
automaticamente.

**CVE (Common Vulnerabilities and Exposures):** identificador único para
uma vulnerabilidade de segurança publicada. Ex: `CVE-2025-32434` (torch
RCE).

**GHSA (GitHub Security Advisory):** identificador alternativo de
vulnerabilidade no GitHub. Ex: `GHSA-2rj9-7h5r-q4h8`.

**SQLite:** banco de dados embutido em arquivo. Usado pelo Optuna como
storage do study (persistência + resumibilidade).

**facenet-pytorch:** biblioteca com MTCNN pronto para PyTorch. Usado no
pré-processamento e na auditoria multi-face.

---

## 9. Regulação e Contexto

**EU AI Act:** regulamentação europeia (em vigor) que classifica sistemas
de IA por risco. Sistemas de **biometria à distância** são de **alto
risco** (Annex III), exigindo evidência empírica de fairness antes do
deploy.

**NIST FRTE (Face Recognition Technology Evaluation):** programa do NIST
que avalia sistemas de reconhecimento facial em vários eixos, incluindo
fairness demográfica. Versão 2024 incluiu protocolos para auditoria de
fairness.

**Annex III, EU AI Act:** lista de aplicações de alto risco. Inclui
reconhecimento facial em espaços públicos.

---

## 10. Termos do Projeto (vocabulário interno)

**R1:** Round 1 do HPO — dataset original (97k), fp32. Concluído em
2026-05-13/14.

**R2:** Round 2 do HPO — dataset limpo (72k, n_faces=1), AMP. Concluído
em 2026-05-14.

**Baseline:** Exp 5 (CE + AdamW + Cosine + dropout=0.2) com head linear.
F1=0.665, IR=1.76 (no dataset original).

**Recipe:** combinação fixa de loss + optimizer + scheduler + dropout +
epochs. Define o ambiente de treino independente de topologia de head.

**Topology / Head topology:** estrutura do classificador final
(`MLPHead`): depth, width por camada, activation, norm, dropout. É o
que o Optuna otimiza no projeto.

**Win #N:** otimização específica de pipeline (#1 AMP, #2
persistent_workers, #4 pre-encode labels, #5 PIL.Image.open). Ver
HISTORICO seção 8.

**Sprint A/B/C/D:** fases do trabalho de refatoração antes do mestrado.
Ver HISTORICO seção 2.

**Diretriz N:** uma das 8 instruções do orientador na reunião de kickoff.
Ver HISTORICO seção 5 e
[meeting_2026-05-11_kickoff.md](meeting_2026-05-11_kickoff.md).

**Opção A/B/C/D:** alternativas analisadas para limpeza do dataset (ver
HISTORICO seção 10). Opção A foi adotada.

**Bug §X.Y:** referência aos bugs identificados na revisão do código MBA
em `REVIEW_AND_PLAN.md`.

---

## Como usar este glossário

- **Para um termo desconhecido:** Ctrl+F neste documento.
- **Para entender o contexto de uso:** consulte o documento referenciado
  (links explícitos onde aplicável).
- **Para a tese:** este glossário pode ser exportado direto como apêndice
  ou usado para construir o glossário do documento formal.
