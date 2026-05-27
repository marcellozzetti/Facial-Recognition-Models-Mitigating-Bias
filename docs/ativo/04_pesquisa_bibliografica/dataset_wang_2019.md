---
name: dataset-wang-2019
status_verificacao: VERIFIED
autores: [Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, Yaohai Huang]
ano: 2019
titulo: "Racial Faces in-the-Wild: Reducing Racial Bias by Information Maximization Adaptation Network"
venue: "IEEE/CVF International Conference on Computer Vision (ICCV)"
tipo_publicacao: conference
arxiv_id: "1812.00194"
doi: null
url_primario: https://arxiv.org/abs/1812.00194
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~70
lente_disrupcao: cobertura
fonte_leitura: PDF integral extraído via pypdf (pdfs/wang_2019_rfw.pdf)
---

# RFW: Racial Faces in-the-Wild (Wang, Deng, Hu, Tao & Huang, 2019)

## 1. Resumo do problema atacado

Em 2019, datasets benchmark de **face recognition** (FR) usados na era
de deep learning (LFW, IJB-A) eram fortemente desbalanceados em raça
(LFW: 69.9% Caucasian; IJB-A: 66.0% Caucasian). Datasets de
treinamento estavam ainda mais enviesados (CASIA-WebFace: 84.5%
Caucasian; VGGFace2: 74.2%; MS-Celeb-1M: 76.3%). Sem benchmark
balanceado, era **impossível medir racial bias em FR profundo**. O
paper contribui: (i) um dataset balanceado para auditoria de viés em
verificação facial (RFW), (ii) evidência empírica de viés em 4 APIs
comerciais + 4 algoritmos SOTA, (iii) um método de mitigação não-
supervisionado (IMAN — Information Maximization Adaptation Network).

**Tarefa central: face verification 1:1**, não classification 1:N.
**Importante:** RFW **não é** dataset de classificação de raça — é
dataset de auditoria de FR balanceado por raça.

## 2. Método

### 2.1 Construção do RFW

- **Fonte:** MS-Celeb-1M (não download de web open).
- **Atribuição de raça:**
  - **Asians e Indians**: atributo "Nationality" do FreeBase celebrity
    knowledge base (direto, autoritativo).
  - **Caucasians e Africans**: Face++ API (estimação).
- **Aceitação de identidade**: aceita só se **a maioria das imagens da
  identidade** for estimada como mesma raça pelo Face++.
- **Verificação manual**: imagens com baixa confiança Face++ foram
  inspecionadas manualmente para mitigar viés do próprio Face++.

### 2.2 Estrutura do dataset

- **4 subsets, um por raça**: Caucasian, Asian, Indian, African.
- **Cada subset**: ~10K imagens de **~3K identidades**.
- **Total**: ~40K imagens, ~12K identidades.
- **Balanceamento**: 25% cada (vs 13.2-69.9% nos benchmarks anteriores
  por categoria).
- **Pose, idade, gênero**: distribuições comparáveis entre raças
  (verificadas via Face++ — Fig 3a-d).
- **Sem overlap** com CASIA-WebFace e VGGFace2 (verificado
  manualmente via embedding ArcFace).

### 2.3 Protocolos de avaliação

- **ROC sobre todos os pares**: ~14K positivos vs ~50M negativos (3K
  identidades × N).
- **LFW-like 6K pares difíceis**: pré-selecionados por cosine
  similarity para **evitar saturação**. Os pares incluem variações
  intra-individual e similaridade inter-individual altas.

### 2.4 Método IMAN (mitigação via domain adaptation)

- **Setup:** UDA (Unsupervised Domain Adaptation). Caucasian = source
  (com rótulos), other races = target (sem rótulos).
- **Desafio único em FR:** classes (identidades) não-sobrepõem entre
  domínios. Métodos UDA padrão (que assumem classes compartilhadas)
  falham.
- **Solução 2-etapas:**
  1. **Pseudo-adaptation**: clustering espectral sobre features deep
     gera pseudo-labels para target; fine-tuning com Softmax.
  2. **MI-adaptation**: Mutual Information loss (estilo InfoGAN)
     maximiza I(X_t; O_t) → aprende margens maiores em
     representação sem labels.
- **Loss MI:**
  L_MI = H(O_t | X_t) − γ H(O_t)
  Primeiro termo: classifier output concentrado em uma classe;
  segundo termo: distribuição marginal uniforme (evita colapso).

## 3. Datasets e setup experimental

- **Test sets**: RFW (4 subsets), GBU (good/bad/ugly), IJB-A.
- **Training (source)**: CASIA-WebFace (rotulado).
- **Training (target)**: subset balanceado por raça (race-balanced
  training set).
- **Backbones**: ResNet-34, baselines com Center-loss, Sphereface,
  ArcFace, VGGFace2.

## 4. Métricas reportadas

- **Verification accuracy (%)** em 6K pares difíceis por raça
  (LFW-like protocol).
- **ROC AUC** sobre 14K pos vs 50M neg pairs (não tabulado no
  excerto lido — fica no supplementary).

## 5. Resultados principais (valores numéricos)

### 5.1 Tabela 1 — Viés em APIs comerciais e SOTA (verification accuracy %)

| Sistema | Caucasian | Indian | Asian | African |
|---|---|---|---|---|
| **APIs comerciais** | | | | |
| Microsoft | 87.60 | 82.83 | 79.67 | 75.83 |
| Face++ | 93.90 | 88.55 | 92.47 | 87.50 |
| Baidu | 89.13 | 86.53 | 90.27 | 77.97 |
| Amazon | 90.45 | 87.20 | 84.87 | 86.27 |
| **mean** | **90.27** | **86.28** | **86.82** | **81.89** |
| **SOTA algorithms** | | | | |
| Center-loss | 87.18 | 81.92 | 79.32 | 78.00 |
| SphereFace | 90.80 | 87.02 | 82.95 | 82.28 |
| ArcFace | 92.15 | 88.00 | 83.98 | 84.93 |
| VGGFace2 | 89.90 | 86.13 | 84.93 | 83.38 |
| **mean** | **90.01** | **85.77** | **82.80** | **82.15** |

**Achados-bandeira:**

- **Gap consistente Caucasian > Asian/African** em todas as 8 sistemas.
- **Erro em African ≈ 2× erro em Caucasian**: e.g., Microsoft tem
  err_Cauc=12.4%, err_Afr=24.2% → ratio ~1.95×.
- **A hierarquia é estável across APIs e algorithms** (não é ruído de
  um sistema isolado).

### 5.2 Achado secundário (importante para nossa pesquisa)

> *"Some specific races are inherently more difficult to recognize
> even trained on the race-balanced training data."*

Ou seja: **mesmo com treino balanceado**, persistem gaps. Concorda
com FairFace (Latino .247) e contrasta com a hipótese simples
"basta balancear os dados".

### 5.3 IMAN reduz mas não elimina o gap

Ablation mostra que IMAN reduz o gap, especialmente a contribuição da
MI-loss. Números absolutos para a tabela final não foram lidos no
excerto, mas a melhoria é incremental sobre o baseline.

## 6. Limitações declaradas pelos autores

- **Taxonomia 4-class apenas** (Caucasian, Asian, Indian, African).
  Não inclui Hispanic/Latino, Middle Eastern, ou subgrupos Asiáticos
  (East vs Southeast).
- **IMAN é unsupervised**: pseudo-labels via clustering podem ser
  ruidosos.
- **Race assignment via Face++** introduz viés circular (Face++ é um
  dos sistemas que o paper audita).
- **Caucasian como source** é uma escolha pragmática (mais dados
  disponíveis), mas reforça padrão "Caucasian = default" criticado em
  outras partes da literatura.

## 7. Limitações que identifiquei (leitura crítica)

- **Confusão tarefa-essencial:** RFW mede **verification (1:1)**, não
  **classification (1:N)** ou **race classification**. Trabalhos
  subsequentes às vezes citam RFW como se fosse análogo a FairFace,
  o que é incorreto:
  - RFW = pares para auditar embeddings de FR.
  - FairFace = imagens rotuladas por raça para auditar classifiers de
    atributo facial.
- **Taxonomia 4-class é coarse** comparado às 7 do FairFace.
  Especificamente, "Asian" do RFW mistura East e Southeast Asians do
  FairFace.
- **Race assignment via Face++** é problemática (já mencionado).
  Embora os autores incluam verificação manual, **inter-annotator
  agreement não é reportado**.
- **Difficult pairs selection via cosine similarity** garante não-
  saturação mas introduz viés: os pares são "difíceis para o modelo
  de seleção" (ArcFace), não necessariamente difíceis em sentido
  absoluto.
- **IMAN testado em conjunto de baselines de 2018-2019**; nada
  garante que o ganho permaneça com backbones modernos (ConvNeXt,
  ViT, DINOv2).
- **Métrica é accuracy de verification @ threshold fixo**, não
  TAR@FAR (true accept @ false accept), que é padrão em biometria.

## 8. Relação com nossa pesquisa

**Centralidade:** RFW é **dataset complementar ao FairFace**, com 3
propriedades úteis para triangulação:

1. **Mesmo problema (viés racial em sistema facial), tarefa diferente
   (verificação vs classificação).** Permite separar achados que são
   **artefatos de classification** vs **invariantes ao tipo de tarefa
   facial**.
2. **Razão de erro ~2× African/Caucasian em FR** estabelece magnitude
   de gap esperada **independente de classification**. Nossa razão
   de disparidade de F1 (max/min) em classification 7-class pode ser
   contextualizada contra este baseline.
3. **Conclusão "race-balanced training não elimina bias"** é evidência
   convergente com FairFace Tabela 6 (White .928 vs Latino .247).
   Reforça motivação para mitigação algorítmica além do balanceamento.

**Limitação de transposição:**

- Não usamos RFW como dataset experimental (nossa pesquisa fica
  in-domain FairFace).
- A taxonomia 4-class do RFW não permite mapeamento direto às 7 do
  FairFace.
- Citação será **contextual**, não experimental.

## 9. Pontos para citar / posicionar

- *"Em paralelo ao FairFace, Wang et al. (2019) introduziram o
  dataset Racial Faces in-the-Wild (RFW), balanceado em 4 categorias
  raciais (Caucasian, Asian, Indian, African), para auditoria de
  sistemas de verificação facial."*
- *"A auditoria de Wang et al. (2019) sobre 4 APIs comerciais
  (Microsoft, Face++, Baidu, Amazon) e 4 algoritmos SOTA documenta
  que a taxa de erro de verificação em faces africanas é
  aproximadamente o dobro da taxa em faces caucasianas — um padrão
  consistente entre todos os sistemas testados."*
- *"Wang et al. (2019) observam que 'algumas raças permanecem
  intrinsecamente mais difíceis de reconhecer, mesmo com dados de
  treinamento balanceados' — conclusão convergente com Kärkkäinen &
  Joo (2021), que reporta disparidade severa entre White e Latino
  no FairFace mesmo balanceado. Esta evidência cruzada motiva
  abordagens algorítmicas de mitigação que vão além do balanceamento
  de dados."*

## 10. Arquivos relacionados

- PDF local: `pdfs/wang_2019_rfw.pdf` (gitignored).
- Texto extraído: `pdfs/wang_2019_rfw.txt` (gitignored).
- Entradas relacionadas: [[dataset_karkkainen_2021]] (FairFace —
  análogo em classification),
  [[buolamwini_2018]] (Gender Shades — predecessor em commercial
  auditing),
  [[grother_2019]] (NIST FRVT — auditoria industrial mais ampla, com
  variação demográfica).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S8.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Conclusion:

- **Estender RFW para mais categorias raciais** — 4 é coarse;
  Middle Eastern, Latinx, Southeast Asian estão ausentes. ⚠ Alinhada
  com Q08 (debate sobre granularidade).
- **Aplicar IMAN a outras tarefas** além de FR verification. ❌
  Tangencial.
- **Combinar IMAN com técnicas supervisionadas** — atualmente UDA
  puro. ⚠ Direção combinada.
- **Investigar por que mesmo race-balanced training não elimina
  disparities** — autores reconhecem mas deixam aberto. ✅ **Alinhada
  com Q10** (matriz skin tone × race) — hipótese fenotípica
  estrutural.
- **Modernizar backbones** — ResNet-34 e Center-loss/SphereFace/
  ArcFace são de 2018-2019. ✅ **Alinhada com Q06**.
- **Validar IMAN em settings com mais raças** — paper foca em
  Caucasian → other races. Generalização não testada. ❌ Fora do
  escopo (RFW é 4-class).
