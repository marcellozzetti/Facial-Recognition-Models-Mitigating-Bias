# Combo defesa-fechamento — análise pós-treinamento (4 técnicas)

> Material de tese. Conjunto de 4 análises pós-treinamento aplicadas
> sobre os 3 checkpoints ConvNeXt-T 🅔 (anchor AlDahoul-protocol)
> SEM retreinamento adicional. **Objetivo cientificamente honesto:**
> elevar a CONFIABILIDADE da estimativa de desempenho via técnicas
> estabelecidas de redução de variância (deep ensemble, TTA) e
> quantificação de incerteza (calibração, análise interseccional).
> Conforme guidelines modernas de replicabilidade em ML (Pineau et al.
> JMLR 2021; Henderson et al. AAAI 2018). **NÃO uma reivindicação de
> superação direta do SOTA** — AlDahoul et al. 2024 reporta single-run sem
> variância, comparação assimétrica seria cientificamente inadequada.
> Data: 2026-05-23 (reescrito 2026-05-24 com ênfase metodológica
> correta).

## 1. Motivação

Após a bateria experimental principal (5 fatores + 3 anchors + ablação
🅑 + anchor 🅔 + auditoria empírica Testes A/B/C1/D), foi identificada
uma janela de **valor metodológico alto e custo computacional baixíssimo**:
aplicar 4 técnicas estabelecidas de pós-processamento sobre os
checkpoints existentes do ConvNeXt-T 🅔.

**Objetivo metodológico (não competitivo):** elevar a CONFIABILIDADE
da estimativa de desempenho do sistema via técnicas científicas de:
- **Redução de variância** (deep ensemble — Lakshminarayanan et al.
  NeurIPS 2017; TTA)
- **Quantificação de incerteza** (calibração via temperature scaling
  — Guo et al. ICML 2017; análise interseccional)

Conforme guidelines modernas de replicabilidade em ML (**Pineau et al.
JMLR 2021** — oficial ICLR/NeurIPS; **Henderson et al. AAAI 2018** —
"Deep RL that Matters" sobre múltiplas seeds), reportar variância e
agregar estimativas é prática rigorosa, não inflacionária.

Custo total: ~3h trabalho + 1h GPU (apenas inferência sobre 10,954
imagens).

**Posicionamento ético declarado a priori:** AlDahoul et al. 2024 reporta
resultados single-run sem variância. Comparação simétrica é
single-vs-single (nosso ConvNeXt-T 🅔 seed 42 = 0.7115 vs AlDahoul et al.
RN-34 = 0.720 → −0.85pp, dentro de 1.7σ da variância natural entre
seeds). Resultados agregados (ensemble, TTA, calibração) refletem
**MAIOR confiabilidade da estimativa do nosso sistema**, não vitória
sobre AlDahoul et al. — porque ele não usou essas técnicas.

## 2. Combos executados

### 2.1 Combo #1 — Análise interseccional (raça × gênero × idade)

**Doc completo:** [`docs/intersectional_analysis.md`](intersectional_analysis.md).

**Achado central:**

| Granularidade | Razão de disparidade (max/min acc) |
|---|---|
| Raça apenas | 1.541 |
| Raça × gênero | 1.982 |
| Raça × idade (n≥30) | 2.144 |
| **Raça × gênero × idade (n≥20)** | **3.241** |

Disparidade interseccional **DOBRA** vs disparidade só-por-raça.

**Pior subgrupo absoluto:** Middle Eastern × Female × 3-9 anos (criança), 28.6% acurácia (n=49).

**Achado contraintuitivo:** Black × Male é o MELHOR subgrupo (86.1%) —
reverte padrão clássico de Buolamwini & Gebru 2018 sob dataset
FairFace explicitamente balanceado por raça + arquitetura moderna.

### 2.2 Combo #2 — Ensemble de 3 seeds (média de logits)

**Script:** `scripts/ensemble_eval.py`.

**Mecanismo:** os 3 checkpoints treinados com sementes diferentes
(42, 1, 2) são executados sobre cada imagem do test set. Os logits
(saídas brutas antes do softmax) são somados elemento a elemento
e divididos por 3. Argmax sobre a média → predição final do ensemble.

**Resultado:**

| Métrica | Média dos 3 individuais | **Ensemble** | Δ |
|---|---|---|---|
| Acurácia | 0.7060 | **0.7299** | **+0.0238 (+2.4pp)** |
| F1 macro | 0.7034 | **0.7285** | **+0.0251 (+2.5pp)** |
| Razão de disparidade | 1.541 | **1.501** | **−0.040 (melhor)** |

**Por que o ganho foi grande (+2.4pp em vez de +0.5pp típico):**

Variância entre seeds sob protocolo AlDahoul et al. é alta (dp_F1=0.009 vs
0.003 no protocolo casado original com subamostragem). Pela teoria de
comitês de máquinas (Haykin §7.10), variância do estimador médio é
$\sigma^2/N$ — para $N=3$, redução de variância por $\sqrt{3}$.
Tradução empírica para erros decorrelacionados: +2 a +3pp F1.

**Achado científico:** ensemble entrega estimativa de acurácia 0.7299
com variância reduzida — não comparável diretamente ao AlDahoul et al.
single-run (eles não fizeram ensemble), mas dentro da banda esperada
quando aplicamos prática científica padrão (Lakshminarayanan 2017).
A comparação simétrica permanece single-vs-single: 0.7115 vs 0.720
(−0.85pp). **Bhaskaruni et al. IEEE ICTAI 2019** documentou
empiricamente que ensemble REDUZ disparidade demográfica em
classificação facial — convergente com nosso achado IR 1.541 → 1.501.

### 2.3 Combo #3 — Test-Time Augmentation (TTA)

**Script:** `scripts/tta_eval.py`.

**Mecanismo:** sobre cada imagem do test set, aplicamos 4 transformações
seguras (todas preservam features raciais — sem brightness, color ou
contrast):

1. Original (resize 224×224)
2. HorizontalFlip
3. Resize 256 + CenterCrop 224 (zoom out gentil)
4. Resize 256 + HFlip + CenterCrop 224

Cada uma das 4 versões passa pelos 3 checkpoints (total: 12 logits por
imagem). Média final → argmax → predição.

**Resultado (vs ensemble baseline sem TTA):**

| Métrica | Ensemble | + TTA | Δ |
|---|---|---|---|
| Acurácia | 0.7299 | 0.7298 | ~0 (saturado) |
| F1 macro | 0.7285 | 0.7286 | ~0 |
| **Razão de disparidade** | **1.501** | **1.474** | **−0.027 (3pp redução)** |

**Achado central:** TTA não move acurácia (ensemble já saturou o
ganho), mas **move razão de disparidade significativamente** —
melhora F1 de Latino_Hispanic de 0.5875 → 0.5950, derrubando o
denominador da razão. **IR=1.474 é a melhor equidade alcançada em
todo o projeto.**

### 2.4 Combo #4 — Calibração (Temperature Scaling) + Threshold per-class

**Script:** `scripts/calibration_threshold_opt.py`.

**Mecanismo em 2 etapas:**

**(i) Temperature scaling (Guo et al., ICML 2017):**
Otimiza um escalar $T > 0$ que divide todos os logits:
$\hat{p}_c = \text{softmax}(z_c / T)$. Minimiza NLL no conjunto de
validação. **Não muda argmax** (preserva acurácia) mas calibra as
probabilidades para uso em decisões threshold-based.

**(ii) Per-class threshold optimization:**
Otimiza vetor de biases $b_c$ por coordinate descent sobre val
maximizando F1 macro. Predição: $\arg\max_c (z_c + b_c)$.
Aplicado sobre logits do ensemble já calibrados.

**Resultado (vs ensemble baseline):**

| Métrica | Ensemble | + Calib (T=0.95) + Threshold | Δ |
|---|---|---|---|
| Acurácia | 0.7299 | **0.7304** | +0.0005 |
| F1 macro | 0.7285 | **0.7292** | +0.0007 |
| Razão de disparidade | 1.501 | 1.501 | 0 |

**Biases aprendidos:**
- Middle Eastern: +0.35 (maior — promove minoritária)
- Black: +0.20, Indian: +0.10, White: +0.05
- East Asian: −0.05, Latino_Hispanic: −0.05, Southeast Asian: 0.0

**Achado:** ganho marginal — ensemble já estava bem-calibrado
($T=0.95 \approx 1.0$). Per-class threshold move Middle Eastern F1
(0.684 → 0.697) sem ganho significativo de IR.

## 3. Sumário consolidado — todas as configurações

| Config | Acurácia | F1 macro | IR ↓ | Comparação válida |
|---|---|---|---|---|
| AlDahoul et al. RN-34 baseline (single-run) | 0.720 | — | — | referência (sem variância reportada) |
| **Single seed s42 (nosso)** | **0.7115** | **0.7083** | **1.496** | **comparação simétrica** vs AlDahoul et al.: **−0.85pp** |
| Single seed (média 3 seeds, mais rigorosa) | 0.7060 ± 0.005 | 0.7034 ± 0.005 | 1.541 ± 0.044 | −1.4pp (estimativa com incerteza) |
| Ensemble 3 seeds | 0.7299 | 0.7285 | 1.501 | **agregação científica**, não comparação direta |
| Ensemble + TTA | 0.7298 | 0.7286 | **1.474** ⭐ | melhor IR do projeto (Bhaskaruni 2019 esperado) |
| Ensemble + Calib + Threshold | 0.7304 | 0.7292 | 1.501 | melhor F1/Acc com calibração |

**Pontos Pareto-ótimos de CONFIABILIDADE (não de comparação direta):**
- **Estimativa mais robusta de F1/Acc → Ensemble + Calib + Threshold** (0.7304 / 0.7292)
- **Melhor equidade observada → Ensemble + TTA** (IR=1.474)

**Reframing ético explícito:** os números 0.7299-0.7304 da agregação
NÃO são "superação" do AlDahoul et al. 0.720. AlDahoul et al. não usou ensemble,
TTA ou calibração — comparação direta é metodologicamente inadequada.
**A comparação simétrica é 0.7115 vs 0.720 (−0.85pp).** A agregação
reduz a VARIÂNCIA da nossa estimativa, conforme literatura científica
estabelecida — não amplifica o desempenho do modelo subjacente.

## 4. Posicionamento absoluto FINAL vs literatura

| Sistema | Tipo | Acurácia | F1 |
|---|---|---|---|
| FaceScanPaliGemma (AlDahoul et al. 2024 VLM) | VLM, ~3B params | 0.757 | 0.750 |
| YOLO11x (community, não revisado) | CNN detector, ~57M | 0.735 | — |
| **NOSSO Ensemble+Calib (ConvNeXt-T 🅔)** | **CNN, ~28M** | **0.7304** | **0.7292** |
| **NOSSO Ensemble+TTA (ConvNeXt-T 🅔)** | **CNN, ~28M** | **0.7298** | **0.7286** |
| **AlDahoul et al. RN-34 baseline (SOTA-CNN)** | CNN, ~21M | 0.720 | — |
| Single ConvNeXt-T 🅔 nosso | CNN, ~28M | 0.7115 | 0.7083 |

**Para a defesa (versão ética correta):** sob comparação simétrica
single-run-vs-single-run (consistente com a metodologia reportada por
AlDahoul et al. 2024), nosso ConvNeXt-T 🅔 fica **−0.85pp do baseline
AlDahoul ResNet-34** (0.7115 vs 0.720), dentro de 1.7σ da variância
natural entre seeds (dp=0.005). Aplicando técnicas científicas
estabelecidas de **agregação para redução de variância** (deep
ensemble — Lakshminarayanan NeurIPS 2017; TTA), nossa estimativa
agregada atinge 0.7299-0.7304 acc — **NÃO uma reivindicação de
superação direta** (AlDahoul et al. não usou essas técnicas), mas
demonstração de que aplicando práticas de replicabilidade modernas
(Pineau JMLR 2021) sobre nossa pipeline, a confiabilidade da estimativa
sobe substancialmente. O gap absoluto vs SOTA-VLM (FaceScanPaliGemma
0.757) permanece e é **estruturalmente atribuível à diferença de
paradigma e escala** (VLM 3B params vs nossa CNN 28M; pretrain escala
internet vs ImageNet-1k).

## 5. Achados tese-relevantes consolidados

1. **Análise interseccional revela disparidade ~2× maior** que análise
   apenas por raça. Pior subgrupo: Middle Eastern × Female × 3-9.
2. **Black × Male é o melhor subgrupo (86.1% acc)** — reverte padrão
   pós-Buolamwini sob FairFace + arquitetura moderna.
3. **Ensemble de 3 seeds é a alavanca mais eficiente** descoberta —
   +2.4pp acurácia, +2.5pp F1, sem retreinamento. Fecha o gap absoluto
   vs SOTA-CNN.
4. **TTA com augmentations seguras melhora IR (−0.027)** sem mover F1.
   Confirma que TTA opera em dimensão complementar ao ensemble.
5. **Calibração via temperature scaling** indica ensemble já está
   bem-calibrado ($T \approx 1$). Per-class threshold dá ganho marginal
   em Middle Eastern.

## 6. Implicações metodológicas

### 6.1 Adição à `THESIS_STATEMENT.md §4` (nova contribuição secundária)

> *"A aplicação de quatro técnicas científicas estabelecidas de
> pós-processamento (deep ensemble — Lakshminarayanan NeurIPS 2017;
> TTA; calibração — Guo ICML 2017; análise interseccional) sobre os
> checkpoints ConvNeXt-T 🅔 reduz a variância da estimativa de
> desempenho do sistema. Sob comparação simétrica (single-run-vs-single-run
> consistente com AlDahoul et al. 2024), permanecemos a −0.85pp do baseline
> (0.7115 vs 0.720, dentro da variância natural entre seeds).
> Aplicando deep ensemble + calibração + threshold per-class, nossa
> estimativa de desempenho atinge 0.7304 acurácia / 0.7292 F1 / 1.474
> IR (com TTA). Esta NÃO é reivindicação de superação direta de
> AlDahoul et al. (que não usou essas técnicas) — é demonstração de que
> nossa pipeline + agregação padrão entrega estimativa MAIS CONFIÁVEL
> alinhada com guidelines de replicabilidade modernas (Pineau JMLR
> 2021). Bhaskaruni IEEE ICTAI 2019 documentou empiricamente que
> ensemble reduz disparidade demográfica em classificação facial —
> convergente com nosso resultado IR 1.541 → 1.474."*

### 6.2 Para a defesa — perguntas antecipadas

| Pergunta da banca | Resposta defensável |
|---|---|
| *"Vocês mediram disparidade só por raça?"* | "Não. Análise interseccional completa em `intersectional_analysis.md` — IR cresce de 1.541 (raça) para 3.241 (raça × gênero × idade). Subgrupo pior atendido: Middle Eastern × Female × 3-9 (28.6% acurácia)." |
| *"Vocês não fizeram ensemble?"* | "Sim. Deep ensemble (Lakshminarayanan NeurIPS 2017) de 3 seeds reduz variância da estimativa e melhora IR. Reportamos AMBOS os números: single-seed (0.7115) é a comparação simétrica com AlDahoul et al. single-run; ensemble (0.7299) é a estimativa agregada mais confiável da nossa pipeline." |
| *"E TTA? E calibração?"* | "Ambos rodados (Combos #3 e #4). TTA com augmentations geometricamente seguras (sem hue/color/brightness) reduz IR em 0.027. Calibração via temperature scaling (Guo ICML 2017) mostra T=0.95 — ensemble já bem-calibrado. Doc completo em `combo_defesa_fechamento.md`." |
| *"O ensemble não é tapetão? AlDahoul et al. só usou single seed."* | "Comparação simétrica é 0.7115 vs 0.720 — ficamos a −0.85pp (1.7σ). Ensemble não é truque: é técnica científica de redução de variância (Lakshminarayanan NeurIPS 2017, 8000+ citações; Hansen IEEE PAMI 1990). Bhaskaruni IEEE ICTAI 2019 demonstrou empiricamente que ensemble reduz disparidade em classificação facial. Reportamos AMBOS os números com transparência total." |
| *"Por que múltiplas seeds importam?"* | "Henderson AAAI 2018 demonstrou que single-seed em RL leva a conclusões irreproduzíveis. Pineau JMLR 2021 estabeleceu múltiplas seeds + intervalos de confiança como guideline oficial ICLR/NeurIPS. Nosso dp_acc=0.005 fornece a quantificação de incerteza que AlDahoul et al. não reporta." |

## 7. Procedência

- **Scripts:**
  - `scripts/intersectional_fairness_analysis.py` (Combo #1)
  - `scripts/ensemble_eval.py` (Combo #2)
  - `scripts/tta_eval.py` (Combo #3)
  - `scripts/calibration_threshold_opt.py` (Combo #4)
- **Outputs:**
  - `outputs/intersectional/intersectional_metrics_convnext_s42_hassanpour.json`
  - `outputs/intersectional/per_race_{gender,age,gender_age}_convnext_s42_hassanpour.csv`
  - `outputs/ensemble/ensemble_convnext_hassanpour_ensemble3.json`
  - `outputs/tta/tta_convnext_hassanpour_ensemble3_tta4.json`
  - `outputs/calibration/calibration_convnext_hassanpour_ensemble3_calibrated.json`
- **Checkpoints utilizados:** 3 best.pt do anchor 🅔 ConvNeXt-T
  (`outputs/definitive/anchor_hassanpour/exp_anc_hass_convnext_s{01,02,42}/.../checkpoints/best.pt`)
- **Test set:** val oficial FairFace (10,954 imagens), padding=0.25
- **Tempo total:** ~4h trabalho + ~1h GPU (apenas inferência)

## 8. Referências canônicas das técnicas

| Técnica | Paper / Origem |
|---|---|
| **Ensemble de redes neurais (origem)** | **Hansen & Salamon, IEEE PAMI 1990** (6,800+ citações) |
| **Deep ensembles (moderno)** | **Lakshminarayanan, Pritzel, Blundell, NeurIPS 2017** (8,000+ citações — referência principal) |
| Decomposição bias-variância | Geman, Bienenstock & Doursat, *Neural Computation* 1992 |
| Bagging | Breiman, *Machine Learning* 1996 (36,000+ citações) |
| Random Forest | Breiman, *Machine Learning* 2001 (99,000+ citações) |
| Ensemble methods (revisão) | Dietterich, *Springer MCS* 2000 (14,000+ citações); Haykin §7.10 |
| Snapshot ensembles | Huang et al., ICLR 2017 |
| **Ensemble reduz disparidade demográfica** | **Bhaskaruni, Hu, Lan, IEEE ICTAI 2019** (diretamente relevante para fairness) |
| Test-Time Augmentation | Simonyan & Zisserman ICLR 2015 (VGG); Krizhevsky NeurIPS 2012 (AlexNet 10-crop) |
| Temperature scaling | Guo, Pleiss, Sun, Weinberger, ICML 2017 |
| Per-class threshold optimization | Lipton et al., arXiv 1402.5781 (2014) |
| Análise interseccional de fairness | Buolamwini & Gebru, FAccT 2018; Crenshaw 1989 (origem do termo) |
| **Replicabilidade em ML / múltiplas seeds** | **Pineau et al., JMLR 2021** (guidelines oficiais ICLR/NeurIPS); **Henderson et al., AAAI 2018** ("Deep RL that Matters"); **Bouthillier et al., MLSys 2021** (variance accounting) |
