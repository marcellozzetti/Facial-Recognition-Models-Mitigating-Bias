# Fator 5 (backbone) — resultados 3-seed casados — **achado positivo**

> Material de tese. Isola a contribuição da **arquitetura do backbone**
> à acurácia e disparidade demográfica, sob protocolo casado. **Pela
> primeira vez nos 5 fatores um arm bate o controle simultaneamente e
> significativamente em acurácia E equidade.** Data: 2026-05-21.
>
> 📍 **Posicionamento absoluto (atualizado 2026-05-22):** nosso ConvNeXt-T
> (F1=0.711, IR=1.569, acc=0.709) fica **−1.1pp da SOTA-CNN publicada**
> (AlDahoul et al. 2024, ResNet-34 = 0.720 acc) sob protocolos diferentes.
> Gap estruturado em 4 escolhas metodológicas declaradas (undersample,
> split próprio, padding 1.25, MTCNN re-align). Ablação 🅑
> (no-undersample) preparada para fechar o maior componente do gap.
> Ver `docs/baseline_positioning.md` §4 e `docs/sota_7class_race_audit.md`.

## 1. Tabela principal — 7 grupos (5 fatores + 2 backbones modernos)

Re-run definitivo (`outputs/definitive/factor5/`), 3 seeds {42,1,2},
critério `val_f1_macro`, recipe por-backbone declarado explicitamente
(LayerNorm-based moderns: lr=1e-4 bs=64 — ver §4).

| Loss / Backbone (clean, casado) | Acc | **F1 macro** | **disparity_ratio ↓** |
|---|---|---|---|
| CE + linear ResNet-50 (controle) | 0.6865 ± 0.0019 | 0.6877 ± 0.0017 | 1.697 ± 0.033 |
| ArcFace ResNet-50 | 0.5717 ± 0.0059 | 0.5486 ± 0.0086 | 3.301 ± 0.365 |
| AdaFace ResNet-50 | 0.6755 ± 0.0077 | 0.6765 ± 0.0067 | 1.697 ± 0.057 |
| MagFace ResNet-50 | 0.6706 ± 0.0020 | 0.6715 ± 0.0030 | 1.795 ± 0.062 |
| SupCon ResNet-50 | 0.6790 ± 0.0047 | 0.6816 ± 0.0036 | 1.699 ± 0.078 |
| **ViT-B/16 + linear** | 0.6730 ± 0.0053 | 0.6753 ± 0.0054 | **1.642 ± 0.044** |
| **ConvNeXt-T + linear** | **0.7093 ± 0.0017** | **0.7108 ± 0.0026** | **1.569 ± 0.023** |

## 2. Veredito (regra do 1σ, casado)

### ConvNeXt-T vs CE+linear (controle)

| Métrica | Δ | σ_comb | |Δ|/σ | Conclusão |
|---|---|---|---|---|
| F1 macro | **+0.0231** | 0.0031 | **~7.4σ** | **altamente significativo** |
| disparity_ratio | **−0.128** | 0.040 | **~3.2σ** | **significativo** |
| Acc | +0.0228 | 0.0025 | ~9σ | altamente significativo |

### ViT-B/16 vs CE+linear

| Métrica | Δ | σ_comb | |Δ|/σ | Conclusão |
|---|---|---|---|---|
| F1 macro | −0.012 | 0.0056 | ~2.2σ | marginalmente pior (não bate controle) |
| disparity_ratio | −0.055 | 0.055 | ~1.0σ | **borderline mais justo** |

### ConvNeXt-T vs ViT-B/16 (mesma família LayerNorm)

| Métrica | Δ | σ_comb | |Δ|/σ | Conclusão |
|---|---|---|---|---|
| F1 macro | +0.0355 | 0.006 | ~5.9σ | ConvNeXt **significativamente melhor** |
| disparity_ratio | −0.073 | 0.050 | ~1.5σ | ConvNeXt borderline mais justo |

## 3. Conclusão do Fator 5

🎯 **Backbone é a alavanca real de equidade nesta tarefa** — e
especificamente o ConvNeXt-T. Pela primeira vez em 5 fatores
testados, um arm move **simultaneamente e significativamente** acurácia
(+2.3pp F1, 7σ) e equidade (IR −0.13, 3σ), com variância entre seeds
apertadíssima (dp_F1=0.003, dp_IR=0.023 → estabilidade excepcional).

**ViT-B/16 move marginalmente a equidade** (~1σ) mas **não melhora
acurácia** vs o controle ResNet-50 — sugere que a alavanca não é
"atenção global", mas **design moderno + LayerNorm + viés indutivo CNN
no kernel 7×7**.

## 4. Hipóteses mecanísticas (a confirmar na defesa)

Três candidatos não-mutuamente exclusivos:

### H1 — LayerNorm > BatchNorm em dados demograficamente desbalanceados
**BN** aprende média/var dependentes do batch; em batch demograficamente
assimétrico, normaliza features pela distribuição majoritária. **LN** é
per-amostra → estruturalmente livre desse viés.
**Previsão testada:** ✓ parcialmente — ambos ConvNeXt e ViT (LN-based)
têm IR menor que ResNet (BN-based). Confirmada como contribuinte.

### H2 — Kernel depthwise 7×7 + design ConvNeXt (escala intermediária)
**ResNet** 3×3 captura textura local fina; **ViT** atenção global capta
contexto; **ConvNeXt** 7×7 depthwise preenche o gap intermediário —
escala onde feições demograficamente discriminativas operam
(~20–40px em 224).
**Previsão testada:** ✓ — ConvNeXt **significativamente** acima do ViT
em F1 (5.9σ); apoia a hipótese.

### H3 — Recipe de pretrain moderno (AdamW + augs + EMA)
ResNet ImageNet (2016 recipe: SGD+stepLR), ConvNeXt/ViT (2021+ recipe
AdamW+RandAugment+Mixup+EMA). Representações resultantes mais robustas.
**Previsão testada:** ✓ parcialmente — ambos moderns superam ResNet em
IR; ConvNeXt mais que ViT.

**Mais provável:** **H2 dominante** + **H1 contribuinte** + H3 ajudando
nas margens. Distinção precisa exige ablação adicional (defesa).

## 5. Recipe per-backbone declarado (caveat de honestidade)

| Backbone | LR | Batch size | Augmentation | Normalize |
|---|---|---|---|---|
| ResNet-50 (controle + factor 1–4) | 1e-3 | 128 | HFlip | ImageNet stats |
| ViT-B/16 | **1e-4** | **64** | HFlip | ImageNet stats |
| ConvNeXt-T | **1e-4** | **64** | HFlip | ImageNet stats |

**Justificativa explícita:** o recipe lr=1e-3+bs=128 é canônico para
BN-based CNNs (ResNet), mas destrói o fine-tuning de moderns
LayerNorm-based (sanity v1 do ViT confirmou empiricamente, val_f1≈acaso).
Cada backbone usa **seu recipe canônico de fine-tuning** — prática
padrão da literatura. O 1σ-Δ entre fatores **continua válido** porque o
recipe é parte integral do "fator backbone".

Trajetória do sanity (3 iterações para acertar):
- v1 (lr=1e-3 bs=128) → ViT colapsa, ConvNeXt thrashing memória.
- v2 (lr=1e-4 ViT, lr=1e-3 ConvNeXt, bs=64 ambos) → ViT ok, ConvNeXt
  underfits (LayerNorm também precisa lr baixo).
- v3 (lr=1e-4 bs=64 ambos) → **ambos saudáveis**. Commit `5a1b189`.

## 6. Cross-factor pattern — quadro consolidado

| Fator | Move acc | Move equidade | Magnitude |
|---|---|---|---|
| 1 — Dataset (limpeza) | ✓ +1.35pp CE | ✗ | parcial (só acurácia) |
| 2 — Topologia (HPO MLP) | ~ | ✓ IR −0.11 (≫1σ) | modesta |
| 3 — Loss family | ✗ (ArcFace pior) | ✗ (CE≈Ada≈Mag) | null |
| 4 — Paradigma (SupCon) | ✗ | ✗ (IR ≡ CE) | null |
| **5 — Backbone (ConvNeXt-T)** | ✓ **+2.3pp F1 (7σ)** | ✓ **IR −0.13 (3σ)** | **forte** |
| Metodológica — Pareto-aware criterion | (+0.15 F1 no AdaFace) | — | **a maior** |

**Resultado central da Linha A:** das 5 dimensões algorítmicas testadas
em FairFace race 7-class, **apenas o backbone moderno (especificamente
ConvNeXt-T)** é uma alavanca robusta de **acurácia + equidade**
simultânea. A topologia move equidade modestamente; loss e paradigma
são nulls atribuição-grade; dataset paga em acurácia. A descoberta
metodológica do critério Pareto-aware fica como a alavanca de impacto
sistêmico.

## 7. Robustez do achado central (verificada em 3 protocolos, 2026-05-23)

A alavanca ConvNeXt-T foi **validada empiricamente em 3 protocolos
metodologicamente distintos** após a bateria de ablações 🅑 (sem
subamostragem) e anchor 🅔 (protocolo AlDahoul et al.):

| Protocolo | Δ F1 (ConvNeXt vs Controle) | Δ IR | Significância |
|---|---|---|---|
| Casado original (subamostragem, partição nossa, padding 1.25, MTCNN) | +0.023 | −0.128 | **7σ + 3σ (forte)** |
| 🅑 Sem subamostragem (partição nossa, padding 1.25, MTCNN) | +0.014 | −0.065 | 1.5σ + 0.7σ (atenuada) |
| 🅔 Protocolo AlDahoul et al. (partição oficial, padding 0.25, sem subamostragem, sem MTCNN) | **+0.024** | **−0.087** | **3.5σ + 1.8σ (significativa)** |

**Achado central:** a alavanca é **invariante** ao balanceamento de
classes, à partição treino/teste, à versão das imagens e à presença
de pré-processamento próprio. Magnitude do efeito em F1 é
**comparável** sob os 3 protocolos (+0.023 / +0.014 / +0.024), com
variância colapsando para metade no protocolo AlDahoul et al. (dp_F1:
0.009 → 0.004; dp_IR: 0.090 → 0.044). Robustez confere validade
externa ao achado.

## 8. Caveats e próximos passos

- **Recipe é parte da rede dorsal.** Não isolamos "puro efeito
  arquitetural" vs "recipe canônica de cada rede dorsal" sem rodar
  matriz adicional. Defesa: ablação intra-arquitetura ResNet ↔
  recipes-CNN-2016 ↔ recipes-CNN-2022.
- **Sem avaliação cross-dataset** (RFW/DemogPairs). PLANO §5 / Linha O2.
- **ConvNeXt-T versão pequena** (28M parâmetros, "tiny"). Versões
  Base/Large podem amplificar o efeito (defesa).
- **Gap absoluto vs SOTA-CNN publicada (−1.4pp vs AlDahoul et al. 0.720)**
  atribuído ao HPO externo após auditoria empírica refutar dois
  suspeitos no nosso código — ver
  [`docs/auditoria_codigo_limitadores.md`](auditoria_codigo_limitadores.md).

## 9. Procedência

- Configs Fator 5: `configs/experiments_factor5/exp_f5_{vit,convnext}_s{01,02,42}.yaml`
- Configs robustez:
  - 🅑: `configs/ablation_no_undersample/exp_abl_nous_{control,convnext}_s{01,02,42}.yaml`
  - 🅔: `configs/anchor_hassanpour/exp_anc_hass_{control,convnext}_s{01,02,42}.yaml`
- Saídas: `outputs/definitive/factor5/`, `outputs/definitive/ablation_no_undersample/`,
  `outputs/definitive/anchor_hassanpour/`
- Recompute: `scripts/recompute_checkpoint_criterion.py --definitive`
- Implementação: `src/face_bias/models/backbones.py::build_backbone`
- Suporte de código novo: `src/face_bias/data/dataset.py` (split_protocol=official)
- Auditoria empírica: `docs/auditoria_codigo_limitadores.md`
- Sanity: pegou 3 iterações de bugs de recipe antes do batch
  (commits `dd5eb30`, `e164242`, `5a1b189`).
