# Fator 3 (família de loss) — resultados 3-seed casados

> Material de tese. Isola a contribuição da **função de perda** para
> acurácia e disparidade demográfica, sob protocolo casado (dataset
> limpo, ResNet-50 ImageNet, 3 seeds {42,1,2}, mesma base, fp32,
> num_workers=0). Data: 2026-05-19.

## 1. Tabela principal (critério de seleção CORRETO: max val_f1_macro)

Recompute casado via `history.json` de todos os runs
(`scripts/recompute_checkpoint_criterion.py`). Controle (CE+linear) e
referência (ArcFace) reaproveitados do batch `dataset_factor` sob
protocolo idêntico.

| Loss (limpo, casado, 3 seeds) | Acc | **F1 macro** | disparity_ratio (f1) ↓ |
|---|---|---|---|
| **CE + linear** (controle) | 0.6865 ± 0.0019 | **0.6877 ± 0.0017** | **1.697 ± 0.033** |
| ArcFace | 0.5717 ± 0.0059 | 0.5486 ± 0.0086 | 3.301 ± 0.365 |
| AdaFace | 0.6693 ± 0.0088 | 0.6709 ± 0.0079 | 1.709 ± 0.007 |
| MagFace | 0.6718 ± 0.0041 | 0.6732 ± 0.0060 | 1.756 ± 0.057 |

## 2. Veredito (regra do 1σ, comparação casada)

1. **ArcFace é o único perdedor real.** F1 0.55 vs ~0.67–0.69
   (Δ≈0.14 ≫ 1σ) e disparidade muito pior (3.30 vs ~1.70 ≫ 1σ).
   Robusto nos 3 seeds.
2. **CE ≈ AdaFace ≈ MagFace.** Agrupados em F1 ~0.67–0.69 e
   disparity_ratio ~1.70–1.76. CE marginalmente melhor em F1 (~2σ sobre
   Ada/Mag — pequeno mas tecnicamente significativo); **em equidade os
   três empatam dentro de 1σ**. AdaFace vs MagFace: Δ 0.002 →
   **não-significativo**.
3. **Contribuição marginal do fator loss:** trocar softmax+CE por uma
   loss de margem **não melhora** acurácia nem equidade nesta tarefa;
   ArcFace **piora** ambos. Achado de *atribuição*, alinhado à Linha A
   (e ao mapa de SOTA: losses de margem são de *verificação*, não
   transferem para classificação de atributo).

## 3. Achado metodológico embutido (delta candidato #2 da tese)

O critério ingênuo de best-epoch (`min val_loss`) contaminava o
resultado de forma **dependente da família de loss**:

| Loss | F1 `min val_loss` → `max val_f1` | Δ (severidade) |
|---|---|---|
| CE+linear | 0.667 → 0.688 | +0.021 |
| ArcFace | 0.543 → 0.549 | +0.006 |
| **AdaFace** | 0.525 → 0.671 | **+0.146** |
| **MagFace** | 0.655 → 0.673 | +0.118 |

Sem a correção, concluir-se-ia (falsamente) "CE esmaga as losses de
margem". A assimetria (Ada/Mag ≫ CE/Arc) é, por si só, evidência
empírica de que **ablação ingênua confunde efeito-do-fator com
efeito-de-critério-quebrado** — núcleo da contribuição metodológica.
Ver [checkpoint_criterion_audit.md](checkpoint_criterion_audit.md),
[magface_diagnosis.md](magface_diagnosis.md).

## 4. Caveat de honestidade (não invalida o ranking)

Números **val-best-epoch** (seleção em validação, padrão). Os runs do
critério antigo (CE, ArcFace, adaface×3, magface_s01) early-stoparam em
`val_loss` cedo → `history.json` truncado → o melhor-F1 deles é **piso,
possivelmente subestimado**; magface s02/s42 (critério novo) treinaram
até convergir (não truncados). Logo:

- O **ranking qualitativo é robusto** (ArcFace perde; CE≈AdaFace≈MagFace).
- A **tabela definitiva** exige o **re-run casado** com critério
  correto + early-stop consistente em todos os fatores (em curso).

## 5. Procedência

- Configs: `configs/experiments_factor3/exp_f3_{adaface,magface}_s{01,02,42}.yaml`
- Saídas: `outputs/factor3/` (6 runs, results.json + history.json)
- Recompute: `scripts/recompute_checkpoint_criterion.py`
- Baseline casado: `outputs/dataset_factor/exp_r2base_exp0{5,6}_*`
- MagFace: forma canônica pós-fix (`λ_g=5`, margem magnitude-detached) —
  ver [magface_diagnosis.md](magface_diagnosis.md).
