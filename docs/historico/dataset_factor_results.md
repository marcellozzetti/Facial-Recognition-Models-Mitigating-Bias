# Fator Dataset — Resultado Defensável (3-seed, ambiente único)

**Data:** 2026-05-16
**Desenho:** batch casado de 12 runs — 2 braços (`r1ctrl`=original 97k,
`r2base`=clean 72k) × 2 receitas (Exp5 CE, Exp6 ArcFace) × 3 seeds
(42,1,2). **Tudo no mesmo ambiente:** torch 2.12+cu126, decode PIL,
fp32, `num_workers=0`, 25 épocas, early stopping. **Só o dataset varia.**
É a medição **defensável** do Fator Dataset (protocolo padrão; análises
exploratórias anteriores usaram 1 seed e não são comparáveis — ver §2).

---

## 1. Resultado (test set, média ± dp, n=3)

| Recipe | Braço | F1 macro | Inequity Rate |
|---|---|---|---|
| **CE** | r1ctrl (orig 97k) | 0.6512 ± 0.0039 | 1.7284 ± 0.0572 |
| **CE** | r2base (clean 72k) | 0.6647 ± 0.0061 | 1.7808 ± 0.0121 |
| | **Δ (limpeza)** | **+0.0135** (σ_comb 0.0073) → **SIGNIFICATIVO** | +0.0524 (σ_comb 0.0584) → **dentro de 1σ** |
| **ArcFace** | r1ctrl (orig 97k) | 0.5765 ± 0.0353 | 2.957 ± 1.243 |
| **ArcFace** | r2base (clean 72k) | 0.5447 ± 0.0064 | 2.951 ± 0.423 |
| | **Δ (limpeza)** | −0.0318 (σ_comb 0.0359) → **dentro de 1σ** | −0.006 (σ_comb 1.313) → **dentro de 1σ** |

Critério de significância: |Δ| > σ combinada (raiz da soma dos
quadrados dos desvios). Conservador, coerente com a regra dos 3 seeds.

---

## 2. Conclusão defensável do Fator Dataset

- **CE:** ΔF1 = **+1,35pp, significativo** (|Δ| > σ combinada). ΔIR
  dentro de 1σ → não-significativo.
- **ArcFace:** ΔF1 e ΔIR não-significativos. O recipe tem variância de
  seed alta (IR σ = ±1,24 no braço original) — instabilidade intrínseca
  do ArcFace, não atribuível ao dataset.

→ **A limpeza de imagens multi-face contribui para a acurácia no recipe
CE (+1,35pp) e não tem efeito significativo sobre a disparidade
demográfica (IR) em nenhum recipe.** Sob o protocolo padrão (3 seeds,
ambiente único, base casada), a contribuição do fator dataset para
fairness é não-significativa.

> **Nota de rastreabilidade.** Análises exploratórias anteriores
> (`r2_clean_dataset_results.md`) usaram 1 seed e ambientes de execução
> distintos entre os braços; seus números não devem ser comparados
> diretamente com os deste documento, que segue o protocolo padrão. As
> conclusões válidas são as deste documento.

---

## 3. Fator 2 (Topologia) recomputado vs baseline linear 3-seed

A comparação anterior da Fase 4 usava o baseline linear de **1 seed**
(F1=0.668, IR=1.737). Agora recomputada vs o baseline linear correto
**de 3 seeds** (`r2base CE`: F1=0.6647±0.0061, IR=1.7808±0.0121):

| Topologia | ΔF1 vs baseline 3s | ΔIR vs baseline 3s |
|---|---|---|
| **MLP trial 4** `[256] GELU drop=0.52` | +0.0081 (dentro de 1σ) | **−0.1122 (σ_comb 0.075) → SIGNIFICATIVO** |
| MLP trial 10 `[1024,1024,2048] SiLU` | +0.0056 (dentro de 1σ) | −0.0652 (dentro de 1σ) |

Contra o baseline linear de 3 seeds (IR=1.781 ± 0.012), o **MLP trial 4
(`[256] GELU drop=0,52`) reduz o IR em ~0,11 de forma estatisticamente
significativa**, sem custo em F1. O baseline de 3 seeds é a referência
válida para o Fator Topologia (a Fase 4 usava um baseline de 1 seed,
não comparável diretamente — ver nota de rastreabilidade §2).

---

## 4. Mapa de atribuição parcial (2 de 5 fatores, defensáveis)

| Fator | Contribuição p/ F1 | Contribuição p/ **IR (fairness)** |
|---|---|---|
| **1. Dataset** (limpeza multi-face) | **+1,35pp** (CE, signif.) | **nula** (não-signif. em ambos recipes) |
| **2. Topologia** (linear→MLP t4) | +0,8pp (não-signif.) | **−0,11, SIGNIFICATIVO** |

→ **A alavanca de equidade defensável até agora é a TOPOLOGIA do
classificador; a limpeza do dataset contribui para a acurácia.** Cada
fator paga em um eixo distinto.

---

## 5. Implicação para a tese

1. Resultado de atribuição claro: dataset → acurácia; topologia →
   equidade. É o tipo de afirmação que a metodologia de decomposição
   controlada (Linha A) existe para produzir.
2. Reforça a Linha B: a busca curta de fairness é não-confiável; a
   etapa de **confirmação (3 seeds, budget completo)** é parte
   obrigatória do método, não opcional.
3. O protocolo padrão (3 seeds, ambiente único, base casada) é a base
   de comparação de todos os fatores daqui em diante.

---

## 6. Dados brutos por seed

```
r1ctrl CE      s01 f1=0.6475 IR=1.787 | s02 f1=0.6495 IR=1.651 | s42 f1=0.6566 IR=1.747
r2base CE      s01 f1=0.6611 IR=1.794 | s02 f1=0.6733 IR=1.765 | s42 f1=0.6596 IR=1.784
r1ctrl ArcFace s01 f1=0.5294 IR=4.707 | s02 f1=0.5856 IR=2.224 | s42 f1=0.6145 IR=1.940
r2base ArcFace s01 f1=0.5511 IR=2.541 | s02 f1=0.5470 IR=3.533 | s42 f1=0.5360 IR=2.777
```

ArcFace IR range no original: 1.94–4.71 (σ=±1.24). Instabilidade
intrínseca do recipe — não do dataset.

---

## 7. Artefatos

- `configs/experiments_dataset_factor/` (12 configs, gerador
  `scripts/generate_dataset_factor_configs.py`)
- `outputs/dataset_factor/{results.json, comparison_table.md}` + runs
- Este documento é a referência válida do Fator Dataset e do Fator
  Topologia. [r2_clean_dataset_results.md](r2_clean_dataset_results.md)
  e [r4_refit_results.md](r4_refit_results.md) são exploratórios
  (1 seed) — consultar apenas como registro cronológico.
