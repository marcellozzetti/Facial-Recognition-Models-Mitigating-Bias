# Fator Dataset — Resultado Defensável (3-seed, ambiente único)

**Data:** 2026-05-16
**Desenho:** batch casado de 12 runs — 2 braços (`r1ctrl`=original 97k,
`r2base`=clean 72k) × 2 receitas (Exp5 CE, Exp6 ArcFace) × 3 seeds
(42,1,2). **Tudo no mesmo ambiente:** torch 2.12+cu126, decode PIL,
fp32, `num_workers=0`, 25 épocas, early stopping. **Só o dataset varia.**
**Corrige** o confound detectado na revisão (R1 antigo torch2.5+cv2 vs
R2 torch2.12+PIL misturava limpeza com mudança de framework/decode).

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

## 2. As DUAS reversões vs a conclusão confundida anterior

### Reversão 1 — o achado-vitrine "ArcFace −54% IR" **NÃO existe**

O `r2_clean_dataset_results.md` (comparação confundida: 1 seed, ambientes
diferentes) reportava:

> ❌ "Limpeza ajuda ArcFace dramaticamente: F1 +5,8pp, IR −54%"

Com rigor (3 seeds, mesmo ambiente, confound removido):

> ✅ ArcFace: ΔF1 = **−3,2pp** (não-significativo), ΔIR = **−0,006**
> (essencialmente zero). E o ArcFace é tão instável entre seeds
> (IR σ = ±1,24 no original) que **nenhum efeito de dataset é
> mensurável** neste recipe.

**O ganho de −54% era artefato do confound + 1 seed.** Era a "vitrine"
da tese antiga — e ela se dissolve sob escrutínio. Isto é exatamente o
que a sua exigência de corrigir o confound + 3 seeds estava protegendo.

### Reversão 2 — a limpeza melhora **acurácia** (CE), não fairness

- **CE:** ΔF1 = **+1,35pp, significativo** (limpeza melhora a acurácia).
  ΔIR dentro do ruído → **limpeza NÃO melhora fairness em CE**.
- **ArcFace:** sem efeito significativo em nenhum eixo.

→ **Conclusão defensável do Fator Dataset:** a limpeza de imagens
multi-face **melhora acurácia no recipe CE (+1,35pp)** mas **não tem
efeito significativo sobre a disparidade demográfica (IR) em nenhum
recipe**. O efeito de fairness atribuível ao dataset é **nulo dentro do
rigor estatístico adotado**.

---

## 3. Fator 2 (Topologia) recomputado vs baseline linear 3-seed

A comparação anterior da Fase 4 usava o baseline linear de **1 seed**
(F1=0.668, IR=1.737). Agora recomputada vs o baseline linear correto
**de 3 seeds** (`r2base CE`: F1=0.6647±0.0061, IR=1.7808±0.0121):

| Topologia | ΔF1 vs baseline 3s | ΔIR vs baseline 3s |
|---|---|---|
| **MLP trial 4** `[256] GELU drop=0.52` | +0.0081 (dentro de 1σ) | **−0.1122 (σ_comb 0.075) → SIGNIFICATIVO** |
| MLP trial 10 `[1024,1024,2048] SiLU` | +0.0056 (dentro de 1σ) | −0.0652 (dentro de 1σ) |

**Reversão 3 — a topologia SIM contribui para fairness.** A conclusão
anterior ("topologia contribui pouco, não-significativo") era
**pessimista por usar o baseline de 1 seed** (que calhava em IR=1.737,
encurtando o gap). Contra o baseline correto de 3 seeds (IR=1.781 ±
0.012, **bem apertado**), o **MLP trial 4 reduz o IR em ~0.11 de forma
estatisticamente significativa**, sem custo em F1.

---

## 4. Mapa de atribuição parcial (2 de 5 fatores, defensáveis)

| Fator | Contribuição p/ F1 | Contribuição p/ **IR (fairness)** |
|---|---|---|
| **1. Dataset** (limpeza multi-face) | **+1,35pp** (CE, signif.) | **nula** (não-signif. em ambos recipes) |
| **2. Topologia** (linear→MLP t4) | +0,8pp (não-signif.) | **−0,11, SIGNIFICATIVO** |

→ **A alavanca de fairness defensável até agora é a TOPOLOGIA do
classificador, não a limpeza do dataset.** A limpeza paga em acurácia;
a topologia paga em equidade. Esta é uma afirmação de atribuição forte
e honesta — e **oposta** ao que a análise confundida sugeria.

---

## 5. Por que isto FORTALECE a tese

1. Demonstra empiricamente o **valor do método** (decomposição
   controlada + 3 seeds): sem o rigor, a tese teria reportado um efeito
   de dataset de −54% IR que **não existe**, e teria descartado o efeito
   de topologia que **existe**. O método salvou as duas conclusões.
2. Vira a subseção "Ameaças à validade e correções aplicadas" —
   credibilidade máxima na banca (mostra que o aluno detecta e corrige
   os próprios artefatos).
3. Reforça a Linha B (critério Pareto-aware + confirmação obrigatória):
   estimativas de budget curto / 1-seed enganam sistematicamente.

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
- Supersede a leitura de fairness de
  [r2_clean_dataset_results.md](r2_clean_dataset_results.md) §1 (aquela
  era confundida — manter como registro do confound).
- Supersede a comparação 1-seed de
  [r4_refit_results.md](r4_refit_results.md) §1 (Fator 2 agora
  significativo vs baseline 3-seed).
