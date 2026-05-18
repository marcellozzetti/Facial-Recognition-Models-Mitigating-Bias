> ℹ️ **Documento exploratório.** A comparação do §2 usa o baseline
> linear de 1 seed. A medição defensável do Fator Topologia (vs
> baseline linear de 3 seeds) está em
> [dataset_factor_results.md](dataset_factor_results.md) §3 — MLP
> trial 4 reduz o IR em −0,11 (estatisticamente significativo). Os
> achados metodológicos deste doc (variância de seed, necessidade de
> confirmação de budget completo) permanecem válidos.

# Fase 4 — Refit dos Vencedores do Pareto (Fator Topologia)

**Data:** 2026-05-15
**Ambiente:** RTX 4070 SUPER, torch 2.12+cu126, **fp32 (use_amp=false)**,
decode PIL, dataset limpo (72 749, n_faces==1), 25 épocas, early
stopping patience=5.
**Desenho:** 2 topologias vencedoras do Pareto R2 × 3 seeds (42, 1, 2),
cada seed com split estratificado independente.
**Objetivo:** confirmar se o ganho de topologia estimado pelo HPO de
budget curto (8 épocas) sobrevive ao budget completo + múltiplas seeds
— i.e., medir a **contribuição marginal real do fator topologia**.

---

## 1. Resultados (test set)

| Topologia | seed | acc | F1 macro | IR (F1) |
|---|---|---:|---:|---:|
| trial 4 `[256] GELU drop=0.52 none` | 42 | 0.6607 | 0.6635 | 1.7316 |
| trial 4 | 1 | 0.6705 | 0.6731 | 1.7095 |
| trial 4 | 2 | 0.6803 | 0.6817 | 1.5645 |
| **trial 4 — média ± dp** | | **0.6705** | **0.6728 ± 0.0075** | **1.6685 ± 0.0741** |
| trial 10 `[1024,1024,2048] SiLU drop=0.087 LN` | 42 | 0.6665 | 0.6689 | 1.7126 |
| trial 10 | 1 | 0.6628 | 0.6617 | 1.7986 |
| trial 10 | 2 | 0.6780 | 0.6801 | 1.6356 |
| **trial 10 — média ± dp** | | **0.6691** | **0.6702 ± 0.0076** | **1.7156 ± 0.0666** |

Pontos de referência (mesmo ambiente, head linear, fp32, 25ep):

- **R2 baseline** (Exp 5, dataset limpo): F1=0.668, IR=1.737
- R1 baseline (Exp 5, dataset original): F1=0.665, IR=1.76 *(exploratório
  1-seed; referência válida do fator topologia em dataset_factor_results.md §3)*

---

## 2. Leitura estatística — o achado honesto

### 2.1 O ganho de topologia NÃO é estatisticamente significativo (n=3)

- **trial 4** melhora IR em 0.737−1.669 = **0.068** sobre o R2 baseline
  linear — mas o **desvio-padrão é ±0.074**. A diferença está **dentro
  de um desvio-padrão**: os intervalos de confiança se sobrepõem.
- **trial 10** é indistinguível do baseline (ΔF1 = +0.2pp, ΔIR =
  −1.2%) — ruído.
- Em F1, ambas as topologias ficam a ≤ 0.5pp do baseline linear, com dp
  de ±0.75pp — também não-significativo.

### 2.2 O HPO de budget curto superestimou sistematicamente

| Topologia | HPO @8ep (estimativa) | Refit @25ep×3seeds (real) | Erro do HPO |
|---|---|---|---|
| trial 4 | F1=0.6935, IR=1.638 | F1=0.673, IR=1.669 | F1 −2.1pp, IR +0.03 |
| trial 10 | F1=0.6886, IR=1.591 | F1=0.670, IR=1.716 | F1 −1.8pp, IR +0.12 |

As duas estimativas otimistas do Pareto R2 **não sobreviveram** à
confirmação. O HPO de 8 épocas captura um ponto de trajetória que não
generaliza ao treino completo.

### 2.3 Variância entre seeds é alta

trial 4 IR oscila 1.564 → 1.732 (Δ 0.168) só trocando a seed. Isso
sozinho já excede o "ganho" médio de 0.068 sobre o baseline. **A escolha
de seed importa mais que a escolha de topologia** nesta faixa.

---

## 3. Interpretação para a tese de atribuição causal

Este NÃO é um resultado negativo — é exatamente o que uma tese de
**atribuição** deve produzir: uma quantificação honesta.

> **Achado (Fator Topologia):** sob protocolo controlado (25 épocas, 3
> seeds, test set, dataset e ambiente fixos), a contribuição marginal
> da topologia do classificador para a disparidade demográfica é
> **pequena e estatisticamente fraca** (ΔIR ≈ −0.07 ± 0.07, dentro do
> ruído). O ganho aparente reportado por HPO de budget curto é
> **artefato de orçamento**, não contribuição real.

Implicações:

1. **Reforça a Linha B (critério Pareto-aware + confirmação
   obrigatória):** prova empírica de que HPO-curto para fairness é
   não-confiável e exige refit de confirmação. Sem a Fase 4, teríamos
   reportado um ganho de topologia que não existe.
2. **Direciona a atribuição:** se topologia contribui pouco, os fatores
   dataset/loss/contrastivo/backbone passam a ser os candidatos a
   contribuição dominante — o que a decomposição restante deve testar.
3. **Honestidade metodológica = credibilidade na banca.** Reportar um
   IC que cruza zero é mais defensável que inflar 0.5pp.

---

## 4. Ameaça à validade detectada — comparação do Fator Dataset

⚠️ **A comparação Fase 4 vs R2 baseline é LIMPA** (mesmo ambiente:
torch 2.12, PIL, fp32, dataset limpo, 25ep) — o achado do §2/§3 é
defensável.

⚠️ **A comparação R1 vs R2 (Fator Dataset) NÃO é limpa.** Entre R1
(08-14/05) e R2 (14/05) mudaram simultaneamente: dataset (97k→72k),
**torch (2.5.1→2.12)**, **decode de imagem (cv2→PIL)**. O Δ atribuído à
limpeza está confundido com versão de framework + biblioteca de decode.

**Correção aplicada (ver `configs/experiments_r1ctrl/`):** re-execução
do Exp 5 + Exp 6 no dataset **original** sob o **ambiente atual** (torch
2.12, PIL, fp32, código pós-wins). Isso isola o Fator Dataset:
R1ctrl ↔ R2 passam a diferir **apenas no dataset**. Ver
`docs/r1_controlled_results.md` (gerado após a re-execução).

Limitação residual documentada: R1ctrl e R2 têm splits diferentes
(datasets diferentes → partições diferentes mesmo com seed=42),
inerente à limpeza. Framing: cada modelo no seu test set; controle
adicional opcional via test set interseção (trabalho futuro).

---

## 5. Artefatos

- `configs/experiments_clean/refit/exp_r4_{t4,t10}_s{42,01,02}.yaml`
- `outputs/r4_refit/exp_r4_*/{train,evaluate}/...`
- `outputs/r4_refit/comparison_table.md` — tabela gerada pelo orquestrador
- `scripts/generate_refit_configs.py` — gerador reproduzível
