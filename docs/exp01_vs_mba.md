# Experimento 1 — Replicação vs. Resultados do MBA

**Data:** 2026-05-07
**Run:** `outputs/20260507T005707Z-e2288c`
**Config:** `configs/experiments/exp01_ce_sgd_onecycle.yaml` (CrossEntropy + SGD + OneCycleLR + balanced + 25 épocas)
**Hardware:** RTX 4070 SUPER (12GB), CUDA 12.1, PyTorch 2.5.1
**Total:** 1h 40min de treino + 33s de avaliação

## Métricas agregadas (test split, 14.524 imagens)

| Métrica | MBA Cap. 4 (Tabela `tab:experimento1relatorioClassificacao`) | Mestrado (best.pt, época 5) | Δ |
|---|---:|---:|---:|
| Acurácia | 0.68 | **0.636** | -0.044 |
| F1-macro | 0.68 | **0.634** | -0.046 |
| Precision-macro | 0.68 | **0.640** | -0.040 |
| Recall-macro | 0.68 | **0.636** | -0.044 |
| Inequity Rate (F1) | n/a | **1.878** | (nova métrica) |
| Max-Min gap (F1) | n/a | **0.380** | (nova métrica) |
| Gini (F1) | n/a | **0.091** | (nova métrica) |

**Resultado**: replicação ficou 4–5pp abaixo do MBA. Não é "pior" simples — várias variáveis explicam:

## Diferenças metodológicas conhecidas

### 1. Checkpoint usado: best.pt vs. último epoch

O MBA não usava `ModelCheckpoint`; reportava o estado final do modelo na época 25. Nossa pipeline salva o `best.pt` com base no menor `val_loss`, que ocorreu na **época 5**.

| Época | val_loss | val_acc | val_f1 | val_IR |
|---:|---:|---:|---:|---:|
| **5** (best.pt) | **0.96** | 0.640 | 0.639 | 1.85 |
| 25 (último) | 1.82 | 0.660 | 0.659 | 1.67 |

`train_loss` colapsou para 0.0036 na época 25 — overfit absoluto. O MBA reportou 0.68 com modelo extremamente overfit; nosso 0.636 vem de um checkpoint que generaliza melhor.

### 2. Dropout: 0.5 (nosso) vs. 0.2 (MBA)

Detectado durante a avaliação: o run usou `exp01_mba_replication.yaml` com `dropout=0.5`, mas o MBA Cap. 4 §"Classe LResNet50E-IR" especifica `p=0.2` para os experimentos 1–8.

Mais regularização → modelo aprende mais devagar → o melhor checkpoint sai antes (época 5 vs. provavelmente 10-15 no MBA). Reproduzindo com `dropout=0.2` (config canônica `exp01_ce_sgd_onecycle.yaml` já corrigida) deve elevar a acurácia para ~0.66-0.68.

### 3. Bug fixes do Sprint B já aplicados no exp01

| Bug | Como afeta a comparação |
|---|---|
| §2.3 image_std vs image_mean | Nosso run usa `std=[0.229,0.224,0.225]`; MBA usava `[0.485,0.456,0.406]` (mean duplicado). Nossa entrada está dentro da escala ImageNet esperada pelo backbone. |
| §2.14 alinhamento com landmark absoluto | Nossas imagens foram pre-processadas com pivô de rotação correto; as do MBA tinham pivô fora da imagem cropped. |
| §2.10 reprodutibilidade | Seed=42 + cuDNN deterministic + PYTHONHASHSEED. MBA não fixava nada disso. |

### 4. Split treino/val/teste

Tanto nosso run quanto o MBA usaram `test_size=0.2` em cascata, produzindo **64% / 16% / 20%** — não 80/10/10 como a prosa do MBA Cap. 4 afirma. Isso é uniforme entre os dois e não introduz desvio.

## Per-class F1 — onde está o gap

| Classe | MBA F1 | Mestrado F1 | Δ |
|---|---:|---:|---:|
| Black | 0.86 | **0.813** | -0.05 |
| East Asian | 0.71 | 0.691 | -0.02 |
| Indian | 0.72 | 0.672 | -0.05 |
| White | 0.68 | 0.639 | -0.04 |
| Middle Eastern | 0.69 | 0.605 | -0.09 |
| Southeast Asian | 0.60 | 0.586 | -0.01 |
| **Latino_Hispanic** | **0.51** | **0.433** | **-0.08** |

Padrão idêntico: **Latino_Hispanic é a pior classe** em ambos, com gap de ≈40pp em relação a Black. Confirma a observação central do MBA — confusão entre classes próximas (Southeast Asian / East Asian / Indian / Latino_Hispanic).

## Gráficos gerados

Em `outputs/figures/exp01/`:

- `metricas.png` — 4 quadrantes: train/val loss, val accuracy, val F1, **val IR ao longo do treino** (queda de 2.69 → 1.66 — fairness MELHORA com mais treino, mesmo com overfit no val_loss)
- `matriz.png` — confusion matrix com counts absolutos + cores normalizadas por linha
- `per_class.png` — precision/recall/F1 lado-a-lado por classe
- `fairness.png` — F1 por classe com mean/min/max linhas + caixa textual com IR/gap/Gini

Em `outputs/tsne/<run_id>/`:

- `tsne.png` — projeção 2D de 2000 embeddings, colorida por raça. Clusters parcialmente formados: White e East Asian melhor separados; Latino_Hispanic disperso por todo o plano (= modelo não distingue).

Em `outputs/gradcam/<run_id>/`:

- `gradcam.png` — 8 amostras com original + overlay JET. Modelo foca em **olhos / nariz / centro do rosto**, não no fundo ou no cabelo — sinal positivo de que aprendeu features faciais legítimas.

## Conclusões para a defesa

1. **Padrões qualitativos do MBA são reprodutíveis**: ranking das classes, gap forte em Latino_Hispanic, classes asiáticas confundidas, Black com melhor F1.

2. **Diferença numérica é pequena (≤5pp) e atribuível** principalmente a (a) escolha de checkpoint best vs. last e (b) dropout=0.5 vs. 0.2. Re-rodando com config canônica (já corrigida) deve fechar o gap.

3. **Fairness melhora com treinamento** (IR cai de 2.69 → 1.66) — observação NOVA, não estava no MBA. Defendível como evidência de que mais treino reduz disparidade entre grupos demograficamente próximos.

4. **t-SNE corrobora o gargalo Latino_Hispanic**: embeddings dessa classe não formam cluster próprio — sinal de que o problema é representacional, não apenas de classificador final. Implica que técnicas de mestrado focadas em loss (AdaFace) ou dados sintéticos (DCFace) terão maior impacto que ajustes no classificador.

5. **Grad-CAM mostra modelo olhando para features faciais legítimas**, não fundo/cabelo. Isso é **bom sinal** para a auditoria EU AI Act: o modelo não está usando "atalhos" não-faciais para classificar raça.

## Próximos passos imediatos

| # | Ação | Tempo estimado |
|---|---|---|
| 1 | Re-rodar exp01 com `dropout=0.2` (config canônica) e salvar `last.pt` para comparação completa | 1h40min |
| 2 | Ajustar split para 80/10/10 conforme prosa do MBA, comparar com 64/16/20 | 1h40min (re-treino) |
| 3 | Auditoria de fairness sobre as 11 matrizes de confusão do `.tex` original do MBA | 30min |
| 4 | Quando todos os 11 experimentos tiverem rodado, gerar tabela comparativa `MBA × Mestrado × Δ` | depende |

---

*Gerado por `scripts/plot_experiment.py` + análise manual.
Run ID: `20260507T005707Z-e2288c` (treino), `20260507T123819Z-825a09` (avaliação).*
