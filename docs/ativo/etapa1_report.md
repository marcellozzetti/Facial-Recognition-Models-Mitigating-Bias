# Etapa 1 — Classificador MST (relatório de preparação)

**Última atualização:** 2026-08-18
**Referência:** Cap. 4 §4.2 (Etapa 1) e §4.9 (validação humana interna)
**Prazo formal do cronograma:** Nov/2026
**Prazo real de trabalho:** Ago–Out/2026 (preparação); Nov/2026 (relatório final)

Este documento registra o estado do código, decisões e pendências da
Etapa 1 antes da execução formal.

## 1. Estratégia (revisada pós-reunião Ago/2026)

**Decisão de escopo (reunião com Prof. Marcos Quiles, 17-Ago/2026):**
adotar **classificador MST próprio treinado internamente** como método
principal, em vez de depender do release do SkinToneNet (Matias 2026).
Motivos:

- Independência do cronograma de release externo (não bloqueia)
- Controle total do preprocessamento (pipeline auto-suficiente)
- Desbloqueia Etapa 5 (RFW/BFW não têm anotação MST — precisamos gerar)

O SkinToneNet permanece como:
- **Comparação externa condicional** — se acesso ao STW for concedido,
  reportamos acurácia comparativa no mesmo protocolo de teste
- **Backend do sensitivity analysis** — se os pesos forem publicados,
  entram como um dos 2-3 backends alternativos

## 2. Datasets de treino

| Dataset | Fonte | Tamanho | Status |
|---|---|---|---|
| **MSTE** (Monk Skin Tone Examples) | Google | ~1.500 imgs | ✅ público |
| **Casual Conversations v2** | Meta / Porgali 2023 | ~5.500 vídeos | ✅ público c/ EULA |
| **STW** (Skin Tone in the Wild) | Matias 2026, ICMC/USP | 42.313 imgs | ⏳ "available soon" (validação externa condicional) |

## 3. Código entregue

### Novos módulos

| Arquivo | Papel |
|---|---|
| [src/face_bias/mst/datasets.py](../../src/face_bias/mst/datasets.py) | loaders MSTE, CCv2, STW + `build_mst_dataset` factory + `class_balance` |
| [src/face_bias/mst/trainer.py](../../src/face_bias/mst/trainer.py) | `MSTTrainer` + `stratified_split` + F1 macro; early stopping por F1 macro em val |
| [src/face_bias/mst/preprocessing.py](../../src/face_bias/mst/preprocessing.py) | `MSTFromRawImage` — detect → align → crop → normalize → classify; `MSTResult` com fallback explicativo |
| [pipelines/03a_train_mst_classifier.py](../../pipelines/03a_train_mst_classifier.py) | CLI de treino end-to-end |

### Módulos revisados

| Arquivo | Mudança |
|---|---|
| [src/face_bias/mst/classifier.py](../../src/face_bias/mst/classifier.py) | Ex-`skintonenet.py`. Classe `MSTClassifier` (aceita qualquer state_dict compatível). |
| [pipelines/03_mst_inference.py](../../pipelines/03_mst_inference.py) | Flag `--auto-preprocess` que ativa `MSTFromRawImage` para datasets crus |
| [src/face_bias/mst/sensitivity.py](../../src/face_bias/mst/sensitivity.py) | SkinToneNet listado como um dos backends (não mais como referência única) |
| [pipelines/03b_train_mst_reproduction.py](../../pipelines/03b_train_mst_reproduction.py) | Mantém como opção de reprodução do SkinToneNet específico (se STW sair) |

### Cobertura de testes

**50 unit tests reais passando**, distribuídos:
- 16 tests em `test_mst_wrapper.py` (classifier)
- 8 tests em `test_mst_datasets.py`
- 6 tests em `test_mst_trainer.py`
- 10 tests em `test_mst_preprocessing.py`
- 10 tests em `test_cross_matrix.py` (Etapa 2)

## 4. Como executar

### Treino do classificador próprio (Etapa 1)

```bash
# Uma vez por seed (rigor experimental exige 3 seeds)
python pipelines/03a_train_mst_classifier.py \
    --mste-root data/MSTE \
    --ccv2-root data/CCv2 \
    --output outputs/etapa1_own/ \
    --seed 42 --max-epochs 30

python pipelines/03a_train_mst_classifier.py --seed 1 ...
python pipelines/03a_train_mst_classifier.py --seed 2 ...
```

### Inferência sobre FairFace (Etapa 2 → 3)

```bash
python pipelines/03_mst_inference.py \
    --config configs/mestrado/stages/etapa1_skintonenet.yaml \
    --weights outputs/etapa1_own/best_seed42.pt \
    --dataset-root data/FairFace \
    --labels-csv data/FairFace/val_labels.csv \
    --output outputs/etapa1/fairface_val_mst.parquet \
    --cache-dir outputs/etapa1/cache/
```

### Inferência em dataset cru (Etapa 5 — RFW/BFW)

```bash
# COM --auto-preprocess: MTCNN detecta o rosto, alinha, e passa para o MST
python pipelines/03_mst_inference.py \
    --weights outputs/etapa1_own/best_seed42.pt \
    --dataset-root data/RFW/images \
    --output outputs/etapa5/rfw_mst.parquet \
    --auto-preprocess \
    --min-face-size 40 --min-confidence 0.9
```

## 5. Pendências para execução formal

- [ ] Baixar MSTE (~1500 imgs, público)
- [ ] Baixar Casual Conversations v2 (requer aceitar EULA da Meta)
- [ ] Treinar 3 seeds do classificador próprio
- [ ] Amostragem estratificada para validação humana (250 imagens FairFace)
- [ ] Sessão de rotulagem (Mestrando + Orientador) — 3-4 h por anotador
- [ ] Sensitivity analysis com `stone_monk` (ChenglongMa) + eventual SkinToneNet
- [ ] Envio do email institucional ao ICMC/USP (docs/ativo/email_skintonenet_authors.md)
- [ ] Se acesso ao STW for concedido: benchmark externo comparativo

## 6. Riscos remapeados

- **Risco 2 (Cap. 5, revisado) — qualidade do classificador MST próprio**:
  mitigação via validação humana interna + sensitivity com `stone_monk`
  + eventual benchmark externo no STW.
- **Novo risco menor — licença GPL do `stone`**: uso como dep opcional,
  invocação out-of-process (lazy import já implementado).
