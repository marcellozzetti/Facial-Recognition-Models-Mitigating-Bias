# Etapa 1 — Classificador MST (relatório de preparação)

**Última atualização:** 2026-08-17
**Referência:** Cap. 4 §4.2 (Etapa 1) e §4.9 (validação humana interna)
**Prazo formal do cronograma:** Nov/2026
**Prazo real de trabalho:** Ago–Out/2026 (preparação); Nov/2026 (relatório final)

Este documento registra o estado do código, decisões e pendências da
Etapa 1 antes da execução formal.

## 1. Estado dos recursos externos

### SkinToneNet (Matias, Costa, Neto & Novello de Brito 2026)

- Paper: [arXiv:2603.02475](https://arxiv.org/abs/2603.02475) — CC BY 4.0
- Autores: **ICMC/USP + IMPA** (mesma instituição do template LaTeX)
- Arquitetura: **ViT-Small** pretrained ImageNet, fine-tuned na STW
- Dataset STW: 42.313 imagens / 3.564 indivíduos, escala Monk 10 tons
- Loss: cross-entropy (10 classes)
- **Weights: `"code and data available soon"` — NÃO publicados em 2026-08**
- **Ação pendente:** contatar autores via ICMC/USP para solicitar acesso
  antecipado (afinal, é a mesma universidade do template — canal
  institucional facilita). Se não vier a tempo do prazo formal, avaliar
  reprodução do treinamento assim que STW for divulgado.

### Alternativa clássica CV — `stone` (ChenglongMa/SkinToneClassifier)

- PyPI: `pip install skin-tone-classifier` (≥ v1.2.6, palette Monk)
- Método: detecção facial + segmentação + k-means (não é DL)
- License: GPL-3.0 (compatível com uso acadêmico; **não linkar código-fonte
  próprio ao GPL**; usar como serviço externo/dep opcional)
- **Papel na Etapa 1:** um dos 2 backends do sensitivity analysis
  declarado em Cap. 3 Objetivo 2

## 2. Código entregue nesta preparação

| Arquivo | Papel | Testes |
|---|---|---|
| [src/face_bias/mst/skintonenet.py](../../src/face_bias/mst/skintonenet.py) | wrapper de inferência (ViT + head 10-classes + cache SQLite) | 6 unit tests |
| [src/face_bias/mst/sensitivity.py](../../src/face_bias/mst/sensitivity.py) | `MSTSensitivityRunner` + backend `stone_monk` + κ pairwise | 2 unit tests |
| [src/face_bias/mst/validation.py](../../src/face_bias/mst/validation.py) | amostragem estratificada, `HumanLabelStore`, κ + bootstrap CI, CLI de rotulagem | 7 unit tests |
| [src/face_bias/mst/__init__.py](../../src/face_bias/mst/__init__.py) | exports públicos do subpacote | — |
| [tests/mestrado/unit/test_mst_wrapper.py](../../tests/mestrado/unit/test_mst_wrapper.py) | **16 unit tests reais** (todos passando) | ✅ |
| [pipelines/03_mst_inference.py](../../pipelines/03_mst_inference.py) | CLI end-to-end: `--config`, `--dataset-root`, `--split`, `--output`, `--cache-dir`, `--allow-imagenet-only` | smoke ok |
| [configs/mestrado/stages/etapa1_skintonenet.yaml](../../configs/mestrado/stages/etapa1_skintonenet.yaml) | config atualizada com `inference:` + `labels_store` + `backends` | — |

Total: **+800 linhas de código funcional, 16 testes passando, smoke test
CLI executando em CPU.**

## 3. Decisões técnicas

1. **Backbone**: torchvision `vit_b_16` como stand-in temporário. Trocar
   por timm `vit_small_patch16_224` quando os weights STW oficiais forem
   divulgados (o paper usa ViT-Small). O pipeline 224×224 + ImageNet
   norm é compatível com ambos.
2. **Weights ausentes**: `WeightsUnavailableError` por default; opt-in
   explícito via `allow_imagenet_only=True` para smoke tests. Modo é
   logado com WARN loud para não vazar em métricas científicas.
3. **Cache**: SQLite por SHA-256 do arquivo. Chave composta com
   `model_id` para evitar contaminação cruzada entre versões de weights.
4. **Sensitivity backend**: `stone` (ChenglongMa) é a única alternativa
   MST 10-classes com licença clara e pip-installable disponível hoje.
   `MST-KD` (Caldeira 2024) e HuggingFace models candidatos ficam como
   registros para plug-in via `runner.register(name, callable)`.
5. **Validação humana**: `HumanLabelStore` JSONL append-only + CLI
   `label_cli` que abre imagens via viewer do SO. Sem GUI própria —
   simplicidade > polish. `stratified_sample` garante piso 1 por
   (raça, mst_pred) para não perder tons raros.
6. **Bootstrap**: 10.000 réplicas, percentile CI 95%, seeds fixas.
   Coerente com [feedback_experimental_rigor](../../../../.claude/projects/c--Users-KABUM-Documents-workspace-github-Facial-Recognition-Models-Mitigating-Bias/memory/feedback_experimental_rigor.md).

## 4. Como executar (quando FairFace estiver disponível)

```bash
# Inferência sobre FairFace val (10.954 imagens)
python pipelines/03_mst_inference.py \
    --config configs/mestrado/stages/etapa1_skintonenet.yaml \
    --dataset-root data/FairFace \
    --labels-csv data/FairFace/val_labels.csv \
    --split val \
    --output outputs/etapa1/fairface_val_mst.parquet \
    --cache-dir outputs/etapa1/cache/

# Sensitivity analysis (após instalar `pip install skin-tone-classifier`)
python - <<'PY'
from pathlib import Path
from face_bias.mst import MSTSensitivityRunner, SkinToneNetInference
runner = MSTSensitivityRunner()
runner.register_stone_monk()
infer = SkinToneNetInference("models_pretrained/skintonenet_vits.pt", device="cuda")
runner.register_skintonenet("skintonenet", infer.infer, infer.preprocess)
preds = runner.run(list(Path("outputs/etapa1/sample").glob("*.jpg")))
print(runner.pairwise_kappa(preds))
PY
```

## 5. Pendências para execução formal (Nov/2026)

- [ ] Contatar autores do SkinToneNet (canal ICMC/USP) ou aguardar
  divulgação dos weights
- [ ] Download do FairFace completo (~13 GB) para o volume DVC
- [ ] Rodar inferência sobre FairFace train+val+test (~108k imagens)
- [ ] Amostragem estratificada para validação humana (250 imagens)
- [ ] Sessão de rotulagem (Mestrando + Orientador) — estimar 3–4 h por
  anotador
- [ ] Sensitivity analysis com `stone_monk` + 1 backend adicional a definir
- [ ] Gerar `outputs/etapa1/relatorio.md` final (para anexo da dissertação)

## 6. Riscos remapeados

- **Risco 2 (Cap. 5) — dependência SkinToneNet**: mitigação executada
  parcialmente. Wrapper aceita qualquer state_dict compatível; sensitivity
  runner permite trocar backend em 1 linha. Modo `allow_imagenet_only`
  garante que o pipeline downstream (Etapas 2+) não fica bloqueado à
  espera dos weights.
- **Novo risco**: dependência de licença GPL do `stone` — não linkar ao
  código próprio; usar como dep opcional invocada out-of-process
  (import lazy já implementado).
