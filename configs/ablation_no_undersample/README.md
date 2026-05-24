# Ablation 🅑 — no-undersample (driver de sucesso da dissertação)

> Ablação de robustez do achado central da dissertação: testa se a alavanca
> **ConvNeXt-T** (Fator 5, F1 +2.3pp / IR −0.13 vs controle) sobrevive
> quando a decisão de **class balance** muda de `undersample` para `none`.
> Configs gerados em 2026-05-22, prontos para dispatch.

## 1. Por que esta ablação é crítica

Conforme `docs/THESIS_STATEMENT.md §3.3` e `docs/sota_7class_race_audit.md §5`:

- Todas as 5 pipelines de fatores e 3 anchors usam `balance: undersample`.
- Hassanpour 2024 (SOTA real: 72% RN-34 / 75.7% VLM) usa **imbalance natural**.
- Decomposição do gap absoluto (−1.1pp vs Hassanpour RN-34): undersample contribui ~−1 a −2pp.
- Esta ablação **fecha** o maior componente desse gap absoluto.

**Outcomes diagnósticos esperados** (`docs/sota_7class_race_audit.md §6.3`):

| Outcome | acc Δ | F1 macro Δ | IR Δ | Interpretação |
|---|---|---|---|---|
| **A (theory)** | +2-3pp ↑ | ~0 ou ↓ pequeno | ↑ piora | undersample trocou acc por equidade (trade-off honesto) |
| **B** | +2-3pp ↑ | +1pp ↑ | ~0 ou ↓ | undersample era custo sem benefício |
| **C** | ~0 | ↓ pior | ↑ piora | undersample era net helpful |
| **D** | −1pp ↓ | ↓ pior | ↑ piora | resultado anômalo (investigar) |

## 2. O que está incluso

| Braço | Backbone | LR | Batch | Seeds | Total runs |
|---|---|---|---|---|---|
| **control** (= baseline definitivo, mas sem undersample) | ResNet-50 ImageNet | 1e-3 | 128 | 42, 1, 2 | 3 |
| **convnext** (= Fator 5 ConvNeXt-T, mas sem undersample) | ConvNeXt-T ImageNet | 1e-4 | 64 | 42, 1, 2 | 3 |
| **TOTAL** | | | | | **6 runs** |

**ETA estimado:** ~7-9h GPU
- Control RN-50: ~75min × 3 = ~3h45
- ConvNeXt-T: ~95min × 3 = ~4h45
- Eval + overhead: ~30min

## 3. Recipe matched (única diferença: `balance: none`)

Tudo idêntico aos experimentos definitivos correspondentes:
- 25 épocas, early stopping patience=5
- val_f1_macro como critério (correto, conforme `docs/checkpoint_criterion_audit.md`)
- num_workers=0 (Windows deadlock-proof)
- fp32 (matched, sem AMP)
- Split 80/10/10 estratificado, seed-fixo (mesmas imagens train/val/test por seed que os demais experimentos)
- Dataset: `data/raw/fairface/fairface_labels_clean.csv` (72k clean)
- Imagens: `data/processed/fairface_aligned/` (MTCNN aligned, mesmas dos F1-F5)

**Única diferença:** `data.balance: "none"` (vs `"undersample"` dos demais). Significa que o train mantém a imbalance natural do FairFace clean (~1.8x entre maior/menor classe).

## 4. Dispatch — quando estiver pronto pra rodar

```powershell
# Verificar GPU livre e fria
nvidia-smi

# Disparar batch
python scripts/run_all_experiments.py `
  --configs-dir configs/ablation_no_undersample `
  --output-dir outputs/definitive/ablation_no_undersample `
  --epochs 25 --device cuda 2>&1 | tee outputs/ablation_no_undersample_batch.log

# (opcional) Armar monitor em outra janela
# tail -F outputs/ablation_no_undersample_batch.log | grep -E "epoch=|ok|FAIL|Done"
```

## 5. Pós-execução — checklist

1. **Recompute via history** (caso o critério naïve tenha pego best-by-val_loss):
   ```powershell
   python scripts/recompute_checkpoint_criterion.py --definitive
   ```
   (precisa adicionar o grupo `("Ablation no-undersample", "definitive/ablation_no_undersample", "exp_abl_nous_")` na lista do script)

2. **Atualizar `docs/anchors_results.md`** com seção 7 (Ablation 🅑 results) — Δ F1, Δ IR vs ConvNeXt-T e controle matched.

3. **Atualizar `docs/THESIS_STATEMENT.md` §4 e §9** com:
   - Δ medido vs estimado (-2 a -3pp predito)
   - Se ConvNeXt-T sem undersample > Hassanpour ResNet-34 (0.720), declarar pareamento de número.
   - Outcome A/B/C/D classificado.

4. **Atualizar `docs/factor5_results.md`** com seção de robustez:
   - "ConvNeXt-T alavanca confirmada robusta a class balance" (se Δ F1 do ConvNeXt vs controle preservado sob no-undersample).

5. **Commit:**
   ```powershell
   git add configs/ablation_no_undersample outputs/definitive/ablation_no_undersample docs/
   git commit -m "feat(ablation): no-undersample robustness — ConvNeXt-T alavanca persiste sob imbalance natural"
   ```

## 6. Caveats declarados a priori

- A ablação NÃO muda split nem padding nem MTCNN re-align. Esses 3 confundidores remanescentes valem ~−1pp adicional vs Hassanpour 2024 setup.
- Se ConvNeXt-T sem undersample chegar a ~0.72 acc, **estamos pareados com Hassanpour ResNet-34** — claim defensável para banca.
- Se chegar a ~0.74 acc, estamos pareados com YOLO11x (community).
- Se chegar a ~0.76, isso seria suspeito (sob 4 confundidores remanescentes, acima da VLM-SOTA é improvável). Verificar metodologia.

## 7. Por que apenas 2 braços?

NÃO rodamos os outros fatores (3, 4) nem os outros anchors sem undersample porque:
- Fatores 3 (loss) e 4 (paradigma) foram nulls sob protocolo matched. Re-rodar sem undersample abriria nova matriz (atribuição em condição diferente) em vez de fechar uma vulnerabilidade.
- Anchors 🅐.1, 🅐.2, 🅓 servem para posicionamento; rodá-los sem undersample multiplica o espaço sem agregar à claim de robustez do achado central.

A ablação tem **escopo cirúrgico**: testa se o achado positivo principal (ConvNeXt-T > controle em acc+IR significativamente) sobrevive à decisão de class balance. **Apenas essa pergunta**.

## 8. Procedência

- Gerador: `scripts/generate_ablation_no_undersample_configs.py`
- Configs gerados (6 files): `configs/ablation_no_undersample/exp_abl_nous_{control,convnext}_s{01,02,42}.yaml`
- Justificativa metodológica: `docs/THESIS_STATEMENT.md §3.3`
- Auditoria de SOTA: `docs/sota_7class_race_audit.md §5-6`
