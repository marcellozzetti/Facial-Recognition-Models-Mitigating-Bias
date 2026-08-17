# Anchor 🅔 — Hassanpour-protocol (Caminho 2+3 combinado)

> Reproduz integralmente o setup metodológico de Hassanpour et al. 2024
> (arXiv 2410.24148) sobre nossas duas arquiteturas-chave, fechando os 3
> confundidores dominantes que separam nosso pipeline do SOTA publicado.
> **Driver de defesa: responde "se rodarmos no setup deles, batemos o número deles?"**.

## 1. O que esta âncora fecha

Os 4 confundidores entre nosso pipeline e Hassanpour 2024
(ver `docs/baseline_positioning.md §4.1`):

| Confundidor | Hassanpour | F1-F5 nosso | 🅔 fecha? |
|---|---|---|---|
| Versão das imagens | padding=0.25 (224 native) | padding=1.25 (448→224 resize) | ✅ sim |
| Split protocol | train/val oficial | 80/10/10 estratificado próprio | ✅ sim |
| Class balance | imbalance natural | undersample por raça | ✅ sim |
| Multi-face cleaning | raw 97k | clean 72k | ❌ não (custo +0.7pp favorável a nós) |
| MTCNN re-align | não aplica | aplica | ❌ não (custo isolado pelo 🅓) |

**🅔 fecha 3 dos 4 dominantes em UM experimento.** O 4º (cleaning + MTCNN) é coberto separadamente pelo anchor 🅓 e tem custo conhecido (~0.7pp F1).

## 2. O que está incluso

| Braço | Backbone | LR | Batch | Seeds | Runs |
|---|---|---|---|---|---|
| **control** | ResNet-50 ImageNet | 1e-3 | 128 | 42, 1, 2 | 3 |
| **convnext** | ConvNeXt-T ImageNet | 1e-4 | 64 | 42, 1, 2 | 3 |
| **TOTAL** | | | | | **6 runs** |

**ETA estimado:** ~9-12h GPU
- 65k train (75% de 86,744) — maior que F1-F5 (54k) → ~+20% por época
- Control: ~9-10min × 15-20 épocas (com early stop) = ~2.5h/seed = 7.5h
- ConvNeXt-T: ~12-14min × 15-20 épocas = ~3.5h/seed = 10.5h (mas pode early-stop antes)

## 3. Recipe matched

Tudo idêntico aos experimentos correspondentes (control = baseline; convnext = Fator 5) EXCETO:
- `data.dataset_image_input_path: data/raw/bucket/fairface-img-margin025` (padding=0.25)
- `data.dataset_file: data/raw/fairface/fairface_labels.csv` (raw 97k, não clean)
- `data.balance: "none"` (não undersample)
- `data.split_protocol: "official"` (usa prefixo `file` do CSV: train/=train_pool, val/=test)
- `training.val_size: 0.25` (fração do train_pool — 25%, matching Hassanpour 75/25)

## 4. Dispatch — quando 🅑 terminar (uma GPU por vez)

```powershell
# 1. Aguardar 🅑 fechar
# (monitor já armado; vai notificar via Done.)

# 2. Verificar GPU livre
nvidia-smi

# 3. Disparar 🅔 em background
python scripts/run_all_experiments.py `
  --configs-dir configs/anchor_hassanpour `
  --output-dir outputs/definitive/anchor_hassanpour `
  --epochs 25 --device cuda 2>&1 | tee outputs/anchor_hassanpour_batch.log
```

## 5. Pós-execução — checklist

1. **Verificar resultado** via `outputs/definitive/anchor_hassanpour/results.json`.
2. **Adicionar grupo no recompute** (`scripts/recompute_checkpoint_criterion.py`):
   ```python
   ("Anchor 🅔 Hassanpour-protocol", "definitive/anchor_hassanpour", "exp_anc_hass_"),
   ```
3. **Atualizar docs:**
   - `docs/anchors_results.md` — adicionar 🅔 na tabela §2 e veredito específico §3.4
   - `docs/baseline_positioning.md` — atualizar §4 com 🅔 + decomposição final do gap
   - `docs/THESIS_STATEMENT.md` — §5 status do gap absoluto
4. **Decisão estratégica final:** se 🅔 control acertar ~0.72 acc, gap fechado contra Hassanpour RN-34. Se ConvNeXt-T acertar ~0.74-0.76, gap fechado contra SOTA-CNN (YOLO11x) e quase contra SOTA-VLM. Definir narrativa da tese conforme outcome.

## 6. Outcomes diagnósticos (predições a priori)

| Outcome | Control acc | ConvNeXt acc | Interpretação |
|---|---|---|---|
| **A (esperado)** | 0.70-0.72 | 0.73-0.75 | Pareamento aproximado com Hassanpour RN-34 (0.72) e YOLO11x (0.74). Gap residual ~1-2pp = "recipe deles é HPO-tuned + test set ligeiramente diferente". Tese **defensável**. |
| **B (otimista)** | 0.72-0.73 | 0.75-0.77 | Pareamento direto com Hassanpour CNN e quase com VLM. Tese **forte**. |
| **C (pessimista)** | 0.66-0.69 | 0.70-0.72 | Apenas ~1-2pp acima do nosso baseline atual. Implica que recipe deles era HPO-tuned além do que conseguimos isolar. Não destrói tese, mas torna a claim absoluta mais modesta. |
| **D (anômalo)** | <0.66 | <0.70 | Algo errado no setup (verificar paths, split, etc). Investigar. |

## 7. Caveats declarados a priori

- **Recipe não é HPO-tuned para padding=0.25.** Nosso recipe AdamW+lr=1e-3 foi calibrado em padding=1.25 (448→224). Padding=0.25 (224 native) pode ter ótimo de hyperparam diferente. Resultado pode subestimar o ConvNeXt sob essa data.
- **Multi-face cleaning + MTCNN ainda diferem.** 🅔 mantém nossos defaults (clean + MTCNN aligned implicit via image_dir). O 🅓 já mostrou que essas escolhas custam ~0.7pp F1 — então o gap residual pós-🅔 é ainda ~0.7pp explicável por isso.
- **Test set ainda diferente do split-validation-leakage de Hassanpour.** Eles fazem hyperparam-tune via val (25% do train) e reportam no val oficial (10,954). Nosso protocolo é o mesmo — mas eles podem ter feito HPO mais agressivo nessa val que nós não fazemos.

## 8. Por que apenas 2 braços (não os 5 fatores)?

NÃO rodamos Fatores 3/4 sob 🅔 porque:
- Fatores 3 (loss) e 4 (paradigma) foram nulls em F1-F5. Re-rodar sob 🅔 abriria nova matriz (atribuição em condição diferente) sem fechar vulnerabilidade da tese.
- A vulnerabilidade que 🅔 fecha é **comparação absoluta vs SOTA** — só precisa do braço com maior F1 esperado (ConvNeXt) e o controle como referência interna.

## 9. Procedência

- Gerador: `scripts/generate_anchor_hassanpour_configs.py`
- Configs (6 files): `configs/anchor_hassanpour/exp_anc_hass_{control,convnext}_s{01,02,42}.yaml`
- Suporte de código: `src/face_bias/data/dataset.py` (split_protocol=official) e `src/face_bias/config/schema.py` (field added)
- Auditoria de SOTA que motiva: `docs/sota_7class_race_audit.md`
- Decomposição de gap: `docs/baseline_positioning.md §4.1`
- Thesis statement: `docs/THESIS_STATEMENT.md §3.3, §5`
